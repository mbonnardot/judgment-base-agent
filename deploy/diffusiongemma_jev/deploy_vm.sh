#!/usr/bin/env bash
#
# Deploy OpenJev (DiffusionGemma-26B-A4B) onto a GCE GPU VM (1x NVIDIA L4).
#
# Usage:
#   ./deploy/diffusiongemma_jev/deploy_vm.sh            # Deploy to existing VM (Docker mode)
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --create   # Provision VM if missing, then deploy
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --systemd  # Deploy via systemd (/opt/venv/vllm) instead of Docker
#
# Environment overrides:
#   DJEV_PROJECT     GCP project ID (default: medquad-assistant-capstone)
#   DJEV_ZONE        GCE zone (default: us-central1-a)
#   DJEV_INSTANCE    GCE instance name (default: djev-vllm-l4)
#   DJEV_REMOTE_PORT OpenJev port on the VM (default: 8011)
#   OPENJEV_IMAGE    Container image (default: razorback16/openjev:0.3.0)
#
set -euo pipefail

PROJECT="${DJEV_PROJECT:-${PROJECT_ID:-medquad-assistant-capstone}}"
ZONE="${DJEV_ZONE:-us-central1-a}"
INSTANCE="${DJEV_INSTANCE:-djev-vllm-l4}"
REMOTE_PORT="${DJEV_REMOTE_PORT:-8011}"
MACHINE_TYPE="${DJEV_MACHINE_TYPE:-g2-standard-8}"
BOOT_DISK_SIZE="${DJEV_BOOT_DISK_SIZE:-200GB}"
OPENJEV_IMAGE="${OPENJEV_IMAGE:-razorback16/openjev:0.3.0}"
DEPLOY_MODE="${DEPLOY_MODE:-docker}"
CREATE_IF_MISSING=false

for arg in "$@"; do
  case "${arg}" in
    --create)
      CREATE_IF_MISSING=true
      ;;
    --systemd)
      DEPLOY_MODE="systemd"
      ;;
    --docker)
      DEPLOY_MODE="docker"
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      echo "Usage: $0 [--create] [--docker|--systemd]" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v gcloud >/dev/null || die "gcloud not found; install the Google Cloud SDK"

echo "========================================================================"
echo " Deploying OpenJev (DiffusionGemma-26B-A4B) to GCE GPU VM"
echo " Project  : ${PROJECT}"
echo " Zone     : ${ZONE}"
echo " Instance : ${INSTANCE} (${MACHINE_TYPE}, 1x NVIDIA L4)"
echo " Mode     : ${DEPLOY_MODE} (port ${REMOTE_PORT})"
echo "========================================================================"

# 1. Ensure GCE GPU VM exists and is RUNNING
if ! STATUS="$(gcloud compute instances describe "${INSTANCE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --format='value(status)' 2>/dev/null)"; then
  if [[ "${CREATE_IF_MISSING}" != true ]]; then
    die "Instance ${INSTANCE} not found in ${PROJECT}/${ZONE}. Re-run with --create to provision it."
  fi
  log "[1/3] Provisioning GCE L4 GPU VM ${INSTANCE} (${MACHINE_TYPE})..."
  gcloud compute instances create "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=nvidia-l4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu124-ubuntu-2204-nvidia-550 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    --no-address \
    --scopes=cloud-platform
elif [[ "${STATUS}" != "RUNNING" ]]; then
  log "[1/3] Instance ${INSTANCE} is ${STATUS}; starting it..."
  gcloud compute instances start "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}"
else
  log "[1/3] Instance ${INSTANCE} is already RUNNING."
fi

# Wait for SSH over IAP to become available
log "Waiting for IAP SSH connectivity to ${INSTANCE}..."
for _ in $(seq 1 30); do
  if gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --command="true" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

# 2. Deploy OpenJev via Docker or systemd
if [[ "${DEPLOY_MODE}" == "systemd" ]]; then
  log "[2/3] Uploading systemd unit and installer to ${INSTANCE}..."
  gcloud compute scp --recurse "${SCRIPT_DIR}/systemd" \
    "${INSTANCE}:~/systemd" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap
  log "Running systemd installer on ${INSTANCE}..."
  gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --command="sudo OPENJEV_PORT=${REMOTE_PORT} bash ~/systemd/install.sh"
else
  log "[2/3] Deploying ${OPENJEV_IMAGE} container on ${INSTANCE}..."
  gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --command="sudo bash -s" <<EOF
set -euo pipefail
# Stop any legacy systemd services holding GPU VRAM
systemctl stop djev-jev.service djev-vllm.service 2>/dev/null || true
systemctl disable djev-jev.service djev-vllm.service 2>/dev/null || true
pkill -f 'vllm serve RedHatAI' 2>/dev/null || true
pkill -f 'structured_server.py' 2>/dev/null || true

mkdir -p /opt/hf
docker pull "${OPENJEV_IMAGE}"
docker rm -f openjev 2>/dev/null || true
docker run -d \
  --name openjev \
  --restart unless-stopped \
  --gpus all \
  -p "127.0.0.1:${REMOTE_PORT}:${REMOTE_PORT}" \
  -v /opt/hf:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e OPENJEV_HOST=0.0.0.0 \
  -e OPENJEV_PORT="${REMOTE_PORT}" \
  -e OPENJEV_MAX_MODEL_LEN=16384 \
  -e OPENJEV_GPU_UTIL=0.90 \
  "${OPENJEV_IMAGE}"
EOF
fi

# 3. Verify health endpoint on the VM
log "[3/3] Waiting for OpenJev on ${INSTANCE}:${REMOTE_PORT}/health (~3-5 min cold load)..."
gcloud compute ssh "${INSTANCE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --tunnel-through-iap \
  --command="for i in \$(seq 1 90); do curl -sf http://127.0.0.1:${REMOTE_PORT}/health && exit 0; sleep 5; done; echo 'Timed out waiting for /health' >&2; exit 1"

cat <<EOF

========================================================================
 OpenJev Deployed Successfully on GCE GPU VM (${INSTANCE})!
 Connect from your local machine via IAP tunnel:
   ./scripts/connect_gpu.sh
 Then export:
   export OPENJEV_BASE_URL="http://127.0.0.1:${REMOTE_PORT}"
========================================================================
EOF
