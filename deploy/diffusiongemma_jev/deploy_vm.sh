#!/usr/bin/env bash
#
# Deploy OpenJev (DiffusionGemma-26B-A4B) onto a GCE GPU VM.
#
# Defaults to `a2-highgpu-1g` (1x NVIDIA A100 40GB VRAM) so `razorback16/openjev:0.4.0`
# runs with 100% unmodified upstream defaults (`OPENJEV_MAX_MODEL_LEN=65536` + CUDA graphs).
# Pass `--l4` to use `g2-standard-8` (1x NVIDIA L4 24GB VRAM).
#
# Usage:
#   ./deploy/diffusiongemma_jev/deploy_vm.sh            # Deploy to existing VM (Docker mode)
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --create   # Provision 40GB A100 VM if missing, then deploy
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --create --l4 # Provision 24GB L4 VM (g2-standard-8)
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --systemd  # Deploy via systemd (/opt/venv/vllm) instead of Docker
#
set -euo pipefail

PROJECT="${DJEV_PROJECT:-${PROJECT_ID:-judgement-base-agent}}"
ZONE="${DJEV_ZONE:-us-central1-c}"
REGION="${ZONE%-*}"
INSTANCE="${DJEV_INSTANCE:-djev-vllm-l4}"
REMOTE_PORT="${DJEV_REMOTE_PORT:-8011}"
MACHINE_TYPE="${DJEV_MACHINE_TYPE:-a2-highgpu-1g}"
ACCELERATOR="${DJEV_ACCELERATOR:-type=nvidia-tesla-a100,count=1}"
BOOT_DISK_SIZE="${DJEV_BOOT_DISK_SIZE:-200GB}"
IMAGE_FAMILY="${DJEV_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
OPENJEV_IMAGE="${OPENJEV_IMAGE:-razorback16/openjev:0.4.0}"
DEPLOY_MODE="${DEPLOY_MODE:-docker}"
CREATE_IF_MISSING=false
EXTRA_DOCKER_ENV="-e OPENJEV_VLLM_ARGS=--kv-cache-dtype=bfloat16"

for arg in "$@"; do
  case "${arg}" in
    --create)
      CREATE_IF_MISSING=true
      ;;
    --l4)
      MACHINE_TYPE="g2-standard-8"
      ACCELERATOR="type=nvidia-l4,count=1"
      EXTRA_DOCKER_ENV="-e OPENJEV_MAX_MODEL_LEN=4096 -e OPENJEV_VLLM_ARGS=--enforce-eager"
      ;;
    --systemd)
      DEPLOY_MODE="systemd"
      ;;
    --docker)
      DEPLOY_MODE="docker"
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      echo "Usage: $0 [--create] [--l4] [--docker|--systemd]" >&2
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
echo " Instance : ${INSTANCE} (${MACHINE_TYPE}, ${ACCELERATOR})"
echo " Mode     : ${DEPLOY_MODE} (port ${REMOTE_PORT})"
echo "========================================================================"

# Ensure required APIs, default VPC network, IAP SSH firewall rule, and Cloud NAT exist
gcloud services enable compute.googleapis.com iap.googleapis.com \
  --project="${PROJECT}" --quiet >/dev/null 2>&1 || true

if ! gcloud compute networks describe default \
  --project="${PROJECT}" --quiet >/dev/null 2>&1; then
  log "Creating default VPC network in ${PROJECT}..."
  gcloud compute networks create default \
    --project="${PROJECT}" \
    --subnet-mode=auto \
    --quiet >/dev/null
fi

if ! gcloud compute firewall-rules describe djev-allow-iap-ssh \
  --project="${PROJECT}" --quiet >/dev/null 2>&1; then
  log "Creating IAP SSH firewall rule (djev-allow-iap-ssh)..."
  gcloud compute firewall-rules create djev-allow-iap-ssh \
    --project="${PROJECT}" \
    --network=default \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --quiet >/dev/null
fi

if ! gcloud compute routers describe djev-router \
  --region="${REGION}" --project="${PROJECT}" --quiet >/dev/null 2>&1; then
  log "Creating Cloud Router & Cloud NAT (djev-router / djev-nat) in ${REGION} for outbound Docker/HF access..."
  gcloud compute routers create djev-router \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --network=default \
    --quiet >/dev/null
  gcloud compute routers nats create djev-nat \
    --router=djev-router \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --nat-all-subnet-ip-ranges \
    --auto-allocate-nat-external-ips \
    --quiet >/dev/null
fi

# 1. Ensure GCE GPU VM exists and is RUNNING
if ! STATUS="$(gcloud compute instances describe "${INSTANCE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --format='value(status)' \
  --quiet 2>/dev/null)"; then
  if [[ "${CREATE_IF_MISSING}" != true ]]; then
    die "Instance ${INSTANCE} not found in ${PROJECT}/${ZONE}. Re-run with --create to provision it."
  fi
  log "[1/3] Provisioning GCE GPU VM ${INSTANCE} (${MACHINE_TYPE}, ${ACCELERATOR})..."
  gcloud compute instances create "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="${ACCELERATOR}" \
    --maintenance-policy=TERMINATE \
    --image-family="${IMAGE_FAMILY}" \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    --shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --no-address \
    --scopes=cloud-platform \
    --quiet
elif [[ "${STATUS}" != "RUNNING" ]]; then
  log "[1/3] Instance ${INSTANCE} is ${STATUS}; starting it..."
  gcloud compute instances start "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --quiet
else
  log "[1/3] Instance ${INSTANCE} is already RUNNING."
fi

# Wait for SSH over IAP to become available
log "Waiting for IAP SSH connectivity to ${INSTANCE}..."
for _ in $(seq 1 36); do
  if gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --quiet \
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
    --tunnel-through-iap \
    --quiet
  log "Running systemd installer on ${INSTANCE}..."
  gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --quiet \
    --command="sudo OPENJEV_PORT=${REMOTE_PORT} bash ~/systemd/install.sh"
else
  log "[2/3] Deploying ${OPENJEV_IMAGE} container on ${INSTANCE}..."
  gcloud compute ssh "${INSTANCE}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --quiet \
    --command="sudo bash -s" <<EOF
set -euo pipefail
# Stop any legacy systemd services holding GPU VRAM
systemctl stop djev-jev.service djev-vllm.service 2>/dev/null || true
systemctl disable djev-jev.service djev-vllm.service 2>/dev/null || true
pkill -f 'vllm serve RedHatAI' 2>/dev/null || true
pkill -f 'structured_server.py' 2>/dev/null || true

# Install Docker + NVIDIA Container Toolkit if missing on the VM
if ! command -v docker >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq curl gnupg docker.io
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl enable --now docker
  systemctl restart docker
fi

mkdir -p /opt/hf
docker pull "${OPENJEV_IMAGE}"
docker rm -f openjev 2>/dev/null || true
docker run -d \
  --name openjev \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  -p "127.0.0.1:${REMOTE_PORT}:8080" \
  -v /opt/hf:/root/.cache/huggingface \
  ${EXTRA_DOCKER_ENV} \
  "${OPENJEV_IMAGE}"
EOF
fi

# 3. Verify health endpoint on the VM
log "[3/3] Waiting for OpenJev on ${INSTANCE}:${REMOTE_PORT}/health (~3-5 min cold load)..."
gcloud compute ssh "${INSTANCE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --tunnel-through-iap \
  --quiet \
  --command="for i in \$(seq 1 120); do curl -sf http://127.0.0.1:${REMOTE_PORT}/health && exit 0; sleep 5; done; sudo docker logs --tail 50 openjev 2>&1 || true; echo 'Timed out waiting for /health' >&2; exit 1"

cat <<EOF

========================================================================
 OpenJev Deployed Successfully on GCE GPU VM (${INSTANCE})!
 Connect from your local machine via IAP tunnel:
   ./scripts/connect_gpu.sh
 Then export:
   export OPENJEV_BASE_URL="http://127.0.0.1:${REMOTE_PORT}"
========================================================================
EOF
