#!/usr/bin/env bash
#
# Open an IAP tunnel to the shared DiffusionGemma GPU VM and health-check it.
#
#   ./scripts/connect_gpu.sh          # tunnel on localhost:8011, stays in foreground
#   ./scripts/connect_gpu.sh --start  # also start the VM if it is stopped
#
# Leave this running in its own terminal, then in a second terminal:
#
#   cp examples/.env.example examples/.env   # uncomment the DIFFUSIONGEMMA_* block
#   uv run adk web examples --port 8008
#
# The VM has no external IP by design, so IAP is the only way in. You need
# roles/iap.tunnelResourceAccessor and roles/compute.viewer on the project.
#
# The tunnel self-heals: IAP SSH sessions get torn down after a few hours
# ("client_loop: send disconnect: Broken pipe"), which would otherwise leave
# adk web pointed at a dead port. This script reconnects until you Ctrl-C.
set -euo pipefail

PROJECT="${DJEV_PROJECT:-judgement-base-agent}"
ZONE="${DJEV_ZONE:-us-central1-c}"
INSTANCE="${DJEV_INSTANCE:-djev-vllm-l4}"
LOCAL_PORT="${DJEV_LOCAL_PORT:-8011}"
REMOTE_PORT="${DJEV_REMOTE_PORT:-8011}"

START_IF_STOPPED=false
[[ "${1:-}" == "--start" ]] && START_IF_STOPPED=true

TUNNEL_PID=""
SHUTTING_DOWN=false

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

port_free() {
  # No lsof (some minimal images): assume free and let ssh report the conflict.
  command -v lsof >/dev/null || return 0
  ! lsof -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1
}

# gcloud runs ssh as a grandchild. Killing the gcloud parent does not always
# take the ssh process with it, and while it lives it keeps ${LOCAL_PORT}
# bound -- which makes the next connect attempt fail to bind and look like a
# flapping tunnel. Reap it and wait for the port to actually come free.
stop_tunnel() {
  [[ -z "${TUNNEL_PID}" ]] && return 0
  kill "${TUNNEL_PID}" 2>/dev/null || true
  wait "${TUNNEL_PID}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    port_free && return 0
    pkill -f "ssh.*-L ${LOCAL_PORT}:127\.0\.0\.1:${REMOTE_PORT}" 2>/dev/null || true
    sleep 1
  done
  warn "port ${LOCAL_PORT} is still held; the next attempt may fail to bind"
}

shutdown() {
  SHUTTING_DOWN=true
  stop_tunnel
}
trap shutdown EXIT INT TERM

command -v gcloud >/dev/null || die "gcloud not found; install the Google Cloud SDK"

if ! port_free; then
  die "localhost:${LOCAL_PORT} is already in use. A tunnel may already be running."
fi

start_tunnel() {
  gcloud compute ssh "${INSTANCE}" \
    --zone "${ZONE}" --project "${PROJECT}" --tunnel-through-iap \
    -- -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=1000 \
    -o ExitOnForwardFailure=yes &
  TUNNEL_PID=$!
}

# Poll until the Jev port answers through the tunnel. Returns non-zero on
# timeout or if the tunnel process dies underneath us.
wait_for_endpoint() {
  local attempts="$1"
  for _ in $(seq 1 "${attempts}"); do
    if curl -sf --max-time 5 "http://127.0.0.1:${LOCAL_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 "${TUNNEL_PID}" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# The first judgment after a cold start pays for CUDA graph capture and an
# empty prefix cache: measured 30.6 s, versus 258 ms once warm. Burn that cost
# here so nobody's first impression is a 30-second hang.
warm_model() {
  curl -sf --max-time 90 "http://127.0.0.1:${LOCAL_PORT}/v1/systemone" \
    -H 'Content-Type: application/json' \
    -d '{"model":"openjev-latest","state":{"warmup":true},"questions":{"q0":{"type":"noul","instructions":"Is this a warmup?"}}}' \
    >/dev/null 2>&1
}

log "Checking ${INSTANCE} in ${PROJECT}/${ZONE}"
STATUS="$(gcloud compute instances describe "${INSTANCE}" \
  --zone "${ZONE}" --project "${PROJECT}" \
  --format='value(status)' 2>/dev/null)" \
  || die "cannot read ${INSTANCE}. Do you have access to project ${PROJECT}?"

if [[ "${STATUS}" != "RUNNING" ]]; then
  if [[ "${START_IF_STOPPED}" != true ]]; then
    die "${INSTANCE} is ${STATUS}. Re-run with --start to boot it (~5 min, then it bills ~\$0.71/hr)."
  fi
  log "${INSTANCE} is ${STATUS}; starting it"
  gcloud compute instances start "${INSTANCE}" --zone "${ZONE}" --project "${PROJECT}"
  log "Waiting for the model to load (systemd brings both services up; ~5 min cold)"
fi

log "Opening tunnel localhost:${LOCAL_PORT} -> ${INSTANCE}:${REMOTE_PORT}"
start_tunnel

log "Waiting for the OpenJev endpoint to answer"
wait_for_endpoint 120 || die "endpoint never came up. On the VM: systemctl status djev-vllm djev-jev"

log "Warming the model (one throwaway judgment, up to ~40 s)"
warm_model || warn "warm-up failed; the endpoint is up but the first real call will be slow"

cat <<EOF

$(printf '\033[1;32m==> Ready.\033[0m') OpenJev System One is at http://127.0.0.1:${LOCAL_PORT}

  OPENJEV_BASE_URL=http://127.0.0.1:${LOCAL_PORT}

Leave this terminal open. Ctrl-C closes the tunnel.
EOF

# Supervise. IAP drops long-lived sessions; reconnect rather than silently
# leaving a dead port behind.
while true; do
  wait "${TUNNEL_PID}" 2>/dev/null || true
  "${SHUTTING_DOWN}" && break

  warn "$(date '+%H:%M:%S') tunnel dropped; reconnecting"
  stop_tunnel          # release the port the dead session may still hold
  "${SHUTTING_DOWN}" && break
  start_tunnel

  if wait_for_endpoint 60; then
    log "$(date '+%H:%M:%S') tunnel restored"
  else
    warn "endpoint did not come back; retrying"
  fi
done
