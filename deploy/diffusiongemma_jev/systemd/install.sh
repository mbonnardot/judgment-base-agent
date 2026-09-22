#!/usr/bin/env bash
#
# Install the unified OpenJev systemd unit on the GPU VM.
#
# Run via deploy_vm.sh:
#   ./deploy/diffusiongemma_jev/deploy_vm.sh --systemd
#
# Or directly on the VM as root:
#   sudo bash ~/systemd/install.sh
set -euo pipefail

UNIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR=/etc/systemd/system

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: must run as root (use: sudo bash $0)" >&2
  exit 1
fi

for required in \
  /opt/venv/vllm/bin/python \
  /opt/hf
do
  if [[ ! -e "${required}" ]]; then
    echo "error: expected path missing on this VM: ${required}" >&2
    exit 1
  fi
done

if ! /opt/venv/vllm/bin/python -c "import openjev" >/dev/null 2>&1; then
  echo "==> Installing razorback16/openjev@v0.3.0 into /opt/venv/vllm"
  /opt/venv/vllm/bin/pip install --no-deps git+https://github.com/razorback16/openjev.git@v0.3.0
fi

echo "==> Stopping any legacy vLLM / Docker / hand-launched processes"
systemctl stop djev-vllm.service 2>/dev/null || true
systemctl disable djev-vllm.service 2>/dev/null || true
rm -f "${SYSTEMD_DIR}/djev-vllm.service"
docker rm -f openjev 2>/dev/null || true
pkill -f 'vllm serve RedHatAI' 2>/dev/null || true
pkill -f 'structured_server.py' 2>/dev/null || true
pkill -f 'python -m openjev' 2>/dev/null || true
sleep 3

echo "==> Installing djev-jev.service into ${SYSTEMD_DIR}"
install -m 0644 "${UNIT_DIR}/djev-jev.service" "${SYSTEMD_DIR}/djev-jev.service"

echo "==> Enabling and starting djev-jev.service"
systemctl daemon-reload
systemctl enable djev-jev.service
systemctl restart djev-jev.service

cat <<'EOF'

==> Installed unified OpenJev systemd unit (djev-jev.service).
    The engine takes ~3-5 minutes to load weights on a cold start.

Watch progress:
  journalctl -u djev-jev -f
  tail -f /var/log/djev-jev.log

Verify when ready:
  curl -sf http://127.0.0.1:8011/health && echo OK
EOF
