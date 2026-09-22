#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-medquad-assistant-capstone}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-openjev}"
REPO_NAME="${REPO_NAME:-medquad-repo}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo " Deploying OpenJev (DiffusionGemma-26B-A4B) to GCP Cloud Run (L4 GPU)"
echo " Project : ${PROJECT_ID}"
echo " Region  : ${REGION}"
echo " Service : ${SERVICE_NAME}"
echo " Image   : ${IMAGE_URI}"
echo "========================================================================"

# 1. Ensure Artifact Registry repository exists
if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" >/dev/null 2>&1; then
  echo "[1/3] Creating Artifact Registry repository ${REPO_NAME}..."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --project="${PROJECT_ID}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="OpenJev Container Repository"
else
  echo "[1/3] Using existing Artifact Registry repository ${REPO_NAME}."
fi

# 2. Build and push container image via Cloud Build
echo "[2/3] Building OpenJev container image via Cloud Build..."
gcloud builds submit "${SCRIPT_DIR}" \
  --quiet \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}"

# 3. Deploy to Cloud Run GPU (1x NVIDIA L4 24GB VRAM)
echo "[3/3] Deploying ${SERVICE_NAME} to Cloud Run (1x NVIDIA L4 GPU)..."
gcloud run deploy "${SERVICE_NAME}" \
  --quiet \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_URI}" \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu=8 \
  --memory=32Gi \
  --no-cpu-throttling \
  --concurrency=16 \
  --timeout=300 \
  --no-allow-unauthenticated \
  --set-env-vars="OPENJEV_PORT=8080,OPENJEV_MAX_MODEL_LEN=16384,OPENJEV_GPU_UTIL=0.90"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo ""
echo "========================================================================"
echo " OpenJev Deployed Successfully!"
echo " URL: ${SERVICE_URL}"
echo " Export for ADK agents:"
echo "   export OPENJEV_BASE_URL=\"${SERVICE_URL}\""
echo "========================================================================"
