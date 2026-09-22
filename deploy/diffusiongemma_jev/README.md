# Deploying OpenJev (`DiffusionGemma-26B-A4B`) for `judgment-base-agent`

This directory packages [`razorback16/openjev`](https://github.com/razorback16/openjev) (`v0.3.0`) as the open-source **System One / Jev** decision server (`POST /v1/systemone`) for `judgment-base-agent`.

## Why OpenJev?

1. **Prebuilt Image & Precompiled vLLM Wheels (`razorback16/openjev:0.3.0`):**
   - Ships [`razorback16/vllm@baa8338`](https://github.com/razorback16/vllm/tree/structured-reads-57250-rebased) (`vllm#57250` structured canvas reads plus 4 upstream crash fixes: `vllm#57416` prefill logit rows, `vllm#54309` image inputs, sampler `torch.compile` fallback dtype fix, and mixed-batch logprob stash widths) using `VLLM_USE_PRECOMPILED=1` — no 60-minute CUDA source compilation.
2. **Automatic Single-Token Label & Canvas Management:**
   - Normalizes all schema question keys internally to `q1..qN`, assigns verified single-token labels (`A..Z`, `a..z`, `AA..ZZ`), switches to a compact `"indexed"` layout when `>10` questions are batched, and chunks large batches (~12 questions per read) in parallel.
3. **Multiple Runtime Targets:**
   - **Codiv Free Hosted OpenJev (`https://api.codiv.ai`):** 100M free input tokens, zero infrastructure required.
   - **Apple Silicon Mac (`OPENJEV_BACKEND=mlx`):** Runs `mlx-community/diffusiongemma-26B-A4B-it-4bit` in-process (~16 GB unified memory, no Docker or vLLM required).
   - **NVIDIA L4 / GPU Container (`razorback16/openjev:0.3.0`):** Serves both `POST /v1/systemone` and OpenAI-compatible `POST /v1/chat/completions` (`model="diffusiongemma-26b"`) on the same GPU.

---

## 1. Using OpenJev with Existing ADK Agents (Zero Code Changes)

Set `OPENJEV_BASE_URL` (or `DIFFUSIONGEMMA_JEV_URL`) in `examples/.env` or your shell:

```bash
# Option 1: Free hosted OpenJev on Codiv
export OPENJEV_BASE_URL="https://api.codiv.ai"
export OPENJEV_API_KEY="sk-codiv-..."

# Option 2: Local Docker / Apple Silicon MLX / IAP GPU Tunnel
export OPENJEV_BASE_URL="http://127.0.0.1:8080"
```

When `OPENJEV_BASE_URL` or `DIFFUSIONGEMMA_JEV_URL` is set, `TypeSafeBackend()` automatically delegates every `JudgmentAgent`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`, and `JudgmentRubricEvaluator` call to [`DiffusionGemmaBackend`](../../judgment_base_agent/backends/diffusiongemma.py).

Or instantiate [`DiffusionGemmaBackend`](../../judgment_base_agent/backends/diffusiongemma.py) explicitly (including optional OpenJev extensions `steps`, `samples`, `think`, `sequential`, `images`):

```python
from judgment_base_agent import DiffusionGemmaBackend, JudgmentSwitch

router = JudgmentSwitch(
    name="openjev_router",
    routes={"billing": "Billing questions", "tech_support": "Technical bugs"},
    backend=DiffusionGemmaBackend(
        base_url="http://127.0.0.1:8080",
        steps=1,
        samples=2,
    ),
)
```

---

## 2. Running OpenJev Locally or on GCP

### A. Docker (`razorback16/openjev:0.3.0` on NVIDIA GPU)

```bash
docker run -d --gpus all --ipc=host -p 127.0.0.1:8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  razorback16/openjev:0.3.0
```

### B. Apple Silicon Mac (`OPENJEV_BACKEND=mlx`, no Docker/vLLM)

```bash
pip install "git+https://github.com/razorback16/openjev.git#egg=openjev[mlx]"
OPENJEV_BACKEND=mlx python -m openjev
```

### C. One-Command GCP Cloud Run Deployment (`1x NVIDIA L4`)

```bash
export PROJECT_ID="your-gcp-project-id"
./deploy/diffusiongemma_jev/deploy_cloud_run.sh
```

### D. GCE L4 GPU VM Deployment (`deploy_vm.sh` + IAP Tunnel)

Provision (or update) a GCE `g2-standard-8` (1x NVIDIA L4) VM and deploy OpenJev via Docker (default) or `systemd`:

```bash
# Deploy to existing VM (or pass --create to provision a new L4 VM)
./deploy/diffusiongemma_jev/deploy_vm.sh --create

# Or deploy using the bare-metal systemd service (/opt/venv/vllm):
./deploy/diffusiongemma_jev/deploy_vm.sh --systemd

# Open self-healing IAP tunnel on localhost:8011
./scripts/connect_gpu.sh
```
