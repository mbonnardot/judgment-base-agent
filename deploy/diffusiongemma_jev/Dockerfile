# OpenJev (DiffusionGemma-26B-A4B) container for 24 GB NVIDIA L4 GPUs (Cloud Run / GCE).
# Keeps all upstream OpenJev defaults except capping OPENJEV_MAX_MODEL_LEN to 4096
# (since 65536 requires ~6.8 GiB KV cache on top of 18.15 GiB weights, exceeding 24 GB L4 VRAM).
FROM razorback16/openjev:0.4.0

ENV OPENJEV_PORT=8080 \
    OPENJEV_MAX_MODEL_LEN=4096 \
    OPENJEV_VLLM_ARGS="--enforce-eager"

EXPOSE 8080
