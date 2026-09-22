# Cloud Run / GPU deployment image built directly on razorback16/openjev
# (includes vLLM with PR #57250 structured reads + 4 sampler/image/logit fixes,
# precompiled CUDA wheels, and the OpenJev /v1/systemone + /v1/chat/completions server).
FROM razorback16/openjev:0.3.0

ENV PYTHONUNBUFFERED=1 \
    OPENJEV_HOST=0.0.0.0 \
    OPENJEV_PORT=8080 \
    OPENJEV_MODEL=nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
    OPENJEV_CANVAS=64 \
    OPENJEV_MAX_MODEL_LEN=16384 \
    OPENJEV_GPU_UTIL=0.90

EXPOSE 8080
