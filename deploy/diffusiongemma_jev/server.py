"""FastAPI 'OpenJev' / DiffusionGemma-as-Jev server for GCP Cloud Run (GPU & CPU).

Exposes POST /v1/system_one (and /v1/judgment) compatible with TypeSafe's Jev API,
scoring Choice, Score, and Noul primitives in parallel via single-step bubble-sheet
canvas denoising on Google's DiffusionGemma (DiffusionGemmaForBlockDiffusion).

This module implements the ``transformers`` engine only: a native single-step
Encoder + Bidirectional Decoder canvas forward pass using HuggingFace
``DiffusionGemmaForBlockDiffusion(input_ids, decoder_input_ids)``.

- On NVIDIA L4 GPU (>=16GB VRAM): loads ``google/diffusiongemma-26B-A4B-it``
  (4-bit NF4 / bfloat16).
- On CPU / fast smoke-test: loads
  ``trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion`` (the exact
  ``DiffusionGemmaForBlockDiffusion`` encoder-decoder architecture, 262k vocab).

The ``vllm`` engine is deliberately NOT implemented here. When
``DIFFUSIONGEMMA_ENGINE=vllm``, ``entrypoint.sh`` runs ``vllm serve`` plus the
upstream PR's own ``examples/features/diffusion_reads/structured_server.py``,
which already speaks this same Jev wire format and drives the canvas correctly
through ``diffusion_seed_canvas`` / ``diffusion_read_only``. Re-implementing
that here would mean one ``/v1/completions`` round trip per question — O(N)
instead of the O(1) single canvas read that is the entire point of the model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
import json
import math
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ENGINE_MODE = os.environ.get("DIFFUSIONGEMMA_ENGINE", "transformers").lower()
DEFAULT_TEMPERATURE = float(os.environ.get("DIFFUSIONGEMMA_TEMPERATURE", "1.0"))
MAX_DENOISING_STEPS = int(os.environ.get("DIFFUSIONGEMMA_DENOISING_STEPS", "1"))


def _resolve_default_model_id() -> str:
    explicit = os.environ.get("DIFFUSIONGEMMA_MODEL_ID")
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 15.0:
                return "google/diffusiongemma-26B-A4B-it"
    except Exception:
        pass
    return "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"


DEFAULT_MODEL_ID = _resolve_default_model_id()


class SystemOneRequest(BaseModel):
    """Incoming request matching TypeSafe's system_one / OpenJev schema."""

    state: Any
    questions: dict[str, dict[str, Any]]
    model: str = Field(default=DEFAULT_MODEL_ID)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, gt=0.0)


def _build_prompt(state: Any, key: str, spec: Mapping[str, Any]) -> str:
    q_type = str(spec.get("type", "choice")).lower()
    instructions = str(spec.get("instructions", ""))
    criteria = spec.get("criteria")
    state_json = json.dumps(state, ensure_ascii=False, default=str)

    if q_type == "choice":
        return (
            f"<|begin_of_text|>[STATE]\n{state_json}\n"
            f"[QUESTION: {key} (CHOICE)]\n{instructions}\n"
            f"[OPTIONS]\n{json.dumps(criteria, ensure_ascii=False)}\n"
            f"[ANSWER_BUBBLE]:"
        )
    if q_type == "score":
        legend = {str(i): str(label) for i, label in enumerate(criteria or [])}
        return (
            f"<|begin_of_text|>[STATE]\n{state_json}\n"
            f"[QUESTION: {key} (SCORE 0..{max(len(legend) - 1, 0)})]\n{instructions}\n"
            f"[SCALE_LEGEND]\n{json.dumps(legend, ensure_ascii=False)}\n"
            f"[ANSWER_BUBBLE_INDEX]:"
        )
    return (
        f"<|begin_of_text|>[STATE]\n{state_json}\n"
        f"[QUESTION: {key} (BOOLEAN NOUL)]\n{instructions}\n"
        f"[CRITERIA]\n{json.dumps(criteria or {}, ensure_ascii=False)}\n"
        f"[ANSWER_BUBBLE_BOOL (true/false)]:"
    )


def _entropy_confidence(probs: Sequence[float]) -> float:
    n = len(probs)
    if n <= 1:
        return 1.0
    max_h = math.log(n)
    h = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - (h / max_h)))


_hf_runtime: dict[str, Any] = {}


def _get_or_load_hf_model() -> tuple[Any, Any, str]:
    """Lazily load DiffusionGemmaForBlockDiffusion + AutoTokenizer for 1-step canvas scoring."""
    if "model" in _hf_runtime:
        return _hf_runtime["model"], _hf_runtime["tokenizer"], _hf_runtime["model_id"]

    import torch
    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    model_id = _resolve_default_model_id()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["dtype"] = torch.bfloat16
        if os.environ.get("LOAD_IN_4BIT", "true").lower() == "true" and "26B" in model_id:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                pass
    else:
        load_kwargs["dtype"] = torch.float32

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id, **load_kwargs)
    model.eval()
    _hf_runtime["model"] = model
    _hf_runtime["tokenizer"] = tokenizer
    _hf_runtime["model_id"] = model_id
    return model, tokenizer, model_id


def _candidate_token_ids(tokenizer: Any, label: str) -> list[int]:
    """Return candidate first-token IDs for a choice/score/noul option in the Gemma 262k vocab."""
    ids: list[int] = []
    for variant in (label, f" {label}", label.lower(), f" {label.lower()}", label.capitalize()):
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return list(dict.fromkeys(ids))


def _evaluate_canvas_transformers_sync(req: SystemOneRequest) -> dict[str, Any]:
    """Execute 1-step Encoder prefill + Bidirectional Decoder Bubble-Sheet Canvas pass."""
    import torch

    t0 = time.perf_counter()
    model, tokenizer, loaded_model_id = _get_or_load_hf_model()
    device = next(model.parameters()).device
    canvas_length = int(getattr(model.config, "canvas_length", 32))

    # 1. Build unified [STATE] + [QUESTIONS] context prompt for the autoregressive Encoder
    state_json = json.dumps(req.state, ensure_ascii=False, default=str)
    questions_summary = json.dumps(req.questions, ensure_ascii=False, default=str)
    prompt_text = (
        f"<|begin_of_text|>[SYSTEM_ONE_BUBBLE_SHEET]\n"
        f"[STATE]\n{state_json}\n"
        f"[QUESTIONS]\n{questions_summary}\n"
        f"[FILL_CANVAS_JSON]:\n"
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    in_tokens = int(prompt_ids.shape[1])

    # 2. Build the Bubble-Sheet Canvas (`decoder_input_ids`) placing each question's slot at a fixed canvas position
    q_keys = list(req.questions.keys())
    canvas_tokens: list[int] = []
    slot_positions: dict[str, int] = {}

    open_brace = tokenizer.encode("{", add_special_tokens=False)
    canvas_tokens.extend(open_brace[:1] if open_brace else [2])

    for idx, key in enumerate(q_keys):
        prefix_str = f'"{key}": ' if idx == 0 else f', "{key}": '
        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        if len(canvas_tokens) + len(prefix_ids) + 2 <= canvas_length:
            canvas_tokens.extend(prefix_ids)
            slot_positions[key] = len(canvas_tokens)
            # Initialize bubble slot with space/pad token to be refined by bidirectional diffusion decoder
            space_ids = tokenizer.encode(" ", add_special_tokens=False)
            canvas_tokens.append(int(space_ids[0]) if space_ids else 0)
        else:
            slot_positions[key] = max(0, min(idx, canvas_length - 1))

    while len(canvas_tokens) < canvas_length:
        canvas_tokens.append(0)
    canvas_tokens = canvas_tokens[:canvas_length]
    decoder_input_ids = torch.tensor([canvas_tokens], dtype=torch.long, device=device)

    # 3. Execute 1 single forward pass (`max_denoising_steps=1`): Encoder KV prefill + Bidirectional Decoder
    with torch.no_grad():
        outputs = model(input_ids=prompt_ids, decoder_input_ids=decoder_input_ids)
        # outputs.logits has shape (1, canvas_length, vocab_size=262144)
        canvas_logits = outputs.logits[0]  # (canvas_length, vocab_size)
        log_probs_canvas = torch.log_softmax(canvas_logits, dim=-1)

    temp = max(req.temperature, 1e-4)
    choices: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    nouls: dict[str, Any] = {}

    for key, spec in req.questions.items():
        pos = slot_positions.get(key, 0)
        slot_lps = log_probs_canvas[pos]
        q_type = str(spec.get("type", "choice")).lower()

        if q_type == "choice":
            crit = spec.get("criteria") or {}
            options = (
                list(crit.keys())
                if isinstance(crit, Mapping)
                else [str(x) for x in crit]
            )
            option_lps: list[float] = []
            for opt in options:
                cand_ids = _candidate_token_ids(tokenizer, str(opt))
                best_lp = max((float(slot_lps[tid].item()) for tid in cand_ids), default=-12.0)
                option_lps.append(best_lp / temp)
            max_lp = max(option_lps) if option_lps else 0.0
            exps = [math.exp(x - max_lp) for x in option_lps]
            total = sum(exps) or 1.0
            probs = [e / total for e in exps]
            prob_map = {str(o): float(p) for o, p in zip(options, probs, strict=True)}
            best = max(prob_map, key=lambda k: prob_map[k])
            choices[key] = {
                "choice": best,
                "probabilities": prob_map,
                "confidence": _entropy_confidence(probs),
            }

        elif q_type == "score":
            criteria_list = [str(x) for x in (spec.get("criteria") or [])]
            level_lps: list[float] = []
            for idx, lbl in enumerate(criteria_list):
                cand_ids = _candidate_token_ids(tokenizer, str(idx)) + _candidate_token_ids(tokenizer, lbl)
                best_lp = max((float(slot_lps[tid].item()) for tid in cand_ids), default=-12.0)
                level_lps.append(best_lp / temp)
            max_lp = max(level_lps) if level_lps else 0.0
            exps = [math.exp(x - max_lp) for x in level_lps]
            total = sum(exps) or 1.0
            probs = [e / total for e in exps]
            expected = sum(float(i) * p for i, p in enumerate(probs))
            legend = {str(i): lbl for i, lbl in enumerate(criteria_list)}
            prob_map = {lbl: float(p) for lbl, p in zip(criteria_list, probs, strict=True)}
            scores[key] = {
                "score": expected,
                "legend": legend,
                "probabilities": prob_map,
                "confidence": _entropy_confidence(probs),
            }

        else:
            true_ids = (
                _candidate_token_ids(tokenizer, "true")
                + _candidate_token_ids(tokenizer, "yes")
                + _candidate_token_ids(tokenizer, "1")
            )
            false_ids = (
                _candidate_token_ids(tokenizer, "false")
                + _candidate_token_ids(tokenizer, "no")
                + _candidate_token_ids(tokenizer, "0")
            )
            lp_true = max((float(slot_lps[tid].item()) for tid in true_ids), default=-12.0) / temp
            lp_false = max((float(slot_lps[tid].item()) for tid in false_ids), default=-12.0) / temp
            m = max(lp_true, lp_false)
            p_t = math.exp(lp_true - m)
            p_f = math.exp(lp_false - m)
            nouls[key] = {"noul": float(p_t / (p_t + p_f))}

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "model": loaded_model_id,
        "engine": "transformers_diffusion_canvas_1step",
        "latency_ms": latency_ms,
        "choices": choices,
        "scores": scores,
        "nouls": nouls,
        "usage": {"input_tokens": in_tokens, "output_tokens": len(q_keys)},
    }


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("PRELOAD_MODEL_ON_STARTUP", "false").lower() == "true":
        await asyncio.to_thread(_get_or_load_hf_model)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DiffusionGemma-as-Jev (OpenJev Cloud Run Server)",
        version="1.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": ENGINE_MODE,
            "model": _hf_runtime.get("model_id", DEFAULT_MODEL_ID),
            "max_denoising_steps": MAX_DENOISING_STEPS,
        }

    @app.post("/v1/system_one")
    @app.post("/v1/judgment")
    async def evaluate_system_one(req: SystemOneRequest) -> dict[str, Any]:
        if not req.questions:
            return {
                "model": DEFAULT_MODEL_ID,
                "choices": {},
                "scores": {},
                "nouls": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        try:
            return await asyncio.to_thread(_evaluate_canvas_transformers_sync, req)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"DiffusionGemmaForBlockDiffusion error: {exc}"
            ) from exc

    return app


app = create_app()
