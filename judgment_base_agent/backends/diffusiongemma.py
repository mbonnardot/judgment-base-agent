"""OpenJev (DiffusionGemma-as-Jev) System One backend implementation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import os
from typing import Any

import httpx

from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    JudgmentResult,
    JudgmentUsage,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)

TransportPostFn = Callable[[str, dict[str, Any], dict[str, str]], Awaitable[dict[str, Any]]]

OPENJEV_MODEL_ALIASES = frozenset({"openjev-latest", "openjev-0.1", "jev-latest", "jev-preview"})


def _normalize_openjev_model(model: str | None) -> str:
    """Normalize model names to valid OpenJev aliases ('openjev-latest', 'jev-latest', etc.)."""
    if not model:
        return "openjev-latest"
    cleaned = model.strip()
    if cleaned in OPENJEV_MODEL_ALIASES:
        return cleaned
    if cleaned in {"judgment-latest", "system-one"} or "diffusiongemma" in cleaned.lower():
        return "openjev-latest"
    return cleaned


def _question_kind(q: Any) -> str:
    """Identify primitive type ('choice', 'score', 'noul')."""
    cls_name = type(q).__name__.lower()
    if isinstance(q, Choice) or cls_name == "choice":
        return "choice"
    if isinstance(q, Score) or cls_name == "score":
        return "score"
    if isinstance(q, Noul) or cls_name == "noul":
        return "noul"
    if isinstance(q, Mapping):
        return str(q.get("type", "")).lower()
    return "unknown"


def _serialize_question(q: Any) -> dict[str, Any]:
    """Serialize Choice, Score, Noul, or dict into an OpenJev / Jev question spec."""
    if isinstance(q, Choice):
        return {
            "type": "choice",
            "instructions": q.instructions,
            "criteria": q.normalized_criteria(),
        }
    if isinstance(q, Score):
        return {
            "type": "score",
            "instructions": q.instructions,
            "criteria": list(q.normalized_criteria()),
        }
    if isinstance(q, Noul):
        return {
            "type": "noul",
            "instructions": q.instructions,
            "criteria": dict(q.criteria) if q.criteria else None,
        }
    if isinstance(q, Mapping):
        return dict(q)
    raise JudgmentConfigError(f"Unsupported question primitive for DiffusionGemmaBackend: {type(q)!r}")


def _pick_answer(
    key: str,
    answers: Mapping[str, Any],
    legacy: Mapping[str, Any],
) -> Any:
    """Resolve one answer from OpenJev's 'answers' map (or legacy per-kind maps)."""
    if key in answers:
        return answers[key]
    if key in legacy:
        return legacy[key]
    raise KeyError(key)


class DiffusionGemmaBackend:
    """Judgment backend targeting OpenJev (`razorback16/openjev`) on NVIDIA GPU (vLLM), Apple Silicon (MLX), or Codiv."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str = "openjev-latest",
        confidence_floor: float = 0.50,
        timeout: float = 30.0,
        transport_post: TransportPostFn | None = None,
        system_one_path: str | None = None,
        steps: int | None = None,
        samples: int | None = None,
        think: int | None = None,
        sequential: bool | None = None,
        images: Sequence[str | Mapping[str, str]] | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OPENJEV_BASE_URL")
            or os.environ.get("DIFFUSIONGEMMA_JEV_URL")
            or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("OPENJEV_API_KEY")
            or os.environ.get("DIFFUSIONGEMMA_API_KEY")
        )
        self.system_one_path = (
            system_one_path
            or os.environ.get("DIFFUSIONGEMMA_SYSTEM_ONE_PATH")
            or "/v1/systemone"
        )
        self.default_model = _normalize_openjev_model(
            os.environ.get("OPENJEV_MODEL_ID")
            or os.environ.get("DIFFUSIONGEMMA_MODEL_ID")
            or default_model
        )
        self.confidence_floor = confidence_floor
        self.timeout = timeout
        self.transport_post = transport_post
        self.steps = steps
        self.samples = samples
        self.think = think
        self.sequential = sequential
        self.images = list(images) if images is not None else None
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> DiffusionGemmaBackend:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def close(self) -> None:
        """Alias for aclose()."""
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_endpoint(self, path: str) -> str:
        """Join base_url and path, tolerating base URLs that already carry /v1 or the full route."""
        normalized = "/" + path.strip("/")
        if self.base_url.endswith(normalized):
            return self.base_url
        leaf = normalized.rsplit("/", 1)[-1]
        if self.base_url.endswith("/v1") and normalized.startswith("/v1/"):
            return f"{self.base_url}/{leaf}"
        return f"{self.base_url}{normalized}"

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._headers()
        if self.transport_post is not None:
            return await self.transport_post(url, payload, headers)
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        resp = await self._http_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return dict(resp.json())

    async def evaluate(
        self,
        state: Any,
        questions: Mapping[str, Any],
        model: str | None = None,
    ) -> JudgmentResult:
        """Evaluate questions against OpenJev's POST /v1/systemone endpoint."""
        target_model = _normalize_openjev_model(model or self.default_model)
        if not questions:
            return JudgmentResult(model=target_model)

        if not self.base_url:
            raise JudgmentConfigError(
                "OPENJEV_BASE_URL or DIFFUSIONGEMMA_JEV_URL (or explicit base_url) is required for DiffusionGemmaBackend."
            )

        try:
            endpoint = self._resolve_endpoint(self.system_one_path)
            serialized_questions = {
                str(k): _serialize_question(v) for k, v in questions.items()
            }
            payload: dict[str, Any] = {
                "state": state,
                "questions": serialized_questions,
                "model": target_model,
            }
            if self.steps is not None:
                payload["steps"] = self.steps
            if self.samples is not None:
                payload["samples"] = self.samples
            if self.think is not None:
                payload["think"] = self.think
            if self.sequential is not None:
                payload["sequential"] = self.sequential
            if self.images is not None:
                payload["images"] = self.images

            data = await self._post_json(endpoint, payload)

            choices_raw = data.get("choices", {}) or {}
            scores_raw = data.get("scores", {}) or {}
            nouls_raw = data.get("nouls", {}) or {}
            answers_raw = data.get("answers", {}) or {}

            parsed_choices: dict[str, ChoiceJudgment] = {}
            parsed_scores: dict[str, ScoreJudgment] = {}
            parsed_nouls: dict[str, NoulJudgment] = {}

            for key, orig_q in questions.items():
                kind = _question_kind(orig_q)
                if kind == "choice":
                    c_item = _pick_answer(str(key), answers_raw, choices_raw)
                    parsed_choices[key] = ChoiceJudgment.from_raw(
                        choice=str(c_item["choice"]),
                        probabilities=dict(c_item.get("probabilities") or {}),
                        confidence=float(c_item.get("confidence", 1.0)),
                        confidence_floor=self.confidence_floor,
                    )
                elif kind == "score":
                    s_item = _pick_answer(str(key), answers_raw, scores_raw)
                    parsed_scores[key] = ScoreJudgment.from_raw(
                        score=float(s_item["score"]),
                        legend=dict(s_item.get("legend") or {}),
                        probabilities=dict(s_item.get("probabilities") or {}),
                        confidence=float(s_item.get("confidence", 1.0)),
                        confidence_floor=self.confidence_floor,
                    )
                elif kind == "noul":
                    n_item = _pick_answer(str(key), answers_raw, nouls_raw)
                    noul_val = n_item["noul"] if isinstance(n_item, Mapping) else n_item
                    parsed_nouls[key] = NoulJudgment(noul=float(noul_val))

            raw_usage = data.get("usage")
            usage = (
                JudgmentUsage(
                    input_tokens=int(raw_usage.get("input_tokens", 0)),
                    output_tokens=int(raw_usage.get("output_tokens", 0)),
                )
                if isinstance(raw_usage, Mapping)
                else None
            )

            return JudgmentResult(
                choices=parsed_choices,
                scores=parsed_scores,
                nouls=parsed_nouls,
                model=str(data.get("model", target_model)),
                usage=usage,
            )
        except JudgmentConfigError:
            raise
        except Exception as exc:
            raise JudgmentEvaluationError(
                f"OpenJev evaluation failed against '{self.base_url}': {exc}"
            ) from exc
