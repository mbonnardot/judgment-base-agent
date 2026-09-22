"""Google DiffusionGemma-as-Jev ('djev' / OpenJev) backend implementation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import json
import math
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


def _env_flag(name: str, *, default: bool) -> bool:
    """Read a boolean environment variable, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    """Serialize Choice, Score, Noul, or dict into a JSON-ready question spec."""
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


def build_bubble_sheet_prompt(
    state: Any,
    question_key: str,
    question: Any,
) -> str:
    """Build a single-step bubble-sheet prompt for DiffusionGemma canvas scoring."""
    spec = _serialize_question(question)
    q_type = spec["type"]
    instructions = spec["instructions"]
    criteria = spec.get("criteria")
    state_json = json.dumps(state, ensure_ascii=False, default=str)

    if q_type == "choice":
        options_str = json.dumps(criteria, ensure_ascii=False)
        return (
            f"<|begin_of_text|>[STATE]\n{state_json}\n"
            f"[QUESTION: {question_key} (CHOICE)]\n{instructions}\n"
            f"[OPTIONS]\n{options_str}\n"
            f"[ANSWER_BUBBLE]:"
        )
    if q_type == "score":
        legend = {str(i): str(label) for i, label in enumerate(criteria or [])}
        return (
            f"<|begin_of_text|>[STATE]\n{state_json}\n"
            f"[QUESTION: {question_key} (SCORE 0..{max(len(legend) - 1, 0)})]\n{instructions}\n"
            f"[SCALE_LEGEND]\n{json.dumps(legend, ensure_ascii=False)}\n"
            f"[ANSWER_BUBBLE_INDEX]:"
        )
    return (
        f"<|begin_of_text|>[STATE]\n{state_json}\n"
        f"[QUESTION: {question_key} (BOOLEAN NOUL)]\n{instructions}\n"
        f"[CRITERIA]\n{json.dumps(criteria or {}, ensure_ascii=False)}\n"
        f"[ANSWER_BUBBLE_BOOL (true/false)]:"
    )


def _match_token_logprob(
    candidate: str,
    top_logprobs: Mapping[str, float],
    default_logprob: float = -12.0,
) -> float:
    """Find the highest logprob among token variants matching candidate (e.g. ' billing', 'billing')."""
    cand_lower = candidate.strip().lower()
    best: float | None = None
    for tok, lp in top_logprobs.items():
        cleaned = str(tok).strip().strip('"').strip("'").lower()
        if cleaned == cand_lower or (cand_lower and cand_lower.startswith(cleaned) and len(cleaned) >= 2):
            val = float(lp)
            if best is None or val > best:
                best = val
    return best if best is not None else default_logprob


def _normalized_entropy_confidence(probs: Sequence[float]) -> float:
    """Compute calibrated confidence in [0, 1] as 1 - normalized Shannon entropy."""
    n = len(probs)
    if n <= 1:
        return 1.0
    max_entropy = math.log(n)
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0.0)
    clarity = max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))
    return clarity


def compute_choice_from_logprobs(
    options: Sequence[str],
    top_logprobs: Mapping[str, float],
    temperature: float = 1.0,
    confidence_floor: float = 0.50,
) -> ChoiceJudgment:
    """Convert single-step canvas slot logprobs into a calibrated ChoiceJudgment."""
    temp = max(temperature, 1e-4)
    raw_lps = [_match_token_logprob(opt, top_logprobs) / temp for opt in options]
    max_lp = max(raw_lps) if raw_lps else 0.0
    exps = [math.exp(lp - max_lp) for lp in raw_lps]
    total = sum(exps) or 1.0
    probs_list = [e / total for e in exps]
    probs_map = {str(opt): float(p) for opt, p in zip(options, probs_list, strict=True)}
    best_opt = max(probs_map, key=lambda k: probs_map[k])
    confidence = _normalized_entropy_confidence(probs_list)
    return ChoiceJudgment.from_raw(
        choice=best_opt,
        probabilities=probs_map,
        confidence=confidence,
        confidence_floor=confidence_floor,
    )


def compute_score_from_logprobs(
    criteria: Sequence[str],
    top_logprobs: Mapping[str, float],
    temperature: float = 1.0,
    confidence_floor: float = 0.50,
) -> ScoreJudgment:
    """Convert single-step canvas slot logprobs into an expected-value ScoreJudgment."""
    temp = max(temperature, 1e-4)
    indices = [str(i) for i in range(len(criteria))]
    raw_lps = [
        max(
            _match_token_logprob(idx, top_logprobs),
            _match_token_logprob(label, top_logprobs),
        )
        / temp
        for idx, label in zip(indices, criteria, strict=True)
    ]
    max_lp = max(raw_lps) if raw_lps else 0.0
    exps = [math.exp(lp - max_lp) for lp in raw_lps]
    total = sum(exps) or 1.0
    probs_list = [e / total for e in exps]
    expected_score = sum(float(i) * p for i, p in enumerate(probs_list))
    legend = {str(i): str(label) for i, label in enumerate(criteria)}
    probs_map = {str(label): float(p) for label, p in zip(criteria, probs_list, strict=True)}
    confidence = _normalized_entropy_confidence(probs_list)
    return ScoreJudgment.from_raw(
        score=expected_score,
        legend=legend,
        probabilities=probs_map,
        confidence=confidence,
        confidence_floor=confidence_floor,
    )


def compute_noul_from_logprobs(
    top_logprobs: Mapping[str, float],
    temperature: float = 1.0,
) -> NoulJudgment:
    """Convert single-step canvas slot logprobs for ('true', 'false') into a NoulJudgment."""
    temp = max(temperature, 1e-4)
    lp_true = max(
        _match_token_logprob("true", top_logprobs),
        _match_token_logprob("yes", top_logprobs),
        _match_token_logprob("1", top_logprobs),
    ) / temp
    lp_false = max(
        _match_token_logprob("false", top_logprobs),
        _match_token_logprob("no", top_logprobs),
        _match_token_logprob("0", top_logprobs),
    ) / temp
    max_lp = max(lp_true, lp_false)
    p_true = math.exp(lp_true - max_lp)
    p_false = math.exp(lp_false - max_lp)
    prob = p_true / (p_true + p_false)
    return NoulJudgment(noul=float(prob))


def _pick_answer(
    key: str,
    alias: str | None,
    answers: Mapping[str, Any],
    legacy: Mapping[str, Any],
) -> Any:
    """Resolve one answer, preferring the wire alias then the original key.

    Servers differ in which key they answer under: the reference
    ``structured_server.py`` echoes back whatever key it received (so, the
    alias), while mocks and the managed Jev API answer under the original
    schema key. Try both against the modern ``answers`` map before falling back
    to the legacy per-kind (``choices`` / ``scores`` / ``nouls``) maps.
    """
    for source in (answers, legacy):
        for candidate in (alias, key):
            if candidate is not None and candidate in source:
                return source[candidate]
    raise KeyError(key)


class DiffusionGemmaBackend:
    """Judgment backend for Google's DiffusionGemma-as-Jev ('djev' / OpenJev on GCP Cloud Run or vLLM)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str = "diffusiongemma-26B-A4B-it-NVFP4",
        confidence_floor: float = 0.50,
        temperature: float = 1.2,
        mode: str = "system_one",
        timeout: float = 30.0,
        transport_post: TransportPostFn | None = None,
        system_one_path: str | None = None,
        alias_question_keys: bool | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("DIFFUSIONGEMMA_JEV_URL")
            or os.environ.get("OPENJEV_BASE_URL")
            or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("DIFFUSIONGEMMA_API_KEY")
            or os.environ.get("OPENJEV_API_KEY")
        )
        # Our own container serves "/v1/system_one". The reference server shipped
        # in vLLM PR #57250 serves "/v1/systemone" (no underscore). Keeping this
        # configurable lets the same agents target either without code changes.
        self.system_one_path = (
            system_one_path
            or os.environ.get("DIFFUSIONGEMMA_SYSTEM_ONE_PATH")
            or "/v1/system_one"
        )
        # The reference server in vLLM PR #57250 lays every question out as a row
        # in one shared canvas template, printing the question key immediately
        # next to its answer slot. Keys whose trailing characters perturb how the
        # adjacent label tokenizes make the server reject its own default "no"
        # label ("label 'no' is not a single token" / "labels do not share one
        # template slot"), but only once enough rows share the canvas. JudgmentMap
        # emits exactly the offending shape ("item_0__urgency_score"). Sending
        # positional keys ("q0", "q1", ...) on the wire and mapping the answers
        # back sidesteps the bug without touching any schema or agent.
        #
        # Tradeoff: opaque keys drop a weak semantic hint from the prompt.
        # Measured impact on review_triage_batch was within noise (2.98/2.02/1.00
        # vs 3.00/2.00/1.19 on managed Jev), but the opt-out exists for backends
        # that read the key as signal.
        self.alias_question_keys = (
            alias_question_keys
            if alias_question_keys is not None
            else _env_flag("DIFFUSIONGEMMA_ALIAS_QUESTION_KEYS", default=True)
        )
        self.default_model = default_model
        self.confidence_floor = confidence_floor
        self.temperature = temperature
        self.mode = mode.lower()
        self.timeout = timeout
        self.transport_post = transport_post
        self._http_client: httpx.AsyncClient | None = None

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
        """Join base_url and path, tolerating base URLs that already carry it.

        Accepts bases like "https://host", "https://host/v1", or a base that
        already ends in the full path, and always yields exactly one copy.
        """
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
        """Evaluate questions against the DiffusionGemma-as-Jev Cloud Run container or vLLM endpoint."""
        target_model = model or self.default_model
        if not questions:
            return JudgmentResult(model=target_model)

        if not self.base_url:
            raise JudgmentConfigError(
                "DIFFUSIONGEMMA_JEV_URL (or explicit base_url) is required for DiffusionGemmaBackend."
            )

        try:
            if self.mode == "vllm":
                return await self._evaluate_via_vllm(state, questions, target_model)
            return await self._evaluate_via_system_one(state, questions, target_model)
        except JudgmentConfigError:
            raise
        except Exception as exc:
            raise JudgmentEvaluationError(
                f"DiffusionGemma evaluation failed against '{self.base_url}': {exc}"
            ) from exc

    async def _evaluate_via_system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        target_model: str,
    ) -> JudgmentResult:
        endpoint = self._resolve_endpoint(self.system_one_path)
        # Positional aliases keep question keys inert on the wire; see the
        # comment on self.alias_question_keys for the upstream bug this dodges.
        alias_for: dict[Any, str] = (
            {key: f"q{i}" for i, key in enumerate(questions)}
            if self.alias_question_keys
            else {}
        )
        serialized_questions = {
            alias_for.get(k, str(k)): _serialize_question(v) for k, v in questions.items()
        }
        payload = {
            "state": state,
            "questions": serialized_questions,
            "model": target_model,
            "temperature": self.temperature,
        }
        data = await self._post_json(endpoint, payload)

        choices_raw = data.get("choices", {}) or {}
        scores_raw = data.get("scores", {}) or {}
        nouls_raw = data.get("nouls", {}) or {}
        answers_raw = data.get("answers", {}) or {}

        parsed_choices: dict[str, ChoiceJudgment] = {}
        parsed_scores: dict[str, ScoreJudgment] = {}
        parsed_nouls: dict[str, NoulJudgment] = {}

        for key, orig_q in questions.items():
            alias = alias_for.get(key)
            kind = _question_kind(orig_q)
            if kind == "choice":
                c_item = _pick_answer(str(key), alias, answers_raw, choices_raw)
                parsed_choices[key] = ChoiceJudgment.from_raw(
                    choice=str(c_item["choice"]),
                    probabilities=dict(c_item.get("probabilities") or {}),
                    confidence=float(c_item.get("confidence", 1.0)),
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "score":
                s_item = _pick_answer(str(key), alias, answers_raw, scores_raw)
                parsed_scores[key] = ScoreJudgment.from_raw(
                    score=float(s_item["score"]),
                    legend=dict(s_item.get("legend") or {}),
                    probabilities=dict(s_item.get("probabilities") or {}),
                    confidence=float(s_item.get("confidence", 1.0)),
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "noul":
                n_item = _pick_answer(str(key), alias, answers_raw, nouls_raw)
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

    async def _evaluate_via_vllm(
        self,
        state: Any,
        questions: Mapping[str, Any],
        target_model: str,
    ) -> JudgmentResult:
        endpoint = (
            self.base_url
            if self.base_url.endswith("/completions")
            else f"{self.base_url}/v1/completions"
            if not self.base_url.endswith("/v1")
            else f"{self.base_url}/completions"
        )
        parsed_choices: dict[str, ChoiceJudgment] = {}
        parsed_scores: dict[str, ScoreJudgment] = {}
        parsed_nouls: dict[str, NoulJudgment] = {}
        total_in = 0
        total_out = 0

        for key, orig_q in questions.items():
            spec = _serialize_question(orig_q)
            prompt = build_bubble_sheet_prompt(state, key, orig_q)
            payload = {
                "model": target_model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": 20,
                "extra_body": {"max_denoising_steps": 1},
            }
            resp_data = await self._post_json(endpoint, payload)
            choices_list = resp_data.get("choices") or [{}]
            top_lps_list = (
                choices_list[0].get("logprobs", {}).get("top_logprobs") or [{}]
            )
            top_logprobs: Mapping[str, float] = top_lps_list[0] if top_lps_list else {}

            usage_dict = resp_data.get("usage") or {}
            total_in += int(usage_dict.get("prompt_tokens", 0))
            total_out += int(usage_dict.get("completion_tokens", 0))

            kind = spec["type"]
            if kind == "choice":
                crit = spec["criteria"]
                options = list(crit.keys()) if isinstance(crit, Mapping) else [str(x) for x in crit]
                parsed_choices[key] = compute_choice_from_logprobs(
                    options=options,
                    top_logprobs=top_logprobs,
                    temperature=self.temperature,
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "score":
                criteria_list = [str(x) for x in spec["criteria"]]
                parsed_scores[key] = compute_score_from_logprobs(
                    criteria=criteria_list,
                    top_logprobs=top_logprobs,
                    temperature=self.temperature,
                    confidence_floor=self.confidence_floor,
                )
            elif kind == "noul":
                parsed_nouls[key] = compute_noul_from_logprobs(
                    top_logprobs=top_logprobs,
                    temperature=self.temperature,
                )

        return JudgmentResult(
            choices=parsed_choices,
            scores=parsed_scores,
            nouls=parsed_nouls,
            model=target_model,
            usage=JudgmentUsage(input_tokens=total_in, output_tokens=total_out),
        )
