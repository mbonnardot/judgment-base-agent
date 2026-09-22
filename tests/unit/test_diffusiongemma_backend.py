"""Unit and integration tests for DiffusionGemmaBackend (OpenJev) and TypeSafeBackend auto-routing."""

from __future__ import annotations

from typing import Any

import pytest

from judgment_base_agent import (
    Choice,
    DiffusionGemmaBackend,
    JudgmentAgent,
    JudgmentSchema,
    Noul,
    Score,
    TypeSafeBackend,
)
from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError


class RoutingSchema(JudgmentSchema):
    dept: Choice = Choice(
        instructions="Route ticket to department",
        criteria={"billing": "Payment issues", "tech": "Technical issues"},
    )
    frustration: Score = Score(
        instructions="Rate customer frustration",
        criteria=["calm", "annoyed", "furious"],
    )
    escalate: Noul = Noul(
        instructions="Does this require human escalation?",
        criteria={"true": "High risk", "false": "Routine"},
    )


def test_removed_redundant_bubble_sheet_and_vllm_helpers() -> None:
    """Verify legacy O(N) /v1/completions prompt and logprob helpers were removed."""
    import judgment_base_agent.backends.diffusiongemma as dg_mod

    for removed in (
        "build_bubble_sheet_prompt",
        "compute_choice_from_logprobs",
        "compute_score_from_logprobs",
        "compute_noul_from_logprobs",
    ):
        assert not hasattr(dg_mod, removed), f"{removed} is redundant with OpenJev and must be removed"


@pytest.mark.asyncio
async def test_openjev_backend_defaults_and_answers_parsing() -> None:
    """Verify DiffusionGemmaBackend targets OpenJev /v1/systemone with openjev-latest and parses answers."""
    captured_payloads: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured_payloads.append({"url": url, "json": json, "headers": headers})
        return {
            "model": "openjev-0.1",
            "usage": {"input_tokens": 64, "output_tokens": 0},
            "answers": {
                "dept": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.92, "tech": 0.08},
                    "confidence": 0.88,
                },
                "frustration": {
                    "type": "score",
                    "score": 1.75,
                    "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                    "probabilities": {"0": 0.05, "1": 0.15, "2": 0.80},
                    "confidence": 0.84,
                },
                "escalate": {
                    "type": "noul",
                    "noul": 0.91,
                },
            },
        }

    backend = DiffusionGemmaBackend(
        base_url="https://openjev-xyz.a.run.app",
        api_key="sk-openjev-123",
        transport_post=fake_post,
    )
    result = await backend.evaluate(
        state={"ticket": "Refund me immediately"},
        questions=RoutingSchema.questions(),
    )

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["url"] == "https://openjev-xyz.a.run.app/v1/systemone"
    assert captured_payloads[0]["headers"]["Authorization"] == "Bearer sk-openjev-123"
    assert captured_payloads[0]["json"]["model"] == "openjev-latest"
    assert list(captured_payloads[0]["json"]["questions"].keys()) == ["dept", "frustration", "escalate"]
    assert "temperature" not in captured_payloads[0]["json"]

    assert result.choice("dept") == "billing"
    assert result.score("frustration") == pytest.approx(1.75)
    assert result.noul("escalate") == pytest.approx(0.91)
    assert result.model == "openjev-0.1"
    assert result.usage is not None
    assert result.usage.input_tokens == 64


@pytest.mark.asyncio
async def test_openjev_model_alias_normalization() -> None:
    """OpenJev only accepts {'openjev-latest', 'openjev-0.1', 'jev-latest', 'jev-preview'}.

    Verify that 'judgment-latest', 'system-one', or raw HuggingFace checkpoint IDs
    are normalized to 'openjev-latest' so OpenJev never returns 400 Unknown model.
    """
    seen_models: list[str] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        seen_models.append(json["model"])
        return {"model": "openjev-0.1", "answers": {"escalate": {"type": "noul", "noul": 0.5}}}

    backend = DiffusionGemmaBackend(base_url="http://127.0.0.1:8080", transport_post=fake_post)
    q = {"escalate": RoutingSchema.questions()["escalate"]}

    for raw_model, expected in [
        ("judgment-latest", "openjev-latest"),
        ("system-one", "openjev-latest"),
        ("RedHatAI/diffusiongemma-26B-A4B-it-NVFP4", "openjev-latest"),
        ("jev-latest", "jev-latest"),
        ("openjev-0.1", "openjev-0.1"),
    ]:
        await backend.evaluate(state="hello", questions=q, model=raw_model)
        assert seen_models[-1] == expected


@pytest.mark.asyncio
async def test_openjev_extensions_passed_when_configured() -> None:
    """Verify optional OpenJev extensions (steps, samples, think, sequential, images) are forwarded."""
    captured: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append(json)
        return {"model": "openjev-0.1", "answers": {"escalate": {"type": "noul", "noul": 0.8}}}

    backend = DiffusionGemmaBackend(
        base_url="http://127.0.0.1:8080",
        steps=2,
        samples=4,
        think=256,
        sequential=True,
        transport_post=fake_post,
    )
    await backend.evaluate(
        state="Complex ticket",
        questions={"escalate": RoutingSchema.questions()["escalate"]},
    )

    sent = captured[0]
    assert sent["steps"] == 2
    assert sent["samples"] == 4
    assert sent["think"] == 256
    assert sent["sequential"] is True


@pytest.mark.asyncio
async def test_typesafe_backend_auto_delegates_to_openjev_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TypeSafeBackend() transparently routes to OpenJev when OPENJEV_BASE_URL or DIFFUSIONGEMMA_JEV_URL is set."""
    monkeypatch.setenv("OPENJEV_BASE_URL", "https://api.codiv.ai")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    captured: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append({"url": url, "json": json})
        return {
            "model": "openjev-0.1",
            "answers": {
                "dept": {
                    "type": "choice",
                    "choice": "tech",
                    "probabilities": {"billing": 0.1, "tech": 0.9},
                    "confidence": 0.90,
                },
                "frustration": {
                    "type": "score",
                    "score": 0.5,
                    "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                    "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1},
                    "confidence": 0.82,
                },
                "escalate": {
                    "type": "noul",
                    "noul": 0.12,
                },
            },
        }

    backend = TypeSafeBackend()
    backend._diffusion_transport_post = fake_post  # type: ignore[attr-defined]

    agent = JudgmentAgent(
        name="triage_judgment",
        schema=RoutingSchema,
        backend=backend,
    )
    decision = await agent.judge({"ticket": "My API key gives 403"})
    assert captured[0]["url"] == "https://api.codiv.ai/v1/systemone"
    assert captured[0]["json"]["model"] == "openjev-latest"
    assert decision.choice("dept") == "tech"
    assert decision.noul("escalate") == pytest.approx(0.12)
    assert decision.model == "openjev-0.1"


@pytest.mark.asyncio
async def test_openjev_backend_missing_url_and_error_wrapping() -> None:
    """Verify missing base_url raises JudgmentConfigError and transport failures raise JudgmentEvaluationError."""
    backend = DiffusionGemmaBackend(base_url="")
    with pytest.raises(JudgmentConfigError):
        await backend.evaluate(state="x", questions={"escalate": RoutingSchema.questions()["escalate"]})

    async def failing_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        raise RuntimeError("connection refused")

    backend2 = DiffusionGemmaBackend(base_url="http://127.0.0.1:8080", transport_post=failing_post)
    with pytest.raises(JudgmentEvaluationError, match="connection refused"):
        await backend2.evaluate(state="x", questions={"escalate": RoutingSchema.questions()["escalate"]})
