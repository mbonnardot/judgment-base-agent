"""Unit and integration tests for DiffusionGemmaBackend, TypeSafeBackend auto-routing, and OpenJev server."""

from __future__ import annotations

import math
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
from judgment_base_agent.backends.diffusiongemma import (
    build_bubble_sheet_prompt,
    compute_choice_from_logprobs,
    compute_noul_from_logprobs,
    compute_score_from_logprobs,
)


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


def test_bubble_sheet_math_helpers() -> None:
    """Verify softmax probability, expected score, noul ratio, and Shannon entropy confidence."""
    prompt = build_bubble_sheet_prompt(
        state={"ticket": "Double charged"},
        question_key="dept",
        question=RoutingSchema.questions()["dept"],
    )
    assert "Double charged" in prompt
    assert "billing" in prompt
    assert "tech" in prompt

    # 1. Choice from logprobs
    top_logprobs = {
        " billing": -0.10536,  # exp(-0.10536) ≈ 0.90
        " tech": -2.30258,     # exp(-2.30258) ≈ 0.10
    }
    choice_j = compute_choice_from_logprobs(
        options=["billing", "tech"],
        top_logprobs=top_logprobs,
        temperature=1.0,
        confidence_floor=0.50,
    )
    assert choice_j.choice == "billing"
    assert choice_j.probabilities["billing"] == pytest.approx(0.90, abs=1e-2)
    assert choice_j.probabilities["tech"] == pytest.approx(0.10, abs=1e-2)
    assert 0.50 <= choice_j.confidence <= 1.0

    # 2. Score from logprobs (0, 1, 2 indices for ["calm", "annoyed", "furious"])
    score_logprobs = {
        "0": -2.30258,  # ~0.10
        "1": -1.60943,  # ~0.20
        "2": -0.35667,  # ~0.70
    }
    score_j = compute_score_from_logprobs(
        criteria=["calm", "annoyed", "furious"],
        top_logprobs=score_logprobs,
        temperature=1.0,
        confidence_floor=0.50,
    )
    # Expected value = 0*0.1 + 1*0.2 + 2*0.7 = 1.6
    assert score_j.score == pytest.approx(1.6, abs=0.05)
    assert score_j.probabilities["furious"] == pytest.approx(0.70, abs=0.02)

    # 3. Noul from logprobs
    noul_logprobs = {
        "true": math.log(0.85),
        "false": math.log(0.15),
    }
    noul_j = compute_noul_from_logprobs(top_logprobs=noul_logprobs, temperature=1.0)
    assert noul_j.noul == pytest.approx(0.85, abs=0.02)


@pytest.mark.asyncio
async def test_diffusiongemma_backend_system_one_container_mode() -> None:
    """Verify DiffusionGemmaBackend talking to the Cloud Run /v1/system_one endpoint."""
    captured_payloads: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured_payloads.append({"url": url, "json": json, "headers": headers})
        return {
            "model": "diffusiongemma-26B-A4B-it-NVFP4",
            "usage": {"input_tokens": 64, "output_tokens": 3},
            "choices": {
                "dept": {
                    "choice": "billing",
                    "probabilities": {"billing": 0.92, "tech": 0.08},
                    "confidence": 0.88,
                }
            },
            "scores": {
                "frustration": {
                    "score": 1.75,
                    "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                    "probabilities": {"calm": 0.05, "annoyed": 0.15, "furious": 0.80},
                    "confidence": 0.84,
                }
            },
            "nouls": {
                "escalate": {"noul": 0.91},
            },
        }

    backend = DiffusionGemmaBackend(
        base_url="https://diffusiongemma-jev-xyz.a.run.app",
        api_key="gcp-token-123",
        transport_post=fake_post,
    )
    result = await backend.evaluate(
        state={"ticket": "Refund me immediately"},
        questions=RoutingSchema.questions(),
    )

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["url"] == "https://diffusiongemma-jev-xyz.a.run.app/v1/system_one"
    assert captured_payloads[0]["headers"]["Authorization"] == "Bearer gcp-token-123"
    assert result.choice("dept") == "billing"
    assert result.score("frustration") == pytest.approx(1.75)
    assert result.noul("escalate") == pytest.approx(0.91)
    assert result.model == "diffusiongemma-26B-A4B-it-NVFP4"
    assert result.usage is not None
    assert result.usage.input_tokens == 64


@pytest.mark.asyncio
async def test_diffusiongemma_backend_direct_vllm_completions_mode() -> None:
    """Verify DiffusionGemmaBackend can score directly against a standard vLLM /v1/completions endpoint."""

    async def fake_vllm_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        prompt = json["prompt"]
        if "dept" in prompt:
            logprobs = {" billing": -0.05, " tech": -3.0}
        elif "frustration" in prompt:
            logprobs = {"0": -3.0, "1": -1.5, "2": -0.2}
        else:
            logprobs = {"true": -0.1, "false": -2.5}
        return {
            "model": "google/diffusiongemma-26B-A4B-it",
            "choices": [
                {
                    "text": " ",
                    "logprobs": {"top_logprobs": [logprobs]},
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 1},
        }

    backend = DiffusionGemmaBackend(
        base_url="http://localhost:8000",
        mode="vllm",
        transport_post=fake_vllm_post,
    )
    result = await backend.evaluate(
        state={"ticket": "Charged twice"},
        questions=RoutingSchema.questions(),
    )
    assert result.choice("dept") == "billing"
    assert result.score("frustration") > 1.3
    assert result.noul("escalate") > 0.80


@pytest.mark.asyncio
async def test_typesafe_backend_auto_delegates_when_diffusiongemma_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify existing agents using default TypeSafeBackend() transparently use DiffusionGemma container when DIFFUSIONGEMMA_JEV_URL is set."""
    monkeypatch.setenv("DIFFUSIONGEMMA_JEV_URL", "https://diffusiongemma-jev-cloudrun.a.run.app")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return {
            "model": "diffusiongemma-26B-A4B-it-NVFP4",
            "choices": {
                "dept": {
                    "choice": "tech",
                    "probabilities": {"billing": 0.1, "tech": 0.9},
                    "confidence": 0.90,
                }
            },
            "scores": {
                "frustration": {
                    "score": 0.5,
                    "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                    "probabilities": {"calm": 0.6, "annoyed": 0.3, "furious": 0.1},
                    "confidence": 0.82,
                }
            },
            "nouls": {
                "escalate": {"noul": 0.12},
            },
        }

    # Default TypeSafeBackend (used by all existing JudgmentAgent / Switch / Guard / Map agents)
    backend = TypeSafeBackend()
    backend._diffusion_transport_post = fake_post  # type: ignore[attr-defined]

    agent = JudgmentAgent(
        name="triage_judgment",
        schema=RoutingSchema,
        backend=backend,
    )
    decision = await agent.judge({"ticket": "My API key gives 403"})
    assert decision.choice("dept") == "tech"
    assert decision.noul("escalate") == pytest.approx(0.12)
    assert decision.model == "diffusiongemma-26B-A4B-it-NVFP4"




@pytest.mark.heavy
@pytest.mark.asyncio
async def test_cloud_run_container_real_diffusiongemma_transformers_e2e(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a real 1-step DiffusionGemmaForBlockDiffusion canvas forward pass through server.py and JudgmentAgent.

    Skipped unless torch and a transformers build exposing
    DiffusionGemmaForBlockDiffusion are installed. Neither is a declared
    dependency of this package, and the test additionally downloads a tiny
    model from Hugging Face, so it cannot run on a bare offline checkout.
    Run it explicitly with `pytest -m heavy`.
    """
    import httpx

    pytest.importorskip("torch", reason="heavy test: requires torch")
    transformers = pytest.importorskip(
        "transformers",
        minversion="5.8.0",
        reason="heavy test: requires transformers>=5.8 for DiffusionGemmaForBlockDiffusion",
    )
    if not hasattr(transformers, "DiffusionGemmaForBlockDiffusion"):
        pytest.skip("installed transformers lacks DiffusionGemmaForBlockDiffusion")

    from deploy.diffusiongemma_jev import server as container_server

    monkeypatch.setenv("DIFFUSIONGEMMA_ENGINE", "transformers")
    monkeypatch.setenv(
        "DIFFUSIONGEMMA_MODEL_ID",
        "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion",
    )
    monkeypatch.setenv("DIFFUSIONGEMMA_JEV_URL", "http://testserver")

    async def asgi_transport_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        transport = httpx.ASGITransport(app=container_server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            path = "/" + url.split("/", 3)[-1]
            resp = await client.post(path, json=json, headers=headers)
            resp.raise_for_status()
            return resp.json()

    backend = TypeSafeBackend()
    backend._diffusion_transport_post = asgi_transport_post  # type: ignore[attr-defined]

    agent = JudgmentAgent(
        name="real_diffusiongemma_canvas_agent",
        schema=RoutingSchema,
        backend=backend,
    )
    decision = await agent.judge({"ticket": "Charged twice on my invoice!"})
    assert decision.choice("dept") in ("billing", "tech")
    assert 0.0 <= decision.score("frustration") <= 2.0
    assert 0.0 <= decision.noul("escalate") <= 1.0
    assert "DiffusionGemmaForBlockDiffusion" in decision.model




@pytest.mark.asyncio
async def test_system_one_path_is_configurable_for_the_vllm_pr_server() -> None:
    """The vLLM PR's reference server serves /v1/systemone (no underscore).

    Our default stays /v1/system_one for the existing container, but the path
    must be overridable so agents can point at the PR server unchanged.
    """
    seen: list[str] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        seen.append(url)
        return {"model": "dgemma", "choices": {"dept": {"choice": "billing", "confidence": 0.9}}}

    backend = DiffusionGemmaBackend(
        base_url="http://127.0.0.1:8011",
        system_one_path="/v1/systemone",
        transport_post=fake_post,
    )
    await backend.evaluate(
        state={"ticket": "Refund me"},
        questions={"dept": Choice(instructions="Route", criteria={"billing": "b", "tech": "t"})},
    )

    assert seen == ["http://127.0.0.1:8011/v1/systemone"]


@pytest.mark.asyncio
async def test_system_one_path_defaults_to_the_existing_container_route() -> None:
    """Omitting system_one_path must not change behaviour for the deployed container."""
    seen: list[str] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        seen.append(url)
        return {"model": "dgemma", "choices": {"dept": {"choice": "billing", "confidence": 0.9}}}

    backend = DiffusionGemmaBackend(
        base_url="https://example.a.run.app",
        transport_post=fake_post,
    )
    await backend.evaluate(
        state={"ticket": "Refund me"},
        questions={"dept": Choice(instructions="Route", criteria={"billing": "b", "tech": "t"})},
    )

    assert seen == ["https://example.a.run.app/v1/system_one"]


@pytest.mark.asyncio
async def test_question_keys_are_aliased_on_the_wire() -> None:
    """Long/underscored keys break the vLLM PR's canvas template, so alias them.

    vllm-project/vllm#57250's structured_server.py prints the question key next to
    the answer slot in a shared canvas template. Keys like `item_0__urgency_score`
    change how the adjacent label tokenizes, and the server then rejects its own
    default `no` label with 422 "label 'no' is not a single token". Short opaque
    keys avoid it. Verified live against DiffusionGemma-26B on an L4.
    """
    captured: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append(json)
        return {
            "model": "dg",
            "answers": {
                "q0": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9},
                "q1": {"type": "score", "score": 1.5, "legend": {}, "probabilities": {}, "confidence": 0.8},
                "q2": {"type": "noul", "noul": 0.7},
            },
        }

    backend = DiffusionGemmaBackend(base_url="http://gpu:8011", transport_post=fake_post)
    result = await backend.evaluate(state={"t": "x"}, questions=RoutingSchema.questions())

    sent_keys = list(captured[0]["questions"].keys())
    assert sent_keys == ["q0", "q1", "q2"], f"expected opaque keys on the wire, got {sent_keys}"

    # The caller still sees its own key names.
    assert result.choice("dept") == "billing"
    assert result.score("frustration") == pytest.approx(1.5)
    assert result.noul("escalate") == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_aliasing_still_reads_servers_that_echo_the_original_keys() -> None:
    """Our own transformers container echoes whatever keys it received; be tolerant."""

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        return {
            "model": "dg",
            "choices": {"dept": {"choice": "tech", "probabilities": {"tech": 1.0}, "confidence": 1.0}},
            "scores": {"frustration": {"score": 2.0, "legend": {}, "probabilities": {}, "confidence": 1.0}},
            "nouls": {"escalate": {"noul": 0.2}},
        }

    backend = DiffusionGemmaBackend(base_url="http://gpu:8011", transport_post=fake_post)
    result = await backend.evaluate(state={"t": "x"}, questions=RoutingSchema.questions())

    assert result.choice("dept") == "tech"
    assert result.noul("escalate") == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_key_aliasing_can_be_disabled() -> None:
    """Opaque keys drop a semantic hint from the prompt, so allow opting out."""
    captured: list[dict[str, Any]] = []

    async def fake_post(url: str, json: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append(json)
        return {"model": "dg", "nouls": {"escalate": {"noul": 0.5}}}

    backend = DiffusionGemmaBackend(
        base_url="http://gpu:8011",
        transport_post=fake_post,
        alias_question_keys=False,
    )
    await backend.evaluate(
        state={"t": "x"},
        questions={"escalate": RoutingSchema.questions()["escalate"]},
    )

    assert list(captured[0]["questions"].keys()) == ["escalate"]
