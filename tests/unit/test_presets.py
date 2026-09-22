"""Unit tests for JudgmentSwitch, JudgmentGuard, JudgmentMap, and JudgmentBatch."""

from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

from judgment_base_agent import (
    JudgmentAgent,
    JudgmentBatch,
    JudgmentConfigError,
    JudgmentField,
    JudgmentGate,
    JudgmentGuard,
    JudgmentMap,
    JudgmentRouter,
    JudgmentSchema,
    JudgmentSwitch,
    MockJudgmentBackend,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
    SystemOneAgent,
    SystemOneGate,
    SystemOneRouter,
)


def test_aliases_are_identical() -> None:
    import judgment_base_agent

    assert SystemOneAgent is JudgmentAgent
    assert JudgmentRouter is JudgmentSwitch
    assert SystemOneRouter is JudgmentSwitch
    assert JudgmentGate is JudgmentGuard
    assert SystemOneGate is JudgmentGuard

    # The aliases must resolve as module attributes too, not only through the
    # `from judgment_base_agent import ...` block above.
    assert judgment_base_agent.SystemOneAgent is judgment_base_agent.JudgmentAgent

    # Guard against __all__ going stale: every exported name must exist.
    unresolved = [name for name in judgment_base_agent.__all__ if not hasattr(judgment_base_agent, name)]
    assert unresolved == []


def test_presets_validation_errors() -> None:
    with pytest.raises(JudgmentConfigError):
        JudgmentSwitch(name="bad_switch")
    with pytest.raises(JudgmentConfigError):
        JudgmentGuard(name="bad_guard")
    with pytest.raises(JudgmentConfigError):
        JudgmentMap(name="bad_map")


@pytest.mark.asyncio
async def test_judgment_switch_confidence_floor_and_custom_policy() -> None:
    low_conf_backend = MockJudgmentBackend(
        responses={"route": {"choice": "billing", "confidence": 0.42}}
    )
    switch = JudgmentSwitch(
        name="ticket_router",
        instructions="Which team should handle `ticket`?",
        routes={"billing": "Charges", "tech": "Bugs"},
        confidence_floor=0.60,
        uncertain_route="human_triage",
        backend=low_conf_backend,
    )

    app = App(name="app", root_agent=switch)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="ambiguous")]),
        )
    ]
    assert events[-1].actions.route == "human_triage"

    # Custom route_policy
    custom_switch = JudgmentSwitch(
        name="custom_router",
        instructions="Which team?",
        routes=["billing", "tech"],
        route_policy=lambda res, state: "override_route",
        backend=low_conf_backend,
    )
    app2 = App(name="app2", root_agent=custom_switch)
    runner2 = InMemoryRunner(app=app2)
    session2 = await runner2.session_service.create_session(app_name="app2", user_id="u1")
    events2 = [
        ev
        async for ev in runner2.run_async(
            user_id="u1",
            session_id=session2.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="test")]),
        )
    ]
    assert events2[-1].actions.route == "override_route"


@pytest.mark.asyncio
async def test_judgment_guard_pass_and_fail_escalation() -> None:
    backend = MockJudgmentBackend(responses={"guard": 0.85})
    guard = JudgmentGuard(
        name="safety_guard",
        instructions="Is the draft completely grounded?",
        threshold=0.75,
        escalate_on_pass=True,
        backend=backend,
    )

    app = App(name="app", root_agent=guard)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="check")]),
        )
    ]
    assert events[-1].actions.route == "pass"
    assert events[-1].actions.escalate is True

    # Custom predicate returning bool or JudgmentDecision
    pred_guard = JudgmentGuard(
        name="pred_guard",
        instructions="Is ok?",
        predicate=lambda res, state: False,
        backend=backend,
    )
    app2 = App(name="app2", root_agent=pred_guard)
    runner2 = InMemoryRunner(app=app2)
    session2 = await runner2.session_service.create_session(app_name="app2", user_id="u1")
    events2 = [
        ev
        async for ev in runner2.run_async(
            user_id="u1",
            session_id=session2.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="check")]),
        )
    ]
    assert events2[-1].actions.route == "fail"
    assert events2[-1].actions.escalate is False


class DocEvalSchema(JudgmentSchema):
    relevant: NoulJudgment = JudgmentField(Noul(instructions="Is doc relevant?"))
    quality: ScoreJudgment = JudgmentField(
        Score(instructions="Doc quality?", criteria=["poor", "good", "excellent"])
    )


@pytest.mark.asyncio
async def test_judgment_map_filter_rank_map_and_reduce() -> None:
    backend = MockJudgmentBackend(
        responses={
            "item_0__relevant": 0.92,
            "item_0__quality": 1.8,
            "item_1__relevant": 0.20,
            "item_1__quality": 0.5,
            "item_2__relevant": 0.85,
            "item_2__quality": 1.95,
            "holistic_ok": 0.90,
        }
    )

    mapper = JudgmentMap(
        name="doc_reranker",
        items_key="docs",
        output_key="reranked_docs",
        item_schema=DocEvalSchema,
        global_questions={"holistic_ok": Noul(instructions="Are docs sufficient?")},
        transform=lambda batch, state: {
            "items_count": len(batch.items()),
            "judgments_count": len(batch.judgments()),
            "serialized": batch.to_dict(),
            "top_docs": (
                batch.filter(lambda item, j: j.relevant.noul >= 0.50)
                .rank_by(lambda item, j: j.quality.score, top_k=2)
                .map(lambda item, j: {"id": item["id"], "score": j.quality.score})
            ),
            "all_relevant": batch.all(lambda item, j: j.relevant.noul >= 0.50),
            "any_relevant": batch.any(lambda item, j: j.relevant.noul >= 0.50),
            "total_score": batch.reduce(
                lambda acc, item, j: acc + j.quality.score, 0.0
            ),
            "holistic": batch.global_result.noul("holistic_ok"),
        },
        backend=backend,
    )

    app = App(name="app", root_agent=mapper)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="app",
        user_id="u1",
        state={
            "docs": [
                {"id": "d0", "text": "First relevant doc"},
                {"id": "d1", "text": "Irrelevant noise"},
                {"id": "d2", "text": "Best relevant doc"},
            ]
        },
    )

    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="run")]),
        )
    ]
    # Single batched backend call for all 3 items + global question!
    assert len(backend.calls) == 1
    out = events[-1].output
    assert out["items_count"] == 3
    assert out["judgments_count"] == 3
    assert len(out["serialized"]["entries"]) == 3
    assert [d["id"] for d in out["top_docs"]] == ["d2", "d0"]
    assert out["all_relevant"] is False
    assert out["any_relevant"] is True
    assert out["total_score"] == pytest.approx(1.8 + 0.5 + 1.95)
    assert out["holistic"] == pytest.approx(0.90)


@pytest.mark.asyncio
async def test_judgment_map_empty_and_items_getter() -> None:
    backend = MockJudgmentBackend(responses={})
    mapper = JudgmentMap(
        name="empty_mapper",
        items_getter=lambda state: [],
        item_questions=lambda item, idx: {"q": Noul(instructions="Ok?")},
        backend=backend,
    )
    app = App(name="app", root_agent=mapper)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="run")]),
        )
    ]
    assert events[-1].output["entries"] == []


# ---------------------------------------------------------------------------
# Regression tests: `judge()` must agree with the ADK `_run_async_impl` path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judgment_map_judge_returns_batch_not_flat_result() -> None:
    """JudgmentMap.judge() must fan out per item instead of falling through to the base agent."""
    backend = MockJudgmentBackend(
        responses={"item_0__relevant": 0.91, "item_1__relevant": 0.10}
    )
    mapper = JudgmentMap(
        name="judge_batch_mapper",
        items_key="docs",
        item_questions=lambda item, idx: {
            "relevant": Noul(instructions=f"Is {item['id']} relevant?")
        },
        backend=backend,
    )

    batch = await mapper.judge({"docs": [{"id": "d0"}, {"id": "d1"}]})

    assert isinstance(batch, JudgmentBatch)
    assert len(batch.entries) == 2
    assert len(backend.calls) == 1  # still a single batched backend call
    assert [entry.item["id"] for entry in batch.entries] == ["d0", "d1"]
    assert batch.entries[0].raw_result.noul("relevant") == pytest.approx(0.91)
    assert batch.entries[1].raw_result.noul("relevant") == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_judge_honors_state_keys_like_the_adk_path() -> None:
    """judge() must apply state_keys so it cannot leak state the ADK path would have filtered."""
    backend = MockJudgmentBackend(responses={"ok": 0.80})
    agent = JudgmentAgent(
        name="scoped_agent",
        state_keys=["ticket"],
        questions={"ok": Noul(instructions="Is this fine?")},
        backend=backend,
    )

    await agent.judge({"ticket": "hello", "secret": "do-not-send"})

    assert backend.calls[0]["state"] == {"ticket": "hello"}


def test_subclass_overriding_evaluate_core_must_also_override_judge() -> None:
    """Guard the silent-wrong-type trap: custom _evaluate_core without judge() is a definition error."""
    with pytest.raises(TypeError, match="must also override"):

        class BrokenAgent(JudgmentAgent):
            async def _evaluate_core(self, session_state, node_input):  # type: ignore[override]
                return "structurally different output"
