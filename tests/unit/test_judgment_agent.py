"""Unit tests for JudgmentAgent, JudgmentDecision, and @judgment_node."""

try:
    import coverage as _coverage

    _cov = _coverage.Coverage.current()
    if _cov is not None and getattr(_cov, "_inorout", None) is not None:
        _io = _cov._inorout
        for _pkg in list(_io.source_pkgs):
            if "/" in _pkg:
                _dotted = _pkg.replace("/", ".").removesuffix(".py")
                _io.source_pkgs.append(_dotted)
                if _pkg in _io.source_pkgs_unmatched:
                    _io.source_pkgs_unmatched.remove(_pkg)
                if _io.source_pkgs_match is not None:
                    _io.source_pkgs_match.modules.append(_dotted)
except Exception:
    pass

from google.adk.apps import App
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
import pytest

from judgment_base_agent.agent import (
    JudgmentAgent,
    JudgmentDecision,
    judgment_node,
    normalize_decision,
)
from judgment_base_agent.backends.mock import MockJudgmentBackend
from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import Choice, ChoiceJudgment, Noul, NoulJudgment
from judgment_base_agent.schema import JudgmentField, JudgmentSchema


class TriageSchema(JudgmentSchema):
    billing: NoulJudgment = JudgmentField(Noul(instructions="Is billing?"))
    department: ChoiceJudgment = JudgmentField(
        Choice(instructions="Department?", criteria=["billing", "support"])
    )


@pytest.mark.asyncio
async def test_judgment_agent_with_schema_and_decide_hook() -> None:
    mock_backend = MockJudgmentBackend(
        responses={
            "billing": 0.95,
            "department": {"choice": "billing", "confidence": 0.88},
        }
    )

    agent = JudgmentAgent(
        name="triage_step",
        schema=TriageSchema,
        state_keys=["ticket"],
        output_key="triage_output",
        backend=mock_backend,
        decide=lambda res, state: JudgmentDecision(
            route="billing_route" if res.billing.noul > 0.8 else "support_route",
            escalate=True,
            state_delta={"routed_to": res.department.choice},
        ),
    )

    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="test_app",
        user_id="u1",
        state={"ticket": "I was charged twice!"},
    )

    events = []
    async for ev in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="go")]),
    ):
        events.append(ev)

    assert len(events) == 1
    ev = events[0]
    assert ev.actions.route == "billing_route"
    assert ev.actions.escalate is True
    assert ev.actions.state_delta["routed_to"] == "billing"
    assert ev.actions.state_delta["triage_output"]["billing"]["noul"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_judgment_agent_on_error_fallback() -> None:
    class FailingBackend:
        async def evaluate(self, state, questions, model=None):
            raise JudgmentEvaluationError("API rate limit")

    agent = JudgmentAgent(
        name="resilient_step",
        questions={"ok": Noul(instructions="Is ok?")},
        backend=FailingBackend(),
        on_error=lambda exc, state: JudgmentDecision(
            route="fallback_edge",
            output={"error": str(exc)},
        ),
    )

    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")

    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
        )
    ]
    assert events[-1].actions.route == "fallback_edge"
    assert events[-1].output == {"error": "API rate limit"}


@pytest.mark.asyncio
async def test_judgment_node_decorator() -> None:
    mock_backend = MockJudgmentBackend(responses={"billing": 0.92, "department": "billing"})

    @judgment_node(name="decorated_triage", schema=TriageSchema, backend=mock_backend)
    def handle_triage(res: TriageSchema, state: dict) -> str:
        return "fast_track" if res.billing.noul > 0.9 else "standard"

    assert isinstance(handle_triage, JudgmentAgent)
    app = App(name="test_app", root_agent=handle_triage)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="check")]),
        )
    ]
    assert events[-1].actions.route == "fast_track"


def test_judgment_agent_requires_schema_or_questions() -> None:
    with pytest.raises(JudgmentConfigError):
        JudgmentAgent(name="invalid_agent")


def test_normalize_decision_variants() -> None:
    assert normalize_decision(None, default_output={"d": 1}) == JudgmentDecision(output={"d": 1})
    assert normalize_decision(True, default_output="def") == JudgmentDecision(
        output="def", route="pass", escalate=True
    )
    assert normalize_decision(False, default_output="def") == JudgmentDecision(
        output="def", route="fail", escalate=False
    )
    assert normalize_decision("route_a", default_output="def", transfer_to_sub_agent=True) == JudgmentDecision(
        output="def", route="route_a", transfer_to_agent="route_a"
    )
    assert normalize_decision(["a", "b"], default_output="def") == JudgmentDecision(
        output="def", route=["a", "b"]
    )
    assert normalize_decision({"custom": 42}, default_output="def") == JudgmentDecision(
        output={"custom": 42}
    )


@pytest.mark.asyncio
async def test_state_builder_callable_questions_and_request_input() -> None:
    mock_backend = MockJudgmentBackend(responses={"ok": 0.4})

    agent = JudgmentAgent(
        name="custom_hooks_agent",
        questions=lambda state_payload, node_in: {"ok": Noul(instructions=f"Check {node_in}")},
        state_builder=lambda s, node_in: {"merged": s.get("k"), "input": node_in},
        backend=mock_backend,
        decide=lambda res, state, node_in: JudgmentDecision(
            request_input_id="req_123",
            request_input_prompt="Please confirm",
        ),
    )

    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="u1", state={"k": "val"}
    )
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="hello")]),
        )
    ]
    assert len(events) == 1
    assert isinstance(events[0], Event)
    assert "req_123" in (events[0].long_running_tool_ids or set())


@pytest.mark.asyncio
async def test_single_arg_builders_and_unhandled_error() -> None:
    mock_backend = MockJudgmentBackend(responses={"ok": 0.9})
    agent = JudgmentAgent(
        name="single_arg_agent",
        questions=lambda s: {"ok": Noul(instructions="Ok?")},
        state_builder=lambda s: {"from_single": True},
        decide=lambda res: "string_output_route",
        backend=mock_backend,
    )
    app = App(name="test_app", root_agent=agent)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="run")]),
        )
    ]
    assert events[-1].actions.route == "string_output_route"

    class FailingBackend:
        async def evaluate(self, state, questions, model=None):
            raise JudgmentEvaluationError("boom")

    bad_agent = JudgmentAgent(
        name="bad_agent",
        questions={"ok": Noul(instructions="Ok?")},
        backend=FailingBackend(),
    )
    bad_app = App(name="bad_app", root_agent=bad_agent)
    bad_runner = InMemoryRunner(app=bad_app)
    bad_session = await bad_runner.session_service.create_session(app_name="bad_app", user_id="u1")
    with pytest.raises(JudgmentEvaluationError, match="boom"):
        async for _ in bad_runner.run_async(
            user_id="u1",
            session_id=bad_session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text="run")]),
        ):
            pass
