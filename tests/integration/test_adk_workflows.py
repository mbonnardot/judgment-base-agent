"""End-to-end integration tests for ADK 2.0 Graph Workflow and Composite Agents (SequentialAgent, LoopAgent)."""

from google.adk.agents import BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.runners import InMemoryRunner
from google.adk.workflow import Workflow
from google.genai import types
import pytest

from judgment_base_agent import (
    JudgmentGuard,
    JudgmentSwitch,
    MockJudgmentBackend,
)


@pytest.mark.asyncio
async def test_adk2_graph_workflow_conditional_routing_with_judgment_switch() -> None:
    backend = MockJudgmentBackend(
        responses={"route": {"choice": "refund", "confidence": 0.94}}
    )

    router = JudgmentSwitch(
        name="intent_router",
        instructions="Classify customer request intent",
        routes={"refund": "Customer wants money back", "tech": "Technical issue"},
        backend=backend,
    )

    def handle_refund(node_input: dict) -> str:
        return f"REFUND_PROCESSED:{node_input['choices']['route']['choice']}"

    def handle_tech(node_input: dict) -> str:
        return "TECH_SUPPORT"

    wf = Workflow(
        name="customer_workflow",
        edges=[
            ("START", router),
            (router, {"refund": handle_refund, "tech": handle_tech}),
        ],
    )

    app = App(name="wf_app", root_agent=wf)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="wf_app", user_id="u1")

    outputs = []
    async for ev in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text="Please refund my duplicate charge")]
        ),
    ):
        if ev.output is not None:
            outputs.append(ev.output)

    assert "REFUND_PROCESSED:refund" in outputs


class DraftProducer(BaseAgent):
    """Test helper agent that increments attempt counter in state."""

    async def _run_async_impl(self, ctx: InvocationContext):
        attempt = int(ctx.session.state.get("attempt", 0)) + 1
        ctx.session.state["attempt"] = attempt
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta={"attempt": attempt}),
        )


@pytest.mark.asyncio
async def test_adk_sequential_and_loop_agent_with_judgment_guard() -> None:
    # Passes on attempt 2
    backend = MockJudgmentBackend(
        responses=lambda state, q, m: {"guard": 0.92 if state.get("attempt", 0) >= 2 else 0.30}
    )

    guard = JudgmentGuard(
        name="quality_gate",
        instructions="Is the draft grounded?",
        threshold=0.80,
        state_keys=["attempt"],
        output_key="gate_result",
        backend=backend,
    )

    loop = LoopAgent(
        name="refinement_loop",
        sub_agents=[DraftProducer(name="producer"), guard],
        max_iterations=5,
    )

    pipeline = SequentialAgent(
        name="main_pipeline",
        sub_agents=[loop],
    )

    app = App(name="loop_app", root_agent=pipeline)
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(app_name="loop_app", user_id="u1")

    async for _ in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text="start")]),
    ):
        pass

    updated_session = await runner.session_service.get_session(
        app_name="loop_app", user_id="u1", session_id=session.id
    )
    assert updated_session.state["attempt"] == 2
    assert updated_session.state["gate_result"]["nouls"]["guard"]["noul"] == pytest.approx(0.92)
