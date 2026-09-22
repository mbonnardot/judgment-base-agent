"""Integration tests verifying the 4 `examples/` applications load in ADK and produce deterministic outcomes."""

from __future__ import annotations

from google.adk.cli.utils.agent_loader import AgentLoader
import pytest

from examples.ai_action_approval_gate.agent import (
    ActionApprovalSchema,
    approval_policy,
)
from examples.policy_fact_checker_loop.agent import policy_fact_guard
from examples.review_triage_batch.agent import (
    INCOMING_APP_REVIEWS,
    ReviewEvaluationSchema,
    filter_and_rank_reviews,
)
from examples.smart_support_router.agent import support_router_switch
from judgment_base_agent import (
    ChoiceJudgment,
    JudgmentBatch,
    JudgmentBatchEntry,
    JudgmentResult,
    MockJudgmentBackend,
    NoulJudgment,
    ScoreJudgment,
)


def test_adk_agent_loader_discovers_all_five_examples() -> None:
    """Verify `adk web examples` discovers and loads all 5 example root_agents."""
    loader = AgentLoader("examples")
    agents = loader.list_agents()
    expected = {
        "smart_support_router",
        "ai_action_approval_gate",
        "policy_fact_checker_loop",
        "review_triage_batch",
        "llm_as_a_judge_rubric",
    }
    assert set(agents) == expected
    for name in expected:
        loaded = loader.load_agent(name)
        assert loaded is not None


@pytest.mark.asyncio
async def test_example_1_smart_support_router_switch_and_fallback() -> None:
    """Verify `support_router_switch` routes clear messages and falls back when clarity Noul < 0.75."""
    clear_backend = MockJudgmentBackend(
        responses={
            "route": ChoiceJudgment.from_raw(
                choice="instant_refund",
                probabilities={"instant_refund": 0.98, "tech_support": 0.01, "cancel_subscription": 0.01},
                confidence=0.98,
            ),
            "is_clear_and_specific": 0.97,
        }
    )
    object.__setattr__(support_router_switch, "backend", clear_backend)
    typed_clear = await support_router_switch._evaluate_core({}, "Duplicate charge refund")
    dec_clear = support_router_switch._invoke_decide(typed_clear, {}, "Duplicate charge refund")
    assert dec_clear.route == "instant_refund"

    vague_backend = MockJudgmentBackend(
        responses={
            "route": ChoiceJudgment.from_raw(
                choice="tech_support",
                probabilities={"tech_support": 0.95, "instant_refund": 0.05},
                confidence=0.95,
            ),
            "is_clear_and_specific": 0.03,
        }
    )
    object.__setattr__(support_router_switch, "backend", vague_backend)
    typed_vague = await support_router_switch._evaluate_core({}, "Vague mixed request")
    dec_vague = support_router_switch._invoke_decide(typed_vague, {}, "Vague mixed request")
    assert dec_vague.route == "ask_clarifying_question"


def test_example_2_ai_action_approval_gate_policy_outcomes() -> None:
    """Verify `approval_policy` maps low, moderate, and critical risks to the 3 expected verdicts."""
    low_risk = ActionApprovalSchema(
        action_type=ChoiceJudgment.from_raw("small_order_refund", confidence=1.0),
        follows_policy=NoulJudgment(noul=0.97),
        risk_level=ScoreJudgment.from_raw(
            score=0.38,
            legend={"0": "low_safe_risk", "1": "moderate_manager_risk", "2": "high_risk", "3": "critical_security_risk"},
        ),
    )
    dec_auto = approval_policy(low_risk, {})
    assert dec_auto.output["approval_verdict"] == "AUTO_APPROVED"
    assert dec_auto.route == "auto_approved"

    med_risk = ActionApprovalSchema(
        action_type=ChoiceJudgment.from_raw("large_credit_or_override", confidence=1.0),
        follows_policy=NoulJudgment(noul=0.93),
        risk_level=ScoreJudgment.from_raw(
            score=1.19,
            legend={"0": "low_safe_risk", "1": "moderate_manager_risk", "2": "high_risk", "3": "critical_security_risk"},
        ),
    )
    dec_mgr = approval_policy(med_risk, {})
    assert dec_mgr.output["approval_verdict"] == "NEEDS_MANAGER_APPROVAL"
    assert dec_mgr.route == "manager_approval"

    high_risk = ActionApprovalSchema(
        action_type=ChoiceJudgment.from_raw("destructive_or_unauthorized", confidence=1.0),
        follows_policy=NoulJudgment(noul=0.01),
        risk_level=ScoreJudgment.from_raw(
            score=3.00,
            legend={"0": "low_safe_risk", "1": "moderate_manager_risk", "2": "high_risk", "3": "critical_security_risk"},
        ),
    )
    dec_block = approval_policy(high_risk, {})
    assert dec_block.output["approval_verdict"] == "BLOCKED_SECURITY_OR_POLICY_VIOLATION"
    assert dec_block.escalate is True


@pytest.mark.asyncio
async def test_example_3_policy_fact_checker_guard() -> None:
    """Verify `policy_fact_guard` rejects ungrounded promises (< 0.85) and escalates on valid replies (>= 0.85)."""
    fail_backend = MockJudgmentBackend(responses={"guard": 0.01})
    object.__setattr__(policy_fact_guard, "backend", fail_backend)
    res_fail = await policy_fact_guard._evaluate_core({"store_policy": "30 days", "draft_reply": "90 days"}, None)
    dec_fail = policy_fact_guard._invoke_decide(res_fail, {}, None)
    assert dec_fail.route == "fail"
    assert dec_fail.escalate is False

    pass_backend = MockJudgmentBackend(responses={"guard": 0.92})
    object.__setattr__(policy_fact_guard, "backend", pass_backend)
    res_pass = await policy_fact_guard._evaluate_core({"store_policy": "30 days", "draft_reply": "30 days"}, None)
    dec_pass = policy_fact_guard._invoke_decide(res_pass, {}, None)
    assert dec_pass.route == "pass"
    assert dec_pass.escalate is True


def test_example_4_review_triage_batch_filter_and_rank() -> None:
    """Verify `filter_and_rank_reviews` blocks spam, skips generic praise, and ranks bugs by urgency."""
    legend = {
        "0": "none_or_praise",
        "1": "minor_ui_polish",
        "2": "moderate_workflow_bug",
        "3": "critical_checkout_or_crash_blocker",
    }
    sim_data = [
        (0.98, 0.98, 3.00),  # REV-101: Apple Pay crash
        (0.01, 0.01, 0.23),  # REV-102: Crypto spam
        (0.99, 0.98, 1.19),  # REV-103: Dark mode contrast
        (0.98, 0.01, 0.00),  # REV-104: 5-star praise
        (0.98, 0.97, 2.00),  # REV-105: Pro sync delay
    ]
    entries = []
    # strict=True: sim_data is hand-maintained to pair 1:1 with INCOMING_APP_REVIEWS.
    # Without it, adding a review would silently drop off the end and go untested.
    for idx, (item, (safe_p, act_p, urg_s)) in enumerate(
        zip(INCOMING_APP_REVIEWS, sim_data, strict=True)
    ):
        j = ReviewEvaluationSchema(
            is_safe_not_spam=NoulJudgment(noul=safe_p),
            has_actionable_issue=NoulJudgment(noul=act_p),
            urgency_score=ScoreJudgment.from_raw(score=urg_s, legend=legend),
        )
        entries.append(
            JudgmentBatchEntry(
                index=idx,
                item=item,
                judgment=j,
                raw_result=JudgmentResult(),
            )
        )
    batch = JudgmentBatch(entries=tuple(entries))
    result = filter_and_rank_reviews(batch)
    assert result["spam_blocked_ids"] == ["REV-102"]
    assert result["praise_skipped_ids"] == ["REV-104"]
    assert [b["id"] for b in result["prioritized_bugs"]] == ["REV-101", "REV-105", "REV-103"]


@pytest.mark.asyncio
async def test_example_5_llm_as_a_judge_rubric_evalset() -> None:
    """Verify `SUPPORT_QUALITY_RUBRIC` and `support_rubric.evalset.json` pass compliant turns and veto-fail unsafe turns."""
    import json
    from pathlib import Path

    from google.adk.evaluation.eval_metrics import EvalStatus
    from google.adk.evaluation.eval_set import EvalSet

    from examples.llm_as_a_judge_rubric.agent import SUPPORT_QUALITY_RUBRIC
    from judgment_base_agent.evals import CLARITY_KEY, JudgmentRubricEvaluator

    eval_set = EvalSet.model_validate(
        json.loads(
            Path("examples/llm_as_a_judge_rubric/support_rubric.evalset.json").read_text()
        )
    )
    assert len(eval_set.eval_cases) == 2

    pass_backend = MockJudgmentBackend(
        {
            "empathy_and_clarity": 0.85,
            "actionable_next_steps": 0.94,
            "policy_accuracy": 0.88,
            "no_credential_solicitation": 0.99,
            CLARITY_KEY: 0.95,
        }
    )
    evaluator_pass = JudgmentRubricEvaluator(
        rubric=SUPPORT_QUALITY_RUBRIC, backend=pass_backend
    )
    res_pass = await evaluator_pass.evaluate_invocations(
        eval_set.eval_cases[0].conversation or []
    )
    assert res_pass.overall_eval_status == EvalStatus.PASSED

    veto_backend = MockJudgmentBackend(
        {
            "empathy_and_clarity": 0.88,
            "actionable_next_steps": 0.70,
            "policy_accuracy": 0.80,
            "no_credential_solicitation": 0.01,
            CLARITY_KEY: 0.95,
        }
    )
    evaluator_veto = JudgmentRubricEvaluator(
        rubric=SUPPORT_QUALITY_RUBRIC, backend=veto_backend
    )
    res_veto = await evaluator_veto.evaluate_invocations(
        eval_set.eval_cases[1].conversation or []
    )
    assert res_veto.overall_eval_status == EvalStatus.FAILED

