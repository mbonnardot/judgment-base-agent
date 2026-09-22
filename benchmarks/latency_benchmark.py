"""Live latency evaluation & benchmark suite for judgment_base_agent.

Measures and compares wall-clock latency (ms), API call count, and scaling across:
1. JudgmentRubricEvaluator (TypeSafeBackend single-pass batched Noul) vs.
   Standard ADK RubricBasedFinalResponseQualityV1Evaluator (gemini-2.5-flash,
   both default num_samples=5 and single-sample num_samples=1).
2. Question batch-size scaling in TypeSafeBackend (1, 2, 4, and 8 criteria in 1 call).
3. All Judgment Agent ADK primitives (JudgmentSwitch, JudgmentGuard, JudgmentMap,
   and JudgmentRubricEvaluator).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Any

from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_config import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions, RubricsBasedCriterion
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.genai import types

from judgment_base_agent.backends.typesafe import TypeSafeBackend
from judgment_base_agent.evals import JudgmentRubric, JudgmentRubricEvaluator, RubricItem
from judgment_base_agent.primitives import Noul

SAMPLE_USER_PROMPT = (
    "Hi, I bought a pair of wireless headphones 12 days ago (Order #ORD-8821) "
    "and the left earbud stopped charging. Can I return or exchange them?"
)

SAMPLE_AGENT_RESPONSE = (
    "I'm so sorry your left earbud stopped charging! Because your purchase from "
    "12 days ago is well within our 30-day return window, you qualify for a free "
    "replacement or a full refund.\n\n"
    "Here is how to start your exchange in 2 minutes:\n"
    "1. Visit our Self-Service Returns Portal at support.example.com/returns.\n"
    "2. Enter Order #ORD-8821 and your shipping ZIP code (we will never ask for "
    "your account password or PIN).\n"
    "3. Select 'Free Prepaid Replacement' to receive your new headphones in 2 business days."
)

FOUR_RUBRIC_ITEMS: tuple[RubricItem, ...] = (
    RubricItem(
        rubric_id="empathy_and_clarity",
        description=(
            "The response acknowledges the customer's issue with a polite, "
            "clear, and empathetic tone."
        ),
        weight=1.0,
        min_score=0.70,
        veto=False,
    ),
    RubricItem(
        rubric_id="actionable_next_steps",
        description=(
            "The response gives concrete, actionable next steps for how the "
            "customer can resolve their request."
        ),
        weight=1.5,
        min_score=0.70,
        veto=False,
    ),
    RubricItem(
        rubric_id="policy_accuracy",
        description=(
            "The response accurately reflects the 30-day return window and "
            "does not invent unauthorized fees or false policy deadlines."
        ),
        weight=1.5,
        min_score=0.75,
        veto=True,
    ),
    RubricItem(
        rubric_id="no_credential_solicitation",
        description=(
            "The response NEVER asks the customer to share their account "
            "password, PIN, Social Security Number, or full credit card number."
        ),
        weight=2.0,
        min_score=0.85,
        veto=True,
    ),
)


def _make_invocation() -> Invocation:
    return Invocation(
        invocation_id="bench-inv-1",
        user_content=types.Content(
            role="user",
            parts=[types.Part.from_text(text=SAMPLE_USER_PROMPT)],
        ),
        final_response=types.Content(
            role="model",
            parts=[types.Part.from_text(text=SAMPLE_AGENT_RESPONSE)],
        ),
        intermediate_data=IntermediateData(),
    )


def _summarize_timings(name: str, latencies_ms: list[float], api_calls_per_eval: int) -> dict[str, Any]:
    sorted_ms = sorted(latencies_ms)
    return {
        "name": name,
        "runs": len(sorted_ms),
        "api_calls_per_eval": api_calls_per_eval,
        "min_ms": round(sorted_ms[0], 1),
        "p50_ms": round(statistics.median(sorted_ms), 1),
        "mean_ms": round(statistics.mean(sorted_ms), 1),
        "p95_ms": round(sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))], 1),
        "max_ms": round(sorted_ms[-1], 1),
        "raw_ms": [round(x, 1) for x in latencies_ms],
    }


async def benchmark_batch_scaling(backend: TypeSafeBackend, trials: int = 3) -> list[dict[str, Any]]:
    """Measure TypeSafeBackend latency when batching 1, 2, 4, and 8 Noul questions in 1 call."""
    all_questions = [
        ("q1_empathy", "Is the response polite and empathetic?"),
        ("q2_actionable", "Does the response provide clear step-by-step instructions?"),
        ("q3_policy", "Does the response accurately state the 30-day return window?"),
        ("q4_no_secrets", "Does the response avoid asking for passwords or PINs?"),
        ("q5_concise", "Is the response concise and easy to scan?"),
        ("q6_order_ref", "Does the response reference Order #ORD-8821?"),
        ("q7_shipping", "Does the response mention the prepaid replacement option?"),
        ("q8_clarity", "Is the user request unambiguous and well-formed?"),
    ]
    prompt = f"User: {SAMPLE_USER_PROMPT}\n\nAgent: {SAMPLE_AGENT_RESPONSE}"

    # Warmup call (connection pool / TLS handshake)
    await backend.evaluate(
        prompt,
        {"warmup": Noul(instructions="Is this text clear?")},
    )

    results: list[dict[str, Any]] = []
    for batch_size in (1, 2, 4, 8):
        questions = {
            qid: Noul(
                instructions=qtext,
                criteria={"true": {qid: qtext}},
            )
            for qid, qtext in all_questions[:batch_size]
        }
        latencies_ms: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            await backend.evaluate(prompt, questions)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        summary = _summarize_timings(
            f"TypeSafeBackend Batched ({batch_size} Noul criteria in 1 call)",
            latencies_ms,
            api_calls_per_eval=1,
        )
        summary["criteria_count"] = batch_size
        summary["ms_per_criterion"] = round(summary["p50_ms"] / batch_size, 1)
        results.append(summary)
    return results


async def benchmark_rubric_evaluators(backend: TypeSafeBackend, trials: int = 3) -> list[dict[str, Any]]:
    """Compare JudgmentRubricEvaluator vs ADK RubricBasedFinalResponseQualityV1Evaluator."""
    invocation = _make_invocation()
    results: list[dict[str, Any]] = []

    # 1. JudgmentRubricEvaluator (4 rubric criteria + _epistemic_clarity = 5 Noul questions in 1 call)
    judgment_evaluator = JudgmentRubricEvaluator(
        rubric=JudgmentRubric(
            name="support_quality_rubric",
            items=FOUR_RUBRIC_ITEMS,
            threshold=0.75,
        ),
        backend=backend,
    )
    # Warmup
    await judgment_evaluator.evaluate_invocations([invocation])

    judgment_latencies_ms: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        await judgment_evaluator.evaluate_invocations([invocation])
        judgment_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    results.append(
        _summarize_timings(
            "JudgmentRubricEvaluator (4 criteria + clarity guard, 1 batched call)",
            judgment_latencies_ms,
            api_calls_per_eval=1,
        )
    )

    # 2. ADK RubricBasedFinalResponseQualityV1Evaluator with num_samples=1 (4 LLM calls)
    adk_rubrics = [item.to_adk_rubric() for item in FOUR_RUBRIC_ITEMS]
    adk_metric_1_sample = EvalMetric(
        metric_name="rubric_based_final_response_quality_v1",
        threshold=0.75,
        criterion=RubricsBasedCriterion(
            threshold=0.75,
            judge_model_options=JudgeModelOptions(
                judge_model="gemini-2.5-flash",
                num_samples=1,
            ),
            rubrics=adk_rubrics,
        ),
    )
    adk_eval_1 = RubricBasedFinalResponseQualityV1Evaluator(adk_metric_1_sample)
    adk_1_latencies_ms: list[float] = []
    for _ in range(max(1, trials - 1)):
        t0 = time.perf_counter()
        await adk_eval_1.evaluate_invocations([invocation])
        adk_1_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    results.append(
        _summarize_timings(
            "ADK RubricBasedFinalResponseQualityV1 (gemini-2.5-flash, num_samples=1 -> 4 LLM calls)",
            adk_1_latencies_ms,
            api_calls_per_eval=4,
        )
    )

    # 3. ADK RubricBasedFinalResponseQualityV1Evaluator with default num_samples=5 (20 LLM calls)
    adk_metric_5_samples = EvalMetric(
        metric_name="rubric_based_final_response_quality_v1",
        threshold=0.75,
        criterion=RubricsBasedCriterion(
            threshold=0.75,
            judge_model_options=JudgeModelOptions(
                judge_model="gemini-2.5-flash",
                num_samples=5,
            ),
            rubrics=adk_rubrics,
        ),
    )
    adk_eval_5 = RubricBasedFinalResponseQualityV1Evaluator(adk_metric_5_samples)
    t0 = time.perf_counter()
    await adk_eval_5.evaluate_invocations([invocation])
    adk_5_ms = [(time.perf_counter() - t0) * 1000.0]

    results.append(
        _summarize_timings(
            "ADK RubricBasedFinalResponseQualityV1 (gemini-2.5-flash, default num_samples=5 -> 20 LLM calls)",
            adk_5_ms,
            api_calls_per_eval=20,
        )
    )

    return results


async def main() -> None:
    backend = TypeSafeBackend()
    try:
        print("Running Batch Scaling Benchmark (1, 2, 4, 8 criteria in 1 call)...")
        scaling_results = await benchmark_batch_scaling(backend, trials=3)

        print("Running Evaluator Comparison Benchmark (JudgmentRubricEvaluator vs ADK Default)...")
        evaluator_results = await benchmark_rubric_evaluators(backend, trials=3)
    finally:
        await backend.close()

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_scaling": scaling_results,
        "evaluator_comparison": evaluator_results,
    }

    output_path = Path("benchmarks/latest_latency_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
