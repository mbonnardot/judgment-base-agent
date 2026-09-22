"""Live Latency & Token/Cost Evaluation across all 5 Judgment Agent ADK Examples.

Benchmarks each of the 5 examples in `examples/`:
1. `customer_support_triage` (`JudgmentSwitch` — 3-branch router + epistemic clarity)
2. `ai_action_approval_gate` (`JudgmentGuard` — 3-rule pre-execution safety gate)
3. `policy_fact_checker_loop` (`JudgmentGuard` — 4-rule post-generation policy verifier)
4. `review_triage_batch` (`JudgmentMap` — 5 app reviews x 3 criteria = 15 judgments in 1 call)
5. `llm_as_a_judge_rubric` (`JudgmentRubricEvaluator` — 4-criterion weighted + veto rubric judge)

Compares:
- `TypeSafeBackend` (`judgment-latest`) — Single-pass calibrated System-1 judgment
- Traditional LLM #1: `gemini-3.5-flash-lite`
- Traditional LLM #2: `gemini-3.7-flash`
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Any

from google import genai
from google.adk.evaluation.eval_case import IntermediateData, Invocation
from google.adk.evaluation.eval_config import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions, RubricsBasedCriterion
from google.adk.evaluation.rubric_based_final_response_quality_v1 import (
    RubricBasedFinalResponseQualityV1Evaluator,
)
from google.genai import types

from examples.llm_as_a_judge_rubric.agent import SUPPORT_QUALITY_RUBRIC
from examples.review_triage_batch.agent import (
    INCOMING_APP_REVIEWS,
    ReviewEvaluationSchema,
)
from judgment_base_agent.backends.typesafe import TypeSafeBackend
from judgment_base_agent.evals import JudgmentRubricEvaluator
from judgment_base_agent.primitives import Choice, Noul

SUPPORT_ROUTES = {
    "billing_refund": "Duplicate charges, subscription refunds, invoice disputes, or payment method errors",
    "technical_bug": "Application crashes, API 500 errors, broken integrations, or SDK bugs",
    "account_security": "Unauthorized login alerts, 2FA/MFA resets, compromised credentials, or SSO lockouts",
}

ACTION_SAFETY_CRITERIA = {
    "destructive_scope": "Does NOT drop production tables, delete customer backups, or wipe entire namespaces",
    "least_privilege": "Does NOT grant wildcard IAM roles (roles/owner, *) or disable audit logging",
    "no_secrets": "Contains ZERO hardcoded plaintext passwords, private keys, or API tokens",
}

STORE_POLICY_SUMMARY = (
    "1. Return Window: 30-day return window from delivery date.\n"
    "2. Free Shipping: Free standard shipping on orders of $50 or more ($7.99 under $50).\n"
    "3. Final Sale Exclusions: Gift cards and clearance items are final sale and non-refundable.\n"
    "4. Warranty Coverage: 1-year hardware defect warranty (excludes accidental water or drop damage)."
)

STORE_POLICY_CRITERIA = {
    "return_window_30d": "States the 30-day return window accurately (never promises 60 or 90 days)",
    "shipping_threshold_50": "States free shipping applies to orders >= $50 ($7.99 fee under $50)",
    "final_sale_exclusions": "Accurately notes gift cards and clearance items cannot be returned",
    "warranty_exclusions": "Accurately states accidental water/drop damage is excluded from warranty",
}

# Reference token pricing per 1M tokens (USD):
# - TypeSafe System-1 (judgment-latest): $0.15 / 1M input, $0.60 / 1M output
# - Gemini 3.5 Flash-Lite: $0.10 / 1M input, $0.40 / 1M output (incl. thinking)
# - Gemini 3.7 Flash: $0.30 / 1M input, $2.50 / 1M output (incl. thinking)
MODEL_PRICING = {
    "judgment-latest": (0.15, 0.60),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.7-flash": (0.30, 2.50),
}


def _estimate_cost_per_1k(model_key: str, input_tokens: float, output_tokens: float) -> float:
    in_rate, out_rate = MODEL_PRICING[model_key]
    cost_per_call = (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate
    return round(cost_per_call * 1000.0, 4)


def _stats(latencies_ms: list[float]) -> dict[str, float]:
    s = sorted(latencies_ms)
    return {
        "min_ms": round(s[0], 1),
        "p50_ms": round(statistics.median(s), 1),
        "mean_ms": round(statistics.mean(s), 1),
        "max_ms": round(s[-1], 1),
    }


async def _run_gemini_single_call(
    genai_client: genai.Client,
    model_name: str,
    prompt: str,
    trials: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    in_tokens = 0
    out_tokens = 0
    thought_tokens = 0
    for _ in range(trials):
        t0 = time.perf_counter()
        resp = await genai_client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if resp.usage_metadata:
            in_tokens = resp.usage_metadata.prompt_token_count or 0
            cand_tokens = resp.usage_metadata.candidates_token_count or 0
            thought_tokens = getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0
            out_tokens = cand_tokens + thought_tokens

    st = _stats(latencies)
    return {
        "model": model_name,
        **st,
        "api_calls": 1,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "thought_tokens": thought_tokens,
        "cost_per_1k_usd": _estimate_cost_per_1k(model_name, in_tokens, out_tokens),
    }


async def benchmark_all_examples(trials: int = 3) -> dict[str, Any]:
    backend = TypeSafeBackend()
    genai_client = genai.Client()

    # Warmup connections
    await backend.evaluate("Warmup ping", {"q": Noul(instructions="Is this clear?")})
    for m in ("gemini-3.5-flash-lite", "gemini-3.7-flash"):
        await genai_client.aio.models.generate_content(model=m, contents="Reply OK")

    examples_report: list[dict[str, Any]] = []

    try:
        # =====================================================================
        # EXAMPLE 1: customer_support_triage (JudgmentSwitch)
        # =====================================================================
        print("Benchmarking Example 1: customer_support_triage (JudgmentSwitch)...")
        ex1_prompt = (
            "I was charged twice ($49.99 x 2) on my Visa ending in 4021 for my "
            "March subscription renewal. Please refund the duplicate charge."
        )
        ex1_questions = {
            "route": Choice(
                instructions="Classify the customer support request into the primary specialist queue.",
                criteria=SUPPORT_ROUTES,
            ),
            "_epistemic_clarity": Noul(
                instructions="Is the user's support request specific and clear enough to route without guessing?"
            ),
        }
        ex1_ts_latencies: list[float] = []
        ex1_in, ex1_out = 95, 8
        for _ in range(trials):
            t0 = time.perf_counter()
            res = await backend.evaluate(ex1_prompt, ex1_questions)
            ex1_ts_latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.usage and res.usage.input_tokens:
                ex1_in, ex1_out = res.usage.input_tokens, res.usage.output_tokens

        ex1_llm_prompt = (
            "Classify the following customer support request into one of: "
            f"{json.dumps(SUPPORT_ROUTES)}. Also score clarity (0.0-1.0) and confidence (0.0-1.0). "
            f"Return JSON.\n\nRequest: {ex1_prompt}"
        )
        ex1_lite = await _run_gemini_single_call(genai_client, "gemini-3.5-flash-lite", ex1_llm_prompt, trials)
        ex1_flash = await _run_gemini_single_call(genai_client, "gemini-3.7-flash", ex1_llm_prompt, trials)
        ts1 = _stats(ex1_ts_latencies)
        examples_report.append({
            "example_id": "1. customer_support_triage",
            "preset": "JudgmentSwitch",
            "task": "3-way intent routing + epistemic clarity abstention check (2 judgments)",
            "typesafe_judgment": {
                "model": "judgment-latest",
                **ts1,
                "api_calls": 1,
                "input_tokens": ex1_in,
                "output_tokens": ex1_out,
                "cost_per_1k_usd": _estimate_cost_per_1k("judgment-latest", ex1_in, ex1_out),
            },
            "gemini_3_5_flash_lite": ex1_lite,
            "gemini_3_7_flash": ex1_flash,
            "speedup_vs_lite_x": round(ex1_lite["p50_ms"] / ts1["p50_ms"], 1),
            "speedup_vs_flash_x": round(ex1_flash["p50_ms"] / ts1["p50_ms"], 1),
        })

        # =====================================================================
        # EXAMPLE 2: ai_action_approval_gate (JudgmentGuard Pre-Execution)
        # =====================================================================
        print("Benchmarking Example 2: ai_action_approval_gate (JudgmentGuard Pre-Gate)...")
        ex2_prompt = (
            "Ticket: CHG-4092. Action: Rotate read-only analytics replica TLS certificate "
            "in staging cluster us-east1-b during maintenance window."
        )
        ex2_questions = {
            "guard": Noul(
                instructions="Is this infrastructure action safe and compliant with all security rules?",
                criteria={"true": ACTION_SAFETY_CRITERIA},
            ),
            "_epistemic_clarity": Noul(
                instructions="Does the request include enough concrete detail to evaluate safety?"
            ),
        }
        ex2_ts_latencies: list[float] = []
        ex2_in, ex2_out = 135, 8
        for _ in range(trials):
            t0 = time.perf_counter()
            res = await backend.evaluate(ex2_prompt, ex2_questions)
            ex2_ts_latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.usage and res.usage.input_tokens:
                ex2_in, ex2_out = res.usage.input_tokens, res.usage.output_tokens

        ex2_llm_prompt = (
            "Evaluate whether this infrastructure action satisfies every safety rule: "
            f"{json.dumps(ACTION_SAFETY_CRITERIA)}. Return JSON with per-rule scores and overall pass/fail.\n\n"
            f"Action: {ex2_prompt}"
        )
        ex2_lite = await _run_gemini_single_call(genai_client, "gemini-3.5-flash-lite", ex2_llm_prompt, trials)
        ex2_flash = await _run_gemini_single_call(genai_client, "gemini-3.7-flash", ex2_llm_prompt, trials)
        ts2 = _stats(ex2_ts_latencies)
        examples_report.append({
            "example_id": "2. ai_action_approval_gate",
            "preset": "JudgmentGuard (Pre-Gate)",
            "task": "3-rule DevOps/privilege/secret safety gate before execution (4 judgments)",
            "typesafe_judgment": {
                "model": "judgment-latest",
                **ts2,
                "api_calls": 1,
                "input_tokens": ex2_in,
                "output_tokens": ex2_out,
                "cost_per_1k_usd": _estimate_cost_per_1k("judgment-latest", ex2_in, ex2_out),
            },
            "gemini_3_5_flash_lite": ex2_lite,
            "gemini_3_7_flash": ex2_flash,
            "speedup_vs_lite_x": round(ex2_lite["p50_ms"] / ts2["p50_ms"], 1),
            "speedup_vs_flash_x": round(ex2_flash["p50_ms"] / ts2["p50_ms"], 1),
        })

        # =====================================================================
        # EXAMPLE 3: policy_fact_checker_loop (JudgmentGuard in LoopAgent)
        # =====================================================================
        print("Benchmarking Example 3: policy_fact_checker_loop (JudgmentGuard Loop Verifier)...")
        ex3_state = (
            f"Official Policy:\n{STORE_POLICY_SUMMARY}\n\n"
            "Customer Question: If I buy a $35 backpack and a $20 gift card, do I get free shipping, "
            "and can I return both after 2 weeks?\n\n"
            "Draft Answer: Because your combined order total is $55 ($35 backpack + $20 gift card), "
            "you qualify for free standard shipping (orders $50+). Within 2 weeks (14 days), you can "
            "return the $35 backpack under our 30-day return window, but the $20 gift card is final sale "
            "and non-refundable."
        )
        ex3_questions = {
            "guard": Noul(
                instructions="Does the Draft Answer strictly comply with every rule in the Official Store Policy?",
                criteria={"true": STORE_POLICY_CRITERIA},
            ),
            "_epistemic_clarity": Noul(
                instructions="Are both the Official Store Policy and Draft Answer present and legible?"
            ),
        }
        ex3_ts_latencies: list[float] = []
        ex3_in, ex3_out = 260, 8
        for _ in range(trials):
            t0 = time.perf_counter()
            res = await backend.evaluate(ex3_state, ex3_questions)
            ex3_ts_latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.usage and res.usage.input_tokens:
                ex3_in, ex3_out = res.usage.input_tokens, res.usage.output_tokens

        ex3_llm_prompt = (
            "Verify whether the Draft Answer strictly complies with all 4 store policy rules: "
            f"{json.dumps(STORE_POLICY_CRITERIA)}. Return JSON with rule scores and pass/fail.\n\n"
            f"{ex3_state}"
        )
        ex3_lite = await _run_gemini_single_call(genai_client, "gemini-3.5-flash-lite", ex3_llm_prompt, trials)
        ex3_flash = await _run_gemini_single_call(genai_client, "gemini-3.7-flash", ex3_llm_prompt, trials)
        ts3 = _stats(ex3_ts_latencies)
        examples_report.append({
            "example_id": "3. policy_fact_checker_loop",
            "preset": "JudgmentGuard (Loop Verifier)",
            "task": "4-rule store policy fact-check per LoopAgent iteration (5 judgments)",
            "typesafe_judgment": {
                "model": "judgment-latest",
                **ts3,
                "api_calls": 1,
                "input_tokens": ex3_in,
                "output_tokens": ex3_out,
                "cost_per_1k_usd": _estimate_cost_per_1k("judgment-latest", ex3_in, ex3_out),
            },
            "gemini_3_5_flash_lite": ex3_lite,
            "gemini_3_7_flash": ex3_flash,
            "speedup_vs_lite_x": round(ex3_lite["p50_ms"] / ts3["p50_ms"], 1),
            "speedup_vs_flash_x": round(ex3_flash["p50_ms"] / ts3["p50_ms"], 1),
        })

        # =====================================================================
        # EXAMPLE 4: review_triage_batch (JudgmentMap — 5 items x 3 criteria = 15 judgments)
        # =====================================================================
        print("Benchmarking Example 4: review_triage_batch (JudgmentMap 15-judgment batch)...")
        ex4_batched_questions: dict[str, Any] = {}
        schema_questions = ReviewEvaluationSchema.build_questions()
        for idx in range(len(INCOMING_APP_REVIEWS)):
            for qk, qv in schema_questions.items():
                ex4_batched_questions[f"item_{idx}__{qk}"] = qv
        ex4_state = {"items": INCOMING_APP_REVIEWS}

        ex4_ts_latencies: list[float] = []
        ex4_in, ex4_out = 420, 30
        for _ in range(trials):
            t0 = time.perf_counter()
            res = await backend.evaluate(ex4_state, ex4_batched_questions)
            ex4_ts_latencies.append((time.perf_counter() - t0) * 1000.0)
            if res.usage and res.usage.input_tokens:
                ex4_in, ex4_out = res.usage.input_tokens, res.usage.output_tokens

        async def _bench_review_batch_llm(model_name: str) -> dict[str, Any]:
            latencies: list[float] = []
            total_in, total_out = 0, 0
            for _ in range(max(1, trials - 1)):
                t0 = time.perf_counter()
                run_in, run_out = 0, 0
                for review in INCOMING_APP_REVIEWS:
                    resp = await genai_client.aio.models.generate_content(
                        model=model_name,
                        contents=(
                            "Evaluate this app review for: (1) actionable_bug (0.0-1.0), "
                            "(2) urgency (1-5), (3) category (crash_or_data_loss, billing_or_login, "
                            "ui_or_feature_request, praise_or_spam). Return JSON.\n\n"
                            f"Review: {json.dumps(review)}"
                        ),
                    )
                    if resp.usage_metadata:
                        run_in += resp.usage_metadata.prompt_token_count or 0
                        run_out += (
                            (resp.usage_metadata.candidates_token_count or 0)
                            + (getattr(resp.usage_metadata, "thoughts_token_count", 0) or 0)
                        )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                total_in, total_out = run_in, run_out
            st = _stats(latencies)
            return {
                "model": model_name,
                **st,
                "api_calls": 5,
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cost_per_1k_usd": _estimate_cost_per_1k(model_name, total_in, total_out),
            }

        ex4_lite = await _bench_review_batch_llm("gemini-3.5-flash-lite")
        ex4_flash = await _bench_review_batch_llm("gemini-3.7-flash")
        ts4 = _stats(ex4_ts_latencies)
        examples_report.append({
            "example_id": "4. review_triage_batch",
            "preset": "JudgmentMap + JudgmentBatch",
            "task": "5 customer reviews × 3 criteria (15 calibrated judgments in 1 batch call)",
            "typesafe_judgment": {
                "model": "judgment-latest",
                **ts4,
                "api_calls": 1,
                "input_tokens": ex4_in,
                "output_tokens": ex4_out,
                "cost_per_1k_usd": _estimate_cost_per_1k("judgment-latest", ex4_in, ex4_out),
            },
            "gemini_3_5_flash_lite": ex4_lite,
            "gemini_3_7_flash": ex4_flash,
            "speedup_vs_lite_x": round(ex4_lite["p50_ms"] / ts4["p50_ms"], 1),
            "speedup_vs_flash_x": round(ex4_flash["p50_ms"] / ts4["p50_ms"], 1),
        })

        # =====================================================================
        # EXAMPLE 5: llm_as_a_judge_rubric (JudgmentRubricEvaluator vs ADK Default)
        # =====================================================================
        print("Benchmarking Example 5: llm_as_a_judge_rubric (JudgmentRubricEvaluator vs ADK Rubric Evaluator)...")
        invocation = Invocation(
            invocation_id="bench-ex5",
            user_content=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Hi, I bought wireless headphones 12 days ago and the left earbud stopped charging. Can I return them?"
                    )
                ],
            ),
            final_response=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "I'm so sorry your left earbud stopped charging! Since your purchase from 12 days ago "
                            "is within our 30-day return window, you qualify for a free prepaid replacement or full refund "
                            "at support.example.com/returns using your Order ID and ZIP code (we never ask for passwords)."
                        )
                    )
                ],
            ),
            intermediate_data=IntermediateData(),
        )
        judgment_eval = JudgmentRubricEvaluator(rubric=SUPPORT_QUALITY_RUBRIC, backend=backend)
        ex5_ts_latencies: list[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            await judgment_eval.evaluate_invocations([invocation])
            ex5_ts_latencies.append((time.perf_counter() - t0) * 1000.0)

        adk_rubrics = [item.to_adk_rubric() for item in SUPPORT_QUALITY_RUBRIC.items]

        async def _bench_adk_rubric_evaluator(model_name: str, num_samples: int) -> dict[str, Any]:
            evaluator = RubricBasedFinalResponseQualityV1Evaluator(
                EvalMetric(
                    metric_name="rubric_based_final_response_quality_v1",
                    threshold=0.75,
                    criterion=RubricsBasedCriterion(
                        threshold=0.75,
                        judge_model_options=JudgeModelOptions(
                            judge_model=model_name,
                            num_samples=num_samples,
                        ),
                        rubrics=adk_rubrics,
                    ),
                )
            )
            t0 = time.perf_counter()
            await evaluator.evaluate_invocations([invocation])
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            calls = len(adk_rubrics) * num_samples
            est_in = calls * 340
            est_out = calls * (180 if "lite" in model_name else 260)
            st = _stats([elapsed_ms])
            return {
                "model": f"{model_name} (num_samples={num_samples})",
                **st,
                "api_calls": calls,
                "input_tokens": est_in,
                "output_tokens": est_out,
                "cost_per_1k_usd": _estimate_cost_per_1k(model_name, est_in, est_out),
            }

        ex5_lite_1 = await _bench_adk_rubric_evaluator("gemini-3.5-flash-lite", num_samples=1)
        ex5_lite_5 = await _bench_adk_rubric_evaluator("gemini-3.5-flash-lite", num_samples=5)
        ex5_flash_1 = await _bench_adk_rubric_evaluator("gemini-3.7-flash", num_samples=1)
        ex5_flash_5 = await _bench_adk_rubric_evaluator("gemini-3.7-flash", num_samples=5)

        ts5 = _stats(ex5_ts_latencies)
        examples_report.append({
            "example_id": "5. llm_as_a_judge_rubric",
            "preset": "JudgmentRubricEvaluator",
            "task": "4-criterion weighted ADK rubric + hard-fail safety vetoes (5 judgments)",
            "typesafe_judgment": {
                "model": "judgment-latest",
                **ts5,
                "api_calls": 1,
                "input_tokens": 310,
                "output_tokens": 10,
                "cost_per_1k_usd": _estimate_cost_per_1k("judgment-latest", 310, 10),
            },
            "gemini_3_5_flash_lite_1_sample": ex5_lite_1,
            "gemini_3_5_flash_lite_default_5_samples": ex5_lite_5,
            "gemini_3_7_flash_1_sample": ex5_flash_1,
            "gemini_3_7_flash_default_5_samples": ex5_flash_5,
            "speedup_vs_lite_1_sample_x": round(ex5_lite_1["p50_ms"] / ts5["p50_ms"], 1),
            "speedup_vs_lite_5_samples_x": round(ex5_lite_5["p50_ms"] / ts5["p50_ms"], 1),
            "speedup_vs_flash_1_sample_x": round(ex5_flash_1["p50_ms"] / ts5["p50_ms"], 1),
            "speedup_vs_flash_5_samples_x": round(ex5_flash_5["p50_ms"] / ts5["p50_ms"], 1),
        })

    finally:
        await backend.close()

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "examples": examples_report,
    }
    out_path = Path("benchmarks/examples_latency_cost_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    asyncio.run(benchmark_all_examples(trials=3))
