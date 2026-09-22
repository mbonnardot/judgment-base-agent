"""Unit tests for BaseJudgmentBackend, TypeSafeBackend, and MockJudgmentBackend."""

from types import SimpleNamespace

import pytest
import typesafe_sdk

from judgment_base_agent.backends import (
    BaseJudgmentBackend,
    MockJudgmentBackend,
    TypeSafeBackend,
)
from judgment_base_agent.backends.typesafe import _question_kind, _to_typesafe_question
from judgment_base_agent.errors import JudgmentConfigError, JudgmentEvaluationError
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
)


@pytest.mark.asyncio
async def test_mock_judgment_backend_static_and_callable() -> None:
    backend = MockJudgmentBackend(
        responses={
            "dept": {
                "choice": "billing",
                "confidence": 0.91,
                "probabilities": {"billing": 0.91, "tech": 0.09},
            },
            "urgent": 0.88,
            "severity": {"score": 1.5, "confidence": 0.72},
        }
    )
    assert isinstance(backend, BaseJudgmentBackend)

    questions = {
        "dept": Choice(instructions="Dept?", criteria=["billing", "tech"]),
        "urgent": Noul(instructions="Urgent?"),
        "severity": Score(instructions="Severity?", criteria=["low", "med", "high"]),
    }
    res = await backend.evaluate(state={"msg": "help"}, questions=questions)
    assert res.choice("dept") == "billing"
    assert res.noul("urgent") == pytest.approx(0.88)
    assert res.score("severity") == pytest.approx(1.5)
    assert len(backend.calls) == 1

    # Test callable responses and direct Judgment objects / scalar fallbacks
    callable_backend = MockJudgmentBackend(
        responses=lambda state, qs, model: {
            "dept": "tech",
            "urgent": NoulJudgment(noul=0.42),
            "severity": 2.0,
            "already_choice": ChoiceJudgment.from_raw(choice="a", confidence=0.95),
            "already_score": ScoreJudgment.from_raw(score=1.0, confidence=0.85),
            "dict_noul": {"noul": 0.77},
        }
    )
    extra_questions = {
        **questions,
        "already_choice": Choice(instructions="A?", criteria=["a", "b"]),
        "already_score": Score(instructions="S?", criteria=["0", "1"]),
        "dict_noul": Noul(instructions="N?"),
    }
    res2 = await callable_backend.evaluate(
        state={"msg": "callable"}, questions=extra_questions, model="custom-mock"
    )
    assert res2.model == "custom-mock"
    assert res2.choice("dept") == "tech"
    assert res2.noul("urgent") == pytest.approx(0.42)
    assert res2.score("severity") == pytest.approx(2.0)
    assert res2.choice("already_choice") == "a"
    assert res2.score("already_score") == pytest.approx(1.0)
    assert res2.noul("dict_noul") == pytest.approx(0.77)


@pytest.mark.asyncio
async def test_typesafe_backend_with_injected_client_and_sdk_primitives() -> None:
    captured: dict = {}

    class FakeClient:
        async def system_one(self, *, state, questions, model=None):
            captured["state"] = state
            captured["questions"] = questions
            captured["model"] = model
            return SimpleNamespace(
                model=model or "judgment-latest",
                usage=SimpleNamespace(input_tokens=15, output_tokens=3),
                choices={
                    "dept": SimpleNamespace(
                        choice="tech",
                        probabilities={"billing": 0.05, "tech": 0.95},
                        confidence=0.92,
                    )
                },
                scores={
                    "frustration": SimpleNamespace(
                        score=1.2,
                        legend={"0": "calm", "1": "annoyed", "2": "angry"},
                        probabilities={"0": 0.1, "1": 0.6, "2": 0.3},
                        confidence=0.65,
                    )
                },
                nouls={"refund": SimpleNamespace(noul=0.12)},
            )

    backend = TypeSafeBackend(client=FakeClient(), default_model="judgment-latest")
    assert isinstance(backend, BaseJudgmentBackend)

    # Empty questions short-circuit
    empty_res = await backend.evaluate(state={}, questions={})
    assert empty_res.model == "judgment-latest"

    questions = {
        "dept": Choice(instructions="Which dept?", criteria=["billing", "tech"]),
        "frustration": typesafe_sdk.Score(
            instructions="Frustration?", criteria=["calm", "annoyed", "angry"]
        ),
        "refund": Noul(
            instructions="Refund?",
            criteria={"true": "Asks for money back", "false": "Does not"},
        ),
    }

    result = await backend.evaluate(state={"text": "500 error"}, questions=questions)
    assert captured["model"] == "judgment-latest"
    assert isinstance(captured["questions"]["dept"], typesafe_sdk.Choice)
    assert isinstance(captured["questions"]["refund"], typesafe_sdk.Noul)
    assert result.choice("dept") == "tech"
    assert result.score("frustration") == pytest.approx(1.2)
    assert result.noul("refund") == pytest.approx(0.12)
    assert result.usage is not None
    assert result.usage.input_tokens == 15


@pytest.mark.asyncio
async def test_typesafe_backend_conversion_and_error_handling() -> None:
    # Test _to_typesafe_question with Score and Mapping dicts
    s_conv = _to_typesafe_question(
        Score(instructions="How severe?", criteria=["low", "high"])
    )
    assert isinstance(s_conv, typesafe_sdk.Score)

    c_dict = _to_typesafe_question(
        {"type": "choice", "instructions": "Pick?", "criteria": {"a": None, "b": None}}
    )
    assert isinstance(c_dict, typesafe_sdk.Choice)

    s_dict = _to_typesafe_question(
        {"type": "score", "instructions": "Rate?", "criteria": ["low", "high"]}
    )
    assert isinstance(s_dict, typesafe_sdk.Score)

    n_dict = _to_typesafe_question({"type": "noul", "instructions": "Yes?"})
    assert isinstance(n_dict, typesafe_sdk.Noul)

    n_custom_criteria = _to_typesafe_question(
        Noul(
            instructions="Compliant?",
            criteria={"ticket_and_scope": "Has ticket", "least_privilege": "Bounded"},
        )
    )
    assert isinstance(n_custom_criteria, typesafe_sdk.Noul)
    assert n_custom_criteria.criteria == {
        "true": {"ticket_and_scope": "Has ticket", "least_privilege": "Bounded"}
    }

    assert _question_kind({"type": "choice"}) == "choice"
    assert _question_kind(object()) == "unknown"

    with pytest.raises(JudgmentConfigError):
        _to_typesafe_question(12345)

    # Client that rejects model kwarg (TypeError fallback) and then failing client
    class LegacyClient:
        async def system_one(self, *, state, questions):
            return SimpleNamespace(
                model="legacy-judgment",
                answers={"q": SimpleNamespace(noul=0.55)},
            )

    legacy_backend = TypeSafeBackend(client=LegacyClient())
    res = await legacy_backend.evaluate(
        state="hi", questions={"q": Noul(instructions="Hi?")}
    )
    assert res.noul("q") == pytest.approx(0.55)

    class BrokenClient:
        async def system_one(self, **kwargs):
            raise RuntimeError("network timeout")

    broken_backend = TypeSafeBackend(client=BrokenClient())
    with pytest.raises(JudgmentEvaluationError, match="network timeout"):
        await broken_backend.evaluate(
            state="hi", questions={"q": Noul(instructions="Hi?")}
        )


@pytest.mark.asyncio
async def test_typesafe_backend_missing_api_key_raises_clean_error(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    backend = TypeSafeBackend(api_key=None)
    with pytest.raises(JudgmentConfigError, match="TYPESAFE_API_KEY"):
        await backend.evaluate(
            state="hello",
            questions={"q": Noul(instructions="Is greeting?")},
        )
