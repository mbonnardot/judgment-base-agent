"""Unit tests for primitives.py, errors.py, and schema.py."""

from dataclasses import FrozenInstanceError

import pytest

from judgment_base_agent.errors import JudgmentConfigError
from judgment_base_agent.primitives import (
    Choice,
    ChoiceJudgment,
    JudgmentResult,
    JudgmentUsage,
    Noul,
    NoulJudgment,
    Score,
    ScoreJudgment,
    classify_confidence_tier,
)
from judgment_base_agent.schema import JudgmentField, JudgmentSchema


def test_question_primitives_immutability_and_validation() -> None:
    c = Choice(instructions="Which dept?", criteria=["billing", "tech"])
    assert c.normalized_criteria() == {"billing": None, "tech": None}
    with pytest.raises(FrozenInstanceError):
        c.instructions = "mutated"  # type: ignore[misc]

    with pytest.raises(JudgmentConfigError):
        Choice(instructions="", criteria=["a"])
    with pytest.raises(JudgmentConfigError):
        Choice(instructions="Valid?", criteria=[])
    with pytest.raises(JudgmentConfigError):
        Score(instructions="Valid?", criteria=["only_one"])
    with pytest.raises(JudgmentConfigError):
        Noul(instructions="   ")


def test_confidence_tier_and_judgment_result_accessors() -> None:
    assert classify_confidence_tier(0.85) == "high"
    assert classify_confidence_tier(0.60, floor=0.50) == "medium"
    assert classify_confidence_tier(0.40, floor=0.50) == "low"

    res = JudgmentResult(
        choices={
            "tone": ChoiceJudgment.from_raw(
                choice="frustrated",
                probabilities={"calm": 0.1, "frustrated": 0.9},
                confidence=0.82,
            )
        },
        scores={
            "urgency": ScoreJudgment.from_raw(
                score=1.8,
                legend={"0": "low", "1": "med", "2": "high"},
                probabilities={"0": 0.0, "1": 0.2, "2": 0.8},
                confidence=0.75,
            )
        },
        nouls={"billing": NoulJudgment(noul=0.94)},
        model="judgment-latest",
        usage=JudgmentUsage(input_tokens=42, output_tokens=7),
    )

    assert res.choice("tone") == "frustrated"
    assert res.confidence("tone") == pytest.approx(0.82)
    assert res.choices["tone"].confidence_tier == "high"
    assert res.score("urgency") == pytest.approx(1.8)
    assert res.confidence("urgency") == pytest.approx(0.75)
    assert res.scores["urgency"].confidence_tier == "medium"
    assert res.noul("billing") == pytest.approx(0.94)
    assert res["billing"].noul == pytest.approx(0.94)

    with pytest.raises(KeyError):
        res.get("unknown_key")


def test_declarative_judgment_schema_roundtrip() -> None:
    class TicketSchema(JudgmentSchema):
        billing: NoulJudgment = JudgmentField(
            Noul(instructions="Is `ticket` about billing?")
        )
        tone: ChoiceJudgment = JudgmentField(
            Choice(
                instructions="What is the tone?",
                criteria={"calm": None, "frustrated": None},
            )
        )
        urgency: ScoreJudgment = JudgmentField(
            Score(
                instructions="How urgent?",
                criteria=["can wait", "today"],
            )
        )

    questions = TicketSchema.build_questions()
    assert set(questions.keys()) == {"billing", "tone", "urgency"}
    assert isinstance(questions["billing"], Noul)
    assert isinstance(questions["tone"], Choice)
    assert isinstance(questions["urgency"], Score)

    raw_result = JudgmentResult(
        choices={
            "tone": ChoiceJudgment.from_raw(
                choice="calm",
                probabilities={"calm": 0.9, "frustrated": 0.1},
                confidence=0.85,
            )
        },
        scores={
            "urgency": ScoreJudgment.from_raw(
                score=0.2,
                legend={"0": "can wait", "1": "today"},
                probabilities={"0": 0.8, "1": 0.2},
                confidence=0.78,
            )
        },
        nouls={"billing": NoulJudgment(noul=0.91)},
    )

    parsed = TicketSchema.from_result(raw_result)
    assert parsed.billing.noul == pytest.approx(0.91)
    assert parsed.tone.choice == "calm"
    assert parsed.urgency.score == pytest.approx(0.2)
    assert parsed.raw_result == raw_result

    # Ensure ADK 2.0 FunctionNode dict->Pydantic auto-conversion works
    dumped = parsed.model_dump()
    rehydrated = TicketSchema.model_validate(dumped)
    assert rehydrated.tone.choice == "calm"

    # Additional edge cases for 100% coverage
    assert Choice(instructions="Q", criteria={"a": "desc", "b": None}).normalized_criteria() == {
        "a": "desc",
        "b": None,
    }
    assert Score(instructions="S", criteria=["l1", "l2"]).normalized_criteria() == ["l1", "l2"]
    with pytest.raises(JudgmentConfigError):
        Score(instructions="   ", criteria=["l1", "l2"])
    with pytest.raises(JudgmentConfigError):
        JudgmentField(None)

    class EmptySchema(JudgmentSchema):
        plain: str = "x"

    with pytest.raises(JudgmentConfigError):
        EmptySchema.build_questions()

    assert raw_result.confidence("billing") == pytest.approx(abs(0.91 - 0.5) * 2.0)
    with pytest.raises(KeyError):
        raw_result.choice("unknown")
    with pytest.raises(KeyError):
        raw_result.score("unknown")
    with pytest.raises(KeyError):
        raw_result.noul("unknown")
    with pytest.raises(KeyError):
        raw_result.confidence("unknown")

