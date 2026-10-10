"""Baseline decision policy: recommend, ask a clarifying question, or abstain.

Order of reasoning:
1. Hard limits first. An option that fails one is excluded, whatever its score.
2. No priorities (weights) given: recommend only if one option dominates on every
   criterion; otherwise ask how much each criterion matters.
3. Score with intervals. Recommend the leader only if no other option could still
   overtake it once the missing values are known. Otherwise ask about exactly the
   missing values that could change the outcome, not every gap.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from adapt.engine.constraints import Status, check, describe
from adapt.engine.scoring import Scored, dominant_option, score_options
from adapt.schema import DecisionFrame

ENGINE_VERSION = "baseline-0.1.0"
NEAR_TIE = 0.02  # score gap below which a recommendation is flagged as a near tie
_EPS = 1e-9


class Outcome(str, Enum):
    RECOMMEND = "recommend"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class Question(BaseModel):
    attribute: str | None
    option_ids: list[str] = Field(default_factory=list)
    prompt: str


class RankedOption(BaseModel):
    option_id: str
    label: str
    score_low: float
    score_high: float
    contributions: dict[str, tuple[float, float]]
    unknown_constraints: list[str] = Field(default_factory=list)


class Excluded(BaseModel):
    option_id: str
    label: str
    failed: list[str]


class DecisionResult(BaseModel):
    outcome: Outcome
    recommended: str | None = None
    reason: str
    ranking: list[RankedOption] = Field(default_factory=list)
    excluded: list[Excluded] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    near_tie: bool = False
    engine_version: str = ENGINE_VERSION


def decide(frame: DecisionFrame) -> DecisionResult:
    labels = {o.id: o.label for o in frame.options}
    assumptions = frame.assumptions()

    if not frame.options:
        return DecisionResult(
            outcome=Outcome.CLARIFY,
            reason="No options were given.",
            questions=[Question(attribute=None, prompt="Which options are you choosing between?")],
            assumptions=assumptions,
        )

    checks = {o.id: [check(o, c) for c in frame.constraints] for o in frame.options}
    excluded = [
        Excluded(
            option_id=o.id,
            label=o.label,
            failed=[describe(ch.constraint) for ch in checks[o.id] if ch.status is Status.FAIL],
        )
        for o in frame.options
        if any(ch.status is Status.FAIL for ch in checks[o.id])
    ]
    excluded_ids = {e.option_id for e in excluded}
    candidates = [o for o in frame.options if o.id not in excluded_ids]
    unknown = {
        o.id: [ch.constraint.attribute for ch in checks[o.id] if ch.status is Status.UNKNOWN]
        for o in candidates
    }
    common = dict(excluded=excluded, assumptions=assumptions)

    if not candidates:
        return DecisionResult(
            outcome=Outcome.ABSTAIN,
            reason="None of the options satisfies every hard limit.",
            **common,
        )

    if not frame.criteria:
        only = candidates[0]
        if len(candidates) == 1 and not unknown[only.id]:
            return DecisionResult(
                outcome=Outcome.RECOMMEND,
                recommended=only.id,
                reason=f"{only.label} is the only option that meets every hard limit.",
                **common,
            )
        return DecisionResult(
            outcome=Outcome.CLARIFY,
            reason="No criteria were given to compare the options.",
            questions=[Question(attribute=None, prompt="What matters most to you when comparing these options?")],
            **common,
        )

    weights = {c.attribute: c.weight for c in frame.criteria}
    if any(w is None for w in weights.values()) or sum(w or 0 for w in weights.values()) <= 0:
        dom = dominant_option(candidates, frame.criteria)
        if dom is not None and not any(unknown.values()):
            return DecisionResult(
                outcome=Outcome.RECOMMEND,
                recommended=dom,
                reason=(
                    f"Your priorities are unknown, but {labels[dom]} is at least as good as "
                    "every other option on every criterion."
                ),
                **common,
            )
        # All-zero weights carry no priority information either, so ask about every criterion.
        to_ask = frame.unweighted_criteria() or [c.attribute for c in frame.criteria]
        return DecisionResult(
            outcome=Outcome.CLARIFY,
            reason="The best option depends on how much each criterion matters to you.",
            questions=[
                Question(
                    attribute=a,
                    prompt=f"How much does {a} matter to you compared with the other criteria?",
                )
                for a in to_ask
            ],
            **common,
        )

    scored = score_options(candidates, frame.criteria, weights)  # type: ignore[arg-type]
    ranking = [
        RankedOption(
            option_id=s.option_id,
            label=labels[s.option_id],
            score_low=round(s.low, 4),
            score_high=round(s.high, 4),
            contributions={k: (round(a, 4), round(b, 4)) for k, (a, b) in s.contributions.items()},
            unknown_constraints=unknown[s.option_id],
        )
        for s in sorted(scored, key=lambda s: (s.mid, s.low), reverse=True)
    ]
    common["ranking"] = ranking

    known = [s for s in scored if not unknown[s.option_id]]
    if not known:
        return DecisionResult(
            outcome=Outcome.CLARIFY,
            reason="Every remaining option has a hard limit that can't be checked yet.",
            questions=_questions(scored, unknown, labels),
            **common,
        )

    leader = max(known, key=lambda s: (s.low, s.high))
    rivals = [s for s in scored if s.option_id != leader.option_id and s.high > leader.low + _EPS]
    if rivals:
        return DecisionResult(
            outcome=Outcome.CLARIFY,
            reason="The best option depends on information that is missing.",
            questions=_questions([leader, *rivals], unknown, labels),
            **common,
        )

    runner_up = max((s.high for s in scored if s is not leader), default=None)
    return DecisionResult(
        outcome=Outcome.RECOMMEND,
        recommended=leader.option_id,
        reason=f"{labels[leader.option_id]} scores highest on your weighted criteria.",
        near_tie=runner_up is not None and leader.low - runner_up < NEAR_TIE,
        **common,
    )


def _questions(
    involved: list[Scored], unknown: dict[str, list[str]], labels: dict[str, str]
) -> list[Question]:
    """One question per attribute whose value could change the outcome."""
    asked: dict[str, list[str]] = {}
    for s in involved:
        for attr in [*s.missing, *unknown[s.option_id]]:
            ids = asked.setdefault(attr, [])
            if s.option_id not in ids:
                ids.append(s.option_id)
    return [
        Question(
            attribute=attr,
            option_ids=ids,
            prompt=f"What is the {attr} of {', '.join(labels[i] for i in ids)}?",
        )
        for attr, ids in asked.items()
    ]
