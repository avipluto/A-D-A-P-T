"""Decision policy: recommend, ask a clarifying question, or abstain.

The scorer is pluggable (linear baseline or a learned model); the rules below
are the same for every scorer, so no model can bypass a hard limit.

Order of reasoning:
1. Hard limits first. An option that fails one is excluded, whatever its score.
2. No priorities (weights) given: recommend only if one option dominates on every
   criterion; otherwise ask how much each criterion matters.
3. Missing values: try plausible values for each (engine/missing.py). Recommend only
   if the same option wins every time; otherwise ask about exactly the values that
   changed the winner, not every gap.
4. Complete information: recommend the top score, and report how far each priority
   would have to move to change the winner (engine/sensitivity.py).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from adapt.engine.constraints import Status, check, describe
from adapt.engine.missing import Slot, missing_slots, robust_winners
from adapt.engine.scoring import BASELINE, Scorer, dominant_option
from adapt.engine.sensitivity import FRAGILE, Flip, weight_flips
from adapt.schema import DecisionFrame

ENGINE_VERSION = "policy-0.2.0"
NEAR_TIE = 0.02  # score gap below which a recommendation is flagged as a near tie
# How far beyond the known values a missing value is assumed to plausibly lie, in units
# of their spread. Chosen on the validation split; see docs/results/phase2b_missing_values.md.
DEFAULT_EXTENSION = 2.0


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
    sensitivity: list[Flip] = Field(default_factory=list)
    fragile: bool = False  # a small change in priorities would change the recommendation
    notes: list[str] = Field(default_factory=list)  # assumptions made and what would flip the result
    engine_version: str = ENGINE_VERSION
    scorer_version: str = BASELINE.version


def decide(
    frame: DecisionFrame, scorer: Scorer = BASELINE, extension: float = DEFAULT_EXTENSION
) -> DecisionResult:
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
    common = dict(excluded=excluded, assumptions=assumptions, scorer_version=scorer.version)

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

    weights_ = {a: float(w) for a, w in weights.items()}  # all known and positive past this point
    scored = scorer(candidates, frame.criteria, weights_)
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

    needed = list(dict.fromkeys([c.attribute for c in frame.criteria] + [c.attribute for c in frame.constraints]))
    slots = missing_slots(candidates, needed)
    if slots:
        fr = robust_winners(
            candidates, frame.options, frame.criteria, frame.constraints, weights_, scorer, slots, extension
        )
        if fr is None or not fr.winners:
            return DecisionResult(
                outcome=Outcome.CLARIFY,
                reason="The best option depends on information that is missing.",
                questions=_questions(slots if fr is None else fr.deciding, labels),
                **common,
            )
        mid = {s.option_id: s.mid for s in scored}
        best = max(fr.winners, key=lambda o: mid[o])
        if unknown[best]:
            # Never recommend on an *assumed* hard limit: a wrong guess here is a violation.
            return DecisionResult(
                outcome=Outcome.CLARIFY,
                reason=f"{labels[best]} looks best, but whether it meets every hard limit is unknown.",
                questions=_questions([Slot(best, a) for a in unknown[best]], labels),
                **common,
            )
        return DecisionResult(
            outcome=Outcome.RECOMMEND,
            recommended=best,
            reason=(
                f"{labels[best]} comes out on top whatever the missing values are, "
                "within the ranges assumed below."
            ),
            notes=[
                f"Assumed {s.attribute} of {labels[s.option_id]} is between {_fmt(lo)} and {_fmt(hi)}."
                for s, (lo, hi) in fr.ranges.items()
            ]
            + (["Checked each missing value separately (too many combinations to try them all)."] if fr.approximate else []),
            **common,
        )

    leader = max(scored, key=lambda s: s.low)
    runner_up = max((s.high for s in scored if s is not leader), default=None)
    flips = weight_flips(scored, weights_, leader.option_id)
    notes = [
        f"{labels[f.new_leader]} would win if {f.attribute}'s share of importance went from "
        f"{f.current_weight:.2f} to {f.flip_weight:.2f}."
        for f in flips[:3]
    ]
    return DecisionResult(
        outcome=Outcome.RECOMMEND,
        recommended=leader.option_id,
        reason=f"{labels[leader.option_id]} scores highest on your weighted criteria.",
        near_tie=runner_up is not None and leader.low - runner_up < NEAR_TIE,
        sensitivity=flips,
        fragile=bool(flips) and flips[0].distance < FRAGILE,
        notes=notes,
        **common,
    )


def _fmt(x: object) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _questions(slots: list[Slot], labels: dict[str, str]) -> list[Question]:
    """One question per attribute whose value could change the outcome."""
    asked: dict[str, list[str]] = {}
    for s in slots:
        ids = asked.setdefault(s.attribute, [])
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
