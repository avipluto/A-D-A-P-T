"""Request and response bodies for the API.

Form input is deliberately simple: attribute values can be a bare number / yes-no /
text, or {"value": ..., "unit": ...}. Priorities can be numeric weights or the words
high / medium / low.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator

from adapt.engine.policy import DecisionResult
from adapt.schema import DecisionFrame, Direction, Operator

Scalar = Union[bool, int, float, str]


class ValueIn(BaseModel):
    value: Scalar
    unit: str | None = None


class OptionIn(BaseModel):
    label: str = Field(min_length=1)
    attributes: dict[str, Union[ValueIn, Scalar]] = Field(default_factory=dict)


class CriterionIn(BaseModel):
    attribute: str
    direction: Direction
    weight: float | None = Field(default=None, ge=0)
    priority: Literal["high", "medium", "low"] | None = None
    plausible_range: tuple[float, float] | None = None


class ConstraintIn(BaseModel):
    attribute: str
    operator: Operator
    value: Scalar
    unit: str | None = None


class DecisionRequest(BaseModel):
    situation: str = ""
    domain_hint: str | None = None  # optional; nothing depends on a fixed list of domains
    goals: list[str] = Field(default_factory=list)
    options: list[OptionIn] = Field(default_factory=list)
    criteria: list[CriterionIn] = Field(default_factory=list)
    constraints: list[ConstraintIn] = Field(default_factory=list)
    # A frame previously returned by this API (e.g. after extraction), edited by the user.
    frame: DecisionFrame | None = None
    confirmed: bool = False  # the user has checked `frame`
    parent_id: str | None = None  # the needs_confirmation response this confirms
    lookup: bool = False  # search the web for missing values (needs retrieval enabled)

    @model_validator(mode="after")
    def _has_input(self) -> DecisionRequest:
        if not (self.situation.strip() or self.options or self.frame):
            raise ValueError("give a situation text, structured options, or a frame")
        return self


class ProposalOut(BaseModel):
    option_id: str
    attribute: str
    status: str  # apply | confirm | none
    value: float | None
    reasons: list[str]
    sources: list[dict]


class DecisionResponse(BaseModel):
    decision_id: str
    status: Literal["decided", "needs_confirmation"]
    input_kind: Literal["form", "text", "confirmed_text"]
    frame: DecisionFrame
    result: DecisionResult | None = None
    lookups: list[ProposalOut] = Field(default_factory=list)
    message: str
    versions: dict


class FeedbackRequest(BaseModel):
    decision_id: str
    kind: Literal["satisfaction", "outcome", "correction"]
    chosen_option_id: str | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    details: dict | None = None
    notes: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def _rating_where_it_belongs(self) -> FeedbackRequest:
        if self.kind in ("satisfaction", "outcome") and self.rating is None:
            raise ValueError(f"{self.kind} feedback needs a rating from 1 to 5")
        return self


class FeedbackResponse(BaseModel):
    feedback_id: str
    review_status: str = "pending"
    message: str = "Saved for review. Feedback never changes the model automatically."
