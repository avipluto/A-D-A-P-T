"""ADAPT HTTP API.

Run:  D:\\adapt-env\\Scripts\\python.exe -m uvicorn --factory adapt.api.app:create_app --port 8000

Flow for text input (phase 3 showed extraction can't be trusted blindly):
  POST /decision {"situation": "..."}               → status "needs_confirmation" + extracted frame
  POST /decision {"frame": <edited>, "confirmed": true, "parent_id": ...}  → status "decided"
Form input (options / criteria / constraints) is decided straight away.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from adapt import config
from adapt.api.models import (
    DecisionRequest,
    DecisionResponse,
    FeedbackRequest,
    FeedbackResponse,
    ProposalOut,
    ValueIn,
)
from adapt.api.services import Services, load
from adapt.api.store import Store
from adapt.engine import decide
from adapt.extraction.schema import to_frame
from adapt.extraction.textgen import LEVEL_WEIGHT
from adapt.retrieval.evidence import domain
from adapt.retrieval.lookup import apply_proposals, lookup_missing
from adapt.schema import Constraint, Criterion, DecisionFrame, Option, Provenance, Value


def form_frame(req: DecisionRequest) -> DecisionFrame:
    def value(v) -> Value:
        v = v if isinstance(v, ValueIn) else ValueIn(value=v)
        return Value(value=v.value, unit=v.unit, provenance=Provenance.STATED)

    return DecisionFrame(
        situation=req.situation or "decision",
        domain_hint=req.domain_hint,
        goals=req.goals,
        options=[
            Option(id=chr(97 + i) if i < 26 else f"o{i}", label=o.label,
                   attributes={k: value(v) for k, v in o.attributes.items()})
            for i, o in enumerate(req.options)
        ],
        criteria=[
            Criterion(attribute=c.attribute, direction=c.direction, plausible_range=c.plausible_range,
                      weight=c.weight if c.weight is not None else (LEVEL_WEIGHT[c.priority] if c.priority else None))
            for c in req.criteria
        ],
        constraints=[Constraint(attribute=c.attribute, operator=c.operator, value=c.value, unit=c.unit)
                     for c in req.constraints],
    )


def has_inferred(f: DecisionFrame) -> bool:
    return (any(v.provenance is Provenance.INFERRED for o in f.options for v in o.attributes.values())
            or any(o.provenance is Provenance.INFERRED for o in f.options)
            or any(c.provenance is Provenance.INFERRED for c in [*f.criteria, *f.constraints]))


def confirm(f: DecisionFrame) -> DecisionFrame:
    """The user checked the frame: what was inferred is now what they stated."""
    def fix(p: Provenance) -> Provenance:
        return Provenance.STATED if p is Provenance.INFERRED else p

    return f.model_copy(update={
        "options": [o.model_copy(update={
            "provenance": fix(o.provenance),
            "attributes": {k: v.model_copy(update={"provenance": fix(v.provenance)}) for k, v in o.attributes.items()},
        }) for o in f.options],
        "criteria": [c.model_copy(update={"provenance": fix(c.provenance)}) for c in f.criteria],
        "constraints": [c.model_copy(update={"provenance": fix(c.provenance)}) for c in f.constraints],
    })


def create_app(services: Services | None = None, store: Store | None = None) -> FastAPI:
    svc = services or load()
    db = store or Store(config.FEEDBACK_DB)
    app = FastAPI(title="ADAPT", version=svc.versions()["engine"])
    app.add_middleware(  # local frontend dev server only
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        return {"versions": svc.versions(), "fallbacks": svc.notes, "store": db.counts(),
                "extraction_requires_confirmation": True}

    @app.post("/decision", response_model=DecisionResponse)
    def decision(req: DecisionRequest) -> DecisionResponse:
        versions = svc.versions()
        request_json = req.model_dump(mode="json")

        if req.frame is not None:
            if has_inferred(req.frame) and not req.confirmed:
                frame, kind = req.frame, "text"
            else:
                frame = confirm(req.frame) if req.confirmed else req.frame
                kind = "confirmed_text" if req.parent_id else "form"
        elif req.options:
            frame, kind = form_frame(req), "form"
        else:
            frame = to_frame(svc.extract(req.situation), req.situation, req.domain_hint)
            kind = "text"

        if kind == "text":  # extracted: show it, don't decide on it yet
            did = db.save_decision(input_kind=kind, status="needs_confirmation", request=request_json,
                                   frame=frame.model_dump(mode="json"), result=None, versions=versions)
            return DecisionResponse(
                decision_id=did, status="needs_confirmation", input_kind=kind, frame=frame, versions=versions,
                message="Check the extracted options, values, priorities and limits, correct anything wrong, "
                        "then send the frame back with confirmed=true.",
            )

        lookups: list[ProposalOut] = []
        message = "Decided."
        if req.lookup:
            if svc.provider is None:
                message = "Decided without web lookup: retrieval is not configured."
            else:
                proposals = lookup_missing(frame, svc.provider)
                frame = apply_proposals(frame, proposals)
                lookups = [
                    ProposalOut(option_id=p.option_id, attribute=p.attribute, status=p.status, value=p.value,
                                reasons=p.reasons,
                                sources=[{"url": c.hit.url, "domain": domain(c.hit.url), "quote": c.quote,
                                          "tier": c.tier, "freshness": c.fresh_note,
                                          "published_at": c.hit.published_at.isoformat() if c.hit.published_at else None}
                                         for c in p.support])
                    for p in proposals
                ]

        result = decide(frame, scorer=svc.scorer)
        did = db.save_decision(input_kind=kind, status="decided", request=request_json,
                               frame=frame.model_dump(mode="json"), result=result.model_dump(mode="json"),
                               versions=versions, parent_id=req.parent_id)
        return DecisionResponse(decision_id=did, status="decided", input_kind=kind, frame=frame, result=result,
                                lookups=lookups, message=message, versions=versions)

    @app.get("/decision/{decision_id}")
    def get_decision(decision_id: str) -> dict:
        d = db.get_decision(decision_id)
        if d is None:
            raise HTTPException(404, "unknown decision_id")
        return {**d, "feedback": db.feedback_for(decision_id)}

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest) -> FeedbackResponse:
        d = db.get_decision(req.decision_id)
        if d is None:
            raise HTTPException(404, "unknown decision_id")
        if req.chosen_option_id is not None:
            ids = {o["id"] for o in DecisionFrame.model_validate_json(d["frame_json"]).model_dump()["options"]}
            if req.chosen_option_id not in ids:
                raise HTTPException(422, f"chosen_option_id must be one of {sorted(ids)}")
        fid = db.save_feedback(decision_id=req.decision_id, kind=req.kind, chosen_option_id=req.chosen_option_id,
                               rating=req.rating, details=req.details, notes=req.notes)
        return FeedbackResponse(feedback_id=fid)

    return app
