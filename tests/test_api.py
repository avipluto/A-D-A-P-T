from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from adapt.api.app import create_app
from adapt.api.services import Services
from adapt.api.store import Store
from adapt.engine.scoring import BASELINE
from adapt.extraction.rules import extract
from adapt.retrieval.lookup import query_for
from adapt.retrieval.providers import Hit, StaticProvider

NOW = datetime.now(timezone.utc)


def make_client(tmp_path, provider=None) -> tuple[TestClient, Store]:
    store = Store(tmp_path / "t.sqlite")
    services = Services(scorer=BASELINE, extract=extract, extractor_version="rules-v1", provider=provider)
    return TestClient(create_app(services, store)), store


FORM = {
    "situation": "Pick a laptop",
    "options": [
        {"label": "Laptop A", "attributes": {"price": {"value": 72000, "unit": "INR"}, "battery_hours": 9}},
        {"label": "Laptop B", "attributes": {"price": {"value": 58000, "unit": "INR"}, "battery_hours": 6}},
    ],
    "criteria": [
        {"attribute": "price", "direction": "minimize", "priority": "high"},
        {"attribute": "battery_hours", "direction": "maximize", "priority": "low"},
    ],
    "constraints": [{"attribute": "price", "operator": "<=", "value": 80000}],
}


def test_health_and_model_info(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["versions"]["scorer"] == "linear-minmax" and info["extraction_requires_confirmation"] is True


def test_form_input_is_decided_and_stored(tmp_path):
    client, store = make_client(tmp_path)
    r = client.post("/decision", json=FORM).json()
    assert r["status"] == "decided" and r["input_kind"] == "form"
    assert r["result"]["recommended"] == "b"  # price matters most
    assert r["frame"]["options"][0]["attributes"]["price"]["provenance"] == "stated"
    saved = client.get(f"/decision/{r['decision_id']}").json()
    assert saved["status"] == "decided" and '"engine"' in saved["versions_json"]


def test_hard_limit_is_respected_through_the_api(tmp_path):
    client, _ = make_client(tmp_path)
    form = {**FORM, "constraints": [{"attribute": "price", "operator": "<=", "value": 60000}],
            "criteria": [{"attribute": "battery_hours", "direction": "maximize", "weight": 1}]}
    r = client.post("/decision", json=form).json()
    assert r["result"]["recommended"] == "b" and r["result"]["excluded"][0]["option_id"] == "a"


TEXT = ("Help me pick a laptop. Laptop A has a price of ₹72,000 and battery life of 9 hours. "
        "Laptop B: price ₹58,000 and battery life 6 hours. Lower price is better, and it matters most to me. "
        "I want more battery life, and it only matters a little.")


def test_text_needs_confirmation_then_decides(tmp_path):
    client, _ = make_client(tmp_path)
    first = client.post("/decision", json={"situation": TEXT}).json()
    assert first["status"] == "needs_confirmation" and first["result"] is None
    assert first["frame"]["options"][0]["attributes"]["price"]["provenance"] == "inferred"

    # sending the frame back WITHOUT confirming still doesn't decide
    again = client.post("/decision", json={"frame": first["frame"]}).json()
    assert again["status"] == "needs_confirmation"

    done = client.post("/decision", json={"frame": first["frame"], "confirmed": True,
                                          "parent_id": first["decision_id"]}).json()
    assert done["status"] == "decided" and done["input_kind"] == "confirmed_text"
    assert done["frame"]["options"][0]["attributes"]["price"]["provenance"] == "stated"
    assert done["result"]["recommended"] == "b"


def test_lookup_fills_missing_values_with_citations(tmp_path):
    hit = Hit(url="https://www.acme.com/laptop-c", title="Acme Laptop C",
              content="The Acme Laptop C battery life is up to 15 hours.", published_at=NOW - timedelta(days=20),
              retrieved_at=NOW, provider="static")
    provider = StaticProvider({query_for("Acme Laptop C", "battery_hours"): [hit]})
    client, _ = make_client(tmp_path, provider)
    form = {**FORM, "options": [*FORM["options"], {"label": "Acme Laptop C", "attributes": {"price": 70000}}],
            "lookup": True}
    r = client.post("/decision", json=form).json()
    assert [(x["option_id"], x["status"]) for x in r["lookups"]] == [("c", "apply")]
    assert r["result"]["citations"][0]["url"] == "https://www.acme.com/laptop-c"


def test_lookup_without_provider_says_so(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/decision", json={**FORM, "lookup": True}).json()
    assert "retrieval is not configured" in r["message"]


def test_feedback_keeps_satisfaction_and_outcome_apart_and_pending(tmp_path):
    client, store = make_client(tmp_path)
    did = client.post("/decision", json=FORM).json()["decision_id"]
    assert client.post("/feedback", json={"decision_id": did, "kind": "satisfaction"}).status_code == 422  # needs rating
    assert client.post("/feedback", json={"decision_id": "nope", "kind": "satisfaction", "rating": 4}).status_code == 404
    assert client.post("/feedback", json={"decision_id": did, "kind": "outcome", "rating": 2,
                                          "chosen_option_id": "z"}).status_code == 422
    ok1 = client.post("/feedback", json={"decision_id": did, "kind": "satisfaction", "rating": 5, "chosen_option_id": "b"})
    ok2 = client.post("/feedback", json={"decision_id": did, "kind": "outcome", "rating": 2,
                                         "notes": "battery was worse than expected"})
    assert ok1.status_code == ok2.status_code == 200 and ok1.json()["review_status"] == "pending"
    rows = store.feedback_for(did)
    assert [(r["kind"], r["rating"], r["review_status"]) for r in rows] == [("satisfaction", 5, "pending"), ("outcome", 2, "pending")]


def test_web_page_is_served_and_does_not_hide_the_api(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200 and "<title>ADAPT</title>" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}  # API routes still win over the static mount


def test_page_never_uses_innerhtml():
    # web quotes and extracted text must be inserted as text, never as markup
    from pathlib import Path

    import adapt.api

    js = (Path(adapt.api.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js.replace("never innerHTML", "")


def test_empty_request_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.post("/decision", json={}).status_code == 422
