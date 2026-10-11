"""SQLite store for decisions and feedback. Local file, standard library only.

Every decision is stored with its request, the frame actually used, the result and all
component versions, so any answer can be traced and reproduced. Feedback keeps three
things apart, because they mean different things:

- satisfaction: did the person like the recommendation? (1-5)
- outcome: how did the choice actually turn out, later? (1-5 plus free text)
- correction: the person says something in the frame or result was wrong

Every feedback row starts as review_status = 'pending'. Nothing here trains anything:
only reviewed rows may be exported for offline retraining (phase 6).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_kind TEXT NOT NULL,            -- 'form' | 'text' | 'confirmed_text'
    status TEXT NOT NULL,                -- 'decided' | 'needs_confirmation'
    request_json TEXT NOT NULL,
    frame_json TEXT NOT NULL,
    result_json TEXT,
    versions_json TEXT NOT NULL,
    parent_id TEXT REFERENCES decisions(id)   -- the extraction this confirmed decision came from
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL REFERENCES decisions(id),
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('satisfaction', 'outcome', 'correction')),
    chosen_option_id TEXT,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    details_json TEXT,
    notes TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK (review_status IN ('pending', 'accepted', 'rejected')),
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS feedback_by_decision ON feedback(decision_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    def save_decision(self, *, input_kind: str, status: str, request: dict, frame: dict, result: dict | None,
                      versions: dict, parent_id: str | None = None) -> str:
        did = str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                (did, _now(), input_kind, status, json.dumps(request), json.dumps(frame),
                 None if result is None else json.dumps(result), json.dumps(versions), parent_id),
            )
        return did

    def get_decision(self, did: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM decisions WHERE id = ?", (did,)).fetchone()
        return None if row is None else dict(row)

    def save_feedback(self, *, decision_id: str, kind: str, chosen_option_id: str | None, rating: int | None,
                      details: dict | None, notes: str | None) -> str:
        fid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute(
                "INSERT INTO feedback (id, decision_id, created_at, kind, chosen_option_id, rating, details_json, notes)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (fid, decision_id, _now(), kind, chosen_option_id, rating,
                 None if details is None else json.dumps(details), notes),
            )
        return fid

    def feedback_for(self, decision_id: str) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM feedback WHERE decision_id = ? ORDER BY created_at",
                                               (decision_id,))]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {
                "decisions": c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
                "feedback_pending": c.execute("SELECT COUNT(*) FROM feedback WHERE review_status='pending'").fetchone()[0],
            }
