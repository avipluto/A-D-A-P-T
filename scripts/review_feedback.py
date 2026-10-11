"""Review feedback before it can ever be used for training.

    python scripts/review_feedback.py list                  # pending feedback with context
    python scripts/review_feedback.py accept <feedback_id> [...]
    python scripts/review_feedback.py reject <feedback_id> [...]
    python scripts/review_feedback.py export                # accepted → versioned dataset

Only 'accepted' feedback is exported. Use ADAPT_FEEDBACK_DB to point at another database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

from adapt import config
from adapt.lifecycle import feedback_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("list", "accept", "reject", "export"))
    p.add_argument("ids", nargs="*")
    args = p.parse_args()
    db = config.FEEDBACK_DB
    if not db.exists():
        raise SystemExit(f"no feedback database at {db}")

    if args.action == "export":
        path = feedback_data.export(db, config.PROCESSED_DIR)
        m = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        print(f"exported {m['examples']} examples ({m['train']} train / {m['heldout']} held out) -> {path}")
        return

    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    if args.action == "list":
        rows = c.execute(
            "SELECT f.id, f.kind, f.rating, f.chosen_option_id, f.notes, f.created_at, d.frame_json, d.result_json "
            "FROM feedback f JOIN decisions d ON d.id = f.decision_id WHERE f.review_status = 'pending' "
            "ORDER BY f.created_at").fetchall()
        for r in rows:
            frame = json.loads(r["frame_json"])
            result = json.loads(r["result_json"]) if r["result_json"] else {}
            labels = {o["id"]: o["label"] for o in frame["options"]}
            print(f"\n{r['id']}  [{r['kind']}]  rating={r['rating']}  {r['created_at'][:16]}")
            print(f"  situation: {frame['situation'][:100]}")
            print(f"  ADAPT said: {result.get('outcome')} {labels.get(result.get('recommended'), '')}")
            print(f"  person chose: {labels.get(r['chosen_option_id'], '-')}   notes: {r['notes'] or '-'}")
        print(f"\n{len(rows)} pending")
        return

    status = "accepted" if args.action == "accept" else "rejected"
    now = datetime.now(timezone.utc).isoformat()
    with c:
        n = sum(c.execute("UPDATE feedback SET review_status=?, reviewed_at=? WHERE id=? AND review_status='pending'",
                          (status, now, i)).rowcount for i in args.ids)
    print(f"{status} {n} of {len(args.ids)}")


if __name__ == "__main__":
    main()
