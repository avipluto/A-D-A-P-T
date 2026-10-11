"""Protection for a public deployment: per-visitor rate limits and request size limits.

In-memory sliding windows (one process, as on a single Hugging Face Space). Visitors
are identified by IP; behind a proxy, uvicorn's --proxy-headers makes that the real
client address. Limits reset if the server restarts, which is acceptable for a demo.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from adapt.schema import DecisionFrame

MAX_TEXT = 4000
MAX_OPTIONS = 12
MAX_ATTRIBUTES = 20
MAX_RULES = 20  # criteria + constraints


class RateLimiter:
    def __init__(self, limits: dict[str, tuple[int, float]]):
        """limits: name -> (max requests, window seconds)"""
        self.limits = limits
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, name: str, request: Request) -> None:
        limit, window = self.limits[name]
        key = (name, request.client.host if request.client else "unknown")
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry = int(window - (now - q[0])) + 1
                raise HTTPException(429, f"Too many requests: limit is {limit} per {int(window // 60)} minutes. "
                                         f"Try again in {retry} s.", headers={"Retry-After": str(retry)})
            q.append(now)


PUBLIC_LIMITS = {"decision": (30, 600.0), "feedback": (60, 600.0)}


def check_frame_size(f: DecisionFrame) -> None:
    attrs = {a for o in f.options for a in o.attributes} | {c.attribute for c in [*f.criteria, *f.constraints]}
    problems = []
    if len(f.options) > MAX_OPTIONS:
        problems.append(f"at most {MAX_OPTIONS} options")
    if len(attrs) > MAX_ATTRIBUTES:
        problems.append(f"at most {MAX_ATTRIBUTES} attributes")
    if len(f.criteria) + len(f.constraints) > MAX_RULES:
        problems.append(f"at most {MAX_RULES} criteria and limits together")
    if len(f.situation) > MAX_TEXT:
        problems.append(f"at most {MAX_TEXT} characters of text")
    if problems:
        raise HTTPException(422, "Request too large: " + "; ".join(problems))
