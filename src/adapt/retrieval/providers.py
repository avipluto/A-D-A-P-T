"""Search providers. Swappable: everything else only sees `SearchProvider` and `Hit`.

TavilyProvider reads TAVILY_API_KEY from the environment (or the repo's .env file,
which is git-ignored). The key is never logged or written anywhere. Results are cached
on disk so repeated runs cost no credits and are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from adapt import config

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    content: str  # text from the page: the ONLY place values may be taken from
    published_at: datetime | None
    retrieved_at: datetime
    provider: str
    score: float | None = None


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[Hit]: ...


class StaticProvider:
    """Returns pre-set hits. For tests, offline demos, and documents a user supplies."""

    name = "static"

    def __init__(self, hits_by_query: dict[str, list[Hit]] | None = None, default: list[Hit] | None = None):
        self.hits_by_query = hits_by_query or {}
        self.default = default or []
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[Hit]:
        self.queries.append(query)
        return self.hits_by_query.get(query, self.default)[:max_results]


def load_dotenv(path: Path = config.REPO_ROOT / ".env") -> None:
    """Minimal .env reader (KEY=value lines). Doesn't override variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in (None, "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00")) if fmt is None else datetime.strptime(s, fmt)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None, cache_ttl_hours: float = 24.0,
                 search_depth: str = "basic", timeout: float = 30.0):
        load_dotenv()
        self._key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self._key:
            raise RuntimeError("TAVILY_API_KEY is not set. Put it in the adapt/.env file (never in code or chat).")
        self.cache_dir = cache_dir or (config.DATA_DIR / "cache" / "search")
        self.cache_ttl = cache_ttl_hours * 3600
        self.search_depth = search_depth  # "basic" = 1 credit per search
        self.timeout = timeout
        self.credits_used = 0

    def _cache_path(self, query: str, max_results: int) -> Path:
        h = hashlib.sha256(f"{self.name}|{self.search_depth}|{max_results}|{query}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def search(self, query: str, max_results: int = 5) -> list[Hit]:
        path = self._cache_path(query, max_results)
        if path.exists() and time.time() - path.stat().st_mtime < self.cache_ttl:
            return [_hit_from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]

        import httpx

        body = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max_results,
            "include_published_date": True,
            "include_answer": False,  # we want sources, not a model-written answer
        }
        r = httpx.post(TAVILY_URL, json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=self.timeout)
        if r.status_code == 401:
            raise RuntimeError("Tavily rejected the API key (401).")
        if r.status_code in (429, 432, 433):
            raise RuntimeError(f"Tavily rate or plan limit reached ({r.status_code}).")
        r.raise_for_status()
        self.credits_used += 1
        now = datetime.now(timezone.utc)
        hits = [
            Hit(url=x["url"], title=x.get("title") or "", content=x.get("content") or "",
                published_at=_parse_date(x.get("published_date")), retrieved_at=now, provider=self.name,
                score=x.get("score"))
            for x in r.json().get("results", [])
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([_hit_to_json(h) for h in hits], indent=1), encoding="utf-8")
        return hits


def _hit_to_json(h: Hit) -> dict:
    d = asdict(h)
    d["published_at"] = h.published_at.isoformat() if h.published_at else None
    d["retrieved_at"] = h.retrieved_at.isoformat()
    return d


def _hit_from_json(d: dict) -> Hit:
    return Hit(**{**d, "published_at": _parse_date(d["published_at"]), "retrieved_at": _parse_date(d["retrieved_at"])})
