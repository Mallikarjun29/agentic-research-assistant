"""Free, keyless web retrieval for scout briefs.

Uses the ``ddgs`` package pinned to DuckDuckGo's own backend: no API key,
account, paid service, ads or trackers, and no multi-engine rotation. If
DuckDuckGo is unavailable or rate limits the request, search returns an
empty list and the caller falls back to clearly labeled model-only notes;
the app never routes around a block or challenge.

Results are search-result snippets (title, URL and the short summary text
the search engine returns). Full pages are not fetched.
"""

from __future__ import annotations

from typing import Dict, List

# DuckDuckGo date filters. "comprehensive" applies no date filter.
_TIMEFRAME_TO_DDG = {"current": "m", "recent": "y"}


def search_web(
    query: str,
    max_results: int = 3,
    timeframe: str = "comprehensive",
    timeout: float = 15.0,
) -> List[Dict]:
    """Return up to ``max_results`` hits as {title, url, snippet} dicts.

    Returns an empty list when search is unavailable or rate limited, so
    callers fall back to model-only notes instead of evading the block.
    """

    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        return []

    kwargs = {"backend": "duckduckgo"}
    timelimit = _TIMEFRAME_TO_DDG.get(timeframe)
    if timelimit:
        kwargs["timelimit"] = timelimit
    try:
        with DDGS(timeout=timeout) as client:
            hits = client.text(query, max_results=max_results, **kwargs)
    except Exception:
        return []
    results: List[Dict] = []
    for hit in hits or []:
        url = (hit.get("href") or "").strip()
        if not url.startswith("http"):
            continue
        results.append(
            {
                "title": (hit.get("title") or "").strip() or url,
                "url": url,
                "snippet": (hit.get("body") or "").strip(),
            }
        )
        if len(results) >= max_results:
            break
    return results


__all__ = ["search_web"]
