"""Free, keyless web retrieval for scout briefs.

Primary path uses the ``ddgs`` package (DuckDuckGo search, no API key or
account). A plain ``requests`` fallback against the DuckDuckGo HTML endpoint
keeps search working when the package is not installed. No paid services,
signups, ads or trackers are involved.
"""

from __future__ import annotations

import html
import re
from typing import Dict, List
from urllib.parse import unquote

import requests

# DuckDuckGo date filters supported by both paths.
_TIMEFRAME_TO_DDG = {"current": "m", "recent": "y"}

_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S
)
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _search_with_ddgs(query: str, max_results: int, timeframe: str) -> List[Dict]:
    from ddgs import DDGS  # type: ignore

    kwargs = {}
    timelimit = _TIMEFRAME_TO_DDG.get(timeframe)
    if timelimit:
        kwargs["timelimit"] = timelimit
    with DDGS() as client:
        hits = client.text(query, max_results=max_results, **kwargs)
    results: List[Dict] = []
    for hit in hits or []:
        url = (hit.get("href") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": (hit.get("title") or "").strip() or url,
                "url": url,
                "snippet": (hit.get("body") or "").strip(),
            }
        )
    return results


def _search_with_html_endpoint(
    query: str, max_results: int, timeframe: str, timeout: float
) -> List[Dict]:
    params = {"q": query}
    date_filter = _TIMEFRAME_TO_DDG.get(timeframe)
    if date_filter:
        params["df"] = date_filter
    response = requests.post(
        "https://html.duckduckgo.com/html/",
        data=params,
        headers=_DEFAULT_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    page = response.text
    links = _RESULT_LINK_RE.findall(page)
    snippets = _RESULT_SNIPPET_RE.findall(page)
    results: List[Dict] = []
    for idx, (raw_url, raw_title) in enumerate(links):
        redirect = _UDDG_RE.search(raw_url)
        url = unquote(redirect.group(1)) if redirect else html.unescape(raw_url)
        if not url.startswith("http"):
            continue
        results.append(
            {
                "title": _strip_tags(raw_title) or url,
                "url": url,
                "snippet": _strip_tags(snippets[idx]) if idx < len(snippets) else "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def search_web(
    query: str,
    max_results: int = 3,
    timeframe: str = "comprehensive",
    timeout: float = 15.0,
) -> List[Dict]:
    """Return up to ``max_results`` web hits as {title, url, snippet} dicts.

    Returns an empty list when search is unavailable or rate limited, so
    callers can fall back to model-only notes.
    """

    try:
        return _search_with_ddgs(query, max_results, timeframe)
    except Exception:
        pass
    try:
        return _search_with_html_endpoint(query, max_results, timeframe, timeout)
    except Exception:
        return []


__all__ = ["search_web"]
