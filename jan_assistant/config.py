"""Configuration primitives for the Jan-v1 research assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class ResearchConfig:
    model_id: str = "janhq/Jan-v1-4B"
    hf_auth_token: str = os.getenv("HF_TOKEN", "")
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    context_length: int = 4096
    device_map: str = "auto"
    torch_dtype: str = "float16"
    # Web retrieval (free DuckDuckGo search, no API key or account).
    use_web_search: bool = True
    web_results_per_query: int = 3
    web_max_queries: int = 10
    web_timeout: float = 15.0


__all__ = ["ResearchConfig"]
