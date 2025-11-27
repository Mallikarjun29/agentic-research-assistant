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


__all__ = ["ResearchConfig"]
