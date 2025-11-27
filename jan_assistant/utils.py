"""Utility helpers for Jan-v1 research workflows."""

from __future__ import annotations

import re
from typing import List


STANDARD_SECTIONS: List[str] = [
    "EXECUTIVE SUMMARY",
    "INTRODUCTION",
    "DETAILED ANALYSIS",
    "CURRENT TRENDS AND DEVELOPMENTS",
    "IMPLICATIONS AND RECOMMENDATIONS",
    "CONCLUSION",
    "ABSTRACT",
    "METHODOLOGY",
    "FINDINGS",
    "DISCUSSION",
    "OVERVIEW",
    "KEY INSIGHTS",
    "RECOMMENDATIONS",
    "NEXT STEPS",
]


def sections_for_format(fmt: str) -> List[str]:
    """Return ordered section headers for a given report format."""

    normalized = (fmt or "").strip().lower()
    if normalized == "executive":
        return ["EXECUTIVE SUMMARY"]
    if normalized == "detailed":
        return [
            "INTRODUCTION",
            "DETAILED ANALYSIS",
            "CURRENT TRENDS AND DEVELOPMENTS",
            "IMPLICATIONS AND RECOMMENDATIONS",
            "CONCLUSION",
        ]
    if normalized == "academic":
        return [
            "ABSTRACT",
            "INTRODUCTION",
            "METHODOLOGY",
            "FINDINGS",
            "DISCUSSION",
            "CONCLUSION",
        ]
    if normalized == "presentation":
        return [
            "OVERVIEW",
            "KEY INSIGHTS",
            "RECOMMENDATIONS",
            "NEXT STEPS",
            "CONCLUSION",
        ]
    return ["INTRODUCTION", "DETAILED ANALYSIS", "CONCLUSION"]


def extract_final_block(text: str) -> str:
    """Extract and sanitize the <final>...</final> region returned by the model."""

    match = re.search(r"<final>([\s\S]*?)</final>", text, flags=re.IGNORECASE)
    cleaned = match.group(1).strip() if match else text
    preamble_patterns = [
        r"^(?:note:|okay,|hmm,|internal|let me|i (?:will|'ll)|as an ai|thinking|plan:|"
        r"here is your report|the following is|i have prepared|i am presenting|"
        r"based on the provided information|below is the report|i hope this meets"
        r" your requirements|this report outlines|this is the final report).*?$",
        r"^(?:here is the report|i have compiled the report|the report is provided"
        r" below|this is the requested report).*?$",
        r"^(?:please find the report below|here's the report).*?$",
    ]
    for pattern in preamble_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"(?m)^\s*[-*•]\s+", "", cleaned)
    cleaned = re.sub(r"[#`*_]{1,3}", "", cleaned)
    first_pos = -1
    for header in sorted(STANDARD_SECTIONS, key=len, reverse=True):
        match = re.search(r"\\b" + re.escape(header) + r"\\b", cleaned, flags=re.IGNORECASE)
        if match and (first_pos == -1 or match.start() < first_pos):
            first_pos = match.start()
    if first_pos >= 0:
        cleaned = cleaned[first_pos:].strip()
    return cleaned


__all__ = ["sections_for_format", "extract_final_block", "STANDARD_SECTIONS"]
