"""Core Jan-v1 research assistant logic powered by Hugging Face models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ResearchConfig
from .utils import extract_final_block, sections_for_format


class DeepResearchAssistant:
    """Orchestrates Jan-v1 GPU inference plus structured research workflows."""

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    def load_model(self) -> bool:
        """Load the Hugging Face model onto the available accelerator."""

        token = self.config.hf_auth_token or None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                use_fast=True,
                trust_remote_code=True,
                token=token,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                device_map=self.config.device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=token,
            )
            self.model.eval()
            return True
        except Exception as exc:  # pragma: no cover - depends on env/GPU
            print(f"Model loading failed: {exc}")
            self.model = None
            self.tokenizer = None
            return False

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        if not self.model or not self.tokenizer:
            return "Model not loaded."
        mt = max_tokens or self.config.max_tokens
        device = self._model_device()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = {
            "max_new_tokens": mt,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        try:
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            generated = output[0][inputs["input_ids"].shape[-1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as exc:  # pragma: no cover - runtime safeguard
            return f"Error generating response: {exc}"

    def generate_search_queries(self, topic: str, focus_area: str, depth: str) -> List[str]:
        base_queries = [
            "overview",
            "recent developments",
            "case studies",
            "policy and regulation",
            "technical approaches",
            "market analysis",
            "statistics",
            "risk considerations",
        ]
        focus_modifiers = {
            "general": "",
            "academic": " academic literature",
            "business": " business impact",
            "technical": " implementation details",
            "policy": " regulatory landscape",
        }
        counts = {"surface": 5, "moderate": 8, "deep": 15, "comprehensive": 25}
        n_queries = counts.get(depth, 8)
        modifier = focus_modifiers.get(focus_area, "")
        expanded = [f"{topic} {phrase}{modifier}".strip() for phrase in base_queries]
        if n_queries > len(expanded):
            expanded.extend([f"{topic} strategic insight {i}" for i in range(n_queries - len(expanded))])
        return expanded[:n_queries]

    def synthesize_research(
        self,
        topic: str,
        research_notes: List[Dict],
        focus_area: str,
        report_format: str,
    ) -> str:
        context_lines = []
        for idx, result in enumerate(research_notes[:20]):
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "")
            context_lines.append(
                f"Source {idx + 1} Title: {title}\nSource {idx + 1} Summary: {snippet}"
            )
        context = "\n".join(context_lines) or "Insufficient scout notes; reason over topic directly."
        sections = sections_for_format(report_format)
        sections_text = "\n".join(sections)
        synthesis_prompt = f"""
You are an expert research analyst. Write the final, polished report on: "{topic}" for a professional audience.
***CRITICAL INSTRUCTIONS:***
- Your entire response MUST be the final report, wrapped EXACTLY inside <final> and </final> tags.
- DO NOT output any text, thoughts, or commentary BEFORE the <final> tag or AFTER the </final> tag.
- DO NOT include conversational filler, internal reflections, or markdown syntax.
- Use full paragraphs (no bullet lists).
- Maintain a formal, evidence-aware tone that matches the focus area "{focus_area}".
- Include the following section headers, in this order, and no others:
{sections_text}
Guidance:
- Base your writing strictly on the Research Notes provided below. If the notes lack specific data, write a methodology-forward analysis without fabricating facts.
Research Notes:
{context}
Now produce ONLY the final report:
<final>
...your report here...
</final>
"""
        raw = self.generate_response(synthesis_prompt, max_tokens=1800)
        final_report = extract_final_block(raw)
        first_section = next((section for section in sections if section in final_report), None)
        if first_section:
            final_report = final_report[final_report.find(first_section) :].strip()
        return final_report

    def build_internal_sources(
        self,
        topic: str,
        queries: List[str],
        focus_area: str,
        timeframe: str,
    ) -> List[Dict]:
        """Use the model itself to draft scout notes per query."""

        sources: List[Dict] = []
        for idx, query in enumerate(queries):
            scout_prompt = f"""
You are a research scout supporting an agentic analyst.
Topic: {topic}
Focus area: {focus_area}
Timeframe: {timeframe}
Sub-question: {query}
Task: Summarize widely reported facts, figures, or narratives relevant to the sub-question.
Constraints:
- Use 3-4 full sentences (no bullets).
- Reference publication windows (e.g., "In 2024 reports...") when possible.
- If evidence is sparse, explain the best methodology to investigate the question.
Respond using the template below:
Title: <concise insight headline>
Summary: <single paragraph insight>
"""
            memo = self.generate_response(scout_prompt, max_tokens=350)
            title, summary = self._split_memo(memo)
            sources.append(
                {
                    "title": title or f"LLM Insight {idx + 1}: {query}",
                    "url": "",
                    "snippet": summary,
                    "source": "jan-v1-local",
                }
            )
        return sources

    @staticmethod
    def _split_memo(text: str) -> Tuple[str, str]:
        title = ""
        summary_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("title:"):
                title = stripped.split(":", 1)[1].strip()
            elif lowered.startswith("summary:"):
                summary_lines.append(stripped.split(":", 1)[1].strip())
            else:
                summary_lines.append(stripped)
        summary = " ".join(summary_lines).strip()
        return title, summary

    def _model_device(self) -> torch.device:
        if hasattr(self.model, "device") and self.model.device is not None:
            return self.model.device
        try:
            param = next(self.model.parameters())
            return param.device
        except StopIteration:  # pragma: no cover - degenerate case
            return torch.device("cpu")


__all__ = ["DeepResearchAssistant"]
