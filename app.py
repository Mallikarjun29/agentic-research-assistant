from __future__ import annotations

import json
import time
from datetime import datetime

import streamlit as st

from jan_assistant import DeepResearchAssistant, ResearchConfig


def init_session_state() -> None:
    if "assistant" not in st.session_state:
        config = ResearchConfig()
        assistant = DeepResearchAssistant(config)
        st.session_state.assistant = assistant
        st.session_state.model_loaded = assistant.load_model()
        st.session_state.results = None


def main() -> None:
    st.set_page_config(page_title="Deep Research Assistant", layout="wide")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.25rem; border-radius: 10px; color: white; text-align: center;
                    margin-bottom: 1rem;">
            <h1 style="margin:0;">Deep Research Assistant (Jan-v1)</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    init_session_state()
    assistant: DeepResearchAssistant = st.session_state.assistant
    st.header("Research Configuration")
    topic = st.text_area(
        "Research Topic",
        placeholder="Impact of AI copilots on enterprise productivity",
        height=100,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        depth = st.selectbox(
            "Research Depth",
            ["surface", "moderate", "deep", "comprehensive"],
            index=1,
            format_func=lambda opt: {
                "surface": "Surface (5-8 sources)",
                "moderate": "Moderate (10-15)",
                "deep": "Deep Dive (20-30)",
                "comprehensive": "Comprehensive (40+)",
            }[opt],
        )
        focus = st.selectbox(
            "Focus Area",
            ["general", "academic", "business", "technical", "policy"],
            index=0,
            format_func=str.title,
        )
    with col_right:
        timeframe = st.selectbox(
            "Time Frame",
            ["current", "recent", "comprehensive"],
            index=1,
            format_func=lambda opt: {
                "current": "Current (≤6 months)",
                "recent": "Recent (≤2 years)",
                "comprehensive": "All time",
            }[opt],
        )
        report_format = st.selectbox(
            "Report Format",
            ["executive", "detailed", "academic", "presentation"],
            index=1,
            format_func=lambda opt: {
                "executive": "Executive Summary",
                "detailed": "Detailed Analysis",
                "academic": "Academic Style",
                "presentation": "Presentation Format",
            }[opt],
        )
    if st.button("Start Deep Research", use_container_width=True):
        if not st.session_state.model_loaded:
            st.error(
                "Model not loaded. Check GPU availability and that the Hugging Face weights "
                "downloaded successfully."
            )
        elif not topic.strip():
            st.error("Please enter a research topic.")
        else:
            run_pipeline(topic, depth, focus, timeframe, report_format)
    if st.session_state.results:
        display_results(st.session_state.results)


def run_pipeline(topic: str, depth: str, focus: str, timeframe: str, report_format: str) -> None:
    assistant: DeepResearchAssistant = st.session_state.assistant
    bar = st.progress(0)
    status = st.empty()
    try:
        status.text("Generating research queries...")
        queries = assistant.generate_search_queries(topic, focus, depth)
        bar.progress(20)
        status.text("Drafting scout briefs with Jan-v1...")
        sources = assistant.build_internal_sources(topic, queries, focus, timeframe)
        bar.progress(65)
        status.text("Synthesizing report...")
        report = assistant.synthesize_research(topic, sources, focus, report_format)
        bar.progress(100)
        st.session_state.results = {
            "topic": topic,
            "report": report,
            "sources": sources,
            "queries": queries,
            "config": {
                "depth": depth,
                "focus": focus,
                "timeframe": timeframe,
                "format": report_format,
            },
            "timestamp": datetime.now().isoformat(),
        }
        time.sleep(0.3)
    except Exception as exc:  # pragma: no cover - runtime UX guard
        st.error(f"Research failed: {exc}")
    finally:
        status.empty()
        bar.empty()


def display_results(results: dict) -> None:
    st.header("Research Report")
    st.subheader(f"Topic: {results['topic']}")
    st.markdown(
        f"<div style='background:#f8f9ff;padding:1rem;border-radius:10px;"
        f"border:1px solid #e1e8ed;color:#0f172a;line-height:1.6;'>{results['report']}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Sources", expanded=False):
        for idx, source in enumerate(results.get("sources", [])[:12]):
            st.markdown(
                f"""
                <div style="background:#fff;padding:0.75rem;border-radius:8px;
                            border:1px solid #e1e8ed;margin:0.4rem 0;color:#0f172a;">
                    <h4 style="margin:0 0 .25rem 0;color:#111827;">{source.get('title','')}</h4>
                    <p style="margin:0 0 .25rem 0;color:#0f172a;">{source.get('snippet','')}</p>
                    <small><a style="color:#2563eb;" href="{source.get('url','')}" target="_blank">{source.get('url','')}</a></small>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("### Export")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        payload = f"Research Report: {results['topic']}\n\n{results['report']}"
        st.download_button(
            "Download Text",
            data=payload,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
    with col_b:
        json_blob = json.dumps(results, indent=2)
        st.download_button(
            "Download JSON",
            data=json_blob,
            file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with col_c:
        if st.button("Start New Research"):
            st.session_state.results = None
            st.experimental_rerun()


if __name__ == "__main__":
    main()
