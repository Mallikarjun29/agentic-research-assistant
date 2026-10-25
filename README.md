# Jan-v1 Deep Research Assistant

A local Streamlit app that plans research queries, retrieves real web evidence, and synthesizes a sectioned report with `janhq/Jan-v1-4B`. Scout briefs are grounded in search-result snippets (title, URL and the search engine's summary text) retrieved through a free DuckDuckGo search (no API key, account or paid service). Full pages are not fetched. If web search is unavailable, the app falls back to clearly labeled model-only notes; the retrieval mode used is shown in the UI and stored in each run's JSON export. Grounding constrains the briefs but is not a factuality guarantee: the model can still write unsupported claims, and snippets are not independently fact-checked. Review important claims before relying on them.

![Streamlit research assistant](https://github.com/user-attachments/assets/5b1b2305-a7bc-4d8e-9a14-73a03753d1db)

## Architecture

```text
Topic + depth / focus / timeframe / format (Streamlit app.py)
  -> deterministic query planning (generate_search_queries)
  -> web search per query via DuckDuckGo only (websearch.py: ddgs package,
       duckduckgo backend; no engine rotation, no retrying around blocks)
  -> Jan-v1 scout briefs grounded in the retrieved snippets (build_web_sources)
       - if the model is unavailable, briefs keep the raw retrieved snippets
         (labeled 'web-snippets', not presented as model-written)
       - if search returns nothing, fall back to model-only notes
         (build_internal_sources, labeled 'jan-v1-local', no URLs)
  -> Jan-v1 report synthesis from up to 20 briefs, with source URLs (synthesize_research)
  -> <final> extraction and section cleanup (utils.py)
  -> report display + TXT/JSON downloads (app.py)
```

`jan_assistant/assistant.py` loads the Hugging Face tokenizer and causal model, generates template-based queries (5, 8, 15, or 25 depending on depth), searches the web for up to `web_max_queries` of them, and prompts the model to write each brief strictly from the retrieved evidence. Every brief carries its real source URLs plus a `references` list in the JSON export. `jan_assistant/websearch.py` performs keyless search through the `ddgs` package pinned to DuckDuckGo's own backend. It does not rotate to other search engines or retry around rate limits: a block simply yields no results and the app falls back to labeled model-only notes. Timeframe maps to the search date filter where supported (current ~ past month, recent ~ past year). The Streamlit app keeps the loaded model and latest run in session state; it does not save a persistent run history.

## Setup

- Python 3.10+ and enough RAM/VRAM for the model. The default `ResearchConfig` uses `device_map="auto"` and float16; an NVIDIA GPU with a compatible PyTorch build is the practical route. CPU operation is not tested here.
- Network access for the initial Hugging Face model download and for live web search. Later offline use can rely on cached weights with the web-search checkbox turned off.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a PyTorch build appropriate for your machine first (see pytorch.org).
pip install -r requirements.txt
streamlit run app.py
```

Set `HF_TOKEN` in your shell if Hugging Face authentication or rate limits require it, for example `export HF_TOKEN=...`. No `.env.example` is checked in, and the code reads `os.getenv("HF_TOKEN")` directly: creating a `.env` file alone will not load it. Do not commit tokens. `jan_assistant/config.py` contains the model ID, generation settings, dtype, device map and web-search defaults (`use_web_search`, `web_results_per_query`, `web_max_queries`, `web_timeout`); edit those defaults if needed. The first launch downloads `janhq/Jan-v1-4B` into the Hugging Face cache.

Enter a topic, choose depth/focus/timeframe/report format, leave **Search the web for evidence** on, and click **Start Deep Research**. The app displays the briefs under "Sources" with their real URLs and a per-brief grounding label (`web+jan-v1-local`, `web-snippets`, or `jan-v1-local` for model-only notes), plus a caption stating which retrieval mode the run used. It offers TXT (report) and JSON (report, query list, briefs with references, retrieval mode and run metadata) downloads. Review the generated report before sharing it.

## Repository layout

- `app.py`: Streamlit UI, web-search toggle, progress, session state and exports.
- `jan_assistant/assistant.py`: model loading, query planning, web-grounded scout prompts and synthesis.
- `jan_assistant/websearch.py`: free DuckDuckGo retrieval (`ddgs` package, duckduckgo backend only; no engine rotation or block evasion).
- `jan_assistant/config.py`: `ResearchConfig` defaults, web-search settings and `HF_TOKEN` environment lookup.
- `jan_assistant/utils.py`: section templates and final-block extraction.
- `requirements.txt`: Python dependencies; install a hardware-compatible PyTorch build separately if required.

If model loading fails, inspect the console error, confirm the model ID/token and check `python -c "import torch; print(torch.cuda.is_available())"`. If memory is tight, reduce `max_tokens` in `ResearchConfig` or use a machine with more memory. If search returns nothing (rate limits or network blocks), the app falls back to model-only notes automatically.

## Verification status

Web retrieval, the grounding fallbacks and the module-level integration were exercised live (real queries returned real results with URLs; model-absent and no-result paths degrade as designed). The full end-to-end run with the Jan-v1 model loaded, including model-written grounded briefs and the Streamlit flow on a GPU machine, has not been executed in CI; run one real research pass after `streamlit run app.py` before demoing it.
