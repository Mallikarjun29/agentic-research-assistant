# Jan-v1 Deep Research Assistant

Run Jan-v1 locally as a Streamlit application that automates deep research workflows without relying on third-party APIs. The stack pulls `janhq/Jan-v1-4B` weights from Hugging Face, generates its own scout notes, and synthesizes polished business-ready reports that you can export or iterate on entirely offline.

<img width="1918" height="951" alt="image" src="https://github.com/user-attachments/assets/5b1b2305-a7bc-4d8e-9a14-73a03753d1db" />


## Feature Highlights
- **Offline-first pipeline** – once the Hugging Face weights are cached, the assistant drafts queries, scout briefs, and final reports without touching the public internet.
- **Agentic reasoning** – Jan-v1 (Lucy + Qwen3-4B-thinking) drives query planning, intermediate note-taking, and sectioned report writing using enforced `<final>...</final>` scaffolding.
- **GPU-accelerated inference** – optimized for Tesla T4, L4, or A10 GPUs via Transformers + PyTorch FP16, with automatic device mapping for larger accelerators.
- **Configurable research knobs** – select topic, focus area (general/academic/business/technical/policy), research depth, timeframe, and target report format (executive/detailed/academic/presentation).
- **Progress-aware UI** – Streamlit surface shows query generation, scout drafting, synthesis progress, previous runs, and export options (TXT + JSON).
- **Structured exports** – downloadable artifacts include the final narrative, underlying scout notes, queries, and run metadata for downstream processing or compliance logging.

## Repository Structure
```
├── app.py                     # Streamlit UI, session management, progress UX, exports
├── jan_assistant/
│   ├── __init__.py            # Package surface
│   ├── assistant.py           # DeepResearchAssistant (HF model load, scout + synthesis logic)
│   ├── config.py              # ResearchConfig dataclass (model id, sampling params, device map)
│   └── utils.py               # Section templates and <final> block extraction helpers
├── requirements.txt           # Streamlit + Transformers + Accelerate stack
├── .env                       # Local secrets (e.g., HF_TOKEN)
└── README.md                  # You are here
```

## Prerequisites
- Linux with Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (Tesla T4/A10 or better) and CUDA 12 compatible drivers
- Enough disk bandwidth to download ~8 GB of Jan-v1 weights the first time you run
- (Optional) `HF_TOKEN` for faster authenticated pulls from Hugging Face

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# (optional) provide HF auth for higher rate limits
cp .env.example .env  # or hand-edit .env
# edit HF_TOKEN=your_token inside .env

streamlit run app.py
```
The first launch will stream `janhq/Jan-v1-4B` into your Hugging Face cache. Subsequent runs re-use the local weights, enabling fully offline operation.

## Configuration
`jan_assistant/config.py` exposes the core knobs:
- `model_id`: Hugging Face identifier (default `janhq/Jan-v1-4B`).
- `hf_auth_token`: read from `HF_TOKEN` environment variable for gated models or higher download quotas.
- `device_map`: set to `"auto"` for multi-GPU placement or override with `"cuda"` for single devices.
- `torch_dtype`: `float16` by default for efficient inference on T4/A10 cards.
- `max_tokens`, `temperature`, `top_p`, `top_k`: generation controls for both scout briefs and final reports.

Add any environment variables to `.env` (gitignored) and export them before launching Streamlit:
```bash
export $(grep -v '^#' .env | xargs)
```

## Using the App
1. Open the Streamlit URL printed in the terminal (e.g., `http://localhost:8501`).
2. Enter a research topic and choose depth/focus/timeframe/report format.
3. Click **Start Deep Research**. The UI will display:
   - Query generation progress.
   - Scout brief drafting (LLM-generated summaries per query instead of remote web calls).
   - Report synthesis phase.
4. Review the rendered report (dark/light theme aware) plus the auto-generated source cards.
5. Export the text summary or the full JSON payload containing queries, scout notes, and metadata.

## How It Works (Under the Hood)
1. **Planning** – deterministic query templates tailored to depth/focus produce a coverage plan.
2. **Scouting** – for each query, Jan-v1 is prompted to produce a short "Title + Summary" memo, simulating research notes without external APIs.
3. **Synthesis** – the memos are concatenated as "Research Notes" and fed into a strict prompt that enforces the requested section ordering, tone, and `<final>` tags.
4. **Post-processing** – regex sanitization keeps the output clean (no markdown bullets, no stray preambles) before displaying it in the Streamlit UI.
5. **Exports** – the full run (topic, config, queries, scout notes, report) is serializable for audits or follow-up tasks.

## Troubleshooting & Tips
- **Model not loaded** – verify `python -c "import torch; print(torch.cuda.is_available())"` returns `True`, and check that the CUDA-enabled PyTorch wheel matches your driver version.
- **CUDA OOM** – lower `max_tokens`, reduce research depth, or switch to a larger GPU. FP16 already minimizes memory footprint, but T4/L4 still max out around 4K context.
- **Slow first run** – download the model ahead of time: `huggingface-cli download janhq/Jan-v1-4B --local-dir ./models/jan-v1-4b`. Point `HF_HOME` to fast storage if needed.
- **Environment vars not loading** – remember to `source .venv/bin/activate` and export `.env` variables before launching Streamlit in new shells.

## Extending the Assistant
- Plug in enterprise retrieval (vector DBs, PDF loaders) by enriching the scout notes before synthesis.
- Add run history or sharing endpoints in `app.py` to coordinate multi-analyst workflows.
- Package `DeepResearchAssistant` into a CLI or FastAPI service for integration with automation pipelines.

Enjoy running Jan-v1 locally and crafting high-quality research briefs without leaving your GPU box!# Jan-v1 Deep Research Assistant

Build a local, privacy-preserving research workflow that combines Jan-v1's agentic reasoning with a Streamlit front-end and Hugging Face Transformers inference tuned for Tesla T4–A10 GPUs.

## What You Get
- **Agent-ready reasoning**: Jan-v1 (Lucy + Qwen3-4B-thinking) orchestrates query planning, scout-note generation, and sectioned reporting.
- **GPU-accelerated deployment**: Load the `janhq/Jan-v1-4B` transformer weights directly from Hugging Face with FP16 on NVIDIA T4/A10 class GPUs.
- **Self-contained research loop**: Instead of Serper, the model drafts its own scout briefs per query and then synthesizes a final report from those notes.
- **Streamlit demo app**: Configure topic, depth, focus, timeframe, and report format with real-time progress bars and export buttons.

## Repository Layout
```
app.py                      # Streamlit UI and user workflow
jan_assistant/
    __init__.py             # Package export surface
    assistant.py            # DeepResearchAssistant orchestrator
    config.py               # ResearchConfig dataclass
    utils.py                # Formatting helpers
requirements.txt            # Python dependencies
```

## Prerequisites
- Python 3.10+
- NVIDIA GPU with ≥16 GB VRAM (Tesla T4, L4, A10, etc.) and CUDA 12 capable drivers
- Sufficient disk bandwidth to stream ~8 GB of model weights from Hugging Face
- Optional but recommended: `HF_TOKEN` environment variable for higher-rate downloads

## 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
# Install a CUDA build of PyTorch first (example shown for CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. (Optional) Pre-download the model from Hugging Face
The first run will automatically download `janhq/Jan-v1-4B`. To prefetch or use offline clusters:
```bash
huggingface-cli download janhq/Jan-v1-4B --local-dir ./models/jan-v1-4b
```
Set `HF_HOME` if you want to control the cache location.

## 3. Run the Streamlit Demo
```bash
streamlit run app.py
```
Open the provided URL and:
1. Enter a research topic (e.g., *Impact of open-source LLM stacks on biotech*).
2. Choose depth, focus (general, academic, business, technical, policy), timeframe, and desired report format.
3. Click **Start Deep Research** to trigger query generation, LLM scout briefs, and Jan-v1 synthesis.
4. Review the structured report, inspect the autogenerated sources, and export in TXT or JSON.

## How It Works
1. **Configuration** (`jan_assistant/config.py`): Holds the Hugging Face model ID, HF token, sampling params, and device preferences (e.g., `device_map="auto"`, FP16).
2. **Assistant Orchestrator** (`jan_assistant/assistant.py`):
  - Loads `janhq/Jan-v1-4B` via `AutoModelForCausalLM.from_pretrained` using FP16 for T4/A10 GPUs.
  - Generates deterministic search queries based on topic depth and focus.
  - Uses the model itself to draft scout briefs for each query, simulating a research crawl without Serper.
  - Synthesizes the final report from those notes, enforcing `<final>...</final>` tags and strict section ordering.
3. **Utilities** (`jan_assistant/utils.py`): Provide report section templates and regex-based `<final>` extraction/cleanup.
4. **UI Layer** (`app.py`): Streamlit manages session-scoped model objects, exposes controls, tracks progress, and offers export buttons.

## Troubleshooting
- **`Model not loaded.`** Confirm PyTorch sees your GPU (`python -c "import torch; print(torch.cuda.is_available())"`) and that you installed the CUDA wheel. Set `HF_TOKEN` if downloads fail.
- **OOM errors**: Reduce `max_tokens` in `ResearchConfig`, set `device_map="auto"`, or run on a larger GPU (A10, L40, etc.).
- **Slow first run**: Hugging Face caches weights under `~/.cache/huggingface`; relocate via `HF_HOME` if needed.

## Next Steps
- Plug in additional tools (e.g., vector retrieval, PDF parsing) inside `DeepResearchAssistant`.
- Extend the Streamlit UI with run histories or multi-report queues.
- Package the assistant as a CLI for batch research jobs.

Enjoy exploring Jan-v1's agentic reasoning on your GPU!
