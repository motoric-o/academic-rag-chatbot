# Academic RAG Chatbot (`regqa`)

**Model Scaling in Grounded Regulation QA**: comparing local quantized Qwen3 models against hosted open-weight models on a fixed RAG pipeline over a single translated campus regulation PDF.

See [docs/project_spec.md](docs/project_spec.md) for the full specification, ownership boundaries, and cross-scope contracts.

## Quick Start

### Docker (Recommended — Live Reload)
Works out of the box with hot reload for both FastAPI (`:8000`) and Vite (`:5173`):

```bash
# All members (CPU-safe backend + frontend)
docker compose up --build

# Stop all containers
docker compose --profile gpu down
```

- Frontend: http://localhost:5173
- Backend health check: http://localhost:8000/api/health

```bash
# GPU worker (optional — for anyone with an NVIDIA GPU running local models)
# Can be run on its own without starting backend/frontend:
docker compose --profile gpu up -d --build gpu

# Or start backend + frontend + GPU worker all together:
docker compose --profile gpu up --build

# Open a shell inside the GPU container
docker compose --profile gpu exec gpu bash

# Verify CUDA + bitsandbytes inside the GPU container
docker compose --profile gpu exec gpu uv run python -c "import torch, bitsandbytes as bnb; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'bnb:', bnb.__version__)"
```

### Local Setup (Without Docker)

#### Python (3.11 via `uv`)
```bash
# Core CPU environment (works for all members)
uv sync --extra dev

# Member A (local GPU inference with RTX 3070 / CUDA)
uv sync --extra dev --extra local
```

### Frontend (React + TypeScript + Vite + Tailwind + Vitest)
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables
Copy `.env.example` to `.env` and fill in any keys required for your role. Never commit `.env`.

---

## Team Sections (each owner maintains their own section)

### Member A — Local Models, Hardware, Compare & Telemetry
- **Scope:** `src/regqa/models/local_hf.py`, `src/regqa/evaluation/efficiency.py`, `src/regqa/analysis/tokenizer_stats.py`, `src/regqa/analysis/scaling_fit.py`, `configs/models/local_*`, `frontend/src/features/compare/`, `frontend/src/features/telemetry/`
- **Notes:** RTX 3070 (8 GB VRAM) is the reference machine for local latency/VRAM reporting. Verify CUDA + `bitsandbytes` in Week 1; Qwen3-0.6B 4-bit smoke test by end of Week 2.
- **Commands:**
  - CPU-safe unit tests: `uv run pytest tests/local/ -v`
  - GPU environment check (Docker GPU worker): `docker compose --profile gpu exec gpu uv run python -m regqa.models.local_hf --check-env`
  - Single-prompt smoke test (GPU worker): `docker compose --profile gpu exec gpu uv run python -m regqa.models.local_hf --config configs/models/local_qwen3_0.6b_4bit.yaml --smoke`
  - 20-call smoke suite + efficiency summary (GPU worker): `docker compose --profile gpu exec gpu uv run python -m regqa.models.local_hf --config configs/models/local_qwen3_0.6b_4bit.yaml --smoke-suite --repeats 2`
  - Efficiency summary CLI: `uv run python -m regqa.evaluation.efficiency <predictions.jsonl>`
  - Experiment A1 tokenizer stats (CPU-only): `uv run python -m regqa.analysis.tokenizer_stats --id-corpus .tmp/corpus.id.json --tokenizers Qwen/Qwen3-0.6B`
  - Scaling fit CLI (CPU-only): `uv run python -m regqa.analysis.scaling_fit --input <metrics.json> --metric strict_correctness`
  - Frontend tests (Compare & Telemetry): `cd frontend && npm test`

### Member B — Corpus, Retrieval & Deterministic Metrics
- **Scope:** `src/regqa/ingest/`, `src/regqa/translate/`, `src/regqa/retrieval/`, `configs/retrieval.yaml`, `data/`, deterministic evaluation modules, `scripts/make_tables.py`
- **Notes:** _TBD by Member B._

### Member C — Hosted Models, Pipeline & Backend API
- **Scope:** `src/regqa/models/base.py`, `hosted_openai_compat.py`, `rate_limiter.py`, `src/regqa/generation/`, `src/regqa/pipeline.py`, `src/regqa/api/`, `scripts/run_experiment.py`, shared configs & utils
- **Notes:** _TBD by Member C._

### Member D — Frontend & Model-Based Evaluation
- **Scope:** `frontend/` (except `compare/` and `telemetry/`), `faithfulness.py`, `judge.py`, `error_taxonomy.py`, `scripts/evaluate.py`, `results/`
- **Notes:** _TBD by Member D._
