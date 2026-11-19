.PHONY: help corpus calibrate cache run eval report app test lint check-api up down logs gpu-up gpu-shell gpu-check

PYTHON ?= uv run python
EXP ?= e1_param_ladder

help:
	@echo "Available targets:"
	@echo "  make up              - Start backend + frontend in Docker with hot reload (CPU)"
	@echo "  make down            - Stop Docker Compose services"
	@echo "  make logs            - Follow Docker Compose logs"
	@echo "  make gpu-up          - Start Member A GPU worker container (RTX 3070)"
	@echo "  make gpu-shell       - Open an interactive shell inside the GPU container"
	@echo "  make gpu-check       - Verify CUDA + bitsandbytes inside the GPU container"
	@echo "  make corpus          - Parse, translate, verify, and freeze corpus (Member B)"
	@echo "  make calibrate       - Run dev-only chunk-size calibration (Member B)"
	@echo "  make cache           - Build frozen retrieval + oracle cache (Member B)"
	@echo "  make run EXP=<name>  - Run experiment config from configs/experiments/<name>.yaml"
	@echo "  make eval            - Compute metrics and statistics (Members B & D)"
	@echo "  make report          - Generate result tables and figures (Members A, B, D)"
	@echo "  make app             - Start backend API and frontend dev server"
	@echo "  make test            - Run Python and frontend unit tests"
	@echo "  make lint            - Run ruff and frontend ESLint"
	@echo "  make check-api       - Verify openapi.json and generated frontend types are in sync"

up:
	docker compose up --build

down:
	docker compose --profile gpu down

logs:
	docker compose logs -f

gpu-up:
	docker compose --profile gpu up -d --build gpu

gpu-shell:
	docker compose --profile gpu exec gpu bash

gpu-check:
	docker compose --profile gpu exec gpu uv run python -c "import torch, bitsandbytes as bnb; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'bnb:', bnb.__version__)"

corpus:
	$(PYTHON) scripts/build_corpus.py

calibrate:
	$(PYTHON) -m regqa.retrieval.calibrate_chunks --config configs/retrieval.yaml

cache:
	$(PYTHON) scripts/build_cache.py --config configs/retrieval.yaml

run:
	$(PYTHON) scripts/run_experiment.py --config configs/experiments/$(EXP).yaml

eval:
	$(PYTHON) scripts/evaluate.py

report:
	$(PYTHON) scripts/make_tables.py

app:
	@echo "Start backend:  uv run uvicorn regqa.api.main:app --reload"
	@echo "Start frontend: cd frontend && npm run dev"

test:
	uv run pytest
	cd frontend && npm test

lint:
	uv run ruff check src tests scripts
	cd frontend && npm run lint

check-api:
	@echo "TODO (Member C & D): export openapi.json and verify frontend/src/api/ is in sync."
