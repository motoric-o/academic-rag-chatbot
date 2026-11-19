# Member A GPU worker image (RTX 3070 / CUDA + bitsandbytes + accelerate)
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# Minimal system libs occasionally needed by bitsandbytes / torch extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --extra dev --extra local

CMD ["sleep", "infinity"]
