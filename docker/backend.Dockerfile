# CPU-safe backend & experiment image (works for all team members)
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install dependencies first for Docker layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "regqa.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src"]
