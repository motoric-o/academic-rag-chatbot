"""FastAPI application entrypoint and CORS setup.

Owner: Member C (spec §12.1).
Minimal bootstrap stub so `docker compose up` starts cleanly with hot reload.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Academic Regulation QA API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health check endpoint (spec §12.1)."""
    return {
        "status": "ok",
        "mode": os.getenv("REGQA_API_MODE", "replay"),
    }
