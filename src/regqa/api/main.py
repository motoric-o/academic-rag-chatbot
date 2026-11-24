"""FastAPI application entrypoint and CORS setup.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §12.1).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from regqa.api.routes import router

app = FastAPI(
    title="Academic Regulation QA API",
    description="Backend API for Grounded Academic Regulation QA across local and hosted models.",
    version="0.1.0",
)

# CORS setup for frontend development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
