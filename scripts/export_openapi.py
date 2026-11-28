"""Export FastAPI OpenAPI schema to JSON.

Owner: Member C (Francisco Valentino; see docs/project_spec.md §12.1 & §12.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regqa.api.main import app


def export_openapi() -> None:
    schema = app.openapi()
    
    # Save to root and frontend/src/api/
    targets = [
        Path("openapi.json"),
        Path("frontend/src/api/openapi.json"),
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print(f"Exported OpenAPI schema to {target}")


if __name__ == "__main__":
    export_openapi()
