"""Export the OpenAPI document so the frontend types can be checked against the real API.

Writes ``frontend/src/api/openapi.json``. The TypeScript types in ``frontend/src/api/types.ts``
are maintained by hand; this file is the reference they are checked against.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import Settings
from backend.main import create_app

OUTPUT = Path("frontend/src/api/openapi.json")


def main() -> int:
    app = create_app(Settings(mode="local"))
    document = app.openapi()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths = len(document["paths"])
    schemas = len(document["components"]["schemas"])
    print(f"Wrote {OUTPUT} ({paths} paths, {schemas} schemas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
