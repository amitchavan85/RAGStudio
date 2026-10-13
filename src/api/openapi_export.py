"""Write the API's OpenAPI spec to web/openapi.json (input for the web app's generated types).

    python -m src.api.openapi_export      # then, in web/: npx openapi-typescript openapi.json -o src/api/schema.d.ts
"""
import json

from src.api.app import _project_root, create_app

if __name__ == "__main__":
    out = _project_root() / "web" / "openapi.json"
    out.write_text(json.dumps(create_app().openapi(), indent=1), encoding="utf-8")
    print(f"wrote {out}")
