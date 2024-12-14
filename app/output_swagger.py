import sys
import yaml

from main import app  # noqa: E402

def format_paths(paths: dict) -> dict:
    def format_response(responses: dict) -> dict:
        return {status_code if status_code != "422" else "400": contents for status_code, contents in responses.items()}

    results = {}
    for path, methods in paths.items():
        result = {}
        for method, items in methods.items():
            if items.get("responses"):
                result[method] = {**items, "responses": format_response(items["responses"])}
            else:
                result[method] = items
        results[path] = result
    return results

if __name__ == "__main__":
    api_json = app.openapi()
    formatted_api_json = {**api_json, "paths": format_paths(api_json["paths"])}
    with open("swagger.yaml", "w") as f:
        yaml.dump(formatted_api_json, f)