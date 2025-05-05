import json


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
    return None
