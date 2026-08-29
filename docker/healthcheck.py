"""Container healthcheck that verifies the API process and dependencies."""
import json
import sys
from urllib.request import urlopen

if __name__ == "__main__":
    try:
        with urlopen("http://127.0.0.1:8000/ready", timeout=2) as response:
            payload = json.load(response)
        sys.exit(0 if payload.get("ready") else 1)
    except Exception:
        sys.exit(1)
