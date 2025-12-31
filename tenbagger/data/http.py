from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_get_json(url: str, *, retries: int = 6, timeout: int = 30) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="ignore")
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= 0:
                text = text[start : end + 1]
            return json.loads(text)
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep((2**attempt) * (0.15 + random.random() * 0.25))
    raise RuntimeError(f"failed GET {url}: {last_err}")


def http_get_bytes(url: str, *, retries: int = 6, timeout: int = 30) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError) as e:
            last_err = e
            time.sleep((2**attempt) * (0.15 + random.random() * 0.25))
    raise RuntimeError(f"failed GET {url}: {last_err}")

