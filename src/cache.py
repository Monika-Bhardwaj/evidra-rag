from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


class DiskCache:
    def __init__(self, cache_dir: Path, ttl_seconds: Optional[int] = None) -> None:
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(name: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return f"{name}-{digest}.json"

    def get(self, name: str, payload: str) -> Optional[Any]:
        path = self.cache_dir / self._key(name, payload)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if self.ttl is not None:
                import time

                age = time.time() - data.get("_ts", 0)
                if age > self.ttl:
                    return None
            return data.get("value")
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, name: str, payload: str, value: Any) -> None:
        import time

        path = self.cache_dir / self._key(name, payload)
        data = {"_ts": time.time(), "value": value}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def clear(self, name: str) -> None:
        for path in self.cache_dir.glob(f"{name}-*.json"):
            path.unlink(missing_ok=True)