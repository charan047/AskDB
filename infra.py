from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional
from redis import Redis

try:
    from pythonjsonlogger import jsonlogger  # type: ignore
except Exception:  # pragma: no cover
    jsonlogger = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


@dataclass
class RedisConfig:
    url: str
    prefix: str = "askdb:"
    socket_timeout_s: float = 2.0
    socket_connect_timeout_s: float = 2.0


_redis_client = None


def get_redis() -> Optional[Redis]:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None

    connect_timeout = int(os.getenv("REDIS_CONNECT_TIMEOUT", "10"))

    try:
        r = redis.Redis.from_url(
            url,
            socket_connect_timeout=connect_timeout,
            socket_timeout=None,          # ✅ required for RQ blocking ops
            health_check_interval=30,
            retry_on_timeout=True,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception as e:
        # IMPORTANT: print/log the real reason instead of hiding it
        print(f"[redis] get_redis failed: {e}")
        return None



def redis_prefix() -> str:
    return (os.getenv("REDIS_PREFIX") or "askdb:").strip()


def rkey(key: str) -> str:
    return f"{redis_prefix()}{key}"


def rget_json(key: str) -> Optional[Any]:
    r = get_redis()
    if not r:
        return None
    try:
        raw = r.get(rkey(key))
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def rset_json(key: str, value: Any, ttl_s: int) -> bool:
    r = get_redis()
    if not r:
        return False
    try:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if ttl_s > 0:
            r.setex(rkey(key), ttl_s, raw)
        else:
            r.set(rkey(key), raw)
        return True
    except Exception:
        return False


def rdel(key: str) -> None:
    r = get_redis()
    if not r:
        return
    try:
        r.delete(rkey(key))
    except Exception:
        pass


def configure_logging() -> logging.Logger:
    """
    Configure a production-friendly logger.
    - JSON logs if python-json-logger available; else pretty text.
    - LOG_LEVEL env supported.
    """
    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logger = logging.getLogger("askdb")
    logger.setLevel(level)
    logger.propagate = False

    # If handlers already exist (e.g. reloader), don't double-add
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    if jsonlogger:
        fmt = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(path)s %(method)s %(status)s %(latency_ms)s",
        )
        handler.setFormatter(fmt)
    else:
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
