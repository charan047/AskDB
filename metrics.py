from __future__ import annotations

import os
from typing import Dict

from prometheus_client import Counter, Histogram

# Labels kept small to avoid cardinality explosions
API_REQUESTS_TOTAL = Counter(
    "askdb_api_requests_total",
    "Total API requests",
    ["path", "method", "status"],
)

API_LATENCY_SECONDS = Histogram(
    "askdb_api_latency_seconds",
    "API request latency (seconds)",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
)

ASKDB_DB_LATENCY_SECONDS = Histogram(
    "askdb_db_latency_seconds",
    "Database execution latency (seconds)",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5),
)

ASKDB_LLM_LATENCY_SECONDS = Histogram(
    "askdb_llm_latency_seconds",
    "LLM generation latency (seconds)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
)

CACHE_HITS_TOTAL = Counter(
    "askdb_cache_hits_total",
    "Cache hits",
    ["layer"],  # api_resp | sql_exec
)

JOBS_ENQUEUED_TOTAL = Counter(
    "askdb_jobs_enqueued_total",
    "Background jobs enqueued",
    ["mode"],
)

JOBS_COMPLETED_TOTAL = Counter(
    "askdb_jobs_completed_total",
    "Background jobs completed",
    ["status"],  # ok | failed
)

JOB_LATENCY_SECONDS = Histogram(
    "askdb_job_latency_seconds",
    "Background job total runtime (seconds)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34),
)


def env_flags() -> Dict[str, str]:
    return {
        "redis_enabled": "1" if bool(os.getenv("REDIS_URL")) else "0",
        "byodb_enabled": os.getenv("BYODB_ENABLED", "1"),
        "sandbox_enabled": os.getenv("SANDBOX_ENABLED", "1"),
    }
