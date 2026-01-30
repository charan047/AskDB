from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv
load_dotenv()
from rq import Worker, Queue, Connection
from redis import Redis

from infra import get_redis, configure_logging
from untitled0 import chain_code

log = configure_logging()

QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "askdb")


def run_askdb_job(
    *,
    question: str,
    session_id: str,
    mode: str,
    include_sql: bool,
    db_url_override: Optional[str] = None,
    schema_csv_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background job:
    - Executes chain_code
    - Returns payload compatible with /api responses
    """
    t0 = time.time()
    res = chain_code(
        question,
        messages=[],
        mode=mode,
        db_url_override=db_url_override,
        schema_csv_text=schema_csv_text,
    )
    total_ms = int((time.time() - t0) * 1000)

    payload: Dict[str, Any] = {
        "answer": res.get("answer", ""),
        "session_id": session_id,
        "latency_ms": int(res.get("latency_ms", 0) or total_ms),
        "db_ms": int(res.get("db_ms", 0) or 0),
        "optimized": bool(res.get("optimized", False)),
        "optimization_reason": res.get("optimization_reason", ""),
        "insights_summary": res.get("insights_summary", ""),
        "chart_spec": res.get("chart_spec", None),
    }

    if include_sql:
        payload.update(
            {
                "sql": res.get("sql", ""),
                "mode": res.get("mode", mode),
                "kind": res.get("kind", ""),
                "rolled_back": bool(res.get("rolled_back", False)),
                "rows_returned": int(res.get("rows_returned", 0) or 0),
                "rows_affected": int(res.get("rows_affected", 0) or 0),
                "sql_fix_retries_used": int(res.get("sql_fix_retries_used", 0) or 0),
                "explain": res.get("explain", None),
                "warnings": res.get("warnings", []),
                "dialect": res.get("dialect", ""),
                "schema_source": res.get("schema_source", ""),
                "host": res.get("host", ""),
                "columns": res.get("columns", []),
                "rows_preview": res.get("rows_preview", []),
            }
        )

    return payload


def main() -> None:
    r: Optional[Redis] = get_redis()
    if not r:
        raise RuntimeError("REDIS_URL is required to run the RQ worker.")

    with Connection(r):
        q = Queue(QUEUE_NAME)
        w = Worker([q])
        log.info("rq_worker_start", extra={"queue": QUEUE_NAME})
        w.work(with_scheduler=True)


if __name__ == "__main__":
    main()
