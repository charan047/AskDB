from __future__ import annotations

import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from uuid import uuid4

from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_community.chat_message_histories import ChatMessageHistory
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from infra import configure_logging, rdel, rget_json, rset_json, get_redis
from metrics import (
    API_REQUESTS_TOTAL,
    API_LATENCY_SECONDS,
    ASKDB_DB_LATENCY_SECONDS,
    ASKDB_LLM_LATENCY_SECONDS,
    CACHE_HITS_TOTAL,
    JOBS_ENQUEUED_TOTAL,
    JOBS_COMPLETED_TOTAL,
    JOB_LATENCY_SECONDS,
)
from rq import Queue
from rq.job import Job

from untitled0 import chain_code, get_schema_tables

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", "262144"))  # 256 KB
log = configure_logging()

# ------------------------------------------------------------
# CORS (restrict in production)
# ------------------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
CORS(app, resources={r"/*": {"origins": [o.strip() for o in origins]}})

# ------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------
storage_uri = os.getenv("RATE_LIMIT_STORAGE_URI") or os.getenv("REDIS_URL")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10 per minute"],
    storage_uri=storage_uri,   # ✅ makes Limiter use Redis
)


# ------------------------------------------------------------
# Feature flags / gating
# ------------------------------------------------------------
BYODB_ENABLED = os.getenv("BYODB_ENABLED", "1") == "1"
SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "1") == "1"
DEMO_KEY = os.getenv("DEMO_KEY", "").strip()  # if set, gates BYODB + sandbox

# Async job system (RQ + Redis)
ASYNC_ENABLED = os.getenv("ASYNC_ENABLED", "1") == "1"
ASYNC_DEFAULT = os.getenv("ASYNC_DEFAULT", "0") == "1"  # if true, /api enqueues by default
ASYNC_THRESHOLD_MS = int(os.getenv("ASYNC_THRESHOLD_MS", "6000"))  # if sync path exceeds this, suggest async
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "askdb")
JOB_RESULT_TTL_SECONDS = int(os.getenv("JOB_RESULT_TTL_SECONDS", "3600"))

CONN_TTL_SECONDS = int(os.getenv("CONN_TTL_SECONDS", "1800"))  # 30 min
MAX_DBURL_CHARS = int(os.getenv("MAX_DBURL_CHARS", "4000"))
MAX_SCHEMA_CHARS = int(os.getenv("MAX_SCHEMA_CHARS", "200000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))  # messages per session

# Caching
API_CACHE_TTL_SECONDS = int(os.getenv("API_CACHE_TTL_SECONDS", "180"))  # response cache for repeated prompts
API_CACHE_ENABLED = os.getenv("API_CACHE_ENABLED", "1") == "1"

# BYODB security controls
BYODB_ALLOWED_SCHEMES = set(
    s.strip().lower()
    for s in (os.getenv("BYODB_ALLOWED_SCHEMES", "postgresql,mysql,sqlite").split(","))
    if s.strip()
)

# ------------------------------------------------------------
# In-memory fallbacks (used if Redis is not configured)
# ------------------------------------------------------------
_histories: Dict[str, ChatMessageHistory] = {}

@dataclass
class SessionConn:
    db_url: str
    schema_csv: Optional[str]
    schema_source: str
    dialect: str
    host: str
    tables: list
    expires_at: float

_connections: Dict[str, SessionConn] = {}


def _now() -> float:
    return time.time()


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]


def _normalize_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def _safe_error(msg: str) -> str:
    """Best-effort redaction for user-facing errors."""
    msg = msg or "Request failed"
    msg = re.sub(r"(postgresql\+psycopg2://)([^:@\s]+):([^@\s]+)@", r"\1***:***@", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(mysql\+pymysql://)([^:@\s]+):([^@\s]+)@", r"\1***:***@", msg, flags=re.IGNORECASE)
    msg = re.sub(r"(sqlite:///)[^\s]+", r"\1***", msg, flags=re.IGNORECASE)
    msg = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "AIza***REDACTED***", msg)
    msg = re.sub(r"lsv2_[0-9A-Za-z\-_]{10,}", "lsv2_***REDACTED***", msg)
    return msg


# ------------------------------------------------------------
# Request lifecycle: request id + security headers + structured logs + metrics
# ------------------------------------------------------------
@app.before_request
def _before_request():
    g.request_id = (request.headers.get("X-Request-ID") or str(uuid4())).strip()
    g.start_time = time.time()


@app.after_request
def _after_request(resp):
    # Security headers
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Metrics
    try:
        latency_s = max(0.0, time.time() - getattr(g, "start_time", time.time()))
        API_REQUESTS_TOTAL.labels(path=request.path, method=request.method, status=str(resp.status_code)).inc()
        API_LATENCY_SECONDS.labels(path=request.path).observe(latency_s)
    except Exception:
        pass

    # Structured log line
    try:
        latency_ms = int((time.time() - getattr(g, "start_time", time.time())) * 1000)
        log.info(
            "request",
            extra={
                "request_id": getattr(g, "request_id", ""),
                "path": request.path,
                "method": request.method,
                "status": resp.status_code,
                "latency_ms": latency_ms,
            },
        )
    except Exception:
        pass
    return resp


# ------------------------------------------------------------
# Demo key gating
# ------------------------------------------------------------
def _require_demo_key() -> Optional[Any]:
    if not DEMO_KEY:
        return None
    incoming = (request.headers.get("X-DEMO-KEY") or "").strip()
    if incoming != DEMO_KEY:
        return jsonify({"error": "DEMO_KEY required"}), 403
    return None


# ------------------------------------------------------------
# Session History store (Redis-backed if REDIS_URL set)
# ------------------------------------------------------------
def _history_key(session_id: str) -> str:
    return f"hist:{session_id}"


def _history_load(session_id: str) -> ChatMessageHistory:
    obj = rget_json(_history_key(session_id))
    if isinstance(obj, list):
        h = ChatMessageHistory()
        for m in obj[-MAX_HISTORY_MESSAGES:]:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "user":
                h.add_user_message(content)
            elif role == "assistant":
                h.add_ai_message(content)
        return h

    if session_id not in _histories:
        _histories[session_id] = ChatMessageHistory()
    return _histories[session_id]


def _history_save(session_id: str, h: ChatMessageHistory) -> None:
    msgs = []
    for msg in h.messages[-MAX_HISTORY_MESSAGES:]:
        role = "user" if msg.type == "user" else "assistant"
        msgs.append({"role": role, "content": msg.content})

    if get_redis():
        rset_json(_history_key(session_id), msgs, ttl_s=int(os.getenv("HISTORY_TTL_SECONDS", "86400")))
    else:
        _histories[session_id] = h


def _history_reset(session_id: str) -> ChatMessageHistory:
    rdel(_history_key(session_id))
    _histories[session_id] = ChatMessageHistory()
    return _histories[session_id]


# ------------------------------------------------------------
# BYODB connection registry (Redis-backed if available)
# ------------------------------------------------------------
def _conn_key(session_id: str) -> str:
    return f"conn:{session_id}"


def _cleanup_connections():
    now = _now()
    for sid, c in list(_connections.items()):
        if c.expires_at <= now:
            _connections.pop(sid, None)


def _get_connection(session_id: str) -> Optional[SessionConn]:
    obj = rget_json(_conn_key(session_id))
    if isinstance(obj, dict) and obj.get("db_url"):
        return SessionConn(
            db_url=obj["db_url"],
            schema_csv=obj.get("schema_csv"),
            schema_source=obj.get("schema_source", "auto"),
            dialect=obj.get("dialect", ""),
            host=obj.get("host", ""),
            tables=obj.get("tables", []) or [],
            expires_at=float(obj.get("expires_at", _now() + CONN_TTL_SECONDS)),
        )

    _cleanup_connections()
    return _connections.get(session_id)


def _set_connection(session_id: str, c: SessionConn) -> None:
    obj = {
        "db_url": c.db_url,
        "schema_csv": c.schema_csv,
        "schema_source": c.schema_source,
        "dialect": c.dialect,
        "host": c.host,
        "tables": c.tables,
        "expires_at": c.expires_at,
    }
    if get_redis():
        rset_json(_conn_key(session_id), obj, ttl_s=CONN_TTL_SECONDS)
    else:
        _connections[session_id] = c


def _del_connection(session_id: str) -> None:
    rdel(_conn_key(session_id))
    _connections.pop(session_id, None)


# ------------------------------------------------------------
# Helpers: demo schema + examples
# ------------------------------------------------------------
def _load_demo_schema_from_csv() -> list:
    csv_path = os.getenv("TABLE_DESCRIPTIONS_PATH", "./database_table_descriptions.csv")
    tables = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tables.append(
                    {
                        "table_name": (row.get("table_name") or "").strip(),
                        "description": (row.get("description") or "").strip(),
                    }
                )
    except Exception:
        pass
    return tables


def get_examples() -> list:
    return [
        {
            "category": "Public (Analytics)",
            "mode": "public",
            "items": [
                "Top 10 customers by total payments and their share of total revenue",
                "Total sales by product line (revenue = quantityordered * priceeach)",
                "For the last 90 days, show monthly revenue and month-over-month % change",
                "Which products are frequently ordered together? Top 10 product pairs by co-occurrence",
            ],
        },
        {
            "category": "Sandbox (What-if, rolled back)",
            "mode": "sandbox",
            "items": [
                "In sandbox mode, increase the credit limit by 15% for customers in France under 30000",
                "In sandbox mode, mark all 'In Process' orders for customer 112 as 'Cancelled'",
                "In sandbox mode, delete order 10100 safely (delete orderdetails first, then the order)",
                "In sandbox mode, delete customer 112 safely by removing dependent rows first",
            ],
        },
    ]


# ------------------------------------------------------------
# Async helpers (RQ)
# ------------------------------------------------------------
def _get_queue() -> Optional[Queue]:
    r = get_redis()
    if not r:
        return None
    return Queue(RQ_QUEUE_NAME, connection=r, default_timeout=int(os.getenv("RQ_JOB_TIMEOUT_SECONDS", "120")))


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _enqueue_job(*, question: str, session_id: str, mode: str, include_sql: bool, db_url_override: Optional[str], schema_csv_text: Optional[str]) -> str:
    q = _get_queue()
    if not q:
        raise RuntimeError("Redis is required for async jobs. Configure REDIS_URL.")

    from worker import run_askdb_job  # local import to keep web lean

    job = q.enqueue(
        run_askdb_job,
        question=question,
        session_id=session_id,
        mode=mode,
        include_sql=include_sql,
        db_url_override=db_url_override,
        schema_csv_text=schema_csv_text,
        result_ttl=JOB_RESULT_TTL_SECONDS,
        ttl=JOB_RESULT_TTL_SECONDS,
        failure_ttl=JOB_RESULT_TTL_SECONDS,
    )
    JOBS_ENQUEUED_TOTAL.labels(mode=mode).inc()
    return job.id


def _job_status(job: Job) -> str:
    if job.is_finished:
        return "finished"
    if job.is_failed:
        return "failed"
    if job.is_started:
        return "running"
    return "queued"


# ------------------------------------------------------------
# API Discovery endpoints
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
@limiter.exempt
def home():
    return jsonify(
        {
            "name": "AskDB",
            "status": "running",
            "usage": {"POST": "/api", "body": {"question": "Top 10 customers by total payments", "session_id": "demo", "mode": "public"}},
            "modes": {"public": "SELECT-only analytics (safe).", "sandbox": "DML simulated and always rolled back."},
            "features": {
                "byodb": BYODB_ENABLED,
                "sandbox": SANDBOX_ENABLED,
                "demo_key_gated": bool(DEMO_KEY),
                "redis_enabled": bool(get_redis()),
                "async_enabled": ASYNC_ENABLED and bool(get_redis()),
            },
        }
    )


@app.route("/health", methods=["GET"])
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics", methods=["GET"])
@limiter.exempt
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/about", methods=["GET"])
@limiter.exempt
def about():
    return jsonify(
        {
            "app": "AskDB",
            "positioning": "Sales Ops / CRM Analytics Copilot + Safe Sandbox Playground",
            "dataset": {
                "name": "ClassicModels-style CRM demo (Postgres)",
                "entities": ["customers", "orders", "orderdetails", "payments", "products", "employees", "offices", "productlines"],
            },
            "modes": {"public": "SELECT-only analytics (safe).", "sandbox": "DML simulated and always rolled back."},
        }
    )


@app.route("/examples", methods=["GET"])
@limiter.exempt
def examples():
    return jsonify({"examples": get_examples()})


@app.route("/connection", methods=["GET"])
@limiter.exempt
def connection_status():
    _cleanup_connections()
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"connected": False, "host": "", "dialect": "", "schema_source": ""})

    c = _get_connection(session_id)
    if not c:
        return jsonify({"connected": False, "host": "", "dialect": "", "schema_source": ""})

    return jsonify({"connected": True, "host": c.host, "dialect": c.dialect, "schema_source": c.schema_source})


@app.route("/schema", methods=["GET"])
@limiter.exempt
def schema():
    session_id = (request.args.get("session_id") or "").strip()
    c = _get_connection(session_id) if session_id else None
    tables = c.tables if c else _load_demo_schema_from_csv()
    return jsonify({"tables": tables})


# ------------------------------------------------------------
# BYODB connect/disconnect
# ------------------------------------------------------------
def _validate_db_url(db_url: str) -> None:
    if not db_url:
        return
    if len(db_url) > MAX_DBURL_CHARS:
        raise ValueError("db_url too long")
    parsed = urlparse(db_url)
    scheme = (parsed.scheme or "").lower()
    root = scheme.split("+", 1)[0]
    if root and root not in BYODB_ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported DB scheme: {root}. Allowed: {', '.join(sorted(BYODB_ALLOWED_SCHEMES))}")


@app.route("/connect", methods=["POST"])
@limiter.limit("6 per minute; 60 per day")
def connect():
    try:
        _cleanup_connections()
        if not BYODB_ENABLED:
            return jsonify({"error": "BYODB disabled"}), 403

        # if DEMO_KEY:
        #     blocked = _require_demo_key()
        #     if blocked:
        #         return blocked

        data = request.get_json() or {}
        session_id = (data.get("session_id") or "").strip()
        db_url = (data.get("db_url") or "").strip()
        schema_csv = (data.get("schema_csv") or "").strip()

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        if not db_url:
            _del_connection(session_id)
            return jsonify({"connected": False, "host": "", "dialect": "", "schema_source": ""})

        _validate_db_url(db_url)
        if schema_csv and len(schema_csv) > MAX_SCHEMA_CHARS:
            return jsonify({"error": "schema_csv too large"}), 400

        tables, meta = get_schema_tables(db_url, schema_csv_text=schema_csv or None)

        expires_at = _now() + CONN_TTL_SECONDS
        c = SessionConn(
            db_url=db_url,
            schema_csv=schema_csv or None,
            schema_source=meta.get("schema_source", "auto"),
            dialect=meta.get("dialect", ""),
            host=meta.get("host", ""),
            tables=tables,
            expires_at=expires_at,
        )
        _set_connection(session_id, c)

        return jsonify({"connected": True, "host": c.host, "dialect": c.dialect, "schema_source": c.schema_source, "tables": c.tables})

    except Exception as e:
        log.exception("connect_failed", extra={"request_id": getattr(g, "request_id", "")})
        return jsonify({"error": _safe_error(str(e))}), 400


@app.route("/disconnect", methods=["POST"])
@limiter.limit("10 per minute")
def disconnect():
    try:
        _cleanup_connections()
        data = request.get_json() or {}
        session_id = (data.get("session_id") or "").strip()
        reset_session = bool(data.get("reset_session", True))
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        _del_connection(session_id)
        if reset_session:
            _history_reset(session_id)

        return jsonify({"connected": False, "host": "", "dialect": "", "schema_source": ""})
    except Exception as e:
        log.exception("disconnect_failed", extra={"request_id": getattr(g, "request_id", "")})
        return jsonify({"error": _safe_error(str(e))}), 400


# ------------------------------------------------------------
# Jobs (polling)
# ------------------------------------------------------------
@app.route("/job/<job_id>", methods=["GET"])
@limiter.exempt
def job_get(job_id: str):
    try:
        q = _get_queue()
        if not q:
            return jsonify({"error": "Redis not configured"}), 400

        job = Job.fetch(job_id, connection=q.connection)
        status = _job_status(job)

        if status == "finished":
            JOBS_COMPLETED_TOTAL.labels(status="ok").inc()
            return jsonify({"status": status, "result": job.result})
        if status == "failed":
            JOBS_COMPLETED_TOTAL.labels(status="failed").inc()
            return jsonify({"status": status, "error": _safe_error(str(job.exc_info or "Job failed"))}), 500

        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"error": _safe_error(str(e))}), 400


# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------
def _db_sig_for_session(conn: Optional[SessionConn]) -> str:
    return "demo" if not conn else _hash_text(conn.db_url)


def _resp_cache_key(db_sig: str, mode: str, q: str) -> str:
    return f"resp:{db_sig}:{mode}:{hashlib.sha256(_normalize_q(q).encode('utf-8')).hexdigest()}"


@app.route("/api", methods=["POST"])
@limiter.limit("6 per minute; 100 per day")
def api():
    try:
        _cleanup_connections()
        data = request.get_json() or {}

        q_text = (data.get("question") or "").strip()
        if not q_text:
            return jsonify({"error": "Missing 'question' in the request body"}), 400

        session_id = (data.get("session_id") or str(uuid4())).strip()
        reset_session = bool(data.get("reset_session", False))
        include_sql = bool(data.get("include_sql", False))
        mode = (data.get("mode") or "public").lower().strip()
        async_requested = bool(data.get("async", ASYNC_DEFAULT))

        if mode not in ("public", "sandbox"):
            return jsonify({"error": "Invalid mode. Use 'public' or 'sandbox'."}), 400

        if mode == "sandbox" and not SANDBOX_ENABLED:
            return jsonify({"error": "Sandbox disabled"}), 403

        if mode == "sandbox" and DEMO_KEY:
            blocked = _require_demo_key()
            if blocked:
                return blocked

        history = _history_load(session_id)
        if reset_session:
            history = _history_reset(session_id)

        conn = _get_connection(session_id)
        db_url_override = None
        schema_csv_override = None
        schema_source = "demo"
        dialect = "demo"
        host = ""
        if conn:
            # if DEMO_KEY:
            #     blocked = _require_demo_key()
            #     if blocked:
            #         return blocked
            db_url_override = conn.db_url
            schema_csv_override = conn.schema_csv
            schema_source = conn.schema_source
            dialect = conn.dialect
            host = conn.host

        # API response cache (public only)
        db_sig = _db_sig_for_session(conn)
        cache_key = _resp_cache_key(db_sig, mode, q_text)
        if API_CACHE_ENABLED and mode == "public" and API_CACHE_TTL_SECONDS > 0:
            cached = rget_json(cache_key)
            if isinstance(cached, dict) and cached.get("payload"):
                CACHE_HITS_TOTAL.labels(layer="api_resp").inc()
                history.add_user_message(q_text)
                history.add_ai_message(cached["payload"].get("answer", ""))
                _history_save(session_id, history)

                payload = cached["payload"]
                payload["session_id"] = session_id
                payload["cache_hit"] = True
                payload["connection"] = {"connected": bool(conn), "host": host, "dialect": dialect, "schema_source": schema_source}
                return jsonify(payload)

        # Async path: enqueue immediately when requested and supported
        if async_requested and ASYNC_ENABLED:
            if not get_redis():
                return jsonify({"error": "Async requires Redis. Configure REDIS_URL."}), 400

            job_id = _enqueue_job(
                question=q_text,
                session_id=session_id,
                mode=mode,
                include_sql=include_sql,
                db_url_override=db_url_override,
                schema_csv_text=schema_csv_override,
            )
            return jsonify({"status": "queued", "job_id": job_id, "session_id": session_id}), 202

        # Sync execution
        history.add_user_message(q_text)
        formatted_messages = [{"role": "user" if msg.type == "user" else "assistant", "content": msg.content} for msg in history.messages]

        t0 = time.time()
        res = chain_code(
            q_text,
            formatted_messages,
            mode=mode,
            db_url_override=db_url_override,
            schema_csv_text=schema_csv_override,
        )
        total_ms = int((time.time() - t0) * 1000)

        answer_text = res.get("answer", "")
        history.add_ai_message(answer_text)
        _history_save(session_id, history)

        db_ms = int(res.get("db_ms", 0) or 0)
        llm_ms = max(0, int(res.get("latency_ms", 0) or total_ms) - db_ms)
        try:
            ASKDB_DB_LATENCY_SECONDS.observe(db_ms / 1000.0)
            ASKDB_LLM_LATENCY_SECONDS.observe(llm_ms / 1000.0)
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "answer": answer_text,
            "session_id": session_id,
            "connection": {"connected": bool(conn), "host": host, "dialect": dialect, "schema_source": schema_source},
            "insights_summary": res.get("insights_summary", ""),
            "chart_spec": res.get("chart_spec", None),
            "latency_ms": int(res.get("latency_ms", 0) or total_ms),
            "db_ms": db_ms,
            "llm_ms": llm_ms,
            "optimized": bool(res.get("optimized", False)),
            "optimization_reason": res.get("optimization_reason", ""),
        }

        if include_sql:
            rows_preview = res.get("rows_preview", []) or []
            columns = res.get("columns", []) or (list(rows_preview[0].keys()) if rows_preview else [])
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
                    "dialect": res.get("dialect", dialect),
                    "schema_source": res.get("schema_source", schema_source),
                    "host": res.get("host", host),
                    "columns": columns,
                    "rows_preview": rows_preview,
                }
            )

        # Cache safe public responses
        if API_CACHE_ENABLED and mode == "public" and API_CACHE_TTL_SECONDS > 0:
            rset_json(cache_key, {"payload": dict(payload)}, ttl_s=API_CACHE_TTL_SECONDS)

        # If latency is high, hint async capability
        if ASYNC_ENABLED and get_redis() and int(payload.get("latency_ms", 0) or 0) >= ASYNC_THRESHOLD_MS:
            payload["async_hint"] = "This query was slow. You can set async:true to run it in the background and poll /job/<id>."

        return jsonify(payload)

    except Exception as e:
        log.exception("api_failed", extra={"request_id": getattr(g, "request_id", "")})
        return jsonify({"error": _safe_error(str(e))}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
