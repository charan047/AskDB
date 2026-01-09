# .\.venv\Scripts\Activate.ps1
from untitled0 import chain_code
from langchain_community.chat_message_histories import ChatMessageHistory
from flask import Flask, request, jsonify
from uuid import uuid4
from flask_cors import CORS
import time
import os
import csv
from dataclasses import dataclass

app = Flask(__name__)

# ----------------------------
# CORS
# ----------------------------
origins = os.getenv(
    "CORS_ORIGINS",
    "https://ask-db.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
).split(",")
CORS(app, resources={r"/*": {"origins": [o.strip() for o in origins]}})

# ----------------------------
# Rate limiting
# ----------------------------
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# ----------------------------
# In-memory session chat history
# ----------------------------
histories = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]

# ----------------------------
# V2 (Phase 1): BYODB connections (in-memory)
# ----------------------------
@dataclass
class SessionConn:
    db_url: str
    schema_csv: str
    expires_at: float

connections = {}
CONN_TTL_SECONDS = int(os.getenv("CONN_TTL_SECONDS", "1800"))  # 30 minutes default
MAX_SCHEMA_CHARS = int(os.getenv("MAX_SCHEMA_CHARS", "50000"))  # safety cap
MAX_DBURL_CHARS = int(os.getenv("MAX_DBURL_CHARS", "4000"))

def _now() -> float:
    return time.time()

def _cleanup_connections() -> None:
    t = _now()
    dead = [sid for sid, c in connections.items() if c.expires_at <= t]
    for sid in dead:
        del connections[sid]

def _is_allowed_db_url(db_url: str) -> bool:
    # MVP allow-list; expand later (e.g., block internal hosts)
    return db_url.startswith(("postgresql", "mysql", "sqlite"))

# ----------------------------
# Helper: load schema CSV for demo dataset (V1 default)
# ----------------------------
def load_schema_from_csv():
    csv_path = os.getenv("TABLE_DESCRIPTIONS_PATH", "./database_table_descriptions.csv")
    tables = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tables.append(
                {
                    "table_name": (row.get("table_name", "") or "").strip(),
                    "description": (row.get("description", "") or "").strip(),
                }
            )
    return tables

def get_examples():
    # Curated examples for  CRM dataset (these are displayed in UI)
    return [
        {
            "category": "Public (analytics)",
            "items": [
                "For the last 90 days, show monthly revenue (sum of payments) and the month-over-month % change",
                "Who are the top 10 customers by total payments, and what percent of total revenue do they contribute?",
                "List customers who have placed orders but have never made a payment, including their total order count",
                "Which products are frequently ordered together? Show top 10 product pairs by co-occurrence count",
            ],
        },
        {
            "category": "Sandbox (what-if writes — rolled back)",
            "items": [
                "In sandbox mode, increase the credit limit by 15% for all customers in France with creditlimit under 30000",
                "In sandbox mode, mark all 'In Process' orders for customer 112 as 'Cancelled'",
                "In sandbox mode, delete order 10100 safely (delete its orderdetails first, then the order)",
                "In sandbox mode, delete customer 112 safely by removing dependent rows first (orderdetails → orders → payments → customer)",
            ],
        },
    ]

# ----------------------------
# Basic endpoints
# ----------------------------
@app.route("/", methods=["GET"])
@limiter.exempt
def home():
    return jsonify(
        {
            "name": "AskDB",
            "status": "running",
            "usage": {
                "POST": "/api",
                "body": {
                    "question": "Top 3 customers by total payments",
                    "session_id": "demo",
                    "mode": "public",
                    "include_sql": True,
                },
            },
            "modes": {
                "public": "SELECT-only (safe). JOINs and analytics queries are allowed.",
                "sandbox": "SELECT + DML simulated (INSERT/UPDATE/DELETE) and always rolled back.",
            },
            "v2": {
                "byodb": {
                    "connect_endpoint": "POST /connect",
                    "status_endpoint": "GET /connection?session_id=...",
                    "ttl_seconds": CONN_TTL_SECONDS,
                }
            },
        }
    )

@app.route("/health", methods=["GET"])
@limiter.exempt
def health():
    return jsonify({"status": "ok"})

@app.route("/about", methods=["GET"])
@limiter.exempt
def about():
    return jsonify(
        {
            "app": "AskDB",
            "positioning": "Sales Ops / CRM Analytics Copilot + Safe Sandbox Playground",
            "dataset": {
                "name": "ClassicModels-style CRM demo (Supabase Postgres) + BYODB (V2 Phase 1)",
                "entities": ["customers", "orders", "orderdetails", "payments", "products", "employees", "offices"],
            },
            "modes": {
                "public": "SELECT-only analytics (safe).",
                "sandbox": "DML simulated (INSERT/UPDATE/DELETE) and always rolled back.",
            },
            "v2": {
                "byodb": "Users can connect their own database per session using POST /connect (in-memory TTL)."
            },
        }
    )

@app.route("/schema", methods=["GET"])
@limiter.exempt
def schema():
    # NOTE: This returns the demo schema CSV contents used for V1.
    # For BYODB sessions, schema is provided through POST /connect.
    return jsonify({"tables": load_schema_from_csv()})

@app.route("/examples", methods=["GET"])
@limiter.exempt
def examples():
    return jsonify({"examples": get_examples()})

# ----------------------------
# V2 (Phase 1): BYODB endpoints
# ----------------------------
@app.route("/connect", methods=["POST"])
@limiter.limit("20 per hour")
def connect_db():
    """
    Register a per-session database connection + schema CSV text.
    Stored in-memory for a short TTL (privacy-friendly, MVP).
    """
    try:
        _cleanup_connections()

        data = request.get_json() or {}
        session_id = (data.get("session_id") or "").strip()
        db_url = (data.get("db_url") or "").strip()
        schema_csv = (data.get("schema_csv") or "").strip()

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        if not db_url:
            return jsonify({"error": "Missing db_url"}), 400
        if not schema_csv:
            return jsonify({"error": "Missing schema_csv"}), 400

        if len(db_url) > MAX_DBURL_CHARS:
            return jsonify({"error": "db_url too long"}), 400
        if len(schema_csv) > MAX_SCHEMA_CHARS:
            return jsonify({"error": "schema_csv too large"}), 400

        if not _is_allowed_db_url(db_url):
            return jsonify({"error": "Unsupported db_url scheme. Use postgresql/mysql/sqlite."}), 400

        connections[session_id] = SessionConn(
            db_url=db_url,
            schema_csv=schema_csv,
            expires_at=_now() + CONN_TTL_SECONDS,
        )

        return jsonify(
            {
                "status": "connected",
                "session_id": session_id,
                "expires_in_seconds": CONN_TTL_SECONDS,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/connection", methods=["GET"])
@limiter.exempt
def connection_status():
    """
    Check if session has an active BYODB connection.
    """
    _cleanup_connections()
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"connected": False, "reason": "missing session_id"}), 400

    c = connections.get(session_id)
    if not c:
        return jsonify({"connected": False}), 200

    return jsonify(
        {
            "connected": True,
            "expires_at": int(c.expires_at),
            "ttl_seconds": max(0, int(c.expires_at - _now())),
        }
    ), 200

@app.route("/disconnect", methods=["POST"])
@limiter.limit("60 per hour")
def disconnect_db():
    """
    Remove the BYODB connection for this session so /api falls back to the default Supabase demo DB.
    Recommended: also reset chat history to avoid cross-DB context confusion.
    """
    try:
        _cleanup_connections()
        data = request.get_json() or {}
        session_id = (data.get("session_id") or "").strip()
        reset_session = bool(data.get("reset_session", True))  # default True

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        # Remove BYODB connection if present
        if session_id in connections:
            del connections[session_id]

        # Optional: reset chat history so user doesn't carry context across DBs
        if reset_session and session_id in histories:
            histories[session_id] = ChatMessageHistory()

        return jsonify(
            {
                "status": "disconnected",
                "session_id": session_id,
                "reset_session": reset_session
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Main API
# ----------------------------
@app.route("/api", methods=["POST"])
@limiter.limit("6 per minute; 100 per day")
def master1():
    try:
        data = request.get_json() or {}

        q = data.get("question")
        if not q:
            return jsonify({"error": "Missing 'question' in the request body"}), 400

        session_id = (data.get("session_id") or str(uuid4())).strip()
        reset_session = bool(data.get("reset_session", False))
        include_sql = bool(data.get("include_sql", False))
        mode = (data.get("mode") or "public").lower().strip()

        # session history (in-memory)
        history = get_history(session_id)
        if reset_session:
            histories[session_id] = ChatMessageHistory()
            history = histories[session_id]

        history.add_user_message(q)

        formatted_messages = [
            {"role": "user" if msg.type == "user" else "assistant", "content": msg.content}
            for msg in history.messages
        ]

        # V2: if BYODB connection exists for this session, use it
        _cleanup_connections()
        conn = connections.get(session_id)

        t0 = time.time()
        if conn:
            # NOTE: untitled0.chain_code must accept these optional kwargs (we’ll update untitled0.py next)
            result = chain_code(
                q,
                formatted_messages,
                mode=mode,
                db_url_override=conn.db_url,
                schema_csv_override=conn.schema_csv,
            )
        else:
            result = chain_code(q, formatted_messages, mode=mode)

        latency_ms = int((time.time() - t0) * 1000)

        answer_text = result.get("answer", "")

        if result.get("mode") == "sandbox" and result.get("kind") == "DML" and result.get("rolled_back") is True:
            answer_text = "✅ Sandbox simulation (no real data changed — rolled back).\n\n" + answer_text

        history.add_ai_message(answer_text)

        payload = {"answer": answer_text, "session_id": session_id}

        if include_sql:
            rows_preview = result.get("rows_preview", []) or []
            columns = result.get("columns", []) or (list(rows_preview[0].keys()) if rows_preview else [])
            payload.update(
                {
                    "sql": result.get("sql", ""),
                    "mode": result.get("mode", mode),
                    "kind": result.get("kind", ""),
                    "rolled_back": bool(result.get("rolled_back", False)),
                    "rows_returned": int(result.get("rows_returned", 0) or 0),
                    "rows_affected": int(result.get("rows_affected", 0) or 0),
                    "latency_ms": latency_ms,
                    "columns": columns,
                    "rows_preview": rows_preview,
                    # helpful for frontend UX
                    "byodb_connected": bool(conn),
                    "byodb_ttl_seconds": (max(0, int(conn.expires_at - _now())) if conn else 0),
                }
            )

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
