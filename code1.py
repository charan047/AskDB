# .\.venv\Scripts\Activate.ps1
from untitled0 import chain_code
from langchain_community.chat_message_histories import ChatMessageHistory
from flask import Flask, request, jsonify
from uuid import uuid4
from flask_cors import CORS
import time

app = Flask(__name__)
import os
import csv
origins = os.getenv("CORS_ORIGINS", "https://ask-db.vercel.app,http://localhost:5173,http://127.0.0.1:5173").split(",")
CORS(app, resources={r"/*": {"origins": [o.strip() for o in origins]}})

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# In-memory session store (local/dev). For production we can swap to Redis.
histories = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in histories:
        histories[session_id] = ChatMessageHistory()
    return histories[session_id]


def load_schema_from_csv():
    csv_path = os.getenv("TABLE_DESCRIPTIONS_PATH", "./database_table_descriptions.csv")
    tables = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tables.append({
                "table_name": row.get("table_name", "").strip(),
                "description": row.get("description", "").strip()
            })
    return tables

def get_examples():
    # Curated public examples for your CRM dataset
    return [
        {
            "category": "Sales & Revenue",
            "items": [
                "Top 3 customers by total payments",
                "Total sales by product line",
                "Show total order value for each order",
                "Customers with orders but no payments",
            ],
        },
        {
            "category": "Customers",
            "items": [
                "List all customers in France with a credit limit over 20000",
                "Show all customers and their sales rep (employee)",
                "Which countries have the highest average credit limit?",
            ],
        },
        {
            "category": "Orders",
            "items": [
                "Show each order with the customer name and status",
                "List all products in order 10100 with quantity and price",
                "Orders that are still In Process",
            ],
        },
        {
            "category": "Inventory",
            "items": [
                "Products with low stock (quantityinstock < 4000)",
                "Show product details for Motorcycles product line",
            ],
        },
        {
            "category": "Sandbox (simulated writes)",
            "items": [
                "Update credit limit of customer 103 to 25000",
                "Delete customer 112",
                "Delete order 10101 and its order details",
            ],
        },
    ]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "AskDB",
        "status": "running",
        "usage": {
            "POST": "/api",
            "body": {
                "question": "What is the price of 1968 Ford Mustang?",
                "session_id": "demo",
                "mode": "public",
                "include_sql": False
            }
        },
        "modes": {
            "public": "SELECT-only (safe). JOINs and analytics queries are allowed.",
            "sandbox": "SELECT + DML simulated (INSERT/UPDATE/DELETE) and always rolled back."
        }
    })


@app.route("/health", methods=["GET"])
@limiter.exempt
def health():
    return jsonify({"status": "ok"})


@app.route("/about", methods=["GET"])
@limiter.exempt
def about():
    return jsonify({
        "app": "AskDB",
        "positioning": "Sales Ops / CRM Analytics Copilot + Safe Sandbox Playground",
        "dataset": {
            "name": "ClassicModels-style CRM demo (Supabase Postgres)",
            "entities": ["customers", "orders", "orderdetails", "payments", "products", "employees", "offices"]
        },
        "modes": {
            "public": "SELECT-only analytics (safe).",
            "sandbox": "DML simulated (INSERT/UPDATE/DELETE) and always rolled back."
        }
    })

@app.route("/schema", methods=["GET"])
@limiter.exempt
def schema():
    return jsonify({
        "tables": load_schema_from_csv()
    })

@app.route("/examples", methods=["GET"])
@limiter.exempt
def examples():
    return jsonify({
        "examples": get_examples()
    })



@app.route("/api", methods=["POST"])
@limiter.limit("6 per minute; 100 per day")
def master1():
    try:
        data = request.get_json() or {}

        q = data.get("question")
        if not q:
            return jsonify({"error": "Missing 'question' in the request body"}), 400

        session_id = data.get("session_id") or str(uuid4())
        reset_session = bool(data.get("reset_session", False))
        include_sql = bool(data.get("include_sql", False))
        mode = (data.get("mode") or "public").lower().strip()

        history = get_history(session_id)
        if reset_session:
            histories[session_id] = ChatMessageHistory()
            history = histories[session_id]

        # Add user message to session history
        history.add_user_message(q)

        # Convert history into the format expected by chain_code
        formatted_messages = [
            {"role": "user" if msg.type == "user" else "assistant", "content": msg.content}
            for msg in history.messages
        ]

        # Run chain
        t0 = time.time()
        result = chain_code(q, formatted_messages, mode=mode)  # <-- now returns dict
        latency_ms = int((time.time() - t0) * 1000)

        answer_text = result.get("answer", "")
        # If sandbox DML was rolled back, make it explicit in the user-facing answer
        if result.get("mode") == "sandbox" and result.get("kind") == "DML" and result.get("rolled_back") is True:
            answer_text = (
                "✅ Sandbox simulation (no real data changed — rolled back).\n\n"
                + answer_text
            )

        # Store assistant message in history (only the final answer text)
        history.add_ai_message(answer_text)

        payload = {
            "answer": answer_text,
            "session_id": session_id
        }

        if include_sql:
            rows_preview = result.get("rows_preview", []) or []
            columns = result.get("columns", []) or (list(rows_preview[0].keys()) if rows_preview else [])
            payload.update({
                "sql": result.get("sql", ""),
                "mode": result.get("mode", mode),
                "kind": result.get("kind", ""),
                "rolled_back": bool(result.get("rolled_back", False)),
                "rows_returned": int(result.get("rows_returned", 0) or 0),
                "rows_affected": int(result.get("rows_affected", 0) or 0),
                "latency_ms": latency_ms,
                
                
                # NEW (for frontend table)
                "columns": columns,
                "rows_preview": rows_preview,
            })

        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
