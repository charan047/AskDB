<div align="center">

# AskDB ⚡
### Natural Language → SQL → Business Insight  
**Proof-first analytics copilot with a rollback sandbox + BYODB**

<a href="https://ask-db.vercel.app"><img src="https://img.shields.io/badge/Live%20Demo-Vercel-111827?style=for-the-badge&logo=vercel" /></a>
<a href="https://askdb-jwz4.onrender.com"><img src="https://img.shields.io/badge/API-Render-111827?style=for-the-badge&logo=render" /></a>
<img src="https://img.shields.io/badge/Public-SELECT%20only-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/Sandbox-Rollback-111827?style=for-the-badge" />
<img src="https://img.shields.io/badge/BYODB-Optional-111827?style=for-the-badge" />

<br/>

**AskDB lets you query a SQL database in plain English** and returns:
**(1) the business answer**, **(2) the generated SQL**, and **(3) a preview table** — so every result is **fast, explainable, and auditable**.

</div>

---

## ✨ What makes AskDB different

Most NL→SQL demos stop at “the model generated a query.” AskDB is built like a real system:

- **Proof-first output**: answer + SQL + preview rows in one place (no guesswork)
- **Safety model**:  
  - **Public mode** = `SELECT`-only (joins, CTEs, aggregations allowed)  
  - **Sandbox mode** = `INSERT/UPDATE/DELETE` allowed, but **every write rolls back**
- **BYODB (Bring Your Own Database)**: connect your own Postgres/MySQL/SQLite URL (schema CSV optional; AskDB can auto-detect)
- **Reliability loop**: optional SQL self-repair on execution errors (configurable retries)
- **Scalable primitives**: Redis-backed caching, async jobs (RQ), and Prometheus metrics (optional)

---

## 🚀 Live Demo

- **Frontend:** https://ask-db.vercel.app  
- **Backend API:** https://askdb-jwz4.onrender.com

> Render free instances may sleep. If the UI shows “Connecting…”, open the backend once to wake it up.

---

## 🎬 30-second demo script (copy/paste)

### 1) Public analytics (safe)
`Top 3 customers by total payments`

### 2) Public analytics (safe)
`Total sales by product line`

### 3) Sandbox “what-if” (rolled back)
Switch **mode = sandbox**  
`Delete customer 112`  
 Expect: `rolled_back: true`

### 4) Proof (customer still exists)
Switch back to **mode = public**  
`Show customer 112`  
 Customer is still present → rollback verified

---

## 🧱 Architecture

```mermaid
flowchart LR
  U[User] --> FE[Frontend - React Vite]
  FE --> API[Flask API]
  API --> SCHEMA[Schema - CSV or introspection]
  API --> LLM[Gemini/OpenRouter - LangChain]
  API --> DB[(SQL database)]
  API --> R[(Redis - optional)]
  API --> M[Metrics - Prometheus]
  API --> FE

```

**Execution path**
1. UI sends a question + session_id + mode.
2. Backend loads the **active connection context** (demo DB or BYODB connection).
3. LLM produces SQL using **schema context + few-shot examples**.
4. Guardrails enforce:
   - public = `SELECT` only (single statement, LIMIT applied)
   - sandbox = DML allowed but **rolled back**
   - DDL blocked always
5. Backend returns answer + SQL + preview rows (+ metadata).

---

## 🧩 Key Features

### Public mode (safe analytics)
- `SELECT` / `WITH ... SELECT` only
- single statement only
- enforced LIMIT + query timeout (dialect-aware best effort)

### Sandbox mode (rollback writes)
- Allows `INSERT / UPDATE / DELETE`
- Runs inside a transaction and **always rolls back**
- Multi-statement allowed only for FK-safe delete batches (child → parent)

### BYODB (Bring Your Own Database)
- Connect a DB URL from the UI
- Schema CSV is optional:
  - If CSV is missing/invalid → AskDB auto-detects schema from the DB
- Disconnect anytime to return to the demo DB
- Can be gated behind a demo key in production

### Observability (optional)
- Structured JSON logs
- `/metrics` endpoint (Prometheus format)
- Request IDs for correlation

### Performance (optional)
- Redis caching layer:
  - LLM output caching (question → SQL)
  - SQL result caching (sql hash + db_id → preview rows)
- Async execution path:
  - queue via RQ + polling endpoints for long queries

---

## 🔌 API Overview

Base URL (local): `http://127.0.0.1:5000`

| Endpoint | Method | Description |
|---|---:|---|
| `/` | GET | status + feature flags |
| `/health` | GET | health check |
| `/about` | GET | app metadata |
| `/examples` | GET | curated example prompts |
| `/schema?session_id=...` | GET | active schema (demo or BYODB) |
| `/connection?session_id=...` | GET | active connection status |
| `/connect` | POST | connect BYODB for a session |
| `/disconnect` | POST | disconnect BYODB for a session |
| `/api` | POST | ask a question (sync or async depending on config) |
| `/jobs/<job_id>` | GET | poll async job result |
| `/metrics` | GET | Prometheus metrics (optional) |

### POST `/api` (primary)
```json
{
  "question": "Top 10 customers by total payments",
  "session_id": "demo",
  "mode": "public",
  "include_sql": true
}
```

---

## 🧰 Tech Stack

**Frontend**
- React + Vite + TailwindCSS (Node.js toolchain)
- Deployed on Vercel

**Backend**
- Flask + SQLAlchemy + LangChain + Gemini/OpenRouter
- Rate limiting + structured logging
- Optional: Redis + RQ + Prometheus

**Database**
- Postgres (Supabase demo DB) + BYODB support for Postgres/MySQL/SQLite

---

## 🛠️ Local Setup (Windows / macOS / Linux)

### 1) Backend
```bash
git clone https://github.com/charan047/AskDB.git
cd AskDB

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2) Environment variables
Create `.env` in the repo root (do **not** commit it):

```env
# --- LLM ---
LLM_PROVIDER=gemini  # gemini | openrouter
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GEMINI_MODEL=gemini-2.5-flash

# --- OpenRouter (optional) ---
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDINGS_MODEL=text-embedding-3-small
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=AskDB
EMBEDDINGS_PROVIDER=gemini  # gemini | openrouter

# --- Demo DB (default) ---
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/postgres?sslmode=require

# --- App ---
PORT=5000
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
TABLE_DESCRIPTIONS_PATH=./database_table_descriptions.csv

# --- Optional: feature flags ---
BYODB_ENABLED=1
SANDBOX_ENABLED=1

# Gate BYODB/sandbox behind a demo key (recommended for public deployments)
DEMO_KEY=

# --- Optional: Redis (caching + sessions + async jobs) ---
REDIS_URL=rediss://default:<password>@<host>:<port>

ASYNC_ENABLED=1
ASYNC_DEFAULT=0
RQ_QUEUE_NAME=askdb
```

### 3) Run backend
```bash
python code1.py
```

### 4) Frontend
```bash
cd frontend
npm install
npm run dev
```

UI: `http://localhost:5173`  
API: `http://127.0.0.1:5000`

---

## 🧪 Tests
```bash
pytest -q
```

---

## 🔐 Security Notes
- Never commit secrets (`.env` should be ignored).
- Public mode is read-only (`SELECT` only).
- Sandbox mode always rolls back DML (no permanent writes).
- DDL (CREATE/ALTER/DROP/TRUNCATE) is blocked.
- For public deployments, keep **DEMO_KEY** enabled to prevent arbitrary DB connections.

If you find a security issue, please follow the process in `SECURITY.md`.

---

## 🗺️ Roadmap (V2)
V1 focuses on “proof-first” analytics with safe guardrails. V2 aims for production-scale workflows:
- Huge schema support (schema RAG + join-path hints)
- Cost-aware query rewrites + indexing suggestions
- Saved dashboards + scheduled insights
- Auth/RBAC + multi-tenant isolation
- Stronger observability (p95 latency + LLM vs DB breakdown)

---

## 🤝 Contributing
We welcome issues + PRs. See `CONTRIBUTING.md` for guidelines.

---

## 📄 License
MIT — see `LICENSE`.
