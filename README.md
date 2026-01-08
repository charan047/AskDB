<div align="center">

# AskDB ⚡  
### Natural Language → SQL → Business Insight  
**CRM Analytics Copilot + Safe Sandbox (Rollback)**

<a href="https://ask-db.vercel.app"><img src="https://img.shields.io/badge/Live%20Demo-Vercel-0ea5e9?style=for-the-badge" /></a>
<a href="https://askdb-jwz4.onrender.com"><img src="https://img.shields.io/badge/API-Render-22c55e?style=for-the-badge" /></a>
<img src="https://img.shields.io/badge/Mode-Public%20SELECT%20Only-64748b?style=for-the-badge" />
<img src="https://img.shields.io/badge/Mode-Sandbox%20Rollback-f97316?style=for-the-badge" />

<br/>
<br/>

**AskDB turns plain-English questions into safe, executable SQL** — and returns **answers + SQL + preview rows** so decisions are **fast, explainable, and trustworthy**.

**V1 ships with a standout “What-If” capability:** ✅ Users can try **INSERT / UPDATE / DELETE** in **Sandbox Mode**, but **every write is rolled back** automatically.  
No broken demos. No corrupted databases. Just safe experimentation.

</div>

---

## 🚀 Live Demo

- **Frontend (Vercel):** [https://ask-db.vercel.app](https://ask-db.vercel.app)  
- **Backend (Render):** [https://askdb-jwz4.onrender.com](https://askdb-jwz4.onrender.com) *(optional)*

> **Note:** Render free instances may sleep. If the UI says “Connecting…”, open the backend URL once to wake it up.

---

## 🎬 30-Second Demo Script

> Use these exactly in the UI to show the full story: analytics → trust → safety → proof.

### 1) Public analytics (safe)
* **Prompt:** `Top 3 customers by total payments`

### 2) Public analytics (safe)
* **Prompt:** `Total sales by product line`

### 3) Sandbox write simulation (🔥 rolled back)
* **Action:** Switch **mode = sandbox**
* **Prompt:** `Delete customer 112`
* ✅ **Expect:** `rolled_back: true` (no real data changed)

### 4) Proof (customer still exists)
* **Action:** Switch back to **mode = public**
* **Prompt:** `Show customer 112`
* ✅ **Result:** The customer is still there — sandbox rollback verified.

---
## 🚧 V2 Roadmap — Under Active Development 👀

This repository represents **V1**. Work on **V2** is currently underway to evolve AskDB into a true multi-tenant analytics copilot built for massive schemas and production-grade workflows.

### 🔥 Planned V2 Upgrades

* **Bring Your Own Database (BYODB):** Securely connect your own database via the UI to run instant natural language analytics.
* **Large Schema Support (RAG):** Implementation of schema indexing and retrieval-augmented generation (RAG) to handle databases with hundreds of tables.
* **Query Optimization:** Cost-aware SQL rewrites, smarter auto-limits, and optimized join strategies to protect database performance.
* **Observability & Analytics:** Detailed request IDs, latency breakdowns, cache hit rates, and full audit logs for every generated query.
* **Auth + RBAC:** Granular access control, distinguishing between public analytics, sandbox testers, and administrative users.
* **Premium UX:** Saved queries, shareable insight links, and "Insight Mode" text summaries of data trends.
---

## 💡 Why this matters

Most teams store critical data in SQL databases, but **most stakeholders can’t write SQL**. This creates bottlenecks:
- Sales Ops waiting on analysts for “simple” numbers.
- Risky ad-hoc queries run without guardrails.
- Slow iteration: “What if we update/delete X?” is scary in production.

**AskDB solves it with:**
- **Self-serve analytics** in plain English.
- **Transparent SQL** for trust and auditability.
- **Strong safety controls** (public read-only + rollback sandbox).

---

## 🧠 How AskDB Works (V1 Pipeline)

1. **User Question** → Input via UI.
2. **Schema-aware Context** → Injected from `database_table_descriptions.csv`.
3. **LLM SQL Generation** → Powered by **Gemini** via **LangChain**.
4. **SQL Guardrails**:
    - **Public:** `SELECT`-only + `LIMIT` enforced.
    - **Sandbox:** DML allowed but **ALWAYS rolled back**.
    - **DDL:** Blocked entirely.
5. **Execution** → Run on Postgres (Supabase).
6. **Output** → Answer + SQL + Preview rows + latency metadata.

---

## 🛡️ Safety Model

### ✅ Public Mode (Default)
* `SELECT`-only analytics.
* `LIMIT` enforced (prevents runaway scans).
* Multi-statement queries blocked.
* DDL blocked.

### 🔥 Sandbox Mode
* Allows `INSERT` / `UPDATE` / `DELETE`.
* Executes inside a **database transaction**.
* **Always rolls back** after execution to ensure no data is changed.

---

## 🧰 Tech Stack

* **Frontend:** React + Vite + Tailwind (Vercel)
* **Backend:** Flask + LangChain + SQLAlchemy + Flask-Limiter (Render)
* **LLM:** Google Gemini (LangChain integration)
* **Database:** Postgres (Supabase)

---

## 🚀 Quick Start (Local)

### 1) Clone & Install Backend
```bash
git clone <repository-url>
cd AskDB
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
### 2) Configure Environment Variables
Create a `.env` file in the repository root:

```env
# ---- API Keys ----
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY

# ---- Database ----
# PostgreSQL example (recommended)
DATABASE_URL=postgresql+psycopg2://user:password@host:5432/db?sslmode=require
GEMINI_MODEL=gemini-2.0-flash

# ---- App ----
TABLE_DESCRIPTIONS_PATH=./database_table_descriptions.csv
CORS_ORIGINS=http://localhost:5173,[http://127.0.0.1:5173](http://127.0.0.1:5173)

# Optional (LangChain tracing):
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=AskDB
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
```
### 3) Connect your database
AskDB uses a standard `DATABASE_URL` format. You can switch database engines without changing any application code:

* **PostgreSQL:** `DATABASE_URL=postgresql+psycopg2://user:password@host:5432/db?sslmode=require`
* **MySQL:** `DATABASE_URL=mysql+pymysql://user:password@host:3306/db`
* **SQLite:** `DATABASE_URL=sqlite:///./sample.db`

### 4) Provide schema descriptions
For the LLM to understand your specific data structure, update the `database_table_descriptions.csv` file:

```csv
table_name,description
customers,Contains customer information (name, country, credit limit, etc.)
orders,Stores order headers (status, dates, customer reference)
orderdetails,Stores line-items for each order (quantity, price)
payments,Stores payment transactions per customer
products,Stores product catalog and pricing
employees,Stores employee directory and reporting relationships
```
### 5) Run backend
```bash
# Ensure you are in the root directory
python code1.py
```
### 6) Run frontend
```bash
cd frontend
npm install
```
## 📡 API Usage

### Ask a question (Public Mode)
```bash
curl -X POST [http://127.0.0.1:5000/api](http://127.0.0.1:5000/api) \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the price of 1968 Ford Mustang?",
    "session_id": "demo",
    "mode": "public",
    "include_sql": true
  }'
```
### Sandbox “what-if” simulation (Rolled back)
```bash

curl -X POST [http://127.0.0.1:5000/api](http://127.0.0.1:5000/api) \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Delete customer 112",
    "session_id": "demo",
    "mode": "sandbox",
    "include_sql": true
  }'
```
### 2. Sandbox Mode (The "Wow" Feature)

## 🔥 Sandbox Mode Details
The Sandbox is designed for safe experimentation. Here is the internal safety flow:

1. **User Request:** User sends a DML command (INSERT/UPDATE/DELETE).
2. **Transaction Start:** AskDB opens a new database transaction.
3. **Execution:** The SQL runs against the live database within that local transaction scope.
4. **Metadata Capture:** The results (rows affected) are captured for the UI.
5. **Mandatory Rollback:** The transaction is explicitly rolled back. **No data is ever committed.**



### ⭐ Stay Tuned
**Something big is building.** Star this repository to get notified when the V2 features drop and the project transitions to a full-scale production tool!
## 🤝 Contributing
Contributions make the open-source community an amazing place to learn and create.
1. **Fork** the Project.
2. **Create** your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the Branch (`git push origin feature/AmazingFeature`).
5. **Open** a Pull Request.
## 🆘 Support & Troubleshooting

If you encounter issues or need help optimizing your AskDB instance, please follow these steps:

* **Open an Issue:** Found a bug or have a feature request? [Create an issue](https://github.com/YOUR_USERNAME/AskDB/issues) in the repository.
* **Check Examples:** Review the curated queries in the UI or access the `/examples` API endpoint to see the best way to phrase natural language questions.
* **Schema Tuning:** If the LLM is selecting the wrong tables, validate your `database_table_descriptions.csv`. Accurate descriptions are the key to high-quality SQL generation.
* **Backend Connectivity:** Ensure your `.env` variables are correctly set and the database user has the appropriate permissions for the mode you are using.

---
## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
