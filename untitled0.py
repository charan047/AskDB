"""AskDB core.

We treat LLM-generated SQL as **untrusted input** and enforce layered safety + performance controls.

Big-tech style reliability/performance features added here:
- **Cost-aware preflight** with EXPLAIN (when supported)
- **Slow-query auto-optimization** retry (LLM rewrite for speed)
- **Response caching** for public SELECT queries (connection-scoped)
"""

import os
import re
import io
import csv
import json
import time
import logging
import hashlib
from collections import OrderedDict
from pathlib import Path
from operator import itemgetter
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import text, inspect
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities.sql_database import SQLDatabase

# create_sql_query_chain lives in langchain for most versions. Keep a fallback for older installs.
try:
    from langchain.chains.sql_database.query import create_sql_query_chain
except Exception:  # pragma: no cover
    from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from pydantic import BaseModel, Field, ValidationError
from infra import get_redis, rget_json, rset_json

logger = logging.getLogger('askdb.core')


load_dotenv()

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# ------------------------------------------------------------
# Defaults / knobs
# ------------------------------------------------------------
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "10"))
PREVIEW_ROW_CAP = int(os.getenv("PREVIEW_ROW_CAP", "20"))
DB_TIMEOUT_MS = int(os.getenv("DB_TIMEOUT_MS", "8000"))
SQL_FIX_RETRIES = int(os.getenv("SQL_FIX_RETRIES", "2"))

# Insights/Charts: avoid adding extra LLM latency unless explicitly enabled
AI_INSIGHTS_LLM = os.getenv("AI_INSIGHTS_LLM", "0") == "1"

# Performance controls (big-tech defaults)
PREFLIGHT_EXPLAIN = os.getenv("PREFLIGHT_EXPLAIN", "1") == "1"
MAX_EST_COST = int(os.getenv("MAX_EST_COST", "250000"))
MAX_EST_ROWS = int(os.getenv("MAX_EST_ROWS", "200000"))
SLOW_QUERY_MS = int(os.getenv("SLOW_QUERY_MS", "4000"))

# Lightweight response cache for public SELECT queries (connection-scoped)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "512"))

# Cache SELECT execution results (safe public mode) to reduce DB latency on repeat queries
SQL_RESULT_CACHE_TTL_SECONDS = int(os.getenv("SQL_RESULT_CACHE_TTL_SECONDS", "120"))

SCHEMA_RAG_ENABLED = os.getenv("SCHEMA_RAG_ENABLED", "0") == "1"
SCHEMA_RAG_K = int(os.getenv("SCHEMA_RAG_K", "8"))
SCHEMA_RAG_PERSIST = os.getenv("SCHEMA_RAG_PERSIST", "0") == "1"

# ------------------------------------------------------------
# SQL cleanup + splitting
# ------------------------------------------------------------
def clean_sql_query(text_: str) -> str:
    text_ = text_ or ""
    # Remove ```sql blocks
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text_ = re.sub(block_pattern, r"\1", text_, flags=re.DOTALL)
    # Remove "SQLQuery:" prefix variants
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQLite|SQL)\s*:\s*"
    text_ = re.sub(prefix_pattern, "", text_, flags=re.IGNORECASE)
    # Remove backticks
    text_ = re.sub(r'`([^`]*)`', r'\1', text_)
    # Normalize whitespace
    text_ = re.sub(r"\s+", " ", text_).strip()
    # Extract first statement-ish
    m = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*", text_, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text_ = m.group(0).strip()
    return text_

def _strip_sql(sql: str) -> str:
    sql = (sql or "").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql

def _split_sql_statements(sql: str) -> List[str]:
    """Split on semicolons NOT inside quotes (good enough for LLM SQL)."""
    s = (sql or "").strip()
    if not s:
        return []
    stmts, buf = [], []
    in_single = False
    in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts

def _sql_kind(sql: str) -> str:
    """SELECT, DML, DDL, OTHER"""
    s = _strip_sql(sql)
    s = re.sub(r"^\s*(--[^\n]*\n\s*)+", "", s, flags=re.MULTILINE).strip()
    if not s:
        return "OTHER"
    first = s.split(None, 1)[0].upper()
    if first in ("WITH", "SELECT"):
        return "SELECT"
    if first in ("INSERT", "UPDATE", "DELETE"):
        return "DML"
    if first in ("CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE"):
        return "DDL"
    return "OTHER"

def _ensure_limit(sql: str, limit: int = 100) -> str:
    s = _strip_sql(sql)
    if re.search(r"\bLIMIT\b", s, flags=re.IGNORECASE):
        return s
    return f"{s} LIMIT {limit}"

# ------------------------------------------------------------
# Dialect helpers
# ------------------------------------------------------------
def _dialect_name(db_: SQLDatabase) -> str:
    try:
        return (db_._engine.dialect.name or "").lower()
    except Exception:
        return "unknown"

def _apply_timeout(conn, dialect: str, timeout_ms: int) -> None:
    """Best-effort timeouts."""
    try:
        if dialect in ("postgresql", "postgres"):
            conn.execute(text(f"SET statement_timeout = {int(timeout_ms)}"))
        elif dialect in ("mysql", "mariadb"):
            # MySQL expects ms
            conn.execute(text(f"SET SESSION MAX_EXECUTION_TIME={int(timeout_ms)}"))
        # sqlite: no server-side timeout
    except Exception:
        pass

# ------------------------------------------------------------
# EXPLAIN preflight (cost-aware guardrails)
# ------------------------------------------------------------
def _explain_preflight(conn, dialect: str, sql: str) -> Optional[Dict[str, Any]]:
    """Return a light-weight cost estimate when supported, else None.

    We use EXPLAIN (FORMAT JSON) for Postgres and JSON EXPLAIN for MySQL when available.
    This is intentionally best-effort; BYODB connections might restrict EXPLAIN.
    """
    try:
        s = _strip_sql(sql)
        if dialect in ("postgresql", "postgres"):
            row = conn.execute(text(f"EXPLAIN (FORMAT JSON) {s}")).fetchone()
            if not row:
                return None
            plan = row[0][0] if isinstance(row[0], list) else row[0]
            # Postgres returns list with a dict. Extract common fields.
            top = plan.get("Plan", {}) if isinstance(plan, dict) else {}
            return {
                "engine": "postgres",
                "total_cost": float(top.get("Total Cost", 0) or 0),
                "plan_rows": int(top.get("Plan Rows", 0) or 0),
                "plan_width": int(top.get("Plan Width", 0) or 0),
                "node_type": top.get("Node Type", ""),
            }
        if dialect in ("mysql", "mariadb"):
            # Some MySQL variants support FORMAT=JSON
            row = conn.execute(text(f"EXPLAIN FORMAT=JSON {s}")).fetchone()
            if not row:
                return None
            raw = row[0]
            if isinstance(raw, str):
                data = json.loads(raw)
            else:
                data = raw
            # MySQL JSON explain doesn't provide a single cost number reliably.
            return {"engine": "mysql", "raw": data}
        if dialect == "sqlite":
            row = conn.execute(text(f"EXPLAIN QUERY PLAN {sql}")).fetchall()
            return {"engine": "sqlite", "raw": [list(r) for r in row]}
    except Exception:
        return None
    return None

def _is_over_budget(explain: Optional[Dict[str, Any]]) -> bool:
    if not explain:
        return False
    if explain.get("engine") == "postgres":
        return (explain.get("total_cost", 0) > MAX_EST_COST) or (explain.get("plan_rows", 0) > MAX_EST_ROWS)
    # For other engines, we don't have portable metrics; skip.
    return False

# ------------------------------------------------------------
# Guardrails execution (supports sandbox rollback)
# ------------------------------------------------------------
def execute_with_guardrails(db_: SQLDatabase, sql: str, mode: str, cache_ns: Optional[str] = None) -> Dict[str, Any]:
    mode = (mode or "public").lower().strip()
    if mode not in ("public", "sandbox"):
        raise ValueError("Invalid mode. Use 'public' or 'sandbox'.")

    statements = _split_sql_statements(sql)
    if not statements:
        raise ValueError("Empty SQL is not allowed.")

    # Public: single statement only
    if mode == "public" and len(statements) > 1:
        raise ValueError("Multi-statement SQL is not allowed in public mode.")

    kinds = [_sql_kind(s) for s in statements]

    # Block DDL always
    if any(k == "DDL" for k in kinds):
        raise ValueError("DDL statements are blocked (CREATE/ALTER/DROP/TRUNCATE/GRANT/REVOKE).")

    # Public: SELECT only
    if mode == "public" and kinds[0] != "SELECT":
        raise ValueError("Public mode supports SELECT queries only. Use sandbox for write simulation.")

    engine = db_._engine
    dialect = _dialect_name(db_)
    rolled_back = False

    # Sandbox multi-statement DML batch
    if mode == "sandbox" and len(statements) > 1:
        if any(k != "DML" for k in kinds):
            raise ValueError("In sandbox mode, multi-statement is allowed only for pure DML batches.")
        conn = engine.connect()
        trans = conn.begin()
        try:
            _apply_timeout(conn, dialect, DB_TIMEOUT_MS)
            total_affected = 0
            breakdown = []
            for i, stmt in enumerate(statements, start=1):
                stmt_sql = _strip_sql(stmt)
                try:
                    res = conn.execute(text(stmt_sql))
                except IntegrityError as e:
                    try:
                        trans.rollback()
                    except Exception:
                        pass
                    rolled_back = True
                    msg = str(getattr(e, "orig", e))
                    raise ValueError(f"Sandbox DML failed due to integrity constraint: {msg}") from e
                affected = int(res.rowcount) if res.rowcount is not None else 0
                total_affected += affected
                breakdown.append(f"{i}) rows_affected={affected}")
            try:
                trans.rollback()
            except Exception:
                pass
            rolled_back = True
            return {
                "sql": ";\n".join(_strip_sql(s) for s in statements),
                "kind": "DML",
                "mode": mode,
                "rolled_back": rolled_back,
                "rowcount": total_affected,
                "rows": [],
                "preview": f"DML batch simulated (rolled back). Statements: {len(statements)}. Total rows affected: {total_affected}. Breakdown: " + " | ".join(breakdown),
            }
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Sandbox single DML
    if mode == "sandbox" and kinds[0] == "DML":
        final_sql = _strip_sql(statements[0])
        conn = engine.connect()
        trans = conn.begin()
        try:
            _apply_timeout(conn, dialect, DB_TIMEOUT_MS)
            try:
                res = conn.execute(text(final_sql))
            except IntegrityError as e:
                try:
                    trans.rollback()
                except Exception:
                    pass
                rolled_back = True
                msg = str(getattr(e, "orig", e))
                raise ValueError(f"Sandbox DML failed due to integrity constraint: {msg}") from e

            rowcount = int(res.rowcount) if res.rowcount is not None else 0
            try:
                trans.rollback()
            except Exception:
                pass
            rolled_back = True
            return {
                "sql": final_sql,
                "kind": "DML",
                "mode": mode,
                "rolled_back": rolled_back,
                "rowcount": rowcount,
                "rows": [],
                "preview": f"DML simulated (rolled back). Rows affected: {rowcount}.",
            }
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # SELECT path
    if kinds[0] != "SELECT":
        raise ValueError("Only SELECT is allowed here. Use sandbox mode for write simulation.")

    final_sql = _ensure_limit(_strip_sql(statements[0]), limit=100)

    # Public SELECT execution cache (safe) — avoids repeated DB hits for identical SQL
    if mode == "public" and SQL_RESULT_CACHE_TTL_SECONDS > 0 and cache_ns and get_redis():
        sql_cache_key = f"sql:{cache_ns}:{hashlib.sha256(final_sql.encode('utf-8')).hexdigest()}"
        cached = rget_json(sql_cache_key)
        if isinstance(cached, dict) and isinstance(cached.get('rows'), list):
            rows = cached.get('rows', [])
            return {
                "sql": final_sql,
                "kind": "SELECT",
                "mode": mode,
                "rolled_back": False,
                "rowcount": int(cached.get('rowcount', len(rows))),
                "rows": rows,
                "preview": f"Rows returned: {int(cached.get('rowcount', len(rows)))}. Preview: {rows[:5]}",
                "db_ms": 0,
                "db_cache_hit": True,
            }

    t_db0 = time.time()
    with engine.connect() as conn:
        _apply_timeout(conn, dialect, DB_TIMEOUT_MS)
        res = conn.execute(text(final_sql))
        cols = list(res.keys())
        fetched = res.fetchmany(100)
        rows = [dict(zip(cols, r)) for r in fetched]
    db_ms = int((time.time() - t_db0) * 1000)

    # Store select result cache (public only)
    if mode == "public" and SQL_RESULT_CACHE_TTL_SECONDS > 0 and cache_ns and get_redis():
        sql_cache_key = f"sql:{cache_ns}:{hashlib.sha256(final_sql.encode('utf-8')).hexdigest()}"
        rset_json(sql_cache_key, {"rowcount": len(rows), "rows": rows}, ttl_s=SQL_RESULT_CACHE_TTL_SECONDS)

    return {
        "sql": final_sql,
        "kind": "SELECT",
        "mode": mode,
        "rolled_back": False,
        "rowcount": len(rows),
        "rows": rows,
        "preview": f"Rows returned: {len(rows)}. Preview: {rows[:5]}",
        "db_ms": db_ms,
        "db_cache_hit": False,
    }

# ------------------------------------------------------------
# Schema CSV parsing + auto schema from DB
# ------------------------------------------------------------
_TABLE_COL_CANDIDATES = {"table_name", "table", "name", "tablename"}
_DESC_COL_CANDIDATES = {"description", "desc", "table_description", "details"}

def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower()).strip("_")

def _detect_delimiter(header_line: str) -> str:
    candidates = [",", "\t", ";", "|"]
    best = ","
    best_count = -1
    for d in candidates:
        c = header_line.count(d)
        if c > best_count:
            best_count = c
            best = d
    return best

def parse_schema_csv_text(schema_csv_text: str) -> Tuple[Optional[str], List[Dict[str, str]], List[str]]:
    """
    Returns (table_details_text or None, tables_list, warnings)
    tables_list item: {table_name, description}
    """
    warnings: List[str] = []
    txt = (schema_csv_text or "").strip()
    if not txt:
        return None, [], ["schema_csv is empty"]

    txt = txt.lstrip("\ufeff")
    lines = txt.splitlines()
    if not lines:
        return None, [], ["schema_csv has no lines"]

    delimiter = _detect_delimiter(lines[0])
    f = io.StringIO(txt)
    try:
        reader = csv.DictReader(f, delimiter=delimiter)
    except Exception as e:
        return None, [], [f"schema_csv could not be parsed: {e}"]

    if not reader.fieldnames:
        return None, [], ["schema_csv missing headers"]

    headers = [_norm_header(h) for h in reader.fieldnames]
    header_map = {h: orig for h, orig in zip(headers, reader.fieldnames)}

    table_key = None
    desc_key = None
    for h in headers:
        if h in _TABLE_COL_CANDIDATES and table_key is None:
            table_key = header_map[h]
        if h in _DESC_COL_CANDIDATES and desc_key is None:
            desc_key = header_map[h]

    if table_key is None:
        return None, [], ["schema_csv missing a table column (expected headers like table_name/table/name)"]
    if desc_key is None:
        warnings.append("schema_csv missing a description column; descriptions will be blank")
        desc_key = table_key  # harmless

    out_blocks: List[str] = []
    tables: List[Dict[str, str]] = []
    row_count = 0
    for row in reader:
        row_count += 1
        tname = (row.get(table_key) or "").strip()
        if not tname:
            continue
        desc = (row.get(desc_key) or "").strip() if desc_key else ""
        tables.append({"table_name": tname, "description": desc})
        out_blocks.append(f"Table Name:{tname}\nTable Description:{desc}\n")

    if not tables:
        return None, [], ["schema_csv parsed but produced no table rows"]

    if row_count == 0:
        warnings.append("schema_csv contains headers but no rows")

    return "\n".join(out_blocks).strip() + "\n", tables, warnings

def auto_schema_from_db(db_: SQLDatabase, max_tables: int = 120, max_cols_per_table: int = 24) -> Tuple[str, List[Dict[str, str]]]:
    """
    Returns (table_details_text, tables_list).
    Auto schema includes columns and types in description.
    """
    insp = inspect(db_._engine)
    table_names = []
    try:
        table_names = insp.get_table_names()
    except Exception:
        # fallback
        table_names = list(db_.get_usable_table_names())

    table_names = table_names[:max_tables]
    blocks: List[str] = []
    tables: List[Dict[str, str]] = []

    for t in table_names:
        cols_desc = []
        try:
            cols = insp.get_columns(t)
            for c in cols[:max_cols_per_table]:
                cols_desc.append(f"{c.get('name')} ({c.get('type')})")
        except Exception:
            pass

        desc = "Columns: " + ", ".join(cols_desc) if cols_desc else "Columns: (unavailable)"
        tables.append({"table_name": t, "description": desc})
        blocks.append(f"Table Name:{t}\nTable Description:{desc}\n")

    return "\n".join(blocks).strip() + "\n", tables

def mask_db_url(db_url: str) -> str:
    s = db_url or ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:••••@", s)

def build_schema_context(db_url: str, schema_csv_text: Optional[str]) -> Dict[str, Any]:
    """
    Creates SQLDatabase, validates connectivity, and returns:
    - db: SQLDatabase
    - table_details: str (for LLM)
    - tables: list[{table_name, description}]
    - schema_source: 'csv' or 'auto'
    - dialect: 'postgresql'/'mysql'/...
    - masked_host: best-effort host string
    - warnings: list[str]
    """
    warnings: List[str] = []
    db_ = SQLDatabase.from_uri(db_url)

    # test connect quickly
    try:
        with db_._engine.connect() as conn:
            _apply_timeout(conn, _dialect_name(db_), min(DB_TIMEOUT_MS, 4000))
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}")

    table_details: Optional[str] = None
    tables: List[Dict[str, str]] = []
    schema_source = "auto"

    if schema_csv_text:
        parsed, parsed_tables, w = parse_schema_csv_text(schema_csv_text)
        warnings.extend(w)
        if parsed is not None:
            table_details = parsed
            tables = parsed_tables
            schema_source = "csv"

    if table_details is None:
        auto_details, auto_tables = auto_schema_from_db(db_)
        table_details = auto_details
        tables = auto_tables
        schema_source = "auto"
        warnings.append("schema_csv missing/invalid → using auto schema from DB")

    dialect = _dialect_name(db_)
    masked = mask_db_url(db_url)
    host = ""
    m = re.search(r"@([^:/\?]+)", db_url)
    if m:
        host = m.group(1)

    return {
        "db": db_,
        "table_details": table_details,
        "tables": tables,
        "schema_source": schema_source,
        "dialect": dialect,
        "masked_url": masked,
        "host": host,
        "warnings": warnings,
    }

# ------------------------------------------------------------
# Few-shot examples (portable)
# ------------------------------------------------------------
EXAMPLES = [
    # Public (SELECT)
    {
        "input": "Top 10 customers by total payments and their share of total revenue.",
        "query": (
            "WITH totals AS ("
            "  SELECT customernumber, SUM(amount) AS total_paid "
            "  FROM payments "
            "  GROUP BY customernumber"
            "), grand AS (SELECT SUM(total_paid) AS grand_total FROM totals), "
            "top10 AS (SELECT customernumber, total_paid FROM totals ORDER BY total_paid DESC LIMIT 10) "
            "SELECT c.customername, t.total_paid, "
            "ROUND(100.0 * t.total_paid / NULLIF(g.grand_total, 0), 2) AS revenue_share_pct "
            "FROM top10 t "
            "JOIN customers c ON c.customernumber = t.customernumber "
            "CROSS JOIN grand g "
            "ORDER BY t.total_paid DESC;"
        )
    },
    {
        "input": "Total sales by product line (revenue = quantityordered * priceeach).",
        "query": (
            "SELECT p.productline, "
            "SUM(od.quantityordered * od.priceeach) AS revenue "
            "FROM orderdetails od "
            "JOIN products p ON p.productcode = od.productcode "
            "GROUP BY p.productline "
            "ORDER BY revenue DESC "
            "LIMIT 10;"
        )
    },
    {
        "input": "Customers who have orders but never made a payment.",
        "query": (
            "SELECT c.customername, COUNT(DISTINCT o.ordernumber) AS order_count "
            "FROM customers c "
            "JOIN orders o ON o.customernumber = c.customernumber "
            "LEFT JOIN payments p ON p.customernumber = c.customernumber "
            "WHERE p.customernumber IS NULL "
            "GROUP BY c.customername "
            "ORDER BY order_count DESC "
            "LIMIT 50;"
        )
    },
    # Sandbox (DML simulated)
    {
        "input": "In sandbox mode, increase creditlimit by 10% for customers in France with creditlimit under 30000.",
        "query": (
            "UPDATE customers "
            "SET creditlimit = creditlimit * 1.10 "
            "WHERE country = 'France' AND creditlimit < 30000;"
        )
    },
    {
        "input": "In sandbox mode, delete order 10100 safely (delete orderdetails first, then the order).",
        "query": (
            "DELETE FROM orderdetails WHERE ordernumber = 10100; "
            "DELETE FROM orders WHERE ordernumber = 10100;"
        )
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}\nSQLQuery:"), ("ai", "{query}")]
)

def build_few_shot_prompt() -> FewShotChatMessagePromptTemplate:
    static_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=EXAMPLES,
        input_variables=["input"],
    )
    if os.getenv("USE_SEMANTIC_EXAMPLES", "0") != "1":
        return static_prompt

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
        selector = SemanticSimilarityExampleSelector.from_examples(
            examples=EXAMPLES,
            embeddings=embeddings,
            vectorstore_cls=Chroma,
            vectorstore_kwargs={
                "collection_name": "askdb_examples",
                "persist_directory": ".chroma_examples",
            },
            k=2,
            input_keys=["input"],
        )
        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=selector,
            input_variables=["input", "top_k"],
        )
    except Exception as e:
        print(f"[AskDB] Semantic examples disabled; falling back to static examples. Reason: {e}")
        return static_prompt

few_shot_prompt = build_few_shot_prompt()

# ------------------------------------------------------------
# Table selection model
# ------------------------------------------------------------
class TableSelection(BaseModel):
    name: List[str] = Field(description="List of table names relevant to the question.")

table_details_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Return the names of ALL SQL tables that might be relevant.\n\n"
         "Tables:\n{table_details}\n\n"
         "Return only the table names."),
        ("human", "{question}")
    ]
)

structured_llm = llm.with_structured_output(TableSelection)
table_chain = table_details_prompt | structured_llm

def get_tables(table_response: TableSelection) -> List[str]:
    return table_response.name

select_table = {"question": itemgetter("question"), "table_details": itemgetter("table_details")} | table_chain | get_tables

# ------------------------------------------------------------
# SQL generation prompt (dialect-aware)
# ------------------------------------------------------------
def make_final_prompt(dialect: str) -> ChatPromptTemplate:
    dialect_disp = dialect or "SQL"
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a {dialect_disp} expert. Output ONE valid {dialect_disp} SQL query.\n"
                "Return ONLY the SQL. No explanations. No markdown.\n\n"
                "Context:\n"
                "- mode: {mode}\n"
                "- top_k: {top_k}\n\n"
                "Rules (STRICT):\n"
                "1) Public mode:\n"
                "   - ONLY generate SELECT (or WITH ... SELECT)\n"
                "   - Must be a SINGLE statement\n"
                "2) Sandbox mode:\n"
                "   - Generate DML (INSERT/UPDATE/DELETE) ONLY if the user explicitly asks to change data\n"
                "   - Multi-statement is allowed ONLY for FK-safe DML batches (delete children then parent)\n"
                "3) Always BLOCK and never generate:\n"
                "   - DDL (CREATE/ALTER/DROP/TRUNCATE)\n"
                "   - GRANT/REVOKE\n\n"
                "Quality:\n"
                "- Prefer explicit columns; avoid SELECT * unless asked\n"
                "- For SELECT: apply LIMIT {top_k} unless user asks for all rows\n"
                "- Optimize for low latency: aggregate/filter BEFORE joining dimension tables; apply ORDER BY+LIMIT inside a subquery/CTE before joins\n"
                "- If the question is about a time period and a date-like column exists (e.g., *date*, created_at), add a reasonable default window (e.g., last 90 days) unless the user asks all-time\n"
                "- Use table/column names exactly from table_info\n"
                "- When joining, use correct keys; avoid cartesian products\n\n"
                "Here is the relevant table info:\n{table_info}\n\n"
                "Examples follow."
            ),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

# ------------------------------------------------------------
# Schema RAG (optional): reduce table_info to relevant subset
# ------------------------------------------------------------
_schema_vs_cache: Dict[str, Any] = {}

def _split_table_blocks(table_details: str) -> List[str]:
    """Split 'Table Name:' blocks into list strings."""
    td = (table_details or "").strip()
    if not td:
        return []
    parts = re.split(r"(?=Table Name:)", td)
    blocks = [p.strip() for p in parts if p.strip()]
    return blocks

def _schema_rag_select(question: str, table_details: str, db_cache_key: str) -> str:
    blocks = _split_table_blocks(table_details)
    if not blocks:
        return table_details

    # Build / reuse vectorstore
    vs_key = f"schema_vs::{db_cache_key}"
    if vs_key not in _schema_vs_cache:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="retrieval_query")
        persist_dir = ".chroma_schema" if SCHEMA_RAG_PERSIST else None
        vs = Chroma(
            collection_name=f"askdb_schema_{hashlib.md5(db_cache_key.encode()).hexdigest()[:12]}",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        # (Re)create collection contents
        try:
            vs._collection.delete(where={})
        except Exception:
            pass
        vs.add_texts(texts=blocks, metadatas=[{"i": i} for i in range(len(blocks))])
        _schema_vs_cache[vs_key] = vs

    vs = _schema_vs_cache[vs_key]
    try:
        docs = vs.similarity_search(question, k=min(SCHEMA_RAG_K, max(4, len(blocks))))
        selected = "\n\n".join(d.page_content for d in docs)
        return selected.strip() + "\n"
    except Exception:
        return table_details

# ------------------------------------------------------------
# Answer prompt + optional AI summary/chart
# ------------------------------------------------------------
answer_prompt = PromptTemplate.from_template(
    """You are answering questions over a SQL database.

Mode: {mode}
Note: In sandbox mode, DML changes are always rolled back (no permanent changes).

Question: {question}
SQL Query: {query}
SQL Result: {result}

Write the answer clearly and concisely in business-friendly language."""
)
rephrase_answer = answer_prompt | llm | StrOutputParser()

class ChartSpec(BaseModel):
    summary: str = Field(description="1-2 sentence executive summary of the results")
    chart_type: str = Field(description="bar or line or none")
    x_key: Optional[str] = None
    y_key: Optional[str] = None
    title: Optional[str] = None

def _infer_chart_and_summary(question: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return an executive summary + a simple chart spec.

    **Performance note:** this runs on every SELECT response, so we default to a
    fast heuristic and only call the LLM if AI_INSIGHTS_LLM=1.
    """
    if not rows:
        return {"insights_summary": "", "chart_spec": None}

    sample = rows[: min(len(rows), 20)]
    cols = list(sample[0].keys())

    def is_num(v) -> bool:
        try:
            float(v)
            return True
        except Exception:
            return False

    def is_date_like(v) -> bool:
        if v is None:
            return False
        s = str(v)
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}", s))

    # Heuristic: pick x (date-like or first stringy col) and y (first numeric col)
    x_key = None
    y_key = None
    for c in cols:
        if x_key is None and ("date" in c.lower() or "time" in c.lower() or is_date_like(sample[0].get(c))):
            x_key = c
    for c in cols:
        if y_key is None and is_num(sample[0].get(c)):
            y_key = c
    if x_key is None:
        for c in cols:
            if not is_num(sample[0].get(c)):
                x_key = c
                break

    chart = None
    if x_key and y_key and len(sample) > 1:
        chart = {
            "type": "line" if ("date" in x_key.lower() or "time" in x_key.lower() or is_date_like(sample[0].get(x_key))) else "bar",
            "x_key": x_key,
            "y_key": y_key,
            "title": "",
        }

    # Basic summary: first row highlights
    summary = ""
    try:
        if len(rows) == 1:
            summary = "Returned 1 row."
        else:
            summary = f"Returned {len(rows)} rows."
        if x_key and y_key:
            summary += f" Key fields: {x_key}, {y_key}."
    except Exception:
        summary = ""

    if not AI_INSIGHTS_LLM:
        return {"insights_summary": summary, "chart_spec": chart}

    # Optional LLM refinement (best-effort)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a data analyst. Given a user question and a small sample of query results, "
             "produce a short executive summary AND suggest a simple chart.\n"
             "Return valid JSON ONLY with keys: summary, chart_type, x_key, y_key, title.\n"
             "chart_type must be one of: bar, line, none.\n"
             "Choose x_key/y_key from the provided columns. If no good chart, chart_type=none."),
            ("human",
             "Question:\n{q}\n\nColumns:\n{cols}\n\nSample rows (JSON):\n{rows}")
        ]
    )
    try:
        msg = prompt.format_messages(q=question, cols=", ".join(cols), rows=json.dumps(sample, default=str))
        raw = str(llm.invoke(msg).content).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        data = json.loads(raw)
        spec = ChartSpec(**data)
        chart2 = None
        if spec.chart_type != "none" and spec.x_key and spec.y_key:
            chart2 = {"type": spec.chart_type, "x_key": spec.x_key, "y_key": spec.y_key, "title": spec.title or ""}
        return {"insights_summary": spec.summary, "chart_spec": chart2}
    except Exception:
        return {"insights_summary": summary, "chart_spec": chart}

# ------------------------------------------------------------
# SQL repair loop
# ------------------------------------------------------------
def _repair_sql(question: str, table_info: str, dialect: str, mode: str, bad_sql: str, error_msg: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"You are a {dialect} SQL expert. Fix the SQL query.\n"
             "Return ONLY the corrected SQL. No markdown. No explanation.\n"
             "Rules: obey mode constraints; never generate DDL; in public mode, SELECT-only and single statement."),
            ("human",
             "Question:\n{q}\n\nMode: {mode}\n\nTable info:\n{ti}\n\nBad SQL:\n{sql}\n\nError:\n{err}\n\nReturn fixed SQL only.")
        ]
    )
    msg = prompt.format_messages(q=question, mode=mode, ti=table_info, sql=bad_sql, err=error_msg)
    fixed = llm.invoke(msg).content
    return clean_sql_query(str(fixed))

def _rewrite_sql_for_budget(question: str, table_info: str, dialect: str, mode: str, sql: str, explain: Dict[str, Any]) -> str:
    """Ask the model to rewrite SQL to fit a query budget (fewer scans / earlier LIMIT / narrower window)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"You are a {dialect} SQL performance engineer. Rewrite the SQL to reduce execution cost and rows scanned.\n"
             "Return ONLY the SQL. No markdown, no commentary.\n\n"
             "Constraints (STRICT):\n"
             "- Obey mode rules (public = SELECT-only, single statement; sandbox = DML only if explicitly requested).\n"
             "- Preserve the question intent.\n"
             "- Prefer: aggregate first, then join; apply ORDER BY + LIMIT inside a subquery/CTE before joins; avoid SELECT *.\n"
             "- If the question is analytics over time and a date-like column exists, add a reasonable default time window (e.g., last 90 days) unless user asked all-time.\n"),
            ("human",
             "Question:\n{q}\n\nMode: {mode}\n\nTable info:\n{ti}\n\nCurrent SQL:\n{sql}\n\nBudget signal (EXPLAIN):\n{ex}\n\nReturn cheaper SQL only."),
        ]
    )
    msg = prompt.format_messages(q=question, mode=mode, ti=table_info, sql=sql, ex=json.dumps(explain))
    out = llm.invoke(msg).content
    return clean_sql_query(str(out))

def _rewrite_sql_for_speed(question: str, table_info: str, dialect: str, mode: str, sql: str, db_ms: int) -> str:
    """Rewrite SQL when it is slow in practice (runtime-based optimization)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"You are a {dialect} SQL performance engineer. Rewrite the SQL to run faster.\n"
             "Return ONLY the SQL. No markdown, no commentary.\n\n"
             "Rules:\n"
             "- Public mode: SELECT-only, single statement.\n"
             "- Add LIMITs early, reduce joins, aggregate before joining dimensions.\n"
             "- Prefer indexed columns in joins/filters when possible (use obvious keys).\n"
             "- Keep output small: LIMIT {top_k} unless user asked otherwise.\n"),
            ("human",
             "Question:\n{q}\n\nMode: {mode}\n\nTable info:\n{ti}\n\nSlow SQL (~{ms}ms):\n{sql}\n\nReturn faster equivalent SQL only."),
        ]
    )
    msg = prompt.format_messages(q=question, mode=mode, ti=table_info, ms=str(db_ms), sql=sql, top_k=str(TOP_K_DEFAULT))
    out = llm.invoke(msg).content
    return clean_sql_query(str(out))

# ------------------------------------------------------------
# Chain build + cache
# ------------------------------------------------------------
_chain_cache: Dict[str, Any] = {}

# Public-response cache (question -> full response). This is intentionally
# lightweight; for production you can swap to Redis.
_response_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

def _cache_key(db_url: str, schema_source: str, dialect: str, mode: str, question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    sig = f"{db_url}::{schema_source}::{dialect}::{mode}::{q}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


# Redis-backed cache (optional). Falls back to in-memory LRU.
def _redis_cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not get_redis():
        return None
    return rget_json(f"resp_cache:{key}")

def _redis_cache_set(key: str, value: Dict[str, Any]) -> None:
    if not get_redis() or CACHE_TTL_SECONDS <= 0:
        return
    rset_json(f"resp_cache:{key}", value, ttl_s=CACHE_TTL_SECONDS)

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    # Redis-first (shared across workers)
    ritem = _redis_cache_get(key)
    if isinstance(ritem, dict) and ritem.get('value') is not None:
        return ritem.get('value')

    now = time.time()
    item = _response_cache.get(key)
    if not item:
        return None
    if item.get("expires_at", 0) <= now:
        _response_cache.pop(key, None)
        return None
    # refresh LRU
    _response_cache.move_to_end(key)
    return item.get("value")

def _cache_set(key: str, value: Dict[str, Any]) -> None:
    # Redis (best-effort)
    _redis_cache_set(key, {"value": value})

    if CACHE_TTL_SECONDS <= 0 or CACHE_MAX_ENTRIES <= 0:
        return
    now = time.time()
    _response_cache[key] = {"expires_at": now + CACHE_TTL_SECONDS, "value": value}
    _response_cache.move_to_end(key)
    # trim
    while len(_response_cache) > CACHE_MAX_ENTRIES:
        _response_cache.popitem(last=False)


_schema_ctx_cache: Dict[str, Any] = {}
SCHEMA_CTX_TTL_SECONDS = int(os.getenv("SCHEMA_CTX_TTL_SECONDS", "600"))

def build_chain_for_db(db_: SQLDatabase, dialect: str, cache_ns: str) -> Any:
    final_prompt = make_final_prompt(dialect)
    generate_query = create_sql_query_chain(llm, db_, final_prompt)

    def _run_exec(inputs: dict) -> dict:
        sql = inputs.get("query", "")
        mode = (inputs.get("mode", "public") or "public").lower().strip()
        q = inputs.get("question", "")
        table_info = inputs.get("table_details", "")

        engine = db_._engine

        def _exec_once(sql_to_run: str) -> Dict[str, Any]:
            t0 = time.time()
            out = execute_with_guardrails(db_, sql_to_run, mode, cache_ns=cache_ns)
            out["db_ms"] = int((time.time() - t0) * 1000)
            return out

        # -------------------------
        # Cost-aware preflight (public SELECT)
        # -------------------------
        explain = None
        base_sql = sql
        candidate_sql = sql
        optimized = False
        optimization_reason = ""

        try:
            if PREFLIGHT_EXPLAIN and mode == "public" and _sql_kind(candidate_sql) == "SELECT":
                with engine.connect() as conn:
                    _apply_timeout(conn, dialect, min(DB_TIMEOUT_MS, 4000))
                    explain = _explain_preflight(conn, dialect, candidate_sql)
                if _is_over_budget(explain):
                    candidate_sql = _rewrite_sql_for_budget(q, table_info, dialect, mode, candidate_sql, explain or {})
                    optimized = True
                    optimization_reason = "over_budget"
        except Exception:
            # best-effort only; never block execution if EXPLAIN fails
            explain = None

        # -------------------------
        # Execute with error-repair loop
        # -------------------------
        last_err = None
        attempt_sql = candidate_sql
        used_repair = 0

        for attempt in range(SQL_FIX_RETRIES + 1):
            try:
                exec_info = _exec_once(attempt_sql)
                exec_info["explain"] = explain
                exec_info["optimized"] = optimized
                exec_info["optimization_reason"] = optimization_reason
                exec_info["sql_fix_retries_used"] = used_repair

                # -------------------------
                # Runtime-based optimization (public SELECT)
                # -------------------------
                if mode == "public" and exec_info.get("kind") == "SELECT" and exec_info.get("db_ms", 0) >= SLOW_QUERY_MS:
                    try:
                        faster_sql = _rewrite_sql_for_speed(q, table_info, dialect, mode, exec_info.get("sql", attempt_sql), int(exec_info["db_ms"]))
                        if faster_sql and faster_sql.strip().lower() != exec_info.get("sql", "").strip().lower():
                            faster_exec = _exec_once(faster_sql)
                            # Keep the faster execution only
                            if int(faster_exec.get("db_ms", 10**9)) < int(exec_info.get("db_ms", 10**9)):
                                faster_exec["explain"] = explain
                                faster_exec["optimized"] = True
                                faster_exec["optimization_reason"] = "slow_query"
                                faster_exec["sql_fix_retries_used"] = used_repair
                                exec_info = faster_exec
                    except Exception:
                        pass

                return exec_info

            except Exception as e:
                last_err = e
                # Do not attempt repairs for sandbox DML integrity errors
                if mode == "sandbox" and _sql_kind(attempt_sql) == "DML":
                    raise

                # If we tried an optimized SQL and it failed, fall back to the base SQL once
                if attempt == 0 and candidate_sql != base_sql:
                    attempt_sql = base_sql
                    continue

                if attempt >= SQL_FIX_RETRIES:
                    break

                used_repair += 1
                attempt_sql = _repair_sql(q, table_info, dialect, mode, attempt_sql, str(e))

        raise ValueError(f"SQL execution failed after retries: {last_err}")

    chain = (
        RunnablePassthrough.assign(table_names_to_use=select_table)
        | RunnablePassthrough.assign(query=generate_query | RunnableLambda(clean_sql_query))
        | RunnablePassthrough.assign(exec_info=RunnableLambda(_run_exec))
        | RunnablePassthrough.assign(
            result=lambda x: x["exec_info"]["preview"],
            mode=lambda x: x.get("mode", "public"),
            query=lambda x: x["exec_info"]["sql"],
        )
        | RunnablePassthrough.assign(answer=rephrase_answer)
        | RunnableLambda(lambda x: {
            "answer": x.get("answer", ""),
            "sql": x["exec_info"]["sql"],
            "mode": x["exec_info"]["mode"],
            "kind": x["exec_info"]["kind"],
            "rolled_back": x["exec_info"]["rolled_back"],
            "db_ms": int(x["exec_info"].get("db_ms", 0) or 0),
            "optimized": bool(x["exec_info"].get("optimized", False)),
            "optimization_reason": x["exec_info"].get("optimization_reason", ""),
            "sql_fix_retries_used": int(x["exec_info"].get("sql_fix_retries_used", 0) or 0),
            "explain": x["exec_info"].get("explain", None),
            "rows_returned": (x["exec_info"]["rowcount"] if x["exec_info"]["kind"] == "SELECT" else 0),
            "rows_affected": (x["exec_info"]["rowcount"] if x["exec_info"]["kind"] == "DML" else 0),
            "rows_preview": (x["exec_info"]["rows"][:PREVIEW_ROW_CAP] if x["exec_info"]["kind"] == "SELECT" else []),
            "columns": (list(x["exec_info"]["rows"][0].keys()) if (x["exec_info"]["kind"] == "SELECT" and x["exec_info"]["rows"]) else []),
        })
    )
    return chain

def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def get_or_build_chain(db_url: str, schema_csv_text: Optional[str]) -> Dict[str, Any]:
    """
    Returns dict with:
      chain, db, table_details, tables, schema_source, dialect, host, warnings
    Uses a short-lived schema context cache to avoid re-introspecting on every request.
    """
    # NOTE: `schema_csv_text` is the only schema override input.
    # Previous iterations referenced `schema_csv_override` here, ...
    # caused a NameError when no CSV was provided.

    cache_sig = f"{db_url}::{_hash_text(schema_csv_text or '')}"
    now = time.time()
    cached = _schema_ctx_cache.get(cache_sig)
    if cached and cached.get("expires_at", 0) > now:
        ctx = cached["ctx"]
    else:
        ctx = build_schema_context(db_url, schema_csv_text)
        _schema_ctx_cache[cache_sig] = {"ctx": ctx, "expires_at": now + SCHEMA_CTX_TTL_SECONDS}

    db_ = ctx["db"]
    table_details = ctx["table_details"]
    dialect = ctx["dialect"]
    schema_source = ctx["schema_source"]
    warnings = ctx["warnings"]
    host = ctx["host"]

    # Chain cache key is stable for a given DB + schema context.
    # NOTE: Schema-RAG selection is applied per-question inside chain_code,
    # not during bootstrap (reduces cold-start and avoids embed failures at import time).
    table_details_used = table_details
    cache_key = f"{db_url}::{_hash_text(table_details)}::{dialect}::{int(SCHEMA_RAG_ENABLED)}"

    if cache_key not in _chain_cache:
        cache_ns = hashlib.sha256(db_url.encode('utf-8')).hexdigest()[:12]
        _chain_cache[cache_key] = build_chain_for_db(db_, dialect, cache_ns)

    return {
        "chain": _chain_cache[cache_key],
        "db": db_,
        "table_details": table_details,
        "table_details_used": table_details_used,
        "tables": ctx["tables"],
        "schema_source": schema_source,
        "dialect": dialect,
        "host": host,
        "warnings": warnings,
    }
# ------------------------------------------------------------
# Default (demo) DB from env
# ------------------------------------------------------------
DEMO_DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DEMO_DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Provide a demo DB URL (e.g., Supabase Postgres).")

_demo_ctx = get_or_build_chain(DEMO_DATABASE_URL, schema_csv_text=(Path(__file__).resolve().parent / "database_table_descriptions.csv").read_text(encoding="utf-8", errors="ignore") if (Path(__file__).resolve().parent / "database_table_descriptions.csv").exists() else None)
_demo_chain = _demo_ctx["chain"]
_demo_table_details = _demo_ctx["table_details"]

# ------------------------------------------------------------
# Public API used by code1.py
# ------------------------------------------------------------
def get_schema_tables(db_url_override: Optional[str], schema_csv_override: Optional[str]) -> Dict[str, Any]:
    """
    Returns tables + schema_source + dialect + host for either demo or BYODB.
    """
    if db_url_override:
        # Use the resolved schema override (if any). This keeps BYODB working
        # when the caller supplies schema text or when it is omitted (auto-introspection).
        ctx = get_or_build_chain(db_url_override, schema_csv_text)
        return {
            "tables": ctx["tables"],
            "schema_source": ctx["schema_source"],
            "dialect": ctx["dialect"],
            "host": ctx["host"],
            "warnings": ctx["warnings"],
        }
    # demo
    return {
        "tables": _demo_ctx["tables"],
        "schema_source": _demo_ctx["schema_source"],
        "dialect": _demo_ctx["dialect"],
        "host": _demo_ctx["host"],
        "warnings": [],
    }

def chain_code(
    q: str,
    m: Optional[List[Dict[str, str]]] = None,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    mode: str = "public",
    db_url_override: Optional[str] = None,
    schema_csv_text: Optional[str] = None,
    schema_csv_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main entrypoint. 'm' kept for compatibility (frontend history), not injected into SQL prompt.
    Returns dict including answer/sql/preview + insights summary/chart_spec.
    """
    # Accept either positional 'm' or keyword 'messages'
    if messages is not None and m is None:
        m = messages
    if m is None:
        m = []

    # Prefer schema_csv_text if provided
    if schema_csv_text is None:
        schema_csv_text = schema_csv_override

    top_k = TOP_K_DEFAULT
    mode = (mode or "public").lower().strip()

    # -------------------------
    # Fast path cache (public, SELECT-only)
    # -------------------------
    mode = (mode or "public").lower().strip()
    q_norm = (q or "").strip()

    # Determine context (demo vs BYODB) for cache key
    if db_url_override:
        # Use resolved schema override (if any). If omitted, schema will be auto-introspected.
        ctx = get_or_build_chain(db_url_override, schema_csv_text)
        cache_db_url = db_url_override
        cache_schema_source = ctx.get("schema_source", "auto")
        cache_dialect = ctx.get("dialect", "unknown")
        if mode == "public":
            ck = _cache_key(cache_db_url, cache_schema_source, cache_dialect, mode, q_norm)
            cached = _cache_get(ck)
            if cached:
                return cached
        table_details = ctx["table_details"]
        table_details_used = ctx["table_details_used"]
        ch = ctx["chain"]
        # Apply schema rag per question if enabled
        if SCHEMA_RAG_ENABLED:
            try:
                table_details_used = _schema_rag_select(q, table_details, f"{db_url_override}::{_hash_text(table_details)}::{ctx['dialect']}")
            except Exception:
                table_details_used = table_details
        out = ch.invoke({
            "question": q,
            "table_details": table_details_used,
            "mode": mode,
            "top_k": top_k,
        })
        # attach warnings/schema metadata
        if ctx["warnings"]:
            out["warnings"] = ctx["warnings"]
        out["schema_source"] = ctx["schema_source"]
        out["dialect"] = ctx["dialect"]
        out["host"] = ctx["host"]
    else:
        cache_db_url = DEMO_DATABASE_URL
        cache_schema_source = _demo_ctx.get("schema_source", "csv")
        cache_dialect = _demo_ctx.get("dialect", "demo")
        if mode == "public":
            ck = _cache_key(cache_db_url, cache_schema_source, cache_dialect, mode, q_norm)
            cached = _cache_get(ck)
            if cached:
                return cached

        out = _demo_chain.invoke({
            "question": q,
            "table_details": _demo_table_details,
            "mode": mode,
            "top_k": top_k,
        })
        out["schema_source"] = _demo_ctx["schema_source"]
        out["dialect"] = _demo_ctx["dialect"]
        out["host"] = _demo_ctx["host"]

    # Add insights + chart for SELECT results
    if out.get("kind") == "SELECT":
        extra = _infer_chart_and_summary(q, out.get("rows_preview") or [])
        out.update(extra)

    # Write to cache for public mode (safe)
    try:
        if mode == "public":
            ck = _cache_key(cache_db_url, out.get("schema_source", cache_schema_source), out.get("dialect", cache_dialect), mode, q_norm)
            _cache_set(ck, out)
    except Exception:
        pass

    return out
