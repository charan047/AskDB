import os
import re
import io
import csv
import time
import hashlib
from pathlib import Path
from operator import itemgetter
from typing import List, Tuple, Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import text, inspect
from sqlalchemy.exc import IntegrityError

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from pydantic import BaseModel, Field

load_dotenv()

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-2.5-pro
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# ------------------------------------------------------------
# Default DB (V1 demo)
# ------------------------------------------------------------
DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL")
if not DEFAULT_DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Example: postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"
    )

# ------------------------------------------------------------
# Helpers: SQL cleanup + guardrails
# ------------------------------------------------------------
def clean_sql_query(text_: str) -> str:
    text_ = text_ or ""

    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text_ = re.sub(block_pattern, r"\1", text_, flags=re.DOTALL)

    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text_ = re.sub(prefix_pattern, "", text_, flags=re.IGNORECASE)

    text_ = re.sub(r'`([^`]*)`', r"\1", text_)  # remove backticks
    text_ = re.sub(r"\s+", " ", text_).strip()

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
    s = (sql or "").strip()
    if not s:
        return []

    stmts = []
    buf = []
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


def execute_with_guardrails(db_: SQLDatabase, sql: str, mode: str) -> dict:
    """
    public: SELECT-only + LIMIT (single statement only)
    sandbox: SELECT + DML; DML always rolled back
             - allows multi-statement DML batches in sandbox only
    blocks: DDL always
    """
    mode = (mode or "public").lower().strip()
    if mode not in ("public", "sandbox"):
        raise ValueError("Invalid mode. Use 'public' or 'sandbox'.")

    statements = _split_sql_statements(sql)
    if not statements:
        raise ValueError("Empty SQL is not allowed.")

    if mode == "public" and len(statements) > 1:
        raise ValueError("Multi-statement SQL is not allowed in public mode.")

    kinds = [_sql_kind(s) for s in statements]

    if any(k == "DDL" for k in kinds):
        raise ValueError("DDL statements are blocked (CREATE/ALTER/DROP/TRUNCATE/GRANT/REVOKE).")

    if mode == "public" and kinds[0] != "SELECT":
        raise ValueError("Public mode supports SELECT queries only. Use mode='sandbox' to simulate DML (rolled back).")

    engine = db_._engine
    rolled_back = False

    # Sandbox: multi-statement DML batch
    if mode == "sandbox" and len(statements) > 1:
        if any(k != "DML" for k in kinds):
            raise ValueError("In sandbox mode, multi-statement is allowed only for pure DML (INSERT/UPDATE/DELETE).")

        conn = engine.connect()
        trans = conn.begin()
        try:
            try:
                conn.execute(text("SET statement_timeout = 8000"))
            except Exception:
                pass

            total_affected = 0
            for stmt in statements:
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
                    if "ForeignKeyViolation" in msg or "violates foreign key constraint" in msg:
                        raise ValueError(
                            "This change is blocked by foreign-key relationships. "
                            "Try deleting dependent rows first, or prefer a soft delete. "
                            "Note: sandbox changes are simulated only and always rolled back."
                        ) from e
                    raise ValueError(f"Sandbox DML failed due to integrity constraint: {msg}") from e

                affected = int(res.rowcount) if res.rowcount is not None else 0
                total_affected += affected

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
                "preview": f"DML batch simulated (rolled back). Statements: {len(statements)}. Rows affected: {total_affected}.",
            }
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Sandbox: single DML
    if mode == "sandbox" and kinds[0] == "DML":
        final_sql = _strip_sql(statements[0])

        conn = engine.connect()
        trans = conn.begin()
        try:
            try:
                conn.execute(text("SET statement_timeout = 8000"))
            except Exception:
                pass

            try:
                res = conn.execute(text(final_sql))
            except IntegrityError as e:
                try:
                    trans.rollback()
                except Exception:
                    pass
                rolled_back = True
                msg = str(getattr(e, "orig", e))
                if "ForeignKeyViolation" in msg or "violates foreign key constraint" in msg:
                    raise ValueError(
                        "This change is blocked by foreign-key relationships. "
                        "Try deleting dependent rows first, or prefer a soft delete. "
                        "Note: sandbox changes are simulated only and always rolled back."
                    ) from e
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

    # SELECT path (public/sandbox) single statement
    if kinds[0] != "SELECT":
        raise ValueError("Only SELECT is allowed here. Use sandbox mode for DML simulation.")

    final_sql = _strip_sql(statements[0])
    final_sql = _ensure_limit(final_sql, limit=100)

    with engine.connect() as conn:
        try:
            conn.execute(text("SET statement_timeout = 8000"))
        except Exception:
            pass

        res = conn.execute(text(final_sql))
        cols = list(res.keys())
        fetched = res.fetchmany(100)
        rows = [dict(zip(cols, r)) for r in fetched]
        rowcount = len(rows)

        return {
            "sql": final_sql,
            "kind": "SELECT",
            "mode": mode,
            "rolled_back": False,
            "rowcount": rowcount,
            "rows": rows,
            "preview": f"Rows returned: {rowcount}. Preview: {rows[:5]}",
        }


# ------------------------------------------------------------
# Answer prompt
# ------------------------------------------------------------
answer_prompt = PromptTemplate.from_template(
    """You are answering questions over a SQL database.

Mode: {mode}
Note: In sandbox mode, DML changes are always rolled back (no permanent changes).

Question: {question}
SQL Query: {query}
SQL Result: {result}

Answer the user question clearly and concisely."""
)
rephrase_answer = answer_prompt | llm | StrOutputParser()

# ------------------------------------------------------------
# Few-shot examples (your updated complex set)
# ------------------------------------------------------------
examples = [
    # PUBLIC
    {
        "input": "For the last 90 days, show monthly revenue (sum of payments) and the month-over-month % change.",
        "query": (
            "SELECT date_trunc('month', paymentdate) AS month, "
            "SUM(amount) AS revenue, "
            "ROUND(100.0 * (SUM(amount) - LAG(SUM(amount)) OVER (ORDER BY date_trunc('month', paymentdate))) / "
            "NULLIF(LAG(SUM(amount)) OVER (ORDER BY date_trunc('month', paymentdate)), 0), 2) AS mom_pct "
            "FROM payments "
            "WHERE paymentdate >= (CURRENT_DATE - INTERVAL '90 days') "
            "GROUP BY 1 "
            "ORDER BY 1;"
        ),
    },
    {
        "input": "Who are the top 10 customers by total payments, and what percent of total revenue do they contribute?",
        "query": (
            "WITH totals AS ("
            "  SELECT customernumber, SUM(amount) AS total_paid "
            "  FROM payments "
            "  GROUP BY customernumber"
            "), grand AS ("
            "  SELECT SUM(total_paid) AS grand_total FROM totals"
            ") "
            "SELECT c.customername, t.total_paid, "
            "ROUND(100.0 * t.total_paid / NULLIF(g.grand_total, 0), 2) AS revenue_share_pct "
            "FROM totals t "
            "JOIN customers c ON c.customernumber = t.customernumber "
            "CROSS JOIN grand g "
            "ORDER BY t.total_paid DESC "
            "LIMIT 10;"
        ),
    },
    {
        "input": "List customers who have placed orders but have never made a payment, including their total order count.",
        "query": (
            "SELECT c.customername, COUNT(o.ordernumber) AS order_count "
            "FROM customers c "
            "JOIN orders o ON o.customernumber = c.customernumber "
            "LEFT JOIN payments p ON p.customernumber = c.customernumber "
            "WHERE p.customernumber IS NULL "
            "GROUP BY c.customername "
            "ORDER BY order_count DESC;"
        ),
    },
    {
        "input": "Which products are frequently ordered together? Show top 10 product pairs by co-occurrence count.",
        "query": (
            "SELECT p1.productname AS product_a, p2.productname AS product_b, COUNT(*) AS co_occurrences "
            "FROM orderdetails od1 "
            "JOIN orderdetails od2 "
            "  ON od1.ordernumber = od2.ordernumber "
            " AND od1.productcode < od2.productcode "
            "JOIN products p1 ON p1.productcode = od1.productcode "
            "JOIN products p2 ON p2.productcode = od2.productcode "
            "GROUP BY product_a, product_b "
            "ORDER BY co_occurrences DESC "
            "LIMIT 10;"
        ),
    },

    # SANDBOX
    {
        "input": "In sandbox mode, increase the credit limit by 15% for all customers in France with creditlimit under 30000.",
        "query": (
            "UPDATE customers "
            "SET creditlimit = ROUND(creditlimit * 1.15, 2) "
            "WHERE country = 'France' AND creditlimit < 30000;"
        ),
    },
    {
        "input": "In sandbox mode, mark all 'In Process' orders for customer 112 as 'Cancelled'.",
        "query": (
            "UPDATE orders "
            "SET status = 'Cancelled' "
            "WHERE customernumber = 112 AND status = 'In Process';"
        ),
    },
    {
        "input": "In sandbox mode, delete order 10100 safely (delete its orderdetails first, then the order).",
        "query": (
            "DELETE FROM orderdetails WHERE ordernumber = 10100; "
            "DELETE FROM orders WHERE ordernumber = 10100;"
        ),
    },
    {
        "input": "In sandbox mode, delete customer 112 safely by removing dependent rows first (orderdetails → orders → payments → customer).",
        "query": (
            "DELETE FROM orderdetails WHERE ordernumber IN (SELECT ordernumber FROM orders WHERE customernumber = 112); "
            "DELETE FROM orders WHERE customernumber = 112; "
            "DELETE FROM payments WHERE customernumber = 112; "
            "DELETE FROM customers WHERE customernumber = 112;"
        ),
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

def build_few_shot_prompt():
    static_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        input_variables=["input"],
    )

    if os.getenv("USE_SEMANTIC_EXAMPLES", "0") != "1":
        return static_prompt

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")

        selector = SemanticSimilarityExampleSelector.from_examples(
            examples=examples,
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
# Schema input (CSV) parsing + auto fallback (A)
# ------------------------------------------------------------
_TABLE_COL_CANDIDATES = {
    "table_name", "tablename", "table", "name", "table name", "table_name ", "table-name"
}
_DESC_COL_CANDIDATES = {
    "description", "desc", "details", "summary", "about", "table_description", "table description"
}

def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip().lower())

def _guess_delimiter(header_line: str) -> str:
    # lightweight delimiter guess
    candidates = [",", "\t", ";", "|"]
    best = ","
    best_count = -1
    for d in candidates:
        c = header_line.count(d)
        if c > best_count:
            best_count = c
            best = d
    return best

def parse_schema_csv_text(schema_csv_text: str) -> Tuple[Optional[str], List[str]]:
    """
    Try to parse CSV text into the "table_details" string used by the table selection prompt.
    Accepts common header variations. If parsing fails, returns (None, warnings).
    """
    warnings = []
    txt = (schema_csv_text or "").strip()
    if not txt:
        return None, ["schema_csv is empty"]

    # allow accidental BOM
    txt = txt.lstrip("\ufeff")

    lines = txt.splitlines()
    if not lines:
        return None, ["schema_csv has no lines"]

    delim = _guess_delimiter(lines[0])
    reader = csv.DictReader(io.StringIO(txt), delimiter=delim)

    if not reader.fieldnames:
        return None, ["Could not detect CSV headers"]

    # map headers
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
        return None, [f"schema_csv missing table name column. Found headers: {reader.fieldnames}"]

    if desc_key is None:
        warnings.append("schema_csv missing description column; using empty descriptions.")
        desc_key = table_key  # placeholder; we'll handle it below

    out = []
    row_count = 0
    for row in reader:
        row_count += 1
        tname = (row.get(table_key) or "").strip()
        if not tname:
            continue

        if desc_key == table_key:
            desc = ""
        else:
            desc = (row.get(desc_key) or "").strip()

        out.append(f"Table Name:{tname}\nTable Description:{desc}\n")

    if not out:
        return None, ["schema_csv parsed but produced no table rows"]

    if row_count == 0:
        warnings.append("schema_csv contains headers but no rows")

    return "\n".join(out).strip() + "\n", warnings

def auto_schema_from_db(db_: SQLDatabase, max_tables: int = 80, max_cols_per_table: int = 18) -> str:
    """
    Auto-generate table details from DB inspection (fallback when CSV is invalid).
    Keeps it compact; enough for table selection + SQL generation.
    """
    engine = db_._engine
    insp = inspect(engine)

    # prefer public schema, but handle other DBs gracefully
    try:
        tables = insp.get_table_names(schema="public")
        if not tables:
            tables = insp.get_table_names()
    except Exception:
        tables = insp.get_table_names()

    tables = sorted(tables)[:max_tables]

    parts = []
    for t in tables:
        try:
            cols = insp.get_columns(t, schema="public")
            if not cols:
                cols = insp.get_columns(t)
        except Exception:
            cols = []

        col_parts = []
        for c in cols[:max_cols_per_table]:
            cname = c.get("name", "")
            ctype = str(c.get("type", ""))
            col_parts.append(f"{cname} ({ctype})")

        desc = "Columns: " + (", ".join(col_parts) if col_parts else "(unknown)")
        parts.append(f"Table Name:{t}\nTable Description:{desc}\n")

    return "\n".join(parts).strip() + "\n"

def load_demo_table_details_from_file() -> str:
    default_csv = Path(__file__).resolve().parent / "database_table_descriptions.csv"
    csv_path = Path(os.getenv("TABLE_DESCRIPTIONS_PATH", str(default_csv)))

    if not csv_path.exists():
        raise FileNotFoundError(
            f"database_table_descriptions.csv not found at: {csv_path}\n"
            f"Fix: Put the file at {default_csv} OR set TABLE_DESCRIPTIONS_PATH in .env"
        )

    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    parsed, warnings = parse_schema_csv_text(txt)
    if parsed:
        if warnings:
            print("[AskDB] demo schema CSV warnings:", warnings)
        return parsed

    # If demo CSV is somehow invalid, fallback to DB inspection for default DB
    demo_db = SQLDatabase.from_uri(DEFAULT_DATABASE_URL)
    return auto_schema_from_db(demo_db)

# ------------------------------------------------------------
# Table selection chain (uses table_details text)
# ------------------------------------------------------------
class Table(BaseModel):
    name: List[str] = Field(description="List of table names in the SQL database.")

table_details_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Return the names of ALL the SQL tables that MIGHT be relevant to the user question.
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""),
        ("human", "{question}")
    ]
)

structured_llm = llm.with_structured_output(Table)
table_chain = table_details_prompt | structured_llm

def get_tables(table_response: Table) -> List[str]:
    return table_response.name

select_table = {"question": itemgetter("question"), "table_details": itemgetter("table_details")} | table_chain | get_tables

# ------------------------------------------------------------
# SQL generation prompt (mode-aware)
# ------------------------------------------------------------
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a PostgreSQL expert. Your job is to output ONE thing: a valid PostgreSQL SQL query.\n"
            "Return ONLY the SQL. No explanations. No markdown. No backticks.\n\n"
            "Context:\n"
            "- mode: {mode}\n"
            "- top_k: {top_k}\n\n"
            "Rules (STRICT):\n"
            "1) Public mode:\n"
            "   - ONLY generate SELECT (or WITH ... SELECT).\n"
            "   - Must be a SINGLE statement.\n"
            "2) Sandbox mode:\n"
            "   - Generate DML (INSERT/UPDATE/DELETE) ONLY if the user explicitly asks to change data.\n"
            "   - Multi-statement is allowed ONLY for FK-safe DML batches (e.g., delete children then parent).\n"
            "3) Always BLOCK and never generate:\n"
            "   - DDL (CREATE/ALTER/DROP/TRUNCATE)\n"
            "   - Permission statements (GRANT/REVOKE)\n"
            "   - Comments or multiple unrelated queries\n\n"
            "Query quality rules:\n"
            "- Prefer explicit column lists; avoid SELECT * unless the user explicitly requests all columns.\n"
            "- For SELECT queries: ALWAYS apply LIMIT {top_k} unless the user explicitly asks for all rows.\n"
            "- If the question is ambiguous, choose a safe, reasonable interpretation (do not ask questions).\n"
            "- Use schema/table/column names exactly as provided in table_info.\n"
            "- When joins are needed, use correct join keys; avoid cartesian products.\n"
            "- Prefer aggregation over returning large raw datasets.\n\n"
            "Write-operation guidance (sandbox DML):\n"
            "- Prefer soft delete (UPDATE a status/flag column) instead of deleting parent rows when relationships exist.\n"
            "- If DELETE is required, respect foreign keys and delete children first.\n\n"
            "Here is the relevant table info:\n"
            "{table_info}\n\n"
            "Below are examples of questions and their corresponding SQL queries. Use them as guidance."
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# ------------------------------------------------------------
# Chain factory + caching (V2 BYODB)
# ------------------------------------------------------------
_CHAIN_CACHE: Dict[str, Any] = {}
_CACHE_MAX = int(os.getenv("CHAIN_CACHE_MAX", "25"))

def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

def _cache_key(db_url: str, table_details: str) -> str:
    return f"{db_url}::{_hash_text(table_details)}"

def build_chain_for_db(db_: SQLDatabase) -> Any:
    """
    Builds the runnable chain bound to a specific SQLDatabase instance.
    Table selection uses table_details passed at invoke-time.
    """
    generate_query = create_sql_query_chain(llm, db_, final_prompt)

    def _run_exec(inputs: dict) -> dict:
        sql = inputs.get("query", "")
        mode = inputs.get("mode", "public")
        return execute_with_guardrails(db_, sql, mode)

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
            "rows_returned": (x["exec_info"]["rowcount"] if x["exec_info"]["kind"] == "SELECT" else 0),
            "rows_affected": (x["exec_info"]["rowcount"] if x["exec_info"]["kind"] == "DML" else 0),
            "rows_preview": (x["exec_info"]["rows"][:20] if x["exec_info"]["kind"] == "SELECT" else []),
            "columns": (list(x["exec_info"]["rows"][0].keys()) if (x["exec_info"]["kind"] == "SELECT" and x["exec_info"]["rows"]) else []),
        })
    )
    return chain

def get_or_build_chain(db_url: str, schema_csv_text: Optional[str]) -> Tuple[Any, str, List[str]]:
    """
    Returns: (chain, table_details, warnings)
    - If schema_csv_text valid: use it
    - Else: auto schema from DB inspection (fallback A)
    Caches chain by (db_url + schema hash)
    """
    warnings: List[str] = []

    db_ = SQLDatabase.from_uri(db_url)

    table_details: Optional[str] = None
    if schema_csv_text is not None:
        parsed, w = parse_schema_csv_text(schema_csv_text)
        warnings.extend(w)
        table_details = parsed

    if not table_details:
        # fallback A: auto-generate schema from DB
        table_details = auto_schema_from_db(db_)
        warnings.append("Schema source: auto-generated from database (CSV missing/invalid).")

    key = _cache_key(db_url, table_details)
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]["chain"], table_details, warnings

    # prune cache if too big
    if len(_CHAIN_CACHE) >= _CACHE_MAX:
        # naive eviction: remove oldest inserted
        oldest = sorted(_CHAIN_CACHE.items(), key=lambda kv: kv[1]["ts"])[0][0]
        del _CHAIN_CACHE[oldest]

    ch = build_chain_for_db(db_)
    _CHAIN_CACHE[key] = {"chain": ch, "ts": time.time()}
    return ch, table_details, warnings

# Build default chain + default table details once
_DEFAULT_TABLE_DETAILS = load_demo_table_details_from_file()
_DEFAULT_DB = SQLDatabase.from_uri(DEFAULT_DATABASE_URL)
_DEFAULT_CHAIN = build_chain_for_db(_DEFAULT_DB)

# ------------------------------------------------------------
# Public API: chain_code (now BYODB-aware)
# ------------------------------------------------------------
def chain_code(q, m, mode="public", db_url_override=None, schema_csv_override=None):
    """
    V1 (default): uses demo DB + demo table descriptions.
    V2 BYODB: uses db_url_override + schema_csv_override
      - if schema csv invalid -> auto schema from DB (fallback A)
    """
    top_k = 10

    if db_url_override:
        ch, table_details, warnings = get_or_build_chain(db_url_override, schema_csv_override)
        out = ch.invoke({
            "question": q,
            "table_details": table_details,
            "mode": mode,
            "top_k": top_k,
        })
        # include warnings (frontend can ignore)
        if warnings:
            out["warnings"] = warnings
        return out

    # default (Supabase demo)
    return _DEFAULT_CHAIN.invoke({
        "question": q,
        "table_details": _DEFAULT_TABLE_DETAILS,
        "mode": mode,
        "top_k": top_k,
    })
