import os
import re
from pathlib import Path
from operator import itemgetter
from typing import List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from dotenv import load_dotenv

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

# ---- Configuration (env-driven) ----
db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise RuntimeError(
        "DATABASE_URL is not set. Example: postgresql+psycopg2://user:pass@host:5432/db?sslmode=require"
    )
db = SQLDatabase.from_uri(db_url)

# ---- LLM ----
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # or gemini-2.5-pro
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# -------------------------
# SQL cleanup + guardrails
# -------------------------
def clean_sql_query(text_: str) -> str:
    text_ = text_ or ""

    # Remove ```sql blocks
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text_ = re.sub(block_pattern, r"\1", text_, flags=re.DOTALL)

    # Remove "SQLQuery:" prefix variants
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text_ = re.sub(prefix_pattern, "", text_, flags=re.IGNORECASE)

    # Remove backticks
    text_ = re.sub(r'`([^`]*)`', r'\1', text_)

    # Normalize whitespace
    text_ = re.sub(r"\s+", " ", text_).strip()

    # Extract first statement-ish (we parse statements later)
    m = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*", text_, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text_ = m.group(0).strip()

    return text_


def _strip_sql(sql: str) -> str:
    sql = (sql or "").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def _split_sql_statements(sql: str) -> list[str]:
    """
    Splits SQL into statements on semicolons that are NOT inside quotes.
    Good enough for typical LLM-generated SQL (Postgres).
    """
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
    """
    Returns one of: SELECT, DML, DDL, OTHER
    """
    s = _strip_sql(sql)
    s = re.sub(r"^\s*(--[^\n]*\n\s*)+", "", s, flags=re.MULTILINE).strip()
    if not s:
        return "OTHER"

    first = s.split(None, 1)[0].upper()

    # Treat CTE as SELECT-family
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

    # Public mode: never allow multi-statement
    if mode == "public" and len(statements) > 1:
        raise ValueError("Multi-statement SQL is not allowed in public mode.")

    kinds = [_sql_kind(s) for s in statements]

    # Block DDL always
    if any(k == "DDL" for k in kinds):
        raise ValueError("DDL statements are blocked (CREATE/ALTER/DROP/TRUNCATE/GRANT/REVOKE).")

    # Public: SELECT only
    if mode == "public" and kinds[0] != "SELECT":
        raise ValueError("Public mode supports SELECT queries only. Use mode='sandbox' to simulate DML (rolled back).")

    engine = db_._engine
    rolled_back = False

    # -----------------------
    # Sandbox: multi-statement DML batch
    # -----------------------
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
                            "Try deleting dependent rows first (orderdetails -> orders -> customers), "
                            "or prefer a soft delete. Note: sandbox changes are simulated only and always rolled back."
                        ) from e
                    raise ValueError(f"Sandbox DML failed due to integrity constraint: {msg}") from e

                affected = int(res.rowcount) if res.rowcount is not None else 0
                total_affected += affected

            # ALWAYS rollback sandbox writes
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

    # -----------------------
    # Sandbox: single DML
    # -----------------------
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
                        "Try deleting dependent rows first (orderdetails -> orders -> customers), "
                        "or prefer a soft delete. Note: sandbox changes are simulated only and always rolled back."
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

    # -----------------------
    # SELECT path (public/sandbox) single statement
    # -----------------------
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

        # Fetch only up to 100 rows (LIMIT already enforced; extra safety)
        fetched = res.fetchmany(100)
        rows = [dict(zip(cols, r)) for r in fetched]
        rowcount = len(rows)

        return {
            "sql": final_sql,
            "kind": "SELECT",
            "mode": mode,
            "rolled_back": False,
            "rowcount": rowcount,
            "rows": rows,  # <= 100
            "preview": f"Rows returned: {rowcount}. Preview: {rows[:5]}",
        }


# -------------------------
# Answer prompt
# -------------------------
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

# -------------------------
# Few-shot examples (lowercase for Postgres)
# -------------------------
examples = [
    {"input": "List all customers in France with a credit limit over 20,000.",
     "query": "SELECT * FROM customers WHERE country = 'France' AND creditlimit > 20000;"},
    {"input": "Get the highest payment amount made by any customer.",
     "query": "SELECT MAX(amount) FROM payments;"},
    {"input": "Show product details for products in the 'Motorcycles' product line.",
     "query": "SELECT * FROM products WHERE productline = 'Motorcycles';"},
    {"input": "Retrieve the names of employees who report to employee number 1002.",
     "query": "SELECT firstname, lastname FROM employees WHERE reportsto = 1002;"},
    {"input": "List all products with a stock quantity less than 7000.",
     "query": "SELECT productname, quantityinstock FROM products WHERE quantityinstock < 7000;"},
    {"input": "what is price of 1968 Ford Mustang",
     "query": "SELECT buyprice, msrp FROM products WHERE productname = '1968 Ford Mustang' LIMIT 1;"},
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

# -------------------------
# Table descriptions from CSV
# -------------------------
def get_table_details():
    default_csv = Path(__file__).resolve().parent / "database_table_descriptions.csv"
    csv_path = Path(os.getenv("TABLE_DESCRIPTIONS_PATH", str(default_csv)))

    if not csv_path.exists():
        raise FileNotFoundError(
            f"database_table_descriptions.csv not found at: {csv_path}\n"
            f"Fix: Put the file at {default_csv} OR set TABLE_DESCRIPTIONS_PATH in .env"
        )

    table_description = pd.read_csv(csv_path)

    table_details = ""
    for _, row in table_description.iterrows():
        table_details += f"Table Name:{row['table_name']}\nTable Description:{row['description']}\n\n"

    return table_details


class Table(BaseModel):
    name: List[str] = Field(description="List of table names in the SQL database.")


table_details = get_table_details()

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

# -------------------------
# SQL generation prompt (Postgres)
# -------------------------
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run.\n"
            "For SELECT queries, keep results small using LIMIT {top_k} when appropriate.\n\n"
            "Write-operation rules (sandbox mode):\n"
            "- Only generate DML when asked: INSERT, UPDATE, DELETE.\n"
            "- Prefer a soft delete (UPDATE a status/flag column) instead of deleting parent rows when relationships exist.\n"
            "- If DELETE is required, respect foreign keys and delete children first (orderdetails -> orders -> customers).\n"
            "- Never generate DDL (CREATE/ALTER/DROP/TRUNCATE) or permission statements (GRANT/REVOKE).\n\n"
            "Here is the relevant table info: {table_info}\n\n"
            "Below are examples of questions and their corresponding SQL queries. Use them as guidance."
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

generate_query = create_sql_query_chain(llm, db, final_prompt)

def _run_exec(inputs: dict) -> dict:
    sql = inputs.get("query", "")
    mode = inputs.get("mode", "public")
    return execute_with_guardrails(db, sql, mode)

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

        # NEW: for frontend table
        "rows_preview": (x["exec_info"]["rows"][:20] if x["exec_info"]["kind"] == "SELECT" else []),
        "columns": (list(x["exec_info"]["rows"][0].keys()) if (x["exec_info"]["kind"] == "SELECT" and x["exec_info"]["rows"]) else []),
    })
)

def chain_code(q, m, mode="public"):
    return chain.invoke({
        "question": q,
        "table_details": table_details,
        "mode": mode,
        "top_k": 10,
    })
