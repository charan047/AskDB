import os
import pytest
from sqlalchemy import text
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from untitled0 import _sql_kind, _split_sql_statements, execute_with_guardrails
from langchain_community.utilities.sql_database import SQLDatabase

@pytest.fixture(scope="module")
def db():
    db = SQLDatabase.from_uri(os.environ["DATABASE_URL"])

    with db._engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS t"))
        conn.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO t (id, name) VALUES (1, 'a'), (2, 'b')"))

    return db

def test_split_statements():
    assert _split_sql_statements("SELECT 1; SELECT 2;") == ["SELECT 1", "SELECT 2"]

def test_kind_select():
    assert _sql_kind("SELECT * FROM t") == "SELECT"
    assert _sql_kind("WITH x AS (SELECT 1) SELECT * FROM x") == "SELECT"

def test_public_blocks_dml(db):
    with pytest.raises(ValueError):
        execute_with_guardrails(db, "DELETE FROM t WHERE id=1", mode="public")

def test_sandbox_rolls_back(db):
    out = execute_with_guardrails(db, "DELETE FROM t WHERE id=1", mode="sandbox")
    assert out["kind"] == "DML"
    assert out["rolled_back"] is True

    out2 = execute_with_guardrails(db, "SELECT COUNT(*) AS c FROM t WHERE id=1", mode="public")
    assert out2["kind"] == "SELECT"
    assert out2["rows"][0]["c"] == 1
