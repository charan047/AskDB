import os
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["DATABASE_URL"] = "sqlite:///./tests/tmp.db"

from untitled0 import _sql_kind, _split_sql_statements, execute_with_guardrails
from langchain_community.utilities.sql_database import SQLDatabase

@pytest.fixture(scope="module")
def db():
    db = SQLDatabase.from_uri(os.environ["DATABASE_URL"])
    # create a tiny table
    db._engine.execute  # ensure exists
    with db._engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT);")
        conn.exec_driver_sql("DELETE FROM customers;")
        conn.exec_driver_sql("INSERT INTO customers (id, name) VALUES (1,'Alice'), (2,'Bob');")
    return db

def test_split_statements():
    assert _split_sql_statements("SELECT 1; SELECT 2;") == ["SELECT 1", "SELECT 2"]

def test_kind_select():
    assert _sql_kind("SELECT * FROM customers") == "SELECT"
    assert _sql_kind("WITH x AS (SELECT 1) SELECT * FROM x") == "SELECT"

def test_public_blocks_dml(db):
    with pytest.raises(ValueError):
        execute_with_guardrails(db, "DELETE FROM customers WHERE id=1", mode="public")

def test_sandbox_rolls_back(db):
    out = execute_with_guardrails(db, "DELETE FROM customers WHERE id=1", mode="sandbox")
    assert out["kind"] == "DML"
    assert out["rolled_back"] is True
    # verify row still exists
    out2 = execute_with_guardrails(db, "SELECT * FROM customers WHERE id=1", mode="public")
    assert out2["rowcount"] == 1
