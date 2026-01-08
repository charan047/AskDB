"""Initialize a tiny demo SQLite DB (sample.db) for AskDB.

Usage:
  python scripts/init_sqlite.py
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "sample.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop & recreate tables for repeatable runs
    cur.executescript("""
    DROP TABLE IF EXISTS cars;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS users;

    CREATE TABLE cars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        make TEXT NOT NULL,
        model TEXT NOT NULL,
        price_usd INTEGER NOT NULL
    );

    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    cars = [
        (1968, "Ford", "Mustang", 95000),
        (1997, "Toyota", "Supra", 72000),
        (2018, "Tesla", "Model 3", 32000),
        (2020, "Honda", "Civic", 21000),
        (2015, "BMW", "M3", 48000),
    ]
    cur.executemany("INSERT INTO cars (year, make, model, price_usd) VALUES (?, ?, ?, ?)", cars)

    users = [("Alice", "alice@example.com"), ("Bob", "bob@example.com")]
    cur.executemany("INSERT INTO users (name, email) VALUES (?, ?)", users)

    orders = [(1, 199.99, "2025-12-01"), (2, 49.50, "2025-12-03")]
    cur.executemany("INSERT INTO orders (user_id, total_amount, created_at) VALUES (?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"✅ Created {DB_PATH}")

if __name__ == "__main__":
    main()
