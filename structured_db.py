# structured_db.py — Robust SQLite backend for DocuMind Lite
# -----------------------------------------------------------
# Features:
#   • Central schema for invoices, resumes, contracts
#   • Auto table creation + column migration (adds new columns if needed)
#   • Generic upsert(table, row) with ON CONFLICT(doc_id)
#   • CSV -> DB import (upserting, no duplicates)
#   • Simple analytics helpers: counts, recent docs, table -> DataFrame
#
# Dependencies: sqlite3, pandas (already used in the project)

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# =====================================================================
# PATHS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "documind.db"

STRUCTURED_DIR = BASE_DIR / "data" / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

INVOICES_CSV = STRUCTURED_DIR / "invoices.csv"
RESUMES_CSV = STRUCTURED_DIR / "resumes.csv"
CONTRACTS_CSV = STRUCTURED_DIR / "contracts.csv"


# =====================================================================
# SCHEMA (single source of truth)
# =====================================================================
# Columns are a mapping: column_name -> SQLite type / constraints.

SCHEMA: Dict[str, Dict[str, str]] = {
    "invoices": {
        "doc_id": "TEXT PRIMARY KEY",
        "source_file": "TEXT",
        "invoice_number": "TEXT",
        "invoice_date": "TEXT",
        "due_date": "TEXT",
        "vendor_name": "TEXT",
        "client_name": "TEXT",
        "total_amount": "TEXT",
        "currency": "TEXT",
        "raw_text_snippet": "TEXT",
    },
    "resumes": {
        "doc_id": "TEXT PRIMARY KEY",
        "source_file": "TEXT",
        "full_name": "TEXT",
        "email": "TEXT",
        "phone": "TEXT",
        "headline": "TEXT",
        "total_experience_years": "REAL",
        "skills": "TEXT",
        "education_summary": "TEXT",
        "raw_text_snippet": "TEXT",
    },
    "contracts": {
        "doc_id": "TEXT PRIMARY KEY",
        "source_file": "TEXT",
        "party_a": "TEXT",
        "party_b": "TEXT",
        "effective_date": "TEXT",
        "end_date": "TEXT",
        "governing_law": "TEXT",
        "contract_title": "TEXT",
        "raw_text_snippet": "TEXT",
    },
}


# =====================================================================
# LOW-LEVEL HELPERS
# =====================================================================

def _get_conn() -> sqlite3.Connection:
    """
    Get a SQLite connection with foreign keys enabled and Row factory.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _init_table(conn: sqlite3.Connection, table: str, cols: Dict[str, str]) -> None:
    """
    Create table if missing, and ensure all columns from SCHEMA exist.
    (Adds new columns if schema was upgraded.)
    """
    cur = conn.cursor()

    # 1) CREATE TABLE IF NOT EXISTS with full schema
    col_defs = ", ".join(f"{name} {ctype}" for name, ctype in cols.items())
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs});")

    # 2) Ensure all columns exist (for upgrades)
    cur.execute(f"PRAGMA table_info({table});")
    existing_cols = {row["name"] for row in cur.fetchall()}

    for name, ctype in cols.items():
        if name not in existing_cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype};")


def init_db() -> None:
    """
    Initialize all tables according to SCHEMA.
    Safe to call many times (idempotent).
    """
    with _get_conn() as conn:
        for table, cols in SCHEMA.items():
            _init_table(conn, table, cols)


# =====================================================================
# GENERIC UPSERT
# =====================================================================

def _upsert(table: str, row: Dict[str, Any]) -> None:
    """
    Generic upsert for any table in SCHEMA.
    - Filters row to known columns
    - Requires 'doc_id' to be present
    - Uses ON CONFLICT(doc_id) DO UPDATE SET ...
    """
    if not row:
        return

    if table not in SCHEMA:
        raise ValueError(f"Unknown table: {table}")

    if "doc_id" not in row:
        raise ValueError("Row must contain 'doc_id' for upsert.")

    init_db()  # ensure tables & columns exist

    schema_cols = list(SCHEMA[table].keys())
    # Filter row to schema columns (extra keys are ignored)
    cols = [c for c in schema_cols if c in row]
    if not cols:
        return

    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)

    # Build update assignments for non-PK columns
    update_cols = [c for c in cols if c != "doc_id"]
    if update_cols:
        assignments = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
    else:
        assignments = ""  # nothing to update

    sql = f"""
        INSERT INTO {table} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT(doc_id) DO UPDATE SET
        {assignments};
    """

    values = [row.get(c) for c in cols]

    with _get_conn() as conn:
        cur = conn.cursor()
        # If there is nothing to update (only doc_id), SQLite allows empty SET
        if assignments.strip():
            cur.execute(sql, values)
        else:
            # In case we only insert doc_id, use simpler INSERT OR IGNORE
            cur.execute(
                f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders});",
                values,
            )
        conn.commit()


# =====================================================================
# PUBLIC UPSERTS (used by extractors)
# =====================================================================

def upsert_invoice_db(row: Dict[str, Any]) -> None:
    _upsert("invoices", row)


def upsert_resume_db(row: Dict[str, Any]) -> None:
    _upsert("resumes", row)


def upsert_contract_db(row: Dict[str, Any]) -> None:
    _upsert("contracts", row)


# =====================================================================
# CSV → DB IMPORT (ONE-TIME OR OCCASIONAL)
# =====================================================================

def _import_csv(path: Path, table: str) -> None:
    """
    Import a CSV into a table, upserting by doc_id.
    Safe to run multiple times.
    """
    if not path.exists():
        return

    try:
        df = pd.read_csv(path)
    except Exception:
        return

    if df.empty or "doc_id" not in df.columns:
        return

    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    for row in records:
        try:
            _upsert(table, row)
        except Exception as e:
            print(f"[structured_db] Failed to import row into {table}: {e}")


def import_csvs_if_present() -> None:
    """
    Import existing CSVs into SQLite tables (upsert).
    You can call this once during setup or occasionally to sync.
    """
    _import_csv(INVOICES_CSV, "invoices")
    _import_csv(RESUMES_CSV, "resumes")
    _import_csv(CONTRACTS_CSV, "contracts")


# =====================================================================
# ANALYTICS / QUERY HELPERS
# =====================================================================

def get_table_df(table: str) -> pd.DataFrame:
    """
    Return a pandas DataFrame of a given table.
    Useful for dashboards / analytics.
    """
    if table not in SCHEMA:
        raise ValueError(f"Unknown table: {table}")

    init_db()
    with _get_conn() as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
        except Exception:
            df = pd.DataFrame(columns=list(SCHEMA[table].keys()))
    return df


def get_counts() -> Dict[str, int]:
    """
    Return a dict with row counts for each table.
    Example: {"invoices": 345, "resumes": 42, "contracts": 17}
    """
    init_db()
    counts: Dict[str, int] = {}
    with _get_conn() as conn:
        cur = conn.cursor()
        for table in SCHEMA.keys():
            try:
                cur.execute(f"SELECT COUNT(*) AS cnt FROM {table};")
                row = cur.fetchone()
                counts[table] = int(row["cnt"]) if row else 0
            except Exception:
                counts[table] = 0
    return counts


def get_recent_docs(table: str, limit: int = 20) -> pd.DataFrame:
    """
    Return the most recent docs (based on rowid) from a table.
    Requires that table exists; safe to call either way.
    """
    if table not in SCHEMA:
        raise ValueError(f"Unknown table: {table}")

    init_db()
    with _get_conn() as conn:
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?;",
                conn,
                params=(limit,),
            )
        except Exception:
            df = pd.DataFrame(columns=list(SCHEMA[table].keys()))
    return df
