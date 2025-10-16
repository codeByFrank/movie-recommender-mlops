# src/data/create_database_mysql.py
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import mysql.connector as mc
from mysql.connector import Error


# ---------- Config ----------
ROOT = Path("/opt/airflow/repo") if Path("/opt/airflow/repo").exists() else Path.cwd()
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)

DB_HOST = os.getenv("DATABASE_HOST", "mysql-ml")
DB_PORT = int(os.getenv("DATABASE_PORT", "3306"))
DB_USER = os.getenv("DATABASE_USER", "app")
DB_PASS = os.getenv("DATABASE_PASSWORD", "mysql")
DB_NAME = os.getenv("DATABASE_NAME", "movielens")


# ---------- MySQL helpers ----------
def connect_server():
    """Connect to MySQL server (no DB selected yet)."""
    try:
        conn = mc.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            connection_timeout=5,
            autocommit=False,
            ssl_disabled=True,
            use_pure=True,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            _ = cur.fetchone()
        return conn
    except Exception as e:
        print(f"❌ Connect error: {e}")
        traceback.print_exc()
        return None


def ensure_database(conn, db_name: str):
    """Ensure database exists and select it."""
    with conn.cursor() as cur:
        try:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        except Error as e:
            print(f"ℹ️ CREATE DATABASE may be restricted: {e}")
        cur.execute("SHOW DATABASES LIKE %s", (db_name,))
        if not cur.fetchone():
            print(f"❌ Database '{db_name}' not found and cannot be created.")
            sys.exit(1)
        cur.execute(f"USE `{db_name}`")


def create_tables(conn):
    """Create tables if missing (idempotent)."""
    movies_sql = """
        CREATE TABLE IF NOT EXISTS movies (
            movieId INT PRIMARY KEY,
            title   VARCHAR(500) NOT NULL,
            genres  VARCHAR(200),
            INDEX idx_title (title(100))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    ratings_sql = """
        CREATE TABLE IF NOT EXISTS ratings (
            userId    INT NOT NULL,
            movieId   INT NOT NULL,
            rating    DECIMAL(2,1) NOT NULL,
            timestamp INT NOT NULL,
            PRIMARY KEY (userId, movieId, timestamp),
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
                ON DELETE CASCADE ON UPDATE CASCADE,
            INDEX idx_user (userId),
            INDEX idx_movie (movieId),
            INDEX idx_rating (rating)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    with conn.cursor() as cur:
        cur.execute(movies_sql)
        cur.execute(ratings_sql)
    conn.commit()


# ---------- Data loaders ----------
def load_movies(conn, movies_csv: Path):
    """Upsert movies (idempotent)."""
    df = pd.read_csv(movies_csv)
    need = {"movieId", "title", "genres"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise SystemExit(f"movies CSV missing {missing}")

    rows = [(int(r.movieId), str(r.title), str(r.genres)) for _, r in df.iterrows()]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO movies (movieId, title, genres) VALUES (%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE title=VALUES(title), genres=VALUES(genres)",
            rows,
        )
    conn.commit()
    print(f"✅ Loaded/updated {len(rows):,} movies")


def _normalize_ratings_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a ratings chunk:
    - Ensure required columns exist (userId, movieId, rating).
    - If 'timestamp' is missing, synthesize it from common date columns or 'now'.
    - Enforce numeric types, drop rows with nulls after coercion.
    """
    # normalize headers (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    REQUIRED = {"userId", "movieId", "rating"}
    missing_required = REQUIRED - set(df.columns)
    if missing_required:
        raise SystemExit(f"ratings CSV missing required columns: {missing_required}")

    # create timestamp if missing (try to parse from common date-ish columns)
    if "timestamp" not in df.columns:
        for cand in ("datetime", "date", "rated_at", "created_at"):
            if cand in df.columns:
                parsed = pd.to_datetime(df[cand], errors="coerce", utc=True)
                if parsed.notna().any():
                    df["timestamp"] = (parsed.view("int64") // 10**9)
                    break
        else:
            df["timestamp"] = int(time.time())

    # enforce numeric types
    df["userId"] = pd.to_numeric(df["userId"], errors="coerce").astype("Int64")
    df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype(float)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")

    # drop rows with missing key fields after coercion
    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()

    # cast back to primitive ints for MySQL driver
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    return df


def ingest_batch(conn, ratings_csv: Path, chunk: int = 100_000):
    """
    Append ratings from a CSV in chunks (idempotent via INSERT IGNORE).
    Accepts CSVs missing 'timestamp' (will synthesize).
    """
    total = 0
    # chunked read; let pandas detect encoding & BOM
    for chunk_df in pd.read_csv(ratings_csv, chunksize=chunk):
        chunk_df = _normalize_ratings_chunk(chunk_df)

        if chunk_df.empty:
            continue

        rows = list(
            zip(
                chunk_df["userId"].tolist(),
                chunk_df["movieId"].tolist(),
                chunk_df["rating"].astype(float).tolist(),
                chunk_df["timestamp"].tolist(),
            )
        )

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT IGNORE INTO ratings (userId, movieId, rating, timestamp) VALUES (%s,%s,%s,%s)",
                rows,
            )
        conn.commit()

        total += len(chunk_df)
        print(f"  → processed {total:,} rows")

    print(f"✅ Ingested batch rows processed: {total:,}")


def test_counts(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM movies")
        m = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM ratings")
        r = cur.fetchone()[0]
    print(f"Movies: {m:,}  Ratings: {r:,}")


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="MySQL setup & ingestion helper")
    p.add_argument("--init-schema", action="store_true", help="Create DB/tables only")
    p.add_argument("--load-movies", type=str, help="Path to movies.csv to upsert")
    p.add_argument("--batch-csv", type=str, help="Path to ratings batch CSV to append (idempotent)")
    args = p.parse_args()

    conn = connect_server()
    if not conn:
        sys.exit(1)

    try:
        ensure_database(conn, DB_NAME)
        # Always ensure tables exist before any data operation
        create_tables(conn)

        did_something = False

        if args.init_schema:
            did_something = True

        if args.load_movies:
            load_movies(conn, Path(args.load_movies))
            did_something = True

        if args.batch_csv:
            ingest_batch(conn, Path(args.batch_csv))
            did_something = True

        if not did_something:
            print("Nothing to do. Use --init-schema and/or --load-movies and/or --batch-csv.")
            sys.exit(2)

        test_counts(conn)

    finally:
        try:
            if conn.is_connected():
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
