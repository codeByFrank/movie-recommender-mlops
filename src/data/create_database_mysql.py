# src/data/create_database_mysql.py
from __future__ import annotations
import os, argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text  # <-- needed
# requires: sqlalchemy, pymysql, pandas

DB_HOST = os.getenv("DATABASE_HOST", "mysql-ml")
DB_PORT = os.getenv("DATABASE_PORT", "3306")
DB_USER = os.getenv("DATABASE_USER", "app")
DB_PASS = os.getenv("DATABASE_PASSWORD", "mysql")
DB_NAME = os.getenv("DATABASE_NAME", "movielens")

def engine(db: str | None = DB_NAME):
    """Return an Engine; if db is None, connect to server w/out a database."""
    if db:
        uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{db}"
    else:
        uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}"
    return create_engine(uri, pool_pre_ping=True, future=True)

DDL = """
CREATE TABLE IF NOT EXISTS ratings (
  userId      INT NOT NULL,
  movieId     INT NOT NULL,
  rating      FLOAT NOT NULL,
  timestamp   BIGINT NOT NULL,
  batch_id    VARCHAR(64) NULL,
  imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (userId, movieId, timestamp),
  INDEX idx_movie (movieId),
  INDEX idx_batch (batch_id),
  INDEX idx_imported (imported_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

def ensure_database(db_name: str):
    """Create DB if missing (idempotent)."""
    eng = engine(db=None)
    with eng.begin() as con:
        con.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
    eng.dispose()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-schema", action="store_true",
                    help="Only (re)create tables and exit")
    ap.add_argument("--load-movies", default=None,
                    help="Optional path to movies.csv to load a movies table")
    ap.add_argument("--batch-csv", default=None,
                    help="Path to a ratings batch CSV to ingest")
    ap.add_argument("--batch-id", default=None,
                    help="Identifier to tag rows from --batch-csv (e.g., ratings_batch_003)")
    args = ap.parse_args()

    # 1) Ensure database exists
    ensure_database(DB_NAME)

    # 2) Ensure ratings schema (and optionally movies)
    eng = engine(DB_NAME)
    with eng.begin() as con:
        con.execute(text(DDL))
        if args.load_movies:
            p = Path(args.load_movies)
            if not p.exists():
                raise FileNotFoundError(p)
            dfm = pd.read_csv(p)
            dfm.to_sql("movies", con, if_exists="replace", index=False)
            print(f"Loaded {len(dfm)} rows into movies.")

    # If only schema init requested
    if args.init_schema and not args.batch_csv:
        print("Schema ensured. Nothing else to do.")
        return

    # 3) Ingest ratings batch (with batch_id)
    if args.batch_csv:
        p = Path(args.batch_csv)
        if not p.exists():
            raise FileNotFoundError(p)

        batch_id = args.batch_id or p.stem
        dfr = pd.read_csv(p)
        required = {"userId", "movieId", "rating", "timestamp"}
        missing = required - set(dfr.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        dfr = dfr[list(required)].copy()
        dfr["batch_id"] = batch_id

        with eng.begin() as con:
            # temp table for fast bulk load
            con.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_ratings"))
            con.execute(text("""
                CREATE TEMPORARY TABLE tmp_ratings (
                  userId INT, movieId INT, rating FLOAT, timestamp BIGINT, batch_id VARCHAR(64)
                )
            """))
            dfr.to_sql("tmp_ratings", con, if_exists="append", index=False)

            # upsert into ratings keyed by (userId, movieId, timestamp)
            con.execute(text("""
                INSERT INTO ratings (userId, movieId, rating, timestamp, batch_id)
                SELECT userId, movieId, rating, timestamp, batch_id
                FROM tmp_ratings
                ON DUPLICATE KEY UPDATE
                    rating      = VALUES(rating),
                    batch_id    = VALUES(batch_id),
                    imported_at = CURRENT_TIMESTAMP
            """))

        print(f"✅ Ingested {len(dfr)} rows into ratings with batch_id={batch_id}")

    eng.dispose()

if __name__ == "__main__":
    main()
