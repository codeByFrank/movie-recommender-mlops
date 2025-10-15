# src/dataops/ingest_to_mysql.py
# Purpose: append one CSV batch into the MySQL 'ratings' table (MovieLens-style).

from __future__ import annotations

import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# If you use dotenv locally, don't override Docker envs
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass


def conn_url_from_env() -> str:
    """
    Build the SQLAlchemy URL from env vars supplied by docker-compose.
    IMPORTANT:
      - Inside Docker, host must be the service name 'mysql-ml' (not 127.0.0.1).
      - From your host machine, you would use 127.0.0.1:3306 instead.
    """
    host = os.getenv("DATABASE_HOST", "mysql-ml")
    port = os.getenv("DATABASE_PORT", "3306")
    user = os.getenv("DATABASE_USER", "app")
    pwd  = os.getenv("DATABASE_PASSWORD", "mysql")
    db   = os.getenv("DATABASE_NAME", "movielens")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"


def ensure_table(engine: Engine) -> None:
    """Create ratings table if it doesn't exist (simple MovieLens schema)."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS ratings (
        userId     INT        NOT NULL,
        movieId    INT        NOT NULL,
        rating     FLOAT      NOT NULL,
        `timestamp` BIGINT    NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


def ingest_csv(engine: Engine, csv_path: str) -> int:
    """Read one CSV and append its rows to ratings."""
    print(f"[ingest] Reading CSV: {csv_path}", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)

    expected = {"userId", "movieId", "rating", "timestamp"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {sorted(missing)}")

    # Pandas -> MySQL append
    print(f"[ingest] Appending {len(df)} rows to table 'ratings' ...", flush=True)
    df.to_sql("ratings", con=engine, if_exists="append", index=False, chunksize=10_000)
    print(f"[ingest] Done. Inserted {len(df)} rows.", flush=True)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a ratings CSV batch into MySQL.")
    ap.add_argument("--csv_path", required=True, help="Path to the batch CSV to ingest.")
    args = ap.parse_args()

    url = conn_url_from_env()
    print(f"[ingest] Connecting via: {url}", flush=True)

    # pool_pre_ping avoids broken connection errors if MySQL idled out
    engine = create_engine(url, pool_pre_ping=True, future=True)

    # Quick connectivity probe
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[ingest] DB connectivity OK.", flush=True)
    except Exception as e:
        print(f"[ingest] DB connection failed: {e}", file=sys.stderr, flush=True)
        sys.exit(2)

    try:
        ensure_table(engine)
        rows = ingest_csv(engine, args.csv_path)
        print(f"[ingest] Success. Rows ingested: {rows}", flush=True)
    except Exception as e:
        print(f"[ingest] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
