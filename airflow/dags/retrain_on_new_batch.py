# airflow/dags/retrain_on_new_batch.py
#
# DAG:
# 1) pick one CSV from data/landing (if none, skip)
# 2) ingest it to MySQL via your script
# 3) train the model (logs to MLflow local folder)
# 4) move the CSV to data/processed

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

from sqlalchemy import create_engine, text
import pandas as pd


# Paths inside the Airflow containers
REPO_ROOT = "/opt/airflow/repo"
LANDING_DIR = os.environ.get("LANDING_DIR", f"{REPO_ROOT}/data/landing")
PROCESSED_DIR = f"{REPO_ROOT}/data/processed"

# Keep MLflow artifacts inside the bind-mounted repo
MLFLOW_URI_DEFAULT = f"file:{REPO_ROOT}/mlruns"

default_args = {
    "owner": "you",
    "retries": 0,
}

with DAG(
    dag_id="retrain_on_new_batch",
    start_date=datetime(2025, 10, 1),
    schedule="*/30 * * * *",
    catchup=False,
    default_args=default_args,
    dagrun_timeout=timedelta(minutes=30),
    description="Ingest one new ratings batch, retrain SVD, archive batch",
) as dag:
    
    def _conn_url():
        host = os.getenv("DATABASE_HOST", "mysql-ml")
        port = os.getenv("DATABASE_PORT", "3306")
        user = os.getenv("DATABASE_USER", "app")
        pwd  = os.getenv("DATABASE_PASSWORD", "mysql")
        db   = os.getenv("DATABASE_NAME", "movielens")
        return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"


    @task
    def maybe_generate_batch_if_empty() -> None:
        """If landing/ is empty, create a small synthetic batch from MySQL."""
        os.makedirs(LANDING_DIR, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(LANDING_DIR, "ratings_batch_*.csv")))
        if existing:
            print(f"Landing not empty ({len(existing)} files) — no generation.")
            return

        print("Landing empty → generating a small batch...")
        eng = create_engine(_conn_url(), pool_pre_ping=True, future=True)
        # sample ~200 rows (RAND() is fine for small local data)
        q = text("""
            SELECT userId, movieId, rating, `timestamp`
            FROM ratings
            ORDER BY RAND() LIMIT 200
        """)
        with eng.connect() as c:
            df = pd.read_sql_query(q, c)

        if df.empty:
            print("No rows in ratings yet — writing a tiny dummy 2-row batch.")
            df = pd.DataFrame(
                [{"userId": 9999, "movieId": 1, "rating": 4.0, "timestamp": 1699999999},
                {"userId": 9999, "movieId": 2, "rating": 3.5, "timestamp": 1699999999}]
            )

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(LANDING_DIR, f"ratings_batch_{ts}.csv")
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows → {out}")

    @task
    def pick_batch() -> str:
        os.makedirs(LANDING_DIR, exist_ok=True)
        pattern = os.path.join(LANDING_DIR, "ratings_batch_*.csv")   # <-- the *
        files = sorted(glob.glob(pattern))
        if not files:
            from airflow.exceptions import AirflowSkipException
            raise AirflowSkipException(f"No batch files found matching {pattern}")
        return files[0]  # oldest first

    @task
    def ingest(csv_path: str):
        cmd = [
            "python", "-u", "-m", "src.dataops.ingest_to_mysql",
            "--csv_path", csv_path,                           # <-- pass-through
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


    @task
    def train() -> None:
        # Ensure the mlruns directory exists (for file: tracking)
        os.makedirs(os.path.join(REPO_ROOT, "mlruns"), exist_ok=True)

        cmd = ["python", "-u", "-m", "src.models.train_model_mysql"]
        env = os.environ.copy()
        env.setdefault("MLFLOW_TRACKING_URI", MLFLOW_URI_DEFAULT)
        env.setdefault("GIT_PYTHON_REFRESH", "quiet")

        print("Env MLFLOW_TRACKING_URI =", env.get("MLFLOW_TRACKING_URI"))
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

    @task
    def archive(csv_path: str) -> None:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        dest = os.path.join(PROCESSED_DIR, os.path.basename(csv_path))
        shutil.move(csv_path, dest)
        print(f"Moved {csv_path} -> {dest}")

    gen = maybe_generate_batch_if_empty()   # <- NEW: create a batch if landing/ is empty
    batch = pick_batch()

    ingest_task = ingest(batch)
    train_task = train()
    archive_task = archive(batch)

    # dependencies
    gen >> batch                # ensure generator runs before we try to pick a file
    ingest_task >> train_task >> archive_task
