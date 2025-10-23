# airflow/dags/retrain_on_new_batch.py
from __future__ import annotations
import os, glob, shutil, json, subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

import pandas as pd  # required by some helpers

# ---- CONFIG ----
PROJECT_ROOT  = "/opt/airflow/repo"
LANDING_DIR   = os.environ.get("LANDING_DIR",  f"{PROJECT_ROOT}/data/landing")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", f"{PROJECT_ROOT}/data/processed")
FAILED_DIR    = os.environ.get("FAILED_DIR",   f"{PROJECT_ROOT}/data/failed")
INCOMING_DIR  = os.environ.get("INCOMING_DIR", f"{PROJECT_ROOT}/data/incoming")

MLFLOW_TRACKING_URI = "http://mlflow-ui:5000"
MODEL_NAME = "movie_recommender_svd"
PRIMARY_METRIC = "rmse"
BETTER_IS_LOWER = True

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="retrain_on_new_batch",
    start_date=datetime(2025, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training"],
) as dag:

    # --- Generate a tiny batch if landing is empty (move first incoming -> landing) ---
    def maybe_generate_batch_if_empty(**ctx):
        os.makedirs(LANDING_DIR, exist_ok=True)
        if glob.glob(os.path.join(LANDING_DIR, "*.csv")):
            print("Landing not empty — skipping generation.")
            return

        incoming = sorted(glob.glob(os.path.join(INCOMING_DIR, "ratings_batch_*.csv")))
        if not incoming:
            print("No incoming batches left.")
            return

        src = incoming[0]
        dst = os.path.join(LANDING_DIR, os.path.basename(src))
        shutil.move(src, dst)
        print(f"Moved {src} -> {dst}")

    pull_one_batch = PythonOperator(
        task_id="maybe_generate_batch_if_empty",
        python_callable=maybe_generate_batch_if_empty,
    )

    # --- Pick batch from landing ---
    def pick_one_batch(**ctx):
        files = sorted(glob.glob(os.path.join(LANDING_DIR, "*.csv")))
        if not files:
            raise AirflowSkipException("No new CSV in landing.")
        batch = files[0]
        print("Picked:", batch)
        ctx["ti"].xcom_push(key="batch_path", value=batch)

    pick_batch = PythonOperator(
        task_id="pick_batch",
        python_callable=pick_one_batch,
    )

    # --- Ingest batch into MySQL ---
    def ingest_mysql(**ctx):
        import sys, shlex, traceback

        print("=== INGEST START ===")
        batch = ctx["ti"].xcom_pull(key="batch_path")
        print("INGEST >> batch_path =", batch)
        print("INGEST >> PROJECT_ROOT =", PROJECT_ROOT)
        print("INGEST >> LANDING_DIR  =", LANDING_DIR)
        print("INGEST >> PYTHON (sys.executable) =", sys.executable)

        if not batch:
            raise RuntimeError("No XCom 'batch_path' value")
        if not os.path.exists(batch):
            raise FileNotFoundError(f"Batch file not found: {batch}")

        batch_id = os.path.splitext(os.path.basename(batch))[0]  # ratings_batch_003
        print("INGEST >> batch_id =", batch_id)

        cmd = [
            sys.executable, "-m", "src.data.create_database_mysql",
            "--batch-csv", batch,
            "--batch-id", batch_id,
        ]
        print("INGEST >> Running:", " ".join(shlex.quote(c) for c in cmd))

        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
            print("---- STDOUT ----\n" + (proc.stdout or ""))
            print("---- STDERR ----\n" + (proc.stderr or ""))
            print("INGEST >> returncode =", proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(f"ingest exited with code {proc.returncode}")

            ctx["ti"].xcom_push(key="batch_id", value=batch_id)
            print("=== INGEST OK ===")
        except Exception:
            print("=== INGEST FAILED ===")
            traceback.print_exc()
            raise

    ingest = PythonOperator(
        task_id="ingest_mysql",
        python_callable=ingest_mysql,
    )

    # --- Train model via API ---
    def train_with_mlflow(**ctx):
        import requests
        from requests.auth import HTTPBasicAuth

        api_user = os.getenv("API_BASIC_USER", "admin")
        api_pass = os.getenv("API_BASIC_PASS", "secret")

        train_scope = os.getenv("TRAIN_SCOPE", "cumulative")  # "batch" or "cumulative"
        batch_id = ctx["ti"].xcom_pull(key="batch_id")
        payload = {"train_scope": train_scope, "train_batch_id": batch_id}

        print("=== TRAINING VIA FASTAPI ===")
        print(f"Calling http://api:8000/train with payload={payload}")

        resp = requests.post(
            "http://api:8000/train",
            json=payload,
            auth=HTTPBasicAuth(api_user, api_pass),
            timeout=3600,
        )
        print(f"Response status: {resp.status_code}")
        if resp.status_code != 200:
            raise RuntimeError(f"Training API returned {resp.status_code}: {resp.text}")

        result = resp.json()
        print(f"Training status: {result.get('status')}")
        if result.get("stdout"):
            print("---- TRAIN STDOUT (from API) ----")
            print(result["stdout"])
        if result.get("stderr"):
            print("---- TRAIN STDERR (from API) ----")
            print(result["stderr"])

        if result.get("status") != "success":
            err = result.get("stderr") or result.get("message") or "Unknown error"
            raise RuntimeError(f"Training failed: {err}")

        # Parse last JSON line from trainer stdout
        stdout = result.get("stdout", "")
        last_json = None
        for line in stdout.strip().splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    last_json = json.loads(line)
                    break
                except Exception:
                    pass

        if not last_json:
            raise RuntimeError("Training script didn't emit final JSON metrics")

        metric = last_json.get(PRIMARY_METRIC)
        if metric is None:
            raise AirflowSkipException("Training produced no PRIMARY_METRIC (rmse).")

        ctx["ti"].xcom_push(key="candidate_run_id", value=last_json.get("run_id"))
        ctx["ti"].xcom_push(key="candidate_metric", value=float(metric))
        print(f"✅ Training completed via API: run_id={last_json.get('run_id')}, {PRIMARY_METRIC}={metric}")

    train = PythonOperator(
        task_id="train_candidate",
        python_callable=train_with_mlflow,
    )

    # --- Compare & promote ---
    def compare_and_promote(**ctx):
        import math
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        candidate_run_id = ctx["ti"].xcom_pull(key="candidate_run_id")
        candidate_metric = float(ctx["ti"].xcom_pull(key="candidate_metric"))

        def get_metric_from_run(run_id: str, metric_name: str):
            run = client.get_run(run_id)
            m = run.data.metrics.get(metric_name)
            return None if m is None else float(m)

        # find current prod via alias (authoritative)
        current_prod_metric = None
        current_prod_version = None
        try:
            prod_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
            current_prod_version = int(prod_mv.version)
            current_prod_metric = get_metric_from_run(prod_mv.run_id, PRIMARY_METRIC)
        except Exception:
            pass  # no production yet

        def better(new, old):
            if old is None or (isinstance(old, float) and math.isnan(old)):
                return True
            return new < old if BETTER_IS_LOWER else new > old

        # find the ModelVersion for this candidate run_id
        cand_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        cand_v = [v for v in cand_versions if v.run_id == candidate_run_id]
        if not cand_v:
            raise RuntimeError("Candidate run not found as a registered model version.")
        cand_v = sorted(cand_v, key=lambda v: int(v.version), reverse=True)[0]

        print(
            f"Comparing: candidate {PRIMARY_METRIC}={candidate_metric} "
            f"vs prod {PRIMARY_METRIC}={current_prod_metric} (v{current_prod_version})"
        )

        if better(candidate_metric, current_prod_metric):
            client.set_registered_model_alias(MODEL_NAME, "production", cand_v.version)
            print(f"✅ Promoted v{cand_v.version} ({PRIMARY_METRIC}={candidate_metric}) to @production")
        else:
            print(f"ℹ️ Kept production v{current_prod_version} ({PRIMARY_METRIC}={current_prod_metric})")

    select_and_promote = PythonOperator(
        task_id="compare_and_promote",
        python_callable=compare_and_promote,
    )

    # --- Archive processed batch ---
    def archive(**ctx):
        batch = ctx["ti"].xcom_pull(key="batch_path")
        if not batch:
            print("No batch_path (upstream skipped); nothing to archive.")
            return
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        dst = os.path.join(PROCESSED_DIR, os.path.basename(batch))
        shutil.move(batch, dst)
        print(f"Moved {batch} -> {dst}")

    archive_batch = PythonOperator(
        task_id="archive_batch",
        python_callable=archive,
        trigger_rule="none_failed_min_one_success",
    )

    # --- Dependencies ---
    pull_one_batch >> pick_batch >> ingest >> train >> select_and_promote >> archive_batch
