# airflow/dags/retrain_on_new_batch.py
from __future__ import annotations
import os, glob, shutil, json, subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

import pandas as pd  # <-- needed for generator

# ---- CONFIG ----
PROJECT_ROOT  = "/opt/airflow/repo"
LANDING_DIR   = os.environ.get("LANDING_DIR",  f"{PROJECT_ROOT}/data/landing")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", f"{PROJECT_ROOT}/data/processed")
FAILED_DIR    = os.environ.get("FAILED_DIR",   f"{PROJECT_ROOT}/data/failed")

PYTHON_BIN = "python"  # <-- define this

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
    tags=["mlops","training"],
) as dag:

    # --- Generate a tiny batch if landing is empty (nice for demos) ---
    def maybe_generate_batch_if_empty(**ctx):
        os.makedirs(LANDING_DIR, exist_ok=True)
        files = glob.glob(os.path.join(LANDING_DIR, "*.csv"))
        if files:
            print(f"Landing has {len(files)} file(s) — no generation.")
            return
        df = pd.DataFrame([
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1700000000},
            {"userId": 1, "movieId": 2, "rating": 3.5, "timestamp": 1700000001},
        ])
        out = os.path.join(LANDING_DIR, "ratings_batch_synth.csv")
        df.to_csv(out, index=False)
        print(f"Generated {out}")

    gen_if_empty = PythonOperator(
        task_id="maybe_generate_batch_if_empty",
        python_callable=maybe_generate_batch_if_empty,
    )

    # --- Pick batch ---
    def pick_one_batch(**ctx):
        files = sorted(glob.glob(os.path.join(LANDING_DIR, "*.csv")))
        if not files:
            raise AirflowSkipException("No new CSV in landing.")
        batch = files[0]  # deterministic: oldest first
        print("Picked:", batch)
        ctx["ti"].xcom_push(key="batch_path", value=batch)

    pick_batch = PythonOperator(
        task_id="pick_batch",
        python_callable=pick_one_batch,
    )

    # --- Ingest batch into MySQL ---
    def ingest_mysql(**ctx):
        import os, sys, shlex, subprocess, traceback

        print("=== INGEST START ===")
        batch = ctx["ti"].xcom_pull(key="batch_path")
        print("INGEST >> batch_path =", batch)
        print("INGEST >> PROJECT_ROOT =", PROJECT_ROOT)
        print("INGEST >> LANDING_DIR  =", LANDING_DIR)
        print("INGEST >> PYTHON (sys.executable) =", sys.executable)

        if not batch:
            raise RuntimeError("No XCom 'batch_path' value")
        print("INGEST >> exists(batch)?", os.path.exists(batch))

        # prefer the same interpreter Airflow uses
        python_bin = sys.executable
        cmd = [python_bin, "-m", "src.data.create_database_mysql", "--batch-csv", batch]
        print("INGEST >> Running:", " ".join(shlex.quote(c) for c in cmd))

        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True)
            print("---- STDOUT ----\n" + (proc.stdout or ""))
            print("---- STDERR ----\n" + (proc.stderr or ""))
            print("INGEST >> returncode =", proc.returncode)
            if proc.returncode != 0:
                raise RuntimeError(f"ingest exited with code {proc.returncode}")
            print("=== INGEST OK ===")
        except Exception as e:
            print("=== INGEST FAILED ===")
            traceback.print_exc()
            raise
    ingest = PythonOperator(
    task_id="ingest_mysql",
    python_callable=ingest_mysql,
)

    # --- Train model ---
    def train_with_mlflow(**ctx):
        """
        Train model via FastAPI endpoint instead of direct subprocess call.
        """
        import requests
        from requests.auth import HTTPBasicAuth
        
        api_user = os.getenv("API_BASIC_USER", "admin")
        api_pass = os.getenv("API_BASIC_PASS", "secret")
        
        print("=== TRAINING VIA FASTAPI ===")
        print(f"Calling http://api:8000/train with user={api_user}")
        
        try:
            response = requests.post(
                "http://api:8000/train",
                auth=HTTPBasicAuth(api_user, api_pass),
                timeout=3600  # 1 hour timeout
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code != 200:
                raise RuntimeError(f"Training API returned {response.status_code}: {response.text}")
            
            result = response.json()
            print(f"Training status: {result.get('status')}")
            
            # Print stdout/stderr from API response
            if result.get("stdout"):
                print("---- TRAIN STDOUT (from API) ----")
                print(result["stdout"])
            if result.get("stderr"):
                print("---- TRAIN STDERR (from API) ----")
                print(result["stderr"])
            
            if result.get("status") != "success":
                error_msg = result.get("stderr") or result.get("message") or "Unknown error"
                raise RuntimeError(f"Training failed: {error_msg}")
            
            # Parse metrics from stdout
            stdout = result.get("stdout", "")
            last_json = None
            for line in stdout.strip().splitlines()[::-1]:
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        last_json = json.loads(line)
                        break
                    except:
                        continue
            
            if not last_json:
                raise RuntimeError("Training script didn't emit final JSON metrics")
            
            metric = last_json.get(PRIMARY_METRIC)
            if metric is None:
                from airflow.exceptions import AirflowSkipException
                raise AirflowSkipException("Training produced no PRIMARY_METRIC (rmse).")
            
            ctx["ti"].xcom_push(key="candidate_run_id", value=last_json.get("run_id"))
            ctx["ti"].xcom_push(key="candidate_metric", value=float(metric))
            
            print(f"✅ Training completed via API: run_id={last_json.get('run_id')}, {PRIMARY_METRIC}={metric}")
            
        except requests.Timeout:
            raise RuntimeError("Training API timeout after 1 hour")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to call training API: {str(e)}")
    train = PythonOperator(
        task_id="train_candidate",
        python_callable=train_with_mlflow,
    )

    # --- Compare & promote ---
    def compare_and_promote(**ctx):
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        candidate_run_id = ctx["ti"].xcom_pull(key="candidate_run_id")
        candidate_metric = float(ctx["ti"].xcom_pull(key="candidate_metric"))

        versions = client.get_registered_model(MODEL_NAME).latest_versions
        prod = [v for v in versions if (v.current_stage == "Production" or
                                        "production" in (v.aliases or []))]
        current_prod_metric = None
        current_prod_version = None
        if prod:
            pv = sorted(prod, key=lambda v: int(v.version), reverse=True)[0]
            current_prod_version = pv.version
            run = client.get_run(pv.run_id)
            m = run.data.metrics.get(PRIMARY_METRIC)
            if m is not None:
                current_prod_metric = float(m)

        def better(new, old):
            return True if old is None else (new < old if BETTER_IS_LOWER else new > old)

        cand_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        cand_v = [v for v in cand_versions if v.run_id == candidate_run_id]
        if not cand_v:
            raise RuntimeError("Candidate run not found as a registered model version.")
        cand_v = sorted(cand_v, key=lambda v: int(v.version), reverse=True)[0]

        if better(candidate_metric, current_prod_metric):
            client.set_registered_model_alias(MODEL_NAME, "production", cand_v.version)
            print(f"Promoted v{cand_v.version} ({candidate_metric}) to @production")
        else:
            print(f"Kept production v{current_prod_version} (metric={current_prod_metric})")

    select_and_promote = PythonOperator(
        task_id="compare_and_promote",
        python_callable=compare_and_promote,
    )

    # --- Archive ---
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
    gen_if_empty >> pick_batch >> ingest >> train >> select_and_promote >> archive_batch
