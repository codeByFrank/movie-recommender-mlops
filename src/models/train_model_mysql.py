# src/models/train_model_mysql.py
from __future__ import annotations

import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix

# ---------- MLflow setup ----------
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("movie_reco_svd")
MODEL_NAME = "movie_recommender_svd"

# ---------- DB config ----------
DB_HOST = os.getenv("DATABASE_HOST", "mysql-ml")
DB_PORT = os.getenv("DATABASE_PORT", "3306")
DB_USER = os.getenv("DATABASE_USER", "app")
DB_PASS = os.getenv("DATABASE_PASSWORD", "mysql")
DB_NAME = os.getenv("DATABASE_NAME", "movielens")

def _engine():
    uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri, pool_pre_ping=True, future=True)

# ---------- Hyperparams & runtime caps (env-overridable) ----------
N_COMPONENTS = int(os.getenv("N_COMPONENTS", "50"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# deterministic subsample: keep rows where MOD(userId, BASE) == REM
USER_MOD_BASE = int(os.getenv("TRAIN_USER_MOD_BASE", "20"))     # 20 => ~5%
USER_MOD_REM  = int(os.getenv("TRAIN_USER_MOD_REMAINDER", "0"))

# caps to bound runtime on huge tables
TRAIN_MAX_ROWS = int(os.getenv("TRAIN_MAX_ROWS", "4000000"))    # max ratings to ingest
CHUNK_SIZE     = int(os.getenv("TRAIN_CHUNK_SIZE", "200000"))   # SQL LIMIT per page
EVAL_CAP       = int(os.getenv("EVAL_CAP", "1000"))             # eval pairs

# ---------- Utilities ----------
def _fetch_distinct_ids(eng):
    with eng.connect() as c:
        users = pd.read_sql_query(
            text("SELECT DISTINCT userId FROM ratings WHERE MOD(userId, :b)=:r"),
            c, params={"b": USER_MOD_BASE, "r": USER_MOD_REM}
        )["userId"].astype(int).tolist()
        movies = pd.read_sql_query(
            text("SELECT DISTINCT movieId FROM ratings"),
            c
        )["movieId"].astype(int).tolist()
    return users, movies

def _build_sparse_matrix(eng, uid2ix, mid2ix):
    rows, cols, vals = [], [], []
    total = 0
    running_sum = 0.0
    running_count = 0

    offset = 0
    while True:
        with eng.connect() as c:
            df = pd.read_sql_query(
                text("""SELECT userId, movieId, rating
                        FROM ratings
                        WHERE MOD(userId, :b)=:r
                        ORDER BY userId, movieId
                        LIMIT :lim OFFSET :off"""),
                c,
                params={"b": USER_MOD_BASE, "r": USER_MOD_REM,
                        "lim": CHUNK_SIZE, "off": offset}
            )
        if df.empty:
            break

        # keep only ids we know
        df = df[df["userId"].isin(uid2ix) & df["movieId"].isin(mid2ix)]
        if not df.empty:
            ui = df["userId"].map(uid2ix).values
            mi = df["movieId"].map(mid2ix).values
            rt = df["rating"].astype(float).values

            rows.extend(ui.tolist())
            cols.extend(mi.tolist())
            vals.extend(rt.tolist())

            running_sum += float(rt.sum())
            running_count += int(rt.size)

        total += int(len(df))
        offset += CHUNK_SIZE
        if total >= TRAIN_MAX_ROWS:
            break

    if running_count == 0:
        raise RuntimeError("No ratings collected for training (sampling too strict?).")

    global_mean = running_sum / running_count

    # mean-center values for SVD
    vals_centered = np.asarray(vals, dtype=float) - float(global_mean)
    X = csr_matrix(
        (vals_centered,
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(uid2ix), len(mid2ix))
    )
    return X, float(global_mean), total

def _sample_eval_pairs(eng, uid2ix, mid2ix, cap=EVAL_CAP):
    with eng.connect() as c:
        df = pd.read_sql_query(
            text("""SELECT userId, movieId, rating
                    FROM ratings
                    WHERE MOD(userId, :b)=:r
                    ORDER BY `timestamp` DESC
                    LIMIT :cap"""),
            c, params={"b": USER_MOD_BASE, "r": USER_MOD_REM, "cap": cap*3}
        )
    df = df[df["userId"].isin(uid2ix) & df["movieId"].isin(mid2ix)]
    if df.empty:
        return None
    return df.head(cap).copy()

# ---------- MLflow PyFunc wrapper ----------
class SVDRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["svd_model"], "rb") as f:
            self.svd = pickle.load(f)
        with open(context.artifacts["user_factors"], "rb") as f:
            self.user_factors = pickle.load(f)
        with open(context.artifacts["item_factors"], "rb") as f:
            self.item_factors = pickle.load(f)
        with open(context.artifacts["baseline_stats"], "rb") as f:
            bs = pickle.load(f)
        with open(context.artifacts["id_maps"], "rb") as f:
            maps = pickle.load(f)

        self.global_mean = float(bs["global_mean"])
        self.uid2ix = maps["uid2ix"]
        self.mid2ix = maps["mid2ix"]

    def predict(self, context, model_input):
        X = np.asarray(model_input, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros(X.shape[0], dtype=float)
        for i, (u, m) in enumerate(X[:, :2].astype(int)):
            ui = self.uid2ix.get(int(u)); mi = self.mid2ix.get(int(m))
            if ui is None or mi is None:
                out[i] = self.global_mean
                continue
            svd_pred = float(np.dot(self.user_factors[ui], self.item_factors[:, mi]))
            out[i] = max(0.5, min(5.0, self.global_mean + svd_pred))
        return out

# ---------- Train entry ----------
def main():
    print("=== TRAINING MOVIE RECOMMENDATION MODEL (sparse) ===")
    print(f"Sampling: MOD(userId, {USER_MOD_BASE}) = {USER_MOD_REM}")
    print(f"Caps: TRAIN_MAX_ROWS={TRAIN_MAX_ROWS}, CHUNK_SIZE={CHUNK_SIZE}, EVAL_CAP={EVAL_CAP}")
    eng = _engine()

    # id maps
    users, movies = _fetch_distinct_ids(eng)
    print(f"Unique users (sampled): {len(users):,}  movies: {len(movies):,}")
    uid2ix = {u: i for i, u in enumerate(users)}
    mid2ix = {m: i for i, m in enumerate(movies)}

    # sparse matrix
    X, global_mean, collected = _build_sparse_matrix(eng, uid2ix, mid2ix)
    print(f"CSR shape={X.shape}, nnz={X.nnz:,}, collected={collected:,}, global_mean={global_mean:.4f}")

    # train SVD
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_
    print("SVD model trained.")

    # eval (fast)
    eval_df = _sample_eval_pairs(eng, uid2ix, mid2ix, cap=EVAL_CAP)
    metrics = None
    if eval_df is not None and not eval_df.empty:
        preds, actuals = [], []
        for _, r in eval_df.iterrows():
            ui = uid2ix.get(int(r.userId)); mi = mid2ix.get(int(r.movieId))
            if ui is None or mi is None:
                continue
            svd_pred = float(np.dot(user_factors[ui], item_factors[:, mi]))
            yhat = max(0.5, min(5.0, global_mean + svd_pred))
            preds.append(yhat); actuals.append(float(r.rating))
        if preds:
            mse = mean_squared_error(actuals, preds)
            mae = mean_absolute_error(actuals, preds)
            rmse = float(np.sqrt(mse))
            metrics = {"rmse": rmse, "mae": float(mae), "predictions": int(len(preds))}
            print(f"Eval: RMSE={rmse:.4f}  MAE={float(mae):.4f}  n={len(preds)}")
        else:
            print("No overlapping eval pairs to score.")
    else:
        print("No eval sample available.")

    # artifacts for pyfunc
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    (models_dir / "svd_model.pkl").write_bytes(pickle.dumps(svd))
    (models_dir / "user_factors.pkl").write_bytes(pickle.dumps(user_factors))
    (models_dir / "item_factors.pkl").write_bytes(pickle.dumps(item_factors))
    baseline = {"global_mean": float(global_mean)}
    (models_dir / "baseline_stats.pkl").write_bytes(pickle.dumps(baseline))
    id_maps = {"uid2ix": uid2ix, "mid2ix": mid2ix}
    (models_dir / "id_maps.pkl").write_bytes(pickle.dumps(id_maps))

    # ---------- MLflow logging + registry ----------
    with mlflow.start_run(run_name=f"svd-sparse-n{N_COMPONENTS}") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "n_components": N_COMPONENTS,
            "random_state": RANDOM_STATE,
            "user_mod_base": USER_MOD_BASE,
            "user_mod_remainder": USER_MOD_REM,
            "train_max_rows": TRAIN_MAX_ROWS,
        })
        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SVDRecommender(),
            artifacts={
                "svd_model": str(models_dir / "svd_model.pkl"),
                "user_factors": str(models_dir / "user_factors.pkl"),
                "item_factors": str(models_dir / "item_factors.pkl"),
                "baseline_stats": str(models_dir / "baseline_stats.pkl"),
                "id_maps": str(models_dir / "id_maps.pkl"),
            },
        )

        client = MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass

        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )

        # one-line JSON for Airflow
        summary = {
            "run_id": run_id,
            "rmse": float(metrics["rmse"]) if metrics else None,
            "mae": float(metrics["mae"]) if metrics else None,
            "version": int(mv.version),
        }
        print(json.dumps(summary))

if __name__ == "__main__":
    main()
