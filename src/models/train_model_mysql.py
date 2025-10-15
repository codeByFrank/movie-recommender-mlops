# src/models/train_model_mysql.py
from __future__ import annotations

import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------- MLflow setup (container-safe) ----------
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
os.environ["MLFLOW_TRACKING_URI"] = f"file:{AIRFLOW_HOME}/mlruns"
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("movie_reco_svd")
MODEL_NAME = "movie_recommender_svd"
BEST_TAG_KEY = "best_rmse"  # tag stored on the registered model

# ---------- DB config (Docker network) ----------
DB_HOST = os.getenv("DATABASE_HOST", "mysql-ml")
DB_PORT = os.getenv("DATABASE_PORT", "3306")
DB_USER = os.getenv("DATABASE_USER", "app")
DB_PASS = os.getenv("DATABASE_PASSWORD", "mysql")
DB_NAME = os.getenv("DATABASE_NAME", "movielens")


def _engine():
    uri = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri)


# ---------- Hyperparams ----------
N_COMPONENTS = 100
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------- Data loading ----------
def load_data_from_database() -> pd.DataFrame:
    print("Loading data from database...")
    eng = _engine()
    with eng.connect() as conn:
        _ = conn.execute(text("SHOW TABLES")).fetchall()
        query = """
        SELECT r.userId, r.movieId, r.rating, m.title, m.genres
        FROM ratings r
        JOIN movies m ON r.movieId = m.movieId
        """
        data = pd.read_sql_query(text(query), conn)
    print(f"Loaded {len(data):,} ratings from database")
    return data


def create_user_item_matrix(data: pd.DataFrame):
    print("Creating user-item matrix...")
    user_item_matrix = data.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    print(f"Matrix shape: {user_item_matrix.shape}")
    global_mean = data["rating"].mean()
    user_means = data.groupby("userId")["rating"].mean()
    movie_means = data.groupby("movieId")["rating"].mean()
    print(f"Global rating mean: {global_mean:.3f}")
    return user_item_matrix, global_mean, user_means, movie_means


def normalize_matrix_for_svd(user_item_matrix: pd.DataFrame, global_mean: float):
    print("Normalizing matrix for SVD...")
    normalized = user_item_matrix.copy()
    normalized[normalized == 0] = global_mean  # fill empties with mean
    normalized = normalized - global_mean      # mean-center
    return normalized


# ---------- Modeling ----------
def train_svd_model(normalized_matrix: pd.DataFrame, n_components: int | None = None):
    n_components = n_components or N_COMPONENTS
    print(f"Training SVD model with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(normalized_matrix)
    item_factors = svd.components_
    print("SVD model trained.")
    return svd, user_factors, item_factors


def evaluate_model(
    data, user_item_matrix, user_factors, item_factors, global_mean, test_size=TEST_SIZE
):
    print("Evaluating model (baseline + SVD)...")
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=RANDOM_STATE
    )
    train_global_mean = train_data["rating"].mean()
    train_user_bias = train_data.groupby("userId")["rating"].mean() - train_global_mean
    train_movie_bias = train_data.groupby("movieId")["rating"].mean() - train_global_mean

    preds, actuals = [], []
    # cap for speed
    for _, row in test_data.head(1000).iterrows():
        u, m, y = int(row["userId"]), int(row["movieId"]), float(row["rating"])
        if u in user_item_matrix.index and m in user_item_matrix.columns:
            ui = user_item_matrix.index.get_loc(u)
            mi = user_item_matrix.columns.get_loc(m)
            ub = float(train_user_bias.get(u, 0.0))
            mb = float(train_movie_bias.get(m, 0.0))
            svd_pred = float(np.dot(user_factors[ui], item_factors[:, mi]))
            yhat = max(0.5, min(5.0, train_global_mean + ub + mb + svd_pred))
            preds.append(yhat)
            actuals.append(y)

    if not preds:
        print("No predictions made for evaluation.")
        return None

    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = float(np.sqrt(mse))
    print(f"RMSE={rmse:.4f}  MAE={float(mae):.4f}  n={len(preds)}")
    return {"rmse": rmse, "mae": float(mae), "predictions": int(len(preds))}


def save_pickles(
    svd, user_factors, item_factors, user_item_matrix, global_mean, user_means, movie_means
) -> Path:
    """Persist the components locally; returns the folder path."""
    print("Saving model artifacts to models/ ...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    (models_dir / "svd_model.pkl").write_bytes(pickle.dumps(svd))
    (models_dir / "user_factors.pkl").write_bytes(pickle.dumps(user_factors))
    (models_dir / "item_factors.pkl").write_bytes(pickle.dumps(item_factors))
    (models_dir / "user_item_matrix.pkl").write_bytes(pickle.dumps(user_item_matrix))
    baseline = {
        "global_mean": float(global_mean),
        "user_means": user_means,
        "movie_means": movie_means,
    }
    (models_dir / "baseline_stats.pkl").write_bytes(pickle.dumps(baseline))
    return models_dir


# ---------- MLflow PyFunc wrapper ----------
class SVDRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["svd_model"], "rb") as f:
            self.svd = pickle.load(f)
        with open(context.artifacts["user_factors"], "rb") as f:
            self.user_factors = pickle.load(f)
        with open(context.artifacts["item_factors"], "rb") as f:
            self.item_factors = pickle.load(f)
        with open(context.artifacts["user_item_matrix"], "rb") as f:
            self.user_item_matrix = pickle.load(f)
        with open(context.artifacts["baseline_stats"], "rb") as f:
            bs = pickle.load(f)
        self.global_mean = bs["global_mean"]
        self.user_means = bs["user_means"]
        self.movie_means = bs["movie_means"]

    # expects a 2D array [[user_id, movie_id], ...] and returns 1D np.array
    def predict(self, context, model_input):
        # ensure numpy array
        X = np.asarray(model_input, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        out = np.zeros(X.shape[0], dtype=float)
        for i, (user_id, movie_id) in enumerate(X[:, :2].astype(int)):
            if (
                user_id not in self.user_item_matrix.index
                or movie_id not in self.user_item_matrix.columns
            ):
                out[i] = float(self.global_mean)
                continue

            ui = self.user_item_matrix.index.get_loc(user_id)
            mi = self.user_item_matrix.columns.get_loc(movie_id)
            user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean
            movie_bias = self.movie_means.get(movie_id, self.global_mean) - self.global_mean
            svd_pred = float(np.dot(self.user_factors[ui], self.item_factors[:, mi]))
            pred = float(self.global_mean + user_bias + movie_bias + svd_pred)
            out[i] = max(0.5, min(5.0, pred))
        return out


# ---------- Train entry ----------
def main():
    print("=== TRAINING MOVIE RECOMMENDATION MODEL ===")
    data = load_data_from_database()
    uim, gmean, umeans, mmeans = create_user_item_matrix(data)
    norm = normalize_matrix_for_svd(uim, gmean)
    svd, uf, if_ = train_svd_model(norm, n_components=N_COMPONENTS)
    metrics = evaluate_model(data, uim, uf, if_, gmean, test_size=TEST_SIZE)

    # Save local copies (also used for PyFunc artifacts)
    out_dir = save_pickles(svd, uf, if_, uim, gmean, umeans, mmeans)

    # ---------- MLflow logging + registry (alias based) ----------
    with mlflow.start_run(run_name=f"svd-n{N_COMPONENTS}") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {"n_components": N_COMPONENTS, "test_size": TEST_SIZE, "random_state": RANDOM_STATE}
        )
        if metrics:
            mlflow.log_metrics(metrics)

        # Log a real MLflow model at artifact_path="model"
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SVDRecommender(),
            artifacts={
                "svd_model": str(out_dir / "svd_model.pkl"),
                "user_factors": str(out_dir / "user_factors.pkl"),
                "item_factors": str(out_dir / "item_factors.pkl"),
                "user_item_matrix": str(out_dir / "user_item_matrix.pkl"),
                "baseline_stats": str(out_dir / "baseline_stats.pkl"),
            },
        )

        client = MlflowClient()
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass

        # Register the just-logged model
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )

        # Decide if it becomes the new production (alias) by RMSE
        this_rmse = float(metrics["rmse"]) if metrics else float("inf")
        current_best = None
        try:
            reg = client.get_registered_model(MODEL_NAME)
            if BEST_TAG_KEY in reg.tags:
                current_best = float(reg.tags[BEST_TAG_KEY])
        except Exception:
            pass

        if (current_best is None) or (this_rmse < current_best):
            # Move alias "production" to this version (no service restart required)
            client.set_registered_model_alias(MODEL_NAME, "production", str(mv.version))
            client.set_registered_model_tag(MODEL_NAME, BEST_TAG_KEY, f"{this_rmse}")

    print("=== MODEL TRAINING COMPLETE ===")
    return metrics


if __name__ == "__main__":
    main()
