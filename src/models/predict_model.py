# src/models/predict_model.py
import os
from pathlib import Path
import pickle
import numpy as np
import mysql.connector
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info


class MovieRecommender:
    def __init__(self):
        # Database configuration
        self.db_host = os.getenv("DB_HOST", "mysql-ml")
        self.db_port = int(os.getenv("DB_PORT", "3306"))
        self.db_user = os.getenv("DB_USER", "app")
        self.db_pass = os.getenv("DB_PASS", "mysql")
        self.db_name = os.getenv("DB_NAME", "movielens")
        
        # MLflow configuration
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/opt/airflow/mlruns"))
        self.model_name = os.getenv("MODEL_NAME", "movie_recommender_svd")
        override_uri = os.getenv("MODEL_URI")  # e.g. models:/movie_recommender_svd@production

        try:
            uri = override_uri or f"models:/{self.model_name}@production"
            print(f"[MovieRecommender] Loading model from URI: {uri}")
            self.model = mlflow.pyfunc.load_model(uri)

            # Health/metadata (alias-based)
            client = MlflowClient()
            self.models_loaded = True
            self._mode = "mlflow"
            self.model_stage = "alias:production"  # for display only

            # Resolve exact version/run pinned by the alias
            try:
                mv = client.get_model_version_by_alias(self.model_name, "production")
                self.model_version = mv.version
                self.model_run_id = mv.run_id
            except Exception:
                # Fallback: ask the loaded artifact for its run_id if available
                info = get_model_info(uri)
                self.model_version = getattr(info, "version", None)
                self.model_run_id = getattr(info, "run_id", None)

        except MlflowException as e:
            self.models_loaded = False
            self._mode = None
            self._load_error = str(e)
            print(f"[MovieRecommender] MLflow load failed: {e}")

    # ------------- DB helpers -------------
    def _connect(self):
        return mysql.connector.connect(
            host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name,ssl_disabled=False
        )

    def _all_movie_ids(self):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT movieId FROM movies")
            return [int(r[0]) for r in cur.fetchall()]
        finally:
            cur.close()
            conn.close()

    def _rated_movie_ids(self, user_id: int):
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT movieId FROM ratings WHERE userId=%s", (user_id,))
            return {int(r[0]) for r in cur.fetchall()}
        finally:
            cur.close()
            conn.close()

    # ------------- Local fallback loader -------------
    def _load_local_models(self):
        current_dir = Path.cwd()
        if (current_dir / "models").exists():
            models_dir = current_dir / "models"
        elif (current_dir.parent / "models").exists():
            models_dir = current_dir.parent / "models"
        else:
            script_dir = Path(__file__).parent
            models_dir = script_dir.parent.parent / "models"

        if not models_dir.exists():
            raise RuntimeError("Local models/ directory not found for fallback.")

        with open(models_dir / "svd_model.pkl", "rb") as f:
            self.svd = pickle.load(f)
        with open(models_dir / "user_factors.pkl", "rb") as f:
            self.user_factors = pickle.load(f)
        with open(models_dir / "item_factors.pkl", "rb") as f:
            self.item_factors = pickle.load(f)
        with open(models_dir / "user_item_matrix.pkl", "rb") as f:
            self.user_item_matrix = pickle.load(f)

        try:
            with open(models_dir / "baseline_stats.pkl", "rb") as f:
                baseline_stats = pickle.load(f)
            self.global_mean = float(baseline_stats["global_mean"])
            self.user_means = baseline_stats["user_means"]
            self.movie_means = baseline_stats["movie_means"]
        except FileNotFoundError:
            self.global_mean = 3.5
            self.user_means = {}
            self.movie_means = {}

        print("[MovieRecommender] Loaded local pickle models (fallback).")

    # ------------- Public API -------------
    def predict_rating(self, user_id: int, movie_id: int):
        """Predict rating using MLflow model if available, otherwise local fallback."""
        if not self.models_loaded:
            return {"error": "Models not loaded"}

        if self._mode == "mlflow":
            pred = self.model.predict(np.array([[user_id, movie_id]], dtype=float)).item()
            info = self.get_movie_info(movie_id)
            return {
                "user_id": int(user_id),
                "movie_id": int(movie_id),
                "predicted_rating": float(pred),
                "title": info.get("title", "Unknown"),
                "genres": info.get("genres", "Unknown"),
                "served_by": "mlflow",
            }

        # Local SVD + baseline (fallback)
        if user_id not in self.user_item_matrix.index:
            return {"error": f"User {user_id} not found"}
        if movie_id not in self.user_item_matrix.columns:
            return {"error": f"Movie {movie_id} not found"}

        ui = self.user_item_matrix.index.get_loc(user_id)
        mi = self.user_item_matrix.columns.get_loc(movie_id)
        user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean
        movie_bias = self.movie_means.get(movie_id, self.global_mean) - self.global_mean
        svd_pred = np.dot(self.user_factors[ui], self.item_factors[:, mi])
        pred = float(np.clip(self.global_mean + user_bias + movie_bias + svd_pred, 0.5, 5.0))
        info = self.get_movie_info(movie_id)
        return {
            "user_id": int(user_id),
            "movie_id": int(movie_id),
            "predicted_rating": pred,
            "title": info.get("title", "Unknown"),
            "genres": info.get("genres", "Unknown"),
            "components": {
                "global_mean": float(self.global_mean),
                "user_bias": float(user_bias),
                "movie_bias": float(movie_bias),
                "svd_component": float(svd_pred),
            },
            "served_by": "local",
        }

    def get_user_recommendations(
        self,
        user_id: int,
        n_recommendations: int = 5,
        max_candidates: int = 5000,
        batch_size: int = 2048,
        min_popularity_count: int = 20,   # require at least N ratings to treat a movie as "popular"
    ):
        """
        Recommend top-N items for a user.
        - If running with an MLflow model, batch-score [user_id, movie_id] pairs.
        - If the user is cold-start (no ratings), fall back to a popularity list.
        - If MLflow isn't loaded, use the local SVD fallback.
        Returns: List[ {movie_id, title, genres, predicted_rating} ]
        """

        if not getattr(self, "models_loaded", False):
            return {"error": "Models not loaded"}

        # ---- small helper: popularity fallback (for cold-start or empty candidates)
        def _popular_from_db(limit: int):
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name,ssl_disabled=False
                )
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT m.movieId, m.title, m.genres, AVG(r.rating) AS avg_rating, COUNT(*) AS c
                    FROM ratings r
                    JOIN movies m ON m.movieId = r.movieId
                    GROUP BY m.movieId, m.title, m.genres
                    HAVING c >= %s
                    ORDER BY avg_rating DESC
                    LIMIT %s
                    """,
                    (min_popularity_count, limit),
                )
                rows = cur.fetchall()
            finally:
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass

            return [
                {
                    "movie_id": int(mid),
                    "title": title,
                    "genres": genres,
                    "predicted_rating": float(avg)
                }
                for (mid, title, genres, avg, _cnt) in rows
            ]

        # ---------- MLflow MODE ----------
        if getattr(self, "_mode", None) == "mlflow":
            # 0) Cold-start guard: does this user have any ratings?
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name,ssl_disabled=False
                )
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ratings WHERE userId=%s", (user_id,))
                n_user_ratings = int(cur.fetchone()[0])
            finally:
                try:
                    cur.close(); conn.close()
                except Exception:
                    pass

            if n_user_ratings == 0:
                # New user → popularity list
                return _popular_from_db(n_recommendations)

            # 1) Candidate set: movies this user hasn't rated yet
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name,ssl_disabled=False
                )
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT m.movieId
                    FROM movies m
                    LEFT JOIN ratings r
                    ON r.movieId = m.movieId AND r.userId = %s
                    WHERE r.userId IS NULL
                    LIMIT %s
                    """,
                    (user_id, max_candidates),
                )
                candidate_ids = [int(row[0]) for row in cur.fetchall()]
            finally:
                try:
                    cur.close(); conn.close()
                except Exception:
                    pass

            if not candidate_ids:
                # Nothing unrated found (rare) → popularity fallback
                return _popular_from_db(n_recommendations)

            # 2) Batch predict [user_id, movie_id]
            preds = []
            X_all = np.array([[user_id, mid] for mid in candidate_ids], dtype=float)
            for start in range(0, len(X_all), batch_size):
                X = X_all[start:start + batch_size]
                y = self.model.predict(X).reshape(-1)
                for mid, score in zip(candidate_ids[start:start + batch_size], y):
                    score = float(max(0.5, min(5.0, score)))
                    preds.append((mid, score))

            # 3) Top-N by score
            preds.sort(key=lambda t: t[1], reverse=True)
            top = preds[:n_recommendations]
            if not top:
                return _popular_from_db(n_recommendations)

            # 4) Hydrate titles/genres in one go
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name,ssl_disabled=False
                )
                cur = conn.cursor()
                placeholders = ",".join(["%s"] * len(top))
                cur.execute(
                    f"SELECT movieId, title, genres FROM movies WHERE movieId IN ({placeholders})",
                    tuple(mid for mid, _ in top),
                )
                meta = {int(r[0]): (r[1], r[2]) for r in cur.fetchall()}
            finally:
                try:
                    cur.close(); conn.close()
                except Exception:
                    pass

            results = []
            for mid, score in top:
                title, genres = meta.get(mid, (f"Movie {mid}", "Unknown"))
                results.append({
                    "movie_id": mid,
                    "title": title,
                    "genres": genres,
                    "predicted_rating": score
                })
            return results

        # ---------- LOCAL SVD FALLBACK ----------
        # (Uses the matrices loaded by _load_local_models)
        if not hasattr(self, "user_item_matrix"):
            return {"error": "Local model not available"}

        # If the user is unknown locally, return popularity fallback
        if user_id not in self.user_item_matrix.index:
            return _popular_from_db(n_recommendations)

        # Vectorized scoring for all movies, then drop those already rated
        ui = self.user_item_matrix.index.get_loc(user_id)
        user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean

        # dot product with all items
        svd_scores = np.dot(self.user_factors[ui], self.item_factors)

        # base = global + user_bias + movie_bias
        movie_biases = np.array([
            self.movie_means.get(mid, self.global_mean) - self.global_mean
            for mid in self.user_item_matrix.columns
        ])
        scores = self.global_mean + user_bias + movie_biases + svd_scores

        # clamp to [0.5, 5.0]
        scores = np.clip(scores, 0.5, 5.0)

        # mask already-rated
        rated_mask = self.user_item_matrix.iloc[ui].to_numpy() > 0
        scores[rated_mask] = -np.inf

        # top-N indices
        top_idx = np.argpartition(-scores, range(min(n_recommendations, len(scores))))[:n_recommendations]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        top_movie_ids = [int(self.user_item_matrix.columns[i]) for i in top_idx if scores[i] != -np.inf]
        top_scores    = [float(scores[i]) for i in top_idx if scores[i] != -np.inf]

        if not top_movie_ids:
            return _popular_from_db(n_recommendations)

        # hydrate from DB
        try:
            conn = mysql.connector.connect(
                host=self.db_host, user=self.db_user, password=self.db_pass, database=self.db_name, ssl_disabled=False
            )
            cur = conn.cursor()
            placeholders = ",".join(["%s"] * len(top_movie_ids))
            cur.execute(
                f"SELECT movieId, title, genres FROM movies WHERE movieId IN ({placeholders})",
                tuple(top_movie_ids),
            )
            meta = {int(r[0]): (r[1], r[2]) for r in cur.fetchall()}
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass

        results = []
        for mid, score in zip(top_movie_ids, top_scores):
            title, genres = meta.get(mid, (f"Movie {mid}", "Unknown"))
            results.append({
                "movie_id": mid,
                "title": title,
                "genres": genres,
                "predicted_rating": score
            })
        return results

    def get_movie_info(self, movie_id: int):
        """Fetch title/genres from MySQL."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT title, genres FROM movies WHERE movieId = %s", (movie_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return {"title": row[0], "genres": row[1]}
            return {"title": f"Movie {movie_id}", "genres": "Unknown"}
        except Exception as e:
            print(f"Error getting movie info: {e}")
            return {"title": f"Movie {movie_id}", "genres": "Unknown"}

    def get_popular_movies(self, n_movies: int = 10):
        """Return high-average, sufficiently-rated movies (DB-based)."""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """
                SELECT m.movieId, m.title, m.genres,
                       AVG(r.rating) AS avg_rating, COUNT(*) AS cnt
                FROM ratings r
                JOIN movies m ON r.movieId = m.movieId
                GROUP BY m.movieId, m.title, m.genres
                HAVING cnt >= 20
                ORDER BY avg_rating DESC
                LIMIT %s
                """,
                (n_movies,),
            )
            rows = cur.fetchall()
            cur.close()
            conn.close()

            return [
                {
                    "movie_id": int(r[0]),
                    "title": r[1],
                    "genres": r[2],
                    "avg_rating": float(round(r[3], 2)),
                    "num_ratings": int(r[4]),
                    "predicted_rating": float(r[3]),
                }
                for r in rows
            ]
        except Exception as e:
            print(f"Error getting popular movies: {e}")
            return []
