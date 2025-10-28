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
            host=self.db_host, 
            user=self.db_user, 
            password=self.db_pass, 
            database=self.db_name,
            ssl_disabled=False
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
        min_popularity_count: int = 20,
        min_user_ratings_for_personalization: int = 5,  # ← KEY: Min ratings for personalization
    ):
        """
        Recommend top-N items for a user with improved cold start handling.
        
        Cold Start Strategy:
        - If user has < 5 ratings → Return popular movies (4.46★)
        - If user has ≥ 5 ratings → Return personalized predictions
        """
        
        if not getattr(self, "models_loaded", False):
            return {"error": "Models not loaded"}

        # ---- Helper: Get popular movies (for cold start) ----
        def _popular_from_db(limit: int):
            """Get top-rated popular movies"""
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, 
                    user=self.db_user, 
                    password=self.db_pass, 
                    database=self.db_name,
                    ssl_disabled=False
                )
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT m.movieId, m.title, m.genres, 
                           AVG(r.rating) AS avg_rating, 
                           COUNT(*) AS num_ratings
                    FROM ratings r
                    JOIN movies m ON m.movieId = r.movieId
                    GROUP BY m.movieId, m.title, m.genres
                    HAVING num_ratings >= %s
                    ORDER BY avg_rating DESC, num_ratings DESC
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
                    "predicted_rating": float(avg),
                    "num_ratings": int(cnt),
                    "served_by": "popularity_fallback"  # Debug flag
                }
                for (mid, title, genres, avg, cnt) in rows
            ]

        # ---------- MLflow MODE ----------
        if getattr(self, "_mode", None) == "mlflow":
            
            # 0) Check: How many ratings does this user have?
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, 
                    user=self.db_user, 
                    password=self.db_pass, 
                    database=self.db_name,
                    ssl_disabled=False
                )
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM ratings WHERE userId=%s", 
                    (user_id,)
                )
                n_user_ratings = int(cur.fetchone()[0])
            finally:
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass

            # ✅ CRITICAL FIX: Use threshold, not == 0
            if n_user_ratings < min_user_ratings_for_personalization:
                print(f"[Cold Start] User {user_id} has only {n_user_ratings} ratings")
                print(f"[Cold Start] Returning {n_recommendations} popular movies instead of predictions")
                
                popular = _popular_from_db(n_recommendations)
                
                if popular:
                    return popular
                else:
                    print("[Error] No popular movies found in database!")
                    return {"error": "No popular movies available"}

            # ---- User has enough ratings → Personalized recommendations ----
            
            # 1) Get candidate movies (not yet rated by this user)
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, 
                    user=self.db_user, 
                    password=self.db_pass, 
                    database=self.db_name,
                    ssl_disabled=False
                )
                cur = conn.cursor()
                
                # Get popular candidates first (better than random)
                cur.execute(
                    """
                    SELECT m.movieId, COUNT(r_all.rating) as popularity
                    FROM movies m
                    LEFT JOIN ratings r_all ON r_all.movieId = m.movieId
                    LEFT JOIN ratings r_user ON r_user.movieId = m.movieId 
                                             AND r_user.userId = %s
                    WHERE r_user.userId IS NULL
                    GROUP BY m.movieId
                    HAVING popularity >= %s
                    ORDER BY popularity DESC
                    LIMIT %s
                    """,
                    (user_id, min_popularity_count, max_candidates),
                )
                candidate_ids = [int(row[0]) for row in cur.fetchall()]
            finally:
                try:
                    cur.close()
                    conn.close()
                except Exception:
                    pass

            if not candidate_ids:
                print(f"[Fallback] No unrated candidates for user {user_id}")
                print(f"[Fallback] Returning popular movies instead")
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

            # 3) Sort by predicted score and take top-N
            preds.sort(key=lambda t: t[1], reverse=True)
            top = preds[:n_recommendations]
            
            if not top:
                print(f"[Fallback] No predictions generated for user {user_id}")
                return _popular_from_db(n_recommendations)

            # 4) Hydrate with movie titles and genres
            try:
                conn = mysql.connector.connect(
                    host=self.db_host, 
                    user=self.db_user, 
                    password=self.db_pass, 
                    database=self.db_name,
                    ssl_disabled=False
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
                    cur.close()
                    conn.close()
                except Exception:
                    pass

            results = []
            for mid, score in top:
                title, genres = meta.get(mid, (f"Movie {mid}", "Unknown"))
                results.append({
                    "movie_id": mid,
                    "title": title,
                    "genres": genres,
                    "predicted_rating": score,
                    "served_by": "personalized_mlflow"  # Debug flag
                })
            return results

        # ---------- LOCAL SVD FALLBACK ----------
        if not hasattr(self, "user_item_matrix"):
            return {"error": "Local model not available"}

        # Check if user exists in local matrix
        if user_id not in self.user_item_matrix.index:
            print(f"[Cold Start - Local] User {user_id} not in matrix → Popular movies")
            return _popular_from_db(n_recommendations)

        # Check how many ratings user has (in the matrix)
        user_ratings = (self.user_item_matrix.loc[user_id] > 0).sum()
        if user_ratings < min_user_ratings_for_personalization:
            print(f"[Cold Start - Local] User {user_id} has only {user_ratings} ratings → Popular movies")
            return _popular_from_db(n_recommendations)

        # ---- Personalized recommendations with local SVD ----
        ui = self.user_item_matrix.index.get_loc(user_id)
        user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean

        # Dot product with all items
        svd_scores = np.dot(self.user_factors[ui], self.item_factors)

        # Add biases
        movie_biases = np.array([
            self.movie_means.get(mid, self.global_mean) - self.global_mean
            for mid in self.user_item_matrix.columns
        ])
        scores = self.global_mean + user_bias + movie_biases + svd_scores
        scores = np.clip(scores, 0.5, 5.0)

        # Mask already-rated movies
        rated_mask = self.user_item_matrix.iloc[ui].to_numpy() > 0
        scores[rated_mask] = -np.inf

        # Get top-N
        top_idx = np.argpartition(-scores, range(min(n_recommendations, len(scores))))[:n_recommendations]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        top_movie_ids = [int(self.user_item_matrix.columns[i]) for i in top_idx if scores[i] != -np.inf]
        top_scores = [float(scores[i]) for i in top_idx if scores[i] != -np.inf]

        if not top_movie_ids:
            print(f"[Fallback - Local] No unrated movies for user {user_id}")
            return _popular_from_db(n_recommendations)

        # Hydrate from DB
        try:
            conn = mysql.connector.connect(
                host=self.db_host, 
                user=self.db_user, 
                password=self.db_pass, 
                database=self.db_name, 
                ssl_disabled=False
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
                cur.close()
                conn.close()
            except Exception:
                pass

        results = []
        for mid, score in zip(top_movie_ids, top_scores):
            title, genres = meta.get(mid, (f"Movie {mid}", "Unknown"))
            results.append({
                "movie_id": mid,
                "title": title,
                "genres": genres,
                "predicted_rating": score,
                "served_by": "personalized_local_svd"  # Debug flag
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