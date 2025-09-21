import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
import sqlite3
import pickle
import logging
import os
from pathlib import Path
import yaml
import json
from datetime import datetime

# DVC and versioning imports
try:
    import dvc.api
    import dvc.repo
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    logging.warning("DVC not available. Install with: pip install dvc")

# Monitoring imports
try:
    from evidently.dashboard import Dashboard
    from evidently.tabs import DataDriftTab, RegressionPerformanceTab
    from evidently.model_profile import Profile
    from evidently.profile_sections import DataDriftProfileSection
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Install with: pip install evidently")

# Prometheus monitoring
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
    
    # Define metrics
    REQUEST_COUNT = Counter('ml_requests_total', 'Total ML API requests')
    REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'ML API request latency')
    MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
    PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence scores')
    
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteMLOpsRecommender:
    def __init__(self, db_path="data/movielens.db", experiment_name="movie_recommender"):
        """Complete MLOps setup for Movie Recommender"""
        self.db_path = db_path
        self.experiment_name = experiment_name
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # MLflow setup
        self.setup_mlflow()
        
        # DVC setup
        if DVC_AVAILABLE:
            self.setup_dvc()
        
        # Monitoring setup
        if PROMETHEUS_AVAILABLE:
            self.setup_monitoring()
            
        # Model artifacts
        self.models = {}
        self.current_experiment_id = None
        
    def setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        # Set MLflow tracking URI (can be local or remote)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite
        # For DagsHub: mlflow.set_tracking_uri("https://dagshub.com/username/repo.mlflow")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_id}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_id}")
        
        mlflow.set_experiment(self.experiment_name)
        self.current_experiment_id = experiment_id
        
    def setup_dvc(self):
        """Setup DVC for data versioning"""
        try:
            # Initialize DVC repo if not exists
            dvc_dir = Path(".dvc")
            if not dvc_dir.exists():
                os.system("dvc init")
                logger.info("Initialized DVC repository")
            
            # Add data to DVC tracking
            data_dir = Path("data")
            if data_dir.exists() and not (data_dir / ".gitignore").exists():
                os.system(f"dvc add {data_dir}")
                logger.info("Added data directory to DVC tracking")
                
        except Exception as e:
            logger.error(f"DVC setup failed: {e}")
    
    def setup_monitoring(self):
        """Setup Prometheus monitoring"""
        if PROMETHEUS_AVAILABLE:
            # Start Prometheus metrics server
            try:
                start_http_server(8001)  # Metrics available at http://localhost:8001
                logger.info("Prometheus metrics server started on port 8001")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")
    
    def version_data(self, message="Data update"):
        """Version data using DVC"""
        if not DVC_AVAILABLE:
            logger.warning("DVC not available for data versioning")
            return
            
        try:
            # Add data changes to DVC
            os.system("dvc add data/")
            os.system("git add data.dv .gitignore")
            os.system(f'git commit -m "{message}"')
            os.system("dvc push")  # Push to remote storage (DagsHub)
            logger.info("Data versioned and pushed to DVC remote")
        except Exception as e:
            logger.error(f"Data versioning failed: {e}")
    
    def load_data(self):
        """Load data with versioning support"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT r.userId, r.movieId, r.rating, m.title, m.genres, m.year
        FROM ratings r
        JOIN movies m ON r.movieId = m.movieId
        """
        
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(self.data)} ratings")
        return self.data
    
    def train_with_mlflow(self, n_components=50, test_size=0.2):
        """Train model with full MLflow tracking"""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("algorithm", "SVD")
            mlflow.log_param("data_version", self.get_data_version())
            
            # Load and prepare data
            self.load_data()
            
            # Train-test split
            train_data, test_data = train_test_split(
                self.data, test_size=test_size, random_state=42
            )
            
            # Create user-item matrix
            train_matrix = train_data.pivot_table(
                index='userId', columns='movieId', values='rating', fill_value=0
            )
            
            # Train SVD model
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(train_matrix)
            item_factors = svd.components_
            
            # Store model components
            self.models = {
                'svd': svd,
                'user_factors': user_factors,
                'item_factors': item_factors,
                'user_item_matrix': train_matrix
            }
            
            # Evaluation
            metrics = self.evaluate_model(train_matrix, test_data)
            
            # Log metrics to MLflow
            if metrics:
                mlflow.log_metric("rmse", metrics['rmse'])
                mlflow.log_metric("mae", metrics['mae'])
                mlflow.log_metric("train_size", len(train_data))
                mlflow.log_metric("test_size", len(test_data))
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    MODEL_ACCURACY.set(1.0 / (1.0 + metrics['rmse']))  # Accuracy proxy
            
            # Log model artifacts
            mlflow.sklearn.log_model(
                svd, 
                "svd_model",
                registered_model_name="MovieRecommenderSVD"
            )
            
            # Log additional artifacts
            self.log_model_artifacts()
            
            # Log code version
            mlflow.log_param("git_commit", self.get_git_commit())
            
            logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
            
            return metrics
    
    def get_data_version(self):
        """Get current data version from DVC"""
        if DVC_AVAILABLE:
            try:
                return dvc.api.get_rev()
            except:
                return "unknown"
        return "no-dvc"
    
    def get_git_commit(self):
        """Get current git commit hash"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"
    
    def log_model_artifacts(self):
        """Log additional model artifacts to MLflow"""
        
        # Save model metadata
        metadata = {
            'model_type': 'collaborative_filtering',
            'algorithm': 'svd',
            'features': ['user_id', 'movie_id'],
            'target': 'rating',
            'created_at': datetime.now().isoformat(),
            'data_shape': self.data.shape
        }
        
        # Save as YAML
        with open("model_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
        mlflow.log_artifact("model_metadata.yaml")
        
        # Log sample predictions
        sample_predictions = self.generate_sample_predictions()
        with open("sample_predictions.json", "w") as f:
            json.dump(sample_predictions, f)
        mlflow.log_artifact("sample_predictions.json")
        
        # Cleanup temp files
        os.remove("model_metadata.yaml")
        os.remove("sample_predictions.json")
    
    def generate_sample_predictions(self, n_samples=10):
        """Generate sample predictions for logging"""
        if not self.models:
            return {}
            
        sample_users = self.data['userId'].sample(n_samples).tolist()
        predictions = {}
        
        for user_id in sample_users:
            try:
                recs = self.get_recommendations(user_id, 3)
                predictions[str(user_id)] = [
                    {'movie_id': rec['movieId'], 'title': rec['title'], 'score': rec['predicted_rating']}
                    for rec in recs
                ]
            except:
                predictions[str(user_id)] = []
                
        return predictions
    
    def evaluate_model(self, train_matrix, test_data):
        """Enhanced model evaluation"""
        
        svd = self.models['svd']
        user_factors = self.models['user_factors']
        item_factors = self.models['item_factors']
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            if (row['userId'] in train_matrix.index and 
                row['movieId'] in train_matrix.columns):
                
                user_idx = train_matrix.index.get_loc(row['userId'])
                item_idx = train_matrix.columns.get_loc(row['movieId'])
                
                pred = np.dot(user_factors[user_idx], item_factors[:, item_idx])
                predictions.append(pred)
                actuals.append(row['rating'])
        
        if predictions:
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)
            
            # Additional metrics
            coverage = len(predictions) / len(test_data)
            
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'mse': mse,
                'coverage': coverage,
                'n_predictions': len(predictions)
            }
            
            logger.info(f"Evaluation Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, Coverage={coverage:.2%}")
            return metrics
        
        return None
    
    def monitor_data_drift(self, reference_data, current_data):
        """Monitor data drift using Evidently"""
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available for drift detection")
            return None
            
        try:
            # Create data drift dashboard
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(reference_data, current_data)
            
            # Save dashboard
            dashboard.save("reports/data_drift_dashboard.html")
            
            # Generate drift profile
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(reference_data, current_data)
            
            drift_report = profile.json()
            
            # Log drift metrics to MLflow
            with mlflow.start_run():
                mlflow.log_metric("data_drift_detected", 1 if "drift" in drift_report else 0)
                mlflow.log_artifact("reports/data_drift_dashboard.html")
            
            logger.info("Data drift analysis completed")
            return drift_report
            
        except Exception as e:
            logger.error(f"Data drift monitoring failed: {e}")
            return None
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations with monitoring"""
        
        # Prometheus monitoring
        if PROMETHEUS_AVAILABLE:
            REQUEST_COUNT.inc()
            request_start = datetime.now()
        
        try:
            if not self.models:
                raise ValueError("No trained model available")
            
            user_item_matrix = self.models['user_item_matrix']
            
            if user_id not in user_item_matrix.index:
                return self._get_popular_recommendations(n_recommendations)
            
            user_idx = user_item_matrix.index.get_loc(user_id)
            user_vector = self.models['user_factors'][user_idx]
            
            # Calculate predictions
            predictions = np.dot(user_vector, self.models['item_factors'])
            
            # Get unrated movies
            user_ratings = user_item_matrix.loc[user_id]
            unrated_movies = user_ratings[user_ratings == 0].index
            
            # Create recommendations
            recommendations = []
            movie_ids = user_item_matrix.columns
            
            for movie_id in unrated_movies:
                movie_idx = movie_ids.get_loc(movie_id)
                score = predictions[movie_idx]
                recommendations.append((movie_id, score))
            
            # Sort and get top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            top_recs = recommendations[:n_recommendations]
            
            # Format results with movie details
            results = []
            conn = sqlite3.connect(self.db_path)
            
            for movie_id, score in top_recs:
                movie_info = conn.execute(
                    "SELECT title, genres FROM movies WHERE movieId = ?", 
                    (movie_id,)
                ).fetchone()
                
                if movie_info:
                    results.append({
                        'movieId': movie_id,
                        'title': movie_info[0],
                        'genres': movie_info[1],
                        'predicted_rating': score
                    })
            
            conn.close()
            
            # Log confidence metrics
            if PROMETHEUS_AVAILABLE:
                avg_confidence = np.mean([rec['predicted_rating'] for rec in results])
                PREDICTION_CONFIDENCE.observe(avg_confidence)
                
                request_duration = (datetime.now() - request_start).total_seconds()
                REQUEST_LATENCY.observe(request_duration)
            
            return results
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _get_popular_recommendations(self, n_recommendations=10):
        """Fallback popular recommendations"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT m.movieId, m.title, m.genres, AVG(r.rating) as avg_rating, COUNT(r.rating) as rating_count
        FROM movies m
        JOIN ratings r ON m.movieId = r.movieId
        GROUP BY m.movieId
        HAVING rating_count >= 50
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT ?
        """
        
        results = []
        for row in conn.execute(query, (n_recommendations,)):
            results.append({
                'movieId': row[0],
                'title': row[1],
                'genres': row[2],
                'avg_rating': row[3],
                'rating_count': row[4]
            })
        
        conn.close()
        return results
    
    def create_model_registry(self):
        """Create and manage model registry"""
        
        # Register best model
        model_name = "MovieRecommenderProduction"
        
        try:
            # Create registered model
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/svd_model",
                model_name
            )
            
            # Transition to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=1,
                stage="Staging"
            )
            
            logger.info(f"Model registered: {model_name}")
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")

def main():
    """Main function with complete MLOps pipeline"""
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Initialize complete MLOps recommender
    recommender = CompleteMLOpsRecommender(
        experiment_name="movie_recommender_complete"
    )
    
    # Version data
    recommender.version_data("Initial data version")
    
    # Train with full tracking
    logger.info("Starting MLOps training pipeline...")
    metrics = recommender.train_with_mlflow(n_components=50)
    
    # Register model
    recommender.create_model_registry()
    
    # Test recommendations
    sample_user = 1
    recommendations = recommender.get_recommendations(sample_user, 5)
    
    print(f"\nRecommendations for user {sample_user}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Score: {rec.get('predicted_rating', 'N/A'):.2f})")
    
    # Simulate data drift monitoring (with dummy data)
    if len(recommender.data) > 1000:
        reference_data = recommender.data.sample(500)
        current_data = recommender.data.sample(500)
        recommender.monitor_data_drift(reference_data, current_data)
    
    logger.info("Complete MLOps pipeline executed successfully!")
    
    print("\n" + "="*50)
    print("MLOPS COMPONENTS INTEGRATED:")
    print("✅ MLflow - Experiment Tracking & Model Registry")
    print("✅ DVC - Data Versioning" if DVC_AVAILABLE else "❌ DVC - Not installed")
    print("✅ Prometheus - Monitoring" if PROMETHEUS_AVAILABLE else "❌ Prometheus - Not installed") 
    print("✅ Evidently - Drift Detection" if EVIDENTLY_AVAILABLE else "❌ Evidently - Not installed")
    print("✅ Git - Code Versioning")
    print("✅ SQLite - Data Storage")
    print("\nNext: Add Airflow, BentoML, Docker containers")
    print("="*50)

if __name__ == "__main__":
    main()