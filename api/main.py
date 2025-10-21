from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import os, secrets
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, "/opt/airflow/repo")

from src.models.predict_model import MovieRecommender

# ---- Basic Auth setup ----
security = HTTPBasic()
API_BASIC_USER = os.getenv("API_BASIC_USER", "admin")
API_BASIC_PASS = os.getenv("API_BASIC_PASS", "secret")

def require_basic(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic credentials; return True if valid, else 401."""
    user_ok = secrets.compare_digest(credentials.username, API_BASIC_USER)
    pass_ok = secrets.compare_digest(credentials.password, API_BASIC_PASS)
    if not (user_ok and pass_ok):
        # The WWW-Authenticate header is important so clients know to prompt for creds
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# Create FastAPI app - SIMPLE VERSION
app = FastAPI(
    title="Movie Recommender API",
    description="Simple movie recommendation system (Basic Auth protected)",
    version="1.0.0"
)

# Create recommender instance
recommender = MovieRecommender()

# Request models - SIMPLE
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 5

class PredictionRequest(BaseModel):
    user_id: int
    movie_id: int

# ===== Public Endpoints =====
@app.get("/")
def home():
    """Welcome message (public)"""
    return {
        "message": "Welcome to Movie Recommender API!",
        "status": "running",
        "endpoints_public": ["/", "/health"],
        "endpoints_protected": ["/recommendations", "/predict", "/movie/{id}", "/popular", "/train", "/model/status"]
    }

@app.get("/health")
def health_check():
    info = {
        "status": "healthy",
        "models_loaded": getattr(recommender, "models_loaded", False),
    }
    if getattr(recommender, "models_loaded", False):
        info.update({
            "served_by": getattr(recommender, "_mode", None),
            "model_name": getattr(recommender, "model_name", None),
            "model_stage": getattr(recommender, "model_stage", None),
            "model_version": getattr(recommender, "model_version", None),
            "model_run_id": getattr(recommender, "model_run_id", None),
        })
    else:
        # show reason if load failed
        info["load_error"] = getattr(recommender, "_load_error", None)
    return info

# ===== Protected Endpoints (Basic Auth) =====
@app.post("/recommendations", dependencies=[Depends(require_basic)])
def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user - SIMPLE"""
    try:
        recommendations = recommender.get_user_recommendations(
            request.user_id, 
            request.n_recommendations
        )
        if isinstance(recommendations, dict) and "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/predict", dependencies=[Depends(require_basic)])
def predict_rating(request: PredictionRequest):
    """Predict rating for user-movie pair - SIMPLE"""
    try:
        prediction = recommender.predict_rating(request.user_id, request.movie_id)
        if isinstance(prediction, dict) and "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

@app.get("/movie/{movie_id}", dependencies=[Depends(require_basic)])
def get_movie_info(movie_id: int):
    """Get information about a specific movie - SIMPLE"""
    try:
        movie_info = recommender.get_movie_info(movie_id)
        return {"movie_id": movie_id, **movie_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting movie info: {str(e)}")

@app.get("/popular", dependencies=[Depends(require_basic)])
def get_popular_movies(n_movies: int = 10):
    """Get popular movies - SIMPLE"""
    try:
        popular = recommender.get_popular_movies(n_movies)
        return {"popular_movies": popular, "count": len(popular)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular movies: {str(e)}")

@app.post("/train", dependencies=[Depends(require_basic)])
def trigger_training():
    """
    Trigger model retraining (protected).
    Used by Airflow DAG for automated retraining.
    """
    try:
        import subprocess
        
        print("🚀 Starting training...")  # ← NEU
        
        # Führe das MySQL-Training-Script aus
        result = subprocess.run(
            ["python", "-m", "src.models.train_model_mysql"],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd="/opt/airflow/repo",
            env={
                **os.environ,
                "MLFLOW_TRACKING_URI": "http://mlflow-ui:5000"
            }
        )
        
        print(f"Training returncode: {result.returncode}")  # ← NEU
        print(f"STDOUT length: {len(result.stdout) if result.stdout else 0}")  # ← NEU
        print(f"STDERR length: {len(result.stderr) if result.stderr else 0}")  # ← NEU
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Model retrained successfully",
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
        else:
            print(f"❌ Training failed with code {result.returncode}")  # ← NEU
            print(f"STDERR: {result.stderr}")  # ← NEU
            return {
                "status": "error",
                "message": "Training failed",
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Training timeout after 1 hour")
    except Exception as e:
        print(f"❌ Exception: {e}")  # ← NEU
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/status", dependencies=[Depends(require_basic)])
def get_model_status():
    """
    Check current model status and version (protected).
    """
    try:
        models_dir = Path("models")
        if not models_dir.exists():
            return {"status": "no_model", "message": "No trained model found"}
        model_files = list(models_dir.glob("*.pkl"))
        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
        return {
            "status": "ready",
            "last_trained": latest_file.stat().st_mtime,
            "model_files": [f.name for f in model_files]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run the app - SIMPLE
if __name__ == "__main__":
    import uvicorn
    print("Starting Movie Recommender API with Basic Auth...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
