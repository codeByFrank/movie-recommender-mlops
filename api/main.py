from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import os, secrets
import sys
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# ===== IMPROVED: Async Model Loading =====
recommender = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, but don't block the API"""
    global recommender
    print("🚀 API starting up...")
    
    # Start model loading in background
    try:
        print("📦 Loading model...")
        recommender = MovieRecommender()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("API will start anyway - use /train to create model")
        recommender = MovieRecommender()  # Empty recommender
    
    yield  # API is now ready
    
    print("👋 API shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Movie Recommender API",
    description="Movie recommendation system with MLOps pipeline (Basic Auth protected)",
    version="1.0.0",
    lifespan=lifespan  # ← NEW: Async startup
)

# Request models
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
        "models_loaded": getattr(recommender, "models_loaded", False) if recommender else False,
        "endpoints_public": ["/", "/health"],
        "endpoints_protected": ["/recommendations", "/predict", "/movie/{id}", "/popular", "/train", "/model/status"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint - always responds quickly"""
    if not recommender:
        return {
            "status": "starting",
            "models_loaded": False,
            "message": "API is starting, model loading in progress"
        }
    
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
        info["load_error"] = getattr(recommender, "_load_error", None)
    
    return info

# ===== Helper function to check if model is ready =====
def ensure_model_loaded():
    """Check if model is loaded, raise error if not"""
    if not recommender:
        raise HTTPException(
            status_code=503,
            detail="API is starting up, please wait a moment and try again"
        )
    if not getattr(recommender, "models_loaded", False):
        raise HTTPException(
            status_code=404,
            detail="Models not loaded. Train a model first using /train endpoint or via Airflow DAG"
        )

# ===== Protected Endpoints (Basic Auth) =====
@app.post("/recommendations", dependencies=[Depends(require_basic)])
def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    ensure_model_loaded()  # ← NEW: Check before processing
    
    try:
        print(f"🔍 Getting recommendations for user {request.user_id}")
        recommendations = recommender.get_user_recommendations(
            request.user_id, 
            request.n_recommendations
        )
        print(f"✅ Got {len(recommendations) if isinstance(recommendations, list) else 0} recommendations")
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exception in recommendations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/predict", dependencies=[Depends(require_basic)])
def predict_rating(request: PredictionRequest):
    """Predict rating for user-movie pair"""
    ensure_model_loaded()  # ← NEW
    
    try:
        prediction = recommender.predict_rating(request.user_id, request.movie_id)
        if isinstance(prediction, dict) and "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

@app.get("/movie/{movie_id}", dependencies=[Depends(require_basic)])
def get_movie_info(movie_id: int):
    """Get information about a specific movie"""
    ensure_model_loaded()  # ← NEW
    
    try:
        movie_info = recommender.get_movie_info(movie_id)
        return {"movie_id": movie_id, **movie_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting movie info: {str(e)}")

@app.get("/popular", dependencies=[Depends(require_basic)])
def get_popular_movies(n_movies: int = 10):
    """Get popular movies"""
    ensure_model_loaded()  # ← NEW
    
    try:
        popular = recommender.get_popular_movies(n_movies)
        return {"popular_movies": popular, "count": len(popular)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular movies: {str(e)}")

@app.post("/train", dependencies=[Depends(require_basic)])
async def trigger_training():  # ← NEW: async
    """
    Trigger model retraining (protected).
    Used by Airflow DAG for automated retraining.
    """
    try:
        import subprocess
        
        print("🚀 Starting training...")
        
        # Run training in background (non-blocking)
        result = await asyncio.to_thread(
            subprocess.run,
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
        
        print(f"Training returncode: {result.returncode}")
        
        if result.returncode == 0:
            # Reload model after successful training
            global recommender
            print("🔄 Reloading model...")
            try:
                recommender = MovieRecommender()
                print("✅ Model reloaded successfully!")
            except Exception as e:
                print(f"⚠️ Model reload failed: {e}")
            
            return {
                "status": "success",
                "message": "Model retrained and reloaded successfully",
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
        else:
            print(f"❌ Training failed with code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {
                "status": "error",
                "message": "Training failed",
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Training timeout after 1 hour")
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/status", dependencies=[Depends(require_basic)])
def get_model_status():
    """Check current model status and version (protected)"""
    try:
        if not recommender:
            return {
                "status": "starting",
                "message": "API is starting up"
            }
        
        models_dir = Path("models")
        if not models_dir.exists():
            return {
                "status": "no_model",
                "message": "No trained model found",
                "models_loaded": False
            }
        
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            return {
                "status": "no_model",
                "message": "No model files found",
                "models_loaded": getattr(recommender, "models_loaded", False)
            }
        
        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
        return {
            "status": "ready",
            "models_loaded": getattr(recommender, "models_loaded", False),
            "last_trained": latest_file.stat().st_mtime,
            "model_files": [f.name for f in model_files],
            "model_version": getattr(recommender, "model_version", None)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run the app
if __name__ == "__main__":
    import uvicorn
    print("Starting Movie Recommender API with Basic Auth...")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)