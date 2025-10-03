from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from models.predict_model import MovieRecommender

# Create FastAPI app - SIMPLE VERSION
app = FastAPI(
    title="Movie Recommender API",
    description="Simple movie recommendation system",
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

# API Endpoints - SIMPLE

@app.get("/")
def home():
    """Welcome message"""
    return {
        "message": "Welcome to Movie Recommender API!",
        "status": "running",
        "endpoints": ["/recommendations", "/predict", "/health"]
    }

@app.get("/health")
def health_check():
    """Check if API is working"""
    return {
        "status": "healthy",
        "models_loaded": recommender.models_loaded,
        "message": "API is running"
    }

@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user - SIMPLE"""
    
    try:
        # Get recommendations from our model
        recommendations = recommender.get_user_recommendations(
            request.user_id, 
            request.n_recommendations
        )
        
        # Check for errors
        if isinstance(recommendations, dict) and "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/predict")
def predict_rating(request: PredictionRequest):
    """Predict rating for user-movie pair - SIMPLE"""
    
    try:
        # Get prediction from our model
        prediction = recommender.predict_rating(request.user_id, request.movie_id)
        
        # Check for errors
        if "error" in prediction:
            raise HTTPException(status_code=404, detail=prediction["error"])
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting rating: {str(e)}")

@app.get("/movie/{movie_id}")
def get_movie_info(movie_id: int):
    """Get information about a specific movie - SIMPLE"""
    
    try:
        # Get movie info
        movie_info = recommender.get_movie_info(movie_id)
        
        return {
            "movie_id": movie_id,
            **movie_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting movie info: {str(e)}")

@app.get("/popular")
def get_popular_movies(n_movies: int = 10):
    """Get popular movies - SIMPLE"""
    
    try:
        popular = recommender.get_popular_movies(n_movies)
        
        return {
            "popular_movies": popular,
            "count": len(popular)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular movies: {str(e)}")

@app.post("/train")
def trigger_training():
    """
    Endpoint to trigger model retraining
    Used by Airflow DAG for automated retraining
    
    Returns:
        dict: Training status and metrics
    """
    try:
        import subprocess
        
        # Run training script
        result = subprocess.run(
            ["python", "src/models/train_model.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Model retrained successfully",
                "output": result.stdout[-500:]  # Last 500 chars
            }
        else:
            return {
                "status": "error",
                "message": "Training failed",
                "error": result.stderr[-500:]
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/status")
def get_model_status():
    """
    Check current model status and version
    Used by Airflow to verify deployment
    """
    try:
        models_dir = Path("models")
        
        if not models_dir.exists():
            return {"status": "no_model", "message": "No trained model found"}
        
        # Check model files
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
    
    print("Starting Movie Recommender API...")
    print("API docs will be available at: http://localhost:8000/docs")
    print("API will be running at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)