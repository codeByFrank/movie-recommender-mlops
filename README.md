# Movie Recommendation System - Phase 2

**Machine Learning Engineering Project**  
**Team:** Frank LEE, Gustavo SILVA BELLO, Nicole DÖHRING  
**Mentor:** Vincent Lalanne  
**Course:** DataScientest ML Engineering  
**Duration:** September 15, 2025 - October 28, 2025 (5 phases)

## What We Built

A **movie recommendation system** (Netflix-style). Users rate movies (1–5★), our SVD collaborative filtering model learns user & item factors, and the API returns recommendations / predictions.
Cold-start is handled via a popularity fallback.

**The system works like this:**

1. User rates some movies (1-5 stars)
2. Our AI finds similar users who liked the same movies
3. We recommend movies that similar users also liked
4. User gets 5 movie suggestions with predicted ratings

## Current Status: Phase 1 Complete ✅

**Deadline:** September 25th, 2025  
**Status:** All requirements finished

### What We Have Now

**Data Pipeline:**

- Downloaded MovieLens dataset (20 million movie ratings)
- Created SQLite database with clean data
- Sample data for testing (50,000 ratings)

**Machine Learning Model:**

- SVD Collaborative Filtering algorithm
- Handles "cold start" problem (new users get popular movies)
- Realistic rating predictions (0.5 to 5.0 stars)
- Model performance: RMSE 1.09, MAE 0.83

**API System:**

- FastAPI web service
- Two main endpoints: `/recommendations` and `/predict`
- Interactive documentation at http://localhost:8000/docs
- Works with any programming language

**Analysis:**

- Jupyter notebook with data exploration
- Charts and statistics about user behavior
- Cold start problem analysis

## Project Structure

```
SEP25_BMLOPS_INT_MOVIE_RECO_2/
├── data/
│   ├── raw/ml-20m/              # Original MovieLens data
│   ├── sample/                  # Small data for testing
│   └── movielens.db             # SQLite database
├── models/
│   ├── svd_model.pkl           # Trained AI model
│   ├── user_factors.pkl        # User preferences
│   ├── item_factors.pkl        # Movie features
│   ├── user_item_matrix.pkl    # Rating matrix
│   └── baseline_stats.pkl      # Statistics for predictions
├── src/
│   ├── data/
│   │   ├── make_dataset.py     # Download and process data
│   │   └── create_database.py  # Setup database
│   └── models/
│       ├── train_model.py      # Train the AI model
│       └── predict_model.py    # Make recommendations
├── notebooks/
│   └── data_exploration.ipynb  # Data analysis
├── mlruns/                     # MLflow experiment tracking data
├── api_app.py                  # Web API server
├── requirements.txt            # Python packages needed
└── README.md                   # This file
```

## How to Use the System

0) Pre-requisites:
  -Docker Desktop (includes Docker Compose)

    Windows: enable WSL2 backend during install

  -Git (to clone the repo)

if fresh repo + empty DB:
# run once, inside the api container (idempotent)
docker compose exec api python -m src.models.create_database_mysql   # or your actual module path


1) Clone the repo

2) Create a .env with this code:
    # API basic auth
    API_BASIC_USER=admin
    API_BASIC_PASS=secret

    # MLflow (experiment tracking & model registry)
    MLFLOW_TRACKING_URI=file:/opt/airflow/mlruns
    MODEL_URI=models:/movie_recommender_svd@production

    # MySQL (database)
    DB_HOST=mysql-ml
    DB_USER=app
    DB_PASS=mysql
    DB_NAME=movielens

3) Start the stack - in terminal:
docker compose up -d
docker compose ps

4) Verify UIs are reachable

  API docs (FastAPI): http://localhost:8000/docs

  MLflow UI: http://localhost:5000

  Airflow UI: http://localhost:8080

5) First training run (creates/updates the model in the registry), run:

  docker compose exec api python -m src.models.train_model_mysql

6) Confirm the API sees the promoted model
  curl http://localhost:8000/health



## Technical Details

### Algorithm: SVD Collaborative Filtering

- **Input:** User-movie rating matrix (very sparse)
- **Method:** Matrix factorization to find hidden patterns
- **Output:** Predicted ratings for unwatched movies
- **Formula:** `Prediction = Global Mean + User Bias + Movie Bias + SVD Component`

### Cold Start Solution

- **Problem:** New users have no rating history
- **Solution:** Recommend most popular movies with high ratings
- **Threshold:** Movies with at least 20 ratings

### Data Processing

- **Original data:** 20 million ratings, 138k users, 27k movies
- **Sample data:** 50k ratings for faster development
- **Matrix sparsity:** 99.98% empty (typical for recommendation systems)

### Model Performance

- **RMSE:** 1.09 (lower is better)
- **MAE:** 0.83 (lower is better)
- **Coverage:** Can predict ratings for 1000+ test cases
- **Prediction range:** 0.5 to 5.0 stars (realistic)

## Phase 1 Requirements ✅

- [x] **Data Pipeline:** MovieLens 20M dataset downloaded and processed
- [x] **Database:** SQLite with movies and ratings tables
- [x] **ML Model:** SVD collaborative filtering trained and tested
- [x] **Training Script:** `src/models/train_model.py` creates and saves model
- [x] **Prediction Script:** `src/models/predict_model.py` loads model and makes recommendations
- [x] **API Endpoints:**
  - [x] `/recommendations` - get movie suggestions for user
  - [x] `/predict` - predict rating for user-movie pair
- [x] **Data Analysis:** Jupyter notebook with visualizations and insights
- [x] **Cold Start Handling:** Popular movies fallback for new users

## Phase 2 Requirements ✅

- [x] **MLflow Setup:** Experiment tracking integrated in training script
- [x] **Model Versioning:** MLflow Registry for model version management
- [x] **Performance Comparison:** Automatic comparison between model versions
- [x] **Best Model Selection:** Flag best performing models in MLflow
- [x] **MySQL Migration:** Database migrated from SQLite to MySQL
- [x] **Docker Preparation:** MySQL configured for containerization
- [x] **Airflow Integration Endpoints:**
  - [x] **/train** - Trigger model training
  - [x] **/model/status** - Get current model information

## Phase 3

- [x] implement MLflow locally for monitoring training experiments and model registry.
- [x] integrate MLflow to track metrics and logs in the training script.
- [x] prepare for future transition from SQLite to a proper SQL database.
- [x] implement automation using Airflow or another tool to collect new data and trigger model training automatically.
- simulate new data arrival by either splitting the CSV file into multiple parts or applying random sampling on the training set.

-[x] implement dockerization of the application components.
    [x] create a Docker Compose file to orchestrate the Docker images.
    [x] create a custom Docker file for the API component.

## Phase 4
-[x] modify the Airflow DAG to request the training endpoint of the API 
-[x]instead of running the Python script directly.
-[x]implement the Streamlit application as the frontend interface for the prediction pipeline.
-[x]ensure the API uses the best model for predictions.


## Dependencies

**Install command:**

```bash
pip install pandas numpy scikit-learn fastapi uvicorn matplotlib seaborn jupyter requests
```

**Phase 2 additions:**

```bash
pip install mlflow mysql-connector-python
```

**External requirements**
MySQL Server (~400 MB)

## Team Information

**Course:** Machine Learning Engineering at DataScientest  
**Project duration:** 5 phases over 2 months  
**Current phase:** 1/5 completed  
**Team size:** 3 people  
**Focus:** Building production-ready ML systems, not just algorithms
