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

### Step 1: Setup

```bash
# Install required packages
pip install -r requirements.txt
```

# Install MySQL Server:

```bash
Download MySQL (~400 MB)
Create database named movielens
Default credentials: user=root, password=mysql, host=localhost
```

### Step 2: Create Database and Train Model

```bash
# Download data and create database
python src/data/make_dataset.py
python src/data/create_database.py

# Train the AI model
python src/models/train_model.py
```

### Step 2a: Start the MLflow

```bash
# View experiments and model versions
mlflow ui

# Open browser: http://localhost:5000
```

### Step 3: Start the API

```bash
# Start web server
python api_app.py

# Open browser: http://localhost:8000/docs
```

### Step 4: Test Recommendations

```bash
# Get 5 movie recommendations for user 123
curl -X POST "http://localhost:8000/recommendations" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 123, "n_recommendations": 5}'

# Predict rating for user 123 and movie 456
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"user_id": 123, "movie_id": 456}'

     # Trigger model training (Airflow integration)
curl -X POST "http://localhost:8000/train"

# Check model status
curl -X GET "http://localhost:8000/model/status"
```

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
