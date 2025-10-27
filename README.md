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
├─ .github
│  └─ workflows
│     └─ python-app.yml
├─ .gitignore
├─ airflow
│  ├─ dags
│  │  └─ retrain_on_new_batch.py
│  ├─ Dockerfile
│  └─ requirements.txt
├─ api
│  ├─ Dockerfile
│  ├─ main.py
│  └─ requirements.txt
├─ data
│  ├─ landing
│  ├─ processed
│  │  └─ ratings_batch_synth.csv
│  ├─ raw
│  │  ├─ ml-20m
│  │  │  ├─ genome-scores.csv
│  │  │  ├─ genome-tags.csv
│  │  │  ├─ links.csv
│  │  │  ├─ movies.csv
│  │  │  ├─ ratings.csv
│  │  │  ├─ README.txt
│  │  │  └─ tags.csv
│  │  └─ ml-20m.zip
│  └─ sample
│     ├─ movies_sample.csv
│     └─ ratings_sample.csv
├─ docker-compose.yml
├─ LICENSE
├─ mlops_setup.py
├─ models
│  ├─ baseline_stats.pkl
│  ├─ id_maps.pkl
│  ├─ item_factors.pkl
│  ├─ svd_model.pkl
│  └─ user_factors.pkl
├─ mysql-init
│  └─ 01_init_mlflow.sql
├─ notebooks
│  ├─ .gitkeep
│  └─ data_exploration.ipynb
├─ probe_db.py
├─ README.md
├─ references
│  └─ .gitkeep
├─ reports
│  ├─ .gitkeep
│  └─ figures
│     └─ .gitkeep
├─ requirements.txt
├─ src
│  ├─ __init__.py
│  ├─ config
│  ├─ data
│  │  ├─ __init__.py
│  │  ├─ create_database.py
│  │  ├─ create_database_mysql.py
│  │  └─ make_dataset.py
│  ├─ dataops
│  │  ├─ ingest_to_mysql.py
│  │  └─ split_csv.py
│  ├─ features
│  │  ├─ __init__.py
│  │  └─ build_features.py
│  ├─ models
│  │  ├─ __init__.py
│  │  ├─ predict_model.py
│  │  └─ train_model_mysql.py
│  └─ visualization
│     ├─ __init__.py
│     └─ visualize.py
└─ streamlit
   ├─ app.py
   └─ Dockerfile
```

## How to Use the System

0. Pre-requisites:
   - Docker Desktop installed and running

   - WSL2 backend enabled (Windows → Docker Desktop → Settings → Resources → WSL Integration)

   - Git installed

   - Ports free: 8080 (Airflow), 8000 (API), 5000 (MLflow), 3306 (MySQL)

   - No virtual environment needed: Docker containers already install everything from requirements.txt.
   You only need a .venv if you plan to run notebooks locally.
   - Git (to clone the repo)
   - Create a env. file:
      API_BASIC_USER=admin
      API_BASIC_PASS=secret

      MLFLOW_TRACKING_URI=file:/opt/airflow/mlruns
      MODEL_NAME=movie_recommender_svd
      MODEL_URI=models:/movie_recommender_svd@production
      MLFLOW_DISABLE_ENV_CREATION=true

      DB_HOST=mysql-ml
      DB_USER=app
      DB_PASS=mysql
      DB_NAME=movielens

      DATABASE_HOST=mysql-ml
      DATABASE_USER=app
      DATABASE_PASSWORD=mysql
      DATABASE_NAME=movielens

1. Open terminal at repo root

   VS Code: Terminal → New Terminal (it opens at the workspace folder).

   Or cd into the project folder manually.

2. Build and start all containers:
   - docker compose down -v      # clean start (wipes old volumes)
   - docker compose up -d --build
   - docker compose ps 
   Note: after running this confirm all services show "Up"


3. if fresh repo + Prepare dataset and populate the database:
   - (a) Generate dataset sample
      docker compose exec airflow-webserver bash -lc "cd /opt/airflow/repo && python src/data/make_dataset.py"

   - (b) Verify the generated files by looking in the data/raw/ml-20m. ratings and movies.csv should be there

   - (c) create csv batches of the big dataset, run:
      python .\src\data\split_ratings_into_batches.py 

   - (d) Create schema and load data

      # create database and tables and load movies
      docker compose exec airflow-webserver bash -lc "python /opt/airflow/repo/src/data/create_database_mysql.py --load-movies /opt/airflow/repo/data/sample/movies_sample.csv"

      # load ratings
      docker compose exec airflow-webserver bash -lc "python /opt/airflow/repo/src/data/create_database_mysql.py --batch-csv /opt/airflow/repo/data/sample/ratings_sample.csv"

   - (d) Verify inside MySQL the number of rows, run:
      docker compose exec -T mysql-ml mysql -N -B -uapp -pmysql -hmysql-ml -D movielens -e "SELECT COUNT(*) FROM ratings;"

      Ratings should be 0 at the time

4. Verify UIs are reachable

   - FastAPI: http://localhost:8000/docs
      User: admin
      Password: secret

   - MLflow UI: http://localhost:5001

   - Airflow UI: http://localhost:8080

   NOTE: If the Airflow login doesn’t work, recreate the user:
   docker compose exec airflow-webserver airflow users create --username recommender --password BestTeam --firstname Recommender --lastname Admin --role Admin --email recommender@example.com


5. First training run:
   5.1 - Open Airflow (http://localhost:8080)
   5.2 - Open Airflow → find DAG retrain_on_new_batch.
   5.3 - Toggle it ON and click Trigger DAG.

6. Check the model in MLflow (movie_recommender_svd experiment).

7. Confirm the API sees the promoted model
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

-[x] modify the Airflow DAG to request the training endpoint of the API -[x]instead of running the Python script directly. -[x]implement the Streamlit application as the frontend interface for the prediction pipeline. -[x]ensure the API uses the best model for predictions.

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
