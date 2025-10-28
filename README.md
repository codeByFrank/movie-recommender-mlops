# Movie Recommendation System - Phase 2

**Machine Learning Engineering Project**  
**Team:** Frank LEE, Gustavo SILVA BELLO, Nicole DÖHRING  
**Mentor:** Nicolas FRADIN
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

**Data Pipeline:**

- Downloaded MovieLens **20M** ratings dataset
- Created **MySQL** database with clean `ratings` and `movies` tables (not SQLite)
- Batch retraining via **Airflow** with folder flow: `data/landing → incoming → processed/failed`
- Sample mode for fast demos (e.g., `TRAIN_USER_MOD_BASE`, `EVAL_CAP`)
- Docker Compose brings up DB, API, Airflow, MLflow, Streamlit

**Machine Learning Model:**

- **SVD** collaborative filtering (Truncated SVD on user–item matrix)
- Handles **cold start** (new/unknown users fall back to popularity/global baseline)
- Predictions clipped to **[0.5, 5.0]** stars
- Model performance (sample run): **RMSE ≈ 0.90, MAE ≈ 0.68**

## Model promotion & rollback (MLflow)

- Training logs metrics and artifacts to MLflow (`movie_recommender_svd`).
- The retrain DAG compares candidate RMSE vs current Champion.
- If strictly better, it transitions the candidate to **Staging**/**Production** (depending on your setting).
- **Rollback:** In MLflow UI, switch the previous version back to Production (Stage change). No code change needed.

**API System:**

- **FastAPI** web service
- Main endpoint: **`/recommend`** (e.g., `?user_id=123&k=10`)
- Interactive docs at `http://localhost:8000/docs`
- Works with any programming language (HTTP + JSON)

**Apps & Dashboards:**

- **Streamlit** UI to trigger recommendations and show sample stats
- **Airflow UI** for DAG runs (retraining pipeline)
- **MLflow UI** for experiments, models, and stage transitions

**Screenshots (Streamlit UI):**

A selection of Streamlit UI screenshots lives in **`images/`**  
(landing page, recommendations, rating prediction, popular movies, system status, outlook and more).

**Analysis:**

- Jupyter notebooks for data exploration
- Charts and statistics about user behavior
- Cold-start behavior confirmed (fallback to popularity for users with no ratings)

## Project Structure

```
sep25_bmlops_int_movie_reco_2/
+-- .github
|  +-- workflows
|     +-- python-app.yml
+-- airflow
|  +-- dags
|  |  +-- retrain_on_new_batch.py
|  +-- Dockerfile
|  +-- requirements.txt
+-- api
|  +-- Dockerfile
|  +-- main.py
|  +-- requirements.txt
+-- data
|  +-- failed
|  +-- incoming
|  |  +-- ... (generated batching file)
|  +-- landing
|  +-- processed
|  |  +-- ... (processed batching files)
|  +-- raw
|  |  +-- ml-20m
|  |     +-- genome-scores.csv
|  |     +-- genome-tags.csv
|  |     +-- links.csv
|  |     +-- movies.csv
|  |     +-- ratings.csv
|  |     +-- tags.csv
|  +-- sample
|     +-- movies_sample.csv
|     +-- ratings_sample.csv
+-- models
+-- mysql-init
|  +-- 01_init_mlflow.sql
+-- notebooks
|  +-- .gitkeep
|  +-- data_exploration.ipynb
+-- references
|  +-- .gitkeep
+-- reports
|  +-- figures
|  |  +-- .gitkeep
|  +-- .gitkeep
+-- src
|  +-- data
|  |  +-- __init__.py
|  |  +-- create_database.py
|  |  +-- create_database_mysql.py
|  |  +-- make_dataset.py
|  |  +-- split_ratings_into_batches.py
|  +-- dataops
|  |  +-- ingest_to_mysql.py
|  |  +-- split_csv.py
|  +-- features
|  |  +-- __init__.py
|  |  +-- build_features.py
|  +-- models
|  |  +-- __init__.py
|  |  +-- predict_model.py
|  |  +-- train_model_mysql.py
|  +-- visualization
|  |  +-- __init__.py
|  |  +-- visualize.py
|  +-- __init__.py
+-- streamlit
|  +-- images
|  +-- app.py
|  +-- Dockerfile
+-- tests
|  +-- test_dummy.py
+-- .gitignore
+-- docker-compose.yml
+-- LICENSE
+-- mlops_setup.py
+-- probe_db.py
+-- README.md
+-- repo_tree.py
+-- requirements.txt
```

## Environment variables

| Area            | Variable              | Default / Example                  | Used by                             |
| --------------- | --------------------- | ---------------------------------- | ----------------------------------- |
| MySQL           | `DATABASE_HOST`       | `mysql-ml`                         | training (`train_model_mysql.py`)   |
|                 | `DATABASE_PORT`       | `3306`                             | training                            |
|                 | `DATABASE_USER`       | `app`                              | training                            |
|                 | `DATABASE_PASSWORD`   | `mysql`                            | training                            |
|                 | `DATABASE_NAME`       | `movielens`                        | training                            |
| MySQL (predict) | `DB_HOST`             | `mysql-ml`                         | prediction/API (`predict_model.py`) |
|                 | `DB_PORT`             | `3306`                             | prediction/API                      |
|                 | `DB_USER`             | `app`                              | prediction/API                      |
|                 | `DB_PASS`             | `mysql`                            | prediction/API                      |
|                 | `DB_NAME`             | `movielens`                        | prediction/API                      |
| MLflow          | `MLFLOW_TRACKING_URI` | `http://mlflow-ui:5000`            | train/predict/DAG                   |
|                 | Model name            | `movie_recommender_svd`            | (constant in code)                  |
| Airflow DAG IO  | `LANDING_DIR`         | `/opt/airflow/repo/data/landing`   | retrain DAG                         |
|                 | `INCOMING_DIR`        | `/opt/airflow/repo/data/incoming`  | retrain DAG                         |
|                 | `PROCESSED_DIR`       | `/opt/airflow/repo/data/processed` | retrain DAG                         |
|                 | `FAILED_DIR`          | `/opt/airflow/repo/data/failed`    | retrain DAG                         |
| Streamlit       | `API_URL`             | `http://api:8000`                  | Streamlit                           |
|                 | `API_BASIC_USER`      | `admin`                            | Streamlit/API                       |
|                 | `API_BASIC_PASS`      | `secret`                           | Streamlit/API                       |

### Training knobs (documented for reproducibility)

- `TRAIN_USER_MOD_BASE` (int): user-sampling modulus used during dev to speed up training (e.g., `6` means keep ~1/6 users).
- `EVAL_CAP` (int): optional cap on the number of interactions used for evaluation to keep runs fast in demos.

## How to Use the System

0. Pre-requisites:

   - Docker Desktop installed and running

   - WSL2 backend enabled (Windows → Docker Desktop → Settings → Resources → WSL Integration)

   - Git installed

   - Ports free: 8080 (Airflow), 8000 (API), 5001 (MLflow), 3306 (MySQL), 8501(Streamlit)

   - No virtual environment needed: Docker containers already install everything from requirements.txt.
     You only need a .venv if you plan to run notebooks locally.
   - Git (to clone the repo)
   - Create a env. file like the next one:
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

   - docker compose down -v # clean start (wipes old volumes)
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

     - create database and tables and load movies
       docker compose exec airflow-webserver bash -lc "python /opt/airflow/repo/src/data/create_database_mysql.py --load-movies /opt/airflow/repo/data/sample/movies_sample.csv"

     - load ratings
       docker compose exec airflow-webserver bash -lc "python /opt/airflow/repo/src/data/create_database_mysql.py --batch-csv /opt/airflow/repo/data/sample/ratings_sample.csv"

   - (e) Verify inside MySQL the number of rows, run:
     docker compose exec -T mysql-ml mysql -N -B -uapp -pmysql -hmysql-ml -D movielens -e "SELECT COUNT(\*) FROM ratings;"

     Ratings should be 0 at the time

4. Verify UIs are reachable

   - FastAPI: http://localhost:8000/docs
     User: admin
     Password: secret

   - MLflow UI: http://localhost:5001

   - Airflow UI: http://localhost:8080

   NOTE: If the Airflow login doesn’t work, recreate the user:
   docker compose exec airflow-webserver airflow users create --username recommender --password BestTeam --firstname Recommender --lastname Admin --role Admin --email recommender@example.com

   - Streamlit: http://localhost:8501/

5. First training run:
   5.1 - Open Airflow (http://localhost:8080)
   5.2 - Open Airflow → find DAG retrain_on_new_batch.
   5.3 - Toggle it ON and click Trigger DAG.

6. Check the model in MLflow (movie_recommender_svd experiment).

7. Confirm the API sees the promoted model
   curl http://localhost:8000/health

## Quickstart

1. Copy `.env.example` → `.env` and adjust env vars (see table above).
2. Start the stack:
   ```bash
   docker compose up -d --build
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

## Phase 3

- [x] implement MLflow locally for monitoring training experiments and model registry.
- [x] integrate MLflow to track metrics and logs in the training script.
- [x] prepare for future transition from SQLite to a proper SQL database.
- [x] implement automation using Airflow or another tool to collect new data and trigger model training automatically.
- simulate new data arrival by either splitting the CSV file into multiple parts or applying random sampling on the training set.

-[x] implement dockerization of the application components. -[x] create a Docker Compose file to orchestrate the Docker images. -[x] create a custom Docker file for the API component.

## Phase 4

-[x] modify the Airflow DAG to request the training endpoint of the API -[x]instead of running the Python script directly. -[x]implement the Streamlit application as the frontend interface for the prediction pipeline. -[x]ensure the API uses the best model for predictions.

## Team Information

**Course:** Machine Learning Engineering at DataScientest  
**Project duration:** 5 phases over 2 months  
**Current phase:** 1/5 completed  
**Team size:** 3 people  
**Focus:** Building production-ready ML systems, not just algorithms
