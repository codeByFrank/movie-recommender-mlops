import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import os
from datetime import datetime
import base64

# ==================== CONFIGURATION ====================
API_URL = os.getenv("API_URL", "http://api:8000")
API_USER = os.getenv("API_BASIC_USER", "admin")
API_PASS = os.getenv("API_BASIC_PASS", "secret")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow-ui:5000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")

# Page config
st.set_page_config(
    page_title="Movie Recommender - MLOps Project",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;  /* Space for fixed buttons */
    }
    
    .big-title {
        font-size: 3.5rem !important;
        font-weight: bold;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.8rem;
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .team-member {
        font-size: 1.3rem;
        text-align: center;
        margin: 0.3rem;
    }
    .slide-title {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        margin-top: 0rem;
    }
    .tech-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .slide-content {
       # padding: 1rem;
       # min-height: 400px;
    }
    .project-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    /* Fixed navigation bar at bottom */
    .fixed-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
        border-top: 2px solid #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
def load_image(image_path):
    """Load image from file"""
    try:
        from PIL import Image
        return Image.open(image_path)
    except:
        return None

def call_api(endpoint, method="GET", json_data=None):
    """Call FastAPI endpoint"""
    try:
        auth = HTTPBasicAuth(API_USER, API_PASS)
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, auth=auth, timeout=10)
        else:
            response = requests.post(url, auth=auth, json=json_data, timeout=30)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ==================== SLIDES ====================
def slide_1_title():
    """Title Slide - FRANK"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="big-title">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">MLOps End-to-End ML Pipeline</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p class="team-member">👨‍💻 Frank Lee</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="team-member">👩‍💻 Nicole Doehring</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="team-member">👨‍💻 Gustavo Silva</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project info
    st.markdown("""
    <div class="project-info">
        <strong>🎓 DataScientest MLOps Bootcamp</strong><br>
        📅 <strong>Project Duration:</strong> September 4 - October 22, 2025<br>
        🛠️ <strong>Tools:</strong> GitHub | Visual Studio Code | PyCharm | Jupyter Notebooks<br>
        🎯 <strong>Objective:</strong> Production-ready ML system with automated training & deployment
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_2_problem():
    """Problem & Solution - FRANK"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">🎯 Problem & Solution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 The Challenge")
        st.markdown("""
        **Business Context:**
        - **Information Overload**: Millions of movies available online
        - **User Experience**: Hard to discover relevant content
        - **Business Impact**: Lost engagement & revenue
        - **Technical Challenge**: Scale, automate, maintain
        
        **ML Problems to Solve:**
        - ❄️ **Cold Start**: New users/movies without ratings
        - 📊 **Data Drift**: User preferences change over time
        - 🎯 **Model Quality**: How to monitor good vs bad recommendations?
        """)
    
    with col2:
        st.markdown("### ✅ Our Solution")
        st.markdown("""
        **Technical Approach:**
        - **Collaborative Filtering**: SVD matrix factorization
        - **Content-Based Fallback**: For cold start scenarios
        - **Popularity Baseline**: For completely new users
        
        **MLOps Implementation:**
        - 🔄 **Automated Retraining**: Daily Airflow pipeline
        - 📈 **Experiment Tracking**: MLflow for metrics & models
        - 🚀 **Continuous Deployment**: Best model auto-promoted
        - 🐳 **Containerization**: Docker for reproducibility
        """)
    
    st.markdown("---")
    st.success("💡 **Goal**: Netflix-like recommendation system with production-grade MLOps practices")
    st.markdown('</div>', unsafe_allow_html=True)

def slide_3_architecture_training():
    """Architecture - Training Pipeline - FRANK"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">🏗️ Architecture: Training Pipeline</p>', unsafe_allow_html=True)
    
    # Show architecture diagram
    img = load_image("images/architecture_training.png")
    if img:
        st.image(img, caption="Training Pipeline Architecture", use_column_width=True)
    else:
        st.warning("📊 Architecture diagram not found - showing text description")
    
    st.info("📌 **Key Innovation**: Airflow calls FastAPI for training (not direct execution)")
    
    st.markdown("### 🔄 Automated Training Flow")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1️⃣ Data Ingestion**")
        st.write("- CSV files arrive")
        st.write("- Airflow ETL pipeline")
        st.write("- Store in MySQL")
    
    with col2:
        st.markdown("**2️⃣ Training Trigger**")
        st.write("- Airflow → FastAPI")
        st.write("- POST /train endpoint")
        st.write("- SVD training starts")
    
    with col3:
        st.markdown("**3️⃣ Evaluation**")
        st.write("- Calculate RMSE, MAE")
        st.write("- Compare with production")
        st.write("- Log to MLflow")
    
    with col4:
        st.markdown("**4️⃣ Deployment**")
        st.write("- Best model selected")
        st.write("- Tagged '@production'")
        st.write("- API auto-loads new model")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔑 Key Components")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌊 Airflow**")
        st.write("- DAG: retrain_on_new_batch")
        st.write("- Schedule: @daily")
        st.write("- 6 tasks: ETL → Train → Compare → Promote")
    with col2:
        st.markdown("**📈 MLflow**")
        st.write("- Experiment tracking")
        st.write("- Model registry")
        st.write("- Version management")
    with col3:
        st.markdown("**🗄️ MySQL**")
        st.write("- 20M ratings")
        st.write("- 6,730 movies")
        st.write("- Batch ingestion")
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_4_architecture_inference():
    """Architecture - Inference Pipeline - NICOLE"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">🚀 Architecture: Inference Pipeline</p>', unsafe_allow_html=True)
    
    # Show architecture diagram
    img = load_image("images/architecture_inference.png")
    if img:
        st.image(img, caption="Inference Pipeline Architecture", use_column_width=True)
    else:
        st.warning("📊 Architecture diagram not found - showing text description")
    
    st.markdown("### 🎯 Real-time Prediction Flow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ User Request**")
        st.write("- Streamlit UI")
        st.write("- User ID + preferences")
        st.write("- Click 'Get Recommendations'")
    
    with col2:
        st.markdown("**2️⃣ API Processing**")
        st.write("- FastAPI receives request")
        st.write("- Loads @production model")
        st.write("- Matrix multiplication")
        st.write("- Generates predictions")
    
    with col3:
        st.markdown("**3️⃣ Response**")
        st.write("- Top-N recommendations")
        st.write("- Predicted ratings (0.5-5.0)")
        st.write("- Movie metadata")
        st.write("- Real-time (<1s)")
    
    st.markdown("---")  # ← ENTFERNE das </div> hier!
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔥 Performance")
        st.metric("Response Time", "< 500ms", "Fast")
        st.metric("Model Load", "On startup", "Cached")
        st.metric("Concurrent Users", "Scalable", "Stateless")
    
    with col2:
        st.markdown("### 🛡️ Production Features")
        st.write("✅ HTTP Basic Auth")
        st.write("✅ Error handling & fallbacks")
        st.write("✅ Cold start handling")
        st.write("✅ Health checks")
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_5_tech_stack():
    """Technology Stack - NICOLE"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">🛠️ Technology Stack</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 **ML & Data**")
        st.markdown('<span class="tech-badge">🐍 Python 3.12</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">📊 Pandas & NumPy</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">🤖 Scikit-learn (SVD)</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">🗄️ MySQL 8.0</span>', unsafe_allow_html=True)
        
        st.markdown("### 🔄 **MLOps Tools**")
        st.markdown('<span class="tech-badge">📈 MLflow 2.x</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">🌊 Apache Airflow</span>', unsafe_allow_html=True)
        
    with col2:
        st.markdown("### 🌐 **API & Frontend**")
        st.markdown('<span class="tech-badge">⚡ FastAPI</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">🎨 Streamlit</span>', unsafe_allow_html=True)
        
        st.markdown("### 🐳 **Infrastructure**")
        st.markdown('<span class="tech-badge">🐳 Docker</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">📦 Docker Compose</span>', unsafe_allow_html=True)
        
        st.markdown("### 🧰 **Development**")
        st.markdown('<span class="tech-badge">💻 VS Code</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">📓 Jupyter</span>', unsafe_allow_html=True)
        st.markdown('<span class="tech-badge">🐙 GitHub</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.success("✨ **All services containerized and orchestrated with Docker Compose**")
    st.markdown('</div>', unsafe_allow_html=True)

def slide_6_data_coldstart():
    """Data Pipeline & Cold Start - NICOLE"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">📊 Data Pipeline & Cold Start Problem</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🎬 MovieLens 20M Dataset")
        st.markdown("""
        - **20 million ratings** from 138,493 users
        - **27,278 movies** with metadata
        - **Rating scale**: 0.5 to 5.0 stars
        - **Sample**: 50,000 ratings for development
        """)
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Stats")
        st.metric("Total Ratings", "50K")
        st.metric("Users", "33.2K")
        st.metric("Movies", "6.7K")
        st.metric("Sparsity", "99.98%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Show rating distribution plot
        img = load_image("images/rating_distribution.png")
        if img:
            st.image(img, caption="Rating Distribution Analysis", use_column_width=True)
    
    st.markdown("---")
    
    st.markdown("### ❄️ Cold Start Problem Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Show cold start analysis plot
        img = load_image("images/cold_start_analysis.png")
        if img:
            st.image(img, caption="Cold Start Analysis", use_column_width=True)
    
    with col2:
        st.warning("""
        **Problem Identified:**
        - 32,423 users with <5 ratings (97.6%) 🥶
        - 4,475 movies with <5 ratings (66.5%) 🧊
        - Very sparse user-item matrix
        """)
        
        st.success("""
        **Solution Implemented:**
        - ✅ Popularity-based fallback for new users
        - ✅ Content-based filtering for new movies
        - ✅ Hybrid approach: CF + Popularity
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_7_model():
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">🤖 Model & Performance</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧮 Algorithm: SVD")
        st.markdown("""
        **Singular Value Decomposition (Truncated)**
        - Matrix factorization for collaborative filtering
        ...
        """)
        
        st.code("""
from sklearn.decomposition import TruncatedSVD

# Train model
model = TruncatedSVD(
    n_components=50,
    random_state=42
)
model.fit(user_item_matrix)
        """, language="python")

    # Predict
    st.code(
        """import numpy as np

    score = np.dot(user_factors, item_factors)
    rating = np.clip(global_mean + biases + score, 0.5, 5.0)
    """,
        language="python",
    )
    
    with col2:
        st.markdown("### 📊 Performance Metrics")
        
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("RMSE (Test)", "0.80", "-0.05 vs baseline")
        st.metric("MAE (Test)", "0.62", "-0.03 vs baseline")
        st.metric("Training Time", "~20s", "On 50K ratings")
        st.metric("Predictions/sec", "~2000", "Batch inference")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📈 Evaluation Strategy")
        st.markdown("""
        - **Train/Test Split**: 80/20
        - **Cross-validation**: 5-fold CV
        - **Baseline**: Global mean + biases
        - **Metric**: RMSE (lower is better)
        - **Threshold**: Must beat baseline by >5%
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_8_mlops():
    """MLOps Implementation - TAVO"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="slide-title">⚙️ MLOps: Automation & Monitoring</p>', unsafe_allow_html=True)
    
    st.markdown("### 🔄 Automated Retraining Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📅 Scheduled**")
        st.write("✅ Daily cron job (Airflow)")
        st.write("✅ Checks for new data")
        st.write("✅ Automatic execution")
        
    with col2:
        st.markdown("**🎯 Event-Driven**")
        st.write("✅ New batch arrives")
        st.write("✅ Performance degradation")
        st.write("✅ Manual trigger available")
    
    with col3:
        st.markdown("**🔍 Monitored**")
        st.write("✅ MLflow experiment tracking")
        st.write("✅ Metric comparison")
        st.write("✅ Model versioning")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Model Deployment Strategy")
    
    st.code("""
# Airflow DAG: retrain_on_new_batch

1. maybe_generate_batch_if_empty  # Create synthetic batch if needed
2. pick_batch                      # Select CSV from landing/
3. ingest_mysql                    # Load into database
4. train_candidate                 # Train new model via FastAPI
5. compare_and_promote             # Compare metrics, promote if better
6. archive_batch                   # Move CSV to processed/
    """, language="python")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Implemented")
        st.write("✅ Automated training pipeline")
        st.write("✅ Model versioning (MLflow)")
        st.write("✅ A/B testing ready (production tag)")
        st.write("✅ Rollback capability")
        st.write("✅ Experiment tracking")
    
    with col2:
        st.markdown("### ⏳ Future Improvements")
        st.write("🔄 Data drift detection (Evidently)")
        st.write("📊 Grafana dashboards")
        st.write("🔔 Alert system (Prometheus)")
        st.write("🧪 A/B testing framework")
        st.write("🎯 Real-time monitoring")
    
    st.markdown('</div>', unsafe_allow_html=True)

def slide_9_transition():
    """Transition to Demo - TAVO"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="big-title">🎬 Live Demo</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Let\'s see the system in action!</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Recommendations")
        st.markdown("**Get Personalized Movies**")
        st.write("Enter a User ID and get top-N movie recommendations")
    
    with col2:
        st.markdown("### ⭐ Rating Prediction")
        st.markdown("**Predict User-Movie Rating**")
        st.write("Predict how a user would rate a specific movie")
    
    with col3:
        st.markdown("### 🔥 Popular Movies")
        st.markdown("**Trending & Top Rated**")
        st.write("View most popular and highest-rated movies")
    
    st.markdown("---")
    st.info("👉 **Switch to 'Demo' tab in the sidebar to interact with the system!**")
    st.markdown('</div>', unsafe_allow_html=True)

def slide_10_conclusion():
    """Conclusion & Next Steps"""
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    st.markdown('<p class="big-title">✅ Summary & Achievements</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What We Built")
        st.markdown("""
        ✅ **End-to-End ML Pipeline**
        - Automated data ingestion
        - SVD-based recommendation model
        - Production-ready FastAPI service
        - Interactive Streamlit dashboard
        
        ✅ **MLOps Best Practices**
        - Experiment tracking (MLflow)
        - Workflow orchestration (Airflow)
        - Model versioning & registry
        - Automated deployment
        
        ✅ **Production Features**
        - Cold start handling
        - Real-time predictions
        - Authentication & security
        - Containerized deployment
        """)
    
    with col2:
        st.markdown("### 🚀 Key Achievements")
        st.markdown("""
        **Technical:**
        - RMSE: 0.80 (beat baseline by 5%)
        - Response time: <500ms
        - Daily automated retraining
        - Zero-downtime deployments
        
        **Learning Outcomes:**
        - MLOps principles in practice
        - Airflow pipeline design
        - Model serving at scale
        - Docker orchestration
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔮 Future Work")
        st.write("- Data drift detection")
        st.write("- A/B testing framework")
        st.write("- Real-time monitoring")
        st.write("- Feature store integration")
    
    with col2:
        st.markdown("### 🛠️ Tech Stack")
        st.write("- Python | Docker | MySQL")
        st.write("- Airflow | MLflow")
        st.write("- FastAPI | Streamlit")
        st.write("- Scikit-learn | NumPy")
    
    with col3:
        st.markdown("### 👥 Team")
        st.write("**Frank Lee**")
        st.write("**Nicole Doehring**")
        st.write("**Gustavo Silva**")
        st.write("")
        st.write("📅 Sep - Oct 2025")
    
    st.markdown("---")
    
    st.success("🎓 **DataScientest MLOps Bootcamp** | Thank you! Questions?")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== DEMO PAGES ====================
def demo_recommendations():
    """Demo: Movie Recommendations"""
    st.markdown('<p class="slide-title">🎯 Movie Recommendations</p>', unsafe_allow_html=True)
    
    st.info("💡 Try different User IDs: 15, 57, 100, 1000, 10000")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input("Enter User ID", min_value=1, max_value=138000, value=15, step=1)
    with col2:
        n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
    
    if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching recommendations..."):
            data, error = call_api(
                "/recommendations",
                method="POST",
                json_data={"user_id": user_id, "n_recommendations": n_recommendations}
            )
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Found {data['count']} recommendations for User {user_id}")
            
            for idx, movie in enumerate(data['recommendations'], 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 2])
                    with col1:
                        st.markdown(f"### #{idx}")
                    with col2:
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"Movie ID: {movie.get('movie_id', movie.get('movieId', 'N/A'))} | Genres: {movie.get('genres', 'N/A')}")
                    with col3:
                        st.metric("Rating", f"{movie['predicted_rating']:.2f} ⭐")
                st.divider()

def demo_predictions():
    """Demo: Rating Predictions"""
    st.markdown('<p class="slide-title">⭐ Rating Predictions</p>', unsafe_allow_html=True)
    
    st.info("💡 Try: User 57 + Movie 1 (Toy Story), or User 14 + Movie 50 (Usual Suspects)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=138000, value=57, step=1, key="pred_user")
    with col2:
        movie_id = st.number_input("Movie ID", min_value=1, max_value=27000, value=1, step=1, key="pred_movie")
    
    if st.button("🔮 Predict Rating", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            data, error = call_api(
                "/predict",
                method="POST",
                json_data={"user_id": user_id, "movie_id": movie_id}
            )
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Prediction complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Rating", f"{data['predicted_rating']:.2f} ⭐")
            with col2:
                st.markdown(f"**Movie:** {data.get('title', 'Unknown')}")
                st.markdown(f"**Genres:** {data.get('genres', 'N/A')}")

def demo_movie_info():
    """Demo: Movie Information"""
    st.markdown('<p class="slide-title">🎬 Movie Information</p>', unsafe_allow_html=True)
    
    st.info("💡 Try Movie IDs: 1 (Toy Story), 2019 (Seven Samurai), 50 (Usual Suspects)")
    
    movie_id = st.number_input("Enter Movie ID", min_value=1, max_value=27000, value=1, step=1, key="info_movie")
    
    if st.button("📋 Get Movie Info", type="primary", use_container_width=True):
        with st.spinner("Fetching movie info..."):
            data, error = call_api(f"/movie/{movie_id}")
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Movie found!")
            
            st.markdown(f"### {data.get('title', 'Unknown Movie')}")
            st.markdown(f"**Movie ID:** {movie_id}")
            st.markdown(f"**Genres:** {data.get('genres', 'Unknown')}")

def demo_popular():
    """Demo: Popular Movies"""
    st.markdown('<p class="slide-title">🔥 Popular Movies</p>', unsafe_allow_html=True)
    
    n_movies = st.slider("Number of movies to show", min_value=5, max_value=20, value=10)
    
    if st.button("🌟 Get Popular Movies", type="primary", use_container_width=True):
        with st.spinner("Fetching popular movies..."):
            data, error = call_api(f"/popular?n_movies={n_movies}")
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Found {data['count']} popular movies")
            
            # Display in table format
            for idx, movie in enumerate(data['popular_movies'], 1):
                col1, col2, col3, col4 = st.columns([1, 5, 2, 2])
                with col1:
                    st.markdown(f"**#{idx}**")
                with col2:
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"{movie.get('genres', 'N/A')}")
                with col3:
                    st.metric("Avg Rating", f"{movie.get('avg_rating', movie.get('predicted_rating', 0)):.2f} ⭐")
                with col4:
                    st.metric("# Ratings", f"{movie.get('num_ratings', 'N/A')}")
                st.divider()

def demo_system_status():
    """Demo: System Status"""
    st.markdown('<p class="slide-title">📊 System Status</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 MLflow Tracking")
        st.info(f"🔗 MLflow UI: http://localhost:5001")
        st.write("- **Experiment**: movie_reco_svd")
        st.write("- **Model Registry**: movie_recommender_svd")
        st.write("- **Metrics**: RMSE, MAE tracked")
        st.write("- **Artifacts**: Model files, parameters")
    
    with col2:
        st.markdown("### 🌊 Airflow Orchestration")
        st.info(f"🔗 Airflow UI: http://localhost:8080")
        st.write("- **DAG**: retrain_on_new_batch")
        st.write("- **Schedule**: @daily")
        st.write("- **Tasks**: 6 (ETL → Train → Promote)")
        st.write("- **Status**: Check UI for latest runs")
    
    st.markdown("---")
    
    # API Health Check
    st.markdown("### ⚡ API Health Check")
    if st.button("🔍 Check API Status", use_container_width=True):
        with st.spinner("Checking..."):
            health, error = call_api("/health")
        
        if health:
            st.success("✅ API is healthy and running!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "Healthy ✅")
            with col2:
                st.metric("Models Loaded", "Yes" if health.get('models_loaded') else "No")
            with col3:
                st.metric("Model Version", health.get('model_version', 'N/A'))
            
            with st.expander("📋 Full Health Response"):
                st.json(health)
        else:
            st.error(f"❌ API health check failed: {error}")

# ==================== MAIN APP ====================
def main():
    # Initialize session state
    if 'slide_number' not in st.session_state:
        st.session_state.slide_number = 0
    if 'mode' not in st.session_state:
        st.session_state.mode = 'presentation'
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🎬 Navigation")
        
        mode = st.radio(
            "Select Mode:",
            ["📊 Presentation", "🎮 Demo"],
            index=0 if st.session_state.mode == 'presentation' else 1
        )
        
        if "Presentation" in mode:
            st.session_state.mode = 'presentation'
            
            st.markdown("---")
            st.markdown("### 📑 Slides")
            
            slides = [
                "1️⃣ Title",
                "2️⃣ Problem & Solution",
                "3️⃣ Architecture: Training",
                "4️⃣ Architecture: Inference",
                "5️⃣ Tech Stack",
                "6️⃣ Data & Cold Start",
                "7️⃣ Model & Metrics",
                "8️⃣ MLOps Implementation",
                "9️⃣ Transition to Demo",
                "🎯 Conclusion"
            ]
            
            selected_slide = st.radio("Jump to:", slides, index=st.session_state.slide_number)
            st.session_state.slide_number = slides.index(selected_slide)
            
            st.markdown("---")
            st.info("💡 Click buttons below to navigate")
            
        else:
            st.session_state.mode = 'demo'
            
            st.markdown("---")
            st.markdown("### 🎮 Demo Features")
            demo_page = st.radio(
                "Select Feature:",
                ["🎯 Recommendations", "⭐ Predictions", "🎬 Movie Info", "🔥 Popular", "📊 System Status"]
            )
    
    # Main Content
    if st.session_state.mode == 'presentation':
        # Show current slide
        slides_functions = [
            slide_1_title,
            slide_2_problem,
            slide_3_architecture_training,
            slide_4_architecture_inference,
            slide_5_tech_stack,
            slide_6_data_coldstart,
            slide_7_model,
            slide_8_mlops,
            slide_9_transition,
            slide_10_conclusion
        ]
        
        slides_functions[st.session_state.slide_number]()
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(st.session_state.slide_number == 0), use_container_width=True):
                st.session_state.slide_number = max(0, st.session_state.slide_number - 1)
                st.rerun()
        
        with col2:
            st.markdown(f"<center>Slide {st.session_state.slide_number + 1} of {len(slides_functions)}</center>", unsafe_allow_html=True)
        
        with col3:
            if st.button("Next ➡️", disabled=(st.session_state.slide_number == len(slides_functions) - 1), use_container_width=True):
                st.session_state.slide_number = min(len(slides_functions) - 1, st.session_state.slide_number + 1)
                st.rerun()
    
    else:
        # Demo mode
        if "Recommendations" in demo_page:
            demo_recommendations()
        elif "Predictions" in demo_page:
            demo_predictions()
        elif "Movie Info" in demo_page:
            demo_movie_info()
        elif "Popular" in demo_page:
            demo_popular()
        else:
            demo_system_status()

if __name__ == "__main__":
    main()