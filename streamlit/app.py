import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import os

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")
API_USER = os.getenv("API_BASIC_USER", "admin")
API_PASS = os.getenv("API_BASIC_PASS", "secret")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page:", ["Get Recommendations", "Predict Rating", "Popular Movies"])

# Helper function to call API
def call_api(endpoint, method="GET", json_data=None):
    try:
        auth = HTTPBasicAuth(API_USER, API_PASS)
        url = f"{API_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, auth=auth, timeout=10)
        else:
            response = requests.post(url, auth=auth, json=json_data, timeout=10)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# Page 1: Get Recommendations
if page == "Get Recommendations":
    st.header("🎯 Get Movie Recommendations")
    st.write("Enter your User ID to get personalized movie recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    with col2:
        n_recommendations = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Fetching recommendations..."):
            data, error = call_api(
                "/recommendations",
                method="POST",
                json_data={"user_id": user_id, "n_recommendations": n_recommendations}
            )
        
        if error:
            st.error(error)
        else:
            st.success(f"Found {data['count']} recommendations for User {user_id}")
            
            # Display recommendations
            for idx, movie in enumerate(data['recommendations'], 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 2])
                    with col1:
                        st.metric("Rank", f"#{idx}")
                    with col2:
                        st.write(f"**{movie['title']}**")
                        st.caption(f"Movie ID: {movie['movieId']}")
                    with col3:
                        st.metric("Predicted Rating", f"{movie['predicted_rating']:.2f}⭐")
                st.divider()

# Page 2: Predict Rating
elif page == "Predict Rating":
    st.header("⭐ Predict Movie Rating")
    st.write("Predict what rating a user would give to a specific movie")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    with col2:
        movie_id = st.number_input("Movie ID", min_value=1, value=1, step=1)
    
    if st.button("Predict Rating", type="primary"):
        with st.spinner("Predicting..."):
            data, error = call_api(
                "/predict",
                method="POST",
                json_data={"user_id": user_id, "movie_id": movie_id}
            )
        
        if error:
            st.error(error)
        else:
            st.success("Prediction complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Rating", f"{data['predicted_rating']:.2f}⭐")
            with col2:
                # Get movie info
                movie_data, _ = call_api(f"/movie/{movie_id}")
                if movie_data:
                    st.write(f"**Movie:** {movie_data.get('title', 'Unknown')}")

# Page 3: Popular Movies
elif page == "Popular Movies":
    st.header("🔥 Popular Movies")
    st.write("Discover the most popular movies")
    
    n_movies = st.slider("Number of movies to show", min_value=5, max_value=50, value=10)
    
    if st.button("Get Popular Movies", type="primary"):
        with st.spinner("Fetching popular movies..."):
            data, error = call_api(f"/popular?n_movies={n_movies}")
        
        if error:
            st.error(error)
        else:
            st.success(f"Found {data['count']} popular movies")
            
            # Display in grid
            cols = st.columns(3)
            for idx, movie in enumerate(data['popular_movies']):
                with cols[idx % 3]:
                    with st.container():
                        st.subheader(movie['title'])
                        st.caption(f"Movie ID: {movie['movieId']}")
                        if 'genres' in movie:
                            st.write(f"📁 {movie['genres']}")
                        st.divider()

# Footer
st.markdown("---")
st.caption("Powered by MLflow, Airflow, and FastAPI")