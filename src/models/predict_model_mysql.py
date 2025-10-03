import pickle
import numpy as np
import mysql.connector
from pathlib import Path
import os

class MovieRecommender:
    """Movie recommender using baseline + SVD model"""
    
    def __init__(self):
        """Load the trained model with baseline statistics"""
        self.models_loaded = False
        self.load_models()
    
    
    def load_models(self):
        """Load all saved model components including baseline statistics"""
        print("Loading trained models...")
        
        # FIXED: Finde models/ Ordner egal von wo das Skript ausgeführt wird 
        current_dir = Path.cwd()
        
        # Suche nach models/ Ordner im aktuellen oder Parent-Verzeichnis
        if (current_dir / "models").exists():
            models_dir = current_dir / "models"
        elif (current_dir.parent / "models").exists():
            models_dir = current_dir.parent / "models"
        else:
            # Letzte Option: vom Skript-Verzeichnis aus
            script_dir = Path(__file__).parent
            models_dir = script_dir.parent.parent / "models"
        
        print(f"Current working directory: {current_dir}")
        print(f"Using models directory: {models_dir.absolute()}")
        print(f"Models directory exists: {models_dir.exists()}")
        
        if not models_dir.exists():
            print("❌ Models directory not found anywhere!")
            self.models_loaded = False
            return
        
        try:
            # Rest des Codes bleibt gleich...
            with open(models_dir / "svd_model.pkl", 'rb') as f:
                self.svd = pickle.load(f)

            print(f"Current working directory: {models_dir / 'svd_model.pkl'}")

            # Load user factors
            with open(models_dir / "user_factors.pkl", 'rb') as f:
                self.user_factors = pickle.load(f)
            

            # Load item factors
            with open(models_dir / "item_factors.pkl", 'rb') as f:
                self.item_factors = pickle.load(f)
            
            # Load user-item matrix
            with open(models_dir / "user_item_matrix.pkl", 'rb') as f:
                self.user_item_matrix = pickle.load(f)
            
            # Load baseline statistics (NEW - this fixes the 0.00 problem)
            try:
                with open(models_dir / "baseline_stats.pkl", 'rb') as f:
                    baseline_stats = pickle.load(f)
                self.global_mean = baseline_stats['global_mean']
                self.user_means = baseline_stats['user_means']
                self.movie_means = baseline_stats['movie_means']
                print(f"Loaded baseline statistics: global_mean={self.global_mean:.3f}")
            except FileNotFoundError:
                print("Warning: No baseline statistics found, using fallback values")
                # Fallback if baseline stats don't exist
                self.global_mean = 3.5
                self.user_means = {}
                self.movie_means = {}
            
            self.models_loaded = True
            print("Models loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found: {e}")
            print("Please run src/models/train_model.py first")
            self.models_loaded = False
    
    def get_user_recommendations(self, user_id, n_recommendations=5):
        """Get movie recommendations using baseline + SVD approach"""
        
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        # Check if user exists in our training data
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found, returning popular movies")
            return self.get_popular_movies(n_recommendations)
        
        # Get user index and bias
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean
        
        # Calculate SVD predictions for all movies
        user_vector = self.user_factors[user_idx]
        svd_scores = np.dot(user_vector, self.item_factors)
        
        # Get movies the user hasn't rated yet
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Create recommendations with baseline + SVD predictions
        recommendations = []
        movie_ids = self.user_item_matrix.columns
        
        for movie_id in unrated_movies:
            movie_idx = movie_ids.get_loc(movie_id)
            
            # Movie bias
            movie_bias = self.movie_means.get(movie_id, self.global_mean) - self.global_mean
            
            # SVD component
            svd_component = svd_scores[movie_idx]
            
            # Final prediction: baseline + SVD (this is the Netflix approach)
            predicted_rating = self.global_mean + user_bias + movie_bias + svd_component
            
            # Clamp to valid rating range
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            
            recommendations.append({
                'movie_id': int(movie_id),
                'predicted_rating': float(predicted_rating)
            })
        
        # Sort by predicted rating (highest first)
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Get top N recommendations with movie details
        top_recommendations = recommendations[:n_recommendations]
        
        # Add movie titles and genres
        for rec in top_recommendations:
            movie_info = self.get_movie_info(rec['movie_id'])
            rec.update(movie_info)
        
        return top_recommendations
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating using baseline + SVD approach"""
        
        if not self.models_loaded:
            return {"error": "Models not loaded"}
        
        # Check if user and movie exist in our data
        if user_id not in self.user_item_matrix.index:
            return {"error": f"User {user_id} not found"}
        
        if movie_id not in self.user_item_matrix.columns:
            return {"error": f"Movie {movie_id} not found"}
        
        # Get indices and biases
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        
        user_bias = self.user_means.get(user_id, self.global_mean) - self.global_mean
        movie_bias = self.movie_means.get(movie_id, self.global_mean) - self.global_mean
        
        # Calculate SVD prediction
        svd_prediction = np.dot(self.user_factors[user_idx], self.item_factors[:, movie_idx])
        
        # Final prediction: baseline + SVD
        predicted_rating = self.global_mean + user_bias + movie_bias + svd_prediction
        
        # Clamp to valid rating range
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        # Get movie info
        movie_info = self.get_movie_info(movie_id)
        
        return {
            'user_id': user_id,
            'movie_id': movie_id,
            'predicted_rating': float(predicted_rating),
            'title': movie_info.get('title', 'Unknown'),
            'genres': movie_info.get('genres', 'Unknown'),
            'components': {
                'global_mean': float(self.global_mean),
                'user_bias': float(user_bias), 
                'movie_bias': float(movie_bias),
                'svd_component': float(svd_prediction)
            }
        }
    
    def get_movie_info(self, movie_id):
        """Get movie title and genres from database"""
        try:
            # Connect to MySQL database
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='mysql',
                database='movielens'
            )
            cursor = conn.cursor()
            
            cursor.execute("SELECT title, genres FROM movies WHERE movieId = %s", (movie_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'title': result[0],
                    'genres': result[1]
                }
            else:
                return {
                    'title': f'Movie {movie_id}',
                    'genres': 'Unknown'
                }
                
        except Exception as e:
            print(f"Error getting movie info: {e}")
            return {
                'title': f'Movie {movie_id}',
                'genres': 'Unknown'
        }

    def get_popular_movies(self, n_movies=5):
        """Get popular movies as fallback"""
        try:
            # Connect to MySQL database
            conn = mysql.connector.connect(
                host='localhost',
                user='root',
                password='mysql',
                database='movielens'
            )
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT m.movieId, m.title, m.genres, AVG(r.rating) as avg_rating, COUNT(r.rating) as num_ratings
                FROM movies m
                JOIN ratings r ON m.movieId = r.movieId
                GROUP BY m.movieId, m.title, m.genres
                HAVING num_ratings >= 20
                ORDER BY avg_rating DESC
                LIMIT %s
            """, (n_movies,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': row[2],
                    'avg_rating': round(row[3], 2),
                    'num_ratings': row[4],
                    'predicted_rating': row[3]  # Use avg rating as prediction
                })
            
            cursor.close()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting popular movies: {e}")
            return []

def test_predictions():
    """Test the recommender system"""
    print("=== TESTING MOVIE RECOMMENDER ===")
    
    # Create recommender
    recommender = MovieRecommender()
    
    if not recommender.models_loaded:
        print("Cannot test - models not loaded")
        return
    
    # Test with first user in our data
    sample_user = recommender.user_item_matrix.index[0]
    print(f"\nGetting recommendations for user {sample_user}:")
    
    recommendations = recommender.get_user_recommendations(sample_user, 3)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} - {rec['genres']}")
        print(f"   Predicted rating: {rec['predicted_rating']:.2f}")
    
    # Test rating prediction with component breakdown
    sample_movie = recommender.user_item_matrix.columns[0]
    print(f"\nPredicting rating for user {sample_user}, movie {sample_movie}:")
    
    prediction = recommender.predict_rating(sample_user, sample_movie)
    if 'error' not in prediction:
        print(f"Predicted rating: {prediction['predicted_rating']:.2f}")
        print(f"Movie: {prediction['title']}")
        print("Components breakdown:")
        comp = prediction['components']
        print(f"  - Global mean: {comp['global_mean']:.3f}")
        print(f"  - User bias: {comp['user_bias']:.3f}")
        print(f"  - Movie bias: {comp['movie_bias']:.3f}")  
        print(f"  - SVD component: {comp['svd_component']:.3f}")
        print(f"  - Total: {comp['global_mean'] + comp['user_bias'] + comp['movie_bias'] + comp['svd_component']:.3f}")

if __name__ == "__main__":
    test_predictions()