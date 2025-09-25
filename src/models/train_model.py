import pandas as pd
import numpy as np
import sqlite3
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from pathlib import Path

def load_data_from_database():
    """Load data from SQLite database"""
    print("Loading data from database...")
    
    # Connect to database
    db_path = "data/movielens.db"
    conn = sqlite3.connect(db_path)
    
    # Load ratings and movies
    query = """
    SELECT r.userId, r.movieId, r.rating, m.title, m.genres
    FROM ratings r
    JOIN movies m ON r.movieId = m.movieId
    """
    
    data = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(data):,} ratings from database")
    return data

def create_user_item_matrix(data):
    """Create user-item rating matrix - FIXED for better SVD performance"""
    print("Creating user-item matrix...")
    
    # Create pivot table: users as rows, movies as columns, ratings as values
    user_item_matrix = data.pivot_table(
        index='userId',
        columns='movieId', 
        values='rating',
        fill_value=0
    )
    
    print(f"Matrix shape: {user_item_matrix.shape}")
    print(f"Users: {len(user_item_matrix.index)}")
    print(f"Movies: {len(user_item_matrix.columns)}")
    
    # Calculate baseline statistics for later normalization
    global_mean = data['rating'].mean()
    user_means = data.groupby('userId')['rating'].mean()
    movie_means = data.groupby('movieId')['rating'].mean()
    
    print(f"Global rating mean: {global_mean:.3f}")
    
    return user_item_matrix, global_mean, user_means, movie_means

def normalize_matrix_for_svd(user_item_matrix, global_mean):
    """Normalize the matrix for better SVD performance"""
    print("Normalizing matrix for SVD...")
    
    # Create a copy for SVD training
    normalized_matrix = user_item_matrix.copy()
    
    # Replace 0s with global mean for SVD training
    # This prevents SVD from learning that missing = 0
    mask = normalized_matrix == 0
    normalized_matrix[mask] = global_mean
    
    # Mean-center the data (subtract global mean)
    normalized_matrix = normalized_matrix - global_mean
    
    print("Matrix normalized (mean-centered)")
    return normalized_matrix

def train_svd_model(normalized_matrix, n_components=50):
    """Train SVD collaborative filtering model on normalized data"""
    print(f"Training SVD model with {n_components} components...")
    
    # Initialize SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit SVD on normalized matrix
    user_factors = svd.fit_transform(normalized_matrix)
    item_factors = svd.components_
    
    print("SVD model trained successfully!")
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
    
    return svd, user_factors, item_factors

def evaluate_model(data, user_item_matrix, user_factors, item_factors, global_mean, test_size=0.2):
    """Evaluate model with proper baseline + SVD predictions"""
    print("Evaluating model with baseline + SVD...")
    
    # Split data into train/test
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=42
    )
    
    # Calculate user and movie biases from training data
    train_global_mean = train_data['rating'].mean()
    train_user_bias = train_data.groupby('userId')['rating'].mean() - train_global_mean
    train_movie_bias = train_data.groupby('movieId')['rating'].mean() - train_global_mean
    
    predictions = []
    actuals = []
    
    # Make predictions for test set
    for _, row in test_data.head(1000).iterrows():  # Test on first 1000 for speed
        try:
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # Check if user and movie are in our matrix
            if user_id in user_item_matrix.index and movie_id in user_item_matrix.columns:
                user_idx = user_item_matrix.index.get_loc(user_id)
                item_idx = user_item_matrix.columns.get_loc(movie_id)
                
                # Get biases
                user_bias = train_user_bias.get(user_id, 0)
                movie_bias = train_movie_bias.get(movie_id, 0)
                
                # Calculate SVD prediction
                svd_prediction = np.dot(user_factors[user_idx], item_factors[:, item_idx])
                
                # Final prediction: global mean + biases + SVD
                predicted_rating = train_global_mean + user_bias + movie_bias + svd_prediction
                
                # Clamp to valid rating range
                predicted_rating = max(0.5, min(5.0, predicted_rating))
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                
        except Exception as e:
            continue
    
    if len(predictions) > 0:
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Model Evaluation Results:")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- MAE: {mae:.4f}")
        print(f"- Predictions made: {len(predictions)}")
        print(f"- Average prediction: {np.mean(predictions):.3f}")
        print(f"- Prediction range: {np.min(predictions):.3f} to {np.max(predictions):.3f}")
        
        return {'rmse': rmse, 'mae': mae, 'predictions': len(predictions)}
    else:
        print("Could not make any predictions for evaluation")
        return None

def save_model(svd, user_factors, item_factors, user_item_matrix, global_mean, user_means, movie_means):
    """Save trained model with baseline statistics"""
    print("Saving model and baseline statistics...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model components
    with open(models_dir / "svd_model.pkl", 'wb') as f:
        pickle.dump(svd, f)
        
    with open(models_dir / "user_factors.pkl", 'wb') as f:
        pickle.dump(user_factors, f)
        
    with open(models_dir / "item_factors.pkl", 'wb') as f:
        pickle.dump(item_factors, f)
        
    with open(models_dir / "user_item_matrix.pkl", 'wb') as f:
        pickle.dump(user_item_matrix, f)
    
    # Save baseline statistics for proper predictions
    baseline_stats = {
        'global_mean': global_mean,
        'user_means': user_means,
        'movie_means': movie_means
    }
    
    with open(models_dir / "baseline_stats.pkl", 'wb') as f:
        pickle.dump(baseline_stats, f)
    
    print("Model and baseline statistics saved to models/ directory")

def test_model_predictions(user_item_matrix, user_factors, item_factors, global_mean, user_means, movie_means):
    """Test the model with proper baseline + SVD predictions"""
    print("Testing model predictions with baseline...")
    
    # Get a random user
    sample_user = user_item_matrix.index[0]
    user_idx = 0
    
    # Calculate SVD scores for all movies for this user
    svd_scores = np.dot(user_factors[user_idx], item_factors)
    
    # Get user's existing ratings
    user_ratings = user_item_matrix.iloc[user_idx]
    unrated_movies = user_ratings[user_ratings == 0].index
    
    # Get user bias
    user_bias = user_means.get(sample_user, 0) - global_mean
    
    # Get top 5 predictions for unrated movies
    recommendations = []
    for movie_id in unrated_movies:
        movie_idx = user_item_matrix.columns.get_loc(movie_id)
        
        # Movie bias
        movie_bias = movie_means.get(movie_id, 0) - global_mean
        
        # SVD component
        svd_component = svd_scores[movie_idx]
        
        # Final prediction
        predicted_rating = global_mean + user_bias + movie_bias + svd_component
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        recommendations.append((movie_id, predicted_rating))
    
    # Sort by prediction and get top 5
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_5 = recommendations[:5]
    
    print(f"Sample recommendations for user {sample_user}:")
    print(f"User bias: {user_bias:.3f}, Global mean: {global_mean:.3f}")
    for i, (movie_id, score) in enumerate(top_5, 1):
        print(f"{i}. Movie ID {movie_id}: Predicted Rating {score:.3f}")

def main():
    """Main training function with fixes"""
    print("=== TRAINING MOVIE RECOMMENDATION MODEL ===")
    
    # Step 1: Load data from database
    data = load_data_from_database()
    
    # Step 2: Create user-item matrix and calculate baselines
    user_item_matrix, global_mean, user_means, movie_means = create_user_item_matrix(data)
    
    # Step 3: Normalize matrix for SVD
    normalized_matrix = normalize_matrix_for_svd(user_item_matrix, global_mean)
    
    # Step 4: Train SVD model on normalized data
    svd, user_factors, item_factors = train_svd_model(normalized_matrix, n_components=50)
    
    # Step 5: Evaluate model with baseline + SVD
    metrics = evaluate_model(data, user_item_matrix, user_factors, item_factors, global_mean)
    
    # Step 6: Save model with baseline statistics
    save_model(svd, user_factors, item_factors, user_item_matrix, global_mean, user_means, movie_means)
    
    # Step 7: Test predictions
    test_model_predictions(user_item_matrix, user_factors, item_factors, global_mean, user_means, movie_means)
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print("Model now uses: Global Mean + User Bias + Movie Bias + SVD")
    print("This should produce realistic rating predictions (0.5 - 5.0)")
    
    return metrics

if __name__ == "__main__":
    main()