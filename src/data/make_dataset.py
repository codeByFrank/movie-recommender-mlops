import pandas as pd
import numpy as np
import os
import requests
import zipfile
from pathlib import Path

def download_movielens_data():
    """Download MovieLens 20M dataset - STREAMING with simple progress"""
    import requests, math
    from pathlib import Path

    print("Starting data download...")
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-20m.zip"

    if not zip_path.exists():
        print("Downloading MovieLens 20M dataset... (1.5 GB)")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            chunk = 1024 * 1024  # 1 MB
            done = 0
            with open(zip_path, "wb") as f:
                for part in r.iter_content(chunk_size=chunk):
                    if part:
                        f.write(part)
                        done += len(part)
                        if total:
                            pct = math.floor(done * 100 / total)
                            if pct % 5 == 0:  # print every ~5%
                                print(f"... {pct}%")
        print("Download completed!")
    else:
        print("Dataset already downloaded!")
    
    # Extract files if not extracted
    extract_dir = data_dir / "ml-20m"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction completed!")
    else:
        print("Dataset already extracted!")
    
    return extract_dir

def load_data(data_path):
    """Load the main datasets - SIMPLE VERSION"""
    print("Loading datasets...")
    
    # Load main files
    ratings = pd.read_csv(data_path / "ratings.csv")
    movies = pd.read_csv(data_path / "movies.csv")
    
    print(f"Loaded {len(ratings):,} ratings")
    print(f"Loaded {len(movies):,} movies")
    
    return ratings, movies

def create_sample_data(ratings, movies, sample_size=50000):
    """Create smaller sample for development - SIMPLE VERSION"""
    print(f"Creating sample with {sample_size:,} ratings...")
    
    # Take random sample of ratings
    sample_ratings = ratings.sample(n=min(sample_size, len(ratings)), random_state=42)
    
    # Get movies that appear in sample
    sample_movie_ids = sample_ratings['movieId'].unique()
    sample_movies = movies[movies['movieId'].isin(sample_movie_ids)]
    
    # Save sample data
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_ratings.to_csv(sample_dir / "ratings_sample.csv", index=False)
    sample_movies.to_csv(sample_dir / "movies_sample.csv", index=False)
    
    print(f"Sample created: {len(sample_ratings):,} ratings, {len(sample_movies):,} movies")
    
    return sample_ratings, sample_movies

def show_basic_info(ratings, movies):
    """Show basic information about the data - SIMPLE VERSION"""
    print("\n=== DATA OVERVIEW ===")
    print(f"Total ratings: {len(ratings):,}")
    print(f"Unique users: {ratings['userId'].nunique():,}")
    print(f"Unique movies: {ratings['movieId'].nunique():,}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    
    print("\nRating distribution:")
    for rating in sorted(ratings['rating'].unique()):
        count = len(ratings[ratings['rating'] == rating])
        print(f"Rating {rating}: {count:,} ({count/len(ratings)*100:.1f}%)")

def main():
    """Main function to run data processing"""
    print("=== MOVIE RECOMMENDATION DATA PIPELINE ===")
    
    # Step 1: Download data
    data_path = download_movielens_data()
    
    # Step 2: Load data
    ratings, movies = load_data(data_path)
    
    # Step 3: Show basic info
    show_basic_info(ratings, movies)
    
    # Step 4: Create sample for development
    sample_ratings, sample_movies = create_sample_data(ratings, movies)
    
    print("\n=== DATA PROCESSING COMPLETE ===")
    print("Next step: Run src/models/train_model.py")

# Run this if script is called directly
if __name__ == "__main__":
    main()