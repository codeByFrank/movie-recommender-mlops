import sqlite3
import pandas as pd
from pathlib import Path

def create_database():
    """Create SQLite database with simple structure"""
    print("Creating database...")
    
    # Create data directory if not exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Connect to database (creates it if not exists)
    db_path = data_dir / "movielens.db"
    conn = sqlite3.connect(db_path)
    
    print(f"Database created at: {db_path}")
    return conn

def create_tables(conn):
    """Create simple database tables"""
    print("Creating tables...")
    
    # Create movies table
    movies_sql = """
    CREATE TABLE IF NOT EXISTS movies (
        movieId INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        genres TEXT
    )
    """
    
    # Create ratings table  
    ratings_sql = """
    CREATE TABLE IF NOT EXISTS ratings (
        userId INTEGER NOT NULL,
        movieId INTEGER NOT NULL,
        rating REAL NOT NULL,
        timestamp INTEGER
    )
    """
    
    # Execute table creation
    conn.execute(movies_sql)
    conn.execute(ratings_sql)
    conn.commit()
    
    print("Tables created: movies, ratings")

def load_sample_data_to_database(conn):
    """Load sample data into database"""
    print("Loading sample data into database...")
    
    # Load sample data
    sample_dir = Path("data/sample")
    
    if not sample_dir.exists():
        print("ERROR: No sample data found!")
        print("Please run src/data/make_dataset.py first")
        return False
    
    # Load CSV files
    movies_df = pd.read_csv(sample_dir / "movies_sample.csv")
    ratings_df = pd.read_csv(sample_dir / "ratings_sample.csv")
    
    # Insert data into database
    movies_df.to_sql('movies', conn, if_exists='replace', index=False)
    ratings_df.to_sql('ratings', conn, if_exists='replace', index=False)
    
    print(f"Loaded {len(movies_df):,} movies into database")
    print(f"Loaded {len(ratings_df):,} ratings into database")
    
    return True

def create_indexes(conn):
    """Create indexes for better performance"""
    print("Creating database indexes...")
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(userId)",
        "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movieId)",
        "CREATE INDEX IF NOT EXISTS idx_movies_id ON movies(movieId)"
    ]
    
    for index_sql in indexes:
        conn.execute(index_sql)
    
    conn.commit()
    print("Indexes created")

def test_database(conn):
    """Test if database works correctly"""
    print("Testing database...")
    
    # Test queries
    cursor = conn.cursor()
    
    # Count movies
    cursor.execute("SELECT COUNT(*) FROM movies")
    movie_count = cursor.fetchone()[0]
    
    # Count ratings  
    cursor.execute("SELECT COUNT(*) FROM ratings")
    rating_count = cursor.fetchone()[0]
    
    # Average rating
    cursor.execute("SELECT AVG(rating) FROM ratings")
    avg_rating = cursor.fetchone()[0]
    
    print(f"Database test results:")
    print(f"- Movies in database: {movie_count:,}")
    print(f"- Ratings in database: {rating_count:,}")
    print(f"- Average rating: {avg_rating:.2f}")
    
    # Test a sample query
    cursor.execute("""
        SELECT m.title, AVG(r.rating) as avg_rating, COUNT(r.rating) as num_ratings
        FROM movies m 
        JOIN ratings r ON m.movieId = r.movieId
        GROUP BY m.movieId
        ORDER BY num_ratings DESC
        LIMIT 5
    """)
    
    print("\nTop 5 most rated movies:")
    for i, (title, avg_rating, num_ratings) in enumerate(cursor.fetchall(), 1):
        print(f"{i}. {title} - {avg_rating:.1f} stars ({num_ratings} ratings)")

def main():
    """Main function to setup database"""
    print("=== SETTING UP MOVIE DATABASE ===")
    
    # Step 1: Create database
    conn = create_database()
    
    # Step 2: Create tables
    create_tables(conn)
    
    # Step 3: Load sample data
    success = load_sample_data_to_database(conn)
    
    if success:
        # Step 4: Create indexes
        create_indexes(conn)
        
        # Step 5: Test database
        test_database(conn)
        
        print("\n=== DATABASE SETUP COMPLETE ===")
        print("Next step: Run src/models/train_model.py")
    else:
        print("Database setup failed!")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main()