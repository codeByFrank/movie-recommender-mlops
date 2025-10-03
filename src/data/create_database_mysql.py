import mysql.connector
from mysql.connector import Error
import pandas as pd
from pathlib import Path
import os

def create_database_connection():
    """
    Create connection to MySQL database
    
    You need to have MySQL running first:
    - Local MySQL installation, OR
    - MySQL Docker container
    
    Returns:
        connection object or None if failed
    """
    print("Connecting to MySQL server...")
    
    try:
        # Connect to MySQL server (without selecting database first)
        connection = mysql.connector.connect(
            host='localhost',        # MySQL server location
            user='root',            # MySQL username (change if different)
            password='mysql',    # MySQL password (change to your password)
            port=3306              # Default MySQL port
        )
        
        if connection.is_connected():
            print("Successfully connected to MySQL server")
            return connection
        
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        print("\nMake sure:")
        print("1. MySQL is running")
        print("2. Username and password are correct")
        print("3. Port 3306 is available")
        return None

def create_database(connection):
    """
    Create the movielens database if it does not exist
    
    Args:
        connection: MySQL connection object
    """
    print("Creating database...")
    
    cursor = connection.cursor()
    
    try:
        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS movielens")
        print("Database 'movielens' created (or already exists)")
        
        # Switch to the database
        cursor.execute("USE movielens")
        print("Switched to 'movielens' database")
        
    except Error as e:
        print(f"Error creating database: {e}")
    
    finally:
        cursor.close()

def create_tables(connection):
    """
    Create database tables with proper primary and foreign keys
    
    Tables created:
    1. movies - stores movie information
       PRIMARY KEY: movieId
       
    2. ratings - stores user ratings for movies
       COMPOSITE PRIMARY KEY: (userId, movieId, timestamp)
       FOREIGN KEY: movieId references movies(movieId)
    
    Args:
        connection: MySQL connection object
    """
    print("Creating tables with keys...")
    
    cursor = connection.cursor()
    
    try:
        # First, use the database
        cursor.execute("USE movielens")
        
        # Create movies table
        # PRIMARY KEY = movieId (unique identifier for each movie)
        movies_sql = """
        CREATE TABLE IF NOT EXISTS movies (
            movieId INT PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            genres VARCHAR(200),
            INDEX idx_title (title(100))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        cursor.execute(movies_sql)
        print("Table 'movies' created")
        print("  - PRIMARY KEY: movieId")
        print("  - INDEX: title (for faster searches)")
        
        # Create ratings table
        # COMPOSITE PRIMARY KEY = (userId, movieId, timestamp)
        # This ensures one user can only rate a movie once at a specific time
        # FOREIGN KEY = movieId references movies table
        ratings_sql = """
        CREATE TABLE IF NOT EXISTS ratings (
            userId INT NOT NULL,
            movieId INT NOT NULL,
            rating DECIMAL(2,1) NOT NULL,
            timestamp INT NOT NULL,
            PRIMARY KEY (userId, movieId, timestamp),
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            INDEX idx_user (userId),
            INDEX idx_movie (movieId),
            INDEX idx_rating (rating)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        cursor.execute(ratings_sql)
        print("Table 'ratings' created")
        print("  - PRIMARY KEY: (userId, movieId, timestamp)")
        print("  - FOREIGN KEY: movieId -> movies(movieId)")
        print("  - INDEX: userId (for user lookups)")
        print("  - INDEX: movieId (for movie lookups)")
        print("  - INDEX: rating (for sorting by rating)")
        
        connection.commit()
        print("\nAll tables created successfully")
        
    except Error as e:
        print(f"Error creating tables: {e}")
        connection.rollback()
    
    finally:
        cursor.close()

def load_sample_data_to_database(connection):
    """
    Load sample data from CSV files into MySQL database
    
    Args:
        connection: MySQL connection object
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Loading sample data into database...")
    
    # Check if sample data exists
    sample_dir = Path("data/sample")
    
    if not sample_dir.exists():
        print("ERROR: No sample data found!")
        print("Please run src/data/make_dataset.py first")
        return False
    
    cursor = connection.cursor()
    
    try:
        # Use the database
        cursor.execute("USE movielens")
        
        # Load movies data
        print("Loading movies...")
        movies_df = pd.read_csv(sample_dir / "movies_sample.csv")
        
        # Clear existing data
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        cursor.execute("TRUNCATE TABLE movies")
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
        
        # Insert movies in batches
        insert_movie_sql = """
        INSERT INTO movies (movieId, title, genres)
        VALUES (%s, %s, %s)
        """
        
        movie_data = [
            (int(row['movieId']), str(row['title']), str(row['genres']))
            for _, row in movies_df.iterrows()
        ]
        
        cursor.executemany(insert_movie_sql, movie_data)
        print(f"Loaded {len(movies_df):,} movies into database")
        
        # Load ratings data
        print("Loading ratings...")
        ratings_df = pd.read_csv(sample_dir / "ratings_sample.csv")
        
        # Clear existing ratings
        cursor.execute("TRUNCATE TABLE ratings")
        
        # Insert ratings in batches
        insert_rating_sql = """
        INSERT INTO ratings (userId, movieId, rating, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        
        rating_data = [
            (
                int(row['userId']), 
                int(row['movieId']), 
                float(row['rating']), 
                int(row['timestamp'])
            )
            for _, row in ratings_df.iterrows()
        ]
        
        # Insert in chunks to avoid timeout
        batch_size = 1000
        for i in range(0, len(rating_data), batch_size):
            batch = rating_data[i:i + batch_size]
            cursor.executemany(insert_rating_sql, batch)
            print(f"  Inserted {min(i + batch_size, len(rating_data)):,}/{len(rating_data):,} ratings")
        
        connection.commit()
        print(f"Loaded {len(ratings_df):,} ratings into database")
        
        return True
        
    except Error as e:
        print(f"Error loading data: {e}")
        connection.rollback()
        return False
    
    finally:
        cursor.close()

def test_database(connection):
    """
    Test if database works correctly by running sample queries
    
    Args:
        connection: MySQL connection object
    """
    print("\nTesting database...")
    
    cursor = connection.cursor()
    
    try:
        cursor.execute("USE movielens")
        
        # Test 1: Count movies
        cursor.execute("SELECT COUNT(*) FROM movies")
        movie_count = cursor.fetchone()[0]
        
        # Test 2: Count ratings
        cursor.execute("SELECT COUNT(*) FROM ratings")
        rating_count = cursor.fetchone()[0]
        
        # Test 3: Average rating
        cursor.execute("SELECT AVG(rating) FROM ratings")
        avg_rating = cursor.fetchone()[0]
        
        # Test 4: Check foreign key relationship
        cursor.execute("""
            SELECT COUNT(*) 
            FROM ratings r 
            LEFT JOIN movies m ON r.movieId = m.movieId 
            WHERE m.movieId IS NULL
        """)
        orphan_ratings = cursor.fetchone()[0]
        
        print(f"\nDatabase test results:")
        print(f"- Movies in database: {movie_count:,}")
        print(f"- Ratings in database: {rating_count:,}")
        print(f"- Average rating: {avg_rating:.2f}")
        print(f"- Orphan ratings (should be 0): {orphan_ratings}")
        
        # Test 5: Top rated movies
        cursor.execute("""
            SELECT m.title, AVG(r.rating) as avg_rating, COUNT(r.rating) as num_ratings
            FROM movies m 
            JOIN ratings r ON m.movieId = r.movieId
            GROUP BY m.movieId, m.title
            ORDER BY num_ratings DESC
            LIMIT 5
        """)
        
        print("\nTop 5 most rated movies:")
        for i, (title, avg_rating, num_ratings) in enumerate(cursor.fetchall(), 1):
            print(f"{i}. {title} - {avg_rating:.1f} stars ({num_ratings} ratings)")
        
        # Test 6: Verify primary and foreign keys
        cursor.execute("""
            SELECT 
                TABLE_NAME,
                COLUMN_NAME,
                CONSTRAINT_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = 'movielens'
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """)
        
        print("\nForeign key relationships:")
        for table, column, constraint, ref_table, ref_column in cursor.fetchall():
            print(f"  {table}.{column} -> {ref_table}.{ref_column}")
        
    except Error as e:
        print(f"Error testing database: {e}")
    
    finally:
        cursor.close()

def main():
    """
    Main function to setup MySQL database
    
    Steps:
    1. Connect to MySQL server
    2. Create database
    3. Create tables with keys
    4. Load sample data
    5. Test database
    """
    print("=== SETTING UP MYSQL MOVIE DATABASE ===")
    print("\nIMPORTANT: Make sure MySQL is running!")
    print("Update username/password in create_database_connection() function\n")
    
    # Step 1: Connect to MySQL
    connection = create_database_connection()
    
    if connection is None:
        print("Failed to connect to MySQL. Exiting...")
        return
    
    try:
        # Step 2: Create database
        create_database(connection)
        
        # Step 3: Create tables
        create_tables(connection)
        
        # Step 4: Load sample data
        success = load_sample_data_to_database(connection)
        
        if success:
            # Step 5: Test database
            test_database(connection)
            
            print("\n=== MYSQL DATABASE SETUP COMPLETE ===")
            print("\nDatabase structure:")
            print("1. movies table:")
            print("   - PRIMARY KEY: movieId")
            print("   - Stores movie information")
            print("\n2. ratings table:")
            print("   - PRIMARY KEY: (userId, movieId, timestamp)")
            print("   - FOREIGN KEY: movieId -> movies(movieId)")
            print("   - Stores user ratings")
            print("\nNext step: Update train_model.py to use MySQL connection")
        else:
            print("Database setup failed!")
    
    finally:
        # Close connection
        if connection.is_connected():
            connection.close()
            print("\nMySQL connection closed")

if __name__ == "__main__":
    main()