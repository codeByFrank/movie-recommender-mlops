# src/data/create_database_mysql.py

print("DEBUG: running create_database_mysql.py")
print("DEBUG: file path:", __file__)

from pathlib import Path
import os
import sys
import traceback
import pandas as pd
from dotenv import load_dotenv

import mysql.connector as mc
from mysql.connector import Error

# ---------- Config & helpers ----------

ROOT = Path.cwd()
ENV_FILE = ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE)

DB_HOST = os.getenv("DATABASE_HOST", "mysql-ml")
DB_PORT = int(os.getenv("DATABASE_PORT", "3306"))
DB_USER = os.getenv("DATABASE_USER", "app")     # recommended non-root user
DB_PASS = os.getenv("DATABASE_PASSWORD", "mysql")
DB_NAME = os.getenv("DATABASE_NAME", "movielens")

SAMPLE_DIR = ROOT / "data" / "sample"
BATCH_SIZE = 1_000  # insert chunk size

def connect_server():
    """
    Connect to the MySQL *server* (no DB selected).
    Returns a connection or None.
    """
    print("Connecting to MySQL server...")
    print(f"→ host={DB_HOST} port={DB_PORT} user={DB_USER}")
    try:
        conn = mc.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            connection_timeout=5,   # fail fast
            autocommit=False,
            ssl_disabled=True,      # avoid Windows SSL quirks
            use_pure=True
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            _ = cur.fetchone()
        print("✅ Connected to MySQL server")
        return conn
    except Error as e:
        print(f"❌ MySQL Error while connecting: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error while connecting: {repr(e)}")
        traceback.print_exc()
    return None

def ensure_database(conn, db_name: str):
    """
    Ensure target database exists and can be used.
    Some users may not have CREATE DATABASE privilege; we handle that gracefully.
    """
    try:
        with conn.cursor() as cur:
            # Try to create if not exists (ok if user has global CREATE)
            try:
                cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
                print(f"Database '{db_name}' ready (created or already exists).")
            except Error as e:
                # If we lack privilege, it's fine as long as the DB already exists
                print(f"ℹ️ Could not CREATE DATABASE (likely privilege): {e}")

            # Verify it exists and switch to it
            cur.execute(f"SHOW DATABASES LIKE %s", (db_name,))
            found = cur.fetchone()
            if not found:
                print(f"❌ Database '{db_name}' does not exist and could not be created.")
                print("   → Start your MySQL container with -e MYSQL_DATABASE=movielens or grant CREATE privilege.")
                sys.exit(1)

            cur.execute(f"USE `{db_name}`")
            print(f"Switched to database '{db_name}'.")
    except Exception as e:
        print(f"❌ Error ensuring database '{db_name}': {e}")
        traceback.print_exc()
        sys.exit(1)

def create_tables(conn):
    """
    Create required tables with appropriate keys and indexes.
    """
    print("Creating tables with keys...")
    movies_sql = """
        CREATE TABLE IF NOT EXISTS movies (
            movieId INT PRIMARY KEY,
            title   VARCHAR(500) NOT NULL,
            genres  VARCHAR(200),
            INDEX idx_title (title(100))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    ratings_sql = """
        CREATE TABLE IF NOT EXISTS ratings (
            userId    INT NOT NULL,
            movieId   INT NOT NULL,
            rating    DECIMAL(2,1) NOT NULL,
            timestamp INT NOT NULL,
            PRIMARY KEY (userId, movieId, timestamp),
            FOREIGN KEY (movieId) REFERENCES movies(movieId)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            INDEX idx_user   (userId),
            INDEX idx_movie  (movieId),
            INDEX idx_rating (rating)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """

    try:
        with conn.cursor() as cur:
            cur.execute(movies_sql)
            print("Table 'movies' ready.")
            cur.execute(ratings_sql)
            print("Table 'ratings' ready.")
        conn.commit()
        print("✅ All tables created successfully.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating tables: {e}")
        traceback.print_exc()
        sys.exit(1)

def load_sample_data(conn):
    """
    Load sample CSVs (movies_sample.csv, ratings_sample.csv) into MySQL.
    """
    print("Loading sample data into database...")

    if not SAMPLE_DIR.exists():
        print("❌ No sample data found! Run: python -m src.data.make_dataset")
        sys.exit(1)

    movies_csv  = SAMPLE_DIR / "movies_sample.csv"
    ratings_csv = SAMPLE_DIR / "ratings_sample.csv"

    if not movies_csv.exists() or not ratings_csv.exists():
        print("❌ Sample CSVs missing in data/sample/. Run: python -m src.data.make_dataset")
        sys.exit(1)

    try:
        movies_df  = pd.read_csv(movies_csv)
        ratings_df = pd.read_csv(ratings_csv)
    except Exception as e:
        print(f"❌ Could not read sample CSVs: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            cur.execute("TRUNCATE TABLE ratings")
            cur.execute("TRUNCATE TABLE movies")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")

            # Movies
            print(f"Inserting {len(movies_df):,} movies...")
            movie_rows = [
                (int(r.movieId), str(r.title), str(r.genres))
                for _, r in movies_df.iterrows()
            ]
            cur.executemany(
                "INSERT INTO movies (movieId, title, genres) VALUES (%s, %s, %s)",
                movie_rows
            )
            print("✅ Movies inserted.")

            # Ratings (batched)
            print(f"Inserting {len(ratings_df):,} ratings in batches of {BATCH_SIZE}...")
            rating_rows = [
                (int(r.userId), int(r.movieId), float(r.rating), int(r.timestamp))
                for _, r in ratings_df.iterrows()
            ]
            for i in range(0, len(rating_rows), BATCH_SIZE):
                batch = rating_rows[i:i+BATCH_SIZE]
                cur.executemany(
                    "INSERT INTO ratings (userId, movieId, rating, timestamp) VALUES (%s, %s, %s, %s)",
                    batch
                )
                print(f"  → {min(i+BATCH_SIZE, len(rating_rows)):,}/{len(rating_rows):,}")

        conn.commit()
        print("✅ Sample data loaded successfully.")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

def test_database(conn):
    """
    Quick sanity checks & a few example queries.
    """
    print("\nTesting database...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM movies")
            movie_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM ratings")
            rating_count = cur.fetchone()[0]

            cur.execute("SELECT AVG(rating) FROM ratings")
            avg_rating = cur.fetchone()[0]

            print(f"- Movies in DB : {movie_count:,}")
            print(f"- Ratings in DB: {rating_count:,}")
            print(f"- Avg rating   : {avg_rating:.2f}" if avg_rating is not None else "- Avg rating   : N/A")

            cur.execute("""
                SELECT m.title, AVG(r.rating) AS avg_rating, COUNT(*) AS n
                FROM ratings r
                JOIN movies m ON m.movieId = r.movieId
                GROUP BY m.movieId, m.title
                ORDER BY n DESC
                LIMIT 5
            """)
            rows = cur.fetchall()
            print("\nTop 5 most rated movies:")
            for i, (title, ar, n) in enumerate(rows, 1):
                print(f"  {i}. {title} — {ar:.2f} ({n} ratings)")

        print("✅ Test queries ran successfully.")
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        traceback.print_exc()
        sys.exit(1)

# ---------- Main ----------

def main():
    print("=== SETTING UP MYSQL MOVIE DATABASE ===")
    print("\nIMPORTANT: Make sure your container is up (docker ps shows mysql-ml Up) and .env is correct.\n")

    conn = connect_server()
    if not conn:
        print("Failed to connect to MySQL. Exiting...")
        sys.exit(1)

    try:
        ensure_database(conn, DB_NAME)
        create_tables(conn)
        load_sample_data(conn)
        test_database(conn)
        print("\n=== MYSQL DATABASE SETUP COMPLETE ✅ ===")
        print("Next step: train the model:  python -m src.models.train_model_mysql")
    finally:
        try:
            if conn.is_connected():
                conn.close()
                print("\nMySQL connection closed")
        except Exception:
            pass

if __name__ == "__main__":
    main()
