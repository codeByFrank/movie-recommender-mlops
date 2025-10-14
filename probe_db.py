from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.cwd() / ".env")

import os, time, traceback
import mysql.connector as mc

host = os.getenv("DATABASE_HOST", "127.0.0.1")
port = int(os.getenv("DATABASE_PORT", "3306"))
user = os.getenv("DATABASE_USER", "app")
pwd  = os.getenv("DATABASE_PASSWORD", "mysql")
db   = os.getenv("DATABASE_NAME", "movielens")

print("ENV:", host, port, user, db)

def try_connect(**kwargs):
    t0 = time.time()
    print("\n-> connecting with params:", kwargs)
    try:
        conn = mc.connect(**kwargs)
        dt = time.time() - t0
        print(f"OK: connect() returned in {dt:.2f}s")
        return conn
    except Exception as e:
        dt = time.time() - t0
        print(f"CONNECT ERROR after {dt:.2f}s -> {repr(e)}")
        traceback.print_exc()
        return None

# 1) Connect to the server WITHOUT selecting a database
conn = try_connect(
    host=host, port=port, user=user, password=pwd,
    connection_timeout=5,
    ssl_disabled=True,   # helps on some Windows setups
    use_pure=True        # pure Python implementation
)

if not conn:
    print("\nFAIL: cannot connect to server as", user)
    raise SystemExit(1)

try:
    with conn.cursor() as cur:
        cur.execute("SELECT VERSION()")
        print("Server version:", cur.fetchone()[0])

        cur.execute("SHOW DATABASES LIKE 'movielens'")
        found = cur.fetchone()
        print("DB 'movielens' exists?" , bool(found))
finally:
    conn.close()

# 2) Now connect WITH the database selected
conn2 = try_connect(
    host=host, port=port, user=user, password=pwd, database=db,
    connection_timeout=5,
    ssl_disabled=True, use_pure=True
)
if not conn2:
    print("\nFAIL: cannot connect to database", db)
    raise SystemExit(1)

with conn2.cursor() as cur:
    cur.execute("SELECT 1")
    print("Ping OK:", cur.fetchone())
conn2.close()
