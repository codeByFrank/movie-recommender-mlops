FROM apache/airflow:2.9.2

# Install Python deps (prefer wheels so nothing compiles)
USER airflow
COPY airflow/requirements.txt /requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /requirements.txt
