FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  streamlit==1.29.0 \
  requests==2.31.0

# Copy app
COPY app.py /app/

# Expose port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]