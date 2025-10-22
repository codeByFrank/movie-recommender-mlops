FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
  streamlit==1.29.0 \
  requests==2.31.0 \
  Pillow==10.1.0

# Copy app and images
COPY app.py /app/
COPY images/ /app/images/

# Expose port
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]