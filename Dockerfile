# ──────────────────────────────────────────────
# Healthcare Knowledge Navigator — Dockerfile
# ──────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="Healthcare Knowledge Navigator Team"
LABEL description="FastAPI backend for the Healthcare RAG Assistant"

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data/raw_xml /app/data/parsed_json /app/data/chunks

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
