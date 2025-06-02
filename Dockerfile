FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install wheel for better package compilation
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies with increased timeout and retry
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

# Copy the rest of the application
COPY . .

# Download NLTK resources (only if needed by your app)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || echo "NLTK download skipped"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "--workers", "2", "--timeout", "300", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "--error-logfile", "-", "application:application"]
