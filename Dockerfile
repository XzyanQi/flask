FROM python:3.10-slim

# Install Debian deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dan install dependencies (pakai source CPU untuk torch)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy semua file
COPY . .

# Set env untuk NLTK
ENV NLTK_DATA=/app/nltk_data
COPY nltk_data /app/nltk_data

EXPOSE 8080

# Jalankan pakai gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
