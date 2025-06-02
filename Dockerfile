FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Salin requirements dulu biar pip install bisa cache step-nya
COPY requirements.txt .

# Upgrade pip dan install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip

# Copy seluruh source code
COPY . .

# Set NLTK environment variable
ENV NLTK_DATA=/app/nltk_data

# Copy folder NLTK jika belum otomatis ikut
COPY nltk_data /app/nltk_data

# Port yang diekspos oleh aplikasi Flask (via gunicorn)
EXPOSE 8080

# Jalankan server gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]

