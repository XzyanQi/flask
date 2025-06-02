FROM python:3.10-alpine

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set NLTK_DATA environment variable
ENV NLTK_DATA /app/nltk_data

# (Karena sudah ada lokal)
# COPY folder nltk_data dari lokal ke container
COPY nltk_data $NLTK_DATA

# Expose port Railway
EXPOSE 8080

# Jalankan aplikasi pakai gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]

