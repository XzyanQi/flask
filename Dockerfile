FROM python:3.10-slim

# Install sistem dependencies agar pip install transformers tidak gagal
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Copy requirements terlebih dahulu
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install semua dependencies tanpa skip error
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file project
COPY . .

# Download data NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Expose port Railway
EXPOSE 8080

# Jalankan aplikasi pakai gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
