FROM python:3.10-slim

# Install dependensi sistem yang ringan & penting
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libgl1-mesa-glx \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements dan install python package
COPY requirements.txt .

# Upgrade pip & install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin semua file project
COPY . .

# Download data NLTK (hindari crash preprocessing)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Buka port Railway
EXPOSE 8080

# Jalankan pakai gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
