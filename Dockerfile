# Gunakan image dasar yang ringan & kompatibel
FROM python:3.10-slim

# Install dependencies dasar sistem + lib untuk audio, NLP, FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container, file berat ada di .dockerignore
COPY . .

# Install Python dependencies versi minimal agar image lebih kecil
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.12.0 \
    transformers \
    faiss-cpu \
    flask \
    numpy \
    nltk \
    gdown \
    gunicorn

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Buka port
EXPOSE 8080

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:8080", "application:application"]
