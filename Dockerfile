# Gunakan image dasar yang kompatibel dengan TensorFlow & FAISS
FROM python:3.10-alpine

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
    && apt-get clean

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (setelah install requirements)
RUN python -m nltk.downloader punkt stopwords

# Buka port
EXPOSE 8000

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:8000", "application:application"]
