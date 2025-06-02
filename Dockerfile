# Gunakan image dasar yang kompatibel dengan TensorFlow & FAISS
FROM python:3.9-slim

# Install dependencies dasar
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

# Salin semua file
COPY . .

# Install dependencies dari requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk Flask
EXPOSE 8000

# Jalankan aplikasi
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "application:application"]
