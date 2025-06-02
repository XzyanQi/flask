FROM python:3.10-slim

# Pasang dependensi sistem dulu (biar Sastrawi & cffi bisa install)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy project
COPY . .

# Install dependencies Python (Sastrawi sekarang harusnya berhasil)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download resource NLTK
RUN python -m nltk.downloader punkt stopwords

# Buka port Railway
EXPOSE 8080

# Jalankan dengan Gunicorn
CMD ["gunicorn", "-w", "2", "--timeout", "300", "-b", "0.0.0.0:8080", "application:application"]
