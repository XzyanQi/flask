FROM python:3.10-slim

# Install dependensi sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements terlebih dahulu
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies satu per satu untuk debug lebih mudah
RUN pip install --no-cache-dir wheel setuptools

# Install packages penting dulu
RUN pip install --no-cache-dir flask gunicorn

# Install packages lain (skip yang error)
RUN pip install --no-cache-dir -r requirements.txt || echo "Beberapa package gagal diinstall, tapi lanjut..."

# Copy aplikasi
COPY . .

# Download NLTK data jika diperlukan
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" 2>/dev/null || echo "NLTK download dilewat"

# Expose port
EXPOSE 8080

# Jalankan aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
