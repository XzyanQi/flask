FROM tensorflow/tensorflow:2.11.0

# Install lib tambahan
RUN apt-get update && apt-get install -y \
    git wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dan install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh project
COPY . .

# NLTK
ENV NLTK_DATA=/app/nltk_data
COPY nltk_data /app/nltk_data

# Expose port
EXPOSE 8080

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
