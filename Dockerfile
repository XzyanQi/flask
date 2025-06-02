FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential git wget libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NLTK_DATA=/app/nltk_data
COPY nltk_data /app/nltk_data

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers=1", "--timeout=300", "application:application"]
