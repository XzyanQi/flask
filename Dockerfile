FROM python:3.10-slim

# Dependensi Debian slim
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ libffi-dev libssl-dev git wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NLTK_DATA /app/nltk_data
COPY nltk_data $NLTK_DATA

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
