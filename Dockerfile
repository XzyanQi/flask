FROM python:3.10-alpine

# Instal dependensi sistem untuk alpine
RUN apk update && apk add --no-cache \
    build-base \
    gcc \
    g++ \
    libffi-dev \
    openssl-dev \
    git \
    wget \
    && rm -rf /var/cache/apk/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NLTK_DATA /app/nltk_data
COPY nltk_data $NLTK_DATA

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "application:application"]
