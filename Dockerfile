FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libreoffice \
    poppler-utils \
    curl \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    aiohttp \
    openai \
    pdf2image \
    Pillow \
    python-pptx \
    tenacity

COPY . .
