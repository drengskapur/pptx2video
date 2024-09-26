# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies:
# ffmpeg - for video conversion and audio processing
# libreoffice - to convert PowerPoint to PDF
# poppler-utils - for converting PDF to images (used by pdf2image)
# curl and fonts - for additional font support and other utilities
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libreoffice \
    poppler-utils \
    curl \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies directly into the container
RUN pip install --no-cache-dir \
    aiohttp \
    openai \
    pdf2image \
    Pillow \
    python-pptx \
    tenacity

# Copy the current directory contents into the container at /app
COPY . .

# Expose the default command to convert a PowerPoint presentation to a video
CMD ["python", "scripts/pptx2video.py"]
