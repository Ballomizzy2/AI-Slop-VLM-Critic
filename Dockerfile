# Critic Pipeline — production Docker image
# Uses CPU-only PyTorch for Whisper (smaller image; GPU optional)

FROM python:3.12-slim-bookworm

# ffmpeg for video/audio extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first (Whisper dependency) — smaller than GPU build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create dirs for uploads/output (server creates if missing, but explicit is fine)
RUN mkdir -p uploads output

# Cloud platforms set PORT; default 7474 for local
ENV PORT=7474
EXPOSE 7474

CMD ["python", "server.py"]
