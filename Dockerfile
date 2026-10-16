# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install ALSA, PortAudio, GPIO runtime packages and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    libasound2 \
    libasound2-plugins \
    libportaudio2 \
    libgpiod2 \
    gpiod \
    systemd \
    dbus-bin \
    curl \
    unzip \
    ca-certificates \
    gcc \
    g++ \
    libasound2-dev \
    portaudio19-dev \
    libgpiod-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade build tools and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install sounddevice numpy piper-tts onnxruntime gpiozero rpi-lgpio vosk

# Copy source code and entrypoint
COPY core/ ./core/
COPY services/ ./services/
COPY scripts/ ./scripts/
COPY main.py ./
COPY data/personalities/ ./data.defaults/personalities/
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && \
    pip install -e .

# Create volume mount targets
RUN mkdir -p /app/data /app/models /app/logs

# Runtime environment defaults
ENV PERSONALITY=grimjaw \
    VOLUME=0.5 \
    MIC_GAIN=1.5 \
    VAD_THRESHOLD=0.0035

VOLUME ["/app/data", "/app/models", "/app/logs"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "main.py"]
