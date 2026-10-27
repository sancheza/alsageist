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
    libatomic1 \
    libportaudio2 \
    libgpiod2 \
    gpiod \
    dbus-bin \
    network-manager \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source code and entrypoint
COPY pyproject.toml constraints.txt README.md main.py ./
COPY core/ ./core/
COPY services/ ./services/
COPY scripts/ ./scripts/
COPY data/personalities/ ./data.defaults/personalities/
COPY docker/entrypoint.sh /entrypoint.sh

# Install AlsaGeist and its dependencies (including Raspberry Pi GPIO libraries) in editable mode,
# pinned to the versions in constraints.txt. The compilers and -dev packages exist only for this
# step and are removed in the same layer, so they do not stay in the image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libasound2-dev portaudio19-dev libgpiod-dev python3-dev && \
    chmod +x /entrypoint.sh && \
    pip install --upgrade pip setuptools wheel && \
    pip install -c constraints.txt -e ".[rpi]" && \
    apt-get purge -y --auto-remove gcc g++ libasound2-dev portaudio19-dev libgpiod-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create non-root alsageist user and volume mount targets
RUN groupadd -f -r gpio && \
    groupadd -g 1000 alsageist && \
    useradd -u 1000 -g alsageist -G audio,gpio -m alsageist && \
    mkdir -p /app/data /app/models /app/logs && \
    chown -R alsageist:alsageist /app

USER alsageist

# Runtime environment defaults
ENV PERSONALITY=grimjaw \
    VOLUME=0.5 \
    MIC_GAIN=1.5 \
    VAD_THRESHOLD=0.0035

VOLUME ["/app/data", "/app/models", "/app/logs"]

# Healthy while the daemon answers on its control socket
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 scripts/alsageist_ctl.py status --json > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "main.py"]
