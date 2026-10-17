#!/bin/bash
# docker/entrypoint.sh - Container runtime entrypoint for AlsaGeist.
#
# Validates audio device access, seeds initial personality profiles,
# ensures default Piper neural voice model exists, and runs the display daemon.

set -e

DATA_DIR="/app/data"
MODELS_DIR="/app/models"
LOGS_DIR="/app/logs"
PIPER_DIR="${MODELS_DIR}/piper"
CACHE_DIR="${MODELS_DIR}/cache"

# 1. Seed default personality profile if external mount is empty
mkdir -p "${DATA_DIR}/personalities"
if [[ ! -f "${DATA_DIR}/personalities/grimjaw.json" && -f "/app/data.defaults/personalities/grimjaw.json" ]]; then
    echo "[AlsaGeist Bootstrap] Seeding default grimjaw personality to external data volume..."
    cp "/app/data.defaults/personalities/grimjaw.json" "${DATA_DIR}/personalities/grimjaw.json"
fi

# 2. Ensure model directories exist
mkdir -p "${PIPER_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${LOGS_DIR}"

# 3. Ensure Grimjaw voice model is available
GRIMJAW_MODEL="${PIPER_DIR}/en_GB-alan-medium.onnx"
if [[ ! -f "${GRIMJAW_MODEL}" ]]; then
    echo "[AlsaGeist Bootstrap] Default Grimjaw voice model not found in ${PIPER_DIR}."
    echo "[AlsaGeist Bootstrap] Downloading Piper neural model (en_GB-alan-medium)..."
    python3 /app/scripts/download_model.py --voice alan
fi

# 4. Ensure offline Vosk STT model is available for offline fallback
VOSK_DIR="${MODELS_DIR}/vosk-model-small-en-us-0.15"
if [[ ! -d "${VOSK_DIR}" ]]; then
    echo "[AlsaGeist Bootstrap] Offline Vosk model missing in ${MODELS_DIR}."
    echo "[AlsaGeist Bootstrap] Downloading Vosk speech recognition model..."
    python3 /app/scripts/download_model.py --vosk
fi

# 5. Display diagnostic sound hardware status
if command -v aplay > /dev/null 2>&1; then
    echo "[AlsaGeist Hardware] Audio output devices:"
    aplay -l 2>&1 | sed 's/^/  /' || true
    echo "[AlsaGeist Hardware] Audio capture devices:"
    arecord -l 2>&1 | sed 's/^/  /' || true
fi

# 6. Hand off execution to main command
exec "$@"
