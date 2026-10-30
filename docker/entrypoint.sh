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

# 1. Ensure data, model and log directories exist and are writable by the container user
mkdir -p "${DATA_DIR}/personalities" "${PIPER_DIR}" "${CACHE_DIR}" "${LOGS_DIR}" 2>/dev/null || true
for dir in "${DATA_DIR}/personalities" "${PIPER_DIR}" "${CACHE_DIR}" "${LOGS_DIR}"; do
    if [[ ! -w "${dir}" ]]; then
        echo "[AlsaGeist Bootstrap] ERROR: ${dir} is not writable by uid $(id -u)." >&2
        echo "[AlsaGeist Bootstrap] On the host, run: sudo chown -R $(id -u):$(id -g) <the mounted data, models and logs directories>" >&2
        exit 1
    fi
done

# 2. Ensure personalities directory exists for user-defined profiles and overrides
mkdir -p "${DATA_DIR}/personalities"

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
    # Offline speech recognition is optional when a cloud key is configured, so a failed
    # download does not stop the service from starting.
    python3 /app/scripts/download_model.py --vosk || \
        echo "[AlsaGeist Bootstrap] WARNING: Vosk download failed; offline speech recognition is unavailable."
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
