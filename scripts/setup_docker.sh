#!/bin/bash
# scripts/setup_docker.sh - Automated Docker deployment & setup for Raspbian OS.
#
# Configures audio drivers, installs Docker, provisions external data directories,
# downloads default voice models, configures environment keys, and sets up autostart.
#
# Usage:
#     ./scripts/setup_docker.sh
#     ./scripts/setup_docker.sh --profile aiy
#     ./scripts/setup_docker.sh --profile usb
#     ./scripts/setup_docker.sh --non-interactive
#     ./scripts/setup_docker.sh --dry-run
#     ./scripts/setup_docker.sh -h | --help
#     ./scripts/setup_docker.sh -v | --version

set -eo pipefail

die() { echo "Error: $*" >&2; exit 1; }

VERSION="1.1.0"
PROFILE="aiy"
DRY_RUN=false
NON_INTERACTIVE=false
PREWARM=false
USER_ARG=""
DIR_ARG=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Automated deployment tool for AlsaGeist container on Raspberry Pi OS (Raspbian).

Options:
  -p, --profile <aiy|usb>    Audio hardware profile (default: aiy)
                             - aiy: Google AIY Voice HAT V1 (I2S sound card + GPIO)
                             - usb: USB Audio adapter / speakerphone
  -u, --user <username>      Target Linux user (default: current user)
  -d, --dir <path>           Installation directory (default: current repo root)
  -w, --prewarm              Pre-warm speech audio cache during setup
  -n, --non-interactive      Run non-interactively using default settings
  --dry-run                  Preview deployment actions without modifying system
  -v, --version              Show script version
  -h, --help                 Show this help message
EOF
}

show_version() {
    echo "$(basename "$0") ${VERSION}"
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            show_version
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -n|--non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        -w|--prewarm)
            PREWARM=true
            shift
            ;;
        -p|--profile)
            [[ $# -ge 2 ]] || die "$1 requires a value."
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        -u|--user)
            [[ $# -ge 2 ]] || die "$1 requires a value."
            USER_ARG="$2"
            shift 2
            ;;
        --user=*)
            USER_ARG="${1#*=}"
            shift
            ;;
        -d|--dir)
            [[ $# -ge 2 ]] || die "$1 requires a value."
            DIR_ARG="$2"
            shift 2
            ;;
        --dir=*)
            DIR_ARG="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

if [[ "$PROFILE" != "aiy" && "$PROFILE" != "usb" ]]; then
    echo "Error: Invalid profile '$PROFILE'. Must be 'aiy' or 'usb'." >&2
    exit 1
fi

TARGET_USER="${USER_ARG:-${SUDO_USER:-$USER}}"
TARGET_DIR="${DIR_ARG:-$REPO_DIR}"
PROFILE_UPPER=$(echo "$PROFILE" | tr '[:lower:]' '[:upper:]')

echo "======================================================================"
echo " AlsaGeist Docker Deployment & Raspbian Onboarding v${VERSION}"
echo " Profile: ${PROFILE_UPPER} | Target User: ${TARGET_USER}"
echo " Install Path: ${TARGET_DIR}"
if [[ "$DRY_RUN" == true ]]; then
    echo " Mode: DRY RUN (no modifications will be made)"
fi
echo "======================================================================"

# Append KEY=VALUE to a file, starting a new line first if the file does not end with one
append_line() {
    local file="$1" line="$2"
    if [[ -s "$file" && -n "$(tail -c 1 "$file")" ]]; then
        echo >> "$file"
    fi
    echo "$line" >> "$file"
}

# Helper for elevated commands
run_root() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would execute: $*"
        return 0
    fi
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# 1. Hardware & Audio Driver Setup
echo ""
echo "[1/5] Configuring Host Audio Drivers & System Permissions..."
if [[ -f "${TARGET_DIR}/scripts/setup_rpi_os.sh" ]]; then
    CONFIG_FLAGS=("--profile" "$PROFILE" "--user" "$TARGET_USER")
    if [[ "$DRY_RUN" == true ]]; then
        CONFIG_FLAGS+=("--dry-run")
    fi
    bash "${TARGET_DIR}/scripts/setup_rpi_os.sh" "${CONFIG_FLAGS[@]}"
else
    echo "  Notice: scripts/setup_rpi_os.sh not found. Skipping driver overlay configuration."
fi

# Ensure asound.conf exists on host so Docker volume mount succeeds
if [[ ! -f "/etc/asound.conf" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would initialize empty /etc/asound.conf for Docker mount"
    else
        run_root touch /etc/asound.conf
    fi
fi

# 2. Docker Engine Verification & Installation
echo ""
echo "[2/5] Verifying Docker and Docker Compose..."
if command -v docker > /dev/null 2>&1; then
    echo "  Docker is installed: $(docker --version)"
else
    echo "  Docker not found. Installing Docker Engine with Docker's installation script (run as root)..."
    if [[ "$DRY_RUN" != true && "$NON_INTERACTIVE" != true ]]; then
        read -r -p "  Download https://get.docker.com and run it as root? (y/N): " DOCKER_CHOICE
        if [[ ! "$DOCKER_CHOICE" =~ ^[Yy]$ ]]; then
            echo "  Install Docker Engine manually, then run this script again." >&2
            exit 1
        fi
    fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would download https://get.docker.com to a temporary file and run it with sh"
    else
        DOCKER_INSTALLER="$(mktemp)"
        curl -fsSL https://get.docker.com -o "$DOCKER_INSTALLER"
        run_root sh "$DOCKER_INSTALLER"
        rm -f "$DOCKER_INSTALLER"
    fi
fi

# Verify Docker compose plugin
if docker compose version > /dev/null 2>&1; then
    echo "  Docker Compose is available: $(docker compose version --short)"
else
    echo "  Installing Docker Compose plugin..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would install docker-compose-plugin via apt"
    else
        run_root apt-get update
        run_root apt-get install -y docker-compose-plugin
    fi
fi

# Add target user to docker group
if getent group docker > /dev/null 2>&1; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would add user '${TARGET_USER}' to group 'docker'"
    else
        run_root usermod -aG docker "$TARGET_USER"
        echo "  Added user '${TARGET_USER}' to 'docker' group."
    fi
fi

# 3. External Directory Provisioning
echo ""
echo "[3/5] Setting Up Externalized Directories..."
DATA_DIR="${TARGET_DIR}/data"
MODELS_DIR="${TARGET_DIR}/models"
LOGS_DIR="${TARGET_DIR}/logs"
PIPER_DIR="${MODELS_DIR}/piper"
CACHE_DIR="${MODELS_DIR}/cache"

if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would create: ${DATA_DIR}/personalities, ${PIPER_DIR}, ${CACHE_DIR}, ${LOGS_DIR}"
else
    mkdir -p "${DATA_DIR}/personalities" "${PIPER_DIR}" "${CACHE_DIR}" "${LOGS_DIR}"
    echo "  Created external directories."
fi

# Seed Grimjaw personality profile if not present
GRIMJAW_PROFILE="${DATA_DIR}/personalities/grimjaw.json"
if [[ ! -f "$GRIMJAW_PROFILE" && -f "${TARGET_DIR}/data/personalities/grimjaw.json" ]]; then
    echo "  Found bundled grimjaw profile."
fi

# Download the Piper voice and Vosk offline STT models; download_model.py verifies pinned SHA-256 digests
DOWNLOAD_ARGS=(--all --models-dir "${MODELS_DIR}")
if [[ "$DRY_RUN" == true ]]; then
    DOWNLOAD_ARGS+=(--dry-run)
fi
echo "  Downloading and verifying voice and speech recognition models..."
python3 "${SCRIPT_DIR}/download_model.py" "${DOWNLOAD_ARGS[@]}"

# 4. Environment Configuration (.env)
echo ""
echo "[4/5] Configuring Environment Variables..."
ENV_FILE="${TARGET_DIR}/.env"

GROQ_KEY=""
GEMINI_KEY=""
SKIP_ENV_PROMPT=false

if [[ -f "$ENV_FILE" ]]; then
    if [[ "$NON_INTERACTIVE" == true ]]; then
        echo "  Existing .env detected. Keeping current configuration."
        SKIP_ENV_PROMPT=true
    else
        read -r -p "  Existing .env detected. Overwrite with new settings? (y/N): " OVERWRITE_CHOICE
        if [[ ! "$OVERWRITE_CHOICE" =~ ^[Yy]$ ]]; then
            echo "  Retaining existing .env file."
            SKIP_ENV_PROMPT=true
        fi
    fi
fi

if [[ "$SKIP_ENV_PROMPT" == false ]]; then
    if [[ "$NON_INTERACTIVE" == true ]]; then
        echo "  Non-interactive mode: Using default offline configuration."
    else
        echo "  AlsaGeist runs 100% offline by default with local intent matching and Piper TTS."
        echo "  You can optionally configure cloud STT (Whisper) or Gemini LLM fallback:"
        echo ""
        read -s -r -p "  Enter Groq API Key [leave empty for offline STT]: " GROQ_KEY
        echo ""
        read -s -r -p "  Enter Gemini API Key [leave empty for offline fallback]: " GEMINI_KEY
        echo ""
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would generate .env file with PERSONALITY=grimjaw, VOLUME=0.5, MIC_GAIN=1.5"
    else
        (
            umask 077
            cat << EOF > "$ENV_FILE"
# ==============================================================================
# AlsaGeist Runtime Configuration
# ==============================================================================
PERSONALITY=grimjaw
LLM_FALLBACK=$([[ -n "$GEMINI_KEY" || -n "$GROQ_KEY" ]] && echo "true" || echo "false")

GROQ_API_KEY=${GROQ_KEY}
GEMINI_API_KEY=${GEMINI_KEY}

VOLUME=0.5
MIC_GAIN=1.5
VAD_THRESHOLD=0.0035
EOF
        )
        chmod 600 "$ENV_FILE"
        echo "  Configured ${ENV_FILE} (mode 600)."
    fi
fi

# Host group IDs and GPIO device for docker-compose.yml
if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would record AUDIO_GID, GPIO_GID, and GPIO_DEVICE from the host in ${ENV_FILE}"
else
    for pair in "AUDIO_GID:audio" "GPIO_GID:gpio"; do
        var="${pair%%:*}"
        grp="${pair##*:}"
        gid="$(getent group "$grp" | cut -d: -f3 || true)"
        if [[ -n "$gid" ]] && ! grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
            append_line "$ENV_FILE" "${var}=${gid}"
            echo "  Recorded ${var}=${gid} in ${ENV_FILE}."
        fi
    done

    # Detect GPIO device: Pi 5 routes header lines to gpiochip4; Pi 3/4 uses gpiochip0
    if [[ -e "/dev/gpiochip4" ]] && grep -qi "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
        GPIO_CHIP="/dev/gpiochip4"
    else
        GPIO_CHIP="/dev/gpiochip0"
    fi
    if ! grep -q "^GPIO_DEVICE=" "$ENV_FILE" 2>/dev/null; then
        append_line "$ENV_FILE" "GPIO_DEVICE=${GPIO_CHIP}"
        echo "  Recorded GPIO_DEVICE=${GPIO_CHIP} in ${ENV_FILE}."
    fi
fi

# Optional Dialogue Speech Cache Pre-Warming
DO_PREWARM=false
if [[ "$PREWARM" == true ]]; then
    DO_PREWARM=true
elif [[ "$NON_INTERACTIVE" == false ]]; then
    echo ""
    read -r -p "  Pre-warm audio dialogue cache now (~25 responses, ~15 min on Pi 3)? (y/N): " PREWARM_CHOICE
    if [[ "$PREWARM_CHOICE" =~ ^[Yy]$ ]]; then
        DO_PREWARM=true
    fi
fi

if [[ "$DO_PREWARM" == true ]]; then
    echo "  Pre-warming dialogue audio cache..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would run: python3 ${TARGET_DIR}/scripts/prewarm_cache.py --all"
    else
        if command -v python3 > /dev/null 2>&1 && python3 -c "import piper" > /dev/null 2>&1; then
            python3 "${TARGET_DIR}/scripts/prewarm_cache.py" --all || true
        else
            echo "  Host Python environment lacks piper-tts. Cache can be pre-warmed inside container after startup:"
            echo "    docker compose exec alsageist python3 scripts/prewarm_cache.py --all"
        fi
    fi
fi

# 5. Systemd Service Deployment
echo ""
echo "[5/5] Configuring Systemd Autostart Service..."
SERVICE_PATH="/etc/systemd/system/alsageist.service"

SERVICE_CONTENT="[Unit]
Description=AlsaGeist Conversational Display Container Service
After=network-online.target sound.target docker.service
Wants=network-online.target sound.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${TARGET_DIR}
ExecStart=/usr/bin/docker compose up -d --remove-orphans
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target"

if [[ "$DRY_RUN" == true ]]; then
    echo "  [DRY RUN] Would write systemd unit to ${SERVICE_PATH}:"
    echo "${SERVICE_CONTENT}"
else
    if [[ -f "$SERVICE_PATH" ]] && ! grep -q "docker compose up" "$SERVICE_PATH"; then
        # The existing unit is the bare-metal service; keep a copy before replacing it
        run_root cp -p "$SERVICE_PATH" "${SERVICE_PATH}.alsageist.bak"
        echo "  Backed up the existing unit to ${SERVICE_PATH}.alsageist.bak"
    fi
    printf "%s\n" "$SERVICE_CONTENT" | run_root tee "$SERVICE_PATH" > /dev/null
    run_root systemctl daemon-reload
    run_root systemctl enable alsageist.service
    echo "  Enabled alsageist.service autostart on system boot."
fi

# Final Summary
echo ""
echo "======================================================================"
if [[ "$DRY_RUN" == true ]]; then
    echo " Dry-run complete. System was not modified."
else
    echo " AlsaGeist Setup Completed Successfully!"
    echo ""
    echo " External Directories:"
    echo "   - Data:   ${DATA_DIR}"
    echo "   - Models: ${MODELS_DIR}"
    echo "   - Logs:   ${LOGS_DIR}"
    echo ""
    echo " Dialogue Speech Cache:"
    echo "   - To pre-warm offline dialogue responses with zero playback latency, run:"
    echo "     docker compose exec alsageist python3 scripts/prewarm_cache.py --all"
    echo ""
    if [[ "$PROFILE" == "aiy" ]]; then
        echo " IMPORTANT: A system reboot is required to activate the AIY Voice HAT overlay:"
        echo "   sudo reboot"
        echo ""
        echo " After reboot, AlsaGeist will start automatically in the background."
    else
        echo " To start AlsaGeist immediately, run:"
        echo "   docker compose up -d"
    fi
    echo ""
    echo " To inspect live conversation logs:"
    echo "   docker compose logs -f"
fi
echo "======================================================================"
exit 0
