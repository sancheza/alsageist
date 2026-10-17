#!/bin/bash
# scripts/setup_systemd_service.sh - Generate and install AlsaGeist systemd service unit.
#
# Generates a systemd unit file from alsageist.service, substituting
# user and directory paths defined in .env or passed via command-line arguments.
#
# Usage:
#     ./scripts/setup_systemd_service.sh
#     ./scripts/setup_systemd_service.sh --user pi --dir /home/pi/alsageist
#     ./scripts/setup_systemd_service.sh --dry-run
#     ./scripts/setup_systemd_service.sh --enable
#     ./scripts/setup_systemd_service.sh -h | --help
#     ./scripts/setup_systemd_service.sh -v | --version

set -eo pipefail

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEMPLATE_PATH="${REPO_DIR}/alsageist.service"
OUTPUT_PATH="/etc/systemd/system/alsageist.service"
ENV_PATH="${REPO_DIR}/.env"

DRY_RUN=false
ENABLE_SERVICE=false
ASSUME_YES=false
USER_ARG=""
DIR_ARG=""

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Generate and install the AlsaGeist systemd unit file from template.

Options:
  -u, --user <username>       Target Linux user (default: ALSAGEIST_USER in .env, or \$USER)
  -p, --dir <path>            Installation directory (default: ALSAGEIST_DIR in .env, or repository root)
  -t, --template <path>       Template file path (default: alsageist.service)
  -o, --output <path>         Output service file path (default: /etc/systemd/system/alsageist.service)
  -e, --enable                Reload systemd daemon and enable/start the service
  -y, --yes                   Skip Docker warning confirmation prompt
  -d, --dry-run               Preview generated service unit without writing to disk
  -v, --version               Show script version
  -h, --help                  Show this help message
EOF
}

show_version() {
    echo "$(basename "$0") ${VERSION}"
}

# Parse command line arguments
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
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -y|--yes|-f|--force)
            ASSUME_YES=true
            shift
            ;;
        -e|--enable)
            ENABLE_SERVICE=true
            shift
            ;;
        -u|--user)
            USER_ARG="$2"
            shift 2
            ;;
        --user=*)
            USER_ARG="${1#*=}"
            shift
            ;;
        -p|--dir)
            DIR_ARG="$2"
            shift 2
            ;;
        --dir=*)
            DIR_ARG="${1#*=}"
            shift
            ;;
        -t|--template)
            TEMPLATE_PATH="$2"
            shift 2
            ;;
        --template=*)
            TEMPLATE_PATH="${1#*=}"
            shift
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --output=*)
            OUTPUT_PATH="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

# Read ALSAGEIST_USER and ALSAGEIST_DIR from .env if present
TARGET_USER=""
TARGET_DIR=""

if [[ -f "$ENV_PATH" ]]; then
    while IFS='=' read -r raw_key raw_val || [[ -n "$raw_key" ]]; do
        key="$(echo "$raw_key" | tr -d '[:space:]')"
        if [[ "$key" =~ ^# ]] || [[ -z "$key" ]]; then
            continue
        fi
        if [[ "$key" == "ALSAGEIST_USER" ]]; then
            val="${raw_val//[\"\'[:space:]]/}"
            TARGET_USER="$val"
        elif [[ "$key" == "ALSAGEIST_DIR" ]]; then
            val="${raw_val//[\"\'[:space:]]/}"
            TARGET_DIR="$val"
        fi
    done < "$ENV_PATH"
fi

# Command-line arguments override .env settings; fallback to system defaults
if [[ -n "$USER_ARG" ]]; then
    TARGET_USER="$USER_ARG"
elif [[ -z "$TARGET_USER" ]]; then
    TARGET_USER="${USER:-pi}"
fi

if [[ -n "$DIR_ARG" ]]; then
    TARGET_DIR="$DIR_ARG"
elif [[ -z "$TARGET_DIR" ]]; then
    TARGET_DIR="$REPO_DIR"
fi

if [[ ! -f "$TEMPLATE_PATH" ]]; then
    echo "Error: Template file not found: ${TEMPLATE_PATH}" >&2
    exit 1
fi

# Render template content
# Escape the characters sed treats specially in a replacement (\\, &, and the | delimiter)
sed_escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }
RENDERED_CONTENT=$(sed \
    -e "s|{{ALSAGEIST_USER}}|$(sed_escape "$TARGET_USER")|g" \
    -e "s|{{ALSAGEIST_DIR}}|$(sed_escape "$TARGET_DIR")|g" \
    "$TEMPLATE_PATH")

echo "======================================================================"
echo " AlsaGeist Systemd Service Configuration"
echo "======================================================================"
echo "  Target User:        ${TARGET_USER}"
echo "  Working Directory:  ${TARGET_DIR}"
echo "  Template Source:    ${TEMPLATE_PATH}"
echo "  Target Output:      ${OUTPUT_PATH}"
echo "======================================================================"

# Docker environment check and warning
DOCKER_DETECTED=false
if command -v docker > /dev/null 2>&1; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "alsageist"; then
        DOCKER_DETECTED=true
    fi
fi

echo ""
echo "======================================================================"
echo " WARNING: BARE-METAL SERVICE INSTALLATION"
echo "======================================================================"
echo " This script generates and installs a systemd unit for running AlsaGeist"
echo " directly on the host OS via a native Python virtual environment."
echo ""
echo " This script is NOT required for Docker installations."
echo " Docker installations manage container lifecycle and autostart via"
echo " Docker Compose (or scripts/setup_docker.sh). Installing this bare-metal"
echo " unit on a Docker host may conflict with containerized processes."
if [[ "$DOCKER_DETECTED" == true ]]; then
    echo ""
    echo " Notice: An AlsaGeist Docker container was detected on this system."
fi
echo "======================================================================"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[DRY RUN] Would prompt for confirmation before installing service file."
    echo "[DRY RUN] Generated unit content for ${OUTPUT_PATH}:"
    echo "----------------------------------------------------------------------"
    echo "$RENDERED_CONTENT"
    echo "----------------------------------------------------------------------"
    echo "Dry-run complete. No files were written."
    exit 0
fi

if [[ "$ASSUME_YES" != true ]]; then
    echo ""
    CONFIRM_PROMPT=""
    read -r -p "Proceed with installing bare-metal systemd service? [y/N]: " CONFIRM_PROMPT || true
    if [[ ! "$CONFIRM_PROMPT" =~ ^[Yy]$ ]]; then
        echo "Installation aborted by user."
        exit 0
    fi
fi

# Write target service unit (owned by root, mode 644, so no unprivileged user can edit a unit that runs at boot)
TMP_UNIT="$(mktemp)"
trap 'rm -f "$TMP_UNIT"' EXIT
echo "$RENDERED_CONTENT" > "$TMP_UNIT"

if [[ $EUID -eq 0 ]]; then
    install -m 644 -o root -g root "$TMP_UNIT" "$OUTPUT_PATH"
elif [[ -w "$(dirname "$OUTPUT_PATH")" ]]; then
    install -m 644 "$TMP_UNIT" "$OUTPUT_PATH"
else
    sudo install -m 644 -o root -g root "$TMP_UNIT" "$OUTPUT_PATH"
fi

echo "  Successfully installed service unit to ${OUTPUT_PATH}."

if [[ "$ENABLE_SERVICE" == true ]]; then
    echo "  Reloading systemd daemon and enabling service..."
    if [[ $EUID -eq 0 ]]; then
        systemctl daemon-reload
        systemctl enable --now alsageist.service
    else
        sudo systemctl daemon-reload
        sudo systemctl enable --now alsageist.service
    fi
    echo "  Service enabled and started."
fi

echo "Done."
exit 0
