#!/bin/bash
# scripts/configure_rpi_os.sh - Configure Raspberry Pi OS audio drivers and user permissions.
#
# Configures /boot/firmware/config.txt (or /boot/config.txt), ALSA sound routing,
# and user group memberships for AlsaGeist hardware operation.
#
# Usage:
#     ./scripts/configure_rpi_os.sh
#     ./scripts/configure_rpi_os.sh --profile aiy
#     ./scripts/configure_rpi_os.sh --profile usb
#     ./scripts/configure_rpi_os.sh --dry-run
#     ./scripts/configure_rpi_os.sh -h | --help
#     ./scripts/configure_rpi_os.sh -v | --version

set -eo pipefail

VERSION="1.0.0"
PROFILE="aiy"
DRY_RUN=false
CONFIG_PATH=""
ASOUND_PATH="/etc/asound.conf"

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Configure Raspberry Pi OS for AlsaGeist audio hardware and user permissions.

Options:
  -p, --profile <aiy|usb>    Hardware profile to configure (default: aiy)
                             - aiy: Google AIY Voice HAT V1 (I2S sound card + GPIO)
                             - usb: USB Audio adapter / speakerphone
  -u, --user <username>      Target Linux user for audio/gpio groups (default: current user)
  -c, --config-file <path>   Path to config.txt (auto-detected if omitted)
  -a, --asound-file <path>   Path to asound.conf (default: /etc/asound.conf)
  -d, --dry-run              Show changes without writing files or modifying system
  -v, --version              Show script version
  -h, --help                 Show this help message
EOF
}

show_version() {
    echo "$(basename "$0") ${VERSION}"
}

USER_ARG=""

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
        -u|--user)
            USER_ARG="$2"
            shift 2
            ;;
        --user=*)
            USER_ARG="${1#*=}"
            shift
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        -c|--config-file)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config-file=*)
            CONFIG_PATH="${1#*=}"
            shift
            ;;
        -a|--asound-file)
            ASOUND_PATH="$2"
            shift 2
            ;;
        --asound-file=*)
            ASOUND_PATH="${1#*=}"
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

# Detect target user
TARGET_USER="${USER_ARG:-${SUDO_USER:-$USER}}"

# Detect config.txt location if not explicitly provided
if [[ -z "$CONFIG_PATH" ]]; then
    if [[ -f "/boot/firmware/config.txt" ]]; then
        CONFIG_PATH="/boot/firmware/config.txt"
    elif [[ -f "/boot/config.txt" ]]; then
        CONFIG_PATH="/boot/config.txt"
    else
        CONFIG_PATH="/boot/firmware/config.txt"
    fi
fi

PROFILE_UPPER=$(echo "$PROFILE" | tr '[:lower:]' '[:upper:]')
echo "======================================================================"
echo " AlsaGeist Raspberry Pi OS Configuration Tool v${VERSION}"
echo " Profile: ${PROFILE_UPPER} | Target user: ${TARGET_USER}"
echo " Target config: ${CONFIG_PATH}"
if [[ "$DRY_RUN" == true ]]; then
    echo " Mode: DRY RUN (no modifications will be made)"
fi
echo "======================================================================"

# 1. User Group Permissions
echo ""
echo "[1/3] User Group Configuration for user '${TARGET_USER}'..."

for grp in audio gpio; do
    if getent group "$grp" > /dev/null 2>&1; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [DRY RUN] Would add user '${TARGET_USER}' to group '${grp}'"
        else
            if id -nG "$TARGET_USER" 2>/dev/null | grep -qw "$grp"; then
                echo "  User '${TARGET_USER}' is already a member of '${grp}'."
            else
                if [[ $EUID -eq 0 ]]; then
                    usermod -aG "$grp" "$TARGET_USER"
                    echo "  Added user '${TARGET_USER}' to group '${grp}'."
                elif sudo -n true 2>/dev/null; then
                    sudo usermod -aG "$grp" "$TARGET_USER"
                    echo "  Added user '${TARGET_USER}' to group '${grp}'."
                else
                    echo "  Notice: Sudo password required to modify user groups. Run with sudo to update '${grp}' membership."
                fi
            fi
        fi
    else
        echo "  Notice: Group '${grp}' does not exist on this host system (skipping)."
    fi
done

# Portable file write helpers
safe_write() {
    local content="$1"
    local dest="$2"
    if [[ -w "$dest" ]] || { [[ ! -e "$dest" ]] && [[ -w "$(dirname "$dest")" ]]; }; then
        printf "%s\n" "$content" > "$dest"
    elif [[ $EUID -eq 0 ]]; then
        printf "%s\n" "$content" > "$dest"
    else
        printf "%s\n" "$content" | sudo tee "$dest" > /dev/null
    fi
}

safe_append() {
    local content="$1"
    local dest="$2"
    if [[ -w "$dest" ]]; then
        printf "%s\n" "$content" >> "$dest"
    elif [[ $EUID -eq 0 ]]; then
        printf "%s\n" "$content" >> "$dest"
    else
        printf "%s\n" "$content" | sudo tee -a "$dest" > /dev/null
    fi
}

# 2. Kernel & Audio Driver Overlays in config.txt
echo ""
echo "[2/3] Kernel & Device Tree Configuration (${CONFIG_PATH})..."

if [[ "$PROFILE" == "aiy" ]]; then
    AIY_CONFIG_BLOCK="# --- AlsaGeist: Google AIY Voice HAT V1 Audio Configuration ---
dtparam=audio=off
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
# -------------------------------------------------------------"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Target file: ${CONFIG_PATH}"
        echo "  [DRY RUN] Would ensure dtparam=audio=off, dtparam=i2s=on, and dtoverlay=googlevoicehat-soundcard exist."
    else
        if [[ ! -f "$CONFIG_PATH" ]]; then
            echo "  Notice: Target file '${CONFIG_PATH}' not found. Initializing target file."
            safe_write "" "$CONFIG_PATH"
        fi

        # Disable default Broadcom audio if set to on
        if grep -q "^dtparam=audio=on" "$CONFIG_PATH" 2>/dev/null; then
            echo "  Disabling onboard Broadcom audio (dtparam=audio=on -> off)..."
            TMP_FILE="${CONFIG_PATH}.tmp.$$"
            while IFS= read -r line || [[ -n "$line" ]]; do
                if [[ "$line" =~ ^dtparam=audio=on ]]; then
                    echo "dtparam=audio=off"
                else
                    echo "$line"
                fi
            done < "$CONFIG_PATH" > "$TMP_FILE"
            if [[ -w "$CONFIG_PATH" ]]; then
                mv "$TMP_FILE" "$CONFIG_PATH"
            elif [[ $EUID -eq 0 ]]; then
                mv "$TMP_FILE" "$CONFIG_PATH"
            else
                sudo mv "$TMP_FILE" "$CONFIG_PATH"
            fi
        fi

        # Append overlay configuration if not already present
        if grep -q "googlevoicehat-soundcard" "$CONFIG_PATH" 2>/dev/null; then
            echo "  Voice HAT overlay 'googlevoicehat-soundcard' is already present in ${CONFIG_PATH}."
        else
            echo "  Appending Voice HAT overlay to ${CONFIG_PATH}..."
            safe_append "$AIY_CONFIG_BLOCK" "$CONFIG_PATH"
            echo "  Voice HAT configuration appended."
        fi
    fi
else
    echo "  USB Profile selected: No device tree overlays required in ${CONFIG_PATH}."
fi

# 3. ALSA Routing Configuration (/etc/asound.conf)
echo ""
echo "[3/3] ALSA Configuration (${ASOUND_PATH})..."

if [[ "$PROFILE" == "aiy" ]]; then
    ASOUND_CONTENT='# AlsaGeist Google AIY Voice HAT Full-Duplex ALSA Configuration
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:sndrpigooglevoi,0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:sndrpigooglevoi,0"
    }
}

ctl.!default {
    type hw
    card sndrpigooglevoi
}'

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would write full-duplex ALSA routing configuration to ${ASOUND_PATH}:"
        echo "$ASOUND_CONTENT"
    else
        echo "  Writing ALSA full-duplex configuration to ${ASOUND_PATH}..."
        safe_write "$ASOUND_CONTENT" "$ASOUND_PATH"
        echo "  ALSA configuration written successfully."
    fi
else
    echo "  USB Profile selected: Standard ALSA defaults used (no custom asound.conf required)."
fi

echo ""
echo "======================================================================"
if [[ "$DRY_RUN" == true ]]; then
    echo " Dry-run complete. No files were modified."
else
    echo " OS configuration complete!"
    if [[ "$PROFILE" == "aiy" ]]; then
        echo " A system reboot is required to load the Google Voice HAT device tree overlay:"
        echo "   sudo reboot"
    fi
fi
echo "======================================================================"
exit 0
