#!/usr/bin/env bash
# scripts/verify_docker_deploy.sh - Post-deployment checkout for the Docker install on a Raspberry Pi.
#
# Run on the Pi, from the repository root, after `docker compose up -d`. It is read-only
# and never powers the host off: shutdown is checked by asking systemd-logind whether the
# container is authorized to power off (CanPowerOff), not by doing it.
#
# Usage:
#   ./scripts/verify_docker_deploy.sh            # run all checks
#   ./scripts/verify_docker_deploy.sh --shutdown # also power the Pi off through the control socket (asks first)
#
# Exit status: 0 if every check passed (warnings allowed), 1 if any check failed.

set -u

CONTAINER="${ALSAGEIST_CONTAINER:-alsageist}"
DO_SHUTDOWN=false
[[ "${1:-}" == "--shutdown" ]] && DO_SHUTDOWN=true

PASS=0; FAIL=0; WARN=0
pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; FAIL=$((FAIL + 1)); }
warn() { echo "  [WARN] $1"; [[ -n "${2:-}" ]] && echo "         -> $2"; WARN=$((WARN + 1)); }
dexec() { docker exec "$CONTAINER" "$@" 2>&1; }

echo "AlsaGeist Docker deployment checkout (container: ${CONTAINER})"

echo ""
echo "[1] Container"
if ! docker inspect "$CONTAINER" > /dev/null 2>&1; then
    fail "Container '${CONTAINER}' does not exist" "Run: docker compose up -d (use sudo if your user is not in the docker group)"
    echo ""; echo "Result: ${FAIL} failed. Remaining checks need a container."; exit 1
fi
STATE="$(docker inspect -f '{{.State.Status}}' "$CONTAINER")"
[[ "$STATE" == "running" ]] && pass "Container is running" || fail "Container state is '${STATE}'" "docker compose logs --tail 50"
RESTARTS="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER")"
[[ "$RESTARTS" -eq 0 ]] && pass "No restarts" || warn "Container has restarted ${RESTARTS} time(s)" "docker compose logs --tail 100"
HEALTH="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$CONTAINER")"
case "$HEALTH" in
    healthy) pass "Healthcheck is healthy" ;;
    starting) warn "Healthcheck is still starting (up to 2 minutes after start)" ;;
    *) fail "Healthcheck is '${HEALTH}'" "docker inspect --format '{{json .State.Health}}' ${CONTAINER}" ;;
esac

echo ""
echo "[2] Daemon control socket"
STATUS="$(dexec python3 scripts/alsageist_ctl.py status --json)"
if echo "$STATUS" | grep -q '"status": "ok"'; then
    pass "Daemon answers status queries ($(echo "$STATUS" | grep -E '"(state|mode|personality)"' | tr -d ' \n'))"
else
    fail "Daemon did not answer on its control socket" "$(echo "$STATUS" | tail -1)"
fi

echo ""
echo "[3] Hardware access from inside the container"
dexec test -e /dev/gpiochip0 && pass "GPIO device /dev/gpiochip0 present" || fail "GPIO device /dev/gpiochip0 missing" "Set GPIO_DEVICE in .env (Pi 5 uses /dev/gpiochip4)"
dexec sh -c 'ls /dev/snd/pcmC*D*p > /dev/null 2>&1' && pass "Audio playback device present" || fail "No audio playback device in /dev/snd" "Check the sound overlay and /etc/asound.conf"
dexec sh -c 'ls /dev/snd/pcmC*D*c > /dev/null 2>&1' && pass "Audio capture device present" || fail "No audio capture device in /dev/snd" "Check the microphone and /etc/asound.conf"
if dexec sh -c 'for d in /dev/gpiochip0 /dev/snd/controlC0; do [ -r "$d" ] && [ -w "$d" ] || exit 1; done'; then
    pass "Container user can open GPIO and sound devices"
else
    fail "Container user cannot open GPIO or sound devices" "Check AUDIO_GID and GPIO_GID in .env against: getent group audio gpio"
fi

echo ""
echo "[4] Host shutdown (voice command and 10 s button hold)"
if [[ "$(docker inspect -f '{{.Config.User}}' "$CONTAINER")" == "0:0" ]]; then
    pass "Container runs as root (needed to be authorized for PowerOff)"
else
    fail "Container does not run as root" "Add COMPOSE_FILE=docker-compose.yml:docker-compose.shutdown.yml to .env, then: docker compose up -d"
fi
if docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' "$CONTAINER" | grep -q '/var/run/dbus/system_bus_socket'; then
    pass "Host D-Bus socket is mounted"
else
    fail "Host D-Bus socket is not mounted" "Add COMPOSE_FILE=docker-compose.yml:docker-compose.shutdown.yml to .env, then: docker compose up -d"
fi
CAN="$(dexec dbus-send --system --print-reply --dest=org.freedesktop.login1 /org/freedesktop/login1 org.freedesktop.login1.Manager.CanPowerOff)"
if echo "$CAN" | grep -q 'string "yes"'; then
    pass "systemd-logind authorizes PowerOff for the container"
else
    fail "systemd-logind does not authorize PowerOff (CanPowerOff: $(echo "$CAN" | grep -o 'string "[a-z-]*"' || echo "$CAN" | tail -1))" "Needs root plus the D-Bus socket; check polkit on the host"
fi
if grep -qE '^BUTTON_SHUTDOWN_ENABLED=false' .env 2>/dev/null; then
    warn "BUTTON_SHUTDOWN_ENABLED=false: the 10 s hold mutes instead of powering off"
fi

echo ""
echo "[5] Host configuration"
for var in AUDIO_GID GPIO_GID GPIO_DEVICE; do
    grep -q "^${var}=" .env 2>/dev/null && pass "${var} recorded in .env" || warn "${var} not in .env" "Compose falls back to a default that may not match this Pi"
done
if systemctl is-enabled alsageist.service > /dev/null 2>&1; then
    pass "alsageist.service is enabled for autostart"
else
    warn "alsageist.service is not enabled" "Re-run scripts/setup_docker.sh or: sudo systemctl enable alsageist.service"
fi
if docker compose config 2>/dev/null | grep -q 'system_bus_socket'; then
    pass "docker compose (and so the systemd unit) includes the shutdown override after a reboot"
else
    fail "docker compose config omits the shutdown override" "After a reboot the container would lose shutdown support. Set COMPOSE_FILE in .env"
fi

echo ""
echo "[6] Models and logs"
dexec sh -c 'ls /app/models/piper/*.onnx > /dev/null 2>&1' && pass "Piper voice model present" || fail "No Piper voice model in /app/models/piper" "Check the entrypoint download in: docker compose logs"
dexec test -d /app/models/vosk-model-small-en-us-0.15 && pass "Offline Vosk model present" || warn "Offline Vosk model missing" "Offline speech recognition is unavailable"
dexec sh -c 'touch /app/logs/.verify && rm /app/logs/.verify' && pass "Logs directory writable" || fail "Logs directory not writable" "sudo chown -R 1000:1000 logs data models"
if docker logs --tail 200 "$CONTAINER" 2>&1 | grep -qiE 'Traceback|Shutdown failed|ERROR'; then
    warn "Recent container logs contain errors" "docker compose logs --tail 200 | grep -iE 'traceback|error|failed'"
else
    pass "No errors in recent container logs"
fi

if [[ "$DO_SHUTDOWN" == true ]]; then
    echo ""
    echo "[7] Live shutdown test"
    read -r -p "  This POWERS OFF the Pi now. Type 'yes' to continue: " ANSWER
    if [[ "$ANSWER" == "yes" ]]; then
        dexec python3 scripts/alsageist_ctl.py shutdown --yes
        sleep 20
        echo "  If you can read this the Pi did not power off. Check: docker compose logs --tail 30"
    fi
fi

echo ""
echo "Result: ${PASS} passed, ${WARN} warning(s), ${FAIL} failed."
[[ "$FAIL" -eq 0 ]]
