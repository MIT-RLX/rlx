#!/usr/bin/env bash
# Refuse to run benchmarks on a thermally throttled Apple Silicon machine.
#
# Borrowed from MAX's `utils/check-gpu-throttle.sh` pattern (#48 in plan.md).
# Silent thermal throttling can produce 5–10× slower numbers; CI / local
# bench runs that don't gate on this drift over time without anyone noticing.
#
# Usage:
#   scripts/check-throttle.sh           # exit 0 if cool, 1 if throttling
#   scripts/check-throttle.sh --warn    # warn only, never exit non-zero
#   scripts/check-throttle.sh --json    # emit a one-line JSON report
#
# Honors RLX_ALLOW_THROTTLE=1 to bypass for one-off runs.

set -u

mode="strict"
case "${1:-}" in
  --warn) mode="warn" ;;
  --json) mode="json" ;;
  "") ;;
  *) echo "usage: $0 [--warn|--json]" >&2; exit 2 ;;
esac

# `pmset -g therm` works without sudo and reports CPU thermal level.
# 0 = nominal, >0 = throttled (1 = light, 5 = heavy).
therm_raw="$(pmset -g therm 2>/dev/null || true)"
cpu_speed="$(echo "$therm_raw" | awk -F= '/CPU_Speed_Limit/ {gsub(/ /,"",$2); print $2}')"
cpu_speed="${cpu_speed:-100}"

# Sched limit (1.0 nominal, <1.0 = thermal pressure scheduling).
sched="$(echo "$therm_raw" | awk -F= '/CPU_Scheduler_Limit/ {gsub(/ /,"",$2); print $2}')"
sched="${sched:-100}"

# Load average — sustained high load skews "are we hot?" answers.
load1="$(uptime | awk -F'load averages?:' '{print $2}' | awk '{print $1}' | tr -d ',')"

throttled=0
if [ "${cpu_speed}" -lt 100 ] 2>/dev/null; then throttled=1; fi
if [ "${sched}" -lt 100 ] 2>/dev/null; then throttled=1; fi

if [ "$mode" = "json" ]; then
  printf '{"throttled":%s,"cpu_speed":%s,"sched":%s,"load1":"%s"}\n' \
    "$throttled" "$cpu_speed" "$sched" "$load1"
  exit 0
fi

if [ "$throttled" -eq 0 ]; then
  echo "[throttle] OK — cpu_speed=${cpu_speed}% sched=${sched}% load1=${load1}"
  exit 0
fi

echo "[throttle] WARN — thermal throttling detected: cpu_speed=${cpu_speed}% sched=${sched}% load1=${load1}" >&2
echo "[throttle]        bench numbers will be unreliable; let the machine cool first." >&2

if [ "$mode" = "warn" ] || [ "${RLX_ALLOW_THROTTLE:-0}" = "1" ]; then
  exit 0
fi
exit 1
