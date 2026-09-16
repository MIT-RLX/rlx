#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Wait for a shared GPU rig to be genuinely free, then run a benchmark command.
#
# Why this exists rather than "just run the benchmark": the rigs are shared, and
# a contended timing run does not fail — it produces numbers. This project has
# been burned twice.
#
#   * A CUDA tuning run on a 100%-busy GPU came out ~25x slow and picked a
#     DIFFERENT winner, which would have been written into a persisted cache.
#   * A ROCm sweep passed a utilization-only pre-check and then hit a job that
#     started mid-run; its control arm (a kernel byte-identical to the baseline)
#     read between 0.79x and 1.25x, i.e. a +/-25% noise floor that no ratio in
#     the table could be read against.
#
# So the gate is TWO conditions, not one:
#
#   * utilization below the threshold for N consecutive samples, AND
#   * zero compute processes holding the device.
#
# The second is the one that matters. A training job in a data-loading or eval
# phase reports 0% for minutes at a stretch while still owning the GPU — on the
# CUDA rig, utilization was observed oscillating 100/0/100/0 — so a
# utilization-only gate fires during the lull and is contended seconds later.
#
# Usage:
#   scripts/rig-gpu-idle-watch.sh --vendor nvidia --log ~/ab.log -- ./bench --args
#   scripts/rig-gpu-idle-watch.sh --vendor amd    --log ~/ab.log -- ./bench --args
#
# Options:
#   --vendor nvidia|amd   which SMI to poll (default: autodetect)
#   --log PATH            progress + command output (default: ./gpu-idle-watch.log)
#   --max-util N          utilization ceiling, percent (default: 15)
#   --samples N           consecutive clean samples required (default: 5)
#   --interval N          seconds between samples (default: 30)
#   --settle N            seconds to let clocks settle once free (default: 60)
#   --timeout N           give up after N seconds (default: 28800 = 8h)
#
# IMPORTANT: keep the log OUTSIDE any directory you rsync to the rig. The repo
# sync uses `rsync --delete`, which silently removes anything in the tree that
# is not in the local checkout — including a running watcher's log and script.
set -uo pipefail

VENDOR=""; LOG="./gpu-idle-watch.log"; MAX_UTIL=15; SAMPLES=5; INTERVAL=30
SETTLE=60; TIMEOUT=28800
while [ $# -gt 0 ]; do
  case "$1" in
    --vendor)   VENDOR="$2"; shift 2 ;;
    --log)      LOG="$2"; shift 2 ;;
    --max-util) MAX_UTIL="$2"; shift 2 ;;
    --samples)  SAMPLES="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --settle)   SETTLE="$2"; shift 2 ;;
    --timeout)  TIMEOUT="$2"; shift 2 ;;
    --)         shift; break ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "no command given after --" >&2
  exit 2
fi

if [ -z "$VENDOR" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then VENDOR=nvidia
  elif command -v rocm-smi >/dev/null 2>&1; then VENDOR=amd
  else echo "no nvidia-smi or rocm-smi found" >&2; exit 2
  fi
fi

gpu_util() {
  case "$VENDOR" in
    nvidia) nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ' ;;
    amd)    rocm-smi --showuse 2>/dev/null | awk '/GPU\[0\]/ {print $NF; exit}' ;;
  esac
}
# Count of compute processes holding the device. This is the condition a
# utilization-only gate misses.
gpu_procs() {
  case "$VENDOR" in
    nvidia) nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true ;;
    amd)    rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true ;;
  esac
}

mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
: > "$LOG"
echo "watch: vendor=$VENDOR need $SAMPLES consecutive samples <${MAX_UTIL}% util AND 0 compute procs" >> "$LOG"

DEADLINE=$(( $(date +%s) + TIMEOUT ))
CLEAN=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  U="$(gpu_util)"; P="$(gpu_procs)"
  if [ -n "$U" ] && [ "$U" -lt "$MAX_UTIL" ] 2>/dev/null && [ "${P:-0}" -eq 0 ]; then
    CLEAN=$((CLEAN+1))
  else
    CLEAN=0
  fi
  if [ "$CLEAN" -ge "$SAMPLES" ]; then
    echo "GPU free at $(date -Is) — settling ${SETTLE}s" >> "$LOG"
    sleep "$SETTLE"
    "$@" >> "$LOG" 2>&1
    RC=$?
    # Re-check AFTER the run. A job that started mid-sweep is exactly how the
    # ROCm numbers were corrupted, and the pre-check cannot see it.
    echo "--- rc=$RC ---" >> "$LOG"
    echo "post-run util=$(gpu_util)% procs=$(gpu_procs) (nonzero here means the run was contended)" >> "$LOG"
    echo "SWEEP_DONE" >> "$LOG"
    exit "$RC"
  fi
  sleep "$INTERVAL"
done
echo "SWEEP_TIMEOUT: GPU never went free within ${TIMEOUT}s" >> "$LOG"
exit 1
