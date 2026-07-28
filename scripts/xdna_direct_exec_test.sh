#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# xdna_direct_exec_test.sh — test the POWER-STATE hypothesis for the direct-path
# EXEC_CMD hang. The NPU autosuspends after 5s (runtime_status=suspended at idle);
# the direct (no-XRT) GEMM submits but the firmware never picks it up (ert state=NEW,
# fence times out). This keeps the array RESUMED (disable runtime-PM autosuspend via
# power/control=on) + holds TURBO, then runs the direct EXEC_CMD 512³ GEMM. If it now
# COMPLETES, Level 1 works and the hang was power-state; if it still hangs, the wall
# is firmware-internal (needs Secure Boot off to diagnose).
#
# Needs root for the sysfs write + TURBO ioctl → the core runs under one `sudo`
# (you'll be prompted for your password once). Build happens first as your user.
# The direct path is XRT-free, so no XILINX_XRT/shim env is needed.
#
#   bash ~/rlx/scripts/xdna_direct_exec_test.sh
set -uo pipefail
REPO="$HOME/rlx"; cd "$REPO" || { echo "no $REPO"; exit 1; }
# shellcheck disable=SC1090
source "$HOME/.cargo/env" 2>/dev/null || true
CARGO="$(command -v cargo || echo "$HOME/.cargo/bin/cargo")"

XCLBIN="$HOME/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x512x512_32x32x32_4c.xclbin"
INSTS="$HOME/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/insts_512x512x512_32x32x32_4c.bin"
DEV="$(readlink -f /sys/class/accel/accel0/device)"
[ -f "$XCLBIN" ] || { echo "missing overlay: $XCLBIN"; exit 1; }
[ -n "$DEV" ] || { echo "no accel0 device"; exit 1; }

echo "=================================================================="
echo " DIRECT EXEC_CMD GEMM — power-state test (keep NPU resumed + TURBO)"
echo "=================================================================="
echo "[build] direct_gemm (--features direct) as $USER ..."
"$CARGO" build --release -q -p rlx-xdna --features direct --example direct_gemm \
  || { echo "BUILD FAILED"; exit 1; }
BIN="$REPO/target/release/examples/direct_gemm"
[ -x "$BIN" ] || { echo "missing $BIN"; exit 1; }

echo "[before] control=$(cat "$DEV/power/control" 2>/dev/null) status=$(cat "$DEV/power/runtime_status" 2>/dev/null)"
echo
echo "== root section (sudo — enter your password once) =="
sudo env DEV="$DEV" BIN="$BIN" XCLBIN="$XCLBIN" INSTS="$INSTS" bash -c '
  set -u
  echo "[root] disabling runtime-PM autosuspend: power/control=on"
  echo on > "$DEV/power/control"
  sleep 1
  echo "[root] NPU now: control=$(cat "$DEV/power/control") status=$(cat "$DEV/power/runtime_status")"
  echo "[root] running DIRECT EXEC_CMD 512x512x512 GEMM with TURBO (timeout 30s) ..."
  M=512 K=512 N=512 ITERS=1 RLX_XDNA_TURBO=1 timeout 30 "$BIN" 2>&1 \
    | grep -iE "turbo|axlf|complete|fence|ert state|submit=|GOP|PASS|FAIL|mism|error|panic" | head -16
  echo "[root] direct_gemm exit=${PIPESTATUS[0]}"
  echo "[root] restoring autosuspend: power/control=auto"
  echo auto > "$DEV/power/control"
'
echo
echo "=================================================================="
echo " DONE. Paste everything above."
echo " If the run shows PASS / a GOP/s number / ert state=COMPLETED, the"
echo " direct EXEC_CMD path WORKS once the NPU is held resumed (Level 1)."
echo " If it still shows 'fence timed out / ert state=NEW / submit=1"
echo " complete=0', the hang is firmware-internal (not power) and needs"
echo " Secure Boot OFF to diagnose."
echo "=================================================================="
