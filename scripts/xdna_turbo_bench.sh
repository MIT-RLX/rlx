#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# xdna_turbo_bench.sh — measure AMD XDNA NPU int8 GEMM throughput at DEFAULT DPM vs
# TURBO (max clocks). TURBO is a root-only ioctl (SET_STATE / CAP_SYS_ADMIN), so the
# second run uses `sudo` — you'll be prompted for your password ONCE. No passwordless
# sudo required. Paste the WHOLE output back for validation of the TURBO uplift.
#
#   bash ~/rlx/scripts/xdna_turbo_bench.sh
#
# Optional overrides:  DIM=64 KT=8 COLS=4 ITERS=200 bash ~/rlx/scripts/xdna_turbo_bench.sh
set -uo pipefail

# Repo + toolchain (absolute paths, so they survive the sudo env reset).
REPO="$HOME/rlx"
cd "$REPO" || { echo "no $REPO"; exit 1; }
# shellcheck disable=SC1090
source "$HOME/.cargo/env" 2>/dev/null || true
CARGO="$(command -v cargo || echo "$HOME/.cargo/bin/cargo")"

export XILINX_XRT="$HOME/xrt-root/usr"
export LD_LIBRARY_PATH="$HOME/xrt-root/usr/lib/x86_64-linux-gnu"
export RLX_XDNA_SHIM="$HOME/librlx_xdna_shim.so"
export AIECC="$HOME/mlir-aie/ironenv/bin/aiecc"
export PEANO="$HOME/mlir-aie/ironenv/lib/python3.14/site-packages/llvm-aie"
export RLX_XDNA_AIE_INCLUDE="$HOME/mlir-aie/ironenv/lib/python3.14/site-packages/mlir_aie/include"
export PATH="$HOME/mlir-aie/ironenv/bin:$PATH"
export DIM="${DIM:-64}" KT="${KT:-8}" COLS="${COLS:-4}" ITERS="${ITERS:-200}"

echo "=================================================================="
echo " XDNA NPU int8 GEMM — DEFAULT DPM vs TURBO   (DIM=$DIM KT=$KT COLS=$COLS ITERS=$ITERS)"
echo "=================================================================="

echo
echo "[build] TURBO-capable microkernel benchmark (--features xrt,direct) ..."
"$CARGO" build --release -q -p rlx-xdna --features xrt,direct --example xdna_matmul_microkernel \
  || { echo "BUILD FAILED"; exit 1; }
BIN="$REPO/target/release/examples/xdna_matmul_microkernel"
[ -x "$BIN" ] || { echo "missing $BIN"; exit 1; }

echo
echo "------------------------------------------------------------------"
echo " [1/2] DEFAULT DPM (no TURBO, normal user)"
echo "------------------------------------------------------------------"
"$BIN"

echo
echo "------------------------------------------------------------------"
echo " [2/2] TURBO  (sudo — enter your password if prompted)"
echo "------------------------------------------------------------------"
# sudo scrubs LD_LIBRARY_PATH + most env, so pass everything the binary needs via env.
sudo env \
  XILINX_XRT="$XILINX_XRT" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  RLX_XDNA_SHIM="$RLX_XDNA_SHIM" \
  AIECC="$AIECC" PEANO="$PEANO" RLX_XDNA_AIE_INCLUDE="$RLX_XDNA_AIE_INCLUDE" \
  PATH="$PATH" \
  DIM="$DIM" KT="$KT" COLS="$COLS" ITERS="$ITERS" \
  RLX_XDNA_TURBO=1 \
  "$BIN" \
  || echo "(TURBO run failed — if it says Permission denied, sudo didn't grant CAP_SYS_ADMIN)"

echo
echo "=================================================================="
echo " DONE. Paste everything above back to validate the TURBO uplift."
echo " Expect: run [2/2] prints '[turbo] NPU power mode -> TURBO' and a"
echo " higher GOP/s than run [1/2]. If [2/2] shows the permission warning,"
echo " sudo lacked CAP_SYS_ADMIN and both numbers will match."
echo "=================================================================="
