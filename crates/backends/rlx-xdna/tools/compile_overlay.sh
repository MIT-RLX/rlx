#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Compile an XDNA NPU overlay (aie.mlir -> .xclbin + insts.bin) with NO Python.
#
# The MLIR-AIE compiler is a native ELF binary (mlir_aie/bin/aiecc, no libpython);
# aiecc.py is only a shim. This drives the native binary directly with Peano (no
# Vitis/Chess). Verified to run with python3 blocked.
#
# Usage:
#   MLIR_AIE_INSTALL_DIR=<.../mlir_aie> PEANO_INSTALL_DIR=<.../llvm-aie> \
#   ./compile_overlay.sh <aie.mlir> <out.xclbin> <out_insts.bin> [tmpdir]
set -euo pipefail

MLIR="${1:?aie.mlir}"; XCLBIN="${2:?out.xclbin}"; INSTS="${3:?out_insts.bin}"
TMP="${4:-$(dirname "$XCLBIN")/ovbuild}"
MB="${MLIR_AIE_INSTALL_DIR:?set MLIR_AIE_INSTALL_DIR (.../mlir_aie)}"
PB="${PEANO_INSTALL_DIR:?set PEANO_INSTALL_DIR (.../llvm-aie)}"

AIECC="$MB/bin/aiecc"   # the NATIVE binary, not aiecc.py
[ -x "$AIECC" ] || { echo "native aiecc not found at $AIECC" >&2; exit 1; }
mkdir -p "$TMP"

"$AIECC" --no-xchesscc \
  --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
  --tmpdir="$TMP" --peano="$PB" \
  --xclbin-name="$XCLBIN" --npu-insts-name="$INSTS" \
  "$MLIR"

echo "overlay: $XCLBIN ($(stat -c %s "$XCLBIN") B) + $INSTS ($(stat -c %s "$INSTS") B)"
