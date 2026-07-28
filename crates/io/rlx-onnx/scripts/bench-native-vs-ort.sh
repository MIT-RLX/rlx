#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Compare native RLX (rlx-onnx --exec native) vs ONNX Runtime (--exec ort)
# on a small MatMul+Relu ONNX and optionally KittenTTS (ORT end-to-end via kittentts-rs).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/target/release/rlx-onnx-run}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-15}"
LEVEL="${LEVEL:-3}"

FEATS="ort-fallback,metal,mlx,cuda,rocm"
cd "$ROOT"

if [[ ! -x "$BIN" ]]; then
  echo ">> building rlx-onnx-run ($FEATS)"
  cargo build -p rlx-onnx --release --features "$FEATS"
fi

MICRO="${MICRO_ONNX:-/tmp/bench_matmul_relu.onnx}"
if [[ ! -f "$MICRO" ]]; then
  echo ">> generating $MICRO (opset 11, IR 8)"
  if [[ ! -x /tmp/bench-venv/bin/python3 ]]; then
    python3 -m venv /tmp/bench-venv
    /tmp/bench-venv/bin/pip install -q onnx torch
  fi
  /tmp/bench-venv/bin/python3 "$ROOT/rlx-onnx/scripts/export_bench_matmul_relu.py" "$MICRO"
fi

DEVICES=(cpu metal mlx cuda rocm gpu)
echo ""
echo "=== rlx-onnx micro graph: MatMul+Add+Relu [1,32,512] (warmup=$WARMUP iters=$ITERS level=$LEVEL) ==="
printf "%-8s %-8s %-10s %-12s %s\n" "device" "backend" "status" "ms/iter" "notes"
printf "%s\n" "----------------------------------------------------------------"

for dev in "${DEVICES[@]}"; do
  for exec in ort native; do
  line="$dev $exec"
  if ! out=$("$BIN" "$MICRO" --exec "$exec" --device "$dev" --level "$LEVEL" \
      --warmup "$WARMUP" --iters "$ITERS" 2>&1); then
    note=$(echo "$out" | tail -1 | tr '\n' ' ')
    printf "%-8s %-8s %-10s %-12s %s\n" "$dev" "$exec" "FAIL" "-" "${note:0:60}"
    continue
  fi
  ms=$(echo "$out" | sed -n 's/.*(\([0-9.]*\) ms\/iter).*/\1/p')
  ep=$(echo "$out" | sed -n 's/.*ort_ep=\([^,]*\).*/\1/p')
  note="${ep:-native}"
  printf "%-8s %-8s %-10s %-12s %s\n" "$dev" "$exec" "ok" "$ms" "$note"
  done
done

if [[ -n "${KITTENTTS_MODEL_DIR:-}" && -f "${KITTENTTS_MODEL_DIR}/kitten_tts_mini_v0_8.onnx" ]]; then
  echo ""
  echo "=== KittenTTS end-to-end (ORT via rlx-kittentts; native ONNX import not ready) ==="
  echo "Run: cd ../kittentts-rs && ./scripts/bench-all-backends.sh"
fi
