#!/usr/bin/env python3
"""Side-by-side compare two `model_weights.{rs,bin}` exports.

Reports:
  - per-array shape + L1/Linf differences (after the i8 pass already
    chose per-channel scales, structural shape *must* match between
    trainers; weight values won't, since SGD-momentum and Adam follow
    different optima)
  - per-scale relative diff (per-channel scales are reduced via
    elementwise relative diff → max)
  - INT8 forward parity on the embedded TEST_IMAGE — runs the model
    twice and reports argmax + logits for each
  - INT8 forward sweep on the test_set.bin (500 images by default) for
    each set of weights, producing accuracy numbers comparable to what
    the on-device firmware reports

The reference `forward()` here implements the same INT8 arithmetic the
firmware kernels (`rlx-cortexm/src/{conv2d,dense,maxpool,relu,argmax}.rs`)
do — i32 accumulation, **per-channel** requantize, `[O, kH, kW, I]`
weight layout. As long as host numpy ↔ host Rust ↔ on-device M4F all
agree to the bit, the kernels are deterministic and platform-
independent. This script is the host half of that contract;
`mnist_validate.rs` and `mnist_client.py` cover the other two paths.

# Loading

The script accepts either:
- a `model_weights.rs` path (it reads the sibling `model_weights.bin`
  in the same directory) — this is the canonical path produced by
  `cargo run -p rlx-cortexm-trainer`
- a `model_weights.bin` path directly (RLXM v1 format)
- a legacy per-tensor `model_weights.rs` (one big literal array per
  tensor, no .bin) — the parser falls back to literal scraping.

Usage:
    python3 rlx-cortexm/tools/compare_weights.py \\
        rlx-cortexm/src/model_weights.rs \\
        /tmp/torch_run/src/model_weights.rs \\
        --test-set rlx-cortexm/tests/data/test_set.bin
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ─────────────────────────── parsing ────────────────────────────

INT_ARR  = re.compile(r"pub static (\w+):\s*\[(i8|i32);\s*(\d+)\]\s*=\s*\[([^\]]+)\];", re.S)
SCALE    = re.compile(r"pub const (\w+):\s*f32\s*=\s*([0-9eE+\-.]+);")
TEST_LBL = re.compile(r"pub const TEST_LABEL:\s*u8\s*=\s*(\d+);")

# RLXM v1 binary format
RLXM_MAGIC = b"RLXM"
HEADER_LEN = 16
DESC_LEN = 16
DTYPE_I8, DTYPE_I32, DTYPE_F32, DTYPE_I4_PKD, DTYPE_I2_PKD = 0, 1, 2, 3, 4
WEIGHT_TENSOR_IDS = {0, 2, 4}  # CONV1_W, CONV2_W, FC_W — packed-weight slots
TENSOR_NAMES = {
    0: "CONV1_W", 1: "CONV1_B", 2: "CONV2_W", 3: "CONV2_B",
    4: "FC_W",    5: "FC_B",
    6: "CONV1_MULT", 7: "CONV2_MULT", 8: "FC_MULT",
    9: "TEST_IMAGE",
}


def _unpack_weights(raw: np.ndarray, n_logical: int, bits: int) -> np.ndarray:
    """Unpack a packed-weight byte buffer into a logical i8 array.

    Symmetric 2's-complement: i4 lanes are sign-extended from 4 bits,
    i2 lanes from 2 bits. For `bits=8` returns `raw` as-is (cast to i8).
    """
    if bits == 8:
        return raw.astype(np.int8)
    if bits == 4:
        u = raw.view(np.uint8)
        lo = (u & 0x0F).astype(np.int32)
        hi = ((u >> 4) & 0x0F).astype(np.int32)
        # sign-extend 4-bit
        lo = np.where(lo & 0x8, lo - 16, lo)
        hi = np.where(hi & 0x8, hi - 16, hi)
        out = np.empty(u.size * 2, dtype=np.int8)
        out[0::2] = lo.astype(np.int8)
        out[1::2] = hi.astype(np.int8)
        return out[:n_logical]
    if bits == 2:
        u = raw.view(np.uint8)
        lanes = []
        for shift in (0, 2, 4, 6):
            crumb = ((u >> shift) & 0x03).astype(np.int32)
            crumb = np.where(crumb & 0x2, crumb - 4, crumb)
            lanes.append(crumb)
        out = np.empty(u.size * 4, dtype=np.int8)
        out[0::4] = lanes[0].astype(np.int8)
        out[1::4] = lanes[1].astype(np.int8)
        out[2::4] = lanes[2].astype(np.int8)
        out[3::4] = lanes[3].astype(np.int8)
        return out[:n_logical]
    raise ValueError(f"unsupported weight bits {bits}")


@dataclass
class Weights:
    arrays: dict[str, np.ndarray]      # name → int8 / int32 / float32
    scales: dict[str, float]           # name → f32 (activation scales: X_SCALE, C1_SCALE, ...)
    test_label: int

    def __getitem__(self, k):
        return self.arrays[k]


def load(path: Path) -> Weights:
    """Accept either a .rs file (with a sibling .bin) or a .bin directly."""
    if path.suffix == ".bin":
        return _load_blob_only(path)
    src = path.read_text()
    sibling_bin = path.with_name("model_weights.bin")
    if sibling_bin.exists():
        # New format: .rs has scalar consts + slice references; the data
        # lives in the .bin. We pull arrays from the .bin and activation
        # scales from the .rs.
        return _load_with_blob(src, sibling_bin)
    # Legacy fallback: literals embedded in the .rs.
    return _load_legacy(src)


def _load_with_blob(rs_src: str, bin_path: Path) -> Weights:
    blob = bin_path.read_bytes()
    if blob[:4] != RLXM_MAGIC:
        raise ValueError(f"{bin_path}: bad magic {blob[:4]!r}")
    n_tensors = struct.unpack_from("<H", blob, 12)[0]
    arrays: dict[str, np.ndarray] = {}
    for i in range(n_tensors):
        base = HEADER_LEN + i * DESC_LEN
        tid, dt = struct.unpack_from("<HB", blob, base)
        off, n = struct.unpack_from("<II", blob, base + 4)
        name = TENSOR_NAMES.get(tid, f"TENSOR_{tid}")
        if dt == DTYPE_I8:
            arrays[name] = np.frombuffer(blob, dtype=np.int8, count=n, offset=off).copy()
        elif dt == DTYPE_I32:
            arrays[name] = np.frombuffer(blob, dtype=np.int32, count=n, offset=off).copy()
        elif dt == DTYPE_F32:
            arrays[name] = np.frombuffer(blob, dtype=np.float32, count=n, offset=off).copy()
        elif dt == DTYPE_I4_PKD:
            n_bytes = (n + 1) // 2
            raw = np.frombuffer(blob, dtype=np.uint8, count=n_bytes, offset=off).copy()
            arrays[name] = _unpack_weights(raw, n, 4)
        elif dt == DTYPE_I2_PKD:
            n_bytes = (n + 3) // 4
            raw = np.frombuffer(blob, dtype=np.uint8, count=n_bytes, offset=off).copy()
            arrays[name] = _unpack_weights(raw, n, 2)
        else:
            raise ValueError(f"unknown dtype {dt} for {name}")
    scales = {m.group(1): float(m.group(2)) for m in SCALE.finditer(rs_src)}
    # `WEIGHT_BITS` lives in the .rs as a `pub const` so host tools can
    # display the quant scheme without re-parsing descriptors.
    wb_m = re.search(r"pub const WEIGHT_BITS:\s*u8\s*=\s*(\d+);", rs_src)
    if wb_m:
        scales["_WEIGHT_BITS"] = float(wb_m.group(1))
    tl_m = TEST_LBL.search(rs_src)
    if tl_m is None:
        raise ValueError("no TEST_LABEL in .rs")
    return Weights(arrays=arrays, scales=scales, test_label=int(tl_m.group(1)))


def _load_blob_only(bin_path: Path) -> Weights:
    blob = bin_path.read_bytes()
    if blob[:4] != RLXM_MAGIC:
        raise ValueError(f"{bin_path}: bad magic {blob[:4]!r}")
    accuracy = struct.unpack_from("<f", blob, 8)[0]
    n_tensors = struct.unpack_from("<H", blob, 12)[0]
    arrays: dict[str, np.ndarray] = {}
    for i in range(n_tensors):
        base = HEADER_LEN + i * DESC_LEN
        tid, dt = struct.unpack_from("<HB", blob, base)
        off, n = struct.unpack_from("<II", blob, base + 4)
        name = TENSOR_NAMES.get(tid, f"TENSOR_{tid}")
        if dt in (DTYPE_I8, DTYPE_I32, DTYPE_F32):
            np_dt = {DTYPE_I8: np.int8, DTYPE_I32: np.int32, DTYPE_F32: np.float32}[dt]
            arrays[name] = np.frombuffer(blob, dtype=np_dt, count=n, offset=off).copy()
        elif dt == DTYPE_I4_PKD:
            raw = np.frombuffer(blob, dtype=np.uint8, count=(n + 1) // 2, offset=off).copy()
            arrays[name] = _unpack_weights(raw, n, 4)
        elif dt == DTYPE_I2_PKD:
            raw = np.frombuffer(blob, dtype=np.uint8, count=(n + 3) // 4, offset=off).copy()
            arrays[name] = _unpack_weights(raw, n, 2)
        else:
            raise ValueError(f"unknown dtype {dt} for {name}")
    return Weights(arrays=arrays, scales={"_FROM_BIN_ONLY_ACC": accuracy}, test_label=-1)


def _load_legacy(src: str) -> Weights:
    """Parse the older per-tensor literal-array format produced by
    pre-blob trainer runs and the legacy `tools/train_mnist.py`."""
    arrays: dict[str, np.ndarray] = {}
    for m in INT_ARR.finditer(src):
        name, tyname, n_str, body = m.group(1), m.group(2), m.group(3), m.group(4)
        n = int(n_str)
        vals = [int(t.strip()) for t in body.split(",") if t.strip()]
        if len(vals) != n:
            raise ValueError(f"{name}: declared {n} elements, found {len(vals)}")
        dtype = np.int8 if tyname == "i8" else np.int32
        arrays[name] = np.array(vals, dtype=dtype)
    scales = {m.group(1): float(m.group(2)) for m in SCALE.finditer(src)}
    # Legacy emits per-tensor W*_SCALE / WFC_SCALE — synthesize
    # per-channel CONV*_MULT / FC_MULT from those so the rest of the
    # script can use the unified path.
    for k_in, k_w, k_out, mult_name, n_oc in [
        ("X_SCALE",  "W1_SCALE",  "C1_SCALE",     "CONV1_MULT", 8),
        ("P1_SCALE", "W2_SCALE",  "C2_SCALE",     "CONV2_MULT", 16),
        ("P2_SCALE", "WFC_SCALE", "FC_OUT_SCALE", "FC_MULT",    10),
    ]:
        if all(k in scales for k in (k_in, k_w, k_out)):
            mult = (scales[k_in] * scales[k_w]) / scales[k_out]
            arrays[mult_name] = np.full((n_oc,), mult, dtype=np.float32)
    tl_m = TEST_LBL.search(src)
    if tl_m is None:
        raise ValueError("no TEST_LABEL")
    return Weights(arrays=arrays, scales=scales, test_label=int(tl_m.group(1)))


# ─────────────────── INT8 forward (numpy reference) ─────────────

def requantize_per_channel(acc: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """`acc` shape ends in C; `mult` shape `[C]`. Per-channel f32
    multiplier, symmetric (zp=0), saturating to i8."""
    out = np.rint(acc.astype(np.float64) * mult.astype(np.float64)).astype(np.int64)
    return np.clip(out, -127, 127).astype(np.int8)


def conv2d_i8(x: np.ndarray, w: np.ndarray, b: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """NHWC int8 valid 3×3 conv with i32 bias and per-channel mult.

    x: [H, W, C_in] int8     w: [C_out, kH, kW, C_in] int8
    bias: [C_out] int32      mult: [C_out] f32
    """
    H, W, _ = x.shape
    Cout, kH, kW, _ = w.shape
    Ho, Wo = H - kH + 1, W - kW + 1
    out = np.zeros((Ho, Wo, Cout), dtype=np.int64)
    x32 = x.astype(np.int64)
    w32 = w.astype(np.int64)
    for oc in range(Cout):
        for h in range(Ho):
            for w_ in range(Wo):
                s = b[oc]
                s += int((x32[h:h+kH, w_:w_+kW, :] * w32[oc]).sum())
                out[h, w_, oc] = s
    return requantize_per_channel(out, mult)


def maxpool_i8(x: np.ndarray) -> np.ndarray:
    """NHWC 2×2 stride-2 max-pool."""
    H, W, C = x.shape
    Ho, Wo = H // 2, W // 2
    out = np.zeros((Ho, Wo, C), dtype=np.int8)
    for h in range(Ho):
        for w_ in range(Wo):
            tile = x[h*2:h*2+2, w_*2:w_*2+2, :]
            out[h, w_, :] = tile.reshape(4, C).max(axis=0)
    return out


def relu_i8(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.int8)


def dense_i8(x: np.ndarray, w: np.ndarray, b: np.ndarray, mult: np.ndarray) -> np.ndarray:
    # x: [I] int8, w: [O, I] int8, b: [O] int32, mult: [O] f32 → [O] int8
    acc = (w.astype(np.int64) @ x.astype(np.int64)) + b.astype(np.int64)
    return requantize_per_channel(acc, mult)


def forward(image_i8_28x28: np.ndarray, w: Weights):
    """Run TinyConv-INT8 and return (predicted_class, post_fc_logits_i8)."""
    x = image_i8_28x28.reshape(28, 28, 1)
    c1 = w["CONV1_W"].reshape(8, 3, 3, 1)
    c2 = w["CONV2_W"].reshape(16, 3, 3, 8)

    a = conv2d_i8(x, c1, w["CONV1_B"], w["CONV1_MULT"])
    a = relu_i8(a)
    a = maxpool_i8(a)
    b = conv2d_i8(a, c2, w["CONV2_B"], w["CONV2_MULT"])
    b = relu_i8(b)
    b = maxpool_i8(b)            # [5, 5, 16] int8

    flat = b.reshape(-1)
    fc_w = w["FC_W"].reshape(10, 400)
    logits = dense_i8(flat, fc_w, w["FC_B"], w["FC_MULT"])
    return int(np.argmax(logits)), logits


# ─────────────────────────── compare ────────────────────────────

def stats(name: str, a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        return f"  {name:<10s}: shape mismatch  A={a.shape}  B={b.shape}"
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = a64 - b64
    return (f"  {name:<10s}: shape={tuple(a.shape)}  "
            f"|A|max={float(np.abs(a64).max()):.4g}  "
            f"|B|max={float(np.abs(b64).max()):.4g}  "
            f"|A-B|max={float(np.abs(diff).max()):.4g}  "
            f"L1/N={float(np.abs(diff).mean()):.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path, help="first model_weights.rs/.bin (e.g. rlx)")
    p.add_argument("b", type=Path, help="second model_weights.rs/.bin (e.g. pytorch)")
    p.add_argument("--label-a", default="A", help="display label for first file")
    p.add_argument("--label-b", default="B", help="display label for second file")
    p.add_argument("--test-set", type=Path, default=None,
                   help="optional path to test_set.bin (500x [784 i8 + 1 u8] = 392500 bytes)")
    args = p.parse_args()

    A = load(args.a); B = load(args.b)
    print(f"\n== {args.label_a}: {args.a}")
    print(f"== {args.label_b}: {args.b}")

    # ── Arrays ────────────────────────────────────────────────
    print("\n[arrays]")
    keys = sorted(set(A.arrays) | set(B.arrays))
    for k in keys:
        if k not in A.arrays:
            print(f"  {k:<10s}: missing in {args.label_a}"); continue
        if k not in B.arrays:
            print(f"  {k:<10s}: missing in {args.label_b}"); continue
        print(stats(k, A[k], B[k]))

    # ── Activation scales ────────────────────────────────────
    print("\n[scales]")
    for k in sorted(set(A.scales) | set(B.scales)):
        a = A.scales.get(k, float("nan"))
        b = B.scales.get(k, float("nan"))
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        print(f"  {k:<13s}: {args.label_a}={a:.4e}  {args.label_b}={b:.4e}  rel={rel:.2%}")

    # ── Embedded test image forward parity ────────────────────
    print("\n[forward on TEST_IMAGE]")
    pa, la = forward(A["TEST_IMAGE"], A)
    pb, lb = forward(B["TEST_IMAGE"], B)
    print(f"  {args.label_a}: TEST_LABEL={A.test_label}  predicted={pa}  match={pa==A.test_label}")
    print(f"     logits={list(int(x) for x in la)}")
    print(f"  {args.label_b}: TEST_LABEL={B.test_label}  predicted={pb}  match={pb==B.test_label}")
    print(f"     logits={list(int(x) for x in lb)}")

    # ── Test-set sweep ────────────────────────────────────────
    if args.test_set and args.test_set.exists():
        raw = args.test_set.read_bytes()
        ROW = 28 * 28 + 1
        n = len(raw) // ROW
        imgs = np.frombuffer(raw, dtype=np.uint8).reshape(n, ROW)
        labels = imgs[:, -1]
        pixels = imgs[:, :-1].view(np.int8).reshape(n, 28, 28)

        print(f"\n[INT8 forward sweep over {n} test images from {args.test_set.name}]")
        for label, W in ((args.label_a, A), (args.label_b, B)):
            preds = np.array([forward(pixels[i], W)[0] for i in range(n)])
            acc = float((preds == labels).mean())
            agree_with_label = sum(1 for i in range(n) if preds[i] == labels[i])
            print(f"  {label}: int8 accuracy = {acc:.4f}  ({agree_with_label}/{n})")
        preds_a = np.array([forward(pixels[i], A)[0] for i in range(n)])
        preds_b = np.array([forward(pixels[i], B)[0] for i in range(n)])
        agree = int((preds_a == preds_b).sum())
        print(f"  cross-trainer agreement: {agree}/{n} = {agree/n:.4f}")
    else:
        print("\n[no --test-set provided; skipping bulk sweep]")


if __name__ == "__main__":
    sys.exit(main())
