#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Train TinyConv-MNIST and export INT8 weights as a Rust source file.

Architecture (NHWC in the exported form):

    Input  : [28, 28, 1]  i8  (symmetric, scale = 1/127)
    Conv1  : 1 -> 8, 3x3, valid   -> [26, 26, 8]
    ReLU
    MaxPool: 2x2, stride 2        -> [13, 13, 8]
    Conv2  : 8 -> 16, 3x3, valid  -> [11, 11, 16]
    ReLU
    MaxPool: 2x2, stride 2        -> [5, 5, 16]
    Flatten                        -> [400]
    Dense  : 400 -> 10

Quantization: per-tensor symmetric (zero_point = 0) for activations and
weights. Bias is i32 in the accumulator's scale (x_scale * w_scale).
Output requantization multiplier `mult = (x_scale * w_scale) / out_scale`
is stored as f32 — fine for an M4F with FPU.

Run:
    python3 rlx-cortexm/tools/train_mnist.py --epochs 2

Output: rlx-cortexm/src/model_weights.rs
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────── model ──

class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3)         # 28 -> 26
        self.c2 = nn.Conv2d(8, 16, 3)        # 13 -> 11
        self.fc = nn.Linear(5 * 5 * 16, 10)

    def forward(self, x):                     # x: [N, 1, 28, 28], in [-1, 1]
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x)


# ──────────────────────────────────────────────── train / eval ──

def train(model, loader, device, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        total = correct = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"  epoch {ep + 1}: train acc = {correct/total:.4f}", file=sys.stderr)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


# ──────────────────────────────────────────────── quantization ──

@torch.no_grad()
def calibrate_act_scales(model, loader, device, n_batches=10):
    """Walk forward, record max-abs of each named activation."""
    maxes = {"x": 0.0, "c1": 0.0, "p1": 0.0, "c2": 0.0, "p2": 0.0}

    model.eval()
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        maxes["x"] = max(maxes["x"], x.abs().max().item())
        a = F.relu(model.c1(x));            maxes["c1"] = max(maxes["c1"], a.abs().max().item())
        a = F.max_pool2d(a, 2);              maxes["p1"] = max(maxes["p1"], a.abs().max().item())
        a = F.relu(model.c2(a));             maxes["c2"] = max(maxes["c2"], a.abs().max().item())
        a = F.max_pool2d(a, 2);              maxes["p2"] = max(maxes["p2"], a.abs().max().item())
        seen += 1
        if seen >= n_batches:
            break
    # final fc output isn't needed — argmax doesn't care about scale
    return {k: v / 127.0 for k, v in maxes.items()}


def quant_w(t: torch.Tensor):
    """Per-tensor symmetric int8 weight quant. Returns (q: np.int8, scale: float)."""
    m = t.detach().abs().max().item()
    s = max(m / 127.0, 1e-12)
    q = torch.clamp((t / s).round(), -127, 127).to(torch.int8).cpu().numpy()
    return q, s


def quant_bias(b: torch.Tensor, acc_scale: float):
    """Bias is i32, in the accumulator's scale."""
    return torch.clamp((b / acc_scale).round(), -2**31, 2**31 - 1) \
                .to(torch.int32).cpu().numpy()


# ────────────────────────────────────────────────── conv layout ──

def conv_oihw_to_oihw_nhwc(w_oihw: np.ndarray) -> np.ndarray:
    """PyTorch conv weights are [O, I, kH, kW]. Our kernel reads
    [O, kH, kW, I] (NHWC-friendly inner stride)."""
    return np.transpose(w_oihw, (0, 2, 3, 1)).copy()


# ─────────────────────────────────────────────── rust emitter ──

def emit_rust(path: Path, *, w1, b1, w2, b2, wfc, bfc,
              x_scale, c1_scale, p1_scale, c2_scale, p2_scale,
              w1_scale, w2_scale, wfc_scale,
              fc_out_scale,
              test_image, test_label):
    def arr_i8(name, arr):
        flat = arr.flatten().astype(np.int8)
        body = ", ".join(str(int(v)) for v in flat)
        return f"pub static {name}: [i8; {flat.size}] = [{body}];"

    def arr_i32(name, arr):
        flat = arr.flatten().astype(np.int32)
        body = ", ".join(str(int(v)) for v in flat)
        return f"pub static {name}: [i32; {flat.size}] = [{body}];"

    out = []
    out.append("// Auto-generated by tools/train_mnist.py — do not edit by hand.")
    out.append("// Layout: weights are [O, kH, kW, I]; biases are i32 in acc-scale.")
    out.append("#![allow(clippy::approx_constant)]")
    out.append("")
    out.append(arr_i8("CONV1_W", w1))
    out.append(arr_i32("CONV1_B", b1))
    out.append(arr_i8("CONV2_W", w2))
    out.append(arr_i32("CONV2_B", b2))
    out.append(arr_i8("FC_W", wfc))
    out.append(arr_i32("FC_B", bfc))
    out.append("")
    out.append(f"pub const X_SCALE: f32   = {x_scale:.9e};")
    out.append(f"pub const C1_SCALE: f32  = {c1_scale:.9e};")
    out.append(f"pub const P1_SCALE: f32  = {p1_scale:.9e};")
    out.append(f"pub const C2_SCALE: f32  = {c2_scale:.9e};")
    out.append(f"pub const P2_SCALE: f32  = {p2_scale:.9e};")
    out.append(f"pub const W1_SCALE: f32  = {w1_scale:.9e};")
    out.append(f"pub const W2_SCALE: f32  = {w2_scale:.9e};")
    out.append(f"pub const WFC_SCALE: f32 = {wfc_scale:.9e};")
    out.append(f"pub const FC_OUT_SCALE: f32 = {fc_out_scale:.9e};")
    out.append("")
    out.append("// Multipliers for each layer's requantization step.")
    out.append("// mult_layer = (in_scale * weight_scale) / out_scale")
    out.append("pub const CONV1_MULT: f32 = (X_SCALE  * W1_SCALE)  / C1_SCALE;")
    out.append("pub const CONV2_MULT: f32 = (P1_SCALE * W2_SCALE)  / C2_SCALE;")
    out.append("pub const FC_MULT:    f32 = (P2_SCALE * WFC_SCALE) / FC_OUT_SCALE;")
    out.append("")
    out.append(f"// One MNIST test image (i8, symmetric) for the e2e test.")
    out.append(f"pub const TEST_LABEL: u8 = {int(test_label)};")
    out.append(arr_i8("TEST_IMAGE", test_image))
    out.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"  wrote {path} ({path.stat().st_size} bytes)", file=sys.stderr)


# ─────────────────────────────────────────────────────── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data", default=os.path.expanduser("~/.cache/torchvision-mnist"))
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "src" / "model_weights.rs"))
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cpu"  # MNIST trains in <1 min on cpu, keeps the script reproducible.

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # → [-1, 1]
    ])
    train_ds = datasets.MNIST(args.data, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.data, train=False, download=True, transform=tfm)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch)

    model = TinyConv().to(device)
    print("Training…", file=sys.stderr)
    train(model, train_ld, device, args.epochs)
    fp_acc = evaluate(model, test_ld, device)
    print(f"  fp32 test accuracy: {fp_acc:.4f}", file=sys.stderr)

    print("Calibrating activation scales…", file=sys.stderr)
    act = calibrate_act_scales(model, train_ld, device)
    for k, v in act.items():
        print(f"  {k}: scale = {v:.6e}", file=sys.stderr)

    # Quantize weights.
    w1_oihw, w1_s = quant_w(model.c1.weight)
    w2_oihw, w2_s = quant_w(model.c2.weight)

    # Convert conv weights to [O, kH, kW, I] for our kernel.
    w1 = conv_oihw_to_oihw_nhwc(w1_oihw)
    w2 = conv_oihw_to_oihw_nhwc(w2_oihw)

    # FC weight permutation. PyTorch's `x.flatten(1)` reads NCHW input
    # in C-major order (`[c, h, w]` → `c*H*W + h*W + w`). Our runtime
    # carries NHWC and flattens in H-major order (`h*W*C + w*C + c`).
    # Permute the input axis of fc.weight to match HWC order.
    wfc_t = model.fc.weight.detach()        # [10, C*H*W] = [10, 16*5*5]
    wfc_chw = wfc_t.reshape(10, 16, 5, 5)
    wfc_hwc = wfc_chw.permute(0, 2, 3, 1).contiguous().reshape(10, -1)
    wfc, wfc_s = quant_w(wfc_hwc)

    # Bias quantization in each layer's accumulator scale.
    b1  = quant_bias(model.c1.bias, act["x"]  * w1_s)
    b2  = quant_bias(model.c2.bias, act["p1"] * w2_s)
    bfc = quant_bias(model.fc.bias, act["p2"] * wfc_s)

    # FC output scale: pick to span observed logit range so requantize doesn't saturate.
    with torch.no_grad():
        loaded = next(iter(train_ld))[0].to(device)
        logits = model(loaded)
        fc_out_max = logits.abs().max().item()
    fc_out_s = max(fc_out_max / 127.0, 1e-6)

    # Pull a single test image for the e2e test.
    img_t, lbl = test_ds[0]                 # img_t: [1, 28, 28] in [-1, 1]
    img_np = img_t.numpy()[0]                # [28, 28]
    img_q = np.clip(np.round(img_np / act["x"]), -127, 127).astype(np.int8)
    # Layout for kernel: NHWC = [28, 28, 1]
    img_q = img_q[..., None]

    # Bulk validation set for the int8 accuracy test. Format is a flat
    # blob: for each of N images, 784 bytes (i8 NHWC) followed by 1
    # byte label. Loaded by tests/mnist_validate.rs at runtime so the
    # repo doesn't need to vendor it.
    n_val = 500
    val_dir = Path(args.out).parent.parent / "tests" / "data"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_path = val_dir / "test_set.bin"
    with val_path.open("wb") as f:
        for i in range(n_val):
            t, y = test_ds[i]
            q = np.clip(np.round(t.numpy()[0] / act["x"]), -127, 127).astype(np.int8)
            f.write(q.tobytes())
            f.write(bytes([int(y)]))
    print(f"  wrote {val_path} ({n_val} images, {val_path.stat().st_size} bytes)",
          file=sys.stderr)

    out_path = Path(args.out)
    emit_rust(out_path,
              w1=w1, b1=b1, w2=w2, b2=b2, wfc=wfc, bfc=bfc,
              x_scale=act["x"], c1_scale=act["c1"], p1_scale=act["p1"],
              c2_scale=act["c2"], p2_scale=act["p2"],
              w1_scale=w1_s, w2_scale=w2_s, wfc_scale=wfc_s,
              fc_out_scale=fc_out_s,
              test_image=img_q, test_label=lbl)
    print(textwrap.dedent(f"""
        Done.
            fp32 test acc : {fp_acc:.4f}
            test image    : label = {lbl}
            output        : {out_path}
    """).strip(), file=sys.stderr)


if __name__ == "__main__":
    main()
