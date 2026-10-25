# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Reference for rlx-nnpack: torch 2.10's own NNPACK (`torch._nnpack_spatial_convolution`,
the kernel `_ConvBackend.NnpackSpatial` runs), macOS arm64 wheel.

    /Users/Shared/fnirsatlas-ref/.venv/bin/python -I parity/export_torch_nnpack.py

Writes tests/fixtures/torch_nnpack.json: cases of (shapes, padding, input, kernel, bias,
output) with every float as its IEEE-754 bits in hex.
"""
import json, os, struct
import numpy as np, torch

assert torch.__version__.startswith("2.10"), torch.__version__
torch.set_num_threads(1)
assert torch._nnpack_available()  # also initializes NNPACK
rng = np.random.default_rng(20261025)
hexs = lambda a: [format(struct.unpack("<I", struct.pack("<f", float(v)))[0], "08x") for v in np.asarray(a, dtype=np.float32).ravel()]

def tensor(shape, special=True):
    x = rng.uniform(-2, 2, size=shape).astype(np.float32)
    if special:
        flat = x.reshape(-1)
        k = max(1, flat.size // 40)
        idx = rng.choice(flat.size, size=3 * k, replace=False)
        flat[idx[:k]] = 0.0
        flat[idx[k:2 * k]] = -0.0
        flat[idx[2 * k:]] = (rng.uniform(-1, 1, size=k) * 1e-39).astype(np.float32)  # denormals
    return x

cases = []
# batch 1 is NOT this kernel: torch's _nnpack_spatial_convolution sends it to
# nnp_convolution_inference (torch's conv backend only picks NNPACK at batch >= 16)
# (batch, in_ch, out_ch, H, W, kh, kw, pad_h, pad_w, bias?)
specs = [
    (16, 1, 8, 1, 76, 1, 9, 0, 4, False),   # Hydra: 1-D, k 9, pad 4
    (18, 1, 8, 1, 60, 1, 9, 0, 4, True),
    (17, 1, 64, 1, 76, 1, 9, 0, 4, False),  # Hydra univariate: all k·h kernels at once
    (3, 2, 3, 12, 37, 9, 9, 4, 4, True),
    (2, 3, 5, 20, 20, 11, 13, 5, 6, True),
    (2, 1, 1, 16, 16, 16, 16, 0, 0, True),
    (4, 2, 2, 5, 50, 3, 10, 2, 9, False),
    (2, 1, 3, 9, 33, 1, 7, 0, 3, True),     # 7-wide kernel that auto still sends to ft16x16
    (2, 300, 2, 3, 20, 1, 9, 0, 0, True),   # > 256 input channels: two GEMM blocks
    (2, 600, 1, 1, 12, 1, 9, 0, 0, False),  # three blocks
]
for (b, ic, oc, h, w, kh, kw, ph, pw, has_bias) in specs:
    assert b >= 2
    x, k = tensor((b, ic, h, w)), tensor((oc, ic, kh, kw))
    bias = tensor((oc,), special=False) if has_bias else np.zeros(oc, np.float32)
    out = torch._nnpack_spatial_convolution(torch.from_numpy(x), torch.from_numpy(k), torch.from_numpy(bias), [ph, pw], [1, 1]).numpy()
    cases.append({"batch": b, "in_ch": ic, "out_ch": oc, "in_h": h, "in_w": w, "k_h": kh, "k_w": kw, "pad": [ph, pw, ph, pw],
                  "input": hexs(x), "kernel": hexs(k), "bias": hexs(bias), "output": hexs(out), "out_shape": list(out.shape)})
    print(b, ic, oc, h, w, kh, kw, ph, pw, "->", out.shape)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "torch_nnpack.json")
json.dump({"torch": torch.__version__, "cases": cases}, open(p, "w"))
print("wrote", p, os.path.getsize(p), "bytes")
