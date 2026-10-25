# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Loading: weights from disk into a compiled graph — a GGUF file (written
here, f32 and quantized) and a safetensors file — and quantizing in memory.
With RLX_TOKENIZER=<tokenizer.json> it also tokenizes and samples.

    python loading.py
"""

import json
import os
import struct
import tempfile

import numpy as np

import pyrlx

tmp = tempfile.gettempdir()
w = ((np.arange(64) % 9) / 4 - 1).astype(np.float32)

# ── GGUF ────────────────────────────────────────────────────────────────────
gguf_path = os.path.join(tmp, "rlx-example.gguf")
pyrlx.write_gguf(
    gguf_path,
    {
        "proj.weight": {"data": w, "shape": [8, 8], "dtype": "F32"},
        "proj.weight_q8": {"data": w, "shape": [8, 8], "dtype": "Q8_0"},
    },
    architecture="demo",
    metadata={"demo.layers": 1},
)
f = pyrlx.GgufFile.load(gguf_path)
print("gguf architecture:", f.architecture)
print("gguf tensors:", " ".join(f.tensor_names()))
exact = np.asarray(f.dequant_tensor("proj.weight"), np.float32).ravel()
q8 = np.asarray(f.dequant_tensor("proj.weight_q8"), np.float32).ravel()
print("gguf f32 exact:", bool((exact == w).all()), f"q8 max|Δ|: {np.abs(exact - q8).max():.4f}")

g = pyrlx.Graph("proj")
x = g.input("x", [1, 8])
g.set_outputs([g.matmul(x, g.param("proj.weight", [8, 8]))])
c = pyrlx.Session("cpu").compile(g)
c.set_param("proj.weight", exact.reshape(8, 8))
print("gguf -> graph:", np.round(c.run({"x": np.ones((1, 8), np.float32)})[0].ravel()[:3], 2).tolist())

# ── safetensors → every tensor as f32 ───────────────────────────────────────
st_path = os.path.join(tmp, "rlx-example.safetensors")
header = json.dumps({"proj.weight": {"dtype": "F32", "shape": [8, 8], "data_offsets": [0, w.nbytes]}})
header += " " * (-len(header) % 8)
with open(st_path, "wb") as fh:
    fh.write(struct.pack("<Q", len(header)) + header.encode() + w.tobytes())
loaded = pyrlx.load_weights("safetensors", st_path)
print("safetensors:", sorted(loaded), "exact:", bool((np.asarray(loaded["proj.weight"]) == w).all()))

# ── quantize in memory ──────────────────────────────────────────────────────
packed = pyrlx.quantize_gguf(w, "Q8_0")
back = pyrlx.dequant_gguf(packed, "Q8_0", num_elements=w.size)
print(f"quantize q8_0: {w.nbytes} -> {len(packed)} bytes, round trip {len(back)} values")
os.remove(gguf_path)
os.remove(st_path)

# ── text ────────────────────────────────────────────────────────────────────
tok_path = os.environ.get("RLX_TOKENIZER")
if tok_path:
    tok = pyrlx.load_tokenizer(tok_path)
    ids = tok.encode("Hello from Python")
    print(f"tokenizer: {len(ids)} ids -> {tok.decode(ids)!r}")
else:
    print("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)")
print("argmax:", pyrlx.argmax([0.1, 2, 0.3]))
