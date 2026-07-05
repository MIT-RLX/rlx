# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""Export existing PyTorch models to RLX — worked examples for every option.

`pyrlx.from_torch(model, example_inputs, out_dir, ...)` converts a live
`torch.nn.Module` to an RLX model. Under the hood it runs `torch.export`, maps
each ATen op directly onto RLX ops, and emits a runnable bundle and/or a
standalone RLX crate — then verifies numeric parity against PyTorch.

────────────────────────────────────────────────────────────────────────────
QUICKSTART
────────────────────────────────────────────────────────────────────────────
    import pyrlx, torch

    model = MyModel().eval()
    example_inputs = (torch.randn(1, 3, 224, 224),)          # a tuple
    summary = pyrlx.from_torch(model, example_inputs, out_dir="out/", verify=True)
    print(summary["rlx_result"]["parity"])                   # cosine + max abs err

Output layout under `out/`:
    torch-ir.json          # the exported graph (aten ops + shapes + dtypes)
    weights.safetensors    # parameters (HF-canonical names)
    reference.safetensors  # golden inputs/outputs (for --verify)
    bundle/                # runnable: model.hir.json + weights + meta
    rlx-<name>/            # standalone generated RLX crate

────────────────────────────────────────────────────────────────────────────
ALL OPTIONS (pyrlx.from_torch)  — one example each below in `option_examples()`
────────────────────────────────────────────────────────────────────────────
    out_dir           : str/Path   where artifacts are written.
    model_name        : str|None   name for the artifacts (default: class name).
    verify            : bool       run the RLX model on CPU and compare to
                                    PyTorch (cosine ≥ 0.999, |err| ≤ 1e-2). Default True.
    emit              : tuple      artifacts to produce:
                                     "bundle" — runnable HIR + weights (run now)
                                     "crate"  — standalone Rust crate (ship/edit)
                                    Default ("bundle", "crate").
    emit_style        : str        authoring layer the generated CRATE targets:
                                     "graph"  — raw HIR builder (all ops; default)
                                     "tensor" — PyTorch-like `Tensor` DSL (readable)
                                     "flow"   — `ModelFlow` blocks (runner ecosystem)
    decomposition     : str        how much torch.export lowers the graph:
                                     "aten" — no decomposition; closest to the
                                              source module (fewest nodes)
                                     "high" — fused high-level ops preserved (default)
                                     "core" — most primitive (full Core ATen)
    run_rlx           : bool|"auto" invoke the Rust worker to build bundle/crate +
                                    verify. "auto" runs it if found. False = only
                                    write torch-ir.json + weights (front-end only).
    strict            : bool       torch.export strict mode (default False).

CLI equivalent (a .py exposing `model` + `example_inputs`, or get_model()/build()):
    rlx-torch-import model.py -o out/ \
        --emit bundle,crate --emit-style tensor \
        --decomposition high --verify --device cpu

Run this file:  `python torch_to_rlx.py [mlp|encdec|cnn|mnist|llama|dino|all|options]`
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyrlx


# ═══════════════════════════════════════════════════════════════════════════
# 1. Simple MLP  (linear + layernorm + gelu)
# ═══════════════════════════════════════════════════════════════════════════
def make_mlp():
    class MLP(nn.Module):
        def __init__(self, d=16, h=32):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, h)
            self.fc2 = nn.Linear(h, d)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(self.ln(x))))

    return MLP().eval(), (torch.randn(2, 4, 16),)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Encoder-decoder  (self-attention + cross-attention)
# ═══════════════════════════════════════════════════════════════════════════
def make_encoder_decoder():
    class Block(nn.Module):
        def __init__(self, d=32, h=4):
            super().__init__()
            self.h, self.hd = h, d // h
            self.qkv = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in "qkvo"})

        def mha(self, x, kv):
            b, s, d = x.shape
            sk = kv.shape[1]
            q = self.qkv["q"](x).view(b, s, self.h, self.hd).transpose(1, 2)
            k = self.qkv["k"](kv).view(b, sk, self.h, self.hd).transpose(1, 2)
            v = self.qkv["v"](kv).view(b, sk, self.h, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v)
            return self.qkv["o"](a.transpose(1, 2).reshape(b, s, d))

    class EncDec(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.enc, self.dec_self, self.dec_cross = Block(d), Block(d), Block(d)
            self.n = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])

        def forward(self, src, tgt):
            mem = src + self.enc.mha(self.n[0](src), self.n[0](src))
            tgt = tgt + self.dec_self.mha(self.n[1](tgt), self.n[1](tgt))
            tgt = tgt + self.dec_cross.mha(self.n[2](tgt), mem)
            return tgt

    return EncDec().eval(), (torch.randn(1, 5, 32), torch.randn(1, 3, 32))


# ═══════════════════════════════════════════════════════════════════════════
# 3. CNN  (conv + batchnorm + relu + pool + head)
# ═══════════════════════════════════════════════════════════════════════════
def make_cnn():
    class CNN(nn.Module):
        def __init__(self, classes=10):
            super().__init__()
            self.c1, self.b1 = nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8)
            self.c2, self.b2 = nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16)
            self.head = nn.Linear(16, classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
            x = F.relu(self.b2(self.c2(x)))
            return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))

    return CNN().eval(), (torch.randn(2, 3, 16, 16),)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MNIST classifier  (LeNet-style; int-free, [B,1,28,28] → [B,10])
# ═══════════════════════════════════════════════════════════════════════════
def make_mnist():
    class MnistNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 8, 3, padding=1)
            self.c2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.c1(x)), 2)
            x = F.max_pool2d(F.relu(self.c2(x)), 2)
            return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

    return MnistNet().eval(), (torch.randn(4, 1, 28, 28),)


# ═══════════════════════════════════════════════════════════════════════════
# 5. LLaMA  (real HF decoder — rotary + GQA + causal mask + SwiGLU)
#    Uses the actual transformers architecture at a tiny config so it exports
#    fast; a real checkpoint works the same way (weight names are HF-canonical).
# ═══════════════════════════════════════════════════════════════════════════
def make_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, ids):
            return self.m(input_ids=ids).logits

    # int64 token ids are fed as f32 + a cast internally, so this runs on CPU,
    # CUDA and Metal alike.
    return Wrap().eval(), (torch.randint(0, 64, (1, 8)),)


# ═══════════════════════════════════════════════════════════════════════════
# 6. DINO / ViT  (patch embed + transformer encoder + pooled head)
# ═══════════════════════════════════════════════════════════════════════════
def make_dino():
    class ViT(nn.Module):
        def __init__(self, d=32, h=4, ffn=64, classes=10):
            super().__init__()
            self.h, self.hd = h, d // h
            self.patch = nn.Conv2d(3, d, 4, stride=4)
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
            self.v, self.o = nn.Linear(d, d), nn.Linear(d, d)
            self.f1, self.f2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
            self.head = nn.Linear(d, classes)

        def forward(self, x):
            x = self.patch(x)
            b, d, hh, ww = x.shape
            x = x.flatten(2).transpose(1, 2)                       # [B, N, d]
            n = x.shape[1]
            hs = self.n1(x)
            q = self.q(hs).view(b, n, self.h, self.hd).transpose(1, 2)
            k = self.k(hs).view(b, n, self.h, self.hd).transpose(1, 2)
            v = self.v(hs).view(b, n, self.h, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
            x = x + self.o(a)
            x = x + self.f2(F.gelu(self.f1(self.n2(x))))
            return self.head(x.mean(1))                            # pooled logits

    return ViT().eval(), (torch.randn(1, 3, 16, 16),)


MODELS = {
    "mlp": make_mlp,
    "encdec": make_encoder_decoder,
    "cnn": make_cnn,
    "mnist": make_mnist,
    "llama": make_llama,
    "dino": make_dino,
}


# ═══════════════════════════════════════════════════════════════════════════
# One example for every `from_torch` OPTION (on the MLP for brevity).
# ═══════════════════════════════════════════════════════════════════════════
def option_examples(out_root="out_options"):
    model, ex = make_mlp()

    # verify: run the converted model and compare to PyTorch (the default).
    pyrlx.from_torch(model, ex, f"{out_root}/verify", verify=True)

    # emit: choose artifacts — a runnable bundle, a standalone crate, or both.
    pyrlx.from_torch(model, ex, f"{out_root}/bundle_only", emit=("bundle",))
    pyrlx.from_torch(model, ex, f"{out_root}/crate_only", emit=("crate",))

    # emit_style: which RLX authoring layer the generated crate uses.
    pyrlx.from_torch(model, ex, f"{out_root}/style_graph", emit=("crate",), emit_style="graph")
    pyrlx.from_torch(model, ex, f"{out_root}/style_tensor", emit=("crate",), emit_style="tensor")
    pyrlx.from_torch(model, ex, f"{out_root}/style_flow", emit=("crate",), emit_style="flow")

    # decomposition: how close the graph stays to the source module.
    pyrlx.from_torch(model, ex, f"{out_root}/decomp_aten", decomposition="aten")
    pyrlx.from_torch(model, ex, f"{out_root}/decomp_high", decomposition="high")
    pyrlx.from_torch(model, ex, f"{out_root}/decomp_core", decomposition="core")

    # run_rlx=False: front-end only — just torch-ir.json + weights (no Rust build).
    pyrlx.from_torch(model, ex, f"{out_root}/frontend_only", run_rlx=False)

    print("wrote per-option examples under", out_root)


def convert_one(name: str) -> None:
    factory = MODELS[name]
    try:
        model, ex = factory()
    except ImportError as e:
        print(f"[{name}] skipped (missing dep: {e})")
        return
    summary = pyrlx.from_torch(model, ex, out_dir=f"out_{name}", model_name=name, verify=True)
    parity = (summary.get("rlx_result") or {}).get("parity")
    if parity and parity.get("outputs"):
        o = parity["outputs"][0]
        print(f"[{name}] parity: cosine={o['cosine']:.6f}  max|err|={o['max_abs_err']:.2e}")
    else:
        print(f"[{name}] {summary.get('num_nodes')} nodes — see out_{name}/ (rlx_ran={summary.get('rlx_ran')})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "mlp"
    if arg == "options":
        option_examples()
    elif arg == "all":
        for name in MODELS:
            convert_one(name)
    elif arg in MODELS:
        convert_one(arg)
    else:
        print(f"usage: python torch_to_rlx.py [{'|'.join(MODELS)}|all|options]")
        raise SystemExit(2)
