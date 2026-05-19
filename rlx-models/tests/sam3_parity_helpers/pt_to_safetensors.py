#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# (license header truncated — see workspace root.)

"""Convert SAM3 .pt checkpoints to safetensors with original key names."""

import sys

import torch
from safetensors.torch import save_file


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} in.pt out.safetensors", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        print("checkpoint is not a tensor state dict", file=sys.stderr)
        return 3
    tensors = {k: v.detach().cpu().contiguous() for k, v in ckpt.items() if hasattr(v, "detach")}
    save_file(tensors, dst)
    print(f"wrote {len(tensors)} tensors to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
