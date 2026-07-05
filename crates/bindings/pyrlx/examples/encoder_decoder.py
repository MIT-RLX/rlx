# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""Encoder-decoder → RLX, exported in every form.  Run: `python encoder_decoder.py`

An encoder block + a decoder block with **self- and cross-attention** — exercises
`scaled_dot_product_attention` where the KV comes from a different tensor than the
query (the cross-attention that a plain transformer stack doesn't have).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import export_all_forms


class Block(nn.Module):
    def __init__(self, d=32, h=4):
        super().__init__()
        self.h, self.hd = h, d // h
        self.p = nn.ModuleDict({k: nn.Linear(d, d, bias=False) for k in "qkvo"})

    def mha(self, x, kv):
        b, s, d = x.shape
        sk = kv.shape[1]
        q = self.p["q"](x).view(b, s, self.h, self.hd).transpose(1, 2)
        k = self.p["k"](kv).view(b, sk, self.h, self.hd).transpose(1, 2)
        v = self.p["v"](kv).view(b, sk, self.h, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.p["o"](a.transpose(1, 2).reshape(b, s, d))


class EncDec(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.enc, self.dec_self, self.dec_cross = Block(d), Block(d), Block(d)
        self.n = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])

    def forward(self, src, tgt):
        mem = src + self.enc.mha(self.n[0](src), self.n[0](src))
        tgt = tgt + self.dec_self.mha(self.n[1](tgt), self.n[1](tgt))
        tgt = tgt + self.dec_cross.mha(self.n[2](tgt), mem)  # cross-attn to memory
        return tgt


def build():
    return EncDec().eval(), (torch.randn(1, 5, 32), torch.randn(1, 3, 32))


if __name__ == "__main__":
    export_all_forms("encoder_decoder", *build())
