# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""DINO / ViT → RLX, exported in every form.  Run: `python dino.py`

A DINO-style Vision Transformer: conv patch-embed → transformer encoder
(self-attention + MLP) → pooled linear head. Same shape as DINOv2 / ViT
backbones; a real checkpoint exports the same way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import export_all_forms


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


def build():
    return ViT().eval(), (torch.randn(1, 3, 16, 16),)


if __name__ == "__main__":
    export_all_forms("dino", *build())
