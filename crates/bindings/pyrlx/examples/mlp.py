# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""MLP → RLX, exported in every form.  Run: `python mlp.py`

A minimal feed-forward block (LayerNorm + Linear + GELU + Linear) — the
simplest end-to-end example. `export_all_forms` writes a runnable HIR-graph
bundle and standalone graph/tensor/flow crates under `out_mlp/`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import export_all_forms


class MLP(nn.Module):
    def __init__(self, d=16, h=32):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln(x))))


def build():
    return MLP().eval(), (torch.randn(2, 4, 16),)


if __name__ == "__main__":
    export_all_forms("mlp", *build())
