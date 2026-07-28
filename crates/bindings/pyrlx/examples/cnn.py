# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""CNN → RLX, exported in every form.  Run: `python cnn.py`

Conv + BatchNorm + ReLU + MaxPool ×2 + adaptive-avg-pool + linear head. BatchNorm
is decomposed to be layout-correct, and its integer `num_batches_tracked` buffer
is handled transparently (fed as f32 + a cast, so the crate runs on GPU too).
The `tensor` DSL has no pooling op, so that form reports as n/a — expected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import export_all_forms


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


def build():
    return CNN().eval(), (torch.randn(2, 3, 16, 16),)


if __name__ == "__main__":
    export_all_forms("cnn", *build())
