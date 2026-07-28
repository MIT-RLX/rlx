# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""MNIST classifier → RLX, exported in every form.  Run: `python mnist.py`

A LeNet-style digit classifier: [B,1,28,28] → [B,10]. A conventional, complete
model — swap the random weights for a trained `state_dict` and the same export
produces a deployable RLX bundle/crate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import export_all_forms


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


def build():
    return MnistNet().eval(), (torch.randn(4, 1, 28, 28),)


if __name__ == "__main__":
    export_all_forms("mnist", *build())
