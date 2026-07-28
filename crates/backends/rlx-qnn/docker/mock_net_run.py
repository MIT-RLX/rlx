#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Stand-in for `qnn-net-run`, used by the SDK-free harness self-test
# (`validate.py harness-test`). It reads the SAME `input_list.txt` + raw
# float32 layout the real tool consumes, and writes `output/Result_0/out.raw`
# exactly where `verify.py --check` looks.
#
# It deliberately does NOT exercise the emitted QNN C++ lowering — that needs
# the real SDK (`validate.py run`). What it validates is the host-harness
# plumbing: input_list parsing, raw dtype/shape, and the output path/naming
# contract between verify.py and qnn-net-run. M/K/N stand in for the shapes the
# compiled libQnnModel.so would carry.

import os
import sys

import numpy as np

M, K, N = (int(x) for x in sys.argv[1:4])

# "in0:=in0.raw in1:=in1.raw" -> {"in0": "in0.raw", "in1": "in1.raw"}
spec = open("input_list.txt").read().split()
files = {tok.split(":=")[0]: tok.split(":=")[1] for tok in spec}

in0 = np.fromfile(files["in0"], dtype=np.float32).reshape(M, K)
in1 = np.fromfile(files["in1"], dtype=np.float32).reshape(K, N)

os.makedirs("output/Result_0", exist_ok=True)
(in0 @ in1).astype(np.float32).tofile("output/Result_0/out.raw")
print(f"[mock qnn-net-run] wrote output/Result_0/out.raw  ({M}x{N} f32)")
