# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Inference: build a graph, compile it for a device, set its weights, run it.
##   nim r --path:src examples/inference.nim [device]
## A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

import std/[math, os, sequtils, strformat, strutils]
import rlx

proc weights(n, seed: int): seq[float] =
  for i in 0 ..< n: result.add float((i * 7919 + seed * 104729) mod 1000) / 1000 - 0.5

let device = if paramCount() > 0: paramStr(1) else: "cpu"
let g = newGraph(name = "classifier")
let x = g.input("x", [1, 4])
let hidden = g.relu(g.matmul(x, g.param("w1", [4, 8])))
g.setOutputs([g.softmax(g.matmul(hidden, g.param("w2", [8, 3])))])

let model = newSession(device = device).compile(g) # consumes g
echo "compiled on: ", model.device()
discard model.setParams({"w1": weights(32, 1), "w2": weights(24, 2)})
let probs = model.run({"x": @[0.5, -1, 2, 0.25]})[0]
echo "probs: ", probs.mapIt(&"{it:.4f}").join(" ")
echo &"probs sum: {sum(probs):.4f}"
echo "output shapes: ", model.outputShapes(), " dtypes: ", model.outputDtypes()
