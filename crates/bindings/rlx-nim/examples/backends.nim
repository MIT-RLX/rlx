# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Picking a backend: what this build has, what suits a graph, and the ways to
## choose — by name, by cost model, by policy, by benchmark, per run.
##   nim r --path:src examples/backends.nim

import std/[strformat, strutils]
import rlx

proc model(): Graph =
  result = newGraph(name = "mlp")
  let x = result.input("x", [4, 16])
  result.setOutputs([result.tanh(result.matmul(x, result.param("w", [16, 16])))])

var w, xs: seq[float]
for i in 0 ..< 256: w.add float(i mod 5) / 10 - 0.2
for i in 0 ..< 64: xs.add float(i mod 7) / 7
let x = {"x": xs}

let names = devices()
echo "devices: ", names.join(" ")
echo "metal available: ", isAvailable("metal")
echo "fastest for graph: ", fastestDeviceFor(model())
var reference: seq[float]
for device in names:
  let c = newSession(device = device).compile(model())
  c.setParam("w", w)
  let y = c.run(x)[0]
  if reference.len == 0: reference = y
  var d = 0.0
  for i in 0 ..< y.len: d = max(d, abs(y[i] - reference[i]))
  echo &"session {device}: max|Δ| vs first = {d:.2e}"
let flex = newFlexibleSession(policy = %*{"prefer": ["metal", "cpu"]})
echo "flexible session picked: ", flex.compile(model()).device()
let runner = newRunner(model())
runner.setParam("w", w)
echo "runner warmed: ", runner.warmAll().join(" ")
for row in runner.benchmark(x, runs = 3):
  echo "  bench ", row["device"].getStr, ": exec_ms=", row["exec_ms"]
echo "runner ran on: ", runner.run(x).device
echo "run_try chain: ", runner.runTry(["metal", "cpu"], x).device
let router = newRouter(model())
router.setParam("w", w)
echo "router ran on: ", router.runChain(x).device
