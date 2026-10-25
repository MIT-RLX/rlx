# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## The Nim binding against the real library; quits non-zero on the first failure.

import std/os
import rlx

var checks = 0
proc check(ok: bool; what: string) =
  inc checks
  if not ok:
    stderr.writeLine "failed: ", what
    quit(1)

proc lossGraph(): (Graph, int) =
  let g = newGraph(name = "mse")
  let x = g.input("x", [1, 2])
  let y = g.input("y", [1, 2])
  let w = g.param("w", [2, 2])
  let d = g.sub(g.matmul(x, w), y)
  g.setOutputs([g.mean(g.mul(d, d), [0, 1])])
  (g, w)

block inference:
  let (g, _) = lossGraph()
  let m = newSession().compile(g)
  check g.isConsumed(), "compile consumes"
  m.setParam("w", [1.0, 0, 0, 1])
  check m.run({"x": @[1.0, 2], "y": @[0.0, 0]})[0][0] == 2.5, "loss 2.5"
  check m.device() == "cpu", "device"

block errors:
  try:
    discard newGraph().relu(42)
    check false, "range error raised"
  except RlxError as e:
    check e.kind == ekRange, "range error"

block training:
  let (g, w) = lossGraph()
  let bwd = newSession().compile(grad(g, [w]))
  bwd.setParam("w", [1.0, 0, 0, 1])
  check bwd.run({"x": @[1.0, 2], "y": @[0.0, 0], "d_output": @[1.0]})[1][3] == 4, "dw"
  let t = newTrainer(g.clone(), ["w"], {"w": @[0.0, 0, 0, 0]},
    options = %*{"optimizer": {"kind": "adamw", "lr": 0.1}})
  let r = t.run(300, proc (s: int): Option[seq[(string, seq[float])]] = some(@[("x", @[1.0, 1]), ("y", @[1.0, -1])]),
    onStep = proc (s: int; l: float): bool = l > 1e-6)
  check r.lastLoss.get < 1e-4, "trainer converges"
  let opt = newOptimizer(kind = some("sgd"), options = %*{"lr": 0.5})
  var p = {"p": @[1.0, -1]}.toTable
  opt.step(p, {"p": @[2.0, -2]}.toTable)
  check p["p"] == @[0.0, 0], "optimizer in place"

block loading:
  let path = getTempDir() / "rlx-nim-test.gguf"
  let w = newGgufWriter()
  w.addF32("w", [4], "F32", [0.5, -1, 2, 0.25])
  check w.write(path, architecture = some("demo")) == 1, "write"
  let f = Gguf.open(path)
  check f.architecture() == some("demo"), "architecture"
  check f.tensor("w")[2] == 2, "tensor"
  check f.info("w")["dtype"].getStr == "F32", "info"
  removeFile(path)

block backends:
  check devices()[0] == "cpu" and isAvailable("cpu"), "cpu"
  let (g, _) = lossGraph()
  let r = newRunner(g, policy = %*{"only": ["cpu"]})
  r.setParam("w", [1.0, 0, 0, 1])
  let res = r.run({"x": @[1.0, 2], "y": @[0.0, 0]})
  check res.device == "cpu" and res.outputs[0][0] == 2.5, "runner"
  check Group.connect(0, 1).allReduce([2.0, 3], op = "max")[1] == 3, "all_reduce"

echo "test_rlx: ", checks, " checks passed"
