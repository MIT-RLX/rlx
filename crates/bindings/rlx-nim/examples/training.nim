# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
## the same fit by hand with grad and an Optimizer.
##   nim r --path:src examples/training.nim [device]

import std/[math, os, sequtils, strformat, strutils]
import rlx

const wTrue = [2.0, -1.0, 0.5]
const batch = 8

proc data(step: int): seq[(string, seq[float])] =
  var x, y: seq[float]
  for i in 0 ..< batch:
    var s = 0.0
    for k in 0 ..< 3:
      let v = float((step * 31 + i * 7 + k * 3) mod 17) / 8.5 - 1
      x.add v
      s += v * wTrue[k]
    y.add s
  @[("x", x), ("y", y)]

proc fmt3(v: seq[float]): string = v.mapIt(&"{it:.3f}").join(" ")

let device = if paramCount() > 0: paramStr(1) else: "cpu"
let g = newGraph(name = "regression")
let x = g.input("x", [batch, 3])
let y = g.input("y", [batch, 1])
let w = g.param("w", [3, 1])
let d = g.sub(g.matmul(x, w), y)
g.setOutputs([g.mean(g.mul(d, d), [0, 1])])

let trainer = newTrainer(g.clone(), ["w"], {"w": @[0.0, 0, 0]},
  options = %*{"device": device, "optimizer": {"kind": "adamw", "lr": 0.1}})
let first = trainer.evaluate(data(0))
discard trainer.run(400, proc (s: int): Option[seq[(string, seq[float])]] = some(data(s)),
  lr = proc (s: int): float = 0.1 * 0.5 * (1 + cos(PI * s.float / 400)),
  onStep = proc (s: int; loss: float): bool = loss > 1e-6)
echo &"trainer loss: {first:.4f} -> {trainer.evaluate(data(0)):.6f} in {trainer.steps()} steps"
echo "trainer w: ", fmt3(trainer.params()["w"])

let path = getTempDir() / "rlx-example-train-nim.gguf"
let written = trainer.save(path)
let restored = newTrainer(g.clone(), ["w"], {"w": @[0.0, 0, 0]}, options = %*{"device": device})
echo &"checkpoint: {written} tensors, restored step {restored.load(path)[\"step\"]}"
removeFile(path)
discard restored.freeze(names = some(@["w"]))
let before = fmt3(restored.params()["w"])
discard restored.step(data(1))
echo "frozen: ", restored.frozen(), " unchanged: ", fmt3(restored.params()["w"]) == before
discard restored.unfreeze()

let backward = newSession(device = device).compile(grad(g, [w])) # outputs [loss, dw]
let opt = newOptimizer(kind = some("sgd"), options = %*{"lr": 0.2})
var params = {"w": @[0.0, 0, 0]}.toTable
var loss = 0.0
for step in 0 ..< 300:
  backward.setParam("w", params["w"])
  let outs = backward.run(data(step) & @[("d_output", @[1.0])])
  loss = outs[0][0]
  opt.step(params, {"w": outs[1]}.toTable)
echo &"manual loss: {loss:.6f} w: {fmt3(params[\"w\"])}"
