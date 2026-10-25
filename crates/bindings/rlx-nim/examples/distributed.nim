# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Distributed: each rank is a process; start one per rank with the same peers
## (here three on loopback; across machines, one per host):
##   ./distributed 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
##   ./distributed 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
##   ./distributed 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
## (build it with ). Rank 0
## coordinates: collectives, pipelined inference (a stage shipped per worker),
## data-parallel training. With no arguments it is a mesh of one.

import std/[os, sequtils, streams, strformat, strutils]
import rlx

let rank = if paramCount() >= 1: parseInt(paramStr(1)) else: 0
let world = if paramCount() >= 2: parseInt(paramStr(2)) else: 1
const perRank = 16
var blobs: seq[string]
proc blob(name: string; values: seq[float]): string =
  let path = getTempDir() / &"rlx-dist-nim-{rank}-{name}"
  let s = newFileStream(path, fmWrite)
  for v in values: s.write(float32(v))
  s.close()
  blobs.add path
  "file://" & path

var options = %*{"device": "cpu"}
if paramCount() >= 3: options["peers"] = %paramStr(3).split(',')
let g = Group.connect(rank, world, options = options)
let total = g.allReduce([float(rank + 1)], op = "sum")
let gathered = g.allGather([float(rank + 1)])
let shared = g.broadcast(0, [if rank == 0: 42.0 else: 0.0]) # the root's length
if rank == 0:
  echo &"mesh: rank {g.rank()} of {g.world()}, leader: {g.isLeader()}"
  echo "all_reduce sum: ", total.mapIt($it).join(",")
  echo "all_gather: ", gathered.mapIt($it).join(",")
  echo "broadcast: ", shared.mapIt($it).join(",")
if world == 1:
  echo "a mesh of one: pass rank world peers to start more ranks"
  quit(0)
if rank != 0:
  discard g.serveWorker(options = %*{"steps": 2})
  let w = g.serveTrainer()["params"]["w"]
  g.send(0, 1, w.getElems.mapIt(it.getFloat)) # rank 0 compares
  quit(0)

let stageW = [toSeq(0 ..< 16).mapIt(float(it * 5 mod 7) / 7 - 0.3), toSeq(0 ..< 16).mapIt(float(it * 3 mod 5) / 5 - 0.2)]
var caps: seq[string]
for c in g.collectCaps(): caps.add &"rank {c[\"rank\"]} on {c[\"platform\"].getStr}"
echo "workers: ", caps.join(", ")
let stage = newGraph(name = "stage")
stage.setOutputs([stage.relu(stage.matmul(stage.input("x", [1, 4]), stage.param("w", [4, 4])))])
for r in 1 ..< world:
  g.shipStage(r, stage, %*{"device": "cpu", "weights": {"w": blob(&"w{r}.bin", stageW[(r - 1) mod 2])}})
for xv in [@[1.0, -1, 0.5, 2], @[0.25, 0.5, -0.75, 1]]:
  g.sendActivation(1, xv)
  echo "pipeline: ", g.recvActivation(world - 1).mapIt(&"{it:.3f}").join(" ")

discard g.collectCaps()
let lr = newGraph(name = "lr")
let bigX = lr.input("X", [4, 3])
let y = lr.input("y", [4, 1])
let wv = lr.param("w", [3, 1])
let d = lr.sub(lr.matmul(bigX, wv), y)
lr.setOutputs([lr.mean(lr.mul(d, d), [0, 1])])
let bwd = grad(lr, [wv])
let wTrue = [1.5, -2.0, 0.5]
let n = perRank * world
let xAll = toSeq(0 ..< n * 3).mapIt(float(it * 37 mod 23) / 11.5 - 1)
var yAll: seq[float]
for i in 0 ..< n: yAll.add xAll[i * 3] * wTrue[0] + xAll[i * 3 + 1] * wTrue[1] + xAll[i * 3 + 2] * wTrue[2]
let ux = blob("X.bin", xAll)
let uy = blob("y.bin", yAll)
let uw = blob("w0.bin", @[0.0, 0, 0])
proc spec(r: int): JsonNode = %*{
  "params": [{"name": "w", "uri": uw}],
  "data": [
    {"input": "X", "uri": ux, "elem": 3, "shard_start": r * perRank, "shard_len": perRank},
    {"input": "y", "uri": uy, "elem": 1, "shard_start": r * perRank, "shard_len": perRank}],
  "batch": 4, "lr_per_epoch": newSeqWith(40, 0.2), "device": "cpu"}
for r in 1 ..< world: g.shipTrain(r, bwd, spec(r))
let mine = g.runTrain(bwd, spec(0))
echo &"training loss: {mine[\"metrics\"][\"first_loss\"].getFloat:.4f} -> {mine[\"metrics\"][\"last_loss\"].getFloat:.2e}"
let w0 = mine["params"]["w"].getElems.mapIt(it.getFloat)
echo "rank 0 w: ", w0.mapIt(&"{it:.3f}").join(" "), " (true: 1.5 -2 0.5)"
for r in 1 ..< world:
  let theirs = g.recv(r, 1)
  var agree = true
  for k in 0 ..< w0.len: agree = agree and abs(theirs[k] - w0[k]) < 1e-5
  echo &"rank {r} w agrees: {agree}"
for p in blobs: removeFile(p)
