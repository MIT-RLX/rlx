# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
## into a compiled graph, and quantizing in memory.
##   nim r --path:src examples/loading.nim

import std/[os, sequtils, strformat, strutils, streams]
import rlx

let tmp = getTempDir()
var w: seq[float]
for i in 0 ..< 64: w.add float(i mod 9) / 4 - 1

let gguf = tmp / "rlx-example-nim.gguf"
let writer = newGgufWriter()
writer.addF32("proj.weight", [8, 8], "F32", w)
writer.addF32("proj.weight_q8", [8, 8], "Q8_0", w)
discard writer.write(gguf, architecture = some("demo"), metadata = %*{"demo.layers": 1})
let f = Gguf.open(gguf)
echo "gguf architecture: ", f.architecture().get("")
echo "gguf tensors: ", f.tensorNames().join(" ")
let exact = f.tensor("proj.weight")
let q8 = f.tensor("proj.weight_q8")
var maxd = 0.0
for i in 0 ..< 64: maxd = max(maxd, abs(exact[i] - q8[i]))
echo &"gguf f32 exact: {exact == w} q8 max|Δ|: {maxd:.4f}"

let g = newGraph(name = "proj")
let x = g.input("x", [1, 8])
g.setOutputs([g.matmul(x, g.param("proj.weight", [8, 8]))])
let c = newSession().compile(g)
c.setParam("proj.weight", exact)
let y = c.run({"x": newSeqWith(8, 1.0)})[0]
echo "gguf -> graph: ", y[0 ..< 3].mapIt(&"{it:.2f}").join(" ")

let st = tmp / "rlx-example-nim.safetensors"
var header = """{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}"""
header.add ' '.repeat((8 - header.len mod 8) mod 8)
let s = newFileStream(st, fmWrite)
s.write(uint64(header.len))
s.write(header)
for v in w: s.write(float32(v))
s.close()
let loaded = loadWeights("safetensors", st)
echo "safetensors: [proj.weight] exact: ", loaded["proj.weight"] == w

let packed = quantize(w, "Q8_0")
echo &"quantize q8_0: {w.len * 4} -> {packed.len} bytes, round trip {dequantize(packed, \"Q8_0\", count = some(w.len)).len} values"
removeFile(gguf)
removeFile(st)
echo "argmax: ", argmax([0.1, 2, 0.3])
