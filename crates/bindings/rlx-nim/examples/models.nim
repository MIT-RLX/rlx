# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

## Models: any provider's model, opened by path or id and run by name. A graph
## saved as an .rlxp package runs anywhere; a plugin's models (a downstream
## collection's LLMs, or the reference plugin's demo:count) load through
## RLX_MODEL_PLUGINS.
##   RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib nim r --path:src examples/models.nim

import std/[json, os, sequtils, strformat, strutils]
import rlx

let g = newGraph(name = "affine")
let x = g.input("x", [1, 2])
g.setOutputs([g.matmul(x, g.param("w", [2, 2]))])
let path = getTempDir() / "rlx-example-nim.rlxp"
saveModel(g, path, {"w": @[2.0, 0, 0, 3]},
          model = %*{"task": "graph", "family": "affine", "outputs": ["y"]})

let m = Model.open(path, %*{"device": "cpu"})
let info = m.info()
echo &"""model: {info["family"].getStr} ({info["task"].getStr}) on {info["device"].getStr}"""
let y = m.run({"x": @[1.0, 1]})[0]
echo &"model run: {y[0]:.3f} {y[1]:.3f}"
removeFile(path)

let lm = Model.open("demo:count")
let lmInfo = lm.info()
echo &"""plugin model: {lmInfo["family"].getStr} ({lmInfo["task"].getStr})"""
echo "model text: ", lm.generateText("c", %*{"max_tokens": 4})
echo "model ids: ", lm.generate([2], maxTokens = 3).mapIt($it).join(" ")

# By task: a chat, speech out and back in, embeddings.
echo "chat: ", lm.chat(%*[{"role": "user", "content": "c"}], %*{"max_tokens": 4})
let speech = Model.open("demo:voice").synthesize("hello")
echo &"voice: {speech.audio.len} samples at {speech.sampleRate} Hz"
echo "heard: ", Model.open("demo:speech").transcribe(speech.audio, speech.sampleRate)
var values: seq[string]
for row in Model.open("demo:embed").embed(["hello", "sky"]):
  for v in row: values.add $int(v.getFloat)
echo "embed: ", values.join(" ")
