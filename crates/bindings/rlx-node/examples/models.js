// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib node examples/models.js

const fs = require("fs");
const os = require("os");
const path = require("path");
const rlx = require("..");

const g = new rlx.Graph({ name: "affine" });
const x = g.input("x", [1, 2]);
g.setOutputs([g.matmul(x, g.param("w", [2, 2]))]);
const file = path.join(os.tmpdir(), `rlx-example-node-${process.pid}.rlxp`);
rlx.saveModel(g, file, { w: [2, 0, 0, 3] }, { model: { task: "graph", family: "affine", outputs: ["y"] } });

const m = rlx.Model.open(file, { options: { device: "cpu" } });
const info = m.info();
console.log(`model: ${info.family} (${info.task}) on ${info.device}`);
const [y] = m.run({ x: [1, 1] });
console.log(`model run: ${y[0].toFixed(3)} ${y[1].toFixed(3)}`);
m.close();
fs.rmSync(file);

const lm = rlx.Model.open("demo:count");
const lmInfo = lm.info();
console.log(`plugin model: ${lmInfo.family} (${lmInfo.task})`);
console.log(`model text: ${lm.generateText("c", { options: { max_tokens: 4 } })}`);
console.log(`model ids: ${Array.from(lm.generate([2], { maxTokens: 3 })).join(" ")}`);

// By task: a chat, speech out and back in, embeddings.
console.log(`chat: ${lm.chat([{ role: "user", content: "c" }], { options: { max_tokens: 4 } })}`);
const { audio, sampleRate } = rlx.Model.open("demo:voice").synthesize("hello");
console.log(`voice: ${audio.length} samples at ${sampleRate} Hz`);
console.log(`heard: ${rlx.Model.open("demo:speech").transcribe(audio, { sampleRate })}`);
const rows = rlx.Model.open("demo:embed").embed(["hello", "sky"]);
console.log(`embed: ${rows.flat().join(" ")}`);
