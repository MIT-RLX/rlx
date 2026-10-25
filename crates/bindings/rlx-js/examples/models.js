// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name.
//
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib rlx-js examples/models.js
//
// A graph saved as an .rlxp package runs anywhere; a plugin's models (a
// downstream collection's LLMs, or the reference plugin's demo:count) load
// through RLX_MODEL_PLUGINS or rlx.loadModelPlugin. The Swift, Python and C
// examples of the same name do the same steps.

const g = new rlx.Graph("affine");
const x = g.input("x", [1, 2]);
g.setOutputs([g.matmul(x, g.param("w", [2, 2]))]);
const path = `${rlx.env("TMPDIR") || "."}/rlx-example-js.rlxp`;
rlx.saveModel(g, path, { w: [2, 0, 0, 3] }, { task: "graph", family: "affine", outputs: ["y"] });

const m = rlx.Model.open(path, { device: "cpu" });
const info = m.info();
console.log(`model: ${info.family} (${info.task}) on ${info.device}`);
const [y] = m.run({ x: new Float32Array([1, 1]) });
console.log(`model run: ${y[0].toFixed(3)} ${y[1].toFixed(3)}`);
rlx.removeFile(path);

const lm = rlx.Model.open("demo:count");
const lmInfo = lm.info();
console.log(`plugin model: ${lmInfo.family} (${lmInfo.task})`);
console.log(`model text: ${lm.generateText("c", { max_tokens: 4 })}`);
console.log(`model ids: ${Array.from(lm.generate([2], 3)).join(" ")}`);

// By task: a chat, speech out and back in, embeddings.
console.log(`chat: ${lm.chat([{ role: "user", content: "c" }], { max_tokens: 4 })}`);
const { audio, sampleRate } = rlx.Model.open("demo:voice").synthesize("hello");
console.log(`voice: ${audio.length} samples at ${sampleRate} Hz`);
console.log(`heard: ${rlx.Model.open("demo:speech").transcribe(audio, sampleRate)}`);
const rows = rlx.Model.open("demo:embed").embed(["hello", "sky"]);
console.log(`embed: ${rows.flatMap((r) => Array.from(r)).join(" ")}`);
