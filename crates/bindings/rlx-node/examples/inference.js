// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//
//   node examples/inference.js [device]
//
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

const rlx = require("..");

const device = process.argv[2] || "cpu";
const weights = (n, seed) => Float32Array.from({ length: n }, (_, i) => ((i * 7919 + seed * 104729) % 1000) / 1000 - 0.5);

const g = new rlx.Graph({ name: "classifier" });
const x = g.input("x", [1, 4]);
const hidden = g.relu(g.matmul(x, g.param("w1", [4, 8])));
g.setOutputs([g.softmax(g.matmul(hidden, g.param("w2", [8, 3])))]);

const model = new rlx.Session({ device }).compile(g); // compile consumes the graph
console.log(`compiled on: ${model.device()}`);
model.setParams({ w1: weights(32, 1), w2: weights(24, 2) });
const [probs] = model.run({ x: [0.5, -1, 2, 0.25] });
console.log(`probs: ${Array.from(probs, (p) => p.toFixed(4)).join(" ")}`);
console.log(`probs sum: ${probs.reduce((a, b) => a + b, 0).toFixed(4)}`);
console.log(`output shapes: ${JSON.stringify(model.outputShapes())} dtypes: ${JSON.stringify(model.outputDtypes())}`);
