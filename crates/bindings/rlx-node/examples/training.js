// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with `grad` and an Optimizer.
//
//   node examples/training.js [device]

const fs = require("fs");
const os = require("os");
const path = require("path");
const rlx = require("..");

const device = process.argv[2] || "cpu";
const W_TRUE = [2, -1, 0.5];
const BATCH = 8;

function data(step) {
  const x = new Float32Array(BATCH * 3);
  const y = new Float32Array(BATCH);
  for (let i = 0; i < BATCH; i++) {
    for (let k = 0; k < 3; k++) x[i * 3 + k] = ((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1;
    y[i] = W_TRUE.reduce((s, wk, k) => s + x[i * 3 + k] * wk, 0);
  }
  return { x, y };
}
const fmt = (v) => Array.from(v, (a) => a.toFixed(3)).join(" ");

const g = new rlx.Graph({ name: "regression" });
const x = g.input("x", [BATCH, 3]);
const y = g.input("y", [BATCH, 1]);
const w = g.param("w", [3, 1]);
const d = g.sub(g.matmul(x, w), y);
g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);

const trainer = new rlx.Trainer(g.clone(), ["w"], { w: [0, 0, 0] }, { options: { device, optimizer: { kind: "adamw", lr: 0.1 } } });
const first = trainer.evaluate(data(0));
trainer.run(400, data, {
  lr: (s) => 0.1 * 0.5 * (1 + Math.cos((Math.PI * s) / 400)),
  onStep: (s, loss) => loss > 1e-6,
});
console.log(`trainer loss: ${first.toFixed(4)} -> ${trainer.evaluate(data(0)).toFixed(6)} in ${trainer.steps()} steps`);
console.log(`trainer w: ${fmt(trainer.params().w)}`);

const ck = path.join(os.tmpdir(), `rlx-example-train-node-${process.pid}.gguf`);
const written = trainer.save(ck);
const restored = new rlx.Trainer(g.clone(), ["w"], { w: [0, 0, 0] }, { options: { device } });
console.log(`checkpoint: ${written} tensors, restored step ${restored.load(ck).step}`);
fs.rmSync(ck);
restored.freeze({ names: ["w"] });
const before = Array.from(restored.params().w).join();
restored.step(data(1));
console.log(`frozen: ${JSON.stringify(restored.frozen())} unchanged: ${Array.from(restored.params().w).join() === before}`);
restored.unfreeze();

const backward = new rlx.Session({ device }).compile(rlx.grad(g, [w])); // outputs [loss, dw]
const opt = new rlx.Optimizer({ kind: "sgd", options: { lr: 0.2 } });
const params = { w: new Float32Array(3) };
let loss = 0;
for (let step = 0; step < 300; step++) {
  backward.setParam("w", params.w);
  const [l, dw] = backward.run({ ...data(step), d_output: [1] });
  loss = l[0];
  opt.step(params, { w: dw });
}
console.log(`manual loss: ${loss.toFixed(6)} w: ${fmt(params.w)}`);
