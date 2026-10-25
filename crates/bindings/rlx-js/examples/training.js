// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer (forward + backward + optimizer
// compiled once), checkpoint and resume it, then the same fit by hand with
// `grad` and an Optimizer.
//
//   rlx-js examples/training.js [device]

const device = scriptArgs[0] || "cpu";
const W_TRUE = [2, -1, 0.5];
const BATCH = 8;

function regression() {
  const g = new rlx.Graph("regression");
  const x = g.input("x", [BATCH, 3]);
  const y = g.input("y", [BATCH, 1]);
  const w = g.param("w", [3, 1]);
  const d = g.sub(g.matmul(x, w), y);
  g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);
  return { g, w };
}

function data(step) {
  const x = new Float32Array(BATCH * 3);
  const y = new Float32Array(BATCH);
  for (let i = 0; i < BATCH; i++) {
    for (let k = 0; k < 3; k++) x[i * 3 + k] = ((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1;
    y[i] = W_TRUE.reduce((s, wk, k) => s + x[i * 3 + k] * wk, 0);
  }
  return { x, y };
}

const fmt = (v) => Array.from(v, (x) => x.toFixed(3)).join(" ");
const { g, w } = regression();

// ── Trainer: `run` pulls batches from a callback ────────────────────────────
const trainer = new rlx.Trainer(g.clone(), {
  wrt: ["w"], init: { w: new Float32Array(3) }, device,
  optimizer: { kind: "adamw", lr: 0.1 },
});
const first = trainer.evaluate(data(0));
trainer.run({
  steps: 400,
  batch: (s) => data(s),
  lr: (s) => 0.1 * 0.5 * (1 + Math.cos((Math.PI * s) / 400)), // cosine decay
  onStep: (s, loss) => loss > 1e-6, // stop once the loss is tiny
});
console.log(`trainer loss: ${first.toFixed(4)} -> ${trainer.evaluate(data(0)).toFixed(6)} in ${trainer.steps()} steps`);
console.log(`trainer w: ${fmt(trainer.params().w)}`);

// Checkpoint (GGUF: params + optimizer state), restore into a fresh trainer.
const path = `${rlx.env("TMPDIR") || "."}/rlx-example-train-js.gguf`;
const written = trainer.save(path);
const restored = new rlx.Trainer(g.clone(), { wrt: ["w"], init: { w: new Float32Array(3) }, device });
const report = restored.load(path);
console.log(`checkpoint: ${written} tensors, restored step ${report.step}`);
rlx.removeFile(path);

// Freezing: a frozen param keeps its value through steps.
restored.freeze(["w"]);
const before = Array.from(restored.params().w).join();
restored.step(data(1));
console.log(`frozen: ${JSON.stringify(restored.frozen())} unchanged: ${Array.from(restored.params().w).join() === before}`);
restored.unfreeze();

// ── By hand: grad → compile → Optimizer ─────────────────────────────────────
const backward = new rlx.Session(device).compile(rlx.grad(g, [w])); // outputs [loss, dw]
const opt = new rlx.Optimizer("sgd", { lr: 0.2 });
let params = new Float32Array(3);
let loss = 0;
for (let step = 0; step < 300; step++) {
  backward.setParam("w", params);
  const [l, dw] = backward.run({ ...data(step), d_output: new Float32Array([1]) });
  loss = l[0];
  params = opt.step({ w: { param: params, grad: dw } }).w;
}
console.log(`manual loss: ${loss.toFixed(6)} w: ${fmt(params)}`);
