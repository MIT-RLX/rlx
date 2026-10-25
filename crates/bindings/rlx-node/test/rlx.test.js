// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The Node binding against the real library: `node --test test/`.

const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const rlx = require("..");

function lossGraph() {
  const g = new rlx.Graph({ name: "mse" });
  const x = g.input("x", [1, 2]);
  const y = g.input("y", [1, 2]);
  const w = g.param("w", [2, 2]);
  const d = g.sub(g.matmul(x, w), y);
  g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);
  return { g, w };
}

test("inference", () => {
  const { g } = lossGraph();
  const m = new rlx.Session().compile(g);
  assert.ok(g.isConsumed());
  m.setParam("w", [1, 0, 0, 1]);
  assert.strictEqual(m.run({ x: [1, 2], y: [0, 0] })[0][0], 2.5);
  assert.strictEqual(m.device(), "cpu");
  m.close();
});

test("errors carry the kind", () => {
  assert.throws(() => new rlx.Graph().relu(42), (e) => e instanceof rlx.RlxError && e.kind === "range");
  assert.throws(() => new rlx.Session({ device: "no-such-device" }), rlx.RlxError);
});

test("training", () => {
  const { g, w } = lossGraph();
  const bwd = new rlx.Session().compile(rlx.grad(g, [w]));
  bwd.setParam("w", [1, 0, 0, 1]);
  assert.deepStrictEqual(Array.from(bwd.run({ x: [1, 2], y: [0, 0], d_output: [1] })[1]), [1, 2, 2, 4]);
  const t = new rlx.Trainer(g.clone(), ["w"], { w: [0, 0, 0, 0] }, { options: { optimizer: { kind: "adamw", lr: 0.1 } } });
  const r = t.run(300, () => ({ x: [1, 1], y: [1, -1] }), { onStep: (s, l) => l > 1e-6 });
  assert.ok(r.lastLoss < 1e-4);
  const ck = path.join(os.tmpdir(), `rlx-node-test-${process.pid}.gguf`);
  assert.ok(t.save(ck) >= 1);
  fs.rmSync(ck);
  const opt = new rlx.Optimizer({ kind: "sgd", options: { lr: 0.5 } });
  const p = { p: new Float32Array([1, -1]) };
  opt.step(p, { p: [2, -2] });
  assert.deepStrictEqual(Array.from(p.p), [0, 0]);
});

test("loading", () => {
  const file = path.join(os.tmpdir(), `rlx-node-${process.pid}.gguf`);
  const w = new Float32Array([0.5, -1, 2, 0.25]);
  const wr = new rlx.GgufWriter();
  wr.addF32("w", [4], "F32", w);
  assert.strictEqual(wr.write(file, { architecture: "demo" }), 1);
  const f = rlx.Gguf.open(file);
  assert.strictEqual(f.architecture(), "demo");
  assert.deepStrictEqual(Array.from(f.tensor("w")), Array.from(w));
  assert.strictEqual(f.info("w").dtype, "F32");
  fs.rmSync(file);
});

test("backends and a one-rank mesh", () => {
  assert.ok(rlx.devices().includes("cpu") && rlx.isAvailable("cpu"));
  const { g } = lossGraph();
  const r = new rlx.Runner(g, { policy: { only: ["cpu"] } });
  r.setParam("w", [1, 0, 0, 1]);
  const res = r.run({ x: [1, 2], y: [0, 0] });
  assert.strictEqual(res.device, "cpu");
  assert.strictEqual(res.outputs[0][0], 2.5);
  const mesh = rlx.Group.connect(0, 1);
  assert.deepStrictEqual(Array.from(mesh.allReduce([2, 3], { op: "max" })), [2, 3]);
});
