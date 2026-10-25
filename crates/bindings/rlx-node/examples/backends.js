// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//
//   node examples/backends.js

const rlx = require("..");

function model() {
  const g = new rlx.Graph({ name: "mlp" });
  const x = g.input("x", [4, 16]);
  g.setOutputs([g.tanh(g.matmul(x, g.param("w", [16, 16])))]);
  return g;
}
const x = { x: Float32Array.from({ length: 64 }, (_, i) => (i % 7) / 7) };
const w = Float32Array.from({ length: 256 }, (_, i) => (i % 5) / 10 - 0.2);

const available = rlx.devices();
console.log(`devices: ${available.join(" ")}`);
console.log(`metal available: ${rlx.isAvailable("metal")}`);
console.log(`fastest for graph: ${rlx.fastestDeviceFor(model())}`);
let reference = null;
for (const device of available) {
  const c = new rlx.Session({ device }).compile(model());
  c.setParam("w", w);
  const [y] = c.run(x);
  reference = reference || y;
  console.log(`session ${device}: max|Δ| vs first = ${Math.max(...y.map((v, i) => Math.abs(v - reference[i]))).toExponential(2)}`);
}
const flex = new rlx.FlexibleSession({ policy: { prefer: ["metal", "cpu"] } });
console.log(`flexible session picked: ${flex.compile(model()).device()}`);
const runner = new rlx.Runner(model());
runner.setParam("w", w);
console.log(`runner warmed: ${runner.warmAll().join(" ")}`);
for (const row of runner.benchmark(x, { runs: 3 })) console.log(`  bench ${row.device}: exec_ms=${row.exec_ms}`);
console.log(`runner ran on: ${runner.run(x).device}`);
console.log(`run_try chain: ${runner.runTry(["metal", "cpu"], x).device}`);
const router = new rlx.Router(model());
router.setParam("w", w);
console.log(`router ran on: ${router.runChain(x).device}`);
