// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: three ranks as worker threads on loopback (each a Node
// Worker); across machines, one rank per host with the same peers.
// Collectives, pipelined inference (a stage shipped per worker), data-parallel
// training.
//
//   node examples/distributed.js

const fs = require("fs");
const os = require("os");
const path = require("path");
const { Worker, isMainThread, workerData, parentPort } = require("worker_threads");
const rlx = require("..");

const WORLD = 3;
const PER_RANK = 16;

if (!isMainThread) {
  // A worker rank: runs whatever the coordinator ships.
  const { rank, peers } = workerData;
  const g = rlx.Group.connect(rank, WORLD, { options: { peers, device: "cpu" } });
  g.allReduce([rank + 1], { op: "sum" });
  g.allGather([rank + 1]);
  g.broadcast(0, [0]); // the root's length; it overwrites
  g.serveWorker({ options: { steps: 2 } });
  parentPort.postMessage(g.serveTrainer().params.w);
  return;
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "rlx-dist-node-"));
const blob = (name, values) => {
  const p = path.join(tmp, name);
  fs.writeFileSync(p, Buffer.from(Float32Array.from(values).buffer));
  return `file://${p}`;
};
const matvec = (x, w) => [0, 1, 2, 3].map((j) => Math.max(0, [0, 1, 2, 3].reduce((s, i) => s + x[i] * w[i * 4 + j], 0)));

const stageW = {
  1: Array.from({ length: 16 }, (_, i) => ((i * 5) % 7) / 7 - 0.3),
  2: Array.from({ length: 16 }, (_, i) => ((i * 3) % 5) / 5 - 0.2),
};
const wTrue = [1.5, -2, 0.5];
const xAll = Array.from({ length: PER_RANK * WORLD * 3 }, (_, i) => ((i * 37) % 23) / 11.5 - 1);
const yAll = Array.from({ length: PER_RANK * WORLD }, (_, i) => wTrue.reduce((s, wk, k) => s + xAll[i * 3 + k] * wk, 0));
const spec = (rank, uris) => ({
  params: [{ name: "w", uri: uris.w }],
  data: [
    { input: "X", uri: uris.X, elem: 3, shard_start: rank * PER_RANK, shard_len: PER_RANK },
    { input: "y", uri: uris.y, elem: 1, shard_start: rank * PER_RANK, shard_len: PER_RANK },
  ],
  batch: 4, lr_per_epoch: new Array(40).fill(0.2), device: "cpu",
});

const peers = rlx.loopbackPeers(WORLD);
const workers = [1, 2].map((rank) => new Promise((resolve, reject) => {
  const t = new Worker(__filename, { workerData: { rank, peers } });
  t.on("message", resolve);
  t.on("error", reject);
}));

const g = rlx.Group.connect(0, WORLD, { options: { peers, device: "cpu" } });
console.log(`mesh: rank ${g.rank()} of ${g.world()}, leader: ${g.isLeader()}`);
console.log(`all_reduce sum: ${Array.from(g.allReduce([1], { op: "sum" }))}`);
console.log(`all_gather: ${Array.from(g.allGather([1]))}`);
console.log(`broadcast: ${Array.from(g.broadcast(0, [42]))}`);
console.log(`workers: ${g.collectCaps().map((c) => `rank ${c.rank} on ${c.platform}`).join(", ")}`);

const stage = new rlx.Graph({ name: "stage" });
stage.setOutputs([stage.relu(stage.matmul(stage.input("x", [1, 4]), stage.param("w", [4, 4])))]);
for (let r = 1; r < WORLD; r++) g.shipStage(r, stage, { device: "cpu", weights: { w: blob(`w${r}.bin`, stageW[r]) } });
for (const xv of [[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]]) {
  g.sendActivation(1, xv);
  const yv = g.recvActivation(WORLD - 1);
  const want = matvec(matvec(xv, stageW[1]), stageW[2]);
  console.log(`pipeline max|Δ| vs local = ${Math.max(...want.map((v, i) => Math.abs(v - yv[i]))).toExponential(1)}`);
}

g.collectCaps();
const lr = new rlx.Graph({ name: "lr" });
const X = lr.input("X", [4, 3]);
const Y = lr.input("y", [4, 1]);
const wv = lr.param("w", [3, 1]);
const d = lr.sub(lr.matmul(X, wv), Y);
lr.setOutputs([lr.mean(lr.mul(d, d), [0, 1])]);
const bwd = rlx.grad(lr, [wv]);
const uris = { X: blob("X.bin", xAll), y: blob("y.bin", yAll), w: blob("w0.bin", [0, 0, 0]) };
for (let r = 1; r < WORLD; r++) g.shipTrain(r, bwd, spec(r, uris));
const mine = g.runTrain(bwd, spec(0, uris));
console.log(`training loss: ${mine.metrics.first_loss.toFixed(4)} -> ${mine.metrics.last_loss.toExponential(2)}`);
const w0 = mine.params.w;
console.log(`rank 0 w: ${w0.map((v) => v.toFixed(3)).join(" ")} (true: 1.5 -2 0.5)`);
Promise.all(workers).then((results) => {
  results.forEach((theirs, i) => {
    console.log(`rank ${i + 1} w agrees: ${theirs.every((v, k) => Math.abs(v - w0[k]) < 1e-5)}`);
  });
  fs.rmSync(tmp, { recursive: true });
});
