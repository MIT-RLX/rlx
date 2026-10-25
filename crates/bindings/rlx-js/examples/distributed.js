// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: a mesh of rlx nodes. A script is one thread, so it is one rank;
// start one process per rank with the same peer list (here three on loopback;
// across machines, one per host):
//
//   rlx-js examples/distributed.js 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   rlx-js examples/distributed.js 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   rlx-js examples/distributed.js 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
//
// Rank 0 coordinates: collectives, then a pipelined inference (it ships each
// worker a stage; the worker compiles it on its own hardware and loads its
// weights locally), then data-parallel training. With no arguments it is a
// mesh of one.

const rank = Number(scriptArgs[0] || 0);
const world = Number(scriptArgs[1] || 1);
const peers = scriptArgs[2] ? scriptArgs[2].split(",") : undefined;
const tmp = rlx.env("TMPDIR") || ".";
const PER_RANK = 16;

const g = rlx.Group.connect(rank, world, { peers, device: "cpu" });
const mine = new Float32Array([rank + 1]);
const sum = g.allReduce(mine, "sum");
const gathered = g.allGather(mine);
const shared = g.broadcast(0, new Float32Array([rank === 0 ? 42 : 0])); // root's length
if (rank === 0) {
  console.log(`mesh: rank ${g.rank()} of ${g.world()}, leader: ${g.isLeader()}`);
  console.log(`all_reduce sum: ${Array.from(sum)}`);
  console.log(`all_gather: ${Array.from(gathered)}`);
  console.log(`broadcast: ${Array.from(shared)}`);
}
if (world === 1) {
  console.log("a mesh of one: pass `rank world peers` to start more ranks");
} else if (rank !== 0) {
  // A worker runs whatever the coordinator ships.
  g.serveWorker({ steps: 2 });
  const w = g.serveTrainer().params.w;
  console.log(`rank ${rank} w: ${Array.from(w, (v) => v.toFixed(3)).join(" ")}`);
} else {
  const blob = (name, values) => {
    const path = `${tmp}/rlx-dist-js-${name}`;
    rlx.writeFile(path, new Uint8Array(Float32Array.from(values).buffer));
    return `file://${path}`;
  };
  const stageW = {
    1: Float32Array.from({ length: 16 }, (_, i) => ((i * 5) % 7) / 7 - 0.3),
    2: Float32Array.from({ length: 16 }, (_, i) => ((i * 3) % 5) / 5 - 0.2),
  };

  // Pipelined inference: stage r is relu(x · w_r).
  const caps = g.collectCaps();
  console.log(`workers: ${caps.map((c) => `rank ${c.rank} on ${c.platform}`).join(", ")}`);
  const stage = new rlx.Graph("stage");
  stage.setOutputs([stage.relu(stage.matmul(stage.input("x", [1, 4]), stage.param("w", [4, 4])))]);
  for (let r = 1; r < world; r++) {
    g.shipStage(r, stage, { device: "cpu", weights: { w: blob(`w${r}.bin`, stageW[((r - 1) % 2) + 1]) } });
  }
  for (const x of [[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]]) {
    g.sendActivation(1, new Float32Array(x));
    const y = g.recvActivation(world - 1);
    console.log(`pipeline: ${Array.from(y, (v) => v.toFixed(3)).join(" ")}`);
  }

  // Data-parallel training of y = X·w_true, every rank on its shard.
  g.collectCaps();
  const lr = new rlx.Graph("lr");
  const X = lr.input("X", [4, 3]);
  const Y = lr.input("y", [4, 1]);
  const wv = lr.param("w", [3, 1]);
  const d = lr.sub(lr.matmul(X, wv), Y);
  lr.setOutputs([lr.mean(lr.mul(d, d), [0, 1])]);
  const bwd = rlx.grad(lr, [wv]);
  const wTrue = [1.5, -2, 0.5];
  const n = PER_RANK * world;
  const xAll = Float32Array.from({ length: n * 3 }, (_, i) => ((i * 37) % 23) / 11.5 - 1);
  const yAll = Float32Array.from({ length: n }, (_, i) => wTrue.reduce((s, wk, k) => s + xAll[i * 3 + k] * wk, 0));
  const uris = { X: blob("X.bin", xAll), y: blob("y.bin", yAll), w: blob("w0.bin", [0, 0, 0]) };
  const spec = (r) => ({
    params: [{ name: "w", uri: uris.w }],
    data: [
      { input: "X", uri: uris.X, elem: 3, shardStart: r * PER_RANK, shardLen: PER_RANK },
      { input: "y", uri: uris.y, elem: 1, shardStart: r * PER_RANK, shardLen: PER_RANK },
    ],
    batch: 4, lrPerEpoch: new Array(40).fill(0.2), device: "cpu",
  });
  for (let r = 1; r < world; r++) g.shipTrain(r, bwd, spec(r));
  const result = g.runTrain(bwd, spec(0));
  console.log(`training loss: ${result.metrics.firstLoss.toFixed(4)} -> ${result.metrics.lastLoss.toExponential(2)}`);
  console.log(`rank 0 w: ${Array.from(result.params.w, (v) => v.toFixed(3)).join(" ")} (true: 1.5 -2 0.5)`);
}
