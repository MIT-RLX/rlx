// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Browser glue: forward + backward on CPU, WebGPU (--webgpu), and WebGL2
// (--webgl). `init` (default export) loads the wasm module; the start hook
// installs the panic handler. Each GPU path is feature-gated in the bundle, so
// we check for the exported functions before calling them.

import init, * as rlx from "./pkg/rlx_web.js";

const $ = (id) => document.getElementById(id);
const set = (id, text, cls) => {
  const el = $(id);
  el.textContent = text;
  if (cls) el.className = "v " + cls;
};
const fmt = (a, n = 4) => "[" + Array.from(a).map((v) => v.toFixed(n)).join(", ") + "]";
const maxDiff = (a, b) =>
  Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

// Small MLP: in=3, hidden=4, out=2, batch=1.
const D = { in: 3, hid: 4, out: 2 };
const x = new Float32Array([0.4, -0.2, 0.9]);
const target = new Float32Array([1.0, -0.5]);
const mk = (n, f) => Float32Array.from({ length: n }, (_, i) => f(i));
const w1 = mk(D.in * D.hid, (i) => 0.1 * (i % 5) - 0.2);
const b1 = new Float32Array(D.hid);
const w2 = mk(D.hid * D.out, (i) => 0.05 * i - 0.1);
const b2 = new Float32Array(D.out);

// CPU reference results, used to check the GPU backends agree.
let cpuY = null;
let cpuG = null;

function runCpu() {
  cpuY = rlx.mlp_forward(x, D.in, D.hid, D.out, w1, b1, w2, b2);
  $("cpu-fwd").textContent = `x = ${fmt(x)}\ny = relu(x·W1+b1)·W2+b2 = ${fmt(cpuY)}`;

  cpuG = rlx.mlp_grads(x, target, D.in, D.hid, D.out, w1, b1, w2, b2);
  const gW1 = cpuG.slice(1, 1 + D.in * D.hid);
  $("cpu-bwd").textContent =
    `target = ${fmt(target)}\nloss   = ${cpuG[0].toFixed(6)}\n∂loss/∂W1 = ${fmt(gW1)}\n(+ ∂b1, ∂W2, ∂b2)`;

  let [tw1, tb1, tw2, tb2] = [w1, b1, w2, b2];
  const l0 = rlx.mlp_loss(x, target, D.in, D.hid, D.out, tw1, tb1, tw2, tb2);
  for (let s = 0; s < 100; s++) {
    const up = rlx.mlp_train_step(x, target, D.in, D.hid, D.out, tw1, tb1, tw2, tb2, 0.05);
    tw1 = up.slice(0, D.in * D.hid);
    tb1 = up.slice(D.in * D.hid, D.in * D.hid + D.hid);
    tw2 = up.slice(D.in * D.hid + D.hid, D.in * D.hid + D.hid + D.hid * D.out);
    tb2 = up.slice(up.length - D.out);
  }
  const l1 = rlx.mlp_loss(x, target, D.in, D.hid, D.out, tw1, tb1, tw2, tb2);
  $("cpu-train").textContent = `loss: ${l0.toFixed(6)}  →  ${l1.toFixed(6)}  (100 SGD steps, lr=0.05)`;
}

function reportGpu(id, label, y, g, ms) {
  $(id).textContent =
    `forward  y = ${fmt(y)}\n` +
    `         max|${label}−cpu| = ${maxDiff(y, cpuY).toExponential(2)}\n` +
    `backward loss = ${g[0].toFixed(6)}\n` +
    `         max|${label}−cpu grads| = ${maxDiff(g, cpuG).toExponential(2)}\n` +
    `time = ${ms} ms`;
}

async function runWebgpu() {
  if (typeof rlx.init_webgpu !== "function") {
    set("webgpu", "not built (add --webgpu)", "warn");
    $("gpu").textContent = "—";
    return;
  }
  let ok = false;
  try {
    ok = await rlx.init_webgpu();
  } catch (e) {
    set("webgpu", "error: " + e, "err");
  }
  if (!ok) {
    set("webgpu", "no adapter (browser lacks WebGPU?)", "warn");
    return;
  }
  set("webgpu", "available", "ok");
  const t0 = performance.now();
  const y = await rlx.mlp_forward_gpu(x, D.in, D.hid, D.out, w1, b1, w2, b2);
  const g = await rlx.mlp_grads_gpu(x, target, D.in, D.hid, D.out, w1, b1, w2, b2);
  reportGpu("gpu", "gpu", y, g, (performance.now() - t0).toFixed(2));
}

function runWebgl() {
  if (typeof rlx.mlp_forward_webgl !== "function") {
    set("webgl", "not built (add --webgl)", "warn");
    $("glout").textContent = "—";
    return;
  }
  try {
    const t0 = performance.now();
    const y = rlx.mlp_forward_webgl(x, D.in, D.hid, D.out, w1, b1, w2, b2);
    const g = rlx.mlp_grads_webgl(x, target, D.in, D.hid, D.out, w1, b1, w2, b2);
    set("webgl", "available", "ok");
    reportGpu("glout", "gl", y, g, (performance.now() - t0).toFixed(2));
  } catch (e) {
    set("webgl", "error: " + e, "err");
    $("glout").textContent = String(e);
  }
}

function runTransformer() {
  // A real decoder-only transformer (RMSNorm + RoPE + causal MHA + SwiGLU)
  // with deterministic synthesized weights — runs end-to-end on the CPU path.
  const cfg = { vocab: 64, dim: 64, layers: 2, heads: 4, headDim: 16, ffn: 128, seed: 1 };
  const tokens = new Float32Array([5, 9, 2, 7, 1, 8, 3]);
  const t0 = performance.now();
  const logits = rlx.transformer_next_logits(
    tokens, cfg.vocab, cfg.dim, cfg.layers, cfg.heads, cfg.headDim, cfg.ffn, cfg.seed,
  );
  const ms = (performance.now() - t0).toFixed(2);
  // greedy next token = argmax
  let arg = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[arg]) arg = i;
  $("tf").textContent =
    `config: dim=${cfg.dim}, layers=${cfg.layers}, heads=${cfg.heads}, head_dim=${cfg.headDim}, vocab=${cfg.vocab}\n` +
    `input tokens (${tokens.length}) = [${Array.from(tokens)}]\n` +
    `next-token logits (${logits.length}) = ${fmt(logits.slice(0, 8))} …\n` +
    `greedy next token = ${arg}   (forward ${ms} ms)`;
}

async function main() {
  await init();
  set("status", "wasm loaded", "ok");
  set("backend", rlx.backend());
  runCpu();
  runTransformer();
  await runWebgpu();
  runWebgl();
}

main().catch((e) => {
  set("status", "failed: " + e, "err");
  console.error(e);
});
