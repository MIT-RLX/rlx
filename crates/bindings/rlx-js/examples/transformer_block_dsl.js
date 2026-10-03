// The same pre-norm transformer block as transformer_block.js, written with the
// chaining DSL — and checked against the imperative build for bit-equality.
//
//   rlx-js examples/transformer_block_dsl.js
//
// The point is readability. Threading node ids reads inside-out: the first
// operation applied is the innermost token, so you unwrap the expression
// backwards. A chain reads in the order the data flows.

const B = 1, S = 8, HEADS = 4, HEAD_DIM = 16, D = HEADS * HEAD_DIM, FF = 2 * D;
const DEVICE = rlx.devices().includes("metal") ? "metal" : "cpu";

// ── DSL ──────────────────────────────────────────────────────
function buildDsl() {
  const m = rlx.dsl("block");
  const x = m.input("x", [B, S, D]);
  const cos = m.input("cos", [S, HEAD_DIM / 2]);
  const sin = m.input("sin", [S, HEAD_DIM / 2]);
  const p = (name, dims) => m.param(name, dims);

  const normed = x.rmsNorm(p("attn_norm", [D]), p("attn_norm_b", [D]), 1e-6);
  const q = normed.matmul(p("wq", [D, D])).rope(cos, sin, HEAD_DIM);
  const k = normed.matmul(p("wk", [D, D])).rope(cos, sin, HEAD_DIM);
  const v = normed.matmul(p("wv", [D, D]));

  const afterAttn = x.add(
    q.attentionKind(k, v, HEADS, HEAD_DIM, "causal").matmul(p("wo", [D, D])),
  );

  const h = afterAttn.rmsNorm(p("ffn_norm", [D]), p("ffn_norm_b", [D]), 1e-6);
  const mlp = h.matmul(p("w_gate", [D, FF])).silu()
               .mul(h.matmul(p("w_up", [D, FF])))
               .matmul(p("w_down", [FF, D]));

  m.output(afterAttn.add(mlp));
  return m;
}

// ── imperative, for comparison ───────────────────────────────
function buildImperative() {
  const g = new rlx.Graph("block");
  const x = g.input("x", [B, S, D], "f32");
  const cos = g.input("cos", [S, HEAD_DIM / 2], "f32");
  const sin = g.input("sin", [S, HEAD_DIM / 2], "f32");
  const p = (name, dims) => g.param(name, dims, "f32");

  const normed = g.rmsNorm(x, p("attn_norm", [D]), p("attn_norm_b", [D]), 1e-6);
  const q = g.rope(g.matmul(normed, p("wq", [D, D])), cos, sin, HEAD_DIM);
  const k = g.rope(g.matmul(normed, p("wk", [D, D])), cos, sin, HEAD_DIM);
  const v = g.matmul(normed, p("wv", [D, D]));
  const afterAttn = g.add(
    x,
    g.matmul(g.attentionKind(q, k, v, HEADS, HEAD_DIM, "causal"), p("wo", [D, D])),
  );
  const h = g.rmsNorm(afterAttn, p("ffn_norm", [D]), p("ffn_norm_b", [D]), 1e-6);
  const mlp = g.matmul(
    g.mul(g.silu(g.matmul(h, p("w_gate", [D, FF]))), g.matmul(h, p("w_up", [D, FF]))),
    p("w_down", [FF, D]),
  );
  g.setOutputs([g.add(afterAttn, mlp)]);
  return g;
}

// ── deterministic weights, shared by both ────────────────────
function fill(n, seed) {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = ((s >>> 8) / 8388608 - 1) * 0.05;
  }
  return out;
}
const weights = {
  attn_norm: new Float32Array(D).fill(1), attn_norm_b: new Float32Array(D),
  ffn_norm: new Float32Array(D).fill(1), ffn_norm_b: new Float32Array(D),
  wq: fill(D * D, 1), wk: fill(D * D, 2), wv: fill(D * D, 3), wo: fill(D * D, 4),
  w_gate: fill(D * FF, 5), w_up: fill(D * FF, 6), w_down: fill(FF * D, 7),
};
const inputs = {
  x: fill(B * S * D, 11),
  cos: new Float32Array(S * (HEAD_DIM / 2)).fill(1),
  sin: new Float32Array(S * (HEAD_DIM / 2)).fill(0),
};

function run(graph) {
  const c = new rlx.Session(DEVICE).compile(graph);
  c.setParams(weights);
  return c.run(inputs)[0];
}

const fromDsl = run(buildDsl());
const fromImperative = run(buildImperative());

let worst = 0;
for (let i = 0; i < fromDsl.length; i++) {
  worst = Math.max(worst, Math.abs(fromDsl[i] - fromImperative[i]));
}
console.log(`device        ${DEVICE}`);
console.log(`dsl out[0..3] ${Array.from(fromDsl.slice(0, 3)).map((v) => v.toFixed(6))}`);
console.log(`max|Δ| vs imperative build: ${worst === 0 ? "0 (bit-identical)" : worst.toExponential(2)}`);

// The DSL is a spelling, not a second graph builder: same ops, same ids.
const dslGraph = buildDsl(), impGraph = buildImperative();
console.log(`nodes: dsl=${dslGraph.nodeCount()} imperative=${impGraph.nodeCount()}`);
console.log(`ops identical: ${JSON.stringify(dslGraph.opKinds()) === JSON.stringify(impGraph.opKinds())}`);
