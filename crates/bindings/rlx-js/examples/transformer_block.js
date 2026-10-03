// A full pre-norm transformer block, built in JavaScript and run on every
// backend this binary was compiled with.
//
//   rlx-js examples/transformer_block.js
//
// This is the surface that matters: RMSNorm, RoPE, GQA-shaped attention and a
// SwiGLU MLP are single IR ops, not JS loops — the script assembles the graph
// and the backend does the work.

const B = 1, S = 8, HEADS = 4, HEAD_DIM = 16, D = HEADS * HEAD_DIM, FF = 2 * D;

function block(g) {
  const x = g.input("x", [B, S, D], "f32");
  const cos = g.input("cos", [S, HEAD_DIM / 2], "f32");
  const sin = g.input("sin", [S, HEAD_DIM / 2], "f32");

  const p = (name, dims) => g.param(name, dims, "f32");

  // ── attention ──
  const normed = g.rmsNorm(x, p("attn_norm", [D]), p("attn_norm_b", [D]), 1e-6);
  const q = g.rope(g.matmul(normed, p("wq", [D, D])), cos, sin, HEAD_DIM);
  const k = g.rope(g.matmul(normed, p("wk", [D, D])), cos, sin, HEAD_DIM);
  const v = g.matmul(normed, p("wv", [D, D]));
  const attn = g.attentionKind(q, k, v, HEADS, HEAD_DIM, "causal");
  const afterAttn = g.add(x, g.matmul(attn, p("wo", [D, D])));

  // ── SwiGLU MLP ──
  const h = g.rmsNorm(afterAttn, p("ffn_norm", [D]), p("ffn_norm_b", [D]), 1e-6);
  const gate = g.silu(g.matmul(h, p("w_gate", [D, FF])));
  const up = g.matmul(h, p("w_up", [D, FF]));
  const down = g.matmul(g.mul(gate, up), p("w_down", [FF, D]));

  g.setOutputs([g.add(afterAttn, down)]);
  return { x, cos, sin };
}

// Deterministic pseudo-random fill, so every backend sees identical weights.
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

let reference = null;
for (const device of rlx.devices()) {
  const g = new rlx.Graph("block");
  block(g);

  const t0 = Date.now();
  const compiled = new rlx.Session(device).compile(g);
  compiled.setParams(weights);
  const [y] = compiled.run(inputs);
  const ms = Date.now() - t0;

  if (reference === null) {
    reference = y;
    console.log(`${device.padEnd(7)} ${ms}ms  y[0..4] = ${Array.from(y.slice(0, 4)).map((v) => v.toFixed(6))}`);
  } else {
    let worst = 0;
    for (let i = 0; i < y.length; i++) worst = Math.max(worst, Math.abs(y[i] - reference[i]));
    console.log(`${device.padEnd(7)} ${ms}ms  max|Δ| vs cpu = ${worst.toExponential(2)}`);
  }
}
