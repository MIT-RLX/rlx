// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: weights from disk into a compiled graph — a GGUF file (written
// here, f32 and quantized) and a safetensors file — and quantizing in memory.
// With RLX_TOKENIZER=<tokenizer.json> it also tokenizes.
//
//   rlx-js examples/loading.js

const tmp = rlx.env("TMPDIR") || ".";
const w = Float32Array.from({ length: 64 }, (_, i) => (i % 9) / 4 - 1);

// ── GGUF ────────────────────────────────────────────────────────────────────
const ggufPath = `${tmp}/rlx-example-js.gguf`;
rlx.writeGguf(
  ggufPath,
  {
    "proj.weight": { data: w, shape: [8, 8], dtype: "F32" },
    "proj.weight_q8": { data: w, shape: [8, 8], dtype: "Q8_0" },
  },
  { architecture: "demo", metadata: { "demo.layers": 1 } },
);
const f = rlx.openGguf(ggufPath);
console.log(`gguf architecture: ${f.architecture()}`);
console.log(`gguf tensors: ${f.tensorNames().join(" ")}`);
const exact = f.tensor("proj.weight");
const q8 = f.tensor("proj.weight_q8");
const q8err = Math.max(...Array.from(exact, (v, i) => Math.abs(v - q8[i])));
console.log(`gguf f32 exact: ${exact.every((v, i) => v === w[i])} q8 max|Δ|: ${q8err.toFixed(4)}`);

const g = new rlx.Graph("proj");
const x = g.input("x", [1, 8]);
g.setOutputs([g.matmul(x, g.param("proj.weight", [8, 8]))]);
const c = new rlx.Session("cpu").compile(g);
c.setParam("proj.weight", exact);
const [y] = c.run({ x: new Float32Array(8).fill(1) });
console.log(`gguf -> graph: ${Array.from(y.slice(0, 3), (v) => v.toFixed(2)).join(" ")}`);

// ── safetensors → every tensor as f32 ───────────────────────────────────────
const stPath = `${tmp}/rlx-example-js.safetensors`;
let header = JSON.stringify({ "proj.weight": { dtype: "F32", shape: [8, 8], data_offsets: [0, w.byteLength] } });
header += " ".repeat((8 - (header.length % 8)) % 8);
const file = new Uint8Array(8 + header.length + w.byteLength);
new DataView(file.buffer).setUint32(0, header.length, true);
for (let i = 0; i < header.length; i++) file[8 + i] = header.charCodeAt(i);
file.set(new Uint8Array(w.buffer), 8 + header.length);
rlx.writeFile(stPath, file);
const loaded = rlx.loadWeights("safetensors", stPath);
console.log(`safetensors: ${JSON.stringify(Object.keys(loaded))} exact: ${loaded["proj.weight"].every((v, i) => v === w[i])}`);

// ── quantize in memory ──────────────────────────────────────────────────────
const packed = rlx.quantize(w, "Q8_0");
const back = rlx.dequantize(packed, "Q8_0", w.length);
console.log(`quantize q8_0: ${w.byteLength} -> ${packed.length} bytes, round trip ${back.length} values`);
rlx.removeFile(ggufPath);
rlx.removeFile(stPath);

// ── text ────────────────────────────────────────────────────────────────────
const tokPath = rlx.env("RLX_TOKENIZER");
if (tokPath) {
  const tok = rlx.loadTokenizer(tokPath);
  const ids = tok.encode("Hello from JavaScript");
  console.log(`tokenizer: ${ids.length} ids -> ${JSON.stringify(tok.decode(ids))}`);
} else {
  console.log("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)");
}
console.log(`argmax: ${rlx.argmax(new Float32Array([0.1, 2, 0.3]))}`);
