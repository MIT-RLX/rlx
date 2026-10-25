// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory. RLX_TOKENIZER=<tokenizer.json>
// adds the tokenizer.
//
//   node examples/loading.js

const fs = require("fs");
const os = require("os");
const path = require("path");
const rlx = require("..");

const w = Float32Array.from({ length: 64 }, (_, i) => (i % 9) / 4 - 1);
const gguf = path.join(os.tmpdir(), `rlx-example-node-${process.pid}.gguf`);
const writer = new rlx.GgufWriter();
writer.addF32("proj.weight", [8, 8], "F32", w);
writer.addF32("proj.weight_q8", [8, 8], "Q8_0", w);
writer.write(gguf, { architecture: "demo", metadata: { "demo.layers": 1 } });
const f = rlx.Gguf.open(gguf);
console.log(`gguf architecture: ${f.architecture()}`);
console.log(`gguf tensors: ${f.tensorNames().join(" ")}`);
const exact = f.tensor("proj.weight");
const q8 = f.tensor("proj.weight_q8");
console.log(`gguf f32 exact: ${exact.every((v, i) => v === w[i])} q8 max|Δ|: ${Math.max(...exact.map((v, i) => Math.abs(v - q8[i]))).toFixed(4)}`);

const g = new rlx.Graph({ name: "proj" });
const x = g.input("x", [1, 8]);
g.setOutputs([g.matmul(x, g.param("proj.weight", [8, 8]))]);
const c = new rlx.Session().compile(g);
c.setParam("proj.weight", exact);
const [y] = c.run({ x: new Float32Array(8).fill(1) });
console.log(`gguf -> graph: ${Array.from(y.slice(0, 3), (v) => v.toFixed(2)).join(" ")}`);

const st = path.join(os.tmpdir(), `rlx-example-node-${process.pid}.safetensors`);
let header = JSON.stringify({ "proj.weight": { dtype: "F32", shape: [8, 8], data_offsets: [0, w.byteLength] } });
header += " ".repeat((8 - (header.length % 8)) % 8);
const len = Buffer.alloc(8);
len.writeBigUInt64LE(BigInt(header.length));
fs.writeFileSync(st, Buffer.concat([len, Buffer.from(header), Buffer.from(w.buffer)]));
const loaded = rlx.loadWeights("safetensors", st);
console.log(`safetensors: ${JSON.stringify(Object.keys(loaded))} exact: ${loaded["proj.weight"].every((v, i) => v === w[i])}`);

const packed = rlx.quantize(w, "Q8_0");
console.log(`quantize q8_0: ${w.byteLength} -> ${packed.length} bytes, round trip ${rlx.dequantize(packed, "Q8_0", { count: w.length }).length} values`);
fs.rmSync(gguf);
fs.rmSync(st);

if (process.env.RLX_TOKENIZER) {
  const tok = rlx.Tokenizer.load(process.env.RLX_TOKENIZER);
  const ids = tok.encode("Hello from Node");
  console.log(`tokenizer: ${ids.length} ids -> ${JSON.stringify(tok.decode(ids))}`);
} else {
  console.log("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)");
}
console.log(`argmax: ${rlx.argmax([0.1, 2, 0.3])}`);
