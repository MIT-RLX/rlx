// Run rlx-wgpu's shipped WGSL matmul kernels on a SPEC-COMPLIANT WebGPU host.
//
// Why this exists: the crate's own test suite talks to wgpu-native, which is a
// superset of WebGPU — it accepts extension directives, non-spec limits and
// naga-specific leniency that a browser will not. The kernels a browser
// actually reaches are the portable ones (`matmul`, `matmul_wide`); the
// cooperative-matrix family requires `enable wgpu_cooperative_matrix`, which is
// wgpu-native-only, so those kernel constructors return `None` when the feature
// is absent and a browser never sees the directive. This script proves the
// browser-reachable path both VALIDATES and COMPUTES correctly there.
//
//   deno run --unstable-webgpu --allow-read crates/backends/rlx-wgpu/webgpu/validate.js
//
// Deno's WebGPU is the same W3C surface a browser exposes, so a pass here is
// the in-browser answer without driving a browser.

const HERE = new URL(".", import.meta.url).pathname;
const KERNELS = HERE + "../src/kernels/";

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error("no WebGPU adapter");
  Deno.exit(1);
}
const device = await adapter.requestDevice();
device.addEventListener("uncapturederror", (e) => {
  console.error("UNCAPTURED:", e.error.message);
  Deno.exit(1);
});
console.log(`adapter: ${adapter.info?.description || adapter.info?.vendor || "?"}`);
console.log(`features: ${[...device.features].join(", ") || "(none)"}`);
console.log(
  `cooperative-matrix exposed to WebGPU: ${
    [...device.features].some((f) => /cooperative/i.test(f)) ? "YES" : "no (expected)"
  }`,
);

// Params: 16 u32 (see `struct Params` in the WGSL).
function params({ m, k, n, batch = 1, aOff = 0, bOff, cOff, actId = 0xffff }) {
  const p = new Uint32Array(16);
  p[0] = m; p[1] = k; p[2] = n;
  p[3] = aOff; p[4] = bOff; p[5] = cOff;
  p[6] = batch;
  p[7] = m * k; p[8] = k * n; p[9] = m * n; // batch strides
  p[10] = 0; p[11] = 0;                     // has_bias, bias_off
  p[12] = actId;
  return p;
}

async function runKernel(file, entry, { m, k, n, wgX, wgY }, A, B) {
  const src = await Deno.readTextFile(KERNELS + file);
  // A spec WGSL validator runs here; a browser-illegal construct throws.
  const module = device.createShaderModule({ code: src });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((x) => x.type === "error");
  if (errs.length) {
    console.error(`${file}: WGSL REJECTED by WebGPU:`);
    for (const e of errs) console.error(`   ${e.lineNum}:${e.linePos} ${e.message}`);
    return null;
  }

  // One flat arena, exactly as the backend lays it out: A | B | C.
  const aOff = 0, bOff = m * k, cOff = m * k + k * n;
  const arena = new Float32Array(cOff + m * n);
  arena.set(A, aOff);
  arena.set(B, bOff);

  const buf = device.createBuffer({
    size: arena.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arena);
  const ubo = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, params({ m, k, n, bOff, cOff }));

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: entry },
  });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buf } },
      { binding: 1, resource: { buffer: ubo } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();

  const readback = device.createBuffer({
    size: m * n * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  enc.copyBufferToBuffer(buf, cOff * 4, readback, 0, m * n * 4);
  device.queue.submit([enc.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return out;
}

// Deterministic non-trivial operands. Both are dense and non-commuting, so a
// transposed or swapped product cannot pass — the same property that the
// native `coop_f32_operand_order_is_not_commuted` test relies on.
function make(rows, cols, seed) {
  const v = new Float32Array(rows * cols);
  let s = seed >>> 0;
  for (let i = 0; i < v.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    v[i] = (s >>> 8) / 8388608 - 1;
  }
  return v;
}
function refMatmul(A, B, m, k, n) {
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let t = 0; t < k; t++) acc += A[i * k + t] * B[t * n + j];
      c[i * n + j] = acc;
    }
  }
  return c;
}

const CASES = [
  // [file, entry, m, k, n, tileN, tileM] — grid matches the Rust dispatcher:
  //   matmul       -> (n/32, m/32)
  //   matmul_wide  -> (n/64, m/32)
  ["matmul.wgsl", "matmul", 64, 128, 96, 32, 32],
  ["matmul.wgsl", "matmul", 33, 40, 31, 32, 32], // ragged: exercises bounds checks
  ["matmul_wide.wgsl", "matmul_wide", 128, 256, 192, 64, 32],
  ["matmul_wide.wgsl", "matmul_wide", 96, 72, 130, 64, 32], // ragged
  // The default wide kernel, and the one a browser actually reaches.
  ["matmul_wide_vec4.wgsl", "matmul_wide_vec4", 128, 256, 192, 64, 64],
  ["matmul_wide_vec4.wgsl", "matmul_wide_vec4", 96, 72, 130, 64, 64], // ragged
];

let failed = 0;
for (const [file, entry, m, k, n, tn, tm] of CASES) {
  const A = make(m, k, 7), B = make(k, n, 99);
  const got = await runKernel(
    file,
    entry,
    { m, k, n, wgX: Math.ceil(n / tn), wgY: Math.ceil(m / tm) },
    A,
    B,
  );
  if (!got) { failed++; continue; }
  const want = refMatmul(A, B, m, k, n);
  let worst = 0, mx = 0;
  for (let i = 0; i < want.length; i++) {
    worst = Math.max(worst, Math.abs(got[i] - want[i]));
    mx = Math.max(mx, Math.abs(want[i]));
  }
  const rel = worst / mx;
  const ok = rel < 1e-5;
  if (!ok) failed++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${entry.padEnd(11)} ${m}x${k}x${n}`.padEnd(40) +
      `max|Δ| ${worst.toExponential(2)}  rel ${rel.toExponential(2)}`,
  );
}

// The coop kernels must be REJECTED here — that is the guarantee that keeps a
// browser from ever seeing them. A silent acceptance would mean the native-only
// directive had leaked into the spec surface.
for (const f of ["matmul_coop_f32.wgsl", "matmul_coop16.wgsl", "matmul_qkv_coop_f32.wgsl"]) {
  const src = await Deno.readTextFile(KERNELS + f);
  // The rejection is the expected outcome, so catch it in an error scope
  // rather than letting it reach the uncaptured-error handler above.
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: src });
  const info = await mod.getCompilationInfo();
  const scoped = await device.popErrorScope();
  const rejected = info.messages.some((x) => x.type === "error") || scoped !== null;
  console.log(
    `${rejected ? "PASS" : "FAIL"}  ${f.padEnd(26)} rejected by spec WebGPU: ${rejected}`,
  );
  if (!rejected) failed++;
}

console.log(failed === 0 ? "\nALL WEBGPU CHECKS PASSED" : `\n${failed} WEBGPU CHECK(S) FAILED`);
Deno.exit(failed === 0 ? 0 : 1);
