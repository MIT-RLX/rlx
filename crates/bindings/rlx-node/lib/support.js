// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

"use strict";

// What the generated rlx.js stands on: loading librlx_capi through koffi,
// calling a C function by its prototype, errors, handles, the argument
// converters, taking ownership of what rlx returns, and the two rows whose
// shape is not a plain call (`Optimizer.step` updates arrays in place;
// `Trainer.run` takes callbacks).
//
// The library is `RLX_CAPI_LIB` when set, else the platform package npm
// installed with this one (`@rlx/rlx-<platform>-<arch>`, or its `-cuda`
// variant when that was installed too), else one bundled in this package
// (`native/<os>-<arch>/`), else librlx_capi in the workspace's target/debug
// (a checkout).

const fs = require("fs");
const path = require("path");
const koffi = require("koffi");

const ext = process.platform === "darwin" ? "dylib" : process.platform === "win32" ? "dll" : "so";
const file = process.platform === "win32" ? "rlx_capi.dll" : `librlx_capi.${ext}`;
const os = process.platform === "darwin" ? "macos" : process.platform === "win32" ? "windows" : "linux";
const bundled = path.join(__dirname, "..", "native", `${os}-${process.arch}`, file);

/** librlx_capi from an installed platform package, the backend variant first. */
function platformPackageLib() {
  const base = `${require("../package.json").name}-${process.platform}-${process.arch}`;
  for (const pkg of [`${base}-cuda`, base]) {
    try {
      return path.join(path.dirname(require.resolve(`${pkg}/package.json`)), file);
    } catch {
      // not installed
    }
  }
  return null;
}

const LIB_PATH =
  process.env.RLX_CAPI_LIB ||
  platformPackageLib() ||
  (fs.existsSync(bundled) ? bundled : path.join(__dirname, "..", "..", "..", "..", "target", "debug", file));
const lib = koffi.load(LIB_PATH);

koffi.struct("RlxF32s", { ptr: "float*", len: "size_t" });
koffi.struct("RlxBytes", { ptr: "uint8_t*", len: "size_t" });
koffi.struct("RlxU32s", { ptr: "uint32_t*", len: "size_t" });

const NONE = 0xffffffff;
const fns = new Map();

/** The koffi function for C function `name`, declared by `proto` on first use. */
function fn(name, proto) {
  let f = fns.get(name);
  if (!f) {
    f = lib.func(proto);
    fns.set(name, f);
  }
  return f;
}

const KINDS = { [-1]: "type", [-2]: "range", [-3]: "runtime", [-4]: "panic", [-5]: "not_found" };

/** A failed rlx call: `kind` is its `RLX_ERR_*` status. */
class RlxError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "RlxError";
    this.kind = kind;
  }
}

function check(rc) {
  if (rc === 0) return;
  const msg = fn("rlx_last_error", "const char* rlx_last_error()")();
  throw new RlxError(KINDS[rc] ?? "other", msg);
}

const registry = new FinalizationRegistry(({ free, handle }) => fn(free, `void ${free}(void*)`)(handle));

/** The base of every handle class: `close()` frees now, collection frees otherwise. */
class Handle {
  close() {
    if (this.handle) {
      registry.unregister(this);
      const free = this.constructor.free;
      fn(free, `void ${free}(void*)`)(this.handle);
      this.handle = null;
    }
  }
  [Symbol.dispose]() {
    this.close();
  }
}

function adopt(obj, handle, free) {
  if (!handle) throw new RlxError("runtime", "rlx returned NULL");
  obj.handle = handle;
  registry.register(obj, { free, handle }, obj);
  return obj;
}

function wrap(Class, handle) {
  return adopt(Object.create(Class.prototype), handle, Class.free);
}

// ── arguments ───────────────────────────────────────────────

const json = (v) => (v == null ? null : JSON.stringify(v));
const f32 = (v) => (v instanceof Float32Array ? v : Float32Array.from(v));
const u8 = (v) => (v instanceof Uint8Array ? v : Uint8Array.from(v));
const u32 = (v) => (v instanceof Uint32Array ? v : Uint32Array.from(v));

/** Rows flattened; with `splat` a one-value row stands for `w` copies. */
function rows(list, w, splat) {
  const out = [];
  list.forEach((r, i) => {
    const row = Array.from(r);
    if (row.length === w) out.push(...row);
    else if (splat && row.length === 1) for (let k = 0; k < w; k++) out.push(row[0]);
    else throw new RlxError("type", `row ${i} needs ${w} values, got ${row.length}`);
  });
  return out;
}

function inputs(obj) {
  const names = Object.keys(obj);
  const data = names.map((k) => f32(obj[k]));
  return { names, data, lens: data.map((d) => d.length) };
}

function typedInputs(obj) {
  const names = Object.keys(obj);
  const data = names.map((k) => {
    const d = obj[k].data;
    return ArrayBuffer.isView(d) ? new Uint8Array(d.buffer, d.byteOffset, d.byteLength) : Uint8Array.from(d);
  });
  return { names, data, lens: data.map((d) => d.length), dtypes: names.map((k) => obj[k].dtype) };
}

// ── results ─────────────────────────────────────────────────

function takeString(p) {
  if (!p) return "";
  const s = koffi.decode(p, "char", -1);
  fn("rlx_string_free", "void rlx_string_free(void*)")(p);
  return s;
}

function takeArray(out, type, free) {
  const v = out.len > 0 ? koffi.decode(out.ptr, type, out.len) : [];
  const ctor = type === "float" ? Float32Array : type === "uint8_t" ? Uint8Array : Uint32Array;
  const copy = ctor.from(v);
  const struct = type === "float" ? "RlxF32s" : type === "uint8_t" ? "RlxBytes" : "RlxU32s";
  fn(free, `void ${free}(${struct})`)(out);
  return copy;
}

function takeOutputs(o) {
  try {
    const n = fn("rlx_outputs_len", "size_t rlx_outputs_len(void*)")(o);
    const get = fn("rlx_outputs_get", "int32_t rlx_outputs_get(void*, size_t, _Out_ void**, _Out_ size_t*)");
    const result = [];
    for (let i = 0; i < n; i++) {
      const p = [null], l = [0];
      check(get(o, i, p, l));
      result.push(Float32Array.from(l[0] > 0 ? koffi.decode(p[0], "float", l[0]) : []));
    }
    return result;
  } finally {
    fn("rlx_outputs_free", "void rlx_outputs_free(void*)")(o);
  }
}

function takeTypedOutputs(o) {
  try {
    const n = fn("rlx_typed_outputs_len", "size_t rlx_typed_outputs_len(void*)")(o);
    const get = fn("rlx_typed_outputs_get", "int32_t rlx_typed_outputs_get(void*, size_t, _Out_ void**, _Out_ size_t*, _Out_ const char**)");
    const result = [];
    for (let i = 0; i < n; i++) {
      const p = [null], l = [0], d = [null];
      check(get(o, i, p, l, d));
      result.push({ data: Uint8Array.from(l[0] > 0 ? koffi.decode(p[0], "uint8_t", l[0]) : []), dtype: d[0] });
    }
    return result;
  } finally {
    fn("rlx_typed_outputs_free", "void rlx_typed_outputs_free(void*)")(o);
  }
}

function takeTensors(m) {
  try {
    const n = fn("rlx_tensor_map_len", "size_t rlx_tensor_map_len(void*)")(m);
    const get = fn("rlx_tensor_map_get", "int32_t rlx_tensor_map_get(void*, size_t, _Out_ const char**, _Out_ void**, _Out_ size_t*)");
    const result = {};
    for (let i = 0; i < n; i++) {
      const name = [null], p = [null], l = [0];
      check(get(m, i, name, p, l));
      result[name[0]] = Float32Array.from(l[0] > 0 ? koffi.decode(p[0], "float", l[0]) : []);
    }
    return result;
  } finally {
    fn("rlx_tensor_map_free", "void rlx_tensor_map_free(void*)")(m);
  }
}

// ── the manual rows ─────────────────────────────────────────

function attachManual({ Optimizer, Trainer }) {
  /** Update each `params[name]` (a Float32Array) in place from `grads[name]`. */
  Optimizer.prototype.step = function step(params, grads) {
    const names = Object.keys(params);
    const ps = names.map((k) => params[k]);
    const gs = names.map((k) => {
      if (!(k in grads) || grads[k].length !== params[k].length) {
        throw new RlxError("type", `Optimizer.step: no matching gradient for '${k}'`);
      }
      return f32(grads[k]);
    });
    check(fn("rlx_optimizer_step",
      "int32_t rlx_optimizer_step(void*, const char**, float**, const float**, const size_t*, void*, void*, size_t)")(
      this.handle, names, ps, gs, ps.map((p) => p.length), null, null, names.length));
  };

  /** Train `steps` steps pulling inputs from `batch(step)`; the run summary. */
  Trainer.prototype.run = function run(steps, batch, opts = {}) {
    const every = Math.max(opts.every ?? 1, 1);
    const start = this.steps();
    const losses = [];
    let stopped = false;
    const t0 = performance.now();
    for (let i = 0; i < steps; i++) {
      const s = start + i;
      const b = batch(s);
      if (b == null) break;
      if (opts.lr) this.setLr(opts.lr(s));
      const loss = this.step(b);
      losses.push(loss);
      if (opts.onStep && (i + 1) % every === 0 && opts.onStep(s, loss) === false) {
        stopped = true;
        break;
      }
    }
    const n = losses.length;
    return {
      steps: n, stopped,
      firstLoss: n ? losses[0] : null, lastLoss: n ? losses[n - 1] : null,
      meanLoss: n ? losses.reduce((a, b) => a + b, 0) / n : null,
      stepsPerSecond: n / Math.max((performance.now() - t0) / 1000, 1e-9),
      gradNorm: this.gradNorm(),
    };
  };
}

module.exports = {
  NONE, RlxError, Handle, fn, check, adopt, wrap, json, f32, u8, u32, rows, inputs, typedInputs,
  takeString, takeArray, takeOutputs, takeTypedOutputs, takeTensors, attachManual, LIB_PATH,
};
