/**
 * RLX browser SDK — JavaScript wrapper around wasm-bindgen exports.
 * @module rlx
 */

import initWasm, * as wasm from "./pkg/rlx_web.js";

function mapInfo(raw) {
  return {
    slug: raw.slug,
    title: raw.title,
    inputDims: raw.input_dims,
    inputLen: raw.input_len,
    numClasses: raw.num_classes,
    paramNames: raw.param_names,
    paramSizes: raw.param_sizes,
    paramFlatLen: raw.param_flat_len,
    webglTrain: raw.webgl_train,
  };
}

/** @typedef {"auto"|"cpu"|"webgpu"|"webgl"} Backend */
/** @typedef {"mnist-cnn"|"mnist-mlp"|"cifar-cnn"|"resnet"} VisionModelSlug */

export class VisionModel {
  constructor(inner, info) {
    this.inner = inner;
    this.info = info;
  }

  initParams(seed = 42) {
    return new Float32Array(this.inner.init_params(seed));
  }

  syntheticBatch(seed = 1) {
    const raw = this.inner.synthetic_batch(seed);
    const label = raw[raw.length - 1];
    return { x: new Float32Array(raw.subarray(0, raw.length - 1)), label };
  }

  forwardCpu(x, params) {
    return new Float32Array(this.inner.forward_cpu(x, params));
  }

  trainStepCpu(x, label, params, lr = 0.01) {
    const out = this.inner.train_step_cpu(x, label, params, lr);
    return { loss: out[0], params: new Float32Array(out.subarray(1)) };
  }

  benchCpu(steps = 50, seed = 42, lr = 0.01) {
    const [initialLoss, finalLoss, stepsPerSec] = this.inner.bench_cpu(steps, seed, lr);
    return { initialLoss, finalLoss, stepsPerSec };
  }

  /**
   * Async CPU SGD loop that yields to the event loop so UIs can paint progress.
   * @param {{ steps?: number, seed?: number, lr?: number, reportEvery?: number, onProgress?: (p: object) => void }} [opts]
   */
  async benchCpuAsync(opts = {}) {
    const steps = opts.steps ?? 50;
    const seed = opts.seed ?? 42;
    const lr = opts.lr ?? 0.01;
    const reportEvery = opts.reportEvery ?? 1;
    const onProgress = opts.onProgress ?? (() => {});
    const yieldUi = () => new Promise((r) => setTimeout(r, 0));

    let params = this.initParams(seed);
    const { x, label } = this.syntheticBatch(seed);
    const losses = [];
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();

    for (let i = 0; i < steps; i++) {
      const step = this.trainStepCpu(x, label, params, lr);
      params = step.params;
      losses.push(step.loss);
      const done = i + 1;
      if (done === 1 || done === steps || done % reportEvery === 0) {
        const elapsed =
          ((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0) / 1000;
        onProgress({
          step: done,
          steps,
          loss: step.loss,
          initialLoss: losses[0],
          finalLoss: step.loss,
          losses: losses.slice(),
          stepsPerSec: done / Math.max(elapsed, 1e-6),
          params,
        });
        await yieldUi();
      }
    }

    const elapsed =
      ((typeof performance !== "undefined" ? performance.now() : Date.now()) - t0) / 1000;
    return {
      initialLoss: losses[0],
      finalLoss: losses[losses.length - 1],
      stepsPerSec: steps / Math.max(elapsed, 1e-6),
      losses,
      params,
    };
  }

  async forwardWebgpu(x, params) {
    if (typeof this.inner.forward_webgpu !== "function") {
      throw new Error("WebGPU not enabled — rebuild with --webgpu");
    }
    return new Float32Array(
      await this.inner.forward_webgpu(Array.from(x), Array.from(params)),
    );
  }

  async trainStepWebgpu(x, label, params, lr = 0.01) {
    if (typeof this.inner.train_step_webgpu !== "function") {
      throw new Error("WebGPU not enabled — rebuild with --webgpu");
    }
    const out = await this.inner.train_step_webgpu(
      Array.from(x),
      label,
      Array.from(params),
      lr,
    );
    return { loss: out[0], params: new Float32Array(out.subarray(1)) };
  }

  forwardWebgl(x, params) {
    if (typeof this.inner.forward_webgl !== "function") {
      throw new Error("WebGL not enabled — rebuild with --webgl");
    }
    return new Float32Array(this.inner.forward_webgl(x, params));
  }

  trainStepWebgl(x, label, params, lr = 0.01) {
    if (typeof this.inner.train_step_webgl !== "function") {
      throw new Error("WebGL not enabled — rebuild with --webgl");
    }
    if (!this.info.webglTrain) {
      throw new Error(`${this.info.slug}: WebGL training unavailable (conv backward)`);
    }
    const out = this.inner.train_step_webgl(x, label, params, lr);
    return { loss: out[0], params: new Float32Array(out.subarray(1)) };
  }
}

export class Rlx {
  /** @param {typeof wasm} w @param {boolean} webgpuReady */
  constructor(w, webgpuReady) {
    this.wasm = w;
    this.webgpuReady = webgpuReady;
  }

  /** @param {{ webgpu?: boolean, wasmInit?: Function }} [options] */
  static async init(options = {}) {
    const wasmInit = options.wasmInit ?? initWasm;
    await wasmInit();
    let webgpuReady = false;
    if (options.webgpu && typeof wasm.init_webgpu === "function") {
      try {
        webgpuReady = await wasm.init_webgpu();
      } catch {
        webgpuReady = false;
      }
    }
    return new Rlx(wasm, webgpuReady);
  }

  listVisionModels() {
    return wasm.list_vision_models();
  }

  /** @param {VisionModelSlug} slug */
  modelInfo(slug) {
    return mapInfo(wasm.vision_model_info(slug));
  }

  /** @param {VisionModelSlug} slug */
  vision(slug) {
    const inner = new wasm.VisionBench(slug);
    return new VisionModel(inner, mapInfo(inner.info()));
  }

  preferredBackend() {
    return wasm.preferred_backend();
  }

  get mlp() {
    const w = this.wasm;
    return {
      forward: w.mlp_forward.bind(w),
      loss: w.mlp_loss.bind(w),
      grads: w.mlp_grads.bind(w),
      trainStep: w.mlp_train_step.bind(w),
      forwardGpu: typeof w.mlp_forward_gpu === "function" ? w.mlp_forward_gpu.bind(w) : undefined,
      gradsGpu: typeof w.mlp_grads_gpu === "function" ? w.mlp_grads_gpu.bind(w) : undefined,
      forwardWebgl:
        typeof w.mlp_forward_webgl === "function" ? w.mlp_forward_webgl.bind(w) : undefined,
      gradsWebgl: typeof w.mlp_grads_webgl === "function" ? w.mlp_grads_webgl.bind(w) : undefined,
    };
  }
}

export { wasm };
export default Rlx;
