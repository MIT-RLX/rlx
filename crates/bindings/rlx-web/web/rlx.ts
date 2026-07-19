/**
 * RLX browser SDK — typed wrapper around wasm-bindgen exports.
 *
 * Usage:
 * ```ts
 * import { Rlx } from "./rlx.js";
 * const rlx = await Rlx.init({ webgpu: true });
 * const bench = rlx.vision("mnist-cnn");
 * const logits = bench.forwardCpu(input, params);
 * ```
 */

import initWasm, * as wasm from "./pkg/rlx_web.js";

export type Backend = "auto" | "cpu" | "webgpu" | "webgl";

export type VisionModelSlug =
  | "mnist-cnn"
  | "mnist-mlp"
  | "cifar-cnn"
  | "resnet";

/** Metadata returned by {@link Rlx.modelInfo}. */
export interface VisionModelInfo {
  slug: VisionModelSlug;
  title: string;
  /** NCHW-ish dims without batch (e.g. `[1,28,28]` or `[784]`). */
  inputDims: number[];
  inputLen: number;
  numClasses: number;
  paramNames: string[];
  paramSizes: number[];
  paramFlatLen: number;
  /** WebGL supports forward for all models; training only for MLP. */
  webglTrain: boolean;
}

export interface TrainStepResult {
  loss: number;
  params: Float32Array;
}

export interface BenchResult {
  initialLoss: number;
  finalLoss: number;
  stepsPerSec: number;
}

export interface BenchProgress {
  step: number;
  steps: number;
  loss: number;
  initialLoss: number;
  finalLoss: number;
  losses: number[];
  stepsPerSec: number;
  params: Float32Array;
}

export interface RlxInitOptions {
  /** Attempt WebGPU device init before first GPU call. Default false. */
  webgpu?: boolean;
  /** Custom wasm module init (advanced). */
  wasmInit?: (input?: WebAssembly.Module | BufferSource | Response | URL) => Promise<unknown>;
}

function mapInfo(raw: wasm.VisionModelInfo): VisionModelInfo {
  return {
    slug: raw.slug as VisionModelSlug,
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

/** High-level handle for one vision classification model. */
export class VisionModel {
  readonly info: VisionModelInfo;
  private readonly inner: wasm.VisionBench;

  constructor(inner: wasm.VisionBench, info: VisionModelInfo) {
    this.inner = inner;
    this.info = info;
  }

  initParams(seed = 42): Float32Array {
    return new Float32Array(this.inner.init_params(seed));
  }

  /** Synthetic normalized input plus label in the last element. */
  syntheticBatch(seed = 1): { x: Float32Array; label: number } {
    const raw = this.inner.synthetic_batch(seed);
    const label = raw[raw.length - 1];
    return { x: new Float32Array(raw.subarray(0, raw.length - 1)), label };
  }

  forwardCpu(x: Float32Array, params: Float32Array): Float32Array {
    return new Float32Array(this.inner.forward_cpu(x, params));
  }

  trainStepCpu(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr = 0.01,
  ): TrainStepResult {
    const out = this.inner.train_step_cpu(x, label, params, lr);
    return { loss: out[0], params: new Float32Array(out.subarray(1)) };
  }

  benchCpu(steps = 50, seed = 42, lr = 0.01): BenchResult {
    const [initialLoss, finalLoss, stepsPerSec] = this.inner.bench_cpu(steps, seed, lr);
    return { initialLoss, finalLoss, stepsPerSec };
  }

  /**
   * Async CPU SGD that yields so UIs can paint progress via `onProgress`.
   */
  async benchCpuAsync(opts: {
    steps?: number;
    seed?: number;
    lr?: number;
    reportEvery?: number;
    onProgress?: (p: BenchProgress) => void;
  } = {}): Promise<BenchResult & { losses: number[]; params: Float32Array }> {
    const steps = opts.steps ?? 50;
    const seed = opts.seed ?? 42;
    const lr = opts.lr ?? 0.01;
    const reportEvery = opts.reportEvery ?? 1;
    const onProgress = opts.onProgress ?? (() => {});
    const yieldUi = () => new Promise<void>((r) => setTimeout(r, 0));

    let params = this.initParams(seed);
    const { x, label } = this.syntheticBatch(seed);
    const losses: number[] = [];
    const t0 = performance.now();

    for (let i = 0; i < steps; i++) {
      const step = this.trainStepCpu(x, label, params, lr);
      params = step.params;
      losses.push(step.loss);
      const done = i + 1;
      if (done === 1 || done === steps || done % reportEvery === 0) {
        const elapsed = (performance.now() - t0) / 1000;
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

    const elapsed = (performance.now() - t0) / 1000;
    return {
      initialLoss: losses[0],
      finalLoss: losses[losses.length - 1],
      stepsPerSec: steps / Math.max(elapsed, 1e-6),
      losses,
      params,
    };
  }

  async forwardWebgpu(x: Float32Array, params: Float32Array): Promise<Float32Array> {
    if (typeof this.inner.forward_webgpu !== "function") {
      throw new Error("WebGPU not enabled — rebuild with --webgpu");
    }
    return new Float32Array(await this.inner.forward_webgpu(Array.from(x), Array.from(params)));
  }

  async trainStepWebgpu(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr = 0.01,
  ): Promise<TrainStepResult> {
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

  forwardWebgl(x: Float32Array, params: Float32Array): Float32Array {
    if (typeof this.inner.forward_webgl !== "function") {
      throw new Error("WebGL not enabled — rebuild with --webgl");
    }
    return new Float32Array(this.inner.forward_webgl(x, params));
  }

  trainStepWebgl(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr = 0.01,
  ): TrainStepResult {
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

/** Entry point for RLX in the browser. */
export class Rlx {
  readonly wasm: typeof wasm;
  readonly webgpuReady: boolean;

  private constructor(w: typeof wasm, webgpuReady: boolean) {
    this.wasm = w;
    this.webgpuReady = webgpuReady;
  }

  static async init(options: RlxInitOptions = {}): Promise<Rlx> {
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

  listVisionModels(): VisionModelSlug[] {
    return wasm.list_vision_models() as VisionModelSlug[];
  }

  modelInfo(slug: VisionModelSlug): VisionModelInfo {
    return mapInfo(wasm.vision_model_info(slug));
  }

  vision(slug: VisionModelSlug): VisionModel {
    const inner = new wasm.VisionBench(slug);
    return new VisionModel(inner, mapInfo(inner.info()));
  }

  preferredBackend(): string {
    return wasm.preferred_backend();
  }

  /** Low-level MLP helpers (legacy demo API). */
  get mlp() {
    const w = this.wasm;
    return {
      forward: w.mlp_forward.bind(w),
      loss: w.mlp_loss.bind(w),
      grads: w.mlp_grads.bind(w),
      trainStep: w.mlp_train_step.bind(w),
      forwardGpu: w.mlp_forward_gpu?.bind(w),
      gradsGpu: w.mlp_grads_gpu?.bind(w),
      forwardWebgl: w.mlp_forward_webgl?.bind(w),
      gradsWebgl: w.mlp_grads_webgl?.bind(w),
    };
  }
}

export { wasm };
export default Rlx;
