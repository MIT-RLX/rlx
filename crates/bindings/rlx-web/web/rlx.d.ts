/**
 * Type definitions for the RLX browser SDK (`rlx.js` / `rlx.ts`).
 * Wasm low-level types come from `./pkg/rlx_web.d.ts` after `just build-web`.
 */

declare module "./pkg/rlx_web.js" {
  export * from "./pkg/rlx_web";
}

export type Backend = "auto" | "cpu" | "webgpu" | "webgl";

export type VisionModelSlug =
  | "mnist-cnn"
  | "mnist-mlp"
  | "cifar-cnn"
  | "resnet";

export interface VisionModelInfo {
  slug: VisionModelSlug;
  title: string;
  inputDims: number[];
  inputLen: number;
  numClasses: number;
  paramNames: string[];
  paramSizes: number[];
  paramFlatLen: number;
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
  webgpu?: boolean;
  wasmInit?: (input?: WebAssembly.Module | BufferSource | Response | URL) => Promise<unknown>;
}

export class VisionModel {
  readonly info: VisionModelInfo;
  initParams(seed?: number): Float32Array;
  syntheticBatch(seed?: number): { x: Float32Array; label: number };
  forwardCpu(x: Float32Array, params: Float32Array): Float32Array;
  trainStepCpu(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr?: number,
  ): TrainStepResult;
  benchCpu(steps?: number, seed?: number, lr?: number): BenchResult;
  benchCpuAsync(opts?: {
    steps?: number;
    seed?: number;
    lr?: number;
    reportEvery?: number;
    onProgress?: (p: BenchProgress) => void;
  }): Promise<BenchResult & { losses: number[]; params: Float32Array }>;
  forwardWebgpu(x: Float32Array, params: Float32Array): Promise<Float32Array>;
  trainStepWebgpu(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr?: number,
  ): Promise<TrainStepResult>;
  forwardWebgl(x: Float32Array, params: Float32Array): Float32Array;
  trainStepWebgl(
    x: Float32Array,
    label: number,
    params: Float32Array,
    lr?: number,
  ): TrainStepResult;
}

export class Rlx {
  readonly wasm: typeof import("./pkg/rlx_web.js");
  readonly webgpuReady: boolean;
  static init(options?: RlxInitOptions): Promise<Rlx>;
  listVisionModels(): VisionModelSlug[];
  modelInfo(slug: VisionModelSlug): VisionModelInfo;
  vision(slug: VisionModelSlug): VisionModel;
  preferredBackend(): string;
  readonly mlp: {
    forward: (...args: unknown[]) => Float32Array;
    loss: (...args: unknown[]) => number;
    grads: (...args: unknown[]) => Float32Array;
    trainStep: (...args: unknown[]) => Float32Array;
    forwardGpu?: (...args: unknown[]) => Promise<Float32Array>;
    gradsGpu?: (...args: unknown[]) => Promise<Float32Array>;
    forwardWebgl?: (...args: unknown[]) => Float32Array;
    gradsWebgl?: (...args: unknown[]) => Float32Array;
  };
}

export { wasm } from "./pkg/rlx_web.js";
export default Rlx;
