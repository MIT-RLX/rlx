# rlx-oneapi

Intel **oneAPI Level Zero** backend for RLX — the dedicated `Device::OneApi`
path for Intel Arc / Data Center Max (Ponte Vecchio) GPUs.

It mirrors the native GPU backends (rlx-cuda / rlx-vulkan): the crate owns the
Level Zero driver / device / context / compute-queue, a USM-shared f32 arena,
and SPIR-V compute modules — the peak Intel path, distinct from the wgpu/Vulkan
portability layers. Packed DiT reverse (`AdaLayerNormBackward` /
`GatedResidualBackward`) has dedicated OpenCL-C SPIR-V kernels
(`kernels/ada_layer_norm_backward.cl` / `gated_residual_backward.cl`); when
kernels are not embedded the same ops host-fallback through `rlx-cpu`.

```
rlx-ir Graph
  → legalize/rewrite to the native primitive set (shared with rlx-vulkan)
  → OneApiExecutable
      ├─ device present + kernels embedded → USM arena + per-op SPIR-V dispatch
      └─ otherwise                          → whole-graph rlx-cpu reference
```

## Driverless by design

`libze_loader` is opened at **runtime** with `libloading` — there is no
link-time dependency on the oneAPI runtime, so the crate compiles and
`cargo build`s on hosts with no Level Zero driver (this macOS dev box, CI).
`rlx_oneapi::is_available()` is always `true` once the crate is linked: with a
Level Zero GPU **and** embedded SPIR-V kernels the native USM path runs;
otherwise every op uses the bit-exact `rlx-cpu` reference (`run_host`). Use
`has_level_zero_device()` / `device_name()` / `has_native_kernels()` to probe
hardware. `Device::OneApi` stays selectable via `RLX_DEVICE=oneapi` even when
only `libze_loader` is present (no `ze_intel_gpu` plugin).

## Why OpenCL-C kernels (not GLSL/naga)

Level Zero's `zeModuleCreate(ZE_MODULE_FORMAT_IL_SPIRV)` +
`zeKernelSetArgumentValue` consume **OpenCL/Kernel-flavor** SPIR-V (entry points
declared `OpEntryPoint Kernel`, arguments as kernel-function parameters,
Physical64 addressing). That is a *different* SPIR-V dialect from the Vulkan
**Shader/GLCompute** flavor naga emits from GLSL (push-constant blocks +
descriptor-bound buffers) — they are **not interchangeable**. So the native
kernels here are authored in OpenCL-C under `kernels/*.cl` and lowered by Intel's
offline compiler `ocloc`, which feeds the GPU compiler the same SPIR-V SYCL /
clBuildProgram would.

Kernel compilation is **opt-in and best-effort** (build.rs): it runs only when
`RLX_ONEAPI_BUILD_KERNELS=1` *and* `ocloc` is on `PATH` (an Intel oneAPI build
host). Otherwise no blobs are embedded, and the backend serves every op through
the bit-exact `rlx-cpu` reference — so it is *correct* everywhere and *native*
on Intel pending bring-up. (Same stance as rlx-cuda's `.cu` sources, validated
only in its Linux Docker image.)

## Status

Claims the full **153/`OpKind`** surface ([`docs/op-coverage.md`](../../docs/op-coverage.md)).
Native OpenCL-C kernels (when embedded) cover norms/fused/SoftmaxCE/RNN/SSM/
vision-bwd/QAT/I8/FFT butterfly — mirroring the Vulkan depth wave. Ops without
an embedded kernel use the bit-exact `rlx-cpu` path.

| Component | State |
|---|---|
| Level Zero FFI + driver/device/context/queue bring-up | implemented |
| USM-shared arena, SPIR-V module/kernel cache, per-op dispatch | implemented |
| OpenCL-C kernels: elementwise + matmul/softmax/rmsnorm + DiT reverse + GroupNorm/fused/SoftmaxCE/RNN/vision-bwd/QAT/I8/… | written (`kernels/*.cl`) |
| CPU-reference path (whole graph / missing kernel) | **validated** (tests green on macOS) |
| Native dispatch on Intel hardware | **NOT yet validated** — no Intel GPU on the dev box |

### Pending hardware validation

Nothing in the Level Zero path runs on the dev box, so these are the bring-up
items to confirm on real Arc / Data Center Max:

- the `ZE_STRUCTURE_TYPE_*` enum values + descriptor layouts in `level_zero.rs`
  against the installed loader version;
- the compute command-queue-group ordinal (assumed `0`);
- `ocloc` SPIR-V ingestion by `zeModuleCreate` and the kernel-argument ABI.

## North-star (peak Intel perf)

Forward-inference correctness first; the perf milestones are: route GEMM through
**oneMKL** (`gemm` on the Level Zero backend) instead of the naive `matmul.cl`,
tile hot kernels with SLM, and deepen remaining host ops (DenseSolve via
device LAPACK only if oneMKL can stay optional). These mirror the
"perf-naive; tile + promote" follow-ups the other native backends carry.

**DenseSolve / BatchedDenseSolve** stay on host by design for now: packed
`HostOpDesc` → rlx-cpu LAPACK (`sgesv` / Accelerate / OpenBLAS) against the
USM-shared arena. Device oneMKL LAPACK is **not** linked (would need a
link-time oneMKL dep that breaks the driverless build). Same HostOpDesc path
as wgpu / Vulkan.

## Build

```sh
# Compiles everywhere (no kernels embedded off an Intel host):
cargo build -p rlx-oneapi
cargo test  -p rlx-oneapi            # CPU-reference compute, green on macOS

# On an Intel oneAPI host (Linux) with `ocloc`:
RLX_ONEAPI_BUILD_KERNELS=1 cargo build -p rlx-oneapi
```

Enable in the runtime with the `oneapi` feature:

```sh
cargo build -p rlx-runtime --features oneapi
```

## License

GPL-3.0-only.
