# Logical kernels (one op, many backends)

RLX can express a **logical kernel** as a single [`OpKind`](src/op.rs) while lowering it two ways at compile time:

| Path | When | Execution |
|------|------|-----------|
| **Native** | Kind is in the backend `supported_ops` | Backend thunk (Metal MSL, CPU ref, wgpu, …) |
| **Common IR** | Kind is missing from `supported_ops`, or `RLX_KERNEL_DISPATCH=common` | Primitive MIR (`MatMul`, `Reduce`, `Binary`, `Expand`, …) |

Native fast paths are **not** removed. Common lowering is additive.

## Policy

[`KernelDispatchPolicy`](src/logical_kernel.rs):

- `PreferNative` (default) — native if claimed, else common
- `ForceCommon` — always common IR
- `ForceNative` — never common-lower; compile fails if unsupported

Environment: `RLX_KERNEL_DISPATCH=common|native`.

Runtime: [`CompileOptions::kernel_dispatch`](../rlx/rlx-runtime/src/options.rs) ([`KernelDispatchConfig`](src/logical_kernel.rs)).

Per-op overrides (without global `ForceCommon`):

```rust
CompileOptions::new()
    .force_common_kinds(&[OpKind::GaussianSplatRender, OpKind::GaussianSplatRenderBackward])
```

`force_native_kinds` wins over `force_common_kinds` and `ForceCommon`.

## Registered kernels

See [`registered_logical_kernels()`](src/logical_kernel.rs). Splat common bodies live in [`logical_kernel/splat_common.rs`](src/logical_kernel/splat_common.rs).

## Adding a kernel

1. Add or reuse an [`OpKind`](src/op.rs) + [`Op`](src/op.rs) variant.
2. Implement `lower_*` in `rlx-fusion` (or call an existing fusion pass).
3. Register in [`lower_logical_kernels.rs`](../rlx/rlx-fusion/src/lower_logical_kernels.rs).
4. List the kind in backend `supported_ops` only when a native thunk exists.

## Splat example

```rust
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_splat::logical_kernel::PRIMITIVE_SPLAT_SUPPORTED_OPS;
use rlx_ir::logical_kernel::KernelDispatchPolicy;

let opts = CompileOptions::new()
    .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
    .kernel_dispatch(KernelDispatchPolicy::PreferNative);
let compiled = Session::new(Device::Cpu).compile_with(graph, &opts);
```

TPU omits `GaussianSplatRender*` from its claim set; compile rewrites to common IR automatically.

`rlx-runtime` feature `metal` uses native MSL splat (`rlx-metal/native-splat`) by default. Use
`rlx_splat::logical_kernel::splat_common_only_config()` or `RLX_KERNEL_DISPATCH=common` for the
primitive IR baseline when debugging or on backends without a native splat thunk.

Autodiff graphs need a `d_output` input (see `rlx_splat` test helper `autodiff_session_inputs`).

## Dispatch transparency

Use these APIs to see what will run **native** vs **common-ir** vs **rewritten** vs **missing** before or after compile.

| API | Crate | Role |
|-----|-------|------|
| [`prepare_graph_for_backend_with_report`](../rlx-compile/src/dispatch_report.rs) | `rlx-compile` / `rlx-opt` | Rewrite + legalize probe (compile path) |
| [`analyze_dispatch`](../rlx-compile/src/dispatch_report.rs) | `rlx-compile` / `rlx-opt` | Static common-ir probe (no unfuse) |
| [`format_dispatch_report`](../rlx-compile/src/dispatch_report.rs) | `rlx-compile` / `rlx-opt` | Human-readable summary |
| [`maybe_log_dispatch_report`](../rlx-compile/src/dispatch_report.rs) | `rlx-compile` / `rlx-opt` | Log when env flags set |
| [`legalize_graph_for_device_with_report`](../rlx-runtime/src/device_ext.rs) | `rlx-runtime` | Pre-compile check + report |
| [`dispatch_report_for_device`](../rlx-runtime/src/device_ext.rs) | `rlx-runtime` | Static probe only |

### Environment

| Variable | Effect |
|----------|--------|
| `RLX_DISPATCH_REPORT=1` | Print dispatch report during compile (`prepare_fused_graph`) |
| `RLX_VERBOSE=1` | Same as above (plus other verbose paths) |
| `RLX_KERNEL_DISPATCH=common\|native` | Global policy override |

### Example report

```
rlx dispatch report — backend "Metal", policy PreferNative, supported_ops claim=142
  common-ir lowering (portable, add to supported_ops for native fast path):
    - GaussianSplatRender (gaussian_splat_render)
  native:
    - MatMul ×12 nodes
  rewritten:
    - FusedMatMulBiasAct ×3 nodes
  compile-ready: yes
```

### Rust usage

```rust
use rlx_runtime::{
    dispatch_report_for_device, legalize_graph_for_device_with_report, Device,
};
use rlx_opt::format_dispatch_report;

let report = dispatch_report_for_device(&graph, Device::Metal)?;
println!("{}", format_dispatch_report(&report));

let (legal, report) = legalize_graph_for_device_with_report(graph, Device::Metal)?;
assert!(report.compile_ready);
```

[`supports_graph`](../rlx-runtime/src/device_ext.rs) and
[`first_unsupported_op`](../rlx-runtime/src/device_ext.rs) use the same
rewrite + legalization probe when a backend is registered.

See also the overview in the [workspace README](../README.md#kernel-dispatch-and-transparency).
