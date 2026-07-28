# rlx-unfuse

Shared IR-level **unfuse / decompose** pass for the GPU-family backends
(`rlx-cuda`, `rlx-rocm`, `rlx-wgpu`), parameterized by a per-backend
`DecomposePolicy`.

## What it does

Before a backend lowers a graph, composite and unsupported ops
(`FusedAttentionBlock`, `FusedTransformerLayer`, `FusedSwiGLU`, `LoraMatMul`,
`DotGeneral`, `If`, `While`, rank-3 `Attention`, …) are rewritten into the
primitive ops the backend can actually execute. Everything else is passed
through untouched.

```rust
pub fn unfuse(graph: Graph, policy: &dyn DecomposePolicy) -> Graph;
```

That pass used to be copy-pasted into each GPU backend (~1.3k lines each,
~80% identical). It now lives here once. The only genuinely
backend-specific behavior is captured by a handful of capability flags:

```rust
pub trait DecomposePolicy {
    fn should_unfuse(&self, op: &Op) -> bool;          // shared allowlist (default)
    fn fab_native(&self, out_shape: &Shape) -> bool;   // keep FusedAttentionBlock native?
    fn promote_attention_backward(&self) -> bool;       // rank-4-only attn-backward kernel?
    fn fold_matmul_bias_act(&self) -> bool;             // emit FusedMatMulBiasAct vs matmul+add
    fn fold_residual_ln(&self) -> bool;                 // emit FusedResidualLN vs add+layer_norm
    fn attention_accepts_rank3(&self) -> bool;          // stride-based rank-3 attn vs rank-4 transpose
}
```

The three attention/transformer expansions (`expand_fab`, `expand_ftl`,
`expand_attention_rank3`) have exactly **two** shapes — a *materialized
rank-4* variant (CUDA ≈ ROCm) and a *fused-op / rank-3-elided* variant
(wgpu) — which collapse into one implementation each, branching on the flags
above.

## Per-backend policies (live in each backend's `unfuse.rs`)

| policy | flags set |
|---|---|
| `CudaPolicy` | `fab_native` = out rank-3 && `seq ≤ 96`; `promote_attention_backward` = true |
| `RocmPolicy` | all defaults (materialize rank-4, no native FAB gate) |
| `WgpuPolicy` | `fold_matmul_bias_act`, `fold_residual_ln`, `attention_accepts_rank3` = true |

Each backend keeps a thin `pub fn unfuse(graph) -> Graph { rlx_unfuse::unfuse(graph, &XPolicy) }`
so existing call sites are unchanged.

Depends only on `rlx-ir`. This is *not* the MIR autodiff-unfuse in
`rlx-fusion` (`unfuse_fused_for_autodiff`) — that is a separate pass at a
different stage.
## License

MIT OR Apache-2.0.
