# rlxsl

Warm-tier kernel DSL for RLX — one scalar-expression manifest, many backend
languages.

The standalone **unary-activation** kernels historically re-expressed the same
scalar math by hand in WGSL, MSL, GLSL, CUDA C, OpenCL-C and Rust — six copies
of `gelu`'s Abramowitz-&-Stegun `erf`, six copies of the softplus stability
trick, six copies of every activation. This crate makes that math a single
source: each activation is one `Sx` expression tree, and the per-language
emitters (`Lang`) render it into each target language.

## Scope

The full standalone elementwise surface: unary activations (forward **and**
auto-differentiated backward), binary elementwise ops (`binary`), compare ops
(`compare`), plus the double-word precision prelude (`dw`).

`dw::double_single_prelude(lang)` is the double-single arithmetic library
(`dw_two_sum`, `dw_two_prod`, `dw_add`, `dw_sub`, `dw_mul`, `dw_div`,
`dw_sqrt` over a `{hi, lo}` f32 pair) behind `Op::DoubleSingle` on every GPU backend. Its error-free
transforms need precise math. For drivers that compile with fast math and offer
no switch (wgpu on Metal), `wgsl_fast_math_guarded()` routes every intermediate
through `dw_opq`, an XOR with a zero the compiler can't see (and adds
`dw_exp`), so neither
identities nor reassociation can cancel the error terms; at run time it is the
identity, and results match the precise prelude.

Deliberately the **warm tier** — the *un-fused* standalone kernels, which are
the fallback path. The hot path is the fused region, hand-written per backend
and never routed through here. So generating these costs no peak performance.

## What keeps it correct

- The `eval` interpreter walks the same `Sx` tree, so the manifest's math can
  be checked numerically against the trusted CPU backend for every
  `Activation` (`rlx-runtime`'s `kernel_dsl_activation_oracle` test). It is an
  *independent* reference, because the CPU kernel stays hand-written.
- Case order follows `Activation::opcode_relu_first` — the canonical opcode
  from `rlx_ir::opcodes` — so codegen and dispatch cannot disagree.
- The dev-test suite parses the emitted WGSL (`wgsl-in`) and GLSL (`glsl-in`)
  with `naga` to prove they are valid shader source. CUDA / MSL / OpenCL have
  no toolchain-free parser, so those emitters are checked structurally
  (balanced delimiters plus full case coverage).

> One trap worth remembering: the GELU approximation constant is `√(2/π)`, and
> Rust's `FRAC_2_SQRT_PI` is `2/√π`. Using the latter here silently broke
> `GeluApprox` on *every* generated backend at once — which is exactly the
> failure mode single-sourcing creates, and exactly why the numeric reference
> exists.

## Emitting

```bash
cargo run -p rlxsl --example export
```

Each file is emitted with the opcode scheme its backend actually dispatches
with — relu-first for CUDA / wgpu, gelu-first for native Vulkan. The written
files are for inspection and diffing only; nothing re-imports them.

## Quickstart

```rust
use rlx_ir::op::Activation;
use rlxsl::{Lang, OpcodeScheme, emit_activation, msl_activation_module};

// A whole per-language module, in dispatch order.
let msl = msl_activation_module(OpcodeScheme::ReluFirst);
assert!(msl.contains("gelu") || msl.contains("erf"));

// Or one activation's body, to splice into a hand-written kernel.
let (lets, expr) = emit_activation(Activation::Softplus, Lang::Msl);
println!("{}\n{expr}", lets.join("\n"));
```

## License

MIT OR Apache-2.0.
