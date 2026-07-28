# rlx-tensor

A native [`ndarray`](https://crates.io/crates/ndarray) alternative for RLX:
NumPy-style tensors with operator-overloaded, expression-like syntax that
**trace into rlx-ir instead of executing eagerly**. The graph stays lazy —
so the compiler fuses the whole expression and plans memory across every
backend — until you force it.

|                       | ndarray / NumPy        | PyTorch                 | JAX             | **rlx-tensor**                              |
|-----------------------|------------------------|-------------------------|-----------------|---------------------------------------------|
| Default execution     | eager, host            | eager, GPU              | trace + XLA     | **trace → fuse → backend**                  |
| Intermediate buffers  | always materialized    | often kept for autograd | XLA decides     | **fusion + memory plan**                    |
| Device                | host only              | CUDA-centric            | TPU / GPU       | **CPU / Metal / MLX / CUDA / ROCm / wgpu / TPU** |
| Variable batch / seq  | manual padding         | manual                  | `vmap` / shapes | **first-class `Dim::Dynamic`**              |
| Slicing               | views (cheap)          | views                   | slice ops       | **`s![]` → `narrow` (fusable)**             |

The base crate is a pure rlx-ir graph builder — symbolic `Tensor` handles carry
**zero payload bytes** and pull in **no backend**. Materialization, autodiff,
and training are opt-in features so you only pay for what you use.

## Quickstart

```rust
use rlx_tensor::Tensor; // requires feature = "eval"

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
let b = Tensor::ones([3]);
let c = (&a + &b).relu();          // graphs auto-merge, nothing runs yet
assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0]); // compiles once, auto-picks the fastest backend
// c.on(Device::Metal).to_vec()    // or pin an explicit backend
```

Materialization is compile-cached per thread (keyed by graph + constants +
device), so repeated `to_vec` / `Func::run` reuse the compiled artifact.

## Building a graph

Two equivalent styles — eager-looking `Tensor` constructors (need `eval` to
read back), or a pure symbolic `graph(..)` scope with named inputs/params:

```rust
use rlx_tensor::{ax, graph, rg, s, shape};

// static shape
let g = graph("mlp", |g| {
    let x = g.input("x", shape![2, 4]);
    let w = g.param("w", shape![4, 3]);
    x.matmul(&w).gelu()
});

// dynamic batch + window slice (no copy in IR)
let g2 = graph("batched", |g| {
    let x = g.input("x", shape![?, 128]);   // `?` = dynamic dim
    x.slice(s![ax(), rg(0, 64)])            // s![] → narrow view
});
```

## Declarative DSL — `rlx!` (feature `dsl`)

For a compact, readable little language, `rlx! { … }` declares the whole graph —
shapes, wiring, and outputs are inferred — and evaluates to an `rlx_ir::Graph`:

```rust
use rlx_tensor::rlx; // feature `dsl` (on for umbrella `rlx` users via `tensor`)

let g = rlx! {
    graph "mlp";
    input x: [?, 784];              // `?` = dynamic batch; shape![] grammar
    param w1: [784, 256];  param b1: [256];
    param w2: [256, 10];   param b2: [10];

    let h = gelu(x @ w1 + b1);      // `@` = matmul (NumPy precedence)
    let y = h @ w2 + b2;
    out y;                          // defaults to the last `let`
};
```

- **`@`** matmul, **`+ - * /`** elementwise (broadcasting + scalar promotion),
  precedence `.method()` > unary `-` > (`@` `*` `/`) > (`+` `-`), matching NumPy.
- **`f(x)`** sugar → `x.f()` for any no-arg method (`gelu`, `relu`, `sqrt`, …).
- **`x.method(args)`** escape hatch reaches the full `Tensor` API. A bare
  argument naming a binding is validated and auto-borrowed, so
  `q.attention(k, v, 8, 64, MaskKind::Causal)` reads naturally; wrap an external
  value as `(value)` to pass it through raw.
- Inputs/params are **auto-named** from the binding ident; feed them by that
  name (`compiled.set_param("w1", …)` / `run(&[("x", …)])`).

Statement forms: `graph "name";` (optional), `input`/`param name: [dims];`,
`const name = value : DType;`, `let name = expr;`, `out a, b;`. Mistakes —
unknown bindings, matmul on a scalar, a `let` that isn't a tensor — are reported
as spanned compile errors, not cryptic downstream type errors.

## Op surface

- **Arithmetic** — `+ - * /` (operator overloads, tensor & scalar rhs),
  `maximum` / `minimum` / `clamp` / `pow`.
- **Activations** — `relu`, `gelu`, `gelu_approx`, `silu`, `sigmoid`, `tanh`,
  `exp`, `log`, `sqrt`, `rsqrt`, `abs`, `sin`, `cos`, `tan`, `atan`, `round`.
- **Reductions** — `sum`, `mean`, `max`, `min`, `prod`, `var`, `std`, `norm`,
  `logsumexp`, `cumsum`, `argmax`, `argmin`.
- **Shape / view** — `reshape`, `transpose` / `t`, `narrow`, `slice`, `select`,
  `split`, `chunk`, `flatten`, `squeeze` / `unsqueeze`, `broadcast_to`, `tile`,
  `flip`, `roll`, `pad`, `cat`, `stack`.
- **Indexing** — `gather`, `index_select`, comparisons (`eq` `ne` `lt` `le`
  `gt` `ge`), `where_`, `masked_fill`.
- **NN / linalg** — `matmul` / `mm`, `softmax`, `layer_norm`, `rms_norm`,
  `conv2d`, `attention`, `rope`, `fft` / `ifft` / `fft_axis`, `inv`, `solve`.
- **Constructors** — `from_vec`, `zeros`, `ones`, `full`, `eye`, `arange`,
  `arange_step`, `rand`, `randn`.

## Autodiff (feature `grad`)

Reverse-mode gradients on a materialized tensor:

```rust
let loss = (&a * &b).sum([0], false);
let g = loss.grad(&[&a, &b]);     // [∂loss/∂a, ∂loss/∂b]
```

## Composable transforms (feature `transforms`)

JAX-shaped transforms operate on `Func` (a traced function of named inputs) and
chain — `vmap(grad(f))`:

```rust
let f = Func::new("f", |s| {
    let x = s.input("x", shape![3]);
    (&x * &x).sum([0], false)
});
let batched_grad = f.grad(&["x"]).vmap(&["x"], 4); // Func -> Func -> Func
let compiled = batched_grad.jit();                 // compile once
let out = compiled.run(&[("x", &xs)]);             // run many, no recompile
```

`Func` also exposes `value_and_grad`, `jvp`, and `hvp`.

## Training (feature `optim`)

`Func::train_step` runs value+grad and applies an `rlx_optim` optimizer to the
bound params:

```rust
use rlx_tensor::{AdamW, Func};

let mut opt = AdamW::new(3e-4);
let (loss, _) = f.train_step(&mut opt, &[("x", &xs)]);
```

`Adam`, `Muon`, `AdamW`, `Lion`, `Sgd`, `Optimizer`, and `LrSchedule` are re-exported.

## Features

| feature        | what it adds                                                                 |
|----------------|------------------------------------------------------------------------------|
| *(default)*    | pure rlx-ir graph builder — symbolic only, no backend                        |
| `dsl`          | the `rlx! { … }` declarative graph DSL (pulls in the `rlx-macros` proc macro) |
| `eval`         | materialize via `rlx_runtime::Session`: `to_vec`, `on(device)`, CPU backend  |
| `eval-metal` / `eval-mlx` / `eval-cuda` / `eval-rocm` / `eval-gpu` / `eval-coreml` | `.on(Device::…)` for that backend |
| `eval-apple`   | all Apple GPU/NPU backends (Metal + MLX + wgpu + CoreML/ANE)                  |
| `eval-blas`    | CPU eval with Accelerate BLAS/LAPACK (needed for `inv` / `solve`)            |
| `grad` / `transforms` / `autodiff` | reverse-mode AD + composable `Func` transforms (implies `eval` where it materializes) |
| `optim`        | `Func::train_step` + `rlx_optim` optimizers (implies `autodiff` + `eval`)    |
| `ndarray`      | `Tensor::from(array)` / `tensor.to_ndarray()` interop with `ndarray`         |

Migrating from `ndarray`? Enable `ndarray` for zero-friction `From`/`to_ndarray`
round-trips.

## Where it sits

`rlx-tensor` is re-exported through the prelude crate as
[`rlx::tensor`](../rlx) (`use rlx::prelude::*;` brings `Tensor`, `graph`, the
`s!` / `shape!` macros, and the slice helpers into scope). It is one layer above
[`rlx-ir`](../rlx-ir) and shares its `Op` set, so any graph it builds compiles
through the same fusion / memory-planning / backend pipeline as the rest of RLX.

## License

MIT OR Apache-2.0.
