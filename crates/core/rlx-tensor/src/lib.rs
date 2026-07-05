// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! # rlx-tensor
//!
//! Symbolic tensor API for RLX — expression syntax with compiler-backed
//! execution. Compared to NumPy / ndarray / PyTorch / JAX:
//!
//! | | ndarray / NumPy | PyTorch | JAX | **rlx-tensor** |
//! |-|-----------------|---------|-----|----------------|
//! | Default execution | eager, host | eager, GPU | trace + XLA | **trace → fuse → backend** |
//! | Intermediate buffers | always materialized | often kept for autograd | XLA decides | **fusion + memory plan** |
//! | Device | host only | CUDA-centric | TPU/GPU | **CPU / Metal / CUDA / ROCm / wgpu / MLX / TPU** |
//! | Variable batch/seq | manual padding | manual | `jax.vmap` / shapes | **first-class `Dim::Dynamic`** |
//! | Slicing | views (cheap) | views | slice ops | **`s![]` → narrow (fusable)** |
//!
//! Symbolic [`Tensor`] handles carry **zero payload bytes** until you compile
//! and run — but with the `eval` feature you can also build from host data
//! NumPy-style and read results straight back, the graph staying lazy (so the
//! compiler fuses + memory-plans the whole expression) until you materialize:
//!
//! ```ignore
//! use rlx_tensor::Tensor; // requires feature = "eval"
//!
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
//! let b = Tensor::ones([3]);
//! let c = (&a + &b).relu();        // graphs auto-merge, nothing runs yet
//! assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0]); // auto-selects fastest backend
//! // c.on(Device::Metal).to_vec()  // or pin an explicit backend
//! ```
//!
//! Reverse-mode autodiff (feature `grad`) returns gradient tensors:
//!
//! ```ignore
//! let loss = (&a * &b).sum([0], false);
//! let g = loss.grad(&[&a, &b]);     // [∂loss/∂a, ∂loss/∂b]
//! assert_eq!(g[0].to_vec(), b.to_vec());
//! ```
//!
//! JAX-shaped composable transforms (feature `transforms`) operate on [`Func`]
//! (a traced function of named inputs) and chain — `vmap(grad(f))`:
//!
//! ```ignore
//! let f = Func::new("f", |s| { let x = s.input("x", shape![3]); (&x * &x).sum([0], false) });
//! let batched_grad = f.grad(&["x"]).vmap(&["x"], 4);  // Func -> Func -> Func
//! let compiled = batched_grad.jit();                  // compile once
//! let out = compiled.run(&[("x", &xs)]);              // run many, no recompile
//! ```
//!
//! Materialization is compile-cached per thread (keyed by graph + constants +
//! device), so even plain `to_vec` / `run` reuse the compiled artifact.
//!
//! ```rust
//! use rlx_tensor::{ax, graph, rg, s, shape};
//!
//! // static shape
//! let g = graph("mlp", |g| {
//!     let x = g.input("x", shape![2, 4]);
//!     let w = g.param("w", shape![4, 3]);
//!     (&x.matmul(&w)).gelu()
//! });
//!
//! // dynamic batch + window slice (no copy in IR)
//! let g2 = graph("batched", |g| {
//!     let x = g.input("x", shape![?, 128]);
//!     x.slice(s![ax(), rg(0, 64)])
//! });
//! # let _ = (g, g2);
//! ```

mod array;
#[cfg(feature = "eval")]
mod cache;
mod handle;
#[cfg(feature = "ndarray")]
mod interop;
mod scalar;
#[cfg(feature = "optim")]
mod schedule;
mod scope;
mod slice;
mod tensor;
mod transform;

pub use array::{cat, stack};
pub use rlx_ir::op::MaskKind;
pub use rlx_ir::{DType, Dim, Graph, NodeId, ScaleLayout, ScaledFormat, Shape};
/// Optimizers for [`Func::train_step`] (re-exported from `rlx_optim`).
/// Available with the `optim` feature.
#[cfg(feature = "optim")]
pub use rlx_optim::{Adam, AdamW, Lion, Optimizer, Sgd};
#[cfg(feature = "optim")]
pub use schedule::LrSchedule;
pub use scope::{GraphScope, graph, graph_with};
pub use slice::{SliceAxis, SliceSpec, ax, ix, rg, tail};
pub use tensor::Tensor;
pub use transform::Func;
#[cfg(feature = "eval")]
pub use transform::Jitted;

#[cfg(feature = "eval")]
pub use cache::{cache_stats, clear_cache};
/// Device selector for [`Tensor::on`] / [`Tensor::to_vec_on`] (re-exported
/// from `rlx_runtime`). Available with the `eval` feature.
#[cfg(feature = "eval")]
pub use rlx_runtime::Device;
/// Backend detection (re-exported from `rlx_runtime`): which devices this build
/// can use, and the fastest one. `to_vec` / `Func::run` pick automatically.
#[cfg(feature = "eval")]
pub use rlx_runtime::{available_devices, fastest_device, fastest_device_for, is_available};
#[cfg(feature = "eval")]
pub use tensor::Materialize;

#[doc(hidden)]
#[macro_export]
macro_rules! __dim {
    (?) => {
        $crate::Dim::Dynamic(0)
    };
    (? $sym:literal) => {
        $crate::Dim::Dynamic($sym)
    };
    ($n:expr) => {
        $crate::Dim::Static($n)
    };
}

/// Build a [`Shape`]. Default dtype is `F32`.
///
/// Each static dimension may be **any `usize` expression** — a literal, an
/// identifier, or arithmetic like `w / 2`, `w + heads`, `cfg.width * 2` — so graph
/// builders can size params/inputs from computed dimensions directly (no need to
/// hoist every derived size into a `let`). Dynamic dimensions use `?` → symbol 0
/// (repeat `?` for the same unknown size), `?1` → symbol 1, etc. A leading
/// `Dtype;` sets the element type (default `F32`).
///
/// ```rust
/// use rlx_tensor::{shape, DType};
///
/// let a = shape![2, 4];
/// let b = shape![F32; ?, 128];
/// let w = 64usize;
/// let c = shape![w + 1, w / 2, 3 * w];   // computed dimensions
/// assert_eq!(c.dtype(), DType::F32);
/// assert!(c.is_static());
/// assert!(!b.is_static());
/// ```
#[macro_export]
macro_rules! shape {
    ($dtype:ident; $($rest:tt)+) => {
        $crate::Shape::from_dims(&$crate::__shape_dims!([] ; $($rest)+), $crate::DType::$dtype)
    };
    ($($rest:tt)+) => {
        $crate::Shape::from_dims(&$crate::__shape_dims!([] ; $($rest)+), $crate::DType::F32)
    };
}

/// Token-tree muncher backing [`shape!`]: accumulates `Dim`s left to right,
/// dispatching `?`/`?N` to dynamic dims and everything else to a `usize` expression.
/// (A `tt`-list can't mix the `?` syntax with multi-token expressions, so we munch.)
#[doc(hidden)]
#[macro_export]
macro_rules! __shape_dims {
    // done (with or without a trailing comma)
    ([$($acc:tt)*] ; ) => { [ $($acc)* ] };
    ([$($acc:tt)*] ; ,) => { [ $($acc)* ] };
    // dynamic `?N`
    ([$($acc:tt)*] ; ? $sym:literal , $($rest:tt)*) => {
        $crate::__shape_dims!([$($acc)* $crate::Dim::Dynamic($sym),] ; $($rest)*)
    };
    ([$($acc:tt)*] ; ? $sym:literal) => {
        [ $($acc)* $crate::Dim::Dynamic($sym) ]
    };
    // dynamic `?` (symbol 0)
    ([$($acc:tt)*] ; ? , $($rest:tt)*) => {
        $crate::__shape_dims!([$($acc)* $crate::Dim::Dynamic(0),] ; $($rest)*)
    };
    ([$($acc:tt)*] ; ?) => {
        [ $($acc)* $crate::Dim::Dynamic(0) ]
    };
    // static dimension: any expression, more following
    ([$($acc:tt)*] ; $e:expr, $($rest:tt)*) => {
        $crate::__shape_dims!([$($acc)* $crate::Dim::Static($e),] ; $($rest)*)
    };
    // static dimension: any expression, last
    ([$($acc:tt)*] ; $e:expr) => {
        [ $($acc)* $crate::Dim::Static($e) ]
    };
}
