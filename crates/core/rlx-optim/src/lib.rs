// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX training-step optimizers.
//!
//! Host-side `f32` step functions for the families surveyed in
//! "A Systematic Review of Optimization Algorithms for Modern Deep
//! Learning" (arXiv:2509.02046v1). Each algorithm exposes a small
//! state struct keyed by parameter *name* (so the same struct holds
//! moments for every tensor in a model) and a `step` method that
//! consumes `(name, shape, &mut params, &grads)`.
//!
//! The API is deliberately minimal: it operates on flat `&mut [f32]`
//! / `&[f32]` slices plus a `&[usize]` shape — matching the
//! [`rlx_umap::adam`](../rlx_umap/adam/index.html) pattern. Backends
//! that already ship a fused step kernel (see e.g.
//! `rlx_metal::splat_adam`) are free to bypass this crate for their
//! hot path; this crate is the portable reference / CPU fallback / the
//! one used when there is no backend fused kernel for the requested
//! algorithm.
//!
//! # Algorithms
//!
//! | Family          | Type                          |
//! |-----------------|-------------------------------|
//! | [`Sgd`]         | SGD ± momentum / Nesterov     |
//! | [`Adam`]        | Adam                          |
//! | [`AdamW`]       | AdamW (decoupled decay)       |
//! | [`NAdamW`]      | Nesterov AdamW                |
//! | [`RAdam`]       | Rectified Adam                |
//! | [`QHAdamW`]     | Quasi-hyperbolic AdamW        |
//! | [`Lamb`]        | LAMB (layer-wise adaptive)    |
//! | [`Adafactor`]   | Adafactor (factored 2nd mom.) |
//! | [`Lion`]        | Lion (sign of EMA)            |
//! | [`Soap`]        | SOAP (Shampoo-in-Adam-basis)  |
//! | [`KronPsgd`]    | Kron / PSGD                   |
//! | [`Muon`]        | Muon (Newton–Schulz orth.)    |
//! | [`Sophia`]      | Sophia-H                      |
//! | [`Mars`]        | MARS (variance-reduced)       |
//! | [`Stiefel`]     | Riemannian SGD on St(m,n)     |

// Pure-safe by default. Relaxed from `forbid` to `deny` so the one
// performance-critical exception — the Accelerate/AMX `cblas_sgemm` shim behind
// Muon's Newton–Schulz on macOS — can opt in with a scoped `#[allow(unsafe_code)]`
// (see `muon::accel`). Every other module stays unsafe-free.
#![deny(unsafe_code)]

mod common;

mod adafactor;
mod adam;
mod adamw;
mod kron_psgd;
mod lamb;
mod lion;
mod mars;
mod muon;
mod nadamw;
mod qhadamw;
mod radam;
mod sgd;
mod soap;
mod sophia;
mod stiefel;

pub use adafactor::Adafactor;
pub use adam::Adam;
pub use adamw::AdamW;
pub use kron_psgd::KronPsgd;
pub use lamb::Lamb;
pub use lion::Lion;
pub use mars::Mars;
pub use muon::{Muon, newton_schulz_orth};
pub use nadamw::NAdamW;
pub use qhadamw::QHAdamW;
pub use radam::RAdam;
pub use sgd::Sgd;
pub use soap::Soap;
pub use sophia::Sophia;
pub use stiefel::Stiefel;

pub use common::{global_grad_clip_scale, l2_norm};

/// Common parameter-update interface.
///
/// `name` keys the per-parameter state (moments, preconditioners),
/// `shape` is the parameter's logical shape (used by matrix-aware
/// algorithms like Adafactor / SOAP / Muon — ignored by elementwise
/// ones), `param` is updated in place from `grad`. `grad` is treated
/// as read-only; callers that need gradient clipping should pre-scale
/// it (see [`global_grad_clip_scale`]).
///
/// # Implementing for a backend
///
/// Every algorithm in this crate provides a CPU reference impl. A
/// backend (e.g. `rlx-metal`, `rlx-cuda`) is free to write its own
/// fused step kernel and impl `Optimizer` for a wrapper struct that
/// owns device buffers — the trait places no requirement on where
/// the state lives, only on the entry-point signature. The
/// `rlx-metal::splat_adam` kernel is the canonical example of a
/// backend that bypasses this crate entirely; you can wrap it with a
/// 5-line `impl Optimizer` if you want a uniform interface from a
/// generic trainer.
///
/// # Per-tensor learning rate
///
/// For optimizers that don't need per-tensor LR variation (most
/// transformer pre-training), set [`lr_scale`](Self::lr_scale) to
/// return `1.0` (the default). For domain-specific use cases — e.g.
/// 3D Gaussian splatting, where different attributes need wildly
/// different step sizes — override [`lr_scale`](Self::lr_scale) to
/// multiply the base `lr` by a per-name factor. The provided method
/// on the trait does NOT scale automatically; algorithms are free to
/// consult it via [`Optimizer::lr_scale`] inside their `step`.
/// One parameter's data for a batched optimizer step ([`Optimizer::step_batch`]):
/// its name, static shape, mutable data slice (owned by the caller), and gradient.
pub struct OptItem<'a> {
    pub name: &'a str,
    pub shape: &'a [usize],
    pub param: &'a mut [f32],
    pub grad: &'a [f32],
}

pub trait Optimizer {
    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]);

    /// Batched step over ALL parameters in one call. Default: sequential
    /// `step` per item — bit-identical to the per-parameter loop. Optimizers
    /// whose parameter groups are **independent** (e.g. Muon on the 2-D weight
    /// matrices vs AdamW on the embeddings/biases/norms) can override this to
    /// run the groups on separate threads; because the groups touch disjoint
    /// parameters and disjoint optimizer state, the result is bit-for-bit the
    /// same as the serial loop — only the wall-clock (the CPU-side optimizer
    /// bubble) shrinks toward `max(group_times)` instead of their sum.
    fn step_batch(&mut self, items: &mut [OptItem<'_>]) {
        for it in items.iter_mut() {
            self.step(it.name, it.shape, it.param, it.grad);
        }
    }

    /// Advance the global step counter. Most algorithms increment per
    /// call to [`step`], so most implementations leave this a no-op.
    fn end_iteration(&mut self) {}

    /// Set the base learning rate (for LR schedules / warmup). Default is a
    /// no-op for algorithms without a scalar `lr` (e.g. Adafactor's relative
    /// step size); every algorithm in this crate that has an `lr` field
    /// overrides this to update it.
    fn set_lr(&mut self, _lr: f32) {}

    /// Per-tensor multiplier on the effective learning rate. Default
    /// is `1.0` for every name. Override when wrapping this crate to
    /// support per-name LR schedules (e.g. embedding-vs-attention
    /// splits, or the Gaussian-splat attribute-typed LR setup). The
    /// CPU impls in this crate currently honor this only when the
    /// caller passes a pre-scaled `lr` for the relevant call —
    /// backends are encouraged to consult it inside their fused
    /// kernel.
    fn lr_scale(&self, _name: &str) -> f32 {
        1.0
    }
}
