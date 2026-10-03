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

use std::collections::HashMap;

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
/// transformer pre-training), set `lr_scale` to
/// return `1.0` (the default). For domain-specific use cases — e.g.
/// 3D Gaussian splatting, where different attributes need wildly
/// different step sizes — override `lr_scale` to
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

/// An optimizer's internal state, for checkpointing a training run.
///
/// `buffers` are the per-parameter accumulators keyed `"<slot>/<param>"` — Adam
/// writes `"m/w1"` and `"v/w1"` — so a reload can match them back up by name
/// even if the parameter order changed. `step` is the bias-correction counter.
///
/// Resuming without this is not resuming: Adam's moments are most of what it
/// knows, and restarting them at zero produces a loss spike and a different
/// trajectory from the one that was interrupted.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OptimizerState {
    pub step: u64,
    pub buffers: Vec<(String, Vec<f32>)>,
}

impl OptimizerState {
    /// Collect one named slot out of a `HashMap<param, Vec<f32>>`.
    pub fn extend_slot(&mut self, slot: &str, map: &HashMap<String, Vec<f32>>) {
        let mut names: Vec<&String> = map.keys().collect();
        // Sorted, so a checkpoint's byte layout does not depend on hash order.
        names.sort();
        for name in names {
            self.buffers
                .push((format!("{slot}/{name}"), map[name].clone()));
        }
    }

    /// Restore one named slot into a `HashMap<param, Vec<f32>>`.
    pub fn take_slot(&self, slot: &str, map: &mut HashMap<String, Vec<f32>>) {
        let prefix = format!("{slot}/");
        for (key, values) in &self.buffers {
            if let Some(name) = key.strip_prefix(&prefix) {
                map.insert(name.to_string(), values.clone());
            }
        }
    }
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
    /// call to `step`, so most implementations leave this a no-op.
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

    /// Snapshot the optimizer's state for a checkpoint.
    ///
    /// `None` means this algorithm has not opted in, so a run using it **cannot
    /// be resumed exactly**. Callers should say so rather than silently
    /// restarting the accumulators at zero.
    fn state_dict(&self) -> Option<OptimizerState> {
        None
    }

    /// Restore a snapshot. Returns false when unsupported or when the state does
    /// not belong to this algorithm.
    fn load_state_dict(&mut self, _state: &OptimizerState) -> bool {
        false
    }
}

#[cfg(test)]
mod state_dict_tests {
    use super::*;

    /// A checkpointed optimizer must produce the *same next step* as one that
    /// was never interrupted. Restarting the moments at zero passes a
    /// "does it train" test and still changes the trajectory, so this compares
    /// against an uninterrupted reference rather than against itself.
    fn resume_matches_uninterrupted<O: Optimizer, F: Fn() -> O>(make: F) {
        let grads: [[f32; 4]; 6] = [
            [0.5, -0.25, 0.125, 1.0],
            [0.4, -0.20, 0.100, 0.9],
            [0.3, -0.15, 0.075, 0.8],
            [0.2, -0.10, 0.050, 0.7],
            [0.1, -0.05, 0.025, 0.6],
            [0.05, -0.02, 0.01, 0.5],
        ];
        let shape = [2usize, 2];

        // Reference: six steps, no interruption.
        let mut reference = make();
        let mut p_ref = [1.0f32, 1.0, 1.0, 1.0];
        for g in &grads {
            reference.step("w", &shape, &mut p_ref, g);
            reference.end_iteration();
        }

        // Interrupted: three steps, snapshot, rebuild, restore, three more.
        let mut first = make();
        let mut p = [1.0f32, 1.0, 1.0, 1.0];
        for g in &grads[..3] {
            first.step("w", &shape, &mut p, g);
            first.end_iteration();
        }
        let state = first.state_dict().expect("this optimizer opted in");
        drop(first);

        let mut second = make();
        assert!(second.load_state_dict(&state), "load_state_dict refused");
        for g in &grads[3..] {
            second.step("w", &shape, &mut p, g);
            second.end_iteration();
        }

        for (i, (a, b)) in p_ref.iter().zip(p.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: resumed {b} != uninterrupted {a}"
            );
        }
    }

    #[test]
    fn adam_resumes_bit_exactly() {
        resume_matches_uninterrupted(|| Adam::new(0.1));
    }

    #[test]
    fn adamw_resumes_bit_exactly() {
        resume_matches_uninterrupted(|| AdamW::new(0.1));
    }

    #[test]
    fn sgd_with_momentum_resumes_bit_exactly() {
        resume_matches_uninterrupted(|| {
            let mut o = Sgd::new(0.1);
            o.momentum = 0.9;
            o
        });
    }

    #[test]
    fn lion_resumes_bit_exactly() {
        resume_matches_uninterrupted(|| Lion::new(0.1));
    }

    #[test]
    fn an_optimizer_without_state_says_so() {
        // Honest `None` beats a snapshot that silently drops the accumulators:
        // the caller can warn that the run is not exactly resumable.
        assert!(Soap::new(0.1).state_dict().is_none());
        assert!(!Soap::new(0.1).load_state_dict(&OptimizerState::default()));
    }
}

#[cfg(test)]
mod hot_loop_tests {
    use super::*;

    fn spread(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0 - 1.0) * 0.1
            })
            .collect()
    }

    fn assert_bit_equal(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}[{i}]: {a} != {b} (bit-exactness is the contract here — the loops \
                 were restructured for speed, not for different arithmetic)"
            );
        }
    }

    /// SGD's hot loop was rewritten into one specialized loop per configuration
    /// (it used to test `mu == 0.0` and read `self.nesterov` per element, which
    /// blocked vectorization and cost 7x). The arithmetic must be untouched, so
    /// this compares against the original formulas written out by hand.
    #[test]
    fn sgd_matches_its_scalar_reference_bit_exactly() {
        const N: usize = 257; // deliberately not a multiple of any vector width
        let grad = spread(N, 11);
        for (momentum, nesterov, weight_decay) in [
            (0.0f32, false, 0.0f32),
            (0.0, false, 0.05),
            (0.9, false, 0.0),
            (0.9, false, 0.05),
            (0.9, true, 0.0),
            (0.9, true, 0.05),
        ] {
            let lr = 0.03f32;
            let mut opt = Sgd::new(lr);
            opt.momentum = momentum;
            opt.nesterov = nesterov;
            opt.weight_decay = weight_decay;
            let mut param = spread(N, 7);

            // The reference: exactly the expressions the original loop used.
            let mut ref_param = spread(N, 7);
            let mut ref_v = vec![0.0f32; N];

            for _ in 0..4 {
                opt.step("w", &[N], &mut param, &grad);
                opt.end_iteration();
                for i in 0..N {
                    let g = grad[i] + weight_decay * ref_param[i];
                    if momentum == 0.0 {
                        ref_param[i] -= lr * g;
                    } else {
                        ref_v[i] = momentum * ref_v[i] + g;
                        let update = if nesterov {
                            g + momentum * ref_v[i]
                        } else {
                            ref_v[i]
                        };
                        ref_param[i] -= lr * update;
                    }
                }
            }
            assert_bit_equal(
                &format!("sgd(mu={momentum}, nesterov={nesterov}, wd={weight_decay})"),
                &param,
                &ref_param,
            );
        }
    }

    /// Adam gained an `f32_math` option (AdamW already had one). The default
    /// must still be the f64-intermediate path, bit for bit, and the new path
    /// must be the f32 arithmetic it claims.
    #[test]
    fn adam_f32_and_f64_paths_each_match_their_reference() {
        const N: usize = 129;
        let grad = spread(N, 11);
        let (lr, b1, b2, eps, wd) = (0.03f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);

        for f32_math in [false, true] {
            let mut opt = Adam::new(lr).with_f32_math(f32_math);
            opt.beta1 = b1;
            opt.beta2 = b2;
            opt.eps = eps;
            opt.weight_decay = wd;
            let mut param = spread(N, 7);

            let mut ref_param = spread(N, 7);
            let mut ref_m = vec![0.0f32; N];
            let mut ref_v = vec![0.0f32; N];

            for step in 0..4 {
                opt.step("w", &[N], &mut param, &grad);
                opt.end_iteration();
                if f32_math {
                    let t = step + 1;
                    let bc1 = 1.0 - b1.powi(t);
                    let bc2 = 1.0 - b2.powi(t);
                    for i in 0..N {
                        let g = grad[i] + wd * ref_param[i];
                        ref_m[i] = b1 * ref_m[i] + (1.0 - b1) * g;
                        ref_v[i] = b2 * ref_v[i] + (1.0 - b2) * (g * g);
                        let m_hat = ref_m[i] / bc1;
                        let v_hat = ref_v[i] / bc2;
                        ref_param[i] -= lr * (m_hat / (v_hat.sqrt() + eps));
                    }
                } else {
                    let t = (step + 1) as f64;
                    let (b1, b2) = (b1 as f64, b2 as f64);
                    let bc1 = 1.0 - b1.powf(t);
                    let bc2 = 1.0 - b2.powf(t);
                    let (eps, lr) = (eps as f64, lr as f64);
                    for i in 0..N {
                        let g = (grad[i] + wd * ref_param[i]) as f64;
                        let new_m = b1 * ref_m[i] as f64 + (1.0 - b1) * g;
                        let new_v = b2 * ref_v[i] as f64 + (1.0 - b2) * g * g;
                        ref_m[i] = new_m as f32;
                        ref_v[i] = new_v as f32;
                        let m_hat = new_m / bc1;
                        let v_hat = new_v / bc2;
                        ref_param[i] -= (lr * m_hat / (v_hat.sqrt() + eps)) as f32;
                    }
                }
            }
            assert_bit_equal(&format!("adam(f32_math={f32_math})"), &param, &ref_param);
        }
    }

    /// The two paths must *differ* — otherwise the option is a no-op and the
    /// 2.1x is coming from somewhere unexplained.
    #[test]
    fn adams_two_precision_paths_are_not_the_same_computation() {
        const N: usize = 64;
        let grad = spread(N, 11);
        let run = |f32_math: bool| {
            let mut opt = Adam::new(0.03).with_f32_math(f32_math);
            let mut param = spread(N, 7);
            for _ in 0..8 {
                opt.step("w", &[N], &mut param, &grad);
                opt.end_iteration();
            }
            param
        };
        let f64_path = run(false);
        let f32_path = run(true);
        assert!(
            f64_path
                .iter()
                .zip(&f32_path)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "f32_math made no difference, so it is not selecting a different path"
        );
        // …but they must agree to f32 tolerance, or one of them is wrong.
        for (i, (a, b)) in f64_path.iter().zip(&f32_path).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                "element {i}: f64 path {a} vs f32 path {b}"
            );
        }
    }

    /// Nothing above should depend on the length being a nice multiple, since a
    /// vectorized loop has a scalar tail.
    #[test]
    fn odd_lengths_are_handled_by_the_tail() {
        for n in [1usize, 3, 7, 15, 16, 17, 31, 33, 63, 65] {
            let grad = spread(n, 11);
            let mut param = spread(n, 7);
            let before = param.clone();
            let mut opt = Sgd::new(0.1);
            opt.momentum = 0.9;
            opt.step("w", &[n], &mut param, &grad);
            assert!(
                param.iter().zip(&before).any(|(a, b)| a != b),
                "n={n}: nothing moved"
            );
            assert!(param.iter().all(|x| x.is_finite()), "n={n}: not finite");
        }
    }
}
