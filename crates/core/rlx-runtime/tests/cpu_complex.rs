// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Complex tensors as a *downstream* graph-construction utility,
//! validated numerically through the CPU pipeline.
//!
//! Same kind of file as `cpu_sparse_lu.rs`: everything here is what a
//! downstream package (`rlx-photonics`, `rlx-complex`, a user's own
//! application code) would write. The rlx framework crates stay
//! agnostic — they expose the IR (`Add`, `Mul`, `MatMul`, `Sqrt`,
//! `Neg`, `Narrow`, `Concat`, `Sum`, …) and autodiff machinery, and
//! application code composes them however it likes.
//!
//! ## What this file demonstrates
//!
//! 1. **Complex as a 2N real-block representation.** `rlx-ir::DType`
//!    has no native `Complex32`/`Complex64`; we represent `[..., N]`
//!    complex tensors as a pair of real `[..., N]` tensors and build
//!    complex arithmetic on top of existing real ops. No new IR
//!    primitives, no new dtypes, no fork of the framework.
//!
//! 2. **Numerical correctness on real hardware.** Each Complex
//!    operation builds a graph, compiles via `Session::new(Cpu)`,
//!    runs on real f32 inputs, and asserts the output element-wise
//!    against an in-test reference. `rlx-ir`'s previous home for this
//!    code was unit-tested only at the node-count level — moving it
//!    here lets us actually check `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
//!    bits-and-all.
//!
//! 3. **Wirtinger autodiff convention** for real losses with complex
//!    parameters — the photonics inverse-design case. With the 2N
//!    real-block representation, real autodiff over `(real, imag)`
//!    gives `(∂L/∂x, ∂L/∂y)`. The Wirtinger conjugate gradient used
//!    by JAX/PyTorch as the "complex gradient" of a real loss is
//!    `∂L/∂z̄ = ½(∂L/∂x + i ∂L/∂y)`. The steepest-descent direction
//!    in `z`-space is `2·∂L/∂z̄` (the conjugate-gradient convention),
//!    which equals `∂L/∂x + i ∂L/∂y` — i.e. just package the two real
//!    gradients into a `Complex` tensor *with the same sign on the
//!    imag component*. The footgun in the prior 2N-real helper was
//!    leaving this convention undocumented, letting users assemble a
//!    gradient with the wrong imag sign. The
//!    `Complex::wirtinger_grad_from_real_grads` helper below pins
//!    the convention down.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

// ── Complex helper (downstream graph-construction utility) ─────────

/// `[..., N]` complex tensor as a pair of real `[..., N]` tensors.
/// Cheap to copy — two `NodeId`s and the convention "real and imag
/// share shape and dtype".
#[derive(Clone, Copy, Debug)]
struct Complex {
    real: NodeId,
    imag: NodeId,
}

// `Complex` is a *downstream* graph-construction utility — every
// method is part of the documented API even if not every method is
// exercised by every test.
#[allow(dead_code)]
impl Complex {
    fn new(real: NodeId, imag: NodeId) -> Self {
        Self { real, imag }
    }

    /// Element-wise complex addition.
    fn add(self, other: Self, g: &mut Graph) -> Self {
        Self {
            real: g.add(self.real, other.real),
            imag: g.add(self.imag, other.imag),
        }
    }

    /// Element-wise complex subtraction.
    fn sub(self, other: Self, g: &mut Graph) -> Self {
        Self {
            real: g.sub(self.real, other.real),
            imag: g.sub(self.imag, other.imag),
        }
    }

    /// Negation.
    fn neg(self, g: &mut Graph) -> Self {
        Self {
            real: g.neg(self.real),
            imag: g.neg(self.imag),
        }
    }

    /// Element-wise complex multiplication: `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`.
    fn mul(self, other: Self, g: &mut Graph) -> Self {
        let ac = g.mul(self.real, other.real);
        let bd = g.mul(self.imag, other.imag);
        let ad = g.mul(self.real, other.imag);
        let bc = g.mul(self.imag, other.real);
        Self {
            real: g.sub(ac, bd),
            imag: g.add(ad, bc),
        }
    }

    /// Conjugate. Real part shared (cheap); imag negated.
    fn conj(self, g: &mut Graph) -> Self {
        Self {
            real: self.real,
            imag: g.neg(self.imag),
        }
    }

    /// `|z|² = a² + b²`. Returns a real `NodeId`.
    fn abs_sq(self, g: &mut Graph) -> NodeId {
        let aa = g.mul(self.real, self.real);
        let bb = g.mul(self.imag, self.imag);
        g.add(aa, bb)
    }

    /// `|z| = sqrt(a² + b²)`. Returns a real `NodeId`.
    fn abs(self, g: &mut Graph) -> NodeId {
        let asq = self.abs_sq(g);
        g.sqrt(asq)
    }

    /// Multiply complex by real (scalar or broadcastable real tensor).
    fn scale_real(self, scalar: NodeId, g: &mut Graph) -> Self {
        Self {
            real: g.mul(self.real, scalar),
            imag: g.mul(self.imag, scalar),
        }
    }

    /// Wirtinger conjugate gradient used by JAX / PyTorch as the
    /// *complex gradient of a real loss*.
    ///
    /// Convention: for real loss `L(z, z̄)` and complex parameter
    /// `z = x + iy`, the steepest-descent direction in z-space is
    /// `∂L/∂x + i·∂L/∂y` — equal to `2·∂L/∂z̄`. Apply with
    /// `z_new = z - lr · grad`.
    ///
    /// `g_x` and `g_y` are the real gradients produced by routing the
    /// 2N real-block representation through `rlx_opt::autodiff` —
    /// these are exactly what `grad_with_loss` returns when the
    /// scalar loss depends on `real` and `imag` separately.
    ///
    /// Saves users from the silent footgun of assembling
    /// `Complex { real: g_x, imag: -g_y }` (the literal Wirtinger
    /// `∂L/∂z̄` without the factor of 2), which would point in
    /// half-magnitude descent direction and be wrong for SGD.
    fn wirtinger_grad_from_real_grads(g_x: NodeId, g_y: NodeId) -> Self {
        Self {
            real: g_x,
            imag: g_y,
        }
    }
}

// ── Test scaffolding ───────────────────────────────────────────────

fn f32s_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Numerical execution tests ──────────────────────────────────────

#[test]
fn complex_mul_matches_textbook_formula_through_cpu_pipeline() {
    // y = a * b where a, b are length-3 complex tensors.
    let mut g = Graph::new("c_mul");
    let n = 3usize;
    let ar = g.input("a_re", Shape::new(&[n], DType::F32));
    let ai = g.input("a_im", Shape::new(&[n], DType::F32));
    let br = g.input("b_re", Shape::new(&[n], DType::F32));
    let bi = g.input("b_im", Shape::new(&[n], DType::F32));
    let a = Complex::new(ar, ai);
    let b = Complex::new(br, bi);
    let y = a.mul(b, &mut g);
    g.set_outputs(vec![y.real, y.imag]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let a_re = [1.0_f32, 0.0, -2.0];
    let a_im = [2.0_f32, 1.0, 3.0];
    let b_re = [3.0_f32, 4.0, 1.0];
    let b_im = [4.0_f32, -2.0, -1.0];
    let outs = compiled.run_typed(&[
        ("a_re", &f32s_to_bytes(&a_re), DType::F32),
        ("a_im", &f32s_to_bytes(&a_im), DType::F32),
        ("b_re", &f32s_to_bytes(&b_re), DType::F32),
        ("b_im", &f32s_to_bytes(&b_im), DType::F32),
    ]);
    assert_eq!(outs.len(), 2);
    let y_re = bytes_to_f32s(&outs[0].0);
    let y_im = bytes_to_f32s(&outs[1].0);

    for i in 0..n {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        let exp_re = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        let exp_im = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        assert!(
            (y_re[i] - exp_re).abs() < 1e-5,
            "real[{i}]: {} vs {exp_re}",
            y_re[i]
        );
        assert!(
            (y_im[i] - exp_im).abs() < 1e-5,
            "imag[{i}]: {} vs {exp_im}",
            y_im[i]
        );
    }
}

#[test]
fn complex_abs_sq_returns_real_norm_squared() {
    let mut g = Graph::new("c_abs_sq");
    let n = 4usize;
    let ar = g.input("a_re", Shape::new(&[n], DType::F32));
    let ai = g.input("a_im", Shape::new(&[n], DType::F32));
    let z = Complex::new(ar, ai);
    let m = z.abs_sq(&mut g);
    g.set_outputs(vec![m]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let a_re = [3.0_f32, 0.0, -1.0, 2.0];
    let a_im = [4.0_f32, 5.0, 1.0, -2.0];
    let outs = compiled.run_typed(&[
        ("a_re", &f32s_to_bytes(&a_re), DType::F32),
        ("a_im", &f32s_to_bytes(&a_im), DType::F32),
    ]);
    let m_got = bytes_to_f32s(&outs[0].0);
    let exp = [25.0_f32, 25.0, 2.0, 8.0];
    for i in 0..n {
        assert!(
            (m_got[i] - exp[i]).abs() < 1e-5,
            "|z|²[{i}]: {} vs {}",
            m_got[i],
            exp[i]
        );
    }
}

// ── Wirtinger autodiff convention test ─────────────────────────────

/// Verify that one Wirtinger SGD step on `L(z) = |z - z*|²` (a real
/// loss with a complex parameter) actually moves `z` toward `z*`.
///
/// Concretely: starting from `z0 = (1, 0)` with target `z* = (3, -2)`,
/// the gradient computed by routing the 2N real-block representation
/// through real autodiff plus the `wirtinger_grad_from_real_grads`
/// convention should produce the SGD update direction. After one
/// step with `lr = 0.5`, the loss should drop strictly.
///
/// The point isn't precise numerics — it's *direction*. A naive sign
/// flip on imag (the silent footgun the Wirtinger convention
/// addresses) would either point sideways or outright increase the
/// loss.
#[test]
fn wirtinger_grad_descent_step_decreases_real_loss() {
    let z0 = (1.0_f32, 0.0_f32);
    let z_star = (3.0_f32, -2.0_f32);

    // Build a graph that computes loss = |z - z*|² as a function of
    // (z_real, z_imag) inputs; reduce to scalar.
    let mut g = Graph::new("c_wirtinger");
    let zr = g.input("z_re", Shape::new(&[1], DType::F32));
    let zi = g.input("z_im", Shape::new(&[1], DType::F32));
    let tr = g.input("t_re", Shape::new(&[1], DType::F32));
    let ti = g.input("t_im", Shape::new(&[1], DType::F32));
    let z = Complex::new(zr, zi);
    let t = Complex::new(tr, ti);
    let diff = z.sub(t, &mut g);
    let m = diff.abs_sq(&mut g);
    let loss = g.sum(m, vec![0], false);
    g.set_outputs(vec![loss]);

    // Initial loss.
    let mut compiled = Session::new(Device::Cpu).compile(g.clone());
    let outs = compiled.run_typed(&[
        ("z_re", &f32s_to_bytes(&[z0.0]), DType::F32),
        ("z_im", &f32s_to_bytes(&[z0.1]), DType::F32),
        ("t_re", &f32s_to_bytes(&[z_star.0]), DType::F32),
        ("t_im", &f32s_to_bytes(&[z_star.1]), DType::F32),
    ]);
    let loss_before = bytes_to_f32s(&outs[0].0)[0];

    // Differentiate w.r.t. (z_re, z_im) — those are the two real
    // components of the complex parameter z.
    let bwd = grad_with_loss(&g, &[zr, zi]);
    assert_eq!(bwd.outputs.len(), 3, "[loss, dL/dzr, dL/dzi]");

    let mut compiled_bwd = Session::new(Device::Cpu).compile(bwd);
    let outs = compiled_bwd.run_typed(&[
        ("z_re", &f32s_to_bytes(&[z0.0]), DType::F32),
        ("z_im", &f32s_to_bytes(&[z0.1]), DType::F32),
        ("t_re", &f32s_to_bytes(&[z_star.0]), DType::F32),
        ("t_im", &f32s_to_bytes(&[z_star.1]), DType::F32),
        ("d_output", &f32s_to_bytes(&[1.0]), DType::F32),
    ]);
    let g_x = bytes_to_f32s(&outs[1].0)[0];
    let g_y = bytes_to_f32s(&outs[2].0)[0];

    // Wirtinger conjugate gradient (real loss, complex param):
    //   grad_z = ∂L/∂x + i·∂L/∂y
    // SGD step: z_new = z - lr · grad_z, applied componentwise to
    // (real, imag) since 2N real-block tracks them separately.
    let lr = 0.5;
    let z1 = (z0.0 - lr * g_x, z0.1 - lr * g_y);

    // Step result: re-evaluate loss at z1.
    let outs = compiled.run_typed(&[
        ("z_re", &f32s_to_bytes(&[z1.0]), DType::F32),
        ("z_im", &f32s_to_bytes(&[z1.1]), DType::F32),
        ("t_re", &f32s_to_bytes(&[z_star.0]), DType::F32),
        ("t_im", &f32s_to_bytes(&[z_star.1]), DType::F32),
    ]);
    let loss_after = bytes_to_f32s(&outs[0].0)[0];

    // For L = |z - z*|² and lr = 0.5, the Wirtinger SGD step lands
    // exactly at z* (the loss is quadratic with curvature 2I), so
    // loss_after should be ~0. More importantly: loss_after must be
    // strictly less than loss_before — the silent footgun (wrong
    // imag sign) would either hold loss constant or increase it.
    assert!(
        loss_after < loss_before,
        "Wirtinger SGD should reduce loss: before={loss_before}, after={loss_after}"
    );
    assert!(
        loss_after < 1e-4,
        "L = |z - z*|² with lr = 0.5 should land at z*: got loss {loss_after}"
    );

    // The `wirtinger_grad_from_real_grads` helper packages
    // `(g_x, g_y)` with the correct imag sign (no negation). The
    // numerical check above confirms that convention is right —
    // applying these grads as `(z.real - lr·g_x, z.imag - lr·g_y)`
    // descends a real loss of a complex parameter.
    let mut g2 = Graph::new("wirtinger_helper_check");
    let gx_n = g2.input("g_x", Shape::new(&[1], DType::F32));
    let gy_n = g2.input("g_y", Shape::new(&[1], DType::F32));
    let grad = Complex::wirtinger_grad_from_real_grads(gx_n, gy_n);
    assert_eq!(
        grad.real, gx_n,
        "Wirtinger grad real == real autodiff dL/dx"
    );
    assert_eq!(
        grad.imag, gy_n,
        "Wirtinger grad imag == real autodiff dL/dy (no sign flip)"
    );
}
