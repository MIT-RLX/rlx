// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Forward-mode dual numbers, and functionals of a solution differentiated with
//! them.
//!
//! # What this is for
//!
//! A finite element solve is rarely the answer on its own; what is wanted is
//! some *functional* of it — a torque, a loss, a stored energy, a peak stress —
//! and, for optimisation, that functional's sensitivity to the nodal solution.
//! Deriving that sensitivity by hand is easy to do once and does not scale: each
//! new quantity needs its own derivation, and a mistake in one is invisible
//! because the value it multiplies is still right.
//!
//! Nearly all such functionals share a shape:
//!
//! ```text
//! J = integral f(gradient(u)) dOmega
//! ```
//!
//! with `f` a *local* function of the gradient — two numbers in, one out. That
//! is the case forward-mode differentiation handles best, because the cost is
//! proportional to the number of *inputs* and there are only two. Two evaluations
//! of `f` with seeded duals give `df/d(grad)` exactly, and the chain from there
//! to the nodal values is the shape-function gradient, which is already known.
//!
//! So the caller writes the integrand once, in ordinary arithmetic, and gets both
//! the integral and its sensitivity. No derivation, and no opportunity for a
//! derivation to disagree with the quantity it belongs to.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A value carried alongside its derivative with respect to one input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// The value.
    pub v: f64,
    /// The derivative.
    pub d: f64,
}

impl Dual {
    /// Zero, with zero derivative.
    pub const ZERO: Dual = Dual { v: 0.0, d: 0.0 };

    /// A constant: no dependence on the variable being differentiated.
    pub const fn constant(v: f64) -> Dual {
        Dual { v, d: 0.0 }
    }

    /// The variable itself, seeded with unit derivative.
    pub const fn variable(v: f64) -> Dual {
        Dual { v, d: 1.0 }
    }

    /// Square root, with its derivative.
    ///
    /// Guarded at zero, where the true derivative is unbounded: a magnitude that
    /// happens to vanish should not put an infinity into a gradient.
    pub fn sqrt(self) -> Dual {
        let v = self.v.max(0.0).sqrt();
        Dual {
            v,
            d: if v > 0.0 { self.d / (2.0 * v) } else { 0.0 },
        }
    }

    /// Absolute value. The derivative at zero is taken as zero.
    ///
    /// Not by `signum`, which is `+1` at zero in Rust and would give the kink a
    /// one-sided slope it has not earned — a sum of magnitudes assembled that way
    /// acquires a derivative out of nothing. This said it took the derivative as
    /// zero and did not, until a duplicate implementation of the same function
    /// disagreed with it.
    pub fn abs(self) -> Dual {
        let sign = if self.v > 0.0 {
            1.0
        } else if self.v < 0.0 {
            -1.0
        } else {
            0.0
        };
        Dual {
            v: self.v.abs(),
            d: sign * self.d,
        }
    }

    /// Raise to an integer power.
    pub fn powi(self, n: i32) -> Dual {
        Dual {
            v: self.v.powi(n),
            d: n as f64 * self.v.powi(n - 1) * self.d,
        }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual {
            v: self.v + o.v,
            d: self.d + o.d,
        }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual {
            v: self.v - o.v,
            d: self.d - o.d,
        }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual {
            v: self.v * o.v,
            d: self.d * o.v + self.v * o.d,
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual {
            v: self.v / o.v,
            d: (self.d * o.v - self.v * o.d) / (o.v * o.v),
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            v: -self.v,
            d: -self.d,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, k: f64) -> Dual {
        Dual {
            v: self.v * k,
            d: self.d * k,
        }
    }
}

/// Arithmetic a coordinate can be expressed in.
///
/// Implemented for `f64` and for [`Dual`], so geometry written once against this
/// trait produces either plain coordinates or coordinates carrying their
/// derivative with respect to a design parameter. That is the whole mechanism by
/// which a parameter becomes differentiable: the construction is not duplicated
/// for the derivative, so the two cannot drift apart.
pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a constant, with no dependence on the parameter.
    fn constant(v: f64) -> Self;
    /// The value, discarding any derivative.
    fn value(self) -> f64;
    /// The derivative, zero for a plain scalar.
    fn tangent(self) -> f64;
    /// Scale by a plain number.
    fn scale(self, k: f64) -> Self;
    /// The smaller of two values, compared on value alone.
    fn min_of(self, other: Self) -> Self;
    /// The larger of two values, compared on value alone.
    fn max_of(self, other: Self) -> Self;

    /// Square root. The argument is clamped at zero and the derivative there is
    /// taken as zero, which is defensive rather than mathematical — the true
    /// slope is unbounded — and is what keeps a magnitude computed as the root of
    /// a sum of squares usable when the sum is zero.
    fn sqrt(self) -> Self;
    /// Raise to a real power.
    fn powf(self, exponent: f64) -> Self;
    /// Magnitude. Not differentiable at zero, where the derivative is taken as
    /// zero — the value that keeps a sum of magnitudes from acquiring a slope it
    /// has not earned.
    fn abs(self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
}

impl Scalar for f64 {
    fn constant(v: f64) -> Self {
        v
    }
    fn value(self) -> f64 {
        self
    }
    fn tangent(self) -> f64 {
        0.0
    }
    fn scale(self, k: f64) -> Self {
        self * k
    }
    fn min_of(self, other: Self) -> Self {
        self.min(other)
    }
    fn max_of(self, other: Self) -> Self {
        self.max(other)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn powf(self, exponent: f64) -> Self {
        f64::powf(self, exponent)
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
}

impl Scalar for Dual {
    fn constant(v: f64) -> Self {
        Dual::constant(v)
    }
    fn value(self) -> f64 {
        self.v
    }
    fn tangent(self) -> f64 {
        self.d
    }
    fn scale(self, k: f64) -> Self {
        self * k
    }
    // Compared on value, so a tie breaks the same way it would for a plain
    // number and the surviving branch keeps its own derivative. Comparing on the
    // derivative as well would make the choice depend on which parameter is
    // being differentiated, which is not a geometric fact.
    fn min_of(self, other: Self) -> Self {
        if self.v <= other.v { self } else { other }
    }
    fn max_of(self, other: Self) -> Self {
        if self.v >= other.v { self } else { other }
    }
    // Delegating, not reimplementing. `Dual` carries inherent `sqrt` and `abs`
    // of its own, and an inherent method wins over a trait method at any
    // concrete call site — so a second implementation here would apply in
    // generic code and not in concrete code, and the two would differ silently.
    // They differed by exactly that: `abs` at zero.
    fn sqrt(self) -> Self {
        Dual::sqrt(self)
    }
    fn powf(self, exponent: f64) -> Self {
        Dual {
            v: self.v.powf(exponent),
            d: exponent * self.v.powf(exponent - 1.0) * self.d,
        }
    }
    fn abs(self) -> Self {
        Dual::abs(self)
    }
    fn sinh(self) -> Self {
        Dual {
            v: self.v.sinh(),
            d: self.v.cosh() * self.d,
        }
    }
    fn cosh(self) -> Self {
        Dual {
            v: self.v.cosh(),
            d: self.v.sinh() * self.d,
        }
    }
    fn sin(self) -> Self {
        Dual {
            v: self.v.sin(),
            d: self.v.cos() * self.d,
        }
    }
    fn cos(self) -> Self {
        Dual {
            v: self.v.cos(),
            d: -self.v.sin() * self.d,
        }
    }
}

use crate::assemble::quadrature;
use crate::mesh::Mesh;

/// Integrate a local function of the gradient, and differentiate it with respect
/// to the nodal solution.
///
/// `integrand` receives an element index and the gradient at a quadrature point,
/// and returns the value to integrate. It is called with seeded duals, so it must
/// be written in ordinary arithmetic over [`Dual`] and must not branch on the
/// derivative component.
///
/// Returns the integral and `dJ/du`, one entry per node. Quadrature is the same
/// rule the assembly uses, so the integral and the solve agree about what an
/// element contains.
///
/// The integrand may return [`Dual::ZERO`] to exclude a region — a torque
/// integral over the air gap alone, say — and its sensitivity is then correctly
/// zero there too, without the caller maintaining a separate mask.
pub fn gradient_functional<F>(mesh: &Mesh, u: &[f64], integrand: F) -> (f64, Vec<f64>)
where
    F: Fn(usize, [Dual; 2]) -> Dual,
{
    let order = mesh.order();
    let n_local = order.nodes_per_element();
    let rule = quadrature(order);
    let mut value = 0.0;
    let mut sensitivity = vec![0.0; mesh.nodes.len()];

    for e in 0..mesh.len() {
        let area = mesh.area(e);
        if area <= 0.0 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        for &(l, w) in rule {
            let g = mesh.gradient_at(e, l, u);
            let weight = w * area;

            // One evaluation per input, seeded in turn. Two, because the
            // integrand takes two numbers however complicated it is inside.
            let dx = integrand(e, [Dual::variable(g[0]), Dual::constant(g[1])]);
            let dy = integrand(e, [Dual::constant(g[0]), Dual::variable(g[1])]);
            value += dx.v * weight;

            let shape = mesh.shape_gradients(e, l);
            for i in 0..n_local {
                // Chain the two partials through the shape-function gradient,
                // which is how a nodal value reaches the gradient at this point.
                sensitivity[nodes[i] as usize] +=
                    weight * (dx.d * shape[i][0] + dy.d * shape[i][1]);
            }
        }
    }

    (value, sensitivity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{self, Band, Segment};

    #[test]
    fn a_plain_scalar_carries_no_derivative() {
        assert_eq!(<f64 as Scalar>::constant(2.5).value(), 2.5);
        assert_eq!(<f64 as Scalar>::constant(2.5).tangent(), 0.0);
    }

    #[test]
    fn a_dual_scalar_carries_its_derivative_through_the_trait() {
        let x = Dual::variable(3.0);
        let y: Dual = x * Dual::constant(4.0);
        assert_eq!(Scalar::value(y), 12.0);
        assert_eq!(Scalar::tangent(y), 4.0);
    }

    #[test]
    fn min_and_max_choose_on_value_and_keep_that_branch_derivative() {
        // A clamp must not pick its branch by comparing derivatives: which
        // parameter is being differentiated is not a geometric fact.
        let a = Dual { v: 1.0, d: 99.0 };
        let b = Dual { v: 2.0, d: -3.0 };
        assert_eq!(a.min_of(b), a);
        assert_eq!(a.max_of(b), b);
    }

    #[test]
    fn the_product_rule_holds() {
        let x = Dual::variable(3.0);
        let y = Dual::constant(5.0);
        let p = x * y;
        assert_eq!(p.v, 15.0);
        assert_eq!(p.d, 5.0);
    }

    #[test]
    fn the_quotient_and_power_rules_hold() {
        let x = Dual::variable(4.0);
        let q = Dual::constant(1.0) / x;
        assert!((q.v - 0.25).abs() < 1e-15);
        assert!((q.d + 1.0 / 16.0).abs() < 1e-15);
        let c = x.powi(3);
        assert!((c.v - 64.0).abs() < 1e-12);
        assert!((c.d - 48.0).abs() < 1e-12);
    }

    #[test]
    fn a_square_root_at_zero_does_not_produce_an_infinity() {
        // A field magnitude that happens to vanish is ordinary; an infinite
        // gradient entry from it is not.
        let z = Dual::variable(0.0).sqrt();
        assert_eq!(z.v, 0.0);
        assert!(z.d.is_finite());
    }

    #[test]
    fn duals_agree_with_a_difference_quotient_on_a_composite() {
        // Something with enough structure that a hand derivative would be worth
        // getting wrong: f(x) = sqrt(x^2 + 3x) / (1 + x).
        let f = |x: Dual| (x.powi(2) + x * 3.0).sqrt() / (Dual::constant(1.0) + x);
        let x0 = 2.7;
        let auto = f(Dual::variable(x0)).d;
        // The dual result is exact; the difference quotient is not, and it is
        // the one setting this tolerance. Subtracting two nearby values at
        // h = 1e-6 loses about eight digits to cancellation, so agreement closer
        // than that would say more about luck than correctness.
        let h = 1e-6;
        let fd = (f(Dual::constant(x0 + h)).v - f(Dual::constant(x0 - h)).v) / (2.0 * h);
        assert!((auto - fd).abs() / fd.abs() < 1e-6, "{auto} against {fd}");
    }

    fn unit_domain() -> (Mesh, Vec<u8>) {
        let bands = vec![Band {
            y0: 0.0,
            y1: 1.0,
            segments: vec![Segment {
                x0: 0.0,
                x1: 1.0,
                tag: 0u8,
            }],
            max_dy: 0.25,
        }];
        mesh::layered::build(1.0, bands, 0.25)
    }

    #[test]
    fn a_functional_of_the_gradient_matches_its_hand_derivative() {
        // J = integral (gx * gy). Its sensitivity is known in closed form, so
        // this pins the chain rule through the shape functions.
        let (grid, _) = unit_domain();
        let u: Vec<f64> = grid
            .nodes
            .iter()
            .map(|&[x, y]| 0.3 * x + 0.7 * y + x * y)
            .collect();
        let (value, sens) = gradient_functional(&grid, &u, |_, g| g[0] * g[1]);

        // By hand: dJ/du_k = integral (gy * dN_k/dx + gx * dN_k/dy).
        let rule = quadrature(grid.order());
        let mut expect = vec![0.0; grid.nodes.len()];
        let mut expect_value = 0.0;
        for e in 0..grid.len() {
            let area = grid.area(e);
            let nodes = grid.element_nodes(e);
            for &(l, w) in rule {
                let g = grid.gradient_at(e, l, &u);
                let shape = grid.shape_gradients(e, l);
                expect_value += g[0] * g[1] * w * area;
                for i in 0..grid.order().nodes_per_element() {
                    expect[nodes[i] as usize] +=
                        w * area * (g[1] * shape[i][0] + g[0] * shape[i][1]);
                }
            }
        }
        assert!((value - expect_value).abs() < 1e-12);
        for (a, b) in sens.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-12, "{a} against {b}");
        }
    }

    #[test]
    fn a_functional_sensitivity_matches_perturbing_the_solution() {
        // The end-to-end statement: perturb one nodal value and the functional
        // moves by the predicted amount. Uses a nonlinear integrand, where a
        // hand derivation is most likely to go wrong.
        let (grid, _) = unit_domain();
        let u: Vec<f64> = grid
            .nodes
            .iter()
            .map(|&[x, y]| 0.4 * x - 0.9 * y + 0.2 * x * y)
            .collect();
        let f = |_: usize, g: [Dual; 2]| (g[0] * g[0] + g[1] * g[1]).sqrt() * g[0];
        let (_, sens) = gradient_functional(&grid, &u, f);

        let value_at = |u: &[f64]| gradient_functional(&grid, u, f).0;
        for &node in &[0usize, 7, 13] {
            let h = 1e-6;
            let mut plus = u.clone();
            plus[node] += h;
            let mut minus = u.clone();
            minus[node] -= h;
            let fd = (value_at(&plus) - value_at(&minus)) / (2.0 * h);
            assert!(
                (sens[node] - fd).abs() <= 1e-6 * fd.abs().max(1e-9),
                "node {node}: {} against {fd}",
                sens[node]
            );
        }
    }

    #[test]
    fn a_masked_region_contributes_nothing_to_either_result() {
        // Returning zero for an element must zero its sensitivity too, or a
        // caller integrating over part of a domain gets a gradient from outside
        // the part it integrated.
        let (grid, _) = unit_domain();
        let u: Vec<f64> = grid.nodes.iter().map(|&[x, y]| x + 2.0 * y).collect();
        let (_, all) = gradient_functional(&grid, &u, |_, g| g[0] * g[1]);
        let (_, half) = gradient_functional(&grid, &u, |e, g| {
            if grid.centroid(e)[1] > 0.5 {
                g[0] * g[1]
            } else {
                Dual::ZERO
            }
        });
        assert!(
            all.iter().any(|v| v.abs() > 1e-9),
            "the test field was trivial"
        );
        // Nodes strictly below the mask cannot have been touched.
        for (n, &[_, y]) in grid.nodes.iter().enumerate() {
            if y < 0.4 {
                assert!(half[n].abs() < 1e-15, "node {n} at y={y} got {}", half[n]);
            }
        }
    }
    #[test]
    fn the_transcendental_derivatives_are_the_ones_calculus_gives() {
        // Differenced rather than asserted against hand-written formulas, so the
        // test does not repeat the implementation it is checking. The step is
        // chosen for a central difference on a well-scaled argument: too small
        // and the subtraction loses more than the truncation costs.
        let h = 1e-6;
        let cases: [(&str, fn(Dual) -> Dual, fn(f64) -> f64); 7] = [
            ("sqrt", |x| x.sqrt(), |x| x.sqrt()),
            ("powf(1.5)", |x| x.powf(1.5), |x| x.powf(1.5)),
            ("powf(-2)", |x| x.powf(-2.0), |x| x.powf(-2.0)),
            ("sinh", |x| x.sinh(), |x| x.sinh()),
            ("cosh", |x| x.cosh(), |x| x.cosh()),
            ("sin", |x| x.sin(), |x| x.sin()),
            ("cos", |x| x.cos(), |x| x.cos()),
        ];
        for (name, dual, plain) in cases {
            for at in [0.3f64, 1.0, 2.5] {
                let seeded = dual(Dual { v: at, d: 1.0 });
                let differenced = (plain(at + h) - plain(at - h)) / (2.0 * h);
                let scale = differenced.abs().max(1.0);
                assert!(
                    (seeded.d - differenced).abs() < 1e-6 * scale,
                    "{name} at {at}: {} against {differenced}",
                    seeded.d
                );
                assert!((seeded.v - plain(at)).abs() < 1e-12, "{name} value at {at}");
            }
        }
    }

    #[test]
    fn magnitude_has_the_slope_of_whichever_side_it_is_on() {
        assert_eq!(Dual { v: 3.0, d: 2.0 }.abs().d, 2.0);
        assert_eq!(Dual { v: -3.0, d: 2.0 }.abs().d, -2.0);
        assert_eq!(Dual { v: -3.0, d: 2.0 }.abs().v, 3.0);
        // No derivative exists at the kink, and zero is the choice that keeps a
        // sum of magnitudes from acquiring a slope it has not earned.
        assert_eq!(Dual { v: 0.0, d: 2.0 }.abs().d, 0.0);
    }

    #[test]
    fn the_plain_scalar_agrees_with_the_dual_on_values() {
        // The whole point of the trait is that one expression serves both, so
        // the two must never disagree about the answer itself.
        for at in [0.25f64, 1.0, 3.0] {
            assert!((Scalar::sqrt(at) - Dual::constant(at).sqrt().v).abs() < 1e-15);
            assert!((Scalar::powf(at, 1.5) - Dual::constant(at).powf(1.5).v).abs() < 1e-15);
            assert!((Scalar::sinh(at) - Dual::constant(at).sinh().v).abs() < 1e-15);
            assert!((Scalar::cos(at) - Dual::constant(at).cos().v).abs() < 1e-15);
            assert_eq!(Scalar::tangent(at), 0.0);
        }
    }
}

use crate::assemble::{Constitutive, d_coefficient};
use crate::dof::DofMap;
use crate::mesh::Order;

/// How an element's material data moves with a design parameter.
///
/// Geometry reaches the residual through node positions, which
/// [`residual_tangent`] handles on its own. Material data does not: a remanence
/// or a current density is attached to the element rather than to its shape, so
/// its derivative has to be supplied.
#[derive(Debug, Clone, Copy, Default)]
pub struct ElementTangent {
    /// Derivative of the scalar source.
    pub source: f64,
    /// Derivative of the vector source.
    pub flux_source: [f64; 2],
}

/// Signed area of a triangle whose vertices carry derivatives.
fn dual_area(p: [[Dual; 2]; 3]) -> Dual {
    ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]))
        * Dual::constant(0.5)
}

/// Shape-function gradients from vertices that carry derivatives.
fn dual_shape_gradients(p: [[Dual; 2]; 3], order: Order, l: [f64; 3]) -> [[Dual; 2]; 6] {
    let area = dual_area(p);
    let inv = Dual::constant(1.0) / (area * Dual::constant(2.0));
    let b = [p[1][1] - p[2][1], p[2][1] - p[0][1], p[0][1] - p[1][1]];
    let c = [p[2][0] - p[1][0], p[0][0] - p[2][0], p[1][0] - p[0][0]];
    let dl = [
        [b[0] * inv, c[0] * inv],
        [b[1] * inv, c[1] * inv],
        [b[2] * inv, c[2] * inv],
    ];
    match order {
        Order::P1 => [
            dl[0],
            dl[1],
            dl[2],
            [Dual::ZERO; 2],
            [Dual::ZERO; 2],
            [Dual::ZERO; 2],
        ],
        Order::P2 => {
            let vertex = |i: usize| {
                let k = Dual::constant(4.0 * l[i] - 1.0);
                [k * dl[i][0], k * dl[i][1]]
            };
            let mid = |a: usize, b_: usize| {
                [
                    (dl[b_][0].scale(l[a]) + dl[a][0].scale(l[b_])).scale(4.0),
                    (dl[b_][1].scale(l[a]) + dl[a][1].scale(l[b_])).scale(4.0),
                ]
            };
            [
                vertex(0),
                vertex(1),
                vertex(2),
                mid(0, 1),
                mid(1, 2),
                mid(2, 0),
            ]
        }
    }
}

/// Derivative of the reduced residual with respect to one design parameter.
///
/// # What this replaces
///
/// The implicit half of a shape derivative is `dR/dp`, and it used to be
/// obtained by rebuilding the geometry at `p ± h`, re-meshing both, and
/// differencing the residual. That works, and it carries every drawback a
/// difference quotient has: a step size to choose, precision lost to
/// cancellation, and a cost that grows with the number of parameters.
///
/// Here the same quantity is differentiated. `node_tangents` says how each node
/// moves — produced by running the *same* geometry construction in dual
/// arithmetic, so it cannot disagree with the mesh it describes — and the element
/// residual is assembled in dual arithmetic on top of it. The result is exact to
/// working precision, with no step size anywhere.
///
/// The mesh topology is held fixed throughout, which is what makes the quantity
/// well defined: `dR/dp` is a covector on a particular set of degrees of freedom,
/// and a perturbation that renumbered them would not have one.
pub fn residual_tangent<C: Constitutive>(
    mesh: &Mesh,
    node_tangents: &[[f64; 2]],
    elements: &[C],
    element_tangents: &[ElementTangent],
    dofs: &DofMap,
    u: &[f64],
) -> Vec<f64> {
    assert_eq!(
        node_tangents.len(),
        mesh.nodes.len(),
        "one node tangent per mesh node is required"
    );
    let order = mesh.order();
    let n_local = order.nodes_per_element();
    let rule = quadrature(order);
    let mut out = vec![0.0; dofs.n_free];

    for e in 0..mesh.len() {
        if mesh.area(e) <= 0.0 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let tri = mesh.tris[e];
        // Vertices carrying their motion. Only the three vertices are needed:
        // sides stay straight, so a midside node is the average of its two ends
        // and contributes nothing the vertices do not already carry.
        let vertices = [0usize, 1, 2].map(|k| {
            let n = tri[k] as usize;
            [
                Dual {
                    v: mesh.nodes[n][0],
                    d: node_tangents[n][0],
                },
                Dual {
                    v: mesh.nodes[n][1],
                    d: node_tangents[n][1],
                },
            ]
        });
        let area = dual_area(vertices);

        let element = &elements[e];
        let tangent = element_tangents.get(e).copied().unwrap_or_default();

        for &(l, w) in rule {
            let grad = dual_shape_gradients(vertices, order, l);
            let shape = mesh.shape_values(l, order);
            let scale = area.scale(w);

            // Gradient of the frozen solution, moving only because the shape
            // functions do.
            let mut gu = [Dual::ZERO; 2];
            for i in 0..n_local {
                let value = u[nodes[i] as usize];
                gu[0] = gu[0] + grad[i][0].scale(value);
                gu[1] = gu[1] + grad[i][1].scale(value);
            }
            let grad_sq = gu[0] * gu[0] + gu[1] * gu[1];

            // The coefficient depends on the parameter only through the gradient
            // magnitude, so the chain rule closes with the same derivative the
            // Newton tangent already uses.
            let kappa_v = element.coefficient(grad_sq.v);
            let kappa = if element.is_nonlinear() {
                Dual {
                    v: kappa_v,
                    d: d_coefficient(element, grad_sq.v) * grad_sq.d,
                }
            } else {
                Dual::constant(kappa_v)
            };

            let f = Dual {
                v: element.source(),
                d: tangent.source,
            };
            let s_v = element.flux_source(grad_sq.v);
            let s = [
                Dual {
                    v: s_v[0],
                    d: tangent.flux_source[0],
                },
                Dual {
                    v: s_v[1],
                    d: tangent.flux_source[1],
                },
            ];

            for i in 0..n_local {
                let Some((row, si)) = dofs.resolve(nodes[i]) else {
                    continue;
                };
                let g = gu[0] * grad[i][0] + gu[1] * grad[i][1];
                let load = f.scale(shape[i]) + s[0] * grad[i][0] + s[1] * grad[i][1];
                out[row] += si * (scale * (kappa * g - load)).d;
            }
        }
    }

    out
}
