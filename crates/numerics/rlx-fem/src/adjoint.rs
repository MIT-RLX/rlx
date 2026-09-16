// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adjoint sensitivities: the gradient of an objective with respect to every
//! parameter, for the price of one extra linear solve.
//!
//! # Why this is the whole point
//!
//! For a converged state `u` satisfying `R(u, p) = 0` and an objective
//! `J(u, p)`, the total derivative is
//!
//! ```text
//! dJ/dp = pJ/pp  +  (pJ/pu)^T * du/dp,      du/dp = -(pR/pu)^-1 * pR/pp
//! ```
//!
//! writing `p` for partial derivatives. Evaluating that directly costs one
//! linearised solve *per parameter*. Grouping it the other way,
//!
//! ```text
//! dJ/dp = pJ/pp  -  lambda^T * pR/pp,       (pR/pu)^T * lambda = pJ/pu
//! ```
//!
//! costs one solve *in total*, and then a cheap contraction per parameter. With
//! twenty design variables that is twenty solves against one.
//!
//! Two properties of this crate's discretisation make it cheaper still. The
//! tangent `pR/pu` is symmetric, so `(pR/pu)^T` is the same operator already
//! assembled for Newton and the same conjugate gradient solves it — there is no
//! transpose to form. And `pR/pp` needs no solve at all: it is a residual
//! evaluation at frozen `u`, which costs one assembly pass.
//!
//! # What the caller supplies
//!
//! Parameter derivatives stay with the caller, because only the caller knows
//! what a parameter *is* — a magnet strength that enters one term of one
//! element, or a radius that moves every node in the mesh. [`residual`] and
//! [`contract`] are the two pieces needed to build them, by whatever mix of
//! analytic and difference quotients suits each parameter.
//!
//! When a parameter perturbs geometry, the mesh must keep its topology: `pR/pp`
//! is a covector on a fixed set of degrees of freedom, and a perturbation that
//! renumbers the nodes does not have one. A structured mesh whose subdivision
//! counts are stable under small changes — see [`crate::mesh::layered`] — gives
//! that property by construction, where a remeshed unstructured domain does not.

use crate::assemble::{Constitutive, SolverOptions, assemble_newton};
use crate::dof::DofMap;
use crate::mesh::Mesh;
use crate::solve::{SolveReport, pcg};

/// A solved adjoint state.
#[derive(Debug, Clone)]
pub struct Adjoint {
    /// Adjoint variable on the reduced system.
    pub lambda: Vec<f64>,
    /// Report from the adjoint solve.
    pub report: SolveReport,
}

/// The reduced residual `R(u)` at a given state.
///
/// Zero to solver tolerance at a converged state, which is what makes it useful
/// as a *sensitivity* probe: re-evaluated with a parameter perturbed and `u`
/// held fixed, it is `pR/pp` times the perturbation, with no solve involved.
pub fn residual<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    u: &[f64],
) -> Vec<f64> {
    let (_, neg_residual) = assemble_newton(mesh, elements, dofs, u);
    neg_residual.into_iter().map(|v| -v).collect()
}

/// Residual of a system whose region is carried by a coupling rather than by
/// elements.
///
/// The coupling is linear in the state — it stands in for a region that has been
/// removed from the mesh, and contributes `C u` to the residual and `C` to the
/// tangent. Leaving it out of either makes the sensitivity describe a different
/// problem from the one that was solved, which is silent: both converge, and
/// they answer about machines that differ by however much the coupled region
/// was worth.
pub fn residual_coupled<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coupling: &[(usize, usize, f64)],
    u: &[f64],
) -> Vec<f64> {
    let mut r = residual(mesh, elements, dofs, u);
    if coupling.is_empty() {
        return r;
    }
    let reduced = dofs.reduce(u);
    for &(row, col, value) in coupling {
        if let (Some(target), Some(state)) = (r.get_mut(row), reduced.get(col)) {
            *target += value * state;
        }
    }
    r
}

/// As [`adjoint_coupled`], with the objective's sensitivity given in the
/// *reduced* space rather than per node.
///
/// A functional written directly on the reduced state — a quadratic form in the
/// degrees of freedom, say — has a gradient that is naturally reduced already,
/// and there is no general way to expand it back into a nodal vector whose
/// reduction returns it. Ties sum, so the expansion is not unique.
pub fn adjoint_coupled_reduced<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coupling: &[(usize, usize, f64)],
    u: &[f64],
    dj_dq: &[f64],
    opts: SolverOptions,
) -> Adjoint {
    let (tangent, _) = assemble_newton(mesh, elements, dofs, u);
    let tangent = crate::assemble::with_coupling(tangent, coupling);
    let cap = 20 * tangent.n.max(1);
    let (lambda, report) = pcg(&tangent, dj_dq, opts.linear_tolerance, cap);
    Adjoint { lambda, report }
}

/// As [`adjoint`], for a system with a coupling.
///
/// The tangent of a coupled system is the element tangent plus the coupling,
/// and the coupling is symmetric where it comes from a Dirichlet-to-Neumann map,
/// so the adjoint system is the same matrix the forward solve used.
pub fn adjoint_coupled<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coupling: &[(usize, usize, f64)],
    u: &[f64],
    dj_du: &[f64],
    opts: SolverOptions,
) -> Adjoint {
    let (tangent, _) = assemble_newton(mesh, elements, dofs, u);
    let tangent = crate::assemble::with_coupling(tangent, coupling);
    let rhs = dofs.reduce(dj_du);
    let cap = 20 * tangent.n.max(1);
    let (lambda, report) = pcg(&tangent, &rhs, opts.linear_tolerance, cap);
    Adjoint { lambda, report }
}

/// Solve the adjoint system for an objective whose sensitivity to the nodal
/// state is `dj_du`, given per node.
///
/// The tangent is taken at `u`, so `u` should be a converged state; at any other
/// point the result is the sensitivity of a problem that is not the one being
/// solved.
pub fn adjoint<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    u: &[f64],
    dj_du: &[f64],
    opts: SolverOptions,
) -> Adjoint {
    let (tangent, _) = assemble_newton(mesh, elements, dofs, u);
    let rhs = dofs.reduce(dj_du);
    let cap = 20 * tangent.n.max(1);
    let (lambda, report) = pcg(&tangent, &rhs, opts.linear_tolerance, cap);
    Adjoint { lambda, report }
}

/// Contract an adjoint state with a residual sensitivity.
///
/// Returns `-lambda^T * dR/dp`, which is the implicit part of `dJ/dp`. Add the
/// explicit part `pJ/pp` — the objective's own direct dependence on the
/// parameter — to obtain the total derivative.
///
/// # Panics
///
/// If the two vectors differ in length.
pub fn contract(lambda: &[f64], dr_dp: &[f64]) -> f64 {
    assert_eq!(
        lambda.len(),
        dr_dp.len(),
        "adjoint and residual sensitivity must live on the same reduced system"
    );
    -lambda.iter().zip(dr_dp).map(|(l, r)| l * r).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::solve;
    use crate::mesh::{self, Band, Segment};

    #[derive(Clone)]
    struct Mat {
        lo: f64,
        hi: f64,
        knee_sq: f64,
        f: f64,
    }

    impl Constitutive for Mat {
        fn coefficient(&self, grad_sq: f64) -> f64 {
            if self.hi == self.lo {
                self.lo
            } else {
                self.lo + (self.hi - self.lo) * grad_sq / (grad_sq + self.knee_sq)
            }
        }
        fn is_nonlinear(&self) -> bool {
            self.hi != self.lo
        }
        fn source(&self) -> f64 {
            self.f
        }
    }

    /// Domain, solved for a given source strength.
    fn build(f: f64, nonlinear: bool) -> (mesh::Mesh, Vec<Mat>, DofMap) {
        let (width, height) = (0.02, 0.01);
        let tag = Mat {
            lo: 1.0,
            hi: if nonlinear { 6.0 } else { 1.0 },
            knee_sq: 4.0e6,
            f,
        };
        let bands = vec![Band {
            y0: 0.0,
            y1: height,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag,
            }],
            max_dy: height / 10.0,
        }];
        let (grid, elements) = mesh::layered::build(width, bands, width / 8.0);
        let dofs = DofMap::builder(grid.nodes.len())
            .fix(&grid.bottom_edge)
            .fix(&grid.top_edge)
            .tie(&grid.left_edge, &grid.right_edge, 1.0)
            .build();
        (grid, elements, dofs)
    }

    /// A symmetric coupling between the two horizontal edges, standing in for a
    /// region that has been removed from the mesh. Any symmetric positive
    /// contribution will do: the point is that it changes the answer, so that a
    /// sensitivity which ignores it describes a different problem.
    fn coupling_of(grid: &mesh::Mesh, dofs: &DofMap, strength: f64) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for node in grid.left_edge.iter() {
            if let Some((row, _)) = dofs.resolve(*node) {
                out.push((row, row, strength));
            }
        }
        out
    }

    #[test]
    fn a_coupled_adjoint_matches_a_difference_quotient_of_the_coupled_solve() {
        // The forward solve carries the coupling and, until now, the adjoint did
        // not. Both converge either way, so the mismatch is silent: the
        // sensitivity simply answers about a different problem. This pins them
        // to each other.
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let f = 1.0e6;
        let (grid, elements, dofs) = build(f, true);
        let coupling = coupling_of(&grid, &dofs, 5.0e3);
        assert!(
            !coupling.is_empty(),
            "the fixture must actually couple something"
        );

        let sol = crate::assemble::solve_coupled(&grid, &elements, &dofs, &coupling, opts);
        assert!(sol.converged);

        let dj_du = vec![1.0; grid.nodes.len()];
        let adj = adjoint_coupled(&grid, &elements, &dofs, &coupling, &sol.u, &dj_du, opts);
        assert!(adj.report.converged);

        let h = f * 1e-6;
        let (_, plus, _) = build(f + h, true);
        let (_, minus, _) = build(f - h, true);
        let r_plus = residual_coupled(&grid, &plus, &dofs, &coupling, &sol.u);
        let r_minus = residual_coupled(&grid, &minus, &dofs, &coupling, &sol.u);
        let dr_dp: Vec<f64> = r_plus
            .iter()
            .zip(&r_minus)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();
        let by_adjoint = contract(&adj.lambda, &dr_dp);

        // The honest check: re-solve the coupled system either side and difference.
        let solve_at = |source: f64| {
            let (g, e, d) = build(source, true);
            let c = coupling_of(&g, &d, 5.0e3);
            objective(&crate::assemble::solve_coupled(&g, &e, &d, &c, opts).u)
        };
        let numeric = (solve_at(f + h) - solve_at(f - h)) / (2.0 * h);
        assert!(
            (by_adjoint - numeric).abs() < 1e-5 * numeric.abs().max(1.0),
            "coupled adjoint {by_adjoint} against difference {numeric}"
        );

        // And the uncoupled adjoint does *not* match, which is what makes the
        // coupled one worth having rather than an equivalent spelling.
        let bare = adjoint(&grid, &elements, &dofs, &sol.u, &dj_du, opts);
        let r_plus_bare = residual(&grid, &plus, &dofs, &sol.u);
        let r_minus_bare = residual(&grid, &minus, &dofs, &sol.u);
        let dr_bare: Vec<f64> = r_plus_bare
            .iter()
            .zip(&r_minus_bare)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();
        let ignoring = contract(&bare.lambda, &dr_bare);
        assert!(
            (ignoring - numeric).abs() > 1e-3 * numeric.abs(),
            "ignoring the coupling should give a different answer, got {ignoring} \
             against {numeric}"
        );
    }

    /// Objective: the sum of the nodal state. Its sensitivity is all ones, which
    /// keeps the test about the adjoint rather than about the objective.
    fn objective(u: &[f64]) -> f64 {
        u.iter().sum()
    }

    fn gradient_by_adjoint(f: f64, nonlinear: bool) -> f64 {
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let (grid, elements, dofs) = build(f, nonlinear);
        let sol = solve(&grid, &elements, &dofs, opts);
        assert!(sol.converged, "state did not converge");

        let dj_du = vec![1.0; grid.nodes.len()];
        let adj = adjoint(&grid, &elements, &dofs, &sol.u, &dj_du, opts);
        assert!(adj.report.converged, "adjoint did not converge");

        // dR/df by a difference quotient on the residual at frozen u. No solve.
        let h = f * 1e-6;
        let (_, plus, _) = build(f + h, nonlinear);
        let (_, minus, _) = build(f - h, nonlinear);
        let r_plus = residual(&grid, &plus, &dofs, &sol.u);
        let r_minus = residual(&grid, &minus, &dofs, &sol.u);
        let dr_df: Vec<f64> = r_plus
            .iter()
            .zip(&r_minus)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect();

        // The objective has no direct dependence on f, so the contraction is all
        // of the total derivative.
        contract(&adj.lambda, &dr_df)
    }

    fn gradient_by_resolving(f: f64, nonlinear: bool) -> f64 {
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let h = f * 1e-5;
        let value = |fv: f64| {
            let (grid, elements, dofs) = build(fv, nonlinear);
            objective(&solve(&grid, &elements, &dofs, opts).u)
        };
        (value(f + h) - value(f - h)) / (2.0 * h)
    }

    #[test]
    fn adjoint_matches_refinement_of_the_whole_solve_when_linear() {
        let f = 5.0e6;
        let adj = gradient_by_adjoint(f, false);
        let fd = gradient_by_resolving(f, false);
        assert!(adj.abs() > 0.0, "gradient was trivially zero");
        assert!(
            (adj - fd).abs() / fd.abs() < 1e-6,
            "adjoint {adj} against re-solved {fd}"
        );
    }

    #[test]
    fn adjoint_matches_refinement_of_the_whole_solve_when_nonlinear() {
        // The case that exercises the tangent: the coefficient moves with the
        // solution, so a gradient taken through a frozen-coefficient operator
        // would be wrong here and right in the linear test above.
        let f = 5.0e6;
        let adj = gradient_by_adjoint(f, true);
        let fd = gradient_by_resolving(f, true);
        assert!(adj.abs() > 0.0, "gradient was trivially zero");
        assert!(
            (adj - fd).abs() / fd.abs() < 1e-4,
            "adjoint {adj} against re-solved {fd}"
        );
    }

    #[test]
    fn a_linear_objective_has_an_exact_analytic_gradient() {
        // For a linear problem the state is proportional to the source, so the
        // objective is too and its derivative is the objective divided by f.
        let f = 5.0e6;
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let (grid, elements, dofs) = build(f, false);
        let value = objective(&solve(&grid, &elements, &dofs, opts).u);
        let adj = gradient_by_adjoint(f, false);
        assert!(
            (adj - value / f).abs() / (value / f).abs() < 1e-6,
            "adjoint {adj} against analytic {}",
            value / f
        );
    }

    #[test]
    fn the_residual_vanishes_at_a_converged_state() {
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let (grid, elements, dofs) = build(5.0e6, true);
        let sol = solve(&grid, &elements, &dofs, opts);
        let r = residual(&grid, &elements, &dofs, &sol.u);
        let peak = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // Scaled against the load, since the residual carries its units.
        let (_, load) = assemble_newton(&grid, &elements, &dofs, &vec![0.0; grid.nodes.len()]);
        let scale = load.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak / scale < 1e-6, "residual {peak} against load {scale}");
    }
}
