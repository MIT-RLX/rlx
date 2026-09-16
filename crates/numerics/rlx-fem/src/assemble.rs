// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Element assembly and the damped Newton driver.

use crate::dof::DofMap;
use crate::mesh::{Mesh, Order};
use crate::solve::{Csr, SolveReport, pcg};

/// The physics of one element.
///
/// Implemented by the caller for whatever it calls a material. Nothing here
/// names a physical quantity: `coefficient` is a reluctivity in magnetostatics,
/// a conductivity in heat conduction, a permittivity in electrostatics.
pub trait Constitutive {
    /// Coefficient at a given squared gradient magnitude.
    ///
    /// Takes `|grad u|^2` rather than `|grad u|` because the caller always has
    /// the square already — it is assembled from two gradient components — and
    /// the square root is measurable when it happens once per element per Newton
    /// step.
    fn coefficient(&self, grad_sq: f64) -> f64;

    /// Whether the coefficient depends on the solution.
    ///
    /// An element that says no is never differentiated and never re-evaluated,
    /// and a mesh where every element says no is solved in exactly one pass.
    fn is_nonlinear(&self) -> bool {
        true
    }

    /// Scalar source `f`, integrated against the test function.
    fn source(&self) -> f64 {
        0.0
    }

    /// Vector source `s`, integrated against the test function's gradient.
    ///
    /// This is the term that makes a region drive the field without carrying a
    /// source density: a permanent magnet's remanence, a prescribed heat flux, a
    /// frozen-in polarisation.
    fn flux_source(&self, _grad_sq: f64) -> [f64; 2] {
        [0.0, 0.0]
    }
}

/// Quadrature points in area coordinates, with weights summing to one.
///
/// Public because a caller integrating its own functional of the solution — an
/// energy, a force, a stress resultant — has to use the same rule the assembly
/// did, or its integral and the solve will disagree about what the element
/// contains.
///
/// A linear element needs one point: its integrands are constant, so the
/// centroid rule is exact and reproduces the closed-form element matrices
/// rather than approximating them.
///
/// A quadratic element carries a gradient that varies linearly, so the stiffness
/// integrand is quadratic — and the Newton tangent's rank-one term, which
/// multiplies two of those gradients together, is quartic. The six-point rule
/// below is exact to degree four, which covers the tangent as well as the
/// stiffness and keeps quadrature out of the error budget entirely.
pub fn quadrature(order: Order) -> &'static [([f64; 3], f64)] {
    const P1: &[([f64; 3], f64)] = &[([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1.0)];
    const A1: f64 = 0.445_948_490_915_965;
    const W1: f64 = 0.223_381_589_678_011;
    const A2: f64 = 0.091_576_213_509_771;
    const W2: f64 = 0.109_951_743_655_322;
    const P2: &[([f64; 3], f64)] = &[
        ([1.0 - 2.0 * A1, A1, A1], W1),
        ([A1, 1.0 - 2.0 * A1, A1], W1),
        ([A1, A1, 1.0 - 2.0 * A1], W1),
        ([1.0 - 2.0 * A2, A2, A2], W2),
        ([A2, 1.0 - 2.0 * A2, A2], W2),
        ([A2, A2, 1.0 - 2.0 * A2], W2),
    ];
    match order {
        Order::P1 => P1,
        Order::P2 => P2,
    }
}

/// Assemble the reduced stiffness matrix and load vector at fixed coefficients.
///
/// `coefficient` holds one value per element, which the Newton driver updates
/// between passes. Useful directly for a linear problem, where it is the whole
/// solve.
pub fn assemble<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coefficient: &[f64],
) -> (Csr, Vec<f64>) {
    assemble_coupled(mesh, elements, dofs, coefficient, &[])
}

/// As [`assemble`], with extra couplings folded into the same matrix.
///
/// `coupling` carries entries a region contributes without having elements —
/// [`crate::airgap`] is the case this exists for. They join the mesh's own
/// triplets before the matrix is built, so the result is one system solved by
/// one solver, and a coupling that is symmetric leaves it symmetric.
pub fn assemble_coupled<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coefficient: &[f64],
    coupling: &[(usize, usize, f64)],
) -> (Csr, Vec<f64>) {
    let order = mesh.order();
    let n_local = order.nodes_per_element();
    let rule = quadrature(order);
    let mut triplets = Vec::with_capacity(n_local * n_local * mesh.len());
    let mut rhs = vec![0.0; dofs.n_free];

    for e in 0..mesh.len() {
        let area = mesh.area(e);
        if area <= 0.0 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let kappa = coefficient[e];
        let element = &elements[e];
        let f = element.source();
        let s = element.flux_source(0.0);

        for &(l, w) in rule {
            let scale = w * area;
            let grad = mesh.shape_gradients(e, l);
            let shape = mesh.shape_values(l, order);

            for i in 0..n_local {
                let Some((row, si)) = dofs.resolve(nodes[i]) else {
                    continue;
                };
                // A scalar source against the shape function, plus a vector
                // source contracted with its gradient.
                let load = f * shape[i] + s[0] * grad[i][0] + s[1] * grad[i][1];
                rhs[row] += si * scale * load;

                for k in 0..n_local {
                    let Some((col, sk)) = dofs.resolve(nodes[k]) else {
                        continue;
                    };
                    let ke = kappa * (grad[i][0] * grad[k][0] + grad[i][1] * grad[k][1]);
                    triplets.push((row, col, si * sk * scale * ke));
                }
            }
        }
    }

    triplets.extend_from_slice(coupling);
    (Csr::from_triplets(dofs.n_free, triplets), rhs)
}

/// Element-wise `d(coefficient)/d(grad_sq)`, by central difference.
///
/// Only the *rate* of Newton convergence depends on this being accurate — the
/// converged answer is fixed by the residual, which is exact — so a difference
/// quotient is safe and spares every implementor of [`Constitutive`] from
/// carrying a hand-derived derivative that could silently disagree with its own
/// curve.
pub(crate) fn d_coefficient<C: Constitutive>(element: &C, grad_sq: f64) -> f64 {
    let h = (1e-6 * grad_sq).max(1e-12);
    let lo_at = (grad_sq - h).max(0.0);
    let hi = element.coefficient(grad_sq + h);
    let lo = element.coefficient(lo_at);
    (hi - lo) / (grad_sq + h - lo_at)
}

/// Assemble the Newton tangent and the negative residual at state `u`.
///
/// With `g_i = grad(u) . grad(N_i)`, the residual is
/// `R_i = integral kappa * g_i - f_i`, and differentiating gives
///
/// ```text
/// J_ij = integral [ kappa * grad(N_i) . grad(N_j)  +  2 * kappa' * g_i * g_j ]
/// ```
///
/// The first term is the ordinary stiffness — what successive substitution uses
/// on its own. The second is a rank-one update per element, symmetric and, where
/// the coefficient rises with gradient, positive semi-definite. So the tangent
/// stays symmetric positive definite and the same conjugate gradient solves it.
/// That is why Newton is affordable here: one extra outer product per element
/// and no change of linear solver.
pub fn assemble_newton<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    u: &[f64],
) -> (Csr, Vec<f64>) {
    assemble_newton_coupled(mesh, elements, dofs, u, &[])
}

/// As [`assemble_newton`], with extra couplings.
///
/// A coupling is linear, so it enters the tangent unchanged and the residual as
/// its action on the current state. Both are needed: adding it to the tangent
/// alone would leave Newton converging to the wrong fixed point.
pub fn assemble_newton_coupled<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    u: &[f64],
    coupling: &[(usize, usize, f64)],
) -> (Csr, Vec<f64>) {
    let order = mesh.order();
    let n_local = order.nodes_per_element();
    let rule = quadrature(order);
    let mut triplets = Vec::with_capacity(n_local * n_local * mesh.len());
    let mut neg_residual = vec![0.0; dofs.n_free];

    for e in 0..mesh.len() {
        let area = mesh.area(e);
        if area <= 0.0 {
            continue;
        }
        let nodes = mesh.element_nodes(e);
        let element = &elements[e];
        let f = element.source();

        for &(l, w) in rule {
            let scale = w * area;
            let grad = mesh.shape_gradients(e, l);
            let shape = mesh.shape_values(l, order);

            // Gradient of the current state at this point.
            let mut gu = [0.0, 0.0];
            for i in 0..n_local {
                let value = u[nodes[i] as usize];
                gu[0] += value * grad[i][0];
                gu[1] += value * grad[i][1];
            }
            let grad_sq = gu[0] * gu[0] + gu[1] * gu[1];

            let kappa = element.coefficient(grad_sq);
            let dkappa = if element.is_nonlinear() {
                d_coefficient(element, grad_sq)
            } else {
                0.0
            };
            let s = element.flux_source(grad_sq);

            let mut g = [0.0; 6];
            for i in 0..n_local {
                g[i] = gu[0] * grad[i][0] + gu[1] * grad[i][1];
            }

            for i in 0..n_local {
                let Some((row, si)) = dofs.resolve(nodes[i]) else {
                    continue;
                };
                let load = f * shape[i] + s[0] * grad[i][0] + s[1] * grad[i][1];
                neg_residual[row] += si * scale * (load - kappa * g[i]);

                for k in 0..n_local {
                    let Some((col, sk)) = dofs.resolve(nodes[k]) else {
                        continue;
                    };
                    let tangent = kappa * (grad[i][0] * grad[k][0] + grad[i][1] * grad[k][1])
                        + 2.0 * dkappa * g[i] * g[k];
                    triplets.push((row, col, si * sk * scale * tangent));
                }
            }
        }
    }

    // The coupling's own contribution to the residual, `-K_gap * u`, taken on
    // the reduced unknowns.
    if !coupling.is_empty() {
        let reduced = dofs.reduce_state(u);
        for &(r, c, v) in coupling {
            neg_residual[r] -= v * reduced[c];
        }
        triplets.extend_from_slice(coupling);
    }
    (Csr::from_triplets(dofs.n_free, triplets), neg_residual)
}

/// Infinity norm of the reduced residual at a given state.
fn residual_norm<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    u: &[f64],
    coupling: &[(usize, usize, f64)],
) -> f64 {
    let (_, neg_r) = assemble_newton_coupled(mesh, elements, dofs, u, coupling);
    neg_r.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Controls for the nonlinear solve.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// Relative residual below which the Newton loop stops.
    pub tolerance: f64,
    /// Cap on Newton passes.
    pub max_iterations: usize,
    /// Smallest accepted backtracking factor before a step is judged to have no
    /// descent direction left.
    pub min_step: f64,
    /// Relative residual for the inner linear solve.
    pub linear_tolerance: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            tolerance: 1e-6,
            max_iterations: 40,
            min_step: 1.0 / 4096.0,
            linear_tolerance: 1e-10,
        }
    }
}

/// A converged (or abandoned) solve.
#[derive(Debug, Clone)]
pub struct Solution {
    /// Potential per node.
    pub u: Vec<f64>,
    /// Gradient of the potential per element.
    pub gradient: Vec<[f64; 2]>,
    /// Coefficient per element at the final state.
    pub coefficient: Vec<f64>,
    /// Newton passes taken.
    pub iterations: usize,
    /// Final relative residual.
    pub residual: f64,
    /// Relative residual after each Newton pass.
    ///
    /// Kept because a converged answer and a *well* converged answer look the
    /// same from the outside. The history shows whether Newton took its
    /// quadratic steps or crawled on backtracked ones, which is the difference
    /// between a solve that can be trusted and one that happened to stop.
    pub residual_history: Vec<f64>,
    /// Whether the Newton loop met its tolerance.
    pub converged: bool,
    /// Report from the last inner linear solve.
    pub linear: SolveReport,
}

/// Solve the problem posed by `mesh`, `elements` and `dofs`.
///
/// A mesh whose elements are all linear takes exactly one pass; the Newton loop
/// runs only where some element reports itself nonlinear, which is checked
/// rather than assumed.
pub fn solve<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    opts: SolverOptions,
) -> Solution {
    solve_coupled(mesh, elements, dofs, &[], opts)
}

/// A stiffness matrix with off-mesh couplings added to it.
///
/// The couplings are triplets in the reduced numbering, the same ones
/// [`assemble_newton_coupled`] adds to its tangent.
pub(crate) fn with_coupling(k: Csr, coupling: &[(usize, usize, f64)]) -> Csr {
    if coupling.is_empty() {
        return k;
    }
    let mut triplets = Vec::with_capacity(k.values.len() + coupling.len());
    for row in 0..k.n {
        for idx in k.indptr[row]..k.indptr[row + 1] {
            triplets.push((row, k.indices[idx], k.values[idx]));
        }
    }
    triplets.extend_from_slice(coupling);
    Csr::from_triplets(k.n, triplets)
}

/// As [`solve`], with a region carried by a coupling rather than by elements.
///
/// See [`crate::airgap`]: a gap can be replaced by an exact relation between its
/// two boundaries, which enters here and nowhere else. The Newton loop, the
/// backtracking and the linear solver are unchanged, because a symmetric
/// coupling leaves the system symmetric positive definite.
pub fn solve_coupled<C: Constitutive>(
    mesh: &Mesh,
    elements: &[C],
    dofs: &DofMap,
    coupling: &[(usize, usize, f64)],
    opts: SolverOptions,
) -> Solution {
    assert_eq!(
        elements.len(),
        mesh.len(),
        "one constitutive model per element is required"
    );
    let nonlinear = elements.iter().any(Constitutive::is_nonlinear);
    let initial: Vec<f64> = elements.iter().map(|e| e.coefficient(0.0)).collect();

    // The linear solve is both the answer for a linear mesh and the starting
    // point for a nonlinear one — a better opening state than zero, which
    // evaluates every coefficient at a gradient of zero and then takes its first
    // step from there.
    //
    // The coupling belongs in it. For a nonlinear mesh leaving it out is merely
    // a worse starting point, since the Newton passes below assemble it and
    // recover; for a linear mesh there are no passes, and dropping it returns a
    // field solved as though the coupled regions were not coupled at all. That
    // failure is silent — the solve converges, and reports it — and where the
    // coupling stands in for a vacated air gap it is the difference between a
    // machine and two halves of one.
    let (k, f) = assemble(mesh, elements, dofs, &initial);
    let k = with_coupling(k, coupling);
    let inner_cap = 20 * k.n.max(1);
    let (reduced, mut linear) = pcg(&k, &f, opts.linear_tolerance, inner_cap);
    let mut u = dofs.expand(&reduced);

    let mut residual = 0.0;
    let mut iterations = 0;
    let mut residual_history = Vec::new();

    if nonlinear {
        let scale = f
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let mut r_now = residual_norm(mesh, elements, dofs, &u, coupling);

        for pass in 1..=opts.max_iterations {
            iterations = pass;
            residual = r_now / scale;
            residual_history.push(residual);
            if residual < opts.tolerance {
                break;
            }

            let (jac, neg_r) = assemble_newton_coupled(mesh, elements, dofs, &u, coupling);
            let (delta, report) = pcg(&jac, &neg_r, opts.linear_tolerance, inner_cap);
            linear = report;
            let step = dofs.expand(&delta);

            // Full Newton steps are right near the solution. Far from it — with
            // an element crossing from one coefficient regime to another in a
            // single step — they overshoot into a state whose residual is worse
            // than where they started. Backtracking until the residual actually
            // falls is what makes a strongly nonlinear coefficient solvable at
            // all; undamped Newton and successive substitution both oscillate
            // and never converge.
            let mut alpha = 1.0;
            let mut accepted = false;
            while alpha >= opts.min_step {
                let trial: Vec<f64> = u.iter().zip(&step).map(|(a, d)| a + alpha * d).collect();
                let r_trial = residual_norm(mesh, elements, dofs, &trial, coupling);
                if r_trial < r_now {
                    u = trial;
                    r_now = r_trial;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                // No descent direction left; report where it stalled rather than
                // spending the remaining passes to arrive at the same place.
                break;
            }
            residual = r_now / scale;
        }
        residual_history.push(residual);
    }

    let gradient = mesh.gradient(&u);
    let coefficient: Vec<f64> = elements
        .iter()
        .zip(&gradient)
        .map(|(el, g)| el.coefficient(g[0] * g[0] + g[1] * g[1]))
        .collect();

    Solution {
        u,
        gradient,
        coefficient,
        iterations,
        residual,
        residual_history,
        converged: residual <= opts.tolerance,
        linear,
    }
}
