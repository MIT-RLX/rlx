// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Verification against problems with closed-form solutions.
//!
//! Where possible each case is arranged so the exact solution lies *inside* the
//! P1 space — piecewise linear in `y`, constant in `x` — which makes the
//! discretisation error zero and lets the assertions run at 1e-10 rather than at
//! a chosen percentage. A tolerance of five percent does not detect a sign error
//! in the flux-source term; this one does. The single case whose solution is not
//! representable is checked on convergence rate instead.

use rlx_fem::assemble::{Constitutive, SolverOptions, solve};
use rlx_fem::dof::DofMap;
use rlx_fem::mesh::{self, Band, Order, Segment};

/// A coefficient that rises from `lo` towards `hi` as the gradient grows, plus
/// the two source terms. Linear when `lo == hi`.
#[derive(Clone)]
struct Mat {
    lo: f64,
    hi: f64,
    knee_sq: f64,
    f: f64,
    s: [f64; 2],
}

impl Mat {
    fn linear(kappa: f64) -> Mat {
        Mat {
            lo: kappa,
            hi: kappa,
            knee_sq: 1.0,
            f: 0.0,
            s: [0.0, 0.0],
        }
    }
    fn with_source(mut self, f: f64) -> Mat {
        self.f = f;
        self
    }
    fn with_flux(mut self, s: [f64; 2]) -> Mat {
        self.s = s;
        self
    }
    fn saturating(lo: f64, hi: f64, knee: f64) -> Mat {
        Mat {
            lo,
            hi,
            knee_sq: knee * knee,
            f: 0.0,
            s: [0.0, 0.0],
        }
    }
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
    fn flux_source(&self, _grad_sq: f64) -> [f64; 2] {
        self.s
    }
}

/// Fixed top and bottom, sides tied with `factor`.
fn pinned(grid: &mesh::Mesh, factor: f64) -> DofMap {
    DofMap::builder(grid.nodes.len())
        .fix(&grid.bottom_edge)
        .fix(&grid.top_edge)
        .tie(&grid.left_edge, &grid.right_edge, factor)
        .build()
}

#[test]
fn a_uniform_source_converges_at_second_order() {
    // -kappa * u'' = f with u(0) = u(H) = 0 has the parabola
    //     u(y) = f * y * (H - y) / (2 * kappa)
    // as its exact solution. A parabola is not in the piecewise-linear space,
    // so nodal values are approximations; the one-dimensional superconvergence
    // that would make them exact does not survive triangulation. What must hold
    // is the rate: halving the element size quarters the error. That is the
    // property that fails if the load vector or the stiffness carries a wrong
    // factor, and it holds for no other reason.
    let (width, height, kappa, f) = (0.02, 0.01, 3.0, 5.0e6);

    let error_at = |divisions: usize| {
        let bands = vec![Band {
            y0: 0.0,
            y1: height,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::linear(kappa).with_source(f),
            }],
            max_dy: height / divisions as f64,
        }];
        let (grid, elements) = mesh::layered::build(width, bands, width / divisions as f64);
        let dofs = pinned(&grid, 1.0);
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let sol = solve(&grid, &elements, &dofs, opts);
        assert!(sol.linear.converged, "linear solve: {:?}", sol.linear);

        let peak = f * height * height / (8.0 * kappa);
        grid.nodes
            .iter()
            .enumerate()
            .map(|(n, &[_, y])| (sol.u[n] - f * y * (height - y) / (2.0 * kappa)).abs())
            .fold(0.0, f64::max)
            / peak
    };

    let coarse = error_at(8);
    let fine = error_at(16);
    let finer = error_at(32);
    let rate1 = (coarse / fine).log2();
    let rate2 = (fine / finer).log2();
    assert!(
        (1.7..2.3).contains(&rate1) && (1.7..2.3).contains(&rate2),
        "rates {rate1:.3}, {rate2:.3} from errors {coarse:.3e}, {fine:.3e}, {finer:.3e}"
    );
    assert!(finer < 1e-3, "error still {finer:.3e} at the finest mesh");
}

/// The two-band flux-source circuit and its exact gradients.
///
/// A band carrying a flux source drives the field one way through itself and
/// back through the band above it. With `C = -(s*h1/k1) / (h1/k1 + h2/k2)`,
/// the gradients are `(s + C)/k1` and `C/k2`, opposed and carrying equal flux.
fn circuit(k1: f64, h1: f64, k2: f64, h2: f64, s: f64) -> (f64, f64) {
    let c = -(s * h1 / k1) / (h1 / k1 + h2 / k2);
    ((s + c) / k1, c / k2)
}

#[test]
fn a_flux_source_matches_the_series_circuit() {
    let (width, k1, h1, k2, h2, s) = (0.02, 2.0, 0.005, 7.0, 0.002, 11.0);
    let bands = vec![
        Band {
            y0: 0.0,
            y1: h1,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::linear(k1).with_flux([0.0, s]),
            }],
            max_dy: h1 / 5.0,
        },
        Band {
            y0: h1,
            y1: h1 + h2,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::linear(k2),
            }],
            max_dy: h2 / 4.0,
        },
    ];
    let (grid, elements) = mesh::layered::build(width, bands, width / 6.0);
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, SolverOptions::default());

    let (expect_lower, expect_upper) = circuit(k1, h1, k2, h2, s);
    for e in 0..grid.len() {
        let g = sol.gradient[e];
        let source_here = elements[e].s[1] != 0.0;
        let expect = if source_here {
            expect_lower
        } else {
            expect_upper
        };
        assert!(
            (g[1] - expect).abs() / expect.abs() < 1e-10,
            "gradient in {} band: {} vs {expect}",
            if source_here { "source" } else { "return" },
            g[1]
        );
        // Nothing drives the field across the bands, so grad_x must vanish.
        assert!(g[0].abs() < 1e-9 * expect.abs(), "cross gradient {}", g[0]);
    }

    // Flux continuity between the two branches, independently of the formula.
    assert!((expect_lower * h1 + expect_upper * h2).abs() < 1e-12);
    // And they oppose, which is what makes it a circuit rather than a leak.
    assert!(expect_lower * expect_upper < 0.0);
}

#[test]
fn refinement_leaves_an_exactly_representable_answer_alone() {
    // The circuit solution is piecewise linear, so it is in the P1 space at any
    // resolution and refining must move it by nothing at all. If it moves,
    // something scales with element size that should not.
    let (width, k1, h1, k2, h2, s) = (0.02, 2.0, 0.005, 7.0, 0.002, 11.0);
    let solve_at = |divisions: usize| {
        let bands = vec![
            Band {
                y0: 0.0,
                y1: h1,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(k1).with_flux([0.0, s]),
                }],
                max_dy: h1 / divisions as f64,
            },
            Band {
                y0: h1,
                y1: h1 + h2,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(k2),
                }],
                max_dy: h2 / divisions as f64,
            },
        ];
        let (grid, elements) = mesh::layered::build(width, bands, width / divisions as f64);
        let dofs = pinned(&grid, 1.0);
        let sol = solve(&grid, &elements, &dofs, SolverOptions::default());
        let e = (0..grid.len())
            .find(|&e| elements[e].s[1] == 0.0)
            .expect("a return element");
        sol.gradient[e][1]
    };
    let coarse = solve_at(2);
    let fine = solve_at(16);
    assert!(
        (coarse - fine).abs() / coarse.abs() < 1e-10,
        "coarse {coarse} vs fine {fine}"
    );
}

#[test]
fn tied_edges_hold_exactly_under_both_signs() {
    // The saving from solving one period instead of all of them is exact only if
    // the constraint is exact, node for node.
    let width = 0.02;
    let build = |factor: f64| {
        let bands = vec![
            Band {
                y0: 0.0,
                y1: 0.005,
                segments: vec![
                    Segment {
                        x0: 0.0,
                        x1: width / 2.0,
                        tag: Mat::linear(2.0).with_flux([3.0, 0.0]),
                    },
                    Segment {
                        x0: width / 2.0,
                        x1: width,
                        tag: Mat::linear(2.0),
                    },
                ],
                max_dy: 0.0025,
            },
            Band {
                y0: 0.005,
                y1: 0.007,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(1.0),
                }],
                max_dy: 0.001,
            },
        ];
        let (grid, elements) = mesh::layered::build(width, bands, width / 10.0);
        let dofs = DofMap::builder(grid.nodes.len())
            .fix(&grid.bottom_edge)
            .tie(&grid.left_edge, &grid.right_edge, factor)
            .build();
        let sol = solve(&grid, &elements, &dofs, SolverOptions::default());
        (grid, sol)
    };

    for factor in [1.0, -1.0] {
        let (grid, sol) = build(factor);
        let mut nontrivial = 0;
        for (&l, &r) in grid.left_edge.iter().zip(&grid.right_edge) {
            let (ul, ur) = (sol.u[l as usize], sol.u[r as usize]);
            assert!(
                (ur - factor * ul).abs() < 1e-15,
                "factor {factor}: u({l}) = {ul}, u({r}) = {ur}"
            );
            if ul.abs() > 1e-12 {
                nontrivial += 1;
            }
        }
        // Guard against the constraint passing trivially on an all-zero field.
        assert!(nontrivial > 2, "only {nontrivial} non-trivial edge nodes");
    }
}

#[test]
fn a_rising_coefficient_limits_the_gradient() {
    // The Newton loop has to do something. Driving a thin return band hard with
    // a fixed coefficient predicts a gradient the band cannot sustain once the
    // coefficient is allowed to rise with it; the nonlinear solve must land
    // materially below the linear prediction, and must converge on the way.
    let (width, k_source, h1, h2, s) = (0.02, 2.0, 0.005, 0.002, 400.0);
    let k_lo = 0.05;

    let peak_return_gradient = |return_mat: Mat| {
        let bands = vec![
            Band {
                y0: 0.0,
                y1: h1,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(k_source).with_flux([0.0, s]),
                }],
                max_dy: h1 / 4.0,
            },
            Band {
                y0: h1,
                y1: h1 + h2,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: return_mat,
                }],
                max_dy: h2 / 3.0,
            },
        ];
        let (grid, elements) = mesh::layered::build(width, bands, width / 6.0);
        let dofs = pinned(&grid, 1.0);
        let sol = solve(&grid, &elements, &dofs, SolverOptions::default());
        let peak = (0..grid.len())
            .filter(|&e| grid.centroid(e)[1] > h1)
            .map(|e| sol.gradient[e][1].abs())
            .fold(0.0, f64::max);
        (peak, sol.converged, sol.iterations)
    };

    let (linear, _, _) = peak_return_gradient(Mat::linear(k_lo));
    let (nonlinear, converged, iters) =
        peak_return_gradient(Mat::saturating(k_lo, 40.0 * k_lo, 20.0));

    // The linear case must match the circuit formula exactly.
    let (_, expect) = circuit(k_source, h1, k_lo, h2, s);
    assert!(
        (linear - expect.abs()).abs() / expect.abs() < 1e-9,
        "linear return: {linear} vs circuit {}",
        expect.abs()
    );
    assert!(converged, "Newton did not converge in {iters} passes");
    assert!(
        nonlinear < 0.9 * linear,
        "rising coefficient gave {nonlinear} against a fixed-coefficient {linear}"
    );
    assert!(nonlinear > 0.0, "the field collapsed entirely");
}

#[test]
fn a_linear_mesh_takes_a_single_pass() {
    // Nothing should iterate when no element reports itself nonlinear.
    let bands = vec![Band {
        y0: 0.0,
        y1: 0.01,
        segments: vec![Segment {
            x0: 0.0,
            x1: 0.02,
            tag: Mat::linear(1.0).with_source(1.0),
        }],
        max_dy: 0.002,
    }];
    let (grid, elements) = mesh::layered::build(0.02, bands, 0.004);
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, SolverOptions::default());
    assert_eq!(sol.iterations, 0);
    assert!(sol.converged);
}

#[test]
fn a_quadratic_element_is_exact_where_a_linear_one_only_converges() {
    // The uniform-source problem has the parabola
    //     u(y) = f * y * (H - y) / (2 * kappa)
    // as its exact solution. A parabola is outside the piecewise-linear space,
    // so P1 approximates it and converges at second order — that is the subject
    // of `a_uniform_source_converges_at_second_order` above.
    //
    // It is *inside* the piecewise-quadratic space. So a P2 solve should not
    // approximate it at all: the discrete solution is the exact one, to solver
    // tolerance, on any mesh however coarse. That is a far sharper statement
    // than a convergence rate, and it fails loudly if the quadratic shape
    // functions, their gradients, or the quadrature rule are wrong.
    let (width, height, kappa, f) = (0.02, 0.01, 3.0, 5.0e6);
    let error_at = |divisions: usize, order: Order| {
        let bands = vec![Band {
            y0: 0.0,
            y1: height,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::linear(kappa).with_source(f),
            }],
            max_dy: height / divisions as f64,
        }];
        let (mut grid, elements) = mesh::layered::build(width, bands, width / divisions as f64);
        if order == Order::P2 {
            grid = grid.into_quadratic();
        }
        let dofs = pinned(&grid, 1.0);
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let sol = solve(&grid, &elements, &dofs, opts);
        assert!(sol.linear.converged, "linear solve: {:?}", sol.linear);

        let peak = f * height * height / (8.0 * kappa);
        grid.nodes
            .iter()
            .enumerate()
            .map(|(n, &[_, y])| (sol.u[n] - f * y * (height - y) / (2.0 * kappa)).abs())
            .fold(0.0, f64::max)
            / peak
    };

    let linear = error_at(4, Order::P1);
    let quadratic = error_at(4, Order::P2);
    assert!(linear > 1e-3, "P1 was unexpectedly accurate: {linear:.3e}");
    assert!(
        quadratic < 1e-9,
        "P2 should reproduce a parabola exactly, got {quadratic:.3e}"
    );
    // And it stays exact under refinement rather than merely converging to it.
    assert!(error_at(16, Order::P2) < 1e-9);
}

#[test]
fn a_quadratic_element_still_gets_the_flux_source_circuit_right() {
    // The piecewise-linear circuit solution is in both spaces, so raising the
    // order must leave it alone. If P2 changed this answer, the extra freedom
    // would be going somewhere it should not.
    let (width, k1, h1, k2, h2, s) = (0.02, 2.0, 0.005, 7.0, 0.002, 11.0);
    let bands = || {
        vec![
            Band {
                y0: 0.0,
                y1: h1,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(k1).with_flux([0.0, s]),
                }],
                max_dy: h1 / 3.0,
            },
            Band {
                y0: h1,
                y1: h1 + h2,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: Mat::linear(k2),
                }],
                max_dy: h2 / 2.0,
            },
        ]
    };
    let (mut grid, elements) = mesh::layered::build(width, bands(), width / 5.0);
    grid = grid.into_quadratic();
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, SolverOptions::default());

    let (expect_lower, expect_upper) = circuit(k1, h1, k2, h2, s);
    for e in 0..grid.len() {
        let g = sol.gradient[e];
        let source_here = elements[e].s[1] != 0.0;
        let expect = if source_here {
            expect_lower
        } else {
            expect_upper
        };
        assert!(
            (g[1] - expect).abs() / expect.abs() < 1e-10,
            "gradient in the {} band: {} against {expect}",
            if source_here { "source" } else { "return" },
            g[1]
        );
    }
}

#[test]
fn a_quadratic_element_converges_on_a_nonlinear_problem() {
    // The Newton machinery has to work at the higher order too: the tangent's
    // rank-one term now varies within an element and is integrated rather than
    // evaluated once.
    let (width, k_source, h1, h2, s) = (0.02, 2.0, 0.005, 0.002, 400.0);
    let bands = vec![
        Band {
            y0: 0.0,
            y1: h1,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::linear(k_source).with_flux([0.0, s]),
            }],
            max_dy: h1 / 3.0,
        },
        Band {
            y0: h1,
            y1: h1 + h2,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: Mat::saturating(0.05, 2.0, 20.0),
            }],
            max_dy: h2 / 2.0,
        },
    ];
    let (grid, elements) = mesh::layered::build(width, bands, width / 5.0);
    let grid = grid.into_quadratic();
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, SolverOptions::default());
    assert!(
        sol.converged,
        "Newton did not converge at P2: {}",
        sol.residual
    );
    let peak = sol.gradient.iter().map(|g| g[1].abs()).fold(0.0, f64::max);
    assert!(peak > 0.0, "the field collapsed");
}

/// The geometry, written once, in whatever arithmetic the caller wants.
///
/// This is the point of making the coordinate type generic: at `f64` it is the
/// mesh the solver uses, and at [`Dual`] it is the same construction carrying
/// derivatives. There is no second copy to drift out of step with the first.
fn split_bands<S: rlx_fem::Scalar>(width: S, split: S) -> Vec<Band<Mat, S>> {
    vec![
        Band {
            y0: S::constant(0.0),
            y1: split,
            segments: vec![Segment {
                x0: S::constant(0.0),
                x1: width,
                tag: Mat::linear(2.0).with_flux([0.0, 11.0]),
            }],
            max_dy: 0.0015,
        },
        Band {
            y0: split,
            y1: S::constant(0.007),
            segments: vec![Segment {
                x0: S::constant(0.0),
                x1: width,
                tag: Mat::linear(7.0),
            }],
            max_dy: 0.0015,
        },
    ]
}

#[test]
fn a_differentiated_residual_matches_differencing_the_geometry() {
    // The implicit half of a shape derivative, `dR/dp`, taken two ways: by
    // differentiating the geometry, and by rebuilding it either side and
    // differencing. Agreement to well past difference-quotient precision is what
    // says the dual chain through node positions, element areas and shape
    // gradients is right.
    use rlx_fem::dual::{ElementTangent, residual_tangent};
    use rlx_fem::mesh::layered::{Plan, node_tangents};
    use rlx_fem::{Dual, adjoint::residual};

    let width = 0.02;
    let split = 0.005;
    let opts = SolverOptions {
        linear_tolerance: 1e-14,
        ..Default::default()
    };

    // Solve once at the nominal geometry.
    let plan = Plan::new(width, &split_bands(width, split), width / 6.0);
    let (grid, elements) =
        mesh::layered::build_planned(width, split_bands(width, split), &plan).expect("fits");
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, opts);

    // Analytic: seed the parameter and run the same construction in duals.
    let tangents = node_tangents(
        Dual::constant(width),
        &split_bands(Dual::constant(width), Dual::variable(split)),
        &plan,
    )
    .expect("plan fits the dual geometry");
    assert!(
        tangents.iter().any(|t| t[1].abs() > 1e-9),
        "no node moved, so the test would pass vacuously"
    );
    let zero = vec![ElementTangent::default(); elements.len()];
    let analytic = residual_tangent(&grid, &tangents, &elements, &zero, &dofs, &sol.u);

    // Differenced: rebuild either side on the same plan, residual at frozen u.
    let h = 1e-7;
    let at = |s: f64| {
        let (g, e) = mesh::layered::build_planned(width, split_bands(width, s), &plan)
            .expect("plan still fits");
        residual(&g, &e, &dofs, &sol.u)
    };
    let (plus, minus) = (at(split + h), at(split - h));
    let differenced: Vec<f64> = plus
        .iter()
        .zip(&minus)
        .map(|(a, b)| (a - b) / (2.0 * h))
        .collect();

    let peak = differenced.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "the differenced tangent was trivially zero");
    for (i, (a, d)) in analytic.iter().zip(&differenced).enumerate() {
        assert!(
            (a - d).abs() <= 1e-5 * peak,
            "dof {i}: differentiated {a} against differenced {d}"
        );
    }
}

#[test]
fn a_geometry_that_does_not_move_has_no_residual_tangent() {
    // Seeding a parameter nothing depends on must give exactly zero, not merely
    // something small: a spurious tangent would push an optimiser in a direction
    // the physics never suggested.
    use rlx_fem::Dual;
    use rlx_fem::dual::{ElementTangent, residual_tangent};
    use rlx_fem::mesh::layered::{Plan, node_tangents};

    let (width, split) = (0.02, 0.005);
    let plan = Plan::new(width, &split_bands(width, split), width / 6.0);
    let (grid, elements) =
        mesh::layered::build_planned(width, split_bands(width, split), &plan).expect("fits");
    let dofs = pinned(&grid, 1.0);
    let sol = solve(&grid, &elements, &dofs, SolverOptions::default());

    // Every coordinate a constant: nothing depends on the parameter.
    let tangents = node_tangents(
        Dual::constant(width),
        &split_bands(Dual::constant(width), Dual::constant(split)),
        &plan,
    )
    .expect("fits");
    assert!(tangents.iter().all(|t| t[0] == 0.0 && t[1] == 0.0));

    let zero = vec![ElementTangent::default(); elements.len()];
    let tangent = residual_tangent(&grid, &tangents, &elements, &zero, &dofs, &sol.u);
    assert!(
        tangent.iter().all(|v| *v == 0.0),
        "a still geometry moved the residual"
    );
}
