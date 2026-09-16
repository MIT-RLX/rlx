// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nonlinear scalar finite elements on P1 triangles.
//!
//! Solves, for one scalar unknown per node,
//!
//! ```text
//! -div( kappa(|grad u|^2) * grad u  -  s )  =  f
//! ```
//!
//! whose weak form on linear triangles is
//!
//! ```text
//! integral kappa * grad(u) . grad(w)  =  integral f * w  +  integral s . grad(w)
//! ```
//!
//! Every one of those integrals is exact and closed-form on a P1 triangle,
//! because `grad(u)`, `kappa`, `f` and `s` are all element-wise constant. That
//! is the entire element library.
//!
//! The caller supplies the three physical terms per element through
//! [`Constitutive`]. Nothing in this crate names a physical quantity, so the
//! same solver serves magnetostatics (`u` a vector potential, `kappa` a
//! reluctivity that rises with saturation, `s` a magnet remanence), steady heat
//! conduction, electrostatics and Darcy flow.
//!
//! # Layout
//!
//! | Module | Contents |
//! |---|---|
//! | [`mesh`] | Triangle geometry and a conforming structured builder over layered domains |
//! | [`dof`] | Fixed and tied degrees of freedom, eliminated rather than penalised |
//! | [`mod@adjoint`] | Objective gradients for every parameter at the cost of one solve |
//! | [`dual`] | Forward-mode duals, and functionals of the solution differentiated with them |
//! | [`airgap`] | A gap strip solved exactly and coupled as a boundary relation, not meshed |
//! | [`solve`] | CSR storage and a Jacobi-preconditioned conjugate gradient |
//! | [`assemble`] | The [`Constitutive`] trait, element assembly, and a damped Newton driver |
//!
//! # Example
//!
//! ```
//! use rlx_fem::{assemble::{solve, Constitutive, SolverOptions}, dof::DofMap, mesh};
//!
//! // Uniform unit coefficient, unit source: a Poisson problem.
//! #[derive(Clone)]
//! struct Uniform;
//! impl Constitutive for Uniform {
//!     fn coefficient(&self, _grad_sq: f64) -> f64 { 1.0 }
//!     fn is_nonlinear(&self) -> bool { false }
//!     fn source(&self) -> f64 { 1.0 }
//! }
//!
//! let band = mesh::Band {
//!     y0: 0.0,
//!     y1: 1.0,
//!     segments: vec![mesh::Segment { x0: 0.0, x1: 1.0, tag: Uniform }],
//!     max_dy: 0.25,
//! };
//! let (grid, elements) = mesh::layered::build(1.0, vec![band], 0.25);
//!
//! let dofs = DofMap::builder(grid.nodes.len())
//!     .fix(&grid.bottom_edge)
//!     .fix(&grid.top_edge)
//!     .tie(&grid.left_edge, &grid.right_edge, 1.0)
//!     .build();
//!
//! let result = solve(&grid, &elements, &dofs, SolverOptions::default());
//! assert!(result.linear.converged);
//! assert!(result.u.iter().cloned().fold(0.0, f64::max) > 0.0);
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub mod adjoint;
pub mod airgap;
pub mod assemble;
pub mod dof;
pub mod dual;
pub mod mesh;
pub mod solve;

pub use adjoint::{Adjoint, adjoint, contract, residual};
pub use airgap::{AirGap, Boundary, Coupling};
pub use assemble::{Constitutive, Solution, SolverOptions, quadrature, solve_coupled};
pub use dof::{Dof, DofMap};
pub use dual::{Dual, ElementTangent, Scalar, gradient_functional, residual_tangent};
pub use mesh::{Mesh, Order};
pub use solve::{Csr, SolveReport};
