// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A periodic D2Q9 reference simulator on the host.
//!
//! Small and deliberately unoptimized — its job is to be *the* semantic
//! reference for [`crate::graph`], which builds the same step as an rlx graph.
//! Streaming is a gather: each cell reconstructs the population it needs from
//! the upwind neighbour's moments, so the populations exist only in registers.

use crate::lattice::D2Q9_C;
use crate::moment::{Moments2d, collide_d2q9, moments_d2q9, reconstruct_d2q9};

/// A periodic `nx × ny` D2Q9 field, stored as moments in row-major `(x, y)`.
#[derive(Debug, Clone)]
pub struct Field2d {
    /// Cells along x.
    pub nx: usize,
    /// Cells along y.
    pub ny: usize,
    /// Kinematic viscosity in lattice units.
    pub viscosity: f64,
    /// Per-cell moments, `nx * ny` entries.
    pub cells: Vec<Moments2d>,
}

impl Field2d {
    /// A field at rest with unit density.
    pub fn new(nx: usize, ny: usize, viscosity: f64) -> Self {
        Self {
            nx,
            ny,
            viscosity,
            cells: vec![Moments2d::equilibrium(1.0, [0.0, 0.0]); nx * ny],
        }
    }

    /// Initialize from a closure returning `(rho, ux, uy)` at each cell, placing
    /// every cell at equilibrium.
    pub fn init<F: FnMut(usize, usize) -> (f64, f64, f64)>(
        nx: usize,
        ny: usize,
        viscosity: f64,
        mut f: F,
    ) -> Self {
        let mut cells = Vec::with_capacity(nx * ny);
        for y in 0..ny {
            for x in 0..nx {
                let (rho, ux, uy) = f(x, y);
                cells.push(Moments2d::equilibrium(rho, [ux, uy]));
            }
        }
        Self {
            nx,
            ny,
            viscosity,
            cells,
        }
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Moments at `(x, y)`.
    pub fn at(&self, x: usize, y: usize) -> Moments2d {
        self.cells[self.idx(x, y)]
    }

    /// Advance one time step: gather-stream, then collide.
    pub fn step(&mut self) {
        let mut next = Vec::with_capacity(self.cells.len());
        for y in 0..self.ny {
            for x in 0..self.nx {
                // Gather: population i arriving at (x,y) left the cell at
                // (x,y) − c_i one step ago. Reconstruct it there rather than
                // reading a stored f — this is the whole point of the scheme.
                let mut f = [0.0f64; 9];
                for (i, c) in D2Q9_C.iter().enumerate() {
                    let sx = (x as isize - c[0] as isize).rem_euclid(self.nx as isize) as usize;
                    let sy = (y as isize - c[1] as isize).rem_euclid(self.ny as isize) as usize;
                    f[i] = reconstruct_d2q9(&self.cells[self.idx(sx, sy)])[i];
                }
                let post = moments_d2q9(&f);
                next.push(collide_d2q9(&post, self.viscosity, [0.0, 0.0]));
            }
        }
        self.cells = next;
    }

    /// Total mass — conserved exactly by streaming and collision.
    pub fn mass(&self) -> f64 {
        self.cells.iter().map(|c| c.rho).sum()
    }

    /// Peak velocity magnitude, for decay measurements.
    pub fn max_speed(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| (c.u[0] * c.u[0] + c.u[1] * c.u[1]).sqrt())
            .fold(0.0, f64::max)
    }

    /// Kinetic energy `Σ ½ρ|u|²`.
    pub fn kinetic_energy(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| 0.5 * c.rho * (c.u[0] * c.u[0] + c.u[1] * c.u[1]))
            .sum()
    }
}

/// Initialize a 2-D Taylor–Green vortex on an `n × n` periodic domain.
///
/// ```text
/// u_x = −u0 cos(2πx/n) sin(2πy/n)
/// u_y =  u0 sin(2πx/n) cos(2πy/n)
/// ```
///
/// The analytic solution decays as `exp(−2νk²t)` with `k = 2π/n`, which gives an
/// external oracle for the solver's effective viscosity — the quantity a
/// mis-scaled reconstruction corrupts while still producing a plausible picture.
pub fn taylor_green(n: usize, u0: f64, viscosity: f64) -> Field2d {
    let k = 2.0 * std::f64::consts::PI / n as f64;
    Field2d::init(n, n, viscosity, |x, y| {
        let (fx, fy) = (x as f64, y as f64);
        (
            1.0,
            -u0 * (k * fx).cos() * (k * fy).sin(),
            u0 * (k * fx).sin() * (k * fy).cos(),
        )
    })
}

/// Analytic Taylor–Green decay factor after `steps` steps.
pub fn taylor_green_decay(n: usize, viscosity: f64, steps: usize) -> f64 {
    let k = 2.0 * std::f64::consts::PI / n as f64;
    (-2.0 * viscosity * k * k * steps as f64).exp()
}
