// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An air-gap macro-element: a strip of Laplace solved exactly, coupling its two
//! boundaries without meshing between them.
//!
//! # What it replaces
//!
//! A gap between two smooth boundaries carries no source, so the potential in it
//! satisfies Laplace's equation. Meshing it therefore spends elements
//! approximating a field that has a closed form — and it spends them where the
//! answer matters most, because the quantity of interest is usually a stress
//! integral taken inside that very gap.
//!
//! Expanding the boundary data in the periodic basis of the strip separates the
//! problem: each harmonic `k` decays across the gap as a hyperbolic sine, and the
//! whole region reduces to a relation between the potential on the two
//! boundaries and its normal derivative there — a Dirichlet-to-Neumann map. That
//! relation is a dense block on the boundary degrees of freedom, and it is exact
//! up to where the series is truncated.
//!
//! # The map
//!
//! Over a gap of thickness `g`, a mode of wavenumber `k` with boundary amplitudes
//! `a` (lower) and `b` (upper) is
//!
//! ```text
//! A(z) = [ a * sinh(k(g - z))  +  b * sinh(k z) ] / sinh(k g)
//! ```
//!
//! whose normal derivatives at the two faces are
//!
//! ```text
//! dA/dz|_0 = k (b - a cosh(k g)) / sinh(k g)
//! dA/dz|_g = k (b cosh(k g) - a) / sinh(k g)
//! ```
//!
//! The two coefficients that appear, `k cosh(kg)/sinh(kg)` and `k/sinh(kg)`, both
//! tend to `1/g` as `k` goes to zero — which is the constant mode spreading
//! linearly across the gap, and is why the uniform mode needs no separate case
//! beyond guarding the arithmetic.
//!
//! # Comparing it against a meshed gap
//!
//! The map is exact for the continuous problem, so a meshed gap agrees with it
//! only as far as the mesh resolves the field it contains — and across a gap
//! that field is hyperbolic, which a single linear element cannot represent at
//! all. Against references of increasing gap resolution the two converge:
//!
//! | rows across the gap | 1 | 2 | 4 | 8 | 16 | 32 |
//! |---|---|---|---|---|---|---|
//! | 2 mm gap | 2.33% | 0.78% | 0.31% | 0.18% | 0.14% | 0.14% |
//! | 8 mm gap | 14.90% | 7.45% | 2.75% | 0.90% | 0.35% | 0.20% |
//!
//! Worth stating because the first version of this comparison used one element
//! across the gap and read the difference as the map's error. It was the
//! reference's: a thicker gap departs further from linear, so the discrepancy
//! grew with thickness and looked convincingly like a truncation failure. A
//! reference has to be resolved before it can judge anything.
//!
//! # Why the block stays symmetric
//!
//! Each mode contributes an outer product of its boundary projections, with the
//! same coefficient on both diagonal blocks and the same on both off-diagonal
//! ones. The assembled block is therefore symmetric, and the conjugate gradient
//! that solves the meshed part solves this too — no separate solver, and no loss
//! of the property the rest of the assembly relies on.

use crate::dof::DofMap;

/// How the two circumferential ends of the modelled strip relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coupling {
    /// The strip spans a whole number of periods.
    Periodic,
    /// The strip spans half a period: the field repeats negated.
    AntiPeriodic,
}

/// One boundary of the gap: its nodes, in order along the strip.
#[derive(Debug, Clone)]
pub struct Boundary {
    /// Node indices, ordered by position.
    pub nodes: Vec<u32>,
    /// Position of each node along the strip, same order and length.
    pub positions: Vec<f64>,
}

/// A gap strip replaced by its exact boundary-to-boundary relation.
#[derive(Debug, Clone)]
pub struct AirGap {
    /// The boundary at the near face.
    pub lower: Boundary,
    /// The boundary at the far face.
    pub upper: Boundary,
    /// Separation between the faces.
    pub thickness: f64,
    /// Extent of the strip along its periodic direction.
    pub width: f64,
    /// How the ends of the strip relate.
    pub coupling: Coupling,
    /// Displacement of the upper boundary relative to the lower, along the strip.
    ///
    /// This is what lets one mesh serve every rotor position. The moving side is
    /// meshed once in its own frame; where it currently sits enters only here, as
    /// a shift applied when its boundary is projected onto the basis. Nothing
    /// about the mesh depends on it, so there is no remeshing, no renumbering,
    /// and the derivative with respect to position is analytic rather than a
    /// difference between two discretisations.
    ///
    /// Wrapping needs no special case: the basis has exactly the periodicity of
    /// the strip, so a shift past the end returns the right value, negated where
    /// the coupling is anti-periodic — which is what the far side of an odd
    /// number of poles is.
    pub offset: f64,
    /// Harmonics retained.
    ///
    /// The series converges geometrically — mode `k` decays as `exp(-k g)`
    /// across the gap — so a gap of any realistic thickness needs few terms, and
    /// the ones past that contribute below rounding.
    pub harmonics: usize,
    /// Coefficient of the region, `1/mu` for magnetostatics.
    pub coefficient: f64,
}

/// One term of the periodic basis on the strip.
#[derive(Debug, Clone, Copy)]
struct Mode {
    wavenumber: f64,
    /// `false` for cosine, `true` for sine.
    odd: bool,
}

impl AirGap {
    /// The basis of the strip, in order.
    ///
    /// A periodic strip carries the constant mode and then cosine/sine pairs at
    /// multiples of `2*pi/W`. An anti-periodic one carries no constant — a field
    /// that repeats negated cannot have a uniform part — and its wavenumbers are
    /// the odd multiples of `pi/W`.
    fn modes(&self) -> Vec<Mode> {
        let mut out = Vec::with_capacity(2 * self.harmonics + 1);
        match self.coupling {
            Coupling::Periodic => {
                out.push(Mode {
                    wavenumber: 0.0,
                    odd: false,
                });
                for n in 1..=self.harmonics {
                    let k = core::f64::consts::TAU * n as f64 / self.width;
                    out.push(Mode {
                        wavenumber: k,
                        odd: false,
                    });
                    out.push(Mode {
                        wavenumber: k,
                        odd: true,
                    });
                }
            }
            Coupling::AntiPeriodic => {
                for n in 0..self.harmonics {
                    let k = (2 * n + 1) as f64 * core::f64::consts::PI / self.width;
                    out.push(Mode {
                        wavenumber: k,
                        odd: false,
                    });
                    out.push(Mode {
                        wavenumber: k,
                        odd: true,
                    });
                }
            }
        }
        out
    }

    /// Value of a mode at a position along the strip.
    fn basis(&self, mode: Mode, x: f64) -> f64 {
        if mode.wavenumber == 0.0 {
            1.0
        } else if mode.odd {
            (mode.wavenumber * x).sin()
        } else {
            (mode.wavenumber * x).cos()
        }
    }

    /// Squared norm of a mode over the strip.
    fn norm(&self, mode: Mode) -> f64 {
        if mode.wavenumber == 0.0 {
            self.width
        } else {
            self.width / 2.0
        }
    }

    /// Project a mode onto the boundary's finite element basis.
    ///
    /// Entry `i` is the integral of the mode against node `i`'s hat function.
    /// Taken by Gauss quadrature on each segment: the product of a hat and a
    /// sinusoid has no convenient closed form, and four points integrate it to
    /// well past the accuracy of anything downstream.
    fn project(&self, boundary: &Boundary, mode: Mode, shift: f64) -> Vec<f64> {
        const GAUSS: [(f64, f64); 4] = [
            (-0.861_136_311_594_053, 0.347_854_845_137_454),
            (-0.339_981_043_584_856, 0.652_145_154_862_546),
            (0.339_981_043_584_856, 0.652_145_154_862_546),
            (0.861_136_311_594_053, 0.347_854_845_137_454),
        ];
        let n = boundary.nodes.len();
        let mut out = vec![0.0; n];
        for s in 0..n.saturating_sub(1) {
            let (x0, x1) = (boundary.positions[s], boundary.positions[s + 1]);
            let h = x1 - x0;
            if h <= 0.0 {
                continue;
            }
            for (t, w) in GAUSS {
                let xi = 0.5 * (t + 1.0);
                let x = x0 + h * xi;
                let value = self.basis(mode, x + shift) * w * 0.5 * h;
                // Linear hats: the segment's two nodes share it.
                out[s] += value * (1.0 - xi);
                out[s + 1] += value * xi;
            }
        }
        out
    }

    /// The two hyperbolic coefficients of the map, guarded at `k = 0`.
    ///
    /// `(diagonal, cross)`: the first multiplies a boundary against itself, the
    /// second against the far boundary. Both approach `1/g` as `k` vanishes,
    /// where the mode simply spreads linearly across the gap.
    fn coefficients(&self, k: f64) -> (f64, f64) {
        let g = self.thickness;
        let kg = k * g;
        // Past this the sinh overflows long before the term contributes
        // anything: the mode has decayed to nothing across the gap.
        if kg > 60.0 {
            return (k, 0.0);
        }
        if kg < 1e-8 {
            return (1.0 / g, 1.0 / g);
        }
        let (c, s) = (kg.cosh(), kg.sinh());
        (k * c / s, k / s)
    }

    /// Contributions of the gap to the reduced stiffness matrix.
    ///
    /// Returned as triplets so the caller assembles them alongside the meshed
    /// regions, into the same matrix and with the same solver.
    pub fn triplets(&self, dofs: &DofMap) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        let resolve = |b: &Boundary| -> Vec<Option<(usize, f64)>> {
            b.nodes.iter().map(|&n| dofs.resolve(n)).collect()
        };
        let (lo_dofs, hi_dofs) = (resolve(&self.lower), resolve(&self.upper));

        for mode in self.modes() {
            let (diagonal, cross) = self.coefficients(mode.wavenumber);
            let norm = self.norm(mode);
            let a = self.project(&self.lower, mode, 0.0);
            let b = self.project(&self.upper, mode, self.offset);
            let scale = self.coefficient / norm;

            // Each mode is an outer product on the boundary unknowns: the same
            // coefficient on both self-blocks, its counterpart on both cross
            // blocks, so the assembled contribution is symmetric.
            let mut push = |rows: &[Option<(usize, f64)>],
                            cols: &[Option<(usize, f64)>],
                            rv: &[f64],
                            cv: &[f64],
                            coefficient: f64| {
                for (j, row) in rows.iter().enumerate() {
                    let Some((r, sr)) = *row else { continue };
                    if rv[j] == 0.0 {
                        continue;
                    }
                    for (i, col) in cols.iter().enumerate() {
                        let Some((c, sc)) = *col else { continue };
                        if cv[i] == 0.0 {
                            continue;
                        }
                        out.push((r, c, sr * sc * scale * coefficient * rv[j] * cv[i]));
                    }
                }
            };

            push(&lo_dofs, &lo_dofs, &a, &a, diagonal);
            push(&hi_dofs, &hi_dofs, &b, &b, diagonal);
            push(&lo_dofs, &hi_dofs, &a, &b, -cross);
            push(&hi_dofs, &lo_dofs, &b, &a, -cross);
        }
        out
    }

    /// Amplitudes of the field on each boundary, per mode.
    ///
    /// Returned in the order of `AirGap::modes`, as `(lower, upper)` pairs.
    /// These are what a stress integral in the gap is built from, and taking
    /// them from the expansion rather than from element values is why the result
    /// does not inherit the mesh's noise.
    pub fn amplitudes(&self, u: &[f64]) -> Vec<(f64, f64)> {
        self.modes()
            .into_iter()
            .map(|mode| {
                let norm = self.norm(mode);
                let a = self.project(&self.lower, mode, 0.0);
                let b = self.project(&self.upper, mode, self.offset);
                let dot = |c: &[f64], nodes: &[u32]| -> f64 {
                    c.iter()
                        .zip(nodes)
                        .map(|(w, &n)| w * u[n as usize])
                        .sum::<f64>()
                        / norm
                };
                (dot(&a, &self.lower.nodes), dot(&b, &self.upper.nodes))
            })
            .collect()
    }

    /// Flux density normal to the strip, sampled at `samples` evenly spaced
    /// positions at `height` above the lower face.
    ///
    /// A replaced gap has no elements, so there is nothing in the mesh to read a
    /// field from — the strongest field in the machine lives exactly where the
    /// discretisation was removed. It is recovered instead from the harmonic
    /// representation the coupling is already built on, which is not an
    /// approximation of the gap field but the thing the coupling is exact for.
    ///
    /// `B_n` is `-dA/du` for a potential directed out of plane.
    pub fn normal_flux(&self, u: &[f64], height: f64, samples: usize) -> Vec<f64> {
        let modes = self.modes();
        let amps = self.amplitudes(u);
        let g = self.thickness;
        let z = height.clamp(0.0, g);
        (0..samples)
            .map(|i| {
                let x = self.width * i as f64 / samples.max(1) as f64;
                let mut b = 0.0;
                for (m, &(a, c)) in modes.iter().zip(&amps) {
                    let k = m.wavenumber;
                    // A constant potential has no in-plane flux to contribute.
                    if k == 0.0 {
                        continue;
                    }
                    let kg = (k * g).min(60.0);
                    let sh = kg.sinh();
                    if sh.abs() < 1e-300 {
                        continue;
                    }
                    let f = (a * (k * (g - z)).sinh() + c * (k * z).sinh()) / sh;
                    // The derivative of the mode along the strip.
                    let d = if m.odd {
                        k * (k * x).cos()
                    } else {
                        -k * (k * x).sin()
                    };
                    b -= f * d;
                }
                b
            })
            .collect()
    }

    /// Tangential force per unit depth, from the harmonic amplitudes.
    ///
    /// Maxwell stress evaluated in the transform rather than on the mesh. Only
    /// matching cosine/sine pairs of one wavenumber contribute — everything else
    /// integrates away over the strip — so the sum is short and carries no
    /// contribution from modes the mesh cannot resolve.
    pub fn tangential_force(&self, u: &[f64], height: f64) -> f64 {
        let modes = self.modes();
        let amps = self.amplitudes(u);
        let g = self.thickness;
        let mut total = 0.0;

        // Cosine and sine of one wavenumber sit adjacent; a pair is what a
        // travelling component of the field is made of.
        let mut i = 0;
        while i + 1 < modes.len() {
            let (mc, ms) = (modes[i], modes[i + 1]);
            if mc.wavenumber == 0.0 || mc.wavenumber != ms.wavenumber || mc.odd || !ms.odd {
                i += 1;
                continue;
            }
            let k = mc.wavenumber;
            let kg = (k * g).min(60.0);
            let (sh, ch) = (kg.sinh(), kg.cosh());
            if sh.abs() < 1e-300 {
                i += 2;
                continue;
            }
            // Field at `height` above the lower face, per component.
            let at = |(a, b): (f64, f64)| {
                let z = height.clamp(0.0, g);
                let f = (a * (k * (g - z)).sinh() + b * (k * z).sinh()) / sh;
                let df = k * (-a * (k * (g - z)).cosh() + b * (k * z).cosh()) / sh;
                (f, df)
            };
            let (fc, dfc) = at(amps[i]);
            let (fs, dfs) = at(amps[i + 1]);
            let _ = ch;
            // B_u = dA/dz and B_z = -dA/du, so the stress B_u B_z integrated
            // over a full period keeps the cross terms of one wavenumber and
            // averages the rest away, leaving `(W/2) k (f_s' f_c - f_c' f_s)`.
            // The order of that difference is the sign of the force and is
            // pinned against a closed form in the tests, having once been the
            // other way round.
            total += 0.5 * self.width * k * (dfs * fc - dfc * fs);
            i += 2;
        }
        total * self.coefficient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::{Constitutive, SolverOptions, assemble, solve_coupled};
    use crate::mesh::{self, Band, Mesh, Segment};
    use crate::solve::{Csr, pcg};

    const MU0: f64 = 1.256_637_062_12e-6;

    #[derive(Clone, PartialEq)]
    struct Mat {
        mu_r: f64,
        br: [f64; 2],
        /// Excluded from assembly: its region is carried by the macro-element.
        vacated: bool,
    }

    impl Constitutive for Mat {
        fn coefficient(&self, _: f64) -> f64 {
            if self.vacated {
                0.0
            } else {
                1.0 / (MU0 * self.mu_r)
            }
        }
        fn is_nonlinear(&self) -> bool {
            false
        }
        fn flux_source(&self, _: f64) -> [f64; 2] {
            if self.vacated {
                return [0.0, 0.0];
            }
            let nu = 1.0 / (MU0 * self.mu_r);
            [-nu * self.br[1], nu * self.br[0]]
        }
    }

    fn air() -> Mat {
        Mat {
            mu_r: 1.0,
            br: [0.0, 0.0],
            vacated: false,
        }
    }
    fn iron() -> Mat {
        Mat {
            mu_r: 2000.0,
            br: [0.0, 0.0],
            vacated: false,
        }
    }
    fn magnet(br: f64) -> Mat {
        Mat {
            mu_r: 1.05,
            br: [br, 0.0],
            vacated: false,
        }
    }

    const W: f64 = 0.02;
    const HM: f64 = 0.005;
    const G: f64 = 0.002;
    const HI: f64 = 0.004;

    /// Magnet, gap, iron. The gap is a single element row, so its two faces are
    /// the only nodes in it and vacating it leaves nothing orphaned.
    fn build(vacate_gap: bool) -> (Mesh, Vec<Mat>) {
        let mut gap = air();
        gap.vacated = vacate_gap;
        let bands = vec![
            Band {
                y0: 0.0,
                y1: HM,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: W,
                    tag: magnet(1.2),
                }],
                max_dy: HM / 4.0,
            },
            Band {
                y0: HM,
                y1: HM + G,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: W,
                    tag: gap,
                }],
                max_dy: G,
            },
            Band {
                y0: HM + G,
                y1: HM + G + HI,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: W,
                    tag: iron(),
                }],
                max_dy: HI / 3.0,
            },
        ];
        mesh::layered::build(W, bands, W / 24.0)
    }

    /// Nodes on a horizontal line, ordered along it.
    fn row(grid: &Mesh, y: f64) -> Boundary {
        let mut found: Vec<(f64, u32)> = grid
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| (n[1] - y).abs() < 1e-12)
            .map(|(i, n)| (n[0], i as u32))
            .collect();
        found.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
        Boundary {
            positions: found.iter().map(|f| f.0).collect(),
            nodes: found.iter().map(|f| f.1).collect(),
        }
    }

    fn dofs_for(grid: &Mesh) -> DofMap {
        DofMap::builder(grid.nodes.len())
            .fix(&grid.bottom_edge)
            .fix(&grid.top_edge)
            .tie(&grid.left_edge, &grid.right_edge, 1.0)
            .build()
    }

    fn solve_linear(
        grid: &Mesh,
        elements: &[Mat],
        dofs: &DofMap,
        extra: &[(usize, usize, f64)],
    ) -> Vec<f64> {
        let coeff: Vec<f64> = elements.iter().map(|e| e.coefficient(0.0)).collect();
        let (k, f) = assemble(grid, elements, dofs, &coeff);
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for r in 0..k.n {
            for idx in k.indptr[r]..k.indptr[r + 1] {
                triplets.push((r, k.indices[idx], k.values[idx]));
            }
        }
        triplets.extend_from_slice(extra);
        let combined = Csr::from_triplets(k.n, triplets);
        let (reduced, report) = pcg(&combined, &f, 1e-14, 40 * k.n.max(1));
        assert!(
            report.converged,
            "coupled solve did not converge: {report:?}"
        );
        dofs.expand(&reduced)
    }

    fn gap_for(grid: &Mesh, harmonics: usize) -> AirGap {
        AirGap {
            lower: row(grid, HM),
            upper: row(grid, HM + G),
            thickness: G,
            width: W,
            coupling: Coupling::Periodic,
            harmonics,
            coefficient: 1.0 / MU0,
            offset: 0.0,
        }
    }

    /// Magnet, gap, iron — but with the magnet alternating in sign across the
    /// strip, as a real rotor does, so the boundary field carries short
    /// wavelengths rather than one smooth hump.
    fn build_alternating(vacate_gap: bool) -> (Mesh, Vec<Mat>) {
        let mut gap = air();
        gap.vacated = vacate_gap;
        let poles = 4;
        let pitch = W / poles as f64;
        let arc = 0.8 * pitch;
        let mut segments = Vec::new();
        let mut x = 0.0;
        for j in 0..poles {
            let centre = (j as f64 + 0.5) * pitch;
            let (a, b) = (centre - arc / 2.0, centre + arc / 2.0);
            if a > x {
                segments.push(Segment {
                    x0: x,
                    x1: a,
                    tag: air(),
                });
            }
            let sign = if j % 2 == 0 { 1.2 } else { -1.2 };
            segments.push(Segment {
                x0: a,
                x1: b,
                tag: magnet(sign),
            });
            x = b;
        }
        if x < W {
            segments.push(Segment {
                x0: x,
                x1: W,
                tag: air(),
            });
        }
        let bands = vec![
            Band {
                y0: 0.0,
                y1: HM,
                segments,
                max_dy: HM / 4.0,
            },
            Band {
                y0: HM,
                y1: HM + G,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: W,
                    tag: gap,
                }],
                // Vacated, the gap must be one row so nothing floats between its
                // faces. Meshed, it needs enough rows to resolve a hyperbolic
                // profile — otherwise the reference is the less accurate of the
                // two and the comparison measures its error, not the map's.
                max_dy: if vacate_gap { G } else { G / 16.0 },
            },
            Band {
                y0: HM + G,
                y1: HM + G + HI,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: W,
                    tag: iron(),
                }],
                max_dy: HI / 3.0,
            },
        ];
        mesh::layered::build(W, bands, W / 48.0)
    }

    #[test]
    fn the_map_reproduces_a_gap_under_an_alternating_source() {
        // The case the first version of this test failed to cover.
        //
        // A single smooth magnet puts almost all of its energy in the lowest
        // modes, which any truncation retains — so the map looked exact when it
        // was merely being asked an easy question. A rotor alternating N and S
        // carries short wavelengths, and the modes past the truncation each
        // contribute a self-coupling that grows with wavenumber. Dropping them
        // under-stiffens the gap.
        let (grid, meshed) = build_alternating(false);
        let dofs = dofs_for(&grid);
        let reference = solve_linear(&grid, &meshed, &dofs, &[]);

        let (grid2, vacated) = build_alternating(true);
        // Its own map: the two meshes no longer have the same nodes, because the
        // reference resolves the gap and this one has vacated it. Reusing the
        // reference's map here silently addresses the wrong unknowns.
        let dofs2 = dofs_for(&grid2);
        let gap = AirGap {
            lower: row(&grid2, HM),
            upper: row(&grid2, HM + G),
            thickness: G,
            width: W,
            coupling: Coupling::Periodic,
            harmonics: (row(&grid2, HM).nodes.len() / 2).saturating_sub(1),
            coefficient: 1.0 / MU0,
            offset: 0.0,
        };
        let coupled = solve_linear(&grid2, &vacated, &dofs2, &gap.triplets(&dofs2));

        let peak = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        let mut compared = 0;
        for (i, p) in grid.nodes.iter().enumerate() {
            if p[1] > HM - 1e-12 && p[1] < HM + G + 1e-12 {
                continue;
            }
            let Some(j) = grid2
                .nodes
                .iter()
                .position(|q| (q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9)
            else {
                continue;
            };
            compared += 1;
            worst = worst.max((reference[i] - coupled[j]).abs() / peak);
        }
        assert!(compared > 100, "only {compared} nodes matched");
        assert!(
            worst < 5e-3,
            "an alternating source disagreed by {:.2}% of peak",
            100.0 * worst
        );
    }

    #[test]
    fn the_map_reproduces_the_meshed_gap() {
        // The claim the macro-element rests on: a gap solved from its harmonic
        // expansion and a gap filled with elements are the same problem, so the
        // field in the regions either side must come out the same.
        let (grid, meshed) = build(false);
        let dofs = dofs_for(&grid);
        let reference = solve_linear(&grid, &meshed, &dofs, &[]);

        let (grid2, vacated) = build(true);
        let gap = gap_for(&grid2, 24);
        let coupled = solve_linear(&grid2, &vacated, &dofs, &gap.triplets(&dofs));

        let peak = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak > 1e-6, "the reference field was trivially small");

        // Compared away from the gap, where both formulations describe the same
        // meshed material.
        let mut compared = 0;
        for (n, node) in grid.nodes.iter().enumerate() {
            if node[1] > HM - 1e-12 && node[1] < HM + G + 1e-12 {
                continue;
            }
            compared += 1;
            assert!(
                (reference[n] - coupled[n]).abs() < 2e-3 * peak,
                "node {n} at z={}: meshed {} against coupled {}",
                node[1],
                reference[n],
                coupled[n]
            );
        }
        assert!(compared > 50, "only {compared} nodes were compared");
    }

    #[test]
    fn truncating_the_series_converges() {
        // The expansion is truncated, so the question is where it stops mattering.
        // Each mode decays as exp(-k g) across the gap, so convergence is
        // geometric and a handful of terms is enough.
        let (grid, meshed) = build(false);
        let dofs = dofs_for(&grid);
        let reference = solve_linear(&grid, &meshed, &dofs, &[]);
        let peak = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));

        let (grid2, vacated) = build(true);
        let error_at = |harmonics: usize| {
            let gap = gap_for(&grid2, harmonics);
            let coupled = solve_linear(&grid2, &vacated, &dofs, &gap.triplets(&dofs));
            grid.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n[1] < HM - 1e-12 || n[1] > HM + G + 1e-12)
                .map(|(n, _)| (reference[n] - coupled[n]).abs())
                .fold(0.0f64, f64::max)
                / peak
        };
        let coarse = error_at(2);
        let fine = error_at(16);
        assert!(
            fine < coarse,
            "more harmonics did not help: {coarse} then {fine}"
        );
        assert!(fine < 2e-3, "still {fine:.2e} at sixteen harmonics");
    }

    #[test]
    fn a_uniform_mode_spreads_linearly_across_the_gap() {
        // The zero-wavenumber limit, which the hyperbolic coefficients reach
        // only as a limit and would otherwise divide by a vanishing sinh.
        let (grid, _) = build(true);
        let gap = gap_for(&grid, 4);
        let (diagonal, cross) = gap.coefficients(0.0);
        assert!((diagonal - 1.0 / G).abs() / (1.0 / G) < 1e-9);
        assert!((cross - 1.0 / G).abs() / (1.0 / G) < 1e-9);

        // And approached continuously from above, not jumped to.
        let (d_small, c_small) = gap.coefficients(1e-6);
        assert!((d_small - 1.0 / G).abs() / (1.0 / G) < 1e-6);
        assert!((c_small - 1.0 / G).abs() / (1.0 / G) < 1e-6);
    }

    #[test]
    fn a_mode_that_cannot_cross_the_gap_stops_coupling_it() {
        // A short wavelength decays to nothing over the gap, so its two faces
        // become independent: the cross coefficient goes to zero while the
        // diagonal tends to k. Guarded, because sinh overflows long before the
        // term stops contributing.
        let (grid, _) = build(true);
        let gap = gap_for(&grid, 4);
        let k = 1.0e5;
        let (diagonal, cross) = gap.coefficients(k);
        assert_eq!(cross, 0.0, "a decayed mode should not couple the faces");
        assert!((diagonal - k).abs() / k < 1e-9);
        assert!(diagonal.is_finite());
    }

    #[test]
    fn a_shift_of_one_period_leaves_the_coupling_unchanged() {
        // The convention, pinned. Displacing the moving side by a whole period
        // puts it back where it started, so the block must be identical — and by
        // half a period under an anti-periodic coupling it must be the negation
        // of nothing, since the *field* flips but the coupling of amplitudes does
        // not. Getting this wrong is how a rotor ends up silently a pole out.
        let (grid, _) = build(true);
        let dofs = dofs_for(&grid);
        let block = |offset: f64| {
            let mut gap = gap_for(&grid, 8);
            gap.offset = offset;
            let mut dense = std::collections::HashMap::<(usize, usize), f64>::new();
            for (r, c, v) in gap.triplets(&dofs) {
                *dense.entry((r, c)).or_insert(0.0) += v;
            }
            dense
        };
        let home = block(0.0);
        let shifted = block(W);
        let scale = home.values().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0);
        for (key, v) in &home {
            let after = shifted.get(key).copied().unwrap_or(0.0);
            assert!(
                (v - after).abs() < 1e-8 * scale,
                "a full period changed {key:?}: {v} against {after}"
            );
        }

        // And a partial shift genuinely changes it, so the test above is not
        // passing because the offset does nothing at all.
        let quarter = block(W / 4.0);
        let moved = home
            .iter()
            .map(|(k, v)| (v - quarter.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0f64, f64::max);
        assert!(moved > 1e-3 * scale, "a quarter period changed nothing");
    }

    #[test]
    fn the_coupling_block_is_symmetric() {
        // The assembled matrix has to stay symmetric positive definite, or the
        // conjugate gradient that solves the meshed part cannot solve this.
        let (grid, _) = build(true);
        let dofs = dofs_for(&grid);
        let triplets = gap_for(&grid, 8).triplets(&dofs);
        assert!(!triplets.is_empty());

        let mut dense = std::collections::HashMap::<(usize, usize), f64>::new();
        for (r, c, v) in &triplets {
            *dense.entry((*r, *c)).or_insert(0.0) += v;
        }
        let scale = dense.values().fold(0.0f64, |m, v| m.max(v.abs()));
        for ((r, c), v) in &dense {
            let mirror = dense.get(&(*c, *r)).copied().unwrap_or(0.0);
            assert!(
                (v - mirror).abs() < 1e-9 * scale,
                "({r},{c}) is {v} but ({c},{r}) is {mirror}"
            );
        }
    }
    /// A gap whose two faces are given directly, bypassing any solve, so a
    /// property of the map can be tested on a state chosen to exercise it rather
    /// than on whatever a particular geometry happens to produce.
    fn synthetic_gap(
        lower: impl Fn(f64) -> f64,
        upper: impl Fn(f64) -> f64,
        n: usize,
    ) -> (AirGap, Vec<f64>) {
        let positions: Vec<f64> = (0..=n).map(|i| W * i as f64 / n as f64).collect();
        let mut state: Vec<f64> = positions.iter().map(|&x| lower(x)).collect();
        state.extend(positions.iter().map(|&x| upper(x)));
        let gap = AirGap {
            lower: Boundary {
                positions: positions.clone(),
                nodes: (0..=n as u32).collect(),
            },
            upper: Boundary {
                positions,
                nodes: (n as u32 + 1..=2 * n as u32 + 1).collect(),
            },
            thickness: G,
            width: W,
            coupling: Coupling::Periodic,
            harmonics: 8,
            coefficient: 1.0 / MU0,
            offset: 0.0,
        };
        (gap, state)
    }

    #[test]
    fn the_stress_does_not_depend_on_the_height_it_is_read_at() {
        // In a gap carrying no current the Maxwell stress is a conserved
        // momentum flux, so the tangential force through any surface spanning
        // the gap is the same one.
        //
        // For the sinh form this is an identity rather than a discovery — the
        // height cancels through `sinh(z + (g - z))`, leaving
        // `k (a_s b_c - a_c b_s) / sinh(k g)` — and that is the useful part: the
        // height the stress is read at is a free choice and not a tuning knob,
        // which is what lets a torque be quoted without saying where in the gap
        // it was taken. Asserted so that an edit to the field expression that
        // breaks the conservation law cannot pass quietly.
        //
        // The two faces are given a quarter period apart, because faces in phase
        // pull straight across the gap and not along it — see the test below.
        let k = 2.0 * std::f64::consts::PI / W;
        let (gap, state) = synthetic_gap(|x| (k * x).cos(), |x| (k * x).sin(), 240);

        let mid = gap.tangential_force(&state, G / 2.0);
        assert!(mid.abs() > 1e-6, "the stress was trivially small: {mid:e}");
        for f in [0.0, 0.1, 0.25, 0.75, 0.9, 1.0] {
            let here = gap.tangential_force(&state, f * G);
            assert!(
                (here - mid).abs() <= 1e-10 * mid.abs(),
                "the stress read at {f} of the gap was {here:e}, against {mid:e} at the middle"
            );
        }
    }

    #[test]
    fn faces_in_phase_pull_across_the_gap_and_not_along_it() {
        // Two faces with no relative displacement have nothing to drag each
        // other with, and the tangential force is zero by symmetry. A map that
        // manufactured force out of an aligned configuration would produce
        // torque from a machine at rest — so this is the cheapest guard there is
        // against the failure that matters most.
        let k = 2.0 * std::f64::consts::PI / W;
        let (gap, state) = synthetic_gap(|x| (k * x).cos(), |x| 0.5 * (k * x).cos(), 240);
        let scale = gap.coefficient * W * k;
        let force = gap.tangential_force(&state, G / 2.0);
        assert!(
            force.abs() < 1e-9 * scale,
            "aligned faces produced a tangential force of {force:e}, against a scale of {scale:e}"
        );

        // And the same geometry does produce force once one face is displaced,
        // so the zero above is symmetry and not a stress that is always zero.
        let (shifted, state2) =
            synthetic_gap(|x| (k * x).cos(), |x| 0.5 * (k * x - 0.7).cos(), 240);
        let pulled = shifted.tangential_force(&state2, G / 2.0);
        assert!(
            pulled.abs() > 1e-3 * scale,
            "displacing a face produced no force either: {pulled:e}"
        );
    }

    #[test]
    fn the_reconstructed_flux_matches_the_harmonic_it_was_built_from() {
        // A boundary carrying exactly one mode has a closed-form field across the
        // gap, so the reconstruction can be checked against arithmetic rather
        // than against another discretisation with its own errors.
        const N: usize = 240;
        let k = 2.0 * std::f64::consts::PI / W;
        // One cosine on the near face, nothing on the far one.
        let (gap, state) = synthetic_gap(|x| (k * x).cos(), |_| 0.0, N);

        let z = G / 3.0;
        const SAMPLES: usize = 64;
        let got = gap.normal_flux(&state, z, SAMPLES);
        let decay = (k * (G - z)).sinh() / (k * G).sinh();
        for (i, &b) in got.iter().enumerate() {
            let x = W * i as f64 / SAMPLES as f64;
            let want = k * (k * x).sin() * decay;
            assert!(
                (b - want).abs() < 1e-3 * k,
                "at x = {x:.5} the flux was {b:.6e}, against {want:.6e}"
            );
        }
    }

    #[test]
    fn the_stress_matches_the_closed_form_for_one_mode_pair() {
        // Sign and magnitude together, against arithmetic rather than against
        // another solve. Faces carrying `cos(ku)` and `sin(ku)` give
        // `f_c = sinh(k(g-z))/sinh(kg)` and `f_s = sinh(kz)/sinh(kg)`, and the
        // stress integral collapses to `W k^2 / (2 sinh(kg))` — positive, so a
        // face displaced a quarter period ahead drags the other forward.
        let k = 2.0 * std::f64::consts::PI / W;
        let (gap, state) = synthetic_gap(|x| (k * x).cos(), |x| (k * x).sin(), 240);
        let want = W * k * k / (2.0 * (k * G).sinh()) * gap.coefficient;
        let got = gap.tangential_force(&state, G / 2.0);
        assert!(
            (got - want).abs() < 2e-3 * want.abs(),
            "the stress was {got:e}, against a closed form of {want:e}"
        );
    }

    #[test]
    fn the_public_solve_applies_the_coupling_to_a_linear_mesh() {
        // Every other test here assembles the coupled matrix by hand, which is
        // how a bug in the entry point everyone else calls went unseen.
        //
        // `solve_coupled` opens with one linear solve, and used to assemble it
        // without the coupling: a starting point for a nonlinear mesh, which the
        // Newton passes then correct, but the whole answer for a linear one,
        // which has no passes. A linear mesh with a vacated gap therefore came
        // back solved as two halves that could not see each other — reporting
        // convergence, with a field six orders of magnitude too large.
        let (grid, vacated) = build_alternating(true);
        let dofs = dofs_for(&grid);
        let harmonics = (row(&grid, HM).nodes.len() / 2).saturating_sub(1);
        let gap = gap_for(&grid, harmonics);
        let coupling = gap.triplets(&dofs);

        let by_hand = solve_linear(&grid, &vacated, &dofs, &coupling);
        let opts = SolverOptions {
            linear_tolerance: 1e-14,
            ..Default::default()
        };
        let through_api = solve_coupled(&grid, &vacated, &dofs, &coupling, opts);
        assert!(through_api.converged);

        let peak = by_hand.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(peak > 1e-9, "the reference field was trivially small");
        let worst = by_hand
            .iter()
            .zip(&through_api.u)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 1e-8 * peak,
            "the two routes disagreed by {:.3e}, which is {:.2}% of peak",
            worst,
            100.0 * worst / peak
        );

        // And the coupling is what makes the difference, so the agreement above
        // is not two routes sharing one mistake.
        let uncoupled = solve_coupled(&grid, &vacated, &dofs, &[], opts);
        let apart = uncoupled
            .u
            .iter()
            .zip(&through_api.u)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            apart > 0.5 * peak,
            "dropping the coupling moved the field by only {:.1}% of peak, so the \
             agreement above may be two routes sharing one mistake",
            100.0 * apart / peak
        );
    }
}
