// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where does the modal basis stop helping?
//!
//! The map is `K = C^T diag(D/norm) C`, of rank at most the number of modes. Too
//! few and the gap is left too permeable in the directions they do not span; too
//! many and the modes are dependent on the boundary's own space, so their
//! stiffness is counted more than once. The optimum should sit at the boundary's
//! dimension — this measures whether it does, and how sharp it is.
use rlx_fem::airgap::{AirGap, Boundary, Coupling};
use rlx_fem::assemble::{Constitutive, assemble};
use rlx_fem::dof::DofMap;
use rlx_fem::mesh::{self, Band, Mesh, Segment};
use rlx_fem::solve::{Csr, pcg};

const MU0: f64 = 1.256_637_062_12e-6;
const W: f64 = 0.02;
const HM: f64 = 0.005;
const HI: f64 = 0.004;

#[derive(Clone, PartialEq)]
struct Mat {
    mu_r: f64,
    br: f64,
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
        [-self.br / (MU0 * self.mu_r), 0.0]
    }
}
fn air() -> Mat {
    Mat {
        mu_r: 1.0,
        br: 0.0,
        vacated: false,
    }
}

fn build(g: f64, vacate: bool, per: usize, gap_rows: usize) -> (Mesh, Vec<Mat>) {
    let mut gap = air();
    gap.vacated = vacate;
    let poles = 4;
    let pitch = W / poles as f64;
    let arc = 0.8 * pitch;
    let mut segs = Vec::new();
    let mut x = 0.0;
    for j in 0..poles {
        let c = (j as f64 + 0.5) * pitch;
        let (a, b) = (c - arc / 2.0, c + arc / 2.0);
        if a > x {
            segs.push(Segment {
                x0: x,
                x1: a,
                tag: air(),
            });
        }
        segs.push(Segment {
            x0: a,
            x1: b,
            tag: Mat {
                mu_r: 1.05,
                br: if j % 2 == 0 { 1.2 } else { -1.2 },
                vacated: false,
            },
        });
        x = b;
    }
    if x < W {
        segs.push(Segment {
            x0: x,
            x1: W,
            tag: air(),
        });
    }
    let bands = vec![
        Band {
            y0: 0.0,
            y1: HM,
            segments: segs,
            max_dy: HM / 4.0,
        },
        Band {
            y0: HM,
            y1: HM + g,
            segments: vec![Segment {
                x0: 0.0,
                x1: W,
                tag: gap,
            }],
            max_dy: g / gap_rows as f64,
        },
        Band {
            y0: HM + g,
            y1: HM + g + HI,
            segments: vec![Segment {
                x0: 0.0,
                x1: W,
                tag: Mat {
                    mu_r: 2000.0,
                    br: 0.0,
                    vacated: false,
                },
            }],
            max_dy: HI / 3.0,
        },
    ];
    mesh::layered::build(W, bands, W / per as f64)
}

fn row(grid: &Mesh, y: f64) -> Boundary {
    let mut f: Vec<(f64, u32)> = grid
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| (n[1] - y).abs() < 1e-12)
        .map(|(i, n)| (n[0], i as u32))
        .collect();
    f.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    Boundary {
        positions: f.iter().map(|v| v.0).collect(),
        nodes: f.iter().map(|v| v.1).collect(),
    }
}

fn solve_it(grid: &Mesh, els: &[Mat], dofs: &DofMap, extra: &[(usize, usize, f64)]) -> Vec<f64> {
    let coeff: Vec<f64> = els.iter().map(|e| e.coefficient(0.0)).collect();
    let (k, f) = assemble(grid, els, dofs, &coeff);
    let mut t = Vec::new();
    for r in 0..k.n {
        for i in k.indptr[r]..k.indptr[r + 1] {
            t.push((r, k.indices[i], k.values[i]));
        }
    }
    t.extend_from_slice(extra);
    let (x, _) = pcg(&Csr::from_triplets(k.n, t), &f, 1e-14, 60 * k.n.max(1));
    dofs.expand(&x)
}

fn main() {
    // The reference is what is being questioned here. A single linear element
    // across the gap cannot represent a hyperbolic profile, and the profile
    // departs further from linear as the gap thickens — which is exactly how the
    // discrepancy behaved.
    for (g, per) in [(0.002f64, 48usize), (0.008, 48)] {
        println!("\ngap {:.0} mm, per {}", g * 1e3, per);
        println!("  reference gap rows   worst disagreement vs the map");
        for rows in [1usize, 2, 4, 8, 16, 32] {
            let (rg, re) = build(g, false, per, rows);
            let dofs = DofMap::builder(rg.nodes.len())
                .fix(&rg.bottom_edge)
                .fix(&rg.top_edge)
                .tie(&rg.left_edge, &rg.right_edge, 1.0)
                .build();
            let reference = solve_it(&rg, &re, &dofs, &[]);
            let peak = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));

            let (mg, me) = build(g, true, per, 1);
            let dofs_m = DofMap::builder(mg.nodes.len())
                .fix(&mg.bottom_edge)
                .fix(&mg.top_edge)
                .tie(&mg.left_edge, &mg.right_edge, 1.0)
                .build();
            let gap = AirGap {
                lower: row(&mg, HM),
                upper: row(&mg, HM + g),
                thickness: g,
                width: W,
                coupling: Coupling::Periodic,
                harmonics: 16,
                coefficient: 1.0 / MU0,
                offset: 0.0,
            };
            let u = solve_it(&mg, &me, &dofs_m, &gap.triplets(&dofs_m));

            let mut worst = 0.0f64;
            for (i, pnt) in rg.nodes.iter().enumerate() {
                if pnt[1] > HM - 1e-12 && pnt[1] < HM + g + 1e-12 {
                    continue;
                }
                if let Some(j) = mg
                    .nodes
                    .iter()
                    .position(|q| (q[0] - pnt[0]).abs() < 1e-9 && (q[1] - pnt[1]).abs() < 1e-9)
                {
                    worst = worst.max((reference[i] - u[j]).abs() / peak);
                }
            }
            println!("  {rows:<20} {:>8.3}%", 100.0 * worst);
        }
    }
}
