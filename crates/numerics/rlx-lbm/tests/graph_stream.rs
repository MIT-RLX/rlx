// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The graph streaming step must match the host reference step.
//!
//! Both implement the same equations, but by different routes: the host gathers
//! per cell with modular indexing, the graph shifts whole fields with
//! `Op::Roll` and works elementwise. Agreement is therefore evidence about the
//! lowering (and about `Op::Roll` itself), not two copies of one formula.

#![cfg(feature = "ir")]

use rlx_lbm::graph::{FIELD_NAMES, stream_step_graph};
use rlx_lbm::lattice::D2Q9_C;
use rlx_lbm::moment::{moments_d2q9, reconstruct_d2q9};
use rlx_lbm::sim::taylor_green;
use rlx_runtime::{Device, Session};

/// Host streaming only (no collision), matching what the graph computes.
fn host_stream(f: &rlx_lbm::sim::Field2d) -> Vec<[f32; 6]> {
    let (nx, ny) = (f.nx, f.ny);
    let mut out = Vec::with_capacity(nx * ny);
    for y in 0..ny {
        for x in 0..nx {
            let mut pop = [0.0f64; 9];
            for (i, c) in D2Q9_C.iter().enumerate() {
                let sx = (x as isize - c[0] as isize).rem_euclid(nx as isize) as usize;
                let sy = (y as isize - c[1] as isize).rem_euclid(ny as isize) as usize;
                pop[i] = reconstruct_d2q9(&f.at(sx, sy))[i];
            }
            let m = moments_d2q9(&pop);
            out.push([
                m.rho as f32,
                m.u[0] as f32,
                m.u[1] as f32,
                m.s[0] as f32,
                m.s[1] as f32,
                m.s[2] as f32,
            ]);
        }
    }
    out
}

#[test]
fn graph_stream_matches_host_stream() {
    let (nx, ny) = (16usize, 16usize);
    let fld = taylor_green(nx, 0.03, 0.01);

    // Flatten the host state into the graph's six input buffers.
    let mut ins: [Vec<f32>; 6] = Default::default();
    for y in 0..ny {
        for x in 0..nx {
            let m = fld.at(x, y);
            ins[0].push(m.rho as f32);
            ins[1].push(m.u[0] as f32);
            ins[2].push(m.u[1] as f32);
            ins[3].push(m.s[0] as f32);
            ins[4].push(m.s[1] as f32);
            ins[5].push(m.s[2] as f32);
        }
    }

    let (g, outs) = stream_step_graph(nx, ny);
    assert_eq!(outs.len(), 6);
    let feeds: Vec<(&str, &[f32])> = FIELD_NAMES
        .iter()
        .zip(ins.iter())
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    let got = Session::new(Device::Cpu).compile(g).run(&feeds);
    assert_eq!(got.len(), 6);

    let want = host_stream(&fld);
    for (fi, name) in FIELD_NAMES.iter().enumerate() {
        for (ci, w) in want.iter().enumerate() {
            let a = got[fi][ci];
            let b = w[fi];
            assert!(
                (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                "{name}[{ci}]: graph {a} vs host {b}"
            );
        }
    }
}

/// Streaming alone conserves mass exactly — it is a permutation of populations.
#[test]
fn graph_stream_conserves_mass() {
    let (nx, ny) = (12usize, 12usize);
    let fld = taylor_green(nx, 0.02, 0.01);
    let mut ins: [Vec<f32>; 6] = Default::default();
    for y in 0..ny {
        for x in 0..nx {
            let m = fld.at(x, y);
            for (k, v) in [m.rho, m.u[0], m.u[1], m.s[0], m.s[1], m.s[2]]
                .iter()
                .enumerate()
            {
                ins[k].push(*v as f32);
            }
        }
    }
    let before: f32 = ins[0].iter().sum();

    let (g, _) = stream_step_graph(nx, ny);
    let feeds: Vec<(&str, &[f32])> = FIELD_NAMES
        .iter()
        .zip(ins.iter())
        .map(|(n, v)| (*n, v.as_slice()))
        .collect();
    let got = Session::new(Device::Cpu).compile(g).run(&feeds);
    let after: f32 = got[0].iter().sum();

    assert!(
        (after - before).abs() / before < 1e-5,
        "graph streaming changed mass {before} -> {after}"
    );
}
