// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ArgMin` / `ArgMax` / `ScatterAdd` over F64 on every backend. Candidates
//! that are equal in f32 and differ only below its precision decide the
//! winner, so an f32 computation (or a double-single lowering that compares
//! only the high halves) picks the wrong index; scattered sums cancel to a
//! residue f32 cannot hold.

mod common;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, NumberFormat, Session};

const ROWS: usize = 4;
const COLS: usize = 6;

fn bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Row `r`'s extreme sits at column `(r + 1) % COLS`, apart from its rivals
/// only by `1e-12`, far below f32 resolution at 1.0.
fn rows() -> Vec<f64> {
    let mut v = vec![1.0f64; ROWS * COLS];
    for r in 0..ROWS {
        v[r * COLS + (r + 1) % COLS] = 1.0 - 1e-12;
        v[r * COLS + (r + 3) % COLS] = 1.0 + 1e-12;
    }
    v
}

fn graph() -> Graph {
    let f = DType::F64;
    let mut g = Graph::new("f64_argmin_scatter");
    let x = g.input("x", Shape::new(&[ROWS, COLS], f));
    let idx = g.input("idx", Shape::new(&[ROWS], DType::F32));
    let amin = g.add_node(
        Op::ArgMin {
            axis: 1,
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[ROWS], DType::F32),
    );
    let amax = g.add_node(
        Op::ArgMax {
            axis: 1,
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[ROWS], DType::F32),
    );
    // Rows 0 and 2 go to slot 0, 1 and 3 to slot 1.
    let scat = g.add_node(
        Op::ScatterAdd { axis: 0 },
        vec![x, idx],
        Shape::new(&[2, COLS], f),
    );
    g.set_outputs(vec![amin, amax, scat]);
    g
}

fn run(device: Device, opts: &CompileOptions) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
    let x = rows();
    let idx: Vec<f32> = (0..ROWS).map(|r| (r % 2) as f32).collect();
    let idx_b: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut cg = Session::new(device).compile_with(graph(), opts);
    let outs = cg.run_typed(&[("x", &bytes(&x), DType::F64), ("idx", &idx_b, DType::F32)]);
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let f64s = |b: &[u8]| -> Vec<f64> {
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    };
    assert_eq!(outs[2].1, DType::F64, "{device:?}: scatter dtype");
    (f32s(&outs[0].0), f32s(&outs[1].0), f64s(&outs[2].0))
}

#[test]
fn f64_argmin_argmax_scatter_on_every_backend() {
    let _gpu = common::serialize_gpu();
    let x = rows();
    let want_min: Vec<f32> = (0..ROWS).map(|r| ((r + 1) % COLS) as f32).collect();
    let want_max: Vec<f32> = (0..ROWS).map(|r| ((r + 3) % COLS) as f32).collect();
    let mut want_sum = vec![0f64; 2 * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            want_sum[(r % 2) * COLS + c] += x[r * COLS + c];
        }
    }
    let mut df = CompileOptions::new();
    df.number_format = Some(NumberFormat::F64x2);
    let mut cases = vec![
        (Device::Cpu, CompileOptions::new(), "native"),
        (Device::Cpu, df.clone(), "f64x2"),
    ];
    for device in [
        Device::Metal,
        Device::Mlx,
        Device::Gpu,
        Device::Vulkan,
        Device::Cuda,
        Device::Rocm,
    ] {
        if !common::skip_unless(device) {
            cases.push((device, CompileOptions::new(), "default"));
            cases.push((device, df.clone(), "f64x2"));
        }
    }
    for (device, opts, label) in cases {
        let (amin, amax, sum) = run(device, &opts);
        assert_eq!(amin, want_min, "{device:?} {label}: argmin");
        assert_eq!(amax, want_max, "{device:?} {label}: argmax");
        // Each slot sums two rows of 1 ± 1e-12: the residue needs ~48 bits.
        let err = sum
            .iter()
            .zip(&want_sum)
            .map(|(g, w)| (g - w).abs())
            .fold(0f64, f64::max);
        eprintln!("{device:?} {label}: argmin/argmax ok, scatter max err {err:.1e}");
        assert!(err < 1e-13, "{device:?} {label}: scatter err {err:e}");
    }
}
