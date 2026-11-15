// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batched matmul with numpy broadcasting over right-aligned batch dims,
//! against a host reference, at f32 and f64.
//!
//! `BatchedSgemm` / `BatchedDgemmF64` describe each operand with one stride:
//! the full batch, or one matrix repeated. An operand broadcast along only
//! some batch axes (`[2,1,M,K]·[1,3,K,N]`, `[2,3,M,K]·[3,K,N]`) is neither, and
//! was read as if it held the full batch: past its end, into whatever the
//! arena held next. Those shapes now carry a per-matrix index table.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

const M: usize = 5;
const K: usize = 7;
const N: usize = 4;

fn val(seed: usize, i: usize) -> f64 {
    let z = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((z >> 40) & 0xffff) as f64 / 65535.0 - 0.5
}

/// `a · b` with batch dims broadcast numpy-style, in f64.
fn reference(sa: &[usize], sb: &[usize], a: &[f64], b: &[f64]) -> Vec<f64> {
    let r = sa.len().max(sb.len());
    let pad = |s: &[usize]| {
        let mut v = vec![1; r - s.len()];
        v.extend_from_slice(s);
        v
    };
    let (pa, pb) = (pad(sa), pad(sb));
    let out: Vec<usize> = (0..r - 2).map(|d| pa[d].max(pb[d])).collect();
    let index = |p: &[usize], mut o: usize| {
        let (mut i, mut stride) = (0, 1);
        for d in (0..r - 2).rev() {
            let c = o % out[d];
            o /= out[d];
            if p[d] != 1 {
                i += c * stride;
            }
            stride *= p[d];
        }
        i
    };
    let total: usize = out.iter().product();
    let mut c = vec![0f64; total * M * N];
    for o in 0..total {
        let (ia, ib) = (index(&pa, o), index(&pb, o));
        for i in 0..M {
            for j in 0..N {
                c[o * M * N + i * N + j] = (0..K)
                    .map(|kk| a[ia * M * K + i * K + kk] * b[ib * K * N + kk * N + j])
                    .sum();
            }
        }
    }
    c
}

fn run(sa: &[usize], sb: &[usize], dt: DType) -> f64 {
    let mut g = Graph::new("bcast_mm");
    let x = g.input("a", Shape::new(sa, dt));
    let y = g.input("b", Shape::new(sb, dt));
    let z = g.mm(x, y);
    g.set_outputs(vec![z]);
    let a: Vec<f64> = (0..sa.iter().product()).map(|i| val(1, i)).collect();
    let b: Vec<f64> = (0..sb.iter().product()).map(|i| val(2, i)).collect();
    let want = reference(sa, sb, &a, &b);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let bytes = |v: &[f64]| -> Vec<u8> {
        if dt == DType::F64 {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else {
            v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect()
        }
    };
    for node in g.nodes() {
        let Op::Input { name } = &node.op else {
            continue;
        };
        let data = bytes(if name == "a" { &a } else { &b });
        let off = arena.byte_offset(node.id);
        arena.raw_buf_mut()[off..off + data.len()].copy_from_slice(&data);
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let off = arena.byte_offset(z);
    let raw = &arena.raw_buf()[off..];
    let got: Vec<f64> = if dt == DType::F64 {
        raw.as_chunks::<8>().0[..want.len()]
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    } else {
        raw.as_chunks::<4>().0[..want.len()]
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect()
    };
    got.iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max)
}

const SHAPES: [(&[usize], &[usize]); 8] = [
    (&[2, 1, M, K], &[1, 3, K, N]),
    (&[2, 3, M, K], &[1, 3, K, N]),
    (&[2, 3, M, K], &[2, 1, K, N]),
    (&[1, 3, M, K], &[2, 3, K, N]),
    (&[2, 1, M, K], &[2, 3, K, N]),
    (&[2, 3, M, K], &[3, K, N]),
    (&[1, 8, M, K], &[8, K, N]),
    (&[2, 3, M, K], &[2, 3, K, N]),
];

#[test]
fn broadcast_batched_matmul_f32() {
    for (sa, sb) in SHAPES {
        let worst = run(sa, sb, DType::F32);
        assert!(worst < 1e-5, "{sa:?} x {sb:?}: max |d| {worst:e}");
    }
}

#[test]
fn broadcast_batched_matmul_f64() {
    for (sa, sb) in SHAPES {
        let worst = run(sa, sb, DType::F64);
        assert!(worst < 1e-12, "{sa:?} x {sb:?}: max |d| {worst:e}");
    }
}
