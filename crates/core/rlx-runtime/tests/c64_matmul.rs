// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Complex (C64) matrix multiply: forward vs a textbook complex GEMM,
//! and the Wirtinger reverse-mode gradient (`∂L/∂z̄` convention) for
//! `L = Σ|A·B|²` against its closed form `dA = C·conj(B)ᵀ`,
//! `dB = conj(A)ᵀ·C`.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn const_c64(g: &mut Graph, re: &[f32], im: &[f32], shape: &[usize]) -> rlx_ir::NodeId {
    let mut bytes = Vec::with_capacity(2 * re.len() * 4);
    for i in 0..re.len() {
        bytes.extend_from_slice(&re[i].to_le_bytes());
        bytes.extend_from_slice(&im[i].to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(shape, DType::C64),
    )
}

fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Textbook complex GEMM `C[m,n] = A[m,k] · B[k,n]` (split re/im).
fn cmatmul(
    ar: &[f32],
    ai: &[f32],
    br: &[f32],
    bi: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cr = vec![0f32; m * n];
    let mut ci = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let (mut re, mut im) = (0f32, 0f32);
            for l in 0..k {
                let (x, y) = (ar[i * k + l], ai[i * k + l]);
                let (u, v) = (br[l * n + j], bi[l * n + j]);
                re += x * u - y * v;
                im += x * v + y * u;
            }
            cr[i * n + j] = re;
            ci[i * n + j] = im;
        }
    }
    (cr, ci)
}

const M: usize = 2;
const K: usize = 3;
const N: usize = 2;

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let ar: Vec<f32> = (0..M * K).map(|i| 0.2 + 0.1 * i as f32).collect();
    let ai: Vec<f32> = (0..M * K).map(|i| -0.3 + 0.05 * i as f32).collect();
    let br: Vec<f32> = (0..K * N).map(|i| 0.5 - 0.07 * i as f32).collect();
    let bi: Vec<f32> = (0..K * N).map(|i| 0.1 + 0.09 * i as f32).collect();
    (ar, ai, br, bi)
}

#[test]
fn c64_matmul_forward_matches_textbook() {
    let (ar, ai, br, bi) = inputs();
    let (cr, ci) = cmatmul(&ar, &ai, &br, &bi, M, K, N);

    let mut g = Graph::new("c64_matmul");
    let a = const_c64(&mut g, &ar, &ai, &[M, K]);
    let b = const_c64(&mut g, &br, &bi, &[K, N]);
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::C64));
    g.set_outputs(vec![c]);

    let out = Session::new(Device::Cpu).compile(g).run(&[]).pop().unwrap();
    assert_eq!(out.len(), 2 * M * N);
    for i in 0..M * N {
        assert!(
            (out[2 * i] - cr[i]).abs() < 1e-5 && (out[2 * i + 1] - ci[i]).abs() < 1e-5,
            "C[{i}] = ({}, {}) vs textbook ({}, {})",
            out[2 * i],
            out[2 * i + 1],
            cr[i],
            ci[i]
        );
    }
}

#[test]
fn c64_matmul_wirtinger_grad_matches_closed_form() {
    let (ar, ai, br, bi) = inputs();
    let (cr, ci) = cmatmul(&ar, &ai, &br, &bi, M, K, N);

    // dA = C · conj(B)ᵀ  [M,K];  conj(B)ᵀ is [N,K] with element (j,l) = conj(B[l,j]).
    let mut cbt_re = vec![0f32; N * K];
    let mut cbt_im = vec![0f32; N * K];
    for l in 0..K {
        for j in 0..N {
            cbt_re[j * K + l] = br[l * N + j];
            cbt_im[j * K + l] = -bi[l * N + j];
        }
    }
    let (da_re, da_im) = cmatmul(&cr, &ci, &cbt_re, &cbt_im, M, N, K);

    // dB = conj(A)ᵀ · C  [K,N];  conj(A)ᵀ is [K,M] with element (l,i) = conj(A[i,l]).
    let mut cat_re = vec![0f32; K * M];
    let mut cat_im = vec![0f32; K * M];
    for i in 0..M {
        for l in 0..K {
            cat_re[l * M + i] = ar[i * K + l];
            cat_im[l * M + i] = -ai[i * K + l];
        }
    }
    let (db_re, db_im) = cmatmul(&cat_re, &cat_im, &cr, &ci, K, M, N);

    // L = Σ|C|² ⇒ ∂L/∂C̄ = C, so the reported ∂L/∂Ā / ∂L/∂B̄ equal the above.
    let mut g = Graph::new("c64_matmul_grad");
    let a = const_c64(&mut g, &ar, &ai, &[M, K]);
    let b = const_c64(&mut g, &br, &bi, &[K, N]);
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::C64));
    let nsq = g.complex_norm_sq(c); // F32 [M,N]
    let loss = g.sum(nsq, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    let outs = Session::new(Device::Cpu).compile(bwd).run_typed(&[(
        "d_output",
        &1.0f32.to_le_bytes(),
        DType::F32,
    )]);
    let da = bytes_to_f32s(&outs[1].0);
    let db = bytes_to_f32s(&outs[2].0);
    assert_eq!(da.len(), 2 * M * K);
    assert_eq!(db.len(), 2 * K * N);

    for i in 0..M * K {
        assert!(
            (da[2 * i] - da_re[i]).abs() < 1e-4 && (da[2 * i + 1] - da_im[i]).abs() < 1e-4,
            "dA[{i}] = ({}, {}) vs closed form ({}, {})",
            da[2 * i],
            da[2 * i + 1],
            da_re[i],
            da_im[i]
        );
    }
    for i in 0..K * N {
        assert!(
            (db[2 * i] - db_re[i]).abs() < 1e-4 && (db[2 * i + 1] - db_im[i]).abs() < 1e-4,
            "dB[{i}] = ({}, {}) vs closed form ({}, {})",
            db[2 * i],
            db[2 * i + 1],
            db_re[i],
            db_im[i]
        );
    }
}
