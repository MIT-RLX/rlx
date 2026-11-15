// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Two cases backends got wrong elsewhere, against host references (not just
//! the CPU backend, which had the broadcast bug itself):
//!
//! - batched matmul with an operand broadcast along only some batch axes
//!   (`[2,1,M,K]·[1,3,K,N]`, `[2,3,M,K]·[3,K,N]`);
//! - attention with a logit soft-cap, applied to the scaled scores before the
//!   mask (the models' own order: a masked key gets weight 0).

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

const DEV: Device = Device::Cuda;

fn available() -> bool {
    !rlx_ir::env::skip_unless_device("cuda", true, rlx_runtime::is_available(DEV))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn host_mm(sa: &[usize], sb: &[usize], a: &[f32], b: &[f32]) -> Vec<f32> {
    let r = sa.len().max(sb.len());
    let pad = |s: &[usize]| {
        let mut v = vec![1; r - s.len()];
        v.extend_from_slice(s);
        v
    };
    let (pa, pb) = (pad(sa), pad(sb));
    let (m, k, n) = (pa[r - 2], pa[r - 1], pb[r - 1]);
    let ob: Vec<usize> = (0..r - 2).map(|d| pa[d].max(pb[d])).collect();
    let idx = |p: &[usize], mut o: usize| {
        let (mut i, mut st) = (0, 1);
        for d in (0..r - 2).rev() {
            let c = o % ob[d];
            o /= ob[d];
            if p[d] != 1 {
                i += c * st;
            }
            st *= p[d];
        }
        i
    };
    let tot: usize = ob.iter().product();
    let mut c = vec![0f32; tot * m * n];
    for o in 0..tot {
        let (ia, ib) = (idx(&pa, o), idx(&pb, o));
        for i in 0..m {
            for j in 0..n {
                c[o * m * n + i * n + j] = (0..k)
                    .map(|kk| a[ia * m * k + i * k + kk] * b[ib * k * n + kk * n + j])
                    .sum();
            }
        }
    }
    c
}

#[test]
fn partial_batch_broadcast_matmul() {
    if !available() {
        return;
    }
    let (m, k, n) = (5, 7, 4);
    for (sa, sb) in [
        (vec![2, 1, m, k], vec![1, 3, k, n]),
        (vec![2, 3, m, k], vec![1, 3, k, n]),
        (vec![2, 1, m, k], vec![2, 3, k, n]),
        (vec![2, 3, m, k], vec![3, k, n]),
        (vec![1, 8, m, k], vec![8, k, n]),
    ] {
        let a: Vec<f32> = (0..sa.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let b: Vec<f32> = (0..sb.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.23).cos())
            .collect();
        let mut g = Graph::new("bcast_mm");
        let x = g.input("a", Shape::new(&sa, DType::F32));
        let y = g.input("b", Shape::new(&sb, DType::F32));
        let z = g.mm(x, y);
        g.set_outputs(vec![z]);
        let got = Session::new(DEV)
            .compile(g)
            .run(&[("a", &a), ("b", &b)])
            .remove(0);
        let err = max_abs(&got, &host_mm(&sa, &sb, &a, &b));
        eprintln!("{sa:?} x {sb:?}: max |d| {err:.2e}");
        assert!(err < 1e-4, "{sa:?} x {sb:?}: {err}");
    }
}

#[test]
fn softcap_attention() {
    if !available() {
        return;
    }
    let (nh, hd, cap) = (2usize, 32usize, 5f32);
    let dim = nh * hd;
    for (seq, causal) in [(8usize, true), (100, true), (40, false)] {
        let q: Vec<f32> = (0..seq * dim)
            .map(|i| ((i as f32) * 0.021).sin() * 1.5)
            .collect();
        let kk: Vec<f32> = (0..seq * dim)
            .map(|i| ((i as f32) * 0.017).cos() * 1.5)
            .collect();
        let v: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.013).sin()).collect();
        let mut want = vec![0f32; seq * dim];
        for h in 0..nh {
            for qi in 0..seq {
                let sc: Vec<f64> = (0..seq)
                    .map(|ki| {
                        let s: f64 = (0..hd)
                            .map(|d| {
                                f64::from(q[qi * dim + h * hd + d] * kk[ki * dim + h * hd + d])
                            })
                            .sum();
                        let s = f64::from(cap) * (s / f64::from(cap)).tanh();
                        if causal && ki > qi { -1e9 } else { s }
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
                let w: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = w.iter().sum();
                for d in 0..hd {
                    want[qi * dim + h * hd + d] = ((0..seq)
                        .map(|ki| w[ki] * f64::from(v[ki * dim + h * hd + d]))
                        .sum::<f64>()
                        / z) as f32;
                }
            }
        }
        let mut g = Graph::new("attn_softcap");
        let qi = g.input("q", Shape::new(&[1, seq, dim], DType::F32));
        let ki = g.input("k", Shape::new(&[1, seq, dim], DType::F32));
        let vi = g.input("v", Shape::new(&[1, seq, dim], DType::F32));
        let y = g.add_node(
            Op::Attention {
                num_heads: nh,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: if causal {
                    MaskKind::Causal
                } else {
                    MaskKind::None
                },
                score_scale: Some(1.0),
                attn_logit_softcap: Some(cap),
            },
            vec![qi, ki, vi],
            Shape::new(&[1, seq, dim], DType::F32),
        );
        g.set_outputs(vec![y]);
        let got = Session::new(DEV)
            .compile(g)
            .run(&[("q", &q), ("k", &kk), ("v", &v)])
            .remove(0);
        let err = max_abs(&got, &want);
        eprintln!("softcap seq={seq} causal={causal}: max |d| {err:.2e}");
        assert!(err < 1e-4, "softcap seq={seq} causal={causal}: {err}");
    }
}
