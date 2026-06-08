// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Basic tests for the CUDA backend.
//!
//! Every test starts with `if !rlx_cuda::is_available() { return; }` —
//! the crate compiles fine on Mac (and any other CUDA-less host) via
//! cudarc's dynamic-loading, so unit-test runs on those machines just
//! no-op. On a real CUDA box the same tests dispatch and assert on
//! actual GPU output.

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Shape};

const QK_K: usize = 256;

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn binary_add_matches_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.input("y", Shape::new(&[4], DType::F32));
    let z = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![z]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[
        ("x", &[1.0_f32, 2.0, 3.0, 4.0]),
        ("y", &[10.0_f32, 20.0, 30.0, 40.0]),
    ]);
    assert_eq!(out[0], vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn relu_clamps_negatives_to_zero() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("relu");
    let x = g.input("x", Shape::new(&[5], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[5], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[("x", &[-2.0_f32, -0.5, 0.0, 1.0, 3.0])]);
    assert_eq!(out[0], vec![0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn matmul_2x3x2_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[3, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param("w", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let xv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outs = exe.run(&[("x", &xv)]);
    // Reference: row-major matmul.
    let mut want = vec![0.0_f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                want[i * 2 + j] += xv[i * 3 + k] * [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][k * 2 + j];
            }
        }
    }
    assert!(
        close(&outs[0], &want, 1e-4),
        "matmul mismatch: got {:?} want {want:?}",
        outs[0]
    );
}

#[test]
fn gated_delta_net_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    use rlx_ir::Op;

    let (b, s, h, n) = (1, 4, 2, 3);
    let mut g = Graph::new("gdn");
    let bshn = Shape::new(&[b, s, h, n], DType::F32);
    let bsh = Shape::new(&[b, s, h], DType::F32);
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let g_in = g.input("g", bsh.clone());
    let beta = g.input("beta", bsh);
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: n,
            carry_state: false,
        },
        vec![q, k, v, g_in, beta],
        bshn,
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);

    let nqkv = b * s * h * n;
    let ngb = b * s * h;
    let q_data: Vec<f32> = (0..nqkv).map(|i| 0.05 + 0.03 * (i as f32)).collect();
    let k_data: Vec<f32> = (0..nqkv).map(|i| 0.10 + 0.02 * (i as f32)).collect();
    let v_data: Vec<f32> = (0..nqkv).map(|i| 0.30 + 0.05 * (i as f32)).collect();
    let g_data: Vec<f32> = (0..ngb).map(|i| -0.20 - 0.01 * (i as f32)).collect();
    let beta_data: Vec<f32> = (0..ngb).map(|i| 0.40 + 0.02 * (i as f32)).collect();

    let r = exe.run(&[
        ("q", &q_data),
        ("k", &k_data),
        ("v", &v_data),
        ("g", &g_data),
        ("beta", &beta_data),
    ]);

    let scale = 1.0f32 / (n as f32).sqrt();
    let mut want = vec![0f32; nqkv];
    let mut state = vec![0f32; h * n * n];
    let mut sk = vec![0f32; n];

    for bi in 0..b {
        for st in state.iter_mut() {
            *st = 0.0;
        }
        for ti in 0..s {
            let step_qkv = bi * s * h * n + ti * h * n;
            let step_gb = bi * s * h + ti * h;
            for hi in 0..h {
                let q_row = &q_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let k_row = &k_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let v_row = &v_data[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                let g_t = g_data[step_gb + hi];
                let beta_t = beta_data[step_gb + hi];

                let s_base = hi * n * n;
                let s_mat = &mut state[s_base..s_base + n * n];

                let g_exp = g_t.exp();
                for v in s_mat.iter_mut() {
                    *v *= g_exp;
                }
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * k_row[i];
                    }
                    sk[j] = acc;
                }
                for j in 0..n {
                    sk[j] = (v_row[j] - sk[j]) * beta_t;
                }
                for i in 0..n {
                    for j in 0..n {
                        s_mat[i * n + j] += k_row[i] * sk[j];
                    }
                }
                let out_row = &mut want[step_qkv + hi * n..step_qkv + (hi + 1) * n];
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for i in 0..n {
                        acc += s_mat[i * n + j] * q_row[i];
                    }
                    out_row[j] = acc * scale;
                }
            }
        }
    }
    assert!(
        close(&r[0], &want, 1e-4),
        "GatedDeltaNet mismatch: got {:?} want {want:?}",
        r[0]
    );
}

#[test]
fn dequant_matmul_gguf_q8k_matches_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let k = 256;
    let n = 1;
    let m = 2;
    let scale = 0.0625f32;
    let qs: [i8; QK_K] = std::array::from_fn(|i| (i as i32 - 128) as i8);
    let mut packed = Vec::new();
    packed.extend_from_slice(&scale.to_le_bytes());
    for &q in &qs {
        packed.push(q as u8);
    }
    for _ in 0..(QK_K / 16) {
        packed.extend_from_slice(&0i16.to_le_bytes());
    }
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001 - 0.5).collect();
    let mut expected = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += x[r * k + kk] * (scale * qs[kk] as f32);
            }
            expected[r * n + c] = acc;
        }
    }

    let mut g = Graph::new("dq_gguf_q8k");
    let x_in = g.input("x", Shape::new(&[m, k], DType::F32));
    let w_param = g.param("w_q", Shape::new(&[packed.len()], DType::U8));
    let y = g.add_node(
        rlx_ir::Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8K,
        },
        vec![x_in, w_param],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param_bytes("w_q", &packed);
    let out = exe.run(&[("x", &x)]);
    assert!(
        close(&out[0], &expected, 1e-3),
        "GGUF Q8K DequantMatMul mismatch: got {:?} want {expected:?}",
        out[0]
    );
}

#[test]
fn layer_norm2d_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let n = 1usize;
    let c = 4usize;
    let h = 3usize;
    let w = 3usize;
    let x: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.01 - 0.1)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + 0.05 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.02 * i as f32).collect();
    let mut want = vec![0f32; x.len()];
    rlx_cpu::kernels::layer_norm2d_nchw(&x, &gamma, &beta, &mut want, n, c, h, w, 1e-5);

    let mut g = Graph::new("ln2d");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[c], DType::F32));
    let b_p = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.layer_norm2d(x_in, g_p, b_p, 1e-5);
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param("gamma", &gamma);
    exe.set_param("beta", &beta);
    let out = exe.run(&[("x", &x)]);
    assert!(
        close(&out[0], &want, 1e-4),
        "LayerNorm2d mismatch: max |Δ| = {:.3e}",
        out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
}

#[test]
fn conv_transpose2d_stride2_k2_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let n = 1usize;
    let c_in = 2usize;
    let h = 4usize;
    let w_in = 4usize;
    let c_out = 3usize;
    let kh = 2usize;
    let kw = 2usize;
    let sh = 2usize;
    let sw = 2usize;
    let ph = 0usize;
    let pw = 0usize;
    let dh = 1usize;
    let dw = 1usize;
    let groups = 1usize;
    let h_out = (h - 1) * sh - 2 * ph + dh * (kh - 1) + 1;
    let w_out = (w_in - 1) * sw - 2 * pw + dw * (kw - 1) + 1;
    let x: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| (i as f32) * 0.02 - 0.2)
        .collect();
    let weight: Vec<f32> = (0..c_in * c_out * kh * kw)
        .map(|i| 0.1 + 0.01 * (i as f32))
        .collect();
    let mut want = vec![0f32; n * c_out * h_out * w_out];
    rlx_cpu::kernels::conv_transpose2d_nchw(
        &x, &weight, &mut want, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh,
        dw, groups,
    );

    let mut g = Graph::new("conv_t2d");
    let x_in = g.input("x", Shape::new(&[n, c_in, h, w_in], DType::F32));
    let w_p = g.param("w", Shape::new(&[c_in, c_out, kh, kw], DType::F32));
    let y = g.conv_transpose2d(
        x_in,
        w_p,
        [kh, kw],
        [sh, sw],
        [ph, pw],
        [dh, dw],
        [0, 0],
        groups,
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param("w", &weight);
    let out = exe.run(&[("x", &x)]);
    assert!(
        close(&out[0], &want, 1e-4),
        "ConvTranspose2d mismatch: max |Δ| = {:.3e}",
        out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
}

#[test]
fn group_norm_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let n = 1usize;
    let c = 8usize;
    let h = 4usize;
    let w = 4usize;
    let num_groups = 2usize;
    let x: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32) * 0.01 - 0.2)
        .collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + 0.02 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| -0.01 * i as f32).collect();
    let mut want = vec![0f32; x.len()];
    rlx_cpu::kernels::group_norm_nchw(&x, &gamma, &beta, &mut want, n, c, h, w, num_groups, 1e-5);

    let mut g = Graph::new("gn");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let g_p = g.param("gamma", Shape::new(&[c], DType::F32));
    let b_p = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x_in, g_p, b_p, num_groups, 1e-5);
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param("gamma", &gamma);
    exe.set_param("beta", &beta);
    let out = exe.run(&[("x", &x)]);
    assert!(
        close(&out[0], &want, 1e-4),
        "GroupNorm mismatch: max |Δ| = {:.3e}",
        out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
}

#[test]
fn resize_nearest_2x_matches_cpu_reference() {
    if !rlx_cuda::is_available() {
        return;
    }
    let n = 1usize;
    let c = 3usize;
    let h = 5usize;
    let w = 7usize;
    let x: Vec<f32> = (0..n * c * h * w).map(|i| (i as f32) * 0.003).collect();
    let mut want = vec![0f32; n * c * h * 2 * w * 2];
    rlx_cpu::kernels::resize_nearest_2x_nchw(&x, &mut want, c, h, w);

    let mut g = Graph::new("up2");
    let x_in = g.input("x", Shape::new(&[n, c, h, w], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::ResizeNearest2x,
        vec![x_in],
        Shape::new(&[n, c, h * 2, w * 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    let out = exe.run(&[("x", &x)]);
    assert!(
        close(&out[0], &want, 1e-6),
        "ResizeNearest2x mismatch: max |Δ| = {:.3e}",
        out[0]
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
}

#[test]
fn attention_bshd_eeg_shape_matches_cpu() {
    if !rlx_cuda::is_available() {
        return;
    }
    use rlx_ir::op::MaskKind;

    let (b, s, nh, dh) = (1, 191, 8, 25);
    let n = b * s * nh * dh;
    let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 0.3).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03) % 1.0 - 0.5).collect();

    let want = rlx_ir::cpu_attention_bshd(&q, &k, &v, b, s, nh, dh);

    let mut g = Graph::new("bshd_eeg");
    let qi = g.input("q", Shape::new(&[b, s, nh, dh], DType::F32));
    let ki = g.input("k", Shape::new(&[b, s, nh, dh], DType::F32));
    let vi = g.input("v", Shape::new(&[b, s, nh, dh], DType::F32));
    let o = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[b, s, nh, dh], DType::F32),
    );
    g.set_outputs(vec![o]);
    let mut exe = CudaExecutable::compile(g);
    let got = exe
        .run(&[("q", &q), ("k", &k), ("v", &v)])
        .into_iter()
        .next()
        .unwrap();
    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-3,
        "BSHD flash attention [1,191,8,25] max_abs={err:.3e}",
    );
}

#[test]
fn packed_bshd_attn_matches_cpu_ref() {
    if !rlx_cuda::is_available() {
        return;
    }
    use rlx_ir::op::{MaskKind, Op};

    let (b, s, nh, dh) = (1, 191, 8, 25);
    let hd = nh * dh;
    let f = DType::F32;
    let n = b * s * hd;
    let x_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
    let w_v: Vec<f32> = (0..(hd * 3 * hd))
        .map(|i| (i as f32 * 0.01).cos() * 0.1)
        .collect();
    let b_v: Vec<f32> = (0..(3 * hd)).map(|i| i as f32 * 0.001).collect();

    let mut g_pack = Graph::new("pack");
    let x = g_pack.input("x", Shape::new(&[b, s, hd], f));
    let w = g_pack.param("w", Shape::new(&[hd, 3 * hd], f));
    let bias = g_pack.param("b", Shape::new(&[3 * hd], f));
    let qkv = g_pack.add_node(
        Op::FusedMatMulBiasAct { activation: None },
        vec![x, w, bias],
        Shape::new(&[b, s, 3 * hd], f),
    );
    let qkv4 = g_pack.reshape_(qkv, vec![b as i64, s as i64, 3, nh as i64, dh as i64]);
    g_pack.set_outputs(vec![qkv4]);

    let mut pack_exe = CudaExecutable::compile(g_pack);
    pack_exe.set_param("w", &w_v);
    pack_exe.set_param("b", &b_v);
    let packed = pack_exe.run(&[("x", &x_v)]).into_iter().next().unwrap();
    let want = rlx_ir::cpu_attention_packed_bshd_qkv(&packed, b, s, nh, dh);

    let mut g_attn = Graph::new("attn_pack");
    let pin = g_attn.input("p", Shape::new(&[b, s, 3, nh, dh], f));
    let q0 = g_attn.add_node(
        Op::Narrow {
            axis: 2,
            start: 0,
            len: 1,
        },
        vec![pin],
        Shape::new(&[b, s, 1, nh, dh], f),
    );
    let k0 = g_attn.add_node(
        Op::Narrow {
            axis: 2,
            start: 1,
            len: 1,
        },
        vec![pin],
        Shape::new(&[b, s, 1, nh, dh], f),
    );
    let v0 = g_attn.add_node(
        Op::Narrow {
            axis: 2,
            start: 2,
            len: 1,
        },
        vec![pin],
        Shape::new(&[b, s, 1, nh, dh], f),
    );
    let q = g_attn.reshape_(q0, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let k = g_attn.reshape_(k0, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let v = g_attn.reshape_(v0, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let out = g_attn.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, nh, dh], f),
    );
    g_attn.set_outputs(vec![out]);
    let mut attn_exe = CudaExecutable::compile(g_attn);
    let got = attn_exe.run(&[("p", &packed)]).into_iter().next().unwrap();
    let err = want
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-3,
        "packed BSHD flash attn max_abs={err:.3e} idx26862 want={} got={}",
        want[26862],
        got[26862]
    );
}

#[test]
fn run_slots_matches_run_single_output() {
    if !rlx_cuda::is_available() {
        return;
    }
    let mut g = Graph::new("slots");
    let x = g.input("x", Shape::new(&[1, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 4], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 4], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CudaExecutable::compile(g);
    exe.set_param(
        "w",
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    let xv = [[1.0_f32, 2.0, 3.0, 4.0]];
    let via_run = exe.run(&[("x", &xv[0])])[0].clone();
    let slots = exe.run_slots(&[&xv[0]]).to_vec();
    assert_eq!(slots.len(), 1);
    let (byte_off, len) = slots[0];
    assert!(!exe.arena_ptr().is_null());
    let ptr = unsafe { exe.arena_ptr().add(byte_off) as *const f32 };
    let got = unsafe { std::slice::from_raw_parts(ptr, len) };
    assert!(
        close(got, &via_run, 1e-5),
        "run_slots readback mismatch: {:?} vs {:?}",
        got,
        via_run
    );
}
