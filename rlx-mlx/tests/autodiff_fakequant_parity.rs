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

//! `Op::FakeQuantize` (PerBatch + Fixed) and `Op::FakeQuantizeBackward`
//! (all 4 STEs) on MLX vs hand-written references that mirror
//! `rlx-cpu/src/thunk.rs`. Closes the last published parity gap
//! between `MLX_SUPPORTED_OPS` and `CPU_SUPPORTED_OPS`.

#![cfg(target_os = "macos")]

use rlx_ir::op::{ScaleMode, SteKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::MlxExecutable;

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile(g);
    exe.run(inputs).into_iter().next().unwrap()
}

fn run_with_param(g: Graph, params: &[(&str, &[f32])], inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile(g);
    for (k, v) in params {
        exe.set_param(k, v);
    }
    exe.run(inputs).into_iter().next().unwrap()
}

fn q_max_for(bits: u8) -> f32 {
    match bits {
        8 => 127.0,
        4 => 7.0,
        2 => 1.0,
        _ => panic!("bad bits {bits}"),
    }
}

/// Reference: forward FakeQuantize with PerBatch scale.
/// `chan_dim`/`inner` derive from `axis`; per-channel max-abs → scale →
/// quantize-dequantize.
fn fakequant_perbatch_ref(x: &[f32], shape: &[usize], axis: Option<usize>, bits: u8) -> Vec<f32> {
    let q_max = q_max_for(bits);
    let len = x.len();
    let (chan_dim, inner) = match axis {
        None => (1usize, 1usize),
        Some(c) => {
            let cd = shape[c];
            let inner: usize = shape[c + 1..].iter().product();
            (cd, inner)
        }
    };
    let chan_idx = |i: usize| -> usize {
        if chan_dim == 1 {
            0
        } else {
            (i / inner) % chan_dim
        }
    };
    let mut max_abs = vec![0f32; chan_dim];
    for i in 0..len {
        let a = x[i].abs();
        let c = chan_idx(i);
        if a > max_abs[c] {
            max_abs[c] = a;
        }
    }
    let scale: Vec<f32> = max_abs.iter().map(|m| (m / q_max).max(1e-12)).collect();
    let mut out = vec![0f32; len];
    for i in 0..len {
        let s = scale[chan_idx(i)];
        let qv = (x[i] / s).round().clamp(-q_max, q_max);
        out[i] = qv * s;
    }
    out
}

/// Reference: forward FakeQuantize with Fixed scale.
fn fakequant_fixed_ref(
    x: &[f32],
    state: &[f32],
    shape: &[usize],
    axis: Option<usize>,
    bits: u8,
) -> Vec<f32> {
    let q_max = q_max_for(bits);
    let len = x.len();
    let (chan_dim, inner) = match axis {
        None => (1usize, 1usize),
        Some(c) => {
            let cd = shape[c];
            let inner: usize = shape[c + 1..].iter().product();
            (cd, inner)
        }
    };
    let chan_idx = |i: usize| -> usize {
        if chan_dim == 1 {
            0
        } else {
            (i / inner) % chan_dim
        }
    };
    let scale: Vec<f32> = state.iter().map(|&v| v.max(1e-12)).collect();
    let mut out = vec![0f32; len];
    for i in 0..len {
        let s = scale[chan_idx(i)];
        let qv = (x[i] / s).round().clamp(-q_max, q_max);
        out[i] = qv * s;
    }
    out
}

/// Reference: FakeQuantizeBackward — recompute scale via PerBatch from
/// `x`, then apply the STE-specific gradient formula.
fn fakequant_backward_ref(
    x: &[f32],
    dy: &[f32],
    shape: &[usize],
    axis: Option<usize>,
    bits: u8,
    ste: SteKind,
) -> Vec<f32> {
    let q_max = q_max_for(bits);
    let len = x.len();
    let (chan_dim, inner) = match axis {
        None => (1usize, 1usize),
        Some(c) => {
            let cd = shape[c];
            let inner: usize = shape[c + 1..].iter().product();
            (cd, inner)
        }
    };
    let chan_idx = |i: usize| -> usize {
        if chan_dim == 1 {
            0
        } else {
            (i / inner) % chan_dim
        }
    };
    let mut max_abs = vec![0f32; chan_dim];
    for i in 0..len {
        let a = x[i].abs();
        let c = chan_idx(i);
        if a > max_abs[c] {
            max_abs[c] = a;
        }
    }
    let scale: Vec<f32> = max_abs.iter().map(|m| (m / q_max).max(1e-12)).collect();
    let mut out = vec![0f32; len];
    match ste {
        SteKind::Identity => out.copy_from_slice(dy),
        SteKind::ClippedIdentity => {
            for i in 0..len {
                let bound = q_max * scale[chan_idx(i)];
                out[i] = if x[i].abs() <= bound { dy[i] } else { 0.0 };
            }
        }
        SteKind::Tanh => {
            for i in 0..len {
                let t = (x[i] / scale[chan_idx(i)]).tanh();
                out[i] = dy[i] * (1.0 - t * t);
            }
        }
        SteKind::HardTanh => {
            for i in 0..len {
                let bound = q_max * scale[chan_idx(i)];
                let attn = (1.0 - (x[i] / bound).abs()).max(0.0);
                out[i] = dy[i] * attn;
            }
        }
    }
    out
}

fn check_fakequant_perbatch(shape: &[usize], axis: Option<usize>, bits: u8) {
    let total: usize = shape.iter().product();
    let xs: Vec<f32> = (0..total).map(|i| 0.07 * (i as f32) - 1.3).collect();

    let mut g = Graph::new("fq_pb");
    let x = g.input("x", Shape::new(shape, DType::F32));
    let q = g.add_node(
        Op::FakeQuantize {
            bits,
            axis,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![x],
        Shape::new(shape, DType::F32),
    );
    g.set_outputs(vec![q]);

    let want = fakequant_perbatch_ref(&xs, shape, axis, bits);
    let got = run(g, &[("x", &xs)]);
    assert!(
        close(&got, &want, 1e-5),
        "FakeQuantize PerBatch (shape={shape:?}, axis={axis:?}, bits={bits}): \
         got {got:?} want {want:?}"
    );
}

fn check_fakequant_fixed(shape: &[usize], axis: Option<usize>, bits: u8) {
    let total: usize = shape.iter().product();
    let xs: Vec<f32> = (0..total).map(|i| 0.07 * (i as f32) - 1.3).collect();

    // Choose a scale per channel that doesn't match PerBatch (so test
    // proves we're really using `state`, not recomputing).
    let chan_dim = match axis {
        None => 1usize,
        Some(c) => shape[c],
    };
    let state: Vec<f32> = (0..chan_dim).map(|c| 0.01 + 0.05 * c as f32).collect();

    let mut g = Graph::new("fq_fx");
    let x = g.input("x", Shape::new(shape, DType::F32));
    let s_shape = match axis {
        None => Shape::new(&[1usize], DType::F32),
        Some(_) => Shape::new(&[chan_dim], DType::F32),
    };
    let s = g.param("scale", s_shape);
    let q = g.add_node(
        Op::FakeQuantize {
            bits,
            axis,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::Fixed,
        },
        vec![x, s],
        Shape::new(shape, DType::F32),
    );
    g.set_outputs(vec![q]);

    let want = fakequant_fixed_ref(&xs, &state, shape, axis, bits);
    let got = run_with_param(g, &[("scale", &state)], &[("x", &xs)]);
    assert!(
        close(&got, &want, 1e-5),
        "FakeQuantize Fixed (shape={shape:?}, axis={axis:?}, bits={bits}): \
         got {got:?} want {want:?}"
    );
}

fn check_fakequant_backward(shape: &[usize], axis: Option<usize>, bits: u8, ste: SteKind) {
    let total: usize = shape.iter().product();
    let xs: Vec<f32> = (0..total).map(|i| 0.07 * (i as f32) - 1.3).collect();
    let dys: Vec<f32> = (0..total).map(|i| 0.13 * (i as f32) - 0.4).collect();

    let mut g = Graph::new("fq_bwd");
    let x = g.input("x", Shape::new(shape, DType::F32));
    let dy = g.input("dy", Shape::new(shape, DType::F32));
    let dx = g.add_node(
        Op::FakeQuantizeBackward { bits, axis, ste },
        vec![x, dy],
        Shape::new(shape, DType::F32),
    );
    g.set_outputs(vec![dx]);

    let want = fakequant_backward_ref(&xs, &dys, shape, axis, bits, ste);
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 1e-5),
        "FakeQuantizeBackward (shape={shape:?}, axis={axis:?}, bits={bits}, \
         ste={ste:?}): got {got:?} want {want:?}"
    );
}

#[test]
fn fq_perbatch_per_tensor_8bit() {
    check_fakequant_perbatch(&[6], None, 8);
}
#[test]
fn fq_perbatch_per_channel_8bit() {
    check_fakequant_perbatch(&[3, 4], Some(0), 8);
}
#[test]
fn fq_perbatch_per_channel_4bit() {
    check_fakequant_perbatch(&[2, 5, 3], Some(1), 4);
}
#[test]
fn fq_perbatch_per_channel_2bit() {
    check_fakequant_perbatch(&[3, 4], Some(0), 2);
}

#[test]
fn fq_fixed_per_tensor_8bit() {
    check_fakequant_fixed(&[6], None, 8);
}
#[test]
fn fq_fixed_per_channel_8bit() {
    check_fakequant_fixed(&[3, 4], Some(0), 8);
}
#[test]
fn fq_fixed_per_channel_4bit() {
    check_fakequant_fixed(&[2, 5, 3], Some(1), 4);
}

#[test]
fn fq_backward_identity_per_tensor() {
    check_fakequant_backward(&[6], None, 8, SteKind::Identity);
}
#[test]
fn fq_backward_identity_per_channel() {
    check_fakequant_backward(&[3, 4], Some(0), 8, SteKind::Identity);
}
#[test]
fn fq_backward_clipped_per_channel_4bit() {
    check_fakequant_backward(&[3, 4], Some(0), 4, SteKind::ClippedIdentity);
}
#[test]
fn fq_backward_clipped_per_tensor_2bit() {
    // Tight 2-bit range exercises the clamp-and-zero path on most elements.
    check_fakequant_backward(&[8], None, 2, SteKind::ClippedIdentity);
}
#[test]
fn fq_backward_tanh_per_channel_4bit() {
    check_fakequant_backward(&[3, 4], Some(0), 4, SteKind::Tanh);
}
#[test]
fn fq_backward_tanh_per_tensor_8bit() {
    check_fakequant_backward(&[6], None, 8, SteKind::Tanh);
}
#[test]
fn fq_backward_hard_tanh_per_channel_4bit() {
    check_fakequant_backward(&[3, 4], Some(0), 4, SteKind::HardTanh);
}
#[test]
fn fq_backward_hard_tanh_3d_per_channel_2bit() {
    check_fakequant_backward(&[2, 3, 5], Some(1), 2, SteKind::HardTanh);
}

#[test]
fn fq_ema_returns_clear_error() {
    // ScaleMode::EMA isn't supported yet — confirm we get a clean
    // error rather than a silent miscompute.
    let mut g = Graph::new("fq_ema");
    let x = g.input("x", Shape::new(&[4usize], DType::F32));
    let s = g.param("state", Shape::new(&[1usize], DType::F32));
    let q = g.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: None,
            ste: SteKind::Identity,
            scale_mode: ScaleMode::EMA { decay: 0.99 },
        },
        vec![x, s],
        Shape::new(&[4usize], DType::F32),
    );
    g.set_outputs(vec![q]);

    let xs: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4];
    let mut exe = MlxExecutable::compile(g);
    exe.set_param("state", &[0.05f32]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| exe.run(&[("x", &xs)])));
    assert!(
        result.is_err(),
        "expected EMA to fail with a clear error; got success instead"
    );
}
