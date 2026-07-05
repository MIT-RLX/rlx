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

//! End-to-end CPU numeric tests for the forward 3-D conv kernels
//! (`Op::Conv3d`, `Op::ConvTranspose3d`) and the separable 3-D resize
//! (`Graph::interpolate3d`) added for MONAI-style 3-D decoders. Each builds an
//! input-fed graph, runs it on the CPU backend via `compile_thunks` /
//! `execute_thunks`, and checks the numbers against a nested-loop reference
//! computed in-test.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::ops::upsample::InterpMode;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn input(g: &mut Graph, name: &str, shape: &[usize]) -> NodeId {
    g.input(name, Shape::new(shape, DType::F32))
}

/// Compile + run `g` on the CPU backend, feeding each `Op::Input` from
/// `inputs` (matched by name). Returns (output values, output dims).
fn run(g: &Graph, inputs: &[(&str, Vec<f32>)]) -> (Vec<f32>, Vec<usize>) {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(g, &arena);
    // Materialize Op::Constant slots (e.g. interpolate3d's resample matrices)
    // into the arena — the same job rlx_runtime::backend does for production.
    for node in g.nodes() {
        if let Op::Constant { data } = &node.op {
            if arena.has_buffer(node.id) && !data.is_empty() {
                let buf = arena.slice_mut(node.id);
                let n = buf.len().min(data.len() / 4);
                for i in 0..n {
                    buf[i] = f32::from_le_bytes([
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ]);
                }
            }
        }
    }
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &inputs
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing data for input {name}"))
                .1;
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                for (i, &v) in data.iter().enumerate() {
                    *p.add(i) = v;
                }
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out_id = g.outputs[0];
    let dims: Vec<usize> = g
        .shape(out_id)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let n: usize = dims.iter().product();
    let off = arena.byte_offset(out_id);
    let out = unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    };
    (out, dims)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Reference NCDHW forward conv (cross-correlation). Weight
/// `[C_out, C_in/g, kD, kH, kW]`.
#[allow(clippy::too_many_arguments)]
fn conv3d_ref(
    inp: &[f32],
    wt: &[f32],
    n: usize,
    c_in: usize,
    d: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 3],
    pad: [usize; 3],
    dil: [usize; 3],
    groups: usize,
) -> (Vec<f32>, [usize; 3]) {
    let out_dim =
        |x: usize, k: usize, i: usize| (x + 2 * pad[i] - dil[i] * (k - 1) - 1) / stride[i] + 1;
    let d_out = out_dim(d, kd, 0);
    let h_out = out_dim(h, kh, 1);
    let w_out = out_dim(w, kw, 2);
    let c_in_pg = c_in / groups;
    let c_out_pg = c_out / groups;
    let mut out = vec![0f32; n * c_out * d_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / c_out_pg;
            for od in 0..d_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc = 0f32;
                        for cioff in 0..c_in_pg {
                            let ci = g * c_in_pg + cioff;
                            for a in 0..kd {
                                for b in 0..kh {
                                    for c in 0..kw {
                                        let di = od * stride[0] + a * dil[0];
                                        let hi = ho * stride[1] + b * dil[1];
                                        let wi = wo * stride[2] + c * dil[2];
                                        if di < pad[0] || hi < pad[1] || wi < pad[2] {
                                            continue;
                                        }
                                        let (di, hi, wi) = (di - pad[0], hi - pad[1], wi - pad[2]);
                                        if di >= d || hi >= h || wi >= w {
                                            continue;
                                        }
                                        let iv =
                                            inp[(((ni * c_in + ci) * d + di) * h + hi) * w + wi];
                                        let wv = wt
                                            [(((co * c_in_pg + cioff) * kd + a) * kh + b) * kw + c];
                                        acc += iv * wv;
                                    }
                                }
                            }
                        }
                        out[(((ni * c_out + co) * d_out + od) * h_out + ho) * w_out + wo] = acc;
                    }
                }
            }
        }
    }
    (out, [d_out, h_out, w_out])
}

#[test]
fn conv3d_matches_reference_stride1() {
    let mut g = Graph::new("conv3d");
    let x = input(&mut g, "x", &[1, 1, 3, 3, 3]);
    let w = input(&mut g, "w", &[1, 1, 2, 2, 2]);
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = (1..=27).map(|v| v as f32).collect();
    let w_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (out, dims) = run(&g, &[("x", x_data.clone()), ("w", w_data.clone())]);
    assert_eq!(dims, vec![1, 1, 2, 2, 2], "conv3d output shape");

    let (expect, _) = conv3d_ref(
        &x_data,
        &w_data,
        1,
        1,
        3,
        3,
        3,
        1,
        2,
        2,
        2,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
    );
    assert!(
        max_abs(&out, &expect) < 1e-5,
        "conv3d mismatch: {out:?} vs {expect:?}"
    );

    // Spot-check the [0,0,0] output by hand: dot of the 2×2×2 corner cube
    // (input values 1,2,4,5,10,11,13,14) with weight 1..8.
    let hand = 1.0 * 1.0
        + 2.0 * 2.0
        + 4.0 * 3.0
        + 5.0 * 4.0
        + 10.0 * 5.0
        + 11.0 * 6.0
        + 13.0 * 7.0
        + 14.0 * 8.0;
    assert!(
        (out[0] - hand).abs() < 1e-5,
        "conv3d[0]={} want {hand}",
        out[0]
    );
}

#[test]
fn conv3d_depthwise_groups() {
    // groups=2 depthwise: C_in=C_out=2, weight [2, 1, 2,2,2].
    let mut g = Graph::new("conv3d_dw");
    let x = input(&mut g, "x", &[1, 2, 3, 3, 3]);
    let w = input(&mut g, "w", &[2, 1, 2, 2, 2]);
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 2);
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = (0..54).map(|v| (v as f32) * 0.5 - 3.0).collect();
    let w_data: Vec<f32> = (0..16).map(|v| ((v % 5) as f32) - 2.0).collect();
    let (out, dims) = run(&g, &[("x", x_data.clone()), ("w", w_data.clone())]);
    assert_eq!(dims, vec![1, 2, 2, 2, 2], "depthwise conv3d shape");

    let (expect, _) = conv3d_ref(
        &x_data,
        &w_data,
        1,
        2,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        2,
    );
    assert!(
        max_abs(&out, &expect) < 1e-5,
        "depthwise conv3d mismatch: {out:?} vs {expect:?}"
    );
}

#[test]
fn conv_transpose3d_stride2_upsample() {
    // Stride-2, kernel-2 transposed conv tiles each input voxel into a disjoint
    // 2×2×2 output block → [1,1,2,2,2] upsamples to [1,1,4,4,4].
    let mut g = Graph::new("ct3d");
    let x = input(&mut g, "x", &[1, 1, 2, 2, 2]);
    let w = input(&mut g, "w", &[1, 1, 2, 2, 2]);
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // all ones
    let (out, dims) = run(&g, &[("x", x_data.clone()), ("w", w_data.clone())]);
    assert_eq!(dims, vec![1, 1, 4, 4, 4], "conv_transpose3d output shape");

    // Reference scatter (weight all ones): out[2*id+kz, 2*iy+ky, 2*ix+kx] =
    // input[id,iy,ix]. Blocks are disjoint (stride == kernel).
    let mut expect = vec![0f32; 4 * 4 * 4];
    for id in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                let v = x_data[(id * 2 + iy) * 2 + ix];
                for kz in 0..2 {
                    for ky in 0..2 {
                        for kx in 0..2 {
                            let oz = id * 2 + kz;
                            let oy = iy * 2 + ky;
                            let ox = ix * 2 + kx;
                            expect[(oz * 4 + oy) * 4 + ox] = v;
                        }
                    }
                }
            }
        }
    }
    assert!(
        max_abs(&out, &expect) < 1e-5,
        "conv_transpose3d mismatch: {out:?} vs {expect:?}"
    );
    // A couple of hand-checked scatter values.
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "out[0,0,0] should be input[0]=1"
    );
    // Corner voxel input[1,1,1]=8 lands in the far 2×2×2 block, e.g. [3,3,3].
    assert!(
        (out[(3 * 4 + 3) * 4 + 3] - 8.0).abs() < 1e-5,
        "out[3,3,3] should be input[1,1,1]=8"
    );
}

#[test]
fn interpolate3d_nearest_replicates_2x() {
    let mut g = Graph::new("interp3d_nn");
    let x = input(&mut g, "x", &[1, 1, 2, 2, 2]);
    let y = g.interpolate3d(x, [4, 4, 4], InterpMode::Nearest, false);
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (out, dims) = run(&g, &[("x", x_data.clone())]);
    assert_eq!(dims, vec![1, 1, 4, 4, 4]);

    // Nearest 2×: out[oz,oy,ox] = in[oz/2, oy/2, ox/2].
    let mut expect = vec![0f32; 64];
    for oz in 0..4 {
        for oy in 0..4 {
            for ox in 0..4 {
                expect[(oz * 4 + oy) * 4 + ox] = x_data[((oz / 2) * 2 + oy / 2) * 2 + ox / 2];
            }
        }
    }
    assert!(
        max_abs(&out, &expect) < 1e-5,
        "nearest interpolate3d mismatch: {out:?} vs {expect:?}"
    );
}

#[test]
fn interpolate3d_linear_endpoints_match() {
    let mut g = Graph::new("interp3d_lin");
    let x = input(&mut g, "x", &[1, 1, 2, 2, 2]);
    let y = g.interpolate3d(x, [4, 4, 4], InterpMode::Linear, false);
    g.set_outputs(vec![y]);

    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (out, dims) = run(&g, &[("x", x_data.clone())]);
    assert_eq!(dims, vec![1, 1, 4, 4, 4]);

    // The 8 output corners must equal the 8 input corners (trilinear endpoints,
    // half-pixel + clamp maps the boundary taps exactly onto the source).
    let corner = |z: usize, yy: usize, xx: usize| out[(z * 4 + yy) * 4 + xx];
    let in_corner = |z: usize, yy: usize, xx: usize| x_data[(z * 2 + yy) * 2 + xx];
    for &z in &[0usize, 3] {
        for &yy in &[0usize, 3] {
            for &xx in &[0usize, 3] {
                let got = corner(z, yy, xx);
                let want = in_corner(z / 3, yy / 3, xx / 3);
                assert!(
                    (got - want).abs() < 1e-5,
                    "linear corner [{z},{yy},{xx}]={got} want {want}"
                );
            }
        }
    }
    // Interior values stay within the input range (no overshoot for linear).
    for &v in &out {
        assert!((1.0..=8.0).contains(&v), "linear value {v} out of range");
    }
}
