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

//! Tier 2 autodiff lowering parity: Conv2dBackwardInput / Conv2dBackwardWeight
//! on MLX vs a hand-written NCHW reference (mirrors `rlx-cpu`'s thunk).

#![cfg(target_os = "macos")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::MlxExecutable;

fn close(got: &[f32], want: &[f32], tol: f32) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
}

fn run(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut exe = MlxExecutable::compile(g);
    exe.run(inputs).into_iter().next().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn conv2d_forward_ref(
    x: &[f32],
    w: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw_dil * (kw - 1) - 1) / sw + 1;
    let mut y = vec![0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0f32;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw_dil;
                                if hi < ph || wi < pw {
                                    continue;
                                }
                                let hi = hi - ph;
                                let wi = wi - pw;
                                if hi >= h || wi >= w_in {
                                    continue;
                                }
                                acc += x[((ni * c_in) + ci) * h * w_in + hi * w_in + wi]
                                    * w[((co * c_in) + ci) * kh * kw + ki * kw + kj];
                            }
                        }
                    }
                    y[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                }
            }
        }
    }
    (y, h_out, w_out)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_input_ref(
    dy: &[f32],
    w: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) -> Vec<f32> {
    let mut dx = vec![0f32; n * c_in * h * w_in];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let upstream = dy[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo];
                    if upstream == 0.0 {
                        continue;
                    }
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sh + ki * dh;
                            let wi = wo * sw + kj * dw_dil;
                            if hi < ph || wi < pw {
                                continue;
                            }
                            let hi = hi - ph;
                            let wi = wi - pw;
                            if hi >= h || wi >= w_in {
                                continue;
                            }
                            for ci in 0..c_in {
                                let dx_idx = ((ni * c_in) + ci) * h * w_in + hi * w_in + wi;
                                let w_idx = ((co * c_in) + ci) * kh * kw + ki * kw + kj;
                                dx[dx_idx] += upstream * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_weight_ref(
    x: &[f32],
    dy: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) -> Vec<f32> {
    let mut dw = vec![0f32; c_out * c_in * kh * kw];
    for co in 0..c_out {
        for ci in 0..c_in {
            for ki in 0..kh {
                for kj in 0..kw {
                    let mut acc = 0f32;
                    for ni in 0..n {
                        for ho in 0..h_out {
                            let hi = ho * sh + ki * dh;
                            if hi < ph {
                                continue;
                            }
                            let hi = hi - ph;
                            if hi >= h {
                                continue;
                            }
                            for wo in 0..w_out {
                                let wi = wo * sw + kj * dw_dil;
                                if wi < pw {
                                    continue;
                                }
                                let wi = wi - pw;
                                if wi >= w_in {
                                    continue;
                                }
                                acc += x[((ni * c_in) + ci) * h * w_in + hi * w_in + wi]
                                    * dy[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo];
                            }
                        }
                    }
                    dw[((co * c_in) + ci) * kh * kw + ki * kw + kj] = acc;
                }
            }
        }
    }
    dw
}

#[allow(clippy::too_many_arguments)]
fn conv2d_forward_ref_grouped(
    x: &[f32],
    w: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
    groups: usize,
) -> (Vec<f32>, usize, usize) {
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w_in + 2 * pw - dw_dil * (kw - 1) - 1) / sw + 1;
    let cig = c_in / groups;
    let cog = c_out / groups;
    let mut y = vec![0f32; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / cog;
            let ci_start = g * cig;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0f32;
                    for ci_off in 0..cig {
                        let ci = ci_start + ci_off;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw_dil;
                                if hi < ph || wi < pw {
                                    continue;
                                }
                                let hi = hi - ph;
                                let wi = wi - pw;
                                if hi >= h || wi >= w_in {
                                    continue;
                                }
                                acc += x[((ni * c_in) + ci) * h * w_in + hi * w_in + wi]
                                    * w[((co * cig) + ci_off) * kh * kw + ki * kw + kj];
                            }
                        }
                    }
                    y[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                }
            }
        }
    }
    (y, h_out, w_out)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_input_ref_grouped(
    dy: &[f32],
    w: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
    groups: usize,
) -> Vec<f32> {
    let cig = c_in / groups;
    let cog = c_out / groups;
    let mut dx = vec![0f32; n * c_in * h * w_in];
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / cog;
            let ci_start = g * cig;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let upstream = dy[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo];
                    if upstream == 0.0 {
                        continue;
                    }
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let hi = ho * sh + ki * dh;
                            let wi = wo * sw + kj * dw_dil;
                            if hi < ph || wi < pw {
                                continue;
                            }
                            let hi = hi - ph;
                            let wi = wi - pw;
                            if hi >= h || wi >= w_in {
                                continue;
                            }
                            for ci_off in 0..cig {
                                let ci = ci_start + ci_off;
                                let dx_idx = ((ni * c_in) + ci) * h * w_in + hi * w_in + wi;
                                let w_idx = ((co * cig) + ci_off) * kh * kw + ki * kw + kj;
                                dx[dx_idx] += upstream * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    dx
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_weight_ref_grouped(
    x: &[f32],
    dy: &[f32],
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
    groups: usize,
) -> Vec<f32> {
    let cig = c_in / groups;
    let cog = c_out / groups;
    let mut dw = vec![0f32; c_out * cig * kh * kw];
    for co in 0..c_out {
        let g = co / cog;
        let ci_start = g * cig;
        for ci_off in 0..cig {
            let ci = ci_start + ci_off;
            for ki in 0..kh {
                for kj in 0..kw {
                    let mut acc = 0f32;
                    for ni in 0..n {
                        for ho in 0..h_out {
                            let hi = ho * sh + ki * dh;
                            if hi < ph {
                                continue;
                            }
                            let hi = hi - ph;
                            if hi >= h {
                                continue;
                            }
                            for wo in 0..w_out {
                                let wi = wo * sw + kj * dw_dil;
                                if wi < pw {
                                    continue;
                                }
                                let wi = wi - pw;
                                if wi >= w_in {
                                    continue;
                                }
                                acc += x[((ni * c_in) + ci) * h * w_in + hi * w_in + wi]
                                    * dy[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo];
                            }
                        }
                    }
                    dw[((co * cig) + ci_off) * kh * kw + ki * kw + kj] = acc;
                }
            }
        }
    }
    dw
}

#[allow(clippy::too_many_arguments)]
fn check_conv_input_grad_grouped(
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
    groups: usize,
) {
    let cig = c_in / groups;
    let xs: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let ws: Vec<f32> = (0..c_out * cig * kh * kw)
        .map(|i| 0.03 * (i as f32) - 0.2)
        .collect();
    let (_y, h_out, w_out) = conv2d_forward_ref_grouped(
        &xs, &ws, n, c_in, h, w_in, c_out, kh, kw, sh, sw, ph, pw, dh, dw_dil, groups,
    );
    let dys: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();

    let mut g = Graph::new("conv_bwd_in_g");
    let dy = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], DType::F32));
    let w = g.input("w", Shape::new(&[c_out, cig, kh, kw], DType::F32));
    let dx = g.conv2d_backward_input(
        dy,
        w,
        Shape::new(&[n, c_in, h, w_in], DType::F32),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![dh, dw_dil],
        groups,
    );
    g.set_outputs(vec![dx]);

    let want = conv2d_backward_input_ref_grouped(
        &dys, &ws, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
        groups,
    );
    let got = run(g, &[("dy", &dys), ("w", &ws)]);
    let max_d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let argmax_d = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .fold(
            (0usize, 0f32),
            |(bi, bd), (i, d)| if d > bd { (i, d) } else { (bi, bd) },
        )
        .0;
    assert!(
        close(&got, &want, 2e-3),
        "Conv2dBackwardInput grouped (g={groups}) @ shape (n={n}, c_in={c_in}, \
         h={h}, w={w_in}, c_out={c_out}, k={kh}x{kw}, s={sh}x{sw}, \
         p={ph}x{pw}, d={dh}x{dw_dil}); max_diff={max_d:.3e} at flat[{argmax_d}]\n\
         got[{argmax_d}..{}]={:?}\nwant[..]={:?}",
        (argmax_d + 8).min(got.len()),
        &got[argmax_d..(argmax_d + 8).min(got.len())],
        &want[argmax_d..(argmax_d + 8).min(want.len())]
    );
}

#[allow(clippy::too_many_arguments)]
fn check_conv_weight_grad_grouped(
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
    groups: usize,
) {
    let cig = c_in / groups;
    let xs: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let ws: Vec<f32> = (0..c_out * cig * kh * kw)
        .map(|i| 0.03 * (i as f32) - 0.2)
        .collect();
    let (_y, h_out, w_out) = conv2d_forward_ref_grouped(
        &xs, &ws, n, c_in, h, w_in, c_out, kh, kw, sh, sw, ph, pw, dh, dw_dil, groups,
    );
    let dys: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();

    let mut g = Graph::new("conv_bwd_w_g");
    let x = g.input("x", Shape::new(&[n, c_in, h, w_in], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], DType::F32));
    let dw = g.conv2d_backward_weight(
        x,
        dy,
        Shape::new(&[c_out, cig, kh, kw], DType::F32),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![dh, dw_dil],
        groups,
    );
    g.set_outputs(vec![dw]);

    let want = conv2d_backward_weight_ref_grouped(
        &xs, &dys, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
        groups,
    );
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    let max_d = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        close(&got, &want, 2e-3),
        "Conv2dBackwardWeight grouped (g={groups}) @ shape (n={n}, c_in={c_in}, \
         h={h}, w={w_in}, c_out={c_out}, k={kh}x{kw}, s={sh}x{sw}, \
         p={ph}x{pw}, d={dh}x{dw_dil}); max_diff={max_d:.3e}"
    );
}

#[test]
fn conv2d_backward_input_groups_2() {
    check_conv_input_grad_grouped(2, 4, 5, 5, 6, 3, 3, 1, 1, 1, 1, 1, 1, 2);
}

#[test]
fn conv2d_backward_input_groups_4_depthwise() {
    // Depthwise: groups == c_in == c_out → cig=1, cog=1.
    check_conv_input_grad_grouped(2, 4, 5, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 4);
}

#[test]
fn conv2d_backward_weight_groups_2() {
    check_conv_weight_grad_grouped(2, 4, 5, 5, 6, 3, 3, 1, 1, 1, 1, 1, 1, 2);
}

#[test]
fn conv2d_backward_weight_groups_4_depthwise() {
    check_conv_weight_grad_grouped(2, 4, 5, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 4);
}

#[test]
fn conv2d_backward_input_groups_3_stride2() {
    check_conv_input_grad_grouped(2, 6, 7, 7, 9, 3, 3, 2, 2, 1, 1, 1, 1, 3);
}

#[test]
fn conv2d_backward_input_groups_2_stride2() {
    check_conv_input_grad_grouped(2, 4, 7, 7, 6, 3, 3, 2, 2, 1, 1, 1, 1, 2);
}

#[test]
fn conv2d_backward_input_groups_2_kernel_1x1() {
    // 1x1 kernel: trivial case to check group axis handling without
    // any spatial dilation interactions.
    check_conv_input_grad_grouped(2, 4, 5, 5, 6, 1, 1, 1, 1, 0, 0, 1, 1, 2);
}

#[test]
fn conv2d_backward_weight_groups_3_stride2() {
    check_conv_weight_grad_grouped(2, 6, 7, 7, 9, 3, 3, 2, 2, 1, 1, 1, 1, 3);
}

#[allow(clippy::too_many_arguments)]
fn check_conv_input_grad(
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let xs: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let ws: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| 0.03 * (i as f32) - 0.2)
        .collect();
    let (_y_ref, h_out, w_out) = conv2d_forward_ref(
        &xs, &ws, n, c_in, h, w_in, c_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
    );
    let dys: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();

    let mut g = Graph::new("conv_bwd_in");
    let dy = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], DType::F32));
    let w = g.input("w", Shape::new(&[c_out, c_in, kh, kw], DType::F32));
    let dx = g.conv2d_backward_input(
        dy,
        w,
        Shape::new(&[n, c_in, h, w_in], DType::F32),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![dh, dw_dil],
        1,
    );
    g.set_outputs(vec![dx]);

    let want = conv2d_backward_input_ref(
        &dys, &ws, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
    );
    let got = run(g, &[("dy", &dys), ("w", &ws)]);
    assert!(
        close(&got, &want, 5e-4),
        "Conv2dBackwardInput @ shape (n={n}, c_in={c_in}, h={h}, w={w_in}, \
         c_out={c_out}, k={kh}x{kw}, s={sh}x{sw}, p={ph}x{pw}, d={dh}x{dw_dil}): \
         got {got:?} want {want:?}"
    );
}

#[allow(clippy::too_many_arguments)]
fn check_conv_weight_grad(
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let xs: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let ws: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| 0.03 * (i as f32) - 0.2)
        .collect();
    let (_y_ref, h_out, w_out) = conv2d_forward_ref(
        &xs, &ws, n, c_in, h, w_in, c_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
    );
    let dys: Vec<f32> = (0..n * c_out * h_out * w_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();

    let mut g = Graph::new("conv_bwd_w");
    let x = g.input("x", Shape::new(&[n, c_in, h, w_in], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], DType::F32));
    let dw = g.conv2d_backward_weight(
        x,
        dy,
        Shape::new(&[c_out, c_in, kh, kw], DType::F32),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![dh, dw_dil],
        1,
    );
    g.set_outputs(vec![dw]);

    let want = conv2d_backward_weight_ref(
        &xs, &dys, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
    );
    let got = run(g, &[("x", &xs), ("dy", &dys)]);
    assert!(
        close(&got, &want, 5e-4),
        "Conv2dBackwardWeight @ shape (n={n}, c_in={c_in}, h={h}, w={w_in}, \
         c_out={c_out}, k={kh}x{kw}, s={sh}x{sw}, p={ph}x{pw}, d={dh}x{dw_dil}): \
         got {got:?} want {want:?}"
    );
}

#[test]
fn conv2d_backward_input_basic_3x3_s1_p0() {
    check_conv_input_grad(2, 3, 5, 5, 4, 3, 3, 1, 1, 0, 0, 1, 1);
}

#[test]
fn conv2d_backward_input_basic_3x3_s1_p1() {
    check_conv_input_grad(1, 2, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1);
}

#[test]
fn conv2d_backward_input_stride2_padding1() {
    check_conv_input_grad(2, 2, 7, 7, 3, 3, 3, 2, 2, 1, 1, 1, 1);
}

#[test]
fn conv2d_backward_input_dilation_2() {
    check_conv_input_grad(1, 2, 7, 7, 2, 3, 3, 1, 1, 2, 2, 2, 2);
}

#[test]
fn conv2d_backward_input_kernel_1x1() {
    check_conv_input_grad(2, 4, 4, 4, 5, 1, 1, 1, 1, 0, 0, 1, 1);
}

#[test]
fn conv2d_backward_weight_basic_3x3_s1_p0() {
    check_conv_weight_grad(2, 3, 5, 5, 4, 3, 3, 1, 1, 0, 0, 1, 1);
}

#[test]
fn conv2d_backward_weight_basic_3x3_s1_p1() {
    check_conv_weight_grad(1, 2, 6, 6, 3, 3, 3, 1, 1, 1, 1, 1, 1);
}

#[test]
fn conv2d_backward_weight_stride2_padding1() {
    check_conv_weight_grad(2, 2, 7, 7, 3, 3, 3, 2, 2, 1, 1, 1, 1);
}

#[test]
fn conv2d_backward_weight_dilation_2() {
    check_conv_weight_grad(1, 2, 7, 7, 2, 3, 3, 1, 1, 2, 2, 2, 2);
}

#[test]
fn conv2d_backward_weight_kernel_1x1() {
    check_conv_weight_grad(2, 4, 4, 4, 5, 1, 1, 1, 1, 0, 0, 1, 1);
}

#[test]
fn end_to_end_grad_through_conv_layer() {
    // Forward: x → conv2d → relu → mean → loss
    // Backward graph runs on MLX; param grads should match a CPU-side
    // hand-computed reference.
    let n = 2usize;
    let c_in = 2usize;
    let h = 4usize;
    let w_in = 4usize;
    let c_out = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let sh = 1usize;
    let sw = 1usize;
    let ph = 1usize;
    let pw = 1usize;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut fwd = Graph::new("conv_mlp");
    let x = fwd.input("x", Shape::new(&[n, c_in, h, w_in], DType::F32));
    let w = fwd.param("w", Shape::new(&[c_out, c_in, kh, kw], DType::F32));
    let y = fwd.add_node(
        Op::Conv {
            kernel_size: vec![kh, kw],
            stride: vec![sh, sw],
            padding: vec![ph, pw],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[n, c_out, h_out, w_out], DType::F32),
    );
    let a = fwd.activation(
        Activation::Relu,
        y,
        Shape::new(&[n, c_out, h_out, w_out], DType::F32),
    );
    let loss = fwd.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Mean,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![a],
        Shape::new(&[], DType::F32),
    );
    fwd.set_outputs(vec![loss]);

    let bwd = rlx_opt::autodiff::grad_with_loss(&fwd, &[w]);

    let xs: Vec<f32> = (0..n * c_in * h * w_in)
        .map(|i| 0.1 * (i as f32) - 0.3)
        .collect();
    let ws: Vec<f32> = (0..c_out * c_in * kh * kw)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let d_out: Vec<f32> = vec![1.0];

    let mut exe = MlxExecutable::compile(bwd);
    exe.set_param("w", &ws);
    let outs = exe.run(&[("x", &xs), ("d_output", &d_out)]);
    assert_eq!(outs.len(), 2, "expected [loss, dW]");

    // Reference: forward, then backward via the rlx-cpu formulas.
    let (yv, _, _) = conv2d_forward_ref(
        &xs, &ws, n, c_in, h, w_in, c_out, kh, kw, sh, sw, ph, pw, 1, 1,
    );
    let av: Vec<f32> = yv.iter().map(|&v| v.max(0.0)).collect();
    let total = (n * c_out * h_out * w_out) as f32;
    let loss_ref: f32 = av.iter().sum::<f32>() / total;

    // d/d a_i = 1/total · d_out
    let scale = d_out[0] / total;
    let da: Vec<f32> = av.iter().map(|_| scale).collect();
    // Through ReLU: dy_i = if y_i > 0 then da_i else 0
    let dy: Vec<f32> = yv
        .iter()
        .zip(da.iter())
        .map(|(&y, &d)| if y > 0.0 { d } else { 0.0 })
        .collect();
    let want_dw = conv2d_backward_weight_ref(
        &xs, &dy, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, 1, 1,
    );

    assert!(
        close(&outs[0], &[loss_ref], 1e-4),
        "loss: got {:?} want {loss_ref}",
        outs[0]
    );
    assert!(
        close(&outs[1], &want_dw, 5e-4),
        "dW: got {:?} want {want_dw:?}",
        outs[1]
    );
}
