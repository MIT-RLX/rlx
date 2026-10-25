// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `conv2d_sg` (implicit-GEMM forward conv) and the direct `conv2d` against an
//! independent f64 reference, over strides, padding, dilation, 1×1 kernels,
//! batches and tiles that the 64×64 blocking leaves ragged.

#![cfg(target_os = "macos")]

use rlx_metal::kernels::kernels;
use rlx_metal::mtl::{Device, MTLResourceOptions, MTLSize};

#[derive(Clone, Copy, Debug)]
struct Conv {
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    k: usize,
    s: usize,
    p: usize,
    d: usize,
}

impl Conv {
    fn out(&self) -> (usize, usize) {
        let span = self.d * (self.k - 1) + 1;
        (
            (self.h + 2 * self.p - span) / self.s + 1,
            (self.w + 2 * self.p - span) / self.s + 1,
        )
    }
}

/// Exact output plus the sum of term magnitudes (the f32 error budget).
fn reference(c: Conv, x: &[f32], wt: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let (ho, wo) = c.out();
    let mut out = vec![0f64; c.n * c.c_out * ho * wo];
    let mut mag = vec![0f64; out.len()];
    for n in 0..c.n {
        for co in 0..c.c_out {
            for y in 0..ho {
                for xo in 0..wo {
                    let (mut s, mut m) = (0f64, 0f64);
                    for ci in 0..c.c_in {
                        for ki in 0..c.k {
                            for kj in 0..c.k {
                                let hi = (y * c.s + ki * c.d) as isize - c.p as isize;
                                let wi = (xo * c.s + kj * c.d) as isize - c.p as isize;
                                if hi < 0 || wi < 0 || hi >= c.h as isize || wi >= c.w as isize {
                                    continue;
                                }
                                let a = f64::from(
                                    x[((n * c.c_in + ci) * c.h + hi as usize) * c.w + wi as usize],
                                );
                                let b = f64::from(wt[((co * c.c_in + ci) * c.k + ki) * c.k + kj]);
                                s += a * b;
                                m += (a * b).abs();
                            }
                        }
                    }
                    let i = ((n * c.c_out + co) * ho + y) * wo + xo;
                    out[i] = s;
                    mag[i] = m;
                }
            }
        }
    }
    (out, mag)
}

/// Runs `conv2d_sg` (`sg`) or `conv2d` `reps` times; output and seconds per run.
fn run(c: Conv, x: &[f32], wt: &[f32], sg: bool, reps: usize) -> (Vec<f32>, f64) {
    let dev = Device::system_default().expect("no Metal device");
    let q = dev.new_command_queue();
    let (ho, wo) = c.out();
    let out_len = c.n * c.c_out * ho * wo;
    let buf = |v: &[f32]| {
        dev.new_buffer_with_data(
            v.as_ptr() as *const _,
            (v.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let (bx, bw) = (buf(x), buf(wt));
    let bo = dev.new_buffer((out_len * 4) as u64, MTLResourceOptions::StorageModeShared);
    let nch = [c.n as u32, c.c_in as u32, c.h as u32, c.w as u32];
    let od = [c.c_out as u32, ho as u32, wo as u32, 1u32];
    let ks = [c.k as u32, c.k as u32, c.s as u32, c.s as u32];
    let pd = [c.p as u32, c.p as u32, c.d as u32, c.d as u32];
    let k = kernels();
    let cb = q.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    for _ in 0..reps {
        enc.set_compute_pipeline_state(if sg { &k.conv2d_sg } else { &k.conv2d });
        enc.set_buffer(0, Some(&bx), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bo), 0);
        enc.set_bytes(3, 16, nch.as_ptr() as *const _);
        enc.set_bytes(4, 16, od.as_ptr() as *const _);
        enc.set_bytes(5, 16, ks.as_ptr() as *const _);
        enc.set_bytes(6, 16, pd.as_ptr() as *const _);
        if sg {
            enc.dispatch_thread_groups(
                MTLSize {
                    width: ((ho * wo) as u64).div_ceil(64),
                    height: (c.c_out as u64).div_ceil(64),
                    depth: c.n as u64,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            enc.dispatch_threads(
                MTLSize {
                    width: wo as u64,
                    height: ho as u64,
                    depth: (c.n * c.c_out) as u64,
                },
                MTLSize {
                    width: 8.min(wo as u64),
                    height: 8.min(ho as u64),
                    depth: 1,
                },
            );
        }
    }
    enc.end_encoding();
    let t0 = std::time::Instant::now();
    cb.commit();
    cb.wait_until_completed();
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    let mut out = vec![0f32; out_len];
    unsafe {
        std::ptr::copy_nonoverlapping(bo.contents() as *const f32, out.as_mut_ptr(), out_len);
    }
    (out, secs)
}

fn data(c: Conv) -> (Vec<f32>, Vec<f32>) {
    let x = (0..c.n * c.c_in * c.h * c.w)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let w = (0..c.c_out * c.c_in * c.k * c.k)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) / 200.0)
        .collect();
    (x, w)
}

fn parity(c: Conv) {
    let (x, wt) = data(c);
    let (want, mag) = reference(c, &x, &wt);
    let k_tot = (c.c_in * c.k * c.k) as f64;
    for sg in [true, false] {
        let got = run(c, &x, &wt, sg, 1).0;
        for (i, (&g, (&r, &m))) in got.iter().zip(want.iter().zip(&mag)).enumerate() {
            let tol = 8.0 * k_tot.sqrt() * f64::from(f32::EPSILON) * m.max(1.0);
            assert!(
                (f64::from(g) - r).abs() <= tol,
                "{} off reference at {i} for {c:?}: got {g} want {r} tol {tol}",
                if sg { "conv2d_sg" } else { "conv2d" }
            );
        }
    }
}

#[test]
fn conv2d_sg_3x3_same() {
    parity(Conv {
        n: 1,
        c_in: 32,
        h: 20,
        w: 20,
        c_out: 64,
        k: 3,
        s: 1,
        p: 1,
        d: 1,
    });
}

#[test]
fn conv2d_sg_ragged_tiles_and_batch() {
    // c_out and pixels both leave partial 64-wide tiles; k_tot = 27 < 32.
    parity(Conv {
        n: 2,
        c_in: 3,
        h: 17,
        w: 19,
        c_out: 20,
        k: 3,
        s: 1,
        p: 1,
        d: 1,
    });
}

#[test]
fn conv2d_sg_stride_dilation() {
    parity(Conv {
        n: 1,
        c_in: 8,
        h: 33,
        w: 30,
        c_out: 24,
        k: 3,
        s: 2,
        p: 1,
        d: 1,
    });
    parity(Conv {
        n: 1,
        c_in: 8,
        h: 24,
        w: 24,
        c_out: 16,
        k: 3,
        s: 1,
        p: 2,
        d: 2,
    });
}

#[test]
fn conv2d_sg_pointwise() {
    parity(Conv {
        n: 1,
        c_in: 96,
        h: 10,
        w: 12,
        c_out: 70,
        k: 1,
        s: 1,
        p: 0,
        d: 1,
    });
}

/// `cargo test -p rlx-metal --release --test conv2d_sg_parity -- --ignored --nocapture`
#[test]
#[ignore]
fn conv2d_sg_speed() {
    for c in [
        Conv {
            n: 1,
            c_in: 512,
            h: 20,
            w: 20,
            c_out: 512,
            k: 3,
            s: 1,
            p: 1,
            d: 1,
        },
        Conv {
            n: 1,
            c_in: 256,
            h: 80,
            w: 80,
            c_out: 256,
            k: 3,
            s: 1,
            p: 1,
            d: 1,
        },
        Conv {
            n: 1,
            c_in: 128,
            h: 160,
            w: 160,
            c_out: 128,
            k: 3,
            s: 1,
            p: 1,
            d: 1,
        },
        Conv {
            n: 1,
            c_in: 64,
            h: 160,
            w: 160,
            c_out: 32,
            k: 3,
            s: 1,
            p: 1,
            d: 1,
        },
    ] {
        let (x, wt) = data(c);
        let (ho, wo) = c.out();
        let flops = 2.0 * (c.c_out * ho * wo * c.c_in * c.k * c.k) as f64;
        let mut line = format!("{c:?}:");
        for (name, sg) in [("direct", false), ("sg", true)] {
            run(c, &x, &wt, sg, 2);
            let (_, t) = run(c, &x, &wt, sg, 5);
            line += &format!(
                "  {name} {:8.2} ms {:5.2} TFLOPS",
                t * 1e3,
                flops / t / 1e12
            );
        }
        println!("{line}");
    }
}
