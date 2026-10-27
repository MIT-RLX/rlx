// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Fft`'s algorithm choice (`FftAlgo`) on CPU: the pocketfft path reproduces
//! `rlx_pocketfft` (itself bit-identical to `scipy.fft`) bit for bit, in every
//! layout, across a batch of rows, through the parallel row loop and through the
//! VJP; `Native` stays the radix/Bluestein kernels. GPUs run the elementwise
//! lowering (`rlx_fusion::lower_pocketfft`) and must reproduce the CPU kernel's bits.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, FftAlgo, FftContraction, FftNorm, FftTwiddles, Graph, Op, Shape};
use rlx_pocketfft::engine::{self, Cx};
use rlx_runtime::{Device, Session};

mod common;

fn sig(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// `outer` rows of `n` complex values (2N block) through `Op::Fft` on CPU.
fn run_f64(
    rows: &[Vec<[f64; 2]>],
    inverse: bool,
    norm: FftNorm,
    algo: FftAlgo,
) -> Vec<Vec<[f64; 2]>> {
    let n = rows[0].len();
    let mut bytes = Vec::new();
    for r in rows {
        for c in r {
            bytes.extend_from_slice(&c[0].to_le_bytes());
        }
        for c in r {
            bytes.extend_from_slice(&c[1].to_le_bytes());
        }
    }
    let mut g = Graph::new("fft_algo");
    let x = g.input("x", Shape::new(&[rows.len(), 2 * n], DType::F64));
    let y = g.fft_algo(x, inverse, norm, algo);
    g.set_outputs(vec![y]);
    let out = Session::new(Device::Cpu)
        .compile(g)
        .run_typed(&[("x", &bytes, DType::F64)]);
    let v: Vec<f64> = out[0]
        .0
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    v.chunks(2 * n)
        .map(|r| (0..n).map(|k| [r[k], r[n + k]]).collect())
        .collect()
}

fn bits(v: &[[f64; 2]]) -> Vec<u64> {
    v.iter()
        .flat_map(|c| [c[0].to_bits(), c[1].to_bits()])
        .collect()
}

#[test]
fn scipy_algo_matches_scipy_fft_bits() {
    let _gpu = common::serialize_gpu();
    for n in [
        1usize, 2, 5, 7, 11, 12, 23, 37, 64, 100, 127, 210, 509, 1000, 1009, 4099,
    ] {
        let rows: Vec<Vec<[f64; 2]>> = (0..3)
            .map(|r| {
                sig(n, (r * 1000 + n) as u64)
                    .into_iter()
                    .zip(sig(n, (r * 77 + n) as u64))
                    .map(|(a, b)| [a, b])
                    .collect()
            })
            .collect();
        let fwd = run_f64(&rows, false, FftNorm::Backward, FftAlgo::SCIPY);
        // scipy's ifft (norm=None) scales by 1/n: RLX's `Forward` norm
        let inv = run_f64(&rows, true, FftNorm::Forward, FftAlgo::SCIPY);
        for (r, row) in rows.iter().enumerate() {
            let want = rlx_pocketfft::fft(row, None, rlx_pocketfft::Norm::Backward);
            assert_eq!(bits(&fwd[r]), bits(&want), "fft n={n} row {r}");
            let want = rlx_pocketfft::ifft(row, None, rlx_pocketfft::Norm::Backward);
            assert_eq!(bits(&inv[r]), bits(&want), "ifft n={n} row {r}");
        }
    }
}

#[test]
fn parallel_rows_and_no_fma_option() {
    let _gpu = common::serialize_gpu();
    // 64 rows × 1024 crosses the parallel-row threshold.
    let (n, outer) = (1024usize, 64usize);
    let rows: Vec<Vec<[f64; 2]>> = (0..outer)
        .map(|r| {
            sig(2 * n, r as u64)
                .chunks(2)
                .map(|c| [c[0], c[1]])
                .collect()
        })
        .collect();
    for algo in [FftAlgo::SCIPY, FftAlgo::POCKETFFT_NO_FMA] {
        let FftAlgo::PocketFft {
            contraction,
            twiddles,
        } = algo
        else {
            unreachable!()
        };
        let opt = engine::Options {
            contraction: match contraction {
                FftContraction::Platform => engine::Contraction::Platform,
                FftContraction::Clang => engine::Contraction::Clang,
                FftContraction::None => engine::Contraction::None,
            },
            twiddles: if twiddles == FftTwiddles::Separate {
                engine::Twiddles::Separate
            } else {
                engine::Twiddles::Platform
            },
        };
        let got = run_f64(&rows, false, FftNorm::Backward, algo);
        for (r, row) in rows.iter().enumerate() {
            let x: Vec<Cx<f64>> = row.iter().map(|c| Cx::new(c[0], c[1])).collect();
            let want: Vec<[f64; 2]> = engine::c2c(&x, true, 1.0, opt)
                .iter()
                .map(|c| [c.r, c.i])
                .collect();
            assert_eq!(bits(&got[r]), bits(&want), "{algo:?} row {r}");
        }
    }
    // the options are live: Native, fused and unfused pocketfft round differently.
    // `Clang` contraction fuses on every target; `SCIPY` fuses only where SciPy's
    // own compiler does (Apple), so on Linux it IS the unfused build.
    let native = run_f64(&rows[..1], false, FftNorm::Backward, FftAlgo::Native);
    let always_fused = FftAlgo::PocketFft {
        contraction: FftContraction::Clang,
        twiddles: FftTwiddles::Platform,
    };
    let fused = run_f64(&rows[..1], false, FftNorm::Backward, always_fused);
    let unfused = run_f64(
        &rows[..1],
        false,
        FftNorm::Backward,
        FftAlgo::POCKETFFT_NO_FMA,
    );
    assert_ne!(bits(&native[0]), bits(&fused[0]));
    assert_ne!(bits(&fused[0]), bits(&unfused[0]));
}

#[test]
fn f32_and_c64_layouts() {
    let _gpu = common::serialize_gpu();
    let n = 360usize;
    let z: Vec<Cx<f32>> = sig(2 * n, 9)
        .chunks(2)
        .map(|c| Cx::new(c[0] as f32, c[1] as f32))
        .collect();
    let want = engine::c2c(&z, true, 1.0f32, engine::Options::default());
    for dtype in [DType::F32, DType::C64] {
        let mut bytes = Vec::new();
        if dtype == DType::F32 {
            z.iter()
                .for_each(|c| bytes.extend_from_slice(&c.r.to_le_bytes()));
            z.iter()
                .for_each(|c| bytes.extend_from_slice(&c.i.to_le_bytes()));
        } else {
            z.iter().for_each(|c| {
                bytes.extend_from_slice(&c.r.to_le_bytes());
                bytes.extend_from_slice(&c.i.to_le_bytes());
            });
        }
        let shape = if dtype == DType::F32 {
            Shape::new(&[2 * n], dtype)
        } else {
            Shape::new(&[n], dtype)
        };
        let mut g = Graph::new("fft_algo32");
        let x = g.input("x", shape);
        let y = g.fft_algo(x, false, FftNorm::Backward, FftAlgo::SCIPY);
        g.set_outputs(vec![y]);
        let out = Session::new(Device::Cpu)
            .compile(g)
            .run_typed(&[("x", &bytes, dtype)]);
        let v: Vec<f32> = out[0]
            .0
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        for (k, w) in want.iter().enumerate() {
            let (r, i) = if dtype == DType::F32 {
                (v[k], v[n + k])
            } else {
                (v[2 * k], v[2 * k + 1])
            };
            assert_eq!(
                (r.to_bits(), i.to_bits()),
                (w.r.to_bits(), w.i.to_bits()),
                "{dtype:?} [{k}]"
            );
        }
    }
}

#[test]
fn vjp_keeps_the_algorithm_and_every_device_takes_it() {
    let mut g = Graph::new("fft_algo_vjp");
    let x = g.input("x", Shape::new(&[2 * 12], DType::F64));
    let y = g.fft_algo(x, false, FftNorm::Backward, FftAlgo::SCIPY);
    let loss = g.sum(y, vec![0], false);
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::autodiff::grad_with_loss(&g, &[x]);
    let ffts: Vec<FftAlgo> = bwd
        .nodes()
        .iter()
        .filter_map(|n| {
            if let Op::Fft { algo, .. } = n.op {
                Some(algo)
            } else {
                None
            }
        })
        .collect();
    assert!(
        ffts.len() >= 2 && ffts.iter().all(|a| *a == FftAlgo::SCIPY),
        "{ffts:?}"
    );
    let op = Op::Fft {
        inverse: false,
        norm: FftNorm::Backward,
        algo: FftAlgo::SCIPY,
    };
    // CPU runs it natively; GPUs through `lower_pocketfft`
    assert!(rlx_runtime::device_ext::supports(Device::Cpu, &op));
}

/// Lengths that exercise every pass: radix 2/3/4/5/7/8/11, generic factors (13,
/// 17, 23, 31, 37 — below 50 pocketfft never picks Bluestein), mixed, and
/// Bluestein (1009, 4099).
const LOWER_LENGTHS: [usize; 18] = [
    1, 2, 3, 8, 12, 13, 17, 23, 31, 37, 60, 77, 121, 169, 360, 1000, 1009, 4099,
];

fn block_bytes(rows: &[Vec<[f64; 2]>], dt: DType) -> Vec<u8> {
    let mut out = Vec::new();
    for r in rows {
        for part in 0..2 {
            for c in r {
                if dt == DType::F32 {
                    out.extend_from_slice(&(c[part] as f32).to_le_bytes());
                } else {
                    out.extend_from_slice(&c[part].to_le_bytes());
                }
            }
        }
    }
    out
}

fn fft_graph(
    rows: usize,
    n: usize,
    dt: DType,
    inverse: bool,
    norm: FftNorm,
    algo: FftAlgo,
) -> Graph {
    let mut g = Graph::new("fft_lower");
    let x = g.input("x", Shape::new(&[rows, 2 * n], dt));
    let y = g.fft_algo(x, inverse, norm, algo);
    g.set_outputs(vec![y]);
    g
}

fn run_bytes(dev: Device, g: Graph, x: &[u8], dt: DType) -> Vec<u8> {
    Session::new(dev).compile(g).run_typed(&[("x", x, dt)])[0]
        .0
        .clone()
}

fn cases() -> Vec<(DType, bool, FftNorm, FftAlgo)> {
    vec![
        (DType::F64, false, FftNorm::Backward, FftAlgo::SCIPY),
        (DType::F64, true, FftNorm::Forward, FftAlgo::SCIPY),
        (DType::F32, false, FftNorm::Backward, FftAlgo::SCIPY),
        (DType::F32, true, FftNorm::Ortho, FftAlgo::SCIPY),
        (
            DType::F32,
            false,
            FftNorm::Backward,
            FftAlgo::POCKETFFT_NO_FMA,
        ),
        (
            DType::F64,
            true,
            FftNorm::Forward,
            FftAlgo::POCKETFFT_NO_FMA,
        ),
    ]
}

/// The elementwise lowering (what GPUs run) replays the CPU kernel bit for bit.
#[test]
fn lowered_pocketfft_matches_the_cpu_kernel() {
    let _gpu = common::serialize_gpu();
    let mut bad = vec![];
    for n in LOWER_LENGTHS {
        let rows: Vec<Vec<[f64; 2]>> = (0..2)
            .map(|r| {
                sig(2 * n, (r * 31 + n) as u64)
                    .chunks(2)
                    .map(|c| [c[0], c[1]])
                    .collect()
            })
            .collect();
        for (dt, inverse, norm, algo) in cases() {
            let x = block_bytes(&rows, dt);
            let native = run_bytes(
                Device::Cpu,
                fft_graph(2, n, dt, inverse, norm, algo),
                &x,
                dt,
            );
            let lowered = rlx_fusion::rewriter::apply_match_rewrite(
                &rlx_fusion::lower_pocketfft::LowerPocketFft,
                fft_graph(2, n, dt, inverse, norm, algo),
            )
            .graph;
            assert!(
                lowered
                    .nodes()
                    .iter()
                    .all(|nd| nd.op.kind() != rlx_ir::OpKind::FftPocketFft),
                "n={n} not lowered"
            );
            let got = run_bytes(Device::Cpu, lowered, &x, dt);
            if got != native {
                bad.push(format!("n={n} {dt:?} inv={inverse} {algo:?}"));
            }
        }
    }
    assert!(bad.is_empty(), "{} cases differ: {bad:?}", bad.len());
}

/// Every GPU backend in the build reproduces the CPU pocketfft kernel (f32; the
/// legalizer lowers the node there).
#[test]
fn gpu_pocketfft_matches_the_cpu_kernel() {
    let _gpu = common::serialize_gpu();
    // every backend compiled into this build and usable on this host
    let devices: Vec<Device> = rlx_runtime::device_ext::available_devices()
        .into_iter()
        .filter(|d| *d != Device::Cpu)
        .collect();
    let devices: Vec<Device> = devices
        .into_iter()
        .filter(|d| rlx_runtime::device_ext::is_available(*d))
        .collect();
    eprintln!("devices: {devices:?}");
    let mut bad = vec![];
    let mut hosted: std::collections::BTreeMap<String, usize> = Default::default();
    for n in LOWER_LENGTHS {
        let rows: Vec<Vec<[f64; 2]>> = (0..3)
            .map(|r| {
                sig(2 * n, (r * 17 + n) as u64)
                    .chunks(2)
                    .map(|c| [c[0], c[1]])
                    .collect()
            })
            .collect();
        for (dt, inverse, norm, algo) in cases().into_iter().filter(|c| c.0 == DType::F32) {
            let x = block_bytes(&rows, dt);
            let want = run_bytes(
                Device::Cpu,
                fft_graph(3, n, dt, inverse, norm, algo),
                &x,
                dt,
            );
            for &d in &devices {
                let mut c = Session::new(d).compile(fft_graph(3, n, dt, inverse, norm, algo));
                // ran on the device, not a silent host fallback
                assert_eq!(c.device(), d);
                // Reported, not failed: a backend may host steps by policy (wgpu on
                // Vulkan/DX12 hosts structural ops); what it returns must still match.
                if let Some(summary) = c.host_execution_summary() {
                    *hosted.entry(format!("{d:?}")).or_insert(0) += 1;
                    if n <= 2 {
                        eprintln!("{d:?} n={n}: {summary}");
                    }
                }
                let got = c.run_typed(&[("x", &x, dt)])[0].0.clone();
                if got != want {
                    let diff = got
                        .chunks(4)
                        .zip(want.chunks(4))
                        .filter(|(a, b)| a != b)
                        .count();
                    bad.push(format!(
                        "{d:?} n={n} inv={inverse} {algo:?}: {diff}/{}",
                        want.len() / 4
                    ));
                }
            }
        }
    }
    eprintln!("graphs with host-executed steps, per device: {hosted:?}");
    assert!(bad.is_empty(), "{} cases differ: {bad:?}", bad.len());
}
