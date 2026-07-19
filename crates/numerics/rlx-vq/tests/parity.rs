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

//! The fused VQ op matches the rlx-ir composition, on CPU and Metal.

use rlx_ir::ops::vq::VqMetric;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut b = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: b },
        vec![],
        Shape::new(dims, DType::F32),
    )
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn run(dev: Device, build: &dyn Fn(&mut Graph) -> Vec<NodeId>) -> Vec<Vec<f32>> {
    let mut g = Graph::new("t");
    let outs = build(&mut g);
    g.set_outputs(outs);
    Session::new(dev)
        .compile(g)
        .run_typed(&[])
        .iter()
        .map(|o| f32s(&o.0))
        .collect()
}

/// Well-separated codebook so nearest-code is unambiguous (no f32 tie-flips).
fn separable(n: usize, d: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cb = vec![0f32; k * d];
    for j in 0..k {
        cb[j * d + (j % d)] = 10.0 + j as f32;
    }
    let mut x = vec![0f32; n * d];
    for i in 0..n {
        let j = i % k;
        x[i * d + (j % d)] = 10.0 + j as f32 + 0.02;
    }
    (x, cb)
}

#[test]
fn fused_matches_composition_cpu() {
    rlx_vq::register();
    let (n, d, k) = (48usize, 16usize, 32usize);
    let (x, cb) = separable(n, d, k);

    let fused = run(Device::Cpu, &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, q) = rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Cpu);
        vec![idx, q]
    });
    let comp = run(Device::Cpu, &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, q) = g.vector_quantize(xn, cbn, VqMetric::L2);
        vec![idx, q]
    });

    assert_eq!(fused[0], comp[0], "indices differ from composition");
    assert_eq!(fused[1], comp[1], "gathered codes differ from composition");
    // sanity: nearest code of input i is i%k
    for i in 0..n {
        assert_eq!(fused[0][i] as usize, i % k);
    }
}

#[test]
fn fused_cosine_selects_by_direction() {
    rlx_vq::register();
    // Codebook rows point along distinct axes; each input points along one axis
    // (any magnitude) → cosine picks that axis regardless of scale.
    let (d, k) = (4usize, 4usize);
    let mut cb = vec![0f32; k * d];
    for j in 0..k {
        cb[j * d + j] = 1.0;
    }
    let x = vec![
        0.0, 5.0, 0.0, 0.0, // axis 1
        0.0, 0.0, 0.0, 9.0, // axis 3
        3.0, 0.0, 0.0, 0.0, // axis 0
    ];
    let n = 3;
    let out = run(Device::Cpu, &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, _q) =
            rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::Cosine, rlx_vq::Target::Cpu);
        vec![idx]
    });
    assert_eq!(out[0], vec![1.0, 3.0, 0.0]);
}

fn seeded(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let z = (i as u32).wrapping_mul(2654435761).wrapping_add(salt);
            ((z >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        })
        .collect()
}

#[test]
fn fused_residual_vq_matches_composition() {
    rlx_vq::register();
    let (n, d, k, levels) = (32usize, 8usize, 16usize, 3usize);
    let x = seeded(n * d, 7);
    let cbs: Vec<Vec<f32>> = (0..levels).map(|l| seeded(k * d, 100 + l as u32)).collect();

    let build_fused: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn: Vec<NodeId> = cbs.iter().map(|c| const_f32(g, c, &[k, d])).collect();
        let (mut idxs, recon) =
            rlx_vq::residual_vq(g, xn, &cbn, rlx_vq::Metric::L2, rlx_vq::Target::Cpu);
        idxs.push(recon);
        idxs
    };
    let build_comp: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn: Vec<NodeId> = cbs.iter().map(|c| const_f32(g, c, &[k, d])).collect();
        let (mut idxs, recon) = g.residual_vq(xn, &cbn, VqMetric::L2);
        idxs.push(recon);
        idxs
    };
    let fused = run(Device::Cpu, build_fused);
    let comp = run(Device::Cpu, build_comp);

    // Per-level indices agree (same proxy); reconstruction matches closely.
    for l in 0..levels {
        assert_eq!(fused[l], comp[l], "level {l} indices differ");
    }
    let (fr, cr) = (fused.last().unwrap(), comp.last().unwrap());
    assert_eq!(fr.len(), n * d, "reconstruction shape");
    for (a, b) in fr.iter().zip(cr.iter()) {
        assert!((a - b).abs() < 1e-4, "recon {a} vs {b}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn fused_matches_cpu_on_metal() {
    rlx_vq::register();
    let (n, d, k) = (48usize, 16usize, 32usize);
    let (x, cb) = separable(n, d, k);
    let build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, q) = rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Cpu);
        vec![idx, q]
    };
    let cpu = run(Device::Cpu, build);
    let metal = run(Device::Metal, build);
    assert_eq!(cpu[0], metal[0], "Metal indices differ from CPU");
    assert_eq!(cpu[1], metal[1], "Metal codes differ from CPU");
}

/// Data point: does the fused Metal host-callback (D2H → CPU loop → H2D) beat
/// the on-GPU matmul+argmin composition? Prints timings (run with --nocapture).
/// `Target::Gpu` lowers to the plain matmul+argmin+gather composition — ops
/// every backend supports — so it is portable to *all* GPU backends with no
/// per-backend kernel. Verify identical results on each backend compiled in.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn gpu_target_portable_across_backends() {
    rlx_vq::register();
    let (n, d, k) = (48usize, 16usize, 32usize);
    let (x, cb) = separable(n, d, k);
    let build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, q) = rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Gpu);
        vec![idx, q]
    };
    let cpu = run(Device::Cpu, build);
    for dev in [Device::Metal, Device::Gpu, Device::Mlx] {
        if !rlx_runtime::is_available(dev) {
            eprintln!("skip {dev:?}: unavailable");
            continue;
        }
        let got = run(dev, build);
        assert_eq!(
            got[0], cpu[0],
            "{dev:?}: Target::Gpu indices differ from CPU"
        );
        assert_eq!(got[1], cpu[1], "{dev:?}: Target::Gpu codes differ from CPU");
    }
}

/// The whole point: the `Target::Gpu` lowering on Metal must be faster than the
/// `Target::Cpu` lowering on CPU (and produce the same indices).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_gpu_beats_cpu() {
    use std::time::Instant;
    rlx_vq::register();
    let (n, d, k) = (1024usize, 128usize, 4096usize);
    let (x, cb) = separable(n, d, k);
    let iters = 20;

    let cpu_build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, _q) =
            rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Cpu);
        vec![idx]
    };
    let gpu_build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let xn = const_f32(g, &x, &[n, d]);
        let cbn = const_f32(g, &cb, &[k, d]);
        let (idx, _q) =
            rlx_vq::vector_quantize(g, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Gpu);
        vec![idx]
    };

    // Same result on separable data.
    let cpu_idx = run(Device::Cpu, cpu_build);
    let gpu_idx = run(Device::Metal, gpu_build);
    assert_eq!(
        cpu_idx[0], gpu_idx[0],
        "GPU lowering must match CPU indices"
    );

    let time = |dev: Device, build: &dyn Fn(&mut Graph) -> Vec<NodeId>| -> u128 {
        let mut g = Graph::new("t");
        let outs = build(&mut g);
        g.set_outputs(outs);
        let mut c = Session::new(dev).compile(g);
        c.run_typed(&[]);
        let t = Instant::now();
        for _ in 0..iters {
            c.run_typed(&[]);
        }
        t.elapsed().as_nanos() / iters
    };
    let cpu_ns = time(Device::Cpu, cpu_build);
    let gpu_ns = time(Device::Metal, gpu_build);
    println!(
        "N={n} K={k}: CPU(fused)={cpu_ns}ns  Metal(Target::Gpu)={gpu_ns}ns  speedup={:.2}x",
        cpu_ns as f64 / gpu_ns as f64
    );
    assert!(
        gpu_ns < cpu_ns,
        "Metal must be faster than CPU (got {gpu_ns} vs {cpu_ns})"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_fused_vs_composition_timing() {
    use std::time::Instant;
    rlx_vq::register();
    let iters = 20;
    // Sweep batch size N: the fused on-GPU kernel avoids the [N,K] distance
    // matrix that the matmul+argmin composition must write and re-read, so it
    // wins once that bandwidth dominates (large N).
    for &(n, d, k) in &[
        (256usize, 128usize, 4096usize),
        (1024, 128, 4096),
        (4096, 128, 4096),
        (8192, 128, 8192),
    ] {
        let (x, cb) = separable(n, d, k);

        let mut gf = Graph::new("f");
        let xn = const_f32(&mut gf, &x, &[n, d]);
        let cbn = const_f32(&mut gf, &cb, &[k, d]);
        let (idx, _q) =
            rlx_vq::vector_quantize(&mut gf, xn, cbn, rlx_vq::Metric::L2, rlx_vq::Target::Cpu);
        gf.set_outputs(vec![idx]);
        let mut cf = Session::new(Device::Metal).compile(gf);

        let mut gc = Graph::new("c");
        let xn = const_f32(&mut gc, &x, &[n, d]);
        let cbn = const_f32(&mut gc, &cb, &[k, d]);
        let (idx, _q) = gc.vector_quantize(xn, cbn, VqMetric::L2);
        gc.set_outputs(vec![idx]);
        let mut cc = Session::new(Device::Metal).compile(gc);

        cf.run_typed(&[]);
        cc.run_typed(&[]);
        let t0 = Instant::now();
        for _ in 0..iters {
            cf.run_typed(&[]);
        }
        let fused_ns = t0.elapsed().as_nanos() / iters;
        let t1 = Instant::now();
        for _ in 0..iters {
            cc.run_typed(&[]);
        }
        let comp_ns = t1.elapsed().as_nanos() / iters;
        println!(
            "Metal N={n:>5} K={k:>5}: fused(GPU)={fused_ns:>9}ns  composition={comp_ns:>9}ns  speedup={:.2}x  [N,K]={} MiB",
            comp_ns as f64 / fused_ns as f64,
            (n * k * 4) / (1024 * 1024),
        );
    }
}
