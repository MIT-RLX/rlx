// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// Sweep vector width `N` for third-order reverse-mode AD on `f(x)=sum(x³)`.
//
// ```sh
// # macOS (all Apple backends)
// scripts/check-throttle.sh && \
//   cargo run -p rlx-bench --release --example bench_nth_order --features metal,mlx,gpu
//
// # CUDA rig (Windows / WSL)
// ./rig.sh bench-nth-order both
// ./rig.sh --wsl bench-nth-order
// ```

use rlx_driver::Device;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape, Tick};
use rlx_opt::nth_order_grad;
use rlx_runtime::CompileCache;
use std::collections::HashMap;

fn parse_usize(flag: &str, args: &[String], default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_string(flag: &str, args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn batch_sizes(args: &[String]) -> Vec<usize> {
    if let Some(list) = parse_string("--batch-sizes", args) {
        return list
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|&n| n > 0)
            .collect();
    }
    let min_n = parse_usize("--batch-min", args, 64).max(1);
    let max_n = parse_usize("--batch-max", args, 4096).max(min_n);
    let mut sizes = Vec::new();
    let mut n = min_n;
    while n <= max_n {
        sizes.push(n);
        if n > max_n / 2 {
            break;
        }
        n = n.saturating_mul(2);
    }
    if sizes.last().copied() != Some(max_n) {
        sizes.push(max_n);
    }
    sizes
}

fn gen_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(n as u64);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f32 / u32::MAX as f32;
        out.push(0.5 + u * 0.5);
    }
    out
}

fn build_cubic_sum(n: usize) -> Graph {
    let mut g = Graph::new("cubic_sum");
    let shape = Shape::new(&[n], DType::F32);
    let x = g.input("x", shape.clone());
    let x2 = g.binary(BinaryOp::Mul, x, x, shape.clone());
    let x3 = g.binary(BinaryOp::Mul, x2, x, shape);
    let f = g.reduce(x3, ReduceOp::Sum, vec![0], false, Shape::scalar(DType::F32));
    g.set_outputs(vec![f]);
    g
}

fn median_ns(mut samples: Vec<u64>) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn devices() -> Vec<(&'static str, Device)> {
    let out = vec![("cpu", Device::Cpu)];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if is_available(Device::Metal) {
        out.push(("metal", Device::Metal));
    }
    #[cfg(feature = "mlx")]
    if is_available(Device::Mlx) {
        out.push(("mlx", Device::Mlx));
    }
    #[cfg(feature = "gpu")]
    if is_available(Device::Gpu) {
        out.push(("wgpu", Device::Gpu));
    }
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        out.push(("cuda", Device::Cuda));
    }
    #[cfg(feature = "rocm")]
    if is_available(Device::Rocm) {
        out.push(("rocm", Device::Rocm));
    }
    out
}

struct Row {
    n: usize,
    ad_ns: u64,
    compile_ns: u64,
    cache_hit_ns: u64,
    exec_median_ns: u64,
}

fn bench_device(
    cache: &mut CompileCache,
    key: u64,
    hg: &Graph,
    x_bytes: &[u8],
    warmup: usize,
    runs: usize,
) -> (u64, u64, u64) {
    let t0 = Tick::now();
    let _ = cache.get_or_compile(key, || hg.clone());
    let compile_ns = Tick::now().elapsed_ns(t0);

    let t0 = Tick::now();
    let compiled = cache.get_or_compile(key, || hg.clone());
    let cache_hit_ns = Tick::now().elapsed_ns(t0);

    for _ in 0..warmup {
        let _ = compiled.run_typed(&[("x", x_bytes, DType::F32)]);
    }

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Tick::now();
        let _ = compiled.run_typed(&[("x", x_bytes, DType::F32)]);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    (compile_ns, cache_hit_ns, median_ns(samples))
}

fn prewarm_cuda(_order: usize) {
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        let forward = build_cubic_sum(64);
        let hg = nth_order_grad(&forward, "x", order);
        let x = gen_x(64, 0xDEAD_BEEF);
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut ex = Session::new(Device::Cuda).compile(hg);
        let _ = ex.run_typed(&[("x", &x_bytes, DType::F32)]);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let order = parse_usize("--order", &args, 3);
    let warmup = parse_usize("--warmup", &args, 5);
    let runs = parse_usize("--runs", &args, 200);
    let sizes = batch_sizes(&args);
    let devs = devices();

    println!("# nth_order_grad bench — f(x)=sum(x³), order={order}");
    println!(
        "# platform: {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "# devices: {}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>().join(", ")
    );
    println!("# sizes N: {sizes:?}, runs={runs}, warmup={warmup}");
    println!("# compile via CompileCache (cache miss + hit columns)");
    if std::env::var("RLX_CUDA_COMPILE_MODE").is_ok_and(|v| v.eq_ignore_ascii_case("aot")) {
        println!("# RLX_CUDA_COMPILE_MODE=aot (NVRTC prewarm once per process)");
    }
    if std::env::var("RLX_CUDA_EXEC_MODE").is_ok_and(|v| v.eq_ignore_ascii_case("graph")) {
        println!("# RLX_CUDA_EXEC_MODE=graph (CUDA Graph replay after 1st run)");
    }
    println!();

    prewarm_cuda(order);

    let mut caches: HashMap<Device, CompileCache> = devs
        .iter()
        .map(|(_, device)| (*device, CompileCache::new(*device, sizes.len().max(8))))
        .collect();

    let mut tables: std::collections::HashMap<&str, Vec<Row>> = std::collections::HashMap::new();
    for &(label, _) in &devs {
        tables.insert(label, Vec::new());
    }

    for &n in &sizes {
        let forward = build_cubic_sum(n);
        let t0 = Tick::now();
        let hg = nth_order_grad(&forward, "x", order);
        let ad_ns = Tick::now().elapsed_ns(t0);

        let x = gen_x(n, 0xC0FFEE);
        let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

        for &(label, device) in &devs {
            let cache = caches.get_mut(&device).expect("compile cache");
            let (compile_ns, cache_hit_ns, exec_median_ns) =
                bench_device(cache, n as u64, &hg, &x_bytes, warmup, runs);
            tables.get_mut(label).unwrap().push(Row {
                n,
                ad_ns,
                compile_ns,
                cache_hit_ns,
                exec_median_ns,
            });
        }
    }

    for &(label, _) in &devs {
        println!("## {label}\n");
        println!("| N | AD build µs | compile µs | cache hit µs | exec median µs |");
        println!("|---:|---:|---:|---:|---:|");
        for row in &tables[label] {
            println!(
                "| {} | {:.1} | {:.1} | {:.1} | {:.1} |",
                row.n,
                row.ad_ns as f64 / 1e3,
                row.compile_ns as f64 / 1e3,
                row.cache_hit_ns as f64 / 1e3,
                row.exec_median_ns as f64 / 1e3,
            );
        }
        println!();
    }
}
