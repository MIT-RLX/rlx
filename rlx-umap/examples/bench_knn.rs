// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Benchmark cosine-distance k-NN parity and speed across RLX backends.
//
// ```sh
// cargo run -p rlx-umap --release --example bench_knn
// cargo run -p rlx-umap --release --example bench_knn --features metal,mlx,gpu
// cargo run -p rlx-umap --release --example bench_knn --features all-backends
// ```

use std::time::{Duration, Instant};

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Session;
use rlx_runtime::device_ext;
use rlx_umap::{
    compare_knn, cosine_knn_graph, cosine_pairwise_reference, knn_forward_packed,
    max_pairwise_error, pairwise_cosine_graph, register, unpack_knn_packed,
};

fn parse_usize(flag: &str, args: &[String], default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn gen_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f32 / u32::MAX as f32;
        out.push(u * 2.0 - 1.0);
    }
    out
}

fn time_it(mut f: impl FnMut(), warmup: usize, runs: usize) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed() / runs as u32
}

fn supports_knn_custom(device: Device) -> bool {
    matches!(
        device,
        Device::Cpu | Device::Metal | Device::Gpu | Device::Cuda | Device::Rocm
    ) || {
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        {
            device == Device::Mlx
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos")))]
        {
            false
        }
    }
}

fn bench_devices() -> Vec<(&'static str, Device)> {
    let candidates: Vec<(&str, Device)> = vec![
        ("cpu", Device::Cpu),
        #[cfg(feature = "metal")]
        ("metal", Device::Metal),
        #[cfg(feature = "mlx")]
        ("mlx", Device::Mlx),
        #[cfg(feature = "gpu")]
        ("wgpu", Device::Gpu),
        #[cfg(feature = "cuda")]
        ("cuda", Device::Cuda),
        #[cfg(feature = "rocm")]
        ("rocm", Device::Rocm),
    ];
    candidates
        .into_iter()
        .filter(|(_, d)| device_ext::is_available(*d))
        .collect()
}

fn run_pairwise_cosine(
    device: Device,
    data: &[f32],
    n: usize,
    d: usize,
) -> Result<Vec<f32>, String> {
    let build = || {
        let mut g = Graph::new("cosine_pw");
        let x = g.input("x", Shape::new(&[n, d], DType::F32));
        let pw = pairwise_cosine_graph(&mut g, x, n);
        g.set_outputs(vec![pw]);
        g
    };
    let mut exe = Session::new(device).compile(build());
    let out = exe.run(&[("x", data)]);
    Ok(out.into_iter().next().unwrap_or_default())
}

fn run_cosine_knn(
    device: Device,
    data: &[f32],
    n: usize,
    d: usize,
    k: u32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    if device == Device::Mlx {
        return rlx_umap::session::cosine_knn_mlx(data, n, d, k);
    }
    if !supports_knn_custom(device) {
        return Err("knn custom op not registered for this backend".into());
    }
    let build = || {
        let mut g = Graph::new("cosine_knn");
        let x = g.input("x", Shape::new(&[n, d], DType::F32));
        let (idx, dist) = cosine_knn_graph(&mut g, x, n, k);
        g.set_outputs(vec![idx, dist]);
        g
    };
    let mut exe = Session::new(device).compile(build());
    let outs = exe.run(&[("x", data)]);
    if outs.len() != 2 {
        return Err(format!("expected 2 outputs, got {}", outs.len()));
    }
    Ok((outs[0].clone(), outs[1].clone()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = parse_usize("--n", &args, 512);
    let d = parse_usize("--d", &args, 64);
    let k = parse_usize("--k", &args, 15);
    let seed = parse_usize("--seed", &args, 42) as u64;
    let warmup = parse_usize("--warmup", &args, 2);
    let runs = parse_usize("--runs", &args, 10);

    register();

    let data = gen_data(n, d, seed);
    let ref_pw = cosine_pairwise_reference(&data, n, d);
    let mut ref_packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&ref_pw, n, k, &mut ref_packed);
    let (ref_idx, ref_dist) = unpack_knn_packed(&ref_packed, n, k);

    println!("# rlx-umap bench — cosine distance + k-NN");
    println!("n={n} d={d} k={k} seed={seed} warmup={warmup} runs={runs}\n");

    let devs = bench_devices();
    if devs.is_empty() {
        eprintln!("no RLX devices available");
        std::process::exit(1);
    }

    println!("## Parity (vs CPU reference)\n");
    println!("| backend | pairwise max err | index match | mean dist err | dist hist L1 |");
    println!("|---------|-------------------|-------------|---------------|--------------|");

    for (label, device) in &devs {
        let pw = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_pairwise_cosine(*device, &data, n, d)
        })) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                println!("| {label} | FAIL pw: {e} | | | |");
                continue;
            }
            Err(_) => {
                println!("| {label} | FAIL pw: panic | | | |");
                continue;
            }
        };
        let pw_err = max_pairwise_error(&ref_pw, &pw);

        if !supports_knn_custom(*device) {
            println!("| {label} | {pw_err:.2e} | knn N/A (no custom op) | | |");
            continue;
        }

        let knn_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_cosine_knn(*device, &data, n, d, k as u32)
        }));
        let (idx, dist) = match knn_result {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                println!("| {label} | {pw_err:.2e} | FAIL knn: {e} | | |");
                continue;
            }
            Err(_) => {
                println!("| {label} | {pw_err:.2e} | FAIL knn: panic | | |");
                continue;
            }
        };
        let report = compare_knn(&ref_idx, &ref_dist, &idx, &dist, n, k);
        println!(
            "| {label} | {pw_err:.2e} | {:.4} | {:.2e} | {:.4} |",
            report.index_match_rate, report.mean_dist_error, report.dist_hist_l1
        );
    }

    println!("\n## Speed (median of {runs} runs after {warmup} warmup)\n");
    println!("| backend | pairwise ms | knn ms | e2e ms | notes |");
    println!("|---------|-------------|--------|--------|-------|");

    for (label, device) in &devs {
        let pw_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            time_it(
                || {
                    let _ = run_pairwise_cosine(*device, &data, n, d).expect("pairwise");
                },
                warmup,
                runs,
            )
        }));
        let pw_ms = match pw_ok {
            Ok(d) => d.as_secs_f64() * 1000.0,
            Err(_) => {
                println!("| {label} | FAIL | — | — | pairwise panic |");
                continue;
            }
        };

        if !supports_knn_custom(*device) {
            println!("| {label} | {pw_ms:.2} | — | {pw_ms:.2} | pairwise only |");
            continue;
        }

        let knn_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            time_it(
                || {
                    let _ = run_cosine_knn(*device, &data, n, d, k as u32).expect("knn");
                },
                warmup,
                runs,
            )
        }));
        let knn_ms = match knn_ok {
            Ok(d) => d.as_secs_f64() * 1000.0,
            Err(_) => {
                println!("| {label} | {pw_ms:.2} | FAIL | — | knn panic |");
                continue;
            }
        };

        let e2e_ms = knn_ms;
        println!("| {label} | {pw_ms:.2} | {knn_ms:.2} | {e2e_ms:.2} | |");
    }

    println!("\n_Reference: `cosine_pairwise_reference` + `knn_forward_packed` on host._");
}
