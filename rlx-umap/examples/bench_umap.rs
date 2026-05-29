// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// ```sh
// cargo run -p rlx-umap --release --example bench_umap --features full
// cargo run -p rlx-umap --release --example bench_umap --features full,metal -- --device metal
// ```

use std::env;
use std::time::Instant;

use rlx_driver::Device;
use rlx_runtime::device_ext;
use rlx_umap::prelude::*;

fn parse_usize(flag: &str, args: &[String], default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_device(args: &[String]) -> Device {
    let name = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("cpu");
    match name {
        "metal" => Device::Metal,
        "mlx" => Device::Mlx,
        "wgpu" | "gpu" => Device::Gpu,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = parse_usize("--n", &args, 256);
    let d = parse_usize("--d", &args, 32);
    let epochs = parse_usize("--epochs", &args, 50);
    let device = parse_device(&args);
    assert!(
        device_ext::is_available(device),
        "device {device:?} not available"
    );

    register();
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| ((i + j) as f64 * 0.11).sin()).collect())
        .collect();

    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: epochs,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let start = Instant::now();
    let fitted = Umap::with_device(config, device).fit(data);
    let elapsed = start.elapsed();

    let emb = fitted.embedding();
    let mut max_abs = 0.0f64;
    for row in emb {
        for &v in row {
            max_abs = max_abs.max(v.abs());
        }
    }

    println!(
        "fit n={n} d={d} device={device:?} epochs={epochs} time={:.2}s best_loss embedding_max_abs={max_abs:.4}",
        elapsed.as_secs_f64()
    );
    println!("embedding[0] = [{:.4}, {:.4}]", emb[0][0], emb[0][1]);
}
