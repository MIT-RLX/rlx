// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Train parametric UMAP using RLX autodiff (no Burn).
//
// ```sh
// cargo run -p rlx-umap --release --bin train-umap --features full
// cargo run -p rlx-umap --release --bin train-umap --features full -- \
//   --csv data.csv --epochs 100 --device metal --save model.ruama --embedding out.csv
// ```

use std::env;
use std::path::PathBuf;

use rlx_driver::Device;
use rlx_runtime::device_ext;
use rlx_umap::config::{GraphParams, OptimizationParams, UmapConfig};
use rlx_umap::data::{load_csv, load_f64_matrix, load_synthetic, write_embedding_csv};
use rlx_umap::training::EpochProgress;
use rlx_umap::training::{FitOptions, fit_with_progress};
use rlx_umap::{register, serialize::model_path};

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_usize(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_f64(args: &[String], name: &str, default: f64) -> f64 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_path(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}

fn parse_string(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_device(args: &[String]) -> Device {
    match parse_string(args, "--device").as_deref() {
        Some("metal") => Device::Metal,
        Some("mlx") => Device::Mlx,
        Some("gpu") | Some("wgpu") => Device::Gpu,
        Some("cuda") => Device::Cuda,
        Some("rocm") => Device::Rocm,
        _ => Device::Cpu,
    }
}

fn print_help() {
    eprintln!(
        "train-umap — parametric UMAP training (RLX autodiff, no Burn)

Usage:
  train-umap [OPTIONS]

Data (one required):
  --csv PATH          CSV rows (comma-separated features)
  --f64 PATH          Binary row-major f64 (u64 n, u64 d, data…)
  --synthetic         Random [0,1) data (default if no file)

Training:
  --n N               Samples for --synthetic (default 1000)
  --d D               Features for --synthetic (default 32)
  --epochs N          Training epochs (default 100)
  --k K               n_neighbors (default 15)
  --lr RATE           Learning rate (default 0.001)
  --components C      Output dims (default 2)
  --hidden LIST       Hidden layers, e.g. 128,64 (default 100)
  --device DEVICE     cpu | metal | mlx | gpu | cuda (default cpu)
  --no-pca            Disable PCA warm-start (needs pca feature)

Output:
  --save PATH         Save model (.safetensors or .gguf by extension)
  --embedding PATH    Write embedding CSV (x,y)
  --quiet             Less logging

Example:
  cargo run -p rlx-umap --release --bin train-umap --features full -- \\
    --synthetic --n 2000 --epochs 80 --save /tmp/m.ruama --embedding /tmp/emb.csv
"
    );
}

fn parse_hidden(args: &[String]) -> Vec<usize> {
    parse_path(args, "--hidden")
        .map(|p| {
            p.to_string_lossy()
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![100])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if flag(&args, "--help") || flag(&args, "-h") {
        print_help();
        return Ok(());
    }

    let device = parse_device(&args);
    assert!(
        device_ext::is_available(device),
        "device {device:?} is not available"
    );

    register();

    let data = if let Some(p) = parse_path(&args, "--csv") {
        eprintln!("Loading CSV {} …", p.display());
        load_csv(&p)?
    } else if let Some(p) = parse_path(&args, "--f64") {
        eprintln!("Loading f64 matrix {} …", p.display());
        load_f64_matrix(&p)?
    } else {
        let n = parse_usize(&args, "--n", 1000);
        let d = parse_usize(&args, "--d", 32);
        let seed = parse_usize(&args, "--seed", 42) as u64;
        eprintln!("Synthetic data n={n} d={d}");
        load_synthetic(n, d, seed)
    };

    let n = data.len();
    let d = data[0].len();
    eprintln!("Training: n={n} features={d} device={device:?} (RLX autodiff)");

    let verbose = !flag(&args, "--quiet");
    let mut config = UmapConfig {
        n_components: parse_usize(&args, "--components", 2),
        hidden_sizes: parse_hidden(&args),
        graph: GraphParams {
            n_neighbors: parse_usize(&args, "--k", 15),
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: parse_usize(&args, "--epochs", 100),
            learning_rate: parse_f64(&args, "--lr", 0.001),
            verbose,
            ..Default::default()
        },
        ..Default::default()
    };

    if flag(&args, "--no-pca") {
        config.optimization.pca_warmstart = false;
    }

    let options = FitOptions::new(device).with_ctrlc();

    let fitted = fit_with_progress(config.clone(), data, options, move |p: EpochProgress| {
        if verbose {
            eprintln!(
                "[train-umap] epoch {}/{} loss={:.6} best={:.6} elapsed={:.1}s",
                p.epoch, p.total_epochs, p.loss, p.best_loss, p.elapsed_secs
            );
        }
    });

    let emb = fitted.embedding();
    eprintln!(
        "Done — best embedding shape {} × {} (components={})",
        emb.len(),
        emb.first().map(|r| r.len()).unwrap_or(0),
        config.n_components
    );

    if let Some(path) = parse_path(&args, "--save") {
        fitted.save(&path)?;
        eprintln!("Saved model {}", path.display());
    } else if verbose {
        let path = model_path(std::env::temp_dir(), "train_umap");
        fitted.save(&path)?;
        eprintln!("Saved model {}", path.display());
    }

    if let Some(path) = parse_path(&args, "--embedding") {
        write_embedding_csv(&path, emb, None)?;
        eprintln!("Wrote embedding {}", path.display());
    }

    Ok(())
}
