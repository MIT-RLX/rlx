// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Fit parametric UMAP on MNIST pixels and write a 2-D scatter plot (SVG + CSV).
//
// ```sh
// cargo run -p rlx-umap --release --example mnist_embedding --features full
// cargo run -p rlx-umap --release --example mnist_embedding --features full,metal -- --n 10000 --epochs 150 --out mnist_umap.svg
// cargo run -p rlx-umap --release --example mnist_embedding --features full -- --synthetic --n 2000  # no download
// ```

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use mnist::{Mnist, MnistBuilder};
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

fn parse_path(flag: &str, args: &[String], default: &str) -> PathBuf {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
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
        "gpu" | "wgpu" => Device::Gpu,
        "cuda" => Device::Cuda,
        _ => Device::Cpu,
    }
}

fn try_load_mnist(n: usize) -> Result<(Vec<Vec<f64>>, Vec<u8>), String> {
    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(n as u32)
        .finalize();

    let d = 28 * 28;
    if trn_img.len() < n * d {
        return Err(format!("expected {} pixels, got {}", n * d, trn_img.len()));
    }
    let data: Vec<Vec<f64>> = trn_img
        .chunks(d)
        .take(n)
        .map(|row| row.iter().map(|&b| b as f64 / 255.0).collect())
        .collect();
    Ok((data, trn_lbl[..n].to_vec()))
}

/// Fallback when MNIST download/extract fails: simple 28×28 digit-like blobs.
fn synthetic_mnist(n: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let d = 28 * 28;
    let mut data = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let lbl = (i % 10) as u8;
        labels.push(lbl);
        let cx = 8 + (lbl as usize * 2) % 12;
        let cy = 10 + (lbl as usize * 3) % 8;
        let mut img = vec![0.0f64; d];
        for r in 0..28 {
            for c in 0..28 {
                let dr = r as i32 - cy as i32;
                let dc = c as i32 - cx as i32;
                let dist2 = (dr * dr + dc * dc) as f64;
                let v = (-dist2 / 18.0).exp();
                img[r * 28 + c] = v;
            }
        }
        // unique jitter per sample
        for (j, px) in img.iter_mut().enumerate() {
            *px += 0.05 * ((i * 17 + j) % 50) as f64 / 50.0;
        }
        data.push(img);
    }
    (data, labels)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn load_mnist(n: usize, args: &[String]) -> (Vec<Vec<f64>>, Vec<u8>) {
    if has_flag(args, "--synthetic") {
        println!("Using synthetic 28×28 digit blobs (--synthetic).");
        return synthetic_mnist(n);
    }

    let loaded = std::panic::catch_unwind(|| try_load_mnist(n));
    match loaded {
        Ok(Ok(v)) => {
            println!("Using MNIST training images (normalized 0–1).");
            v
        }
        Ok(Err(e)) => {
            eprintln!("MNIST unavailable ({e}); using synthetic fallback.");
            synthetic_mnist(n)
        }
        Err(_) => {
            eprintln!("MNIST loader panicked; using synthetic fallback.");
            synthetic_mnist(n)
        }
    }
}

/// Digit colors (tab10-inspired, sRGB hex).
const COLORS: [&str; 10] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
];

fn write_csv(path: &PathBuf, embedding: &[Vec<f64>], labels: &[u8]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "x,y,label")?;
    for (pt, &lbl) in embedding.iter().zip(labels) {
        writeln!(f, "{:.6},{:.6},{}", pt[0], pt[1], lbl)?;
    }
    Ok(())
}

fn write_svg(
    path: &PathBuf,
    embedding: &[Vec<f64>],
    labels: &[u8],
    width: u32,
    height: u32,
) -> std::io::Result<()> {
    let margin = 48.0f64;
    let w = width as f64 - 2.0 * margin;
    let h = height as f64 - 2.0 * margin;

    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for pt in embedding {
        min_x = min_x.min(pt[0]);
        max_x = max_x.max(pt[0]);
        min_y = min_y.min(pt[1]);
        max_y = max_y.max(pt[1]);
    }
    let dx = (max_x - min_x).max(1e-9);
    let dy = (max_y - min_y).max(1e-9);

    let mut f = File::create(path)?;
    writeln!(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(
        f,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )?;
    writeln!(f, "<rect width=\"100%\" height=\"100%\" fill=\"#fafafa\"/>")?;
    writeln!(
        f,
        "<text x=\"{:.0}\" y=\"28\" font-family=\"system-ui,sans-serif\" font-size=\"16\" fill=\"#111111\">rlx-umap MNIST embedding (n={})</text>",
        margin,
        embedding.len()
    )?;

    for (pt, &lbl) in embedding.iter().zip(labels) {
        let x = margin + (pt[0] - min_x) / dx * w;
        let y = margin + (1.0 - (pt[1] - min_y) / dy) * h;
        let color = COLORS[lbl as usize % 10];
        writeln!(
            f,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"2.2\" fill=\"{color}\" fill-opacity=\"0.65\"/>"
        )?;
    }

    let legend_x = width as f64 - margin - 80.0;
    for digit in 0..10u8 {
        let legend_y = margin + digit as f64 * 18.0;
        writeln!(
            f,
            "<circle cx=\"{legend_x:.0}\" cy=\"{legend_y:.0}\" r=\"5\" fill=\"{}\"/>",
            COLORS[digit as usize]
        )?;
        writeln!(
            f,
            "<text x=\"{:.0}\" y=\"{:.0}\" dominant-baseline=\"middle\" font-family=\"monospace\" font-size=\"12\" fill=\"#333333\">{}</text>",
            legend_x + 14.0,
            legend_y,
            digit
        )?;
    }

    writeln!(f, "</svg>")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let n = parse_usize("--n", &args, 5_000);
    let epochs = parse_usize("--epochs", &args, 100);
    let device = parse_device(&args);
    let svg_path = parse_path("--out", &args, "mnist_embedding.svg");
    let csv_path = parse_path("--csv", &args, "mnist_embedding.csv");

    assert!(
        device_ext::is_available(device),
        "device {device:?} is not available"
    );

    println!("Loading MNIST training set (n={n}) …");
    let (data, labels) = load_mnist(n, &args);
    let d = data[0].len();
    println!("features={d} labels 0..9");

    register();

    let config = UmapConfig {
        n_components: 2,
        hidden_sizes: vec![128, 64],
        graph: GraphParams {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: epochs,
            learning_rate: 0.001,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    println!("Fitting UMAP on {device:?} …");
    let fitted = Umap::with_device(config, device).fit(data);
    let embedding = fitted.embedding();

    write_svg(&svg_path, embedding, &labels, 900, 700)?;
    write_csv(&csv_path, embedding, &labels)?;
    println!("Wrote {}", svg_path.display());
    println!("Wrote {}", csv_path.display());
    println!(
        "embedding range: x=[{:.3}, {:.3}] y=[{:.3}, {:.3}]",
        embedding.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        embedding
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max),
        embedding.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        embedding
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max),
    );

    Ok(())
}
