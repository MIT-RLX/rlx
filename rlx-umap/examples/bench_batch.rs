// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Sweep batch size (= number of points `n`) and report timing per backend.
//
// ```sh
// cargo run -p rlx-umap --release --example bench_batch
// cargo run -p rlx-umap --release --example bench_batch --features metal,mlx,gpu
// cargo run -p rlx-umap --release --example bench_batch -- --batch-min 1 --batch-max 4096 --batch-step 1
// cargo run -p rlx-umap --release --example bench_batch -- --csv /tmp/umap_batch.csv
// ```

use std::io::Write;
use std::time::{Duration, Instant};

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::device_ext;
use rlx_runtime::{CompiledGraph, Session};
use rlx_umap::{cosine_knn_graph, pairwise_cosine_graph, register};

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

fn gen_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(n as u64);
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (state >> 33) as f32 / u32::MAX as f32;
        out.push(u * 2.0 - 1.0);
    }
    out
}

fn effective_k(n: usize, k: usize) -> Option<usize> {
    if n < 2 {
        return None;
    }
    Some(k.min(n - 1).max(1))
}

/// Batch sizes to test (default: powers of two from `batch_min` to `batch_max`).
fn batch_sizes(args: &[String]) -> Vec<usize> {
    if let Some(list) = parse_string("--batch-sizes", args) {
        return list
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .filter(|&n| n > 0)
            .collect();
    }
    let min_n = parse_usize("--batch-min", args, 1).max(1);
    let max_n = parse_usize("--batch-max", args, 4096).max(min_n);
    let step = parse_usize("--batch-step", args, 0);
    if step > 0 {
        return (min_n..=max_n).step_by(step).collect();
    }
    let mut sizes = Vec::new();
    let mut n = min_n;
    while n <= max_n {
        sizes.push(n);
        if n > max_n / 2 {
            break;
        }
        n = n.saturating_mul(2);
    }
    if sizes.last().copied() != Some(max_n) && max_n > *sizes.last().unwrap_or(&0) {
        sizes.push(max_n);
    }
    sizes
}

fn time_run(mut f: impl FnMut(), warmup: usize, runs: usize) -> Duration {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..runs {
        f();
    }
    t0.elapsed() / runs.max(1) as u32
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

struct PwExe {
    exe: CompiledGraph,
}

impl PwExe {
    fn compile(device: Device, n: usize, d: usize) -> Self {
        let mut g = Graph::new("cosine_pw");
        let x = g.input("x", Shape::new(&[n, d], DType::F32));
        let pw = pairwise_cosine_graph(&mut g, x, n);
        g.set_outputs(vec![pw]);
        Self {
            exe: Session::new(device).compile(g),
        }
    }

    fn run(&mut self, data: &[f32]) {
        let _ = self.exe.run(&[("x", data)]);
    }
}

struct KnnExe {
    exe: CompiledGraph,
}

impl KnnExe {
    fn compile(device: Device, n: usize, d: usize, k: u32) -> Self {
        let mut g = Graph::new("cosine_knn");
        let x = g.input("x", Shape::new(&[n, d], DType::F32));
        let (idx, dist) = cosine_knn_graph(&mut g, x, n, k);
        g.set_outputs(vec![idx, dist]);
        Self {
            exe: Session::new(device).compile(g),
        }
    }

    fn run(&mut self, data: &[f32]) {
        let _ = self.exe.run(&[("x", data)]);
    }
}

/// MLX: pairwise on device, k-NN on CPU from precomputed pairwise buffer.
struct MlxSplitExe {
    pw: PwExe,
    knn_cpu: CompiledGraph,
}

impl MlxSplitExe {
    fn compile(n: usize, d: usize, k: u32) -> Self {
        let mut g_knn = Graph::new("mlx_knn_cpu");
        let pw_in = g_knn.input("pairwise", Shape::new(&[n, n], DType::F32));
        let packed = rlx_umap::knn_graph(&mut g_knn, pw_in, k);
        let (idx, dist) = rlx_umap::split_knn_packed(&mut g_knn, packed, k);
        g_knn.set_outputs(vec![idx, dist]);
        Self {
            pw: PwExe::compile(Device::Mlx, n, d),
            knn_cpu: Session::new(Device::Cpu).compile(g_knn),
        }
    }

    fn run_e2e(&mut self, data: &[f32]) {
        let pw = self.pw.exe.run(&[("x", data)]).remove(0);
        let _ = self.knn_cpu.run(&[("pairwise", &pw)]);
    }
}

enum BackendExe {
    Standard {
        pw: PwExe,
        knn: KnnExe,
    },
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    MlxSplit(MlxSplitExe),
}

impl BackendExe {
    fn compile(device: Device, n: usize, d: usize, k: u32) -> Result<Self, String> {
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        if device == Device::Mlx {
            return Ok(Self::MlxSplit(MlxSplitExe::compile(n, d, k)));
        }
        Ok(Self::Standard {
            pw: PwExe::compile(device, n, d),
            knn: KnnExe::compile(device, n, d, k),
        })
    }

    fn time_pairwise(&mut self, data: &[f32], warmup: usize, runs: usize) -> Duration {
        match self {
            Self::Standard { pw, .. } => time_run(|| pw.run(data), warmup, runs),
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            Self::MlxSplit(s) => time_run(|| s.pw.run(data), warmup, runs),
        }
    }

    fn time_knn(&mut self, data: &[f32], warmup: usize, runs: usize) -> Duration {
        match self {
            Self::Standard { knn, .. } => time_run(|| knn.run(data), warmup, runs),
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            Self::MlxSplit(s) => time_run(
                || {
                    let pw = s.pw.exe.run(&[("x", data)]).remove(0);
                    let _ = s.knn_cpu.run(&[("pairwise", &pw)]);
                },
                warmup,
                runs,
            ),
        }
    }

    fn time_e2e(&mut self, data: &[f32], warmup: usize, runs: usize) -> Duration {
        match self {
            Self::Standard { knn, .. } => time_run(|| knn.run(data), warmup, runs),
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            Self::MlxSplit(s) => time_run(|| s.run_e2e(data), warmup, runs),
        }
    }
}

struct Row {
    n: usize,
    k: usize,
    backend: String,
    pairwise_ms: f64,
    knn_ms: f64,
    e2e_ms: f64,
    status: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let d = parse_usize("--d", &args, 64);
    let k_req = parse_usize("--k", &args, 15);
    let seed = parse_usize("--seed", &args, 42) as u64;
    let warmup = parse_usize("--warmup", &args, 1);
    let runs = parse_usize("--runs", &args, 5);
    let csv_path = parse_string("--csv", &args);
    let sizes = batch_sizes(&args);

    register();

    let devs = bench_devices();
    if devs.is_empty() {
        eprintln!("no RLX devices available");
        std::process::exit(1);
    }

    println!("# rlx-umap batch sweep (batch size = n points)");
    println!("d={d} k_req={k_req} seed={seed} warmup={warmup} runs={runs}");
    println!(
        "sizes ({}) = {:?}",
        sizes.len(),
        if sizes.len() <= 24 {
            format!("{sizes:?}")
        } else {
            format!(
                "{}..{} ({} values)",
                sizes.first().unwrap(),
                sizes.last().unwrap(),
                sizes.len()
            )
        }
    );
    println!();

    let mut rows: Vec<Row> = Vec::new();

    for &n in &sizes {
        let Some(k) = effective_k(n, k_req) else {
            for (label, _) in &devs {
                rows.push(Row {
                    n,
                    k: 0,
                    backend: label.to_string(),
                    pairwise_ms: f64::NAN,
                    knn_ms: f64::NAN,
                    e2e_ms: f64::NAN,
                    status: "skip (n<2)".into(),
                });
            }
            continue;
        };
        let k_u32 = k as u32;
        let data = gen_data(n, d, seed);

        for (label, device) in &devs {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut exe = BackendExe::compile(*device, n, d, k_u32)?;
                let pw_ms = exe.time_pairwise(&data, warmup, runs).as_secs_f64() * 1000.0;
                let knn_ms = exe.time_knn(&data, warmup, runs).as_secs_f64() * 1000.0;
                let e2e_ms = exe.time_e2e(&data, warmup, runs).as_secs_f64() * 1000.0;
                Ok::<_, String>((pw_ms, knn_ms, e2e_ms))
            }));

            match result {
                Ok(Ok((pw_ms, knn_ms, e2e_ms))) => {
                    rows.push(Row {
                        n,
                        k,
                        backend: label.to_string(),
                        pairwise_ms: pw_ms,
                        knn_ms,
                        e2e_ms,
                        status: "ok".into(),
                    });
                }
                Ok(Err(e)) => {
                    rows.push(Row {
                        n,
                        k,
                        backend: label.to_string(),
                        pairwise_ms: f64::NAN,
                        knn_ms: f64::NAN,
                        e2e_ms: f64::NAN,
                        status: format!("err: {e}"),
                    });
                }
                Err(_) => {
                    rows.push(Row {
                        n,
                        k,
                        backend: label.to_string(),
                        pairwise_ms: f64::NAN,
                        knn_ms: f64::NAN,
                        e2e_ms: f64::NAN,
                        status: "panic".into(),
                    });
                }
            }
        }
    }

    // Markdown: e2e ms per backend, rows = batch size
    println!("## End-to-end time (ms) — rows=batch size `n`, columns=backend\n");
    print!("| n |");
    for (label, _) in &devs {
        print!(" {label} |");
    }
    println!();
    print!("|---:|");
    for _ in &devs {
        print!("---:|");
    }
    println!();

    for &n in &sizes {
        print!("| {n} |");
        for (label, _) in &devs {
            let r = rows
                .iter()
                .find(|r| r.n == n && r.backend == *label)
                .map(|r| {
                    if r.status == "ok" {
                        format!("{:.3}", r.e2e_ms)
                    } else {
                        r.status.clone()
                    }
                })
                .unwrap_or_else(|| "—".into());
            print!(" {r} |");
        }
        println!();
    }

    println!("\n## Pairwise only (ms)\n");
    print!("| n |");
    for (label, _) in &devs {
        print!(" {label} |");
    }
    println!();
    print!("|---:|");
    for _ in &devs {
        print!("---:|");
    }
    println!();
    for &n in &sizes {
        print!("| {n} |");
        for (label, _) in &devs {
            let cell = rows
                .iter()
                .find(|r| r.n == n && r.backend == *label)
                .map(|r| {
                    if r.status == "ok" {
                        format!("{:.3}", r.pairwise_ms)
                    } else {
                        r.status.clone()
                    }
                })
                .unwrap_or_else(|| "—".into());
            print!(" {cell} |");
        }
        println!();
    }

    if let Some(path) = csv_path {
        let mut f = std::fs::File::create(&path).expect("create csv");
        writeln!(f, "n,k,d,backend,pairwise_ms,knn_ms,e2e_ms,status").expect("csv header");
        for r in &rows {
            writeln!(
                f,
                "{},{},{},{},{:.6},{:.6},{:.6},{}",
                r.n, r.k, d, r.backend, r.pairwise_ms, r.knn_ms, r.e2e_ms, r.status
            )
            .expect("csv row");
        }
        eprintln!("\nWrote CSV to {path}");
    }

    println!(
        "\n_Timing: compile once per (backend, n), then median of {runs} runs (warmup {warmup})._"
    );
    println!(
        "_Default sizes: powers of 2 in [--batch-min, --batch-max]. Use --batch-step 1 for every n._"
    );
}
