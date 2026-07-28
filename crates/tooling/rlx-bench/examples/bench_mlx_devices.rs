// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Matmul sweep for rlx-mlx on different MLX device backends:
//
// ```sh
// # Apple Silicon (MLX Metal, default device)
// cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx
//
// # WSL / Linux — CPU and CUDA legs (separate processes; MLX device is fixed at init)
// cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx -- --leg mlx-cpu
// RLX_MLX_CUDA=1 cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx-cuda -- --leg mlx-cuda
//
// # Or run all legs (spawns child processes on Linux; CUDA leg needs mlx-cuda feature)
// cargo run -p rlx-bench --release --example bench_mlx_devices --features mlx,mlx-cuda
//
// Via rig:
// ./rig.sh bench-mlx-devices wsl
// ```

use rlx_bench::{patterns::MatmulPattern, run_benchmark};
use rlx_driver::Device;
use std::env;
use std::process::{Command, ExitCode};

fn gflops(m: usize, k: usize, n: usize, median_ns: u64) -> f64 {
    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    flops / (median_ns as f64 / 1e9)
}

fn mlx_device_label() -> String {
    #[cfg(feature = "mlx")]
    {
        let name = rlx_mlx::array::device_name();
        if name.is_empty() {
            format!(
                "mlx(env={})",
                env::var("RLX_MLX_DEVICE").unwrap_or_else(|_| "default".into())
            )
        } else {
            format!("mlx({name})")
        }
    }
    #[cfg(not(feature = "mlx"))]
    {
        "mlx".into()
    }
}

fn run_matmul_sweep(leg: &str) -> i32 {
    let host = env::var("RLX_RIG_RUNTIME").unwrap_or_else(|_| "local".into());
    println!(
        "rlx-bench mlx devices — leg={leg} host={host} backend={}",
        mlx_device_label()
    );

    let shapes = [
        (8, 64, 64),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    for &(m, k, n) in &shapes {
        println!("\n# matmul {m}x{k}x{n}");
        let pattern = MatmulPattern { m, k, n };
        let r = run_benchmark(&pattern, Device::Mlx, 3, 20);
        let med = r.median_ns();
        println!(
            "  {leg:14} median={:.2}µs mean={:.2}µs  {:.2} GFLOP/s",
            med as f64 / 1000.0,
            r.mean_ns() as f64 / 1000.0,
            gflops(m, k, n, med)
        );
    }
    0
}

fn spawn_leg(leg: &str, mlx_device: Option<&str>) -> i32 {
    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bench_mlx_devices: current_exe: {e}");
            return 1;
        }
    };
    let mut cmd = Command::new(exe);
    cmd.arg("--leg").arg(leg);
    if let Some(dev) = mlx_device {
        cmd.env("RLX_MLX_DEVICE", dev);
    } else {
        cmd.env_remove("RLX_MLX_DEVICE");
    }
    match cmd.status() {
        Ok(s) if s.success() => 0,
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!("bench_mlx_devices: failed to spawn {leg}: {e}");
            1
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--leg") {
        let leg = args.get(pos + 1).map(String::as_str).unwrap_or("mlx");
        return ExitCode::from(run_matmul_sweep(leg) as u8);
    }

    let mut fail = 0;
    if cfg!(target_os = "macos") {
        fail |= spawn_leg("apple-silicon", None);
    } else {
        fail |= spawn_leg("mlx-cpu", Some("cpu"));
        #[cfg(feature = "mlx-cuda")]
        {
            fail |= spawn_leg("mlx-cuda", Some("gpu"));
        }
        #[cfg(not(feature = "mlx-cuda"))]
        {
            eprintln!("skip mlx-cuda leg: rebuild with --features mlx-cuda (RLX_MLX_CUDA=1)");
        }
    }
    ExitCode::from(fail as u8)
}
