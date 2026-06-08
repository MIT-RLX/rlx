// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! On-disk wgpu calibration for cost-model ranking.

use rlx_ir::{DType, Graph, Shape, Tick};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::backend::WgpuExecutable;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub device_name: String,
    pub sgemm_gflops: f64,
    pub roundtrip_overhead_ns: f64,
    pub memory_bw_gbps: f64,
}

fn cache_path(device_name: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let dir = PathBuf::from(home).join(".cache").join("rlx");
    let _ = std::fs::create_dir_all(&dir);
    let slug: String = device_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    dir.join(format!("wgpu-calib-{slug}.json"))
}

impl Calibration {
    pub fn load(device_name: &str) -> Option<Self> {
        let raw = std::fs::read_to_string(cache_path(device_name)).ok()?;
        let cal: Calibration = serde_json::from_str(&raw).ok()?;
        (cal.device_name == device_name).then_some(cal)
    }

    pub fn save(&self) -> std::io::Result<()> {
        let raw = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(cache_path(&self.device_name), raw)
    }

    pub fn measure(device_name: &str) -> Self {
        const M: usize = 512;
        const K: usize = 512;
        const N: usize = 512;
        let mut g = Graph::new("calib_mm");
        let x = g.input("x", Shape::new(&[M, K], DType::F32));
        let w = g.param("w", Shape::new(&[K, N], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[M, N], DType::F32));
        g.set_outputs(vec![y]);

        let mut exe = WgpuExecutable::compile(g);
        let wv: Vec<f32> = vec![1.0; K * N];
        let xv: Vec<f32> = vec![1.0; M * K];
        exe.set_param("w", &wv);

        for _ in 0..3 {
            let _ = exe.run(&[("x", &xv)]);
        }
        const ITERS: usize = 20;
        let t0 = Tick::now();
        for _ in 0..ITERS {
            let _ = exe.run(&[("x", &xv)]);
        }
        let total_ns = Tick::now().elapsed_ns(t0) as f64;
        let flops = 2.0 * (M * K * N) as f64 * ITERS as f64;
        let sgemm_gflops = flops / (total_ns / 1e9);

        let t1 = Tick::now();
        let _ = exe.run(&[("x", &xv)]);
        let roundtrip_overhead_ns = Tick::now().elapsed_ns(t1) as f64;

        Self {
            device_name: device_name.to_string(),
            sgemm_gflops,
            roundtrip_overhead_ns,
            memory_bw_gbps: 120.0,
        }
    }

    pub fn load_or_measure() -> Self {
        if !crate::is_available() {
            return Self {
                device_name: "wgpu-unavailable".into(),
                sgemm_gflops: 2_500.0,
                roundtrip_overhead_ns: 80_000.0,
                memory_bw_gbps: 120.0,
            };
        }
        let name = crate::device::adapter_name().unwrap_or_else(|| "wgpu".into());
        if let Some(cal) = Self::load(&name) {
            return cal;
        }
        let cal = Self::measure(&name);
        let _ = cal.save();
        cal
    }
}
