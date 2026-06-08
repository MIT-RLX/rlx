// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Warm-all and micro-benchmark helpers for backend ranking.

use rlx_driver::Device;
use rlx_ir::Tick;

use crate::device_parse::device_label;
use crate::graph_devices::GraphDevices;

/// One backend timed after compile + repeated execution.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceBenchResult {
    pub device: Device,
    pub label: &'static str,
    pub compile_ns: u64,
    pub median_exec_ns: u64,
}

/// Compile every viable backend once.
pub fn warm_all(runner: &mut GraphDevices) -> Result<Vec<Device>, String> {
    let devices: Vec<Device> = runner.devices().to_vec();
    for device in &devices {
        runner.compile(*device)?;
    }
    Ok(devices)
}

/// Compile (if needed) and time each backend; returns results sorted fastest first.
pub fn benchmark_devices(
    runner: &mut GraphDevices,
    inputs: &[(&str, &[f32])],
    runs: usize,
) -> Result<Vec<DeviceBenchResult>, String> {
    let mut results = Vec::new();
    let devices: Vec<Device> = runner.devices().to_vec();
    for device in devices {
        let t0 = Tick::now();
        runner.compile(device)?;
        let compile_ns = Tick::now().elapsed_ns(t0);
        let mut samples = Vec::with_capacity(runs.max(1));
        for _ in 0..runs.max(1) {
            let t1 = Tick::now();
            runner.run(device, inputs)?;
            samples.push(Tick::now().elapsed_ns(t1));
        }
        samples.sort_unstable();
        let median_exec_ns = samples[samples.len() / 2];
        results.push(DeviceBenchResult {
            device,
            label: device_label(device),
            compile_ns,
            median_exec_ns,
        });
    }
    results.sort_by_key(|r| r.median_exec_ns);
    Ok(results)
}
