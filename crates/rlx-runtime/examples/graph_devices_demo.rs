// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Multi-backend pick, resolve, and fallback chain.
//!
//! ```sh
//! cargo run --example graph_devices_demo -p rlx-runtime --features cpu
//! RLX_DEVICE_CHAIN=cpu cargo run --example graph_devices_demo -p rlx-runtime --features cpu,metal
//! ```

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, DevicePolicy, DeviceRouter, GraphDevices};

fn identity_graph() -> Graph {
    let mut g = Graph::new("identity");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    g.set_outputs(vec![x]);
    g
}

fn main() -> Result<(), String> {
    let g = identity_graph();
    let inputs = [("x", [1.0_f32, 2.0, 3.0, 4.0].as_slice())];

    let policy = DevicePolicy::only([Device::Cpu]);
    let mut runner = GraphDevices::with_policy(g.clone(), policy.clone());

    println!("devices: {:?}", runner.devices());
    println!("fastest: {:?}", runner.fastest());
    for row in runner.report() {
        println!(
            "  {} available={} supports={} recommended={}",
            row.label, row.available, row.supports_graph, row.recommended
        );
    }

    let out = runner.run(Device::Cpu, &inputs)?;
    println!("run(cpu): {:?}", out);

    let out = runner.run_resolved_with_inputs(None, &inputs)?;
    println!("run_resolved: {:?}", out);

    let (device, out) = runner.run_chain(None, &inputs).map_err(|e| e.to_string())?;
    println!("run_chain on {device:?}: {:?}", out);

    let (device, out) = runner
        .run_try(&[Device::Cpu], &inputs)
        .map_err(|e| e.to_string())?;
    println!("run_try on {device:?}: {:?}", out);

    let mut router = DeviceRouter::new(g, policy)?;
    let (device, out) = router.run(&inputs, None)?;
    println!("router.run on {device:?}: {:?}", out);

    Ok(())
}
