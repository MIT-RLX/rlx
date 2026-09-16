// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Device::Egpu` seam contract.
//!
//! The eGPU is wired through the runtime — it parses, labels, and resolves a
//! backend — but has no execution path, because device bring-up over the PCIe
//! tunnel is unimplemented. These tests pin the contract that keeps the two
//! facts from drifting: the device is addressable, and it is never dispatchable.
//!
//! Run with `--features egpu` for the registry half; the parsing half holds in
//! every build.

use rlx_driver::Device;
use std::str::FromStr;

#[test]
fn device_token_parses_and_round_trips() {
    // Both parsers accept the device, and the canonical token survives a trip
    // through a `--device` boundary.
    assert_eq!(Device::from_str("egpu").unwrap(), Device::Egpu);
    assert_eq!(Device::from_str("EGPU").unwrap(), Device::Egpu);
    assert_eq!(Device::from_str("thunderbolt").unwrap(), Device::Egpu);
    assert_eq!(
        rlx_runtime::device_parse::parse_device("tbgpu").unwrap(),
        Device::Egpu
    );
    assert_eq!(Device::Egpu.as_arg(), "egpu");
    assert_eq!(
        Device::from_str(Device::Egpu.as_arg()).unwrap(),
        Device::Egpu
    );
    assert_eq!(
        rlx_runtime::device_parse::device_label(Device::Egpu),
        "egpu"
    );
    assert!(Device::all().contains(&Device::Egpu));
}

#[test]
fn the_device_is_never_dispatchable_without_bring_up() {
    // No bring-up path exists, so the eGPU must not appear as a runnable
    // backend regardless of what is plugged into the tunnel. If this ever
    // fails, either bring-up landed (and this test should assert the new
    // contract) or something started reporting a capability it does not have.
    assert!(!rlx_runtime::is_available(Device::Egpu));
    assert!(!rlx_runtime::available_devices().contains(&Device::Egpu));

    // It must also never win device selection for an ordinary graph.
    let graph = identity_graph();
    assert!(!rlx_runtime::devices_for(&graph).contains(&Device::Egpu));
}

#[cfg(feature = "egpu")]
#[test]
fn the_backend_resolves_but_claims_no_ops() {
    // The seam registers, so a dispatch report can name the device and say why
    // it is idle instead of the device silently not existing.
    let backend = rlx_runtime::backend_for(Device::Egpu);
    assert!(backend.is_some(), "the eGPU backend factory must resolve");
    assert!(
        backend.unwrap().supported_ops().is_empty(),
        "the eGPU backend must not claim ops it cannot lower"
    );
    assert!(rlx_runtime::feature_compiled(Device::Egpu));
}

#[cfg(feature = "egpu")]
#[test]
fn discovery_and_availability_stay_separate() {
    // Hardware presence is an inventory signal, not an execution claim: a card
    // on the tunnel is reported by `detected_unavailable_devices` with a
    // diagnostic, and never by `available_devices`.
    let detected = rlx_runtime::detected_unavailable_devices();
    let listed = detected.iter().any(|(d, _)| *d == Device::Egpu);
    assert_eq!(
        listed,
        rlx_egpu::hardware_present(),
        "a discovered eGPU must appear in the detected-but-idle inventory"
    );
    for (device, diagnostic) in detected {
        if device == Device::Egpu {
            assert!(diagnostic.starts_with("eGPU:"));
        }
    }
}

/// Minimal graph the CPU backend can run — used to check device selection never
/// routes to the eGPU.
fn identity_graph() -> rlx_ir::Graph {
    use rlx_ir::{DType, Graph, Op, Shape};
    let mut graph = Graph::new("egpu_seam");
    let x = graph.add_node(
        Op::Input {
            name: "x".to_string(),
        },
        vec![],
        Shape::new(&[4], DType::F32),
    );
    let y = graph.add_node(
        Op::Activation(rlx_ir::op::Activation::Relu),
        vec![x],
        Shape::new(&[4], DType::F32),
    );
    graph.outputs = vec![y];
    graph
}
