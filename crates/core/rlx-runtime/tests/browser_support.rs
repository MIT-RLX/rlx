// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! Browser support preflight and device selection (native CI).

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{
    BrowserSession, Device, browser_block_reason, collective_block_reason,
    passes_browser_preflight, select_browser_device_for_graph, supports_browser_graph,
};

#[test]
fn collective_ops_blocked_in_browser_preflight() {
    let mut g = Graph::new("coll");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::Custom {
            name: "collective.all_reduce".into(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![y]);

    assert!(!passes_browser_preflight(&g));
    assert!(collective_block_reason(&g).is_some());
    assert!(browser_block_reason(&g).is_some());
    assert!(!supports_browser_graph(&g));
    assert!(select_browser_device_for_graph(&g).is_none());
}

#[test]
fn mlp_supported_on_browser_cpu_path() {
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);

    assert!(passes_browser_preflight(&g));
    assert!(supports_browser_graph(&g));
    let device = select_browser_device_for_graph(&g).expect("browser device");
    assert!(matches!(
        device,
        Device::Cpu | Device::WebGpu | Device::OpenGl
    ));

    let compiled = BrowserSession::compile_for_graph(&g).expect("compile");
    assert!(matches!(
        compiled.device(),
        Device::Cpu | Device::WebGpu | Device::OpenGl
    ));
    let mut c = compiled;
    c.set_param("w", &[1.0; 8]);
    let out = c.run(&[("x", &[1.0; 8])]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 4);
}

#[cfg(feature = "opengl")]
#[test]
fn webgl_backend_registers_and_compiles_mlp() {
    let mut g = Graph::new("mlp");
    let x = g.input("x", Shape::new(&[2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 1], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1], DType::F32));
    g.set_outputs(vec![y]);

    let mut compiled = BrowserSession::new(Device::OpenGl)
        .expect("OpenGl available")
        .compile(g)
        .expect("webgl compile");
    compiled.set_param("w", &[1.0, 0.0]);
    let out = compiled.run(&[("x", &[1.0, 2.0])]);
    assert_eq!(out[0].len(), 1);
    assert!((out[0][0] - 1.0).abs() < 1e-5);
}

#[cfg(feature = "webgpu")]
#[test]
fn webgpu_device_registered() {
    assert!(rlx_runtime::registry::registered_devices().contains(&Device::WebGpu));
}
