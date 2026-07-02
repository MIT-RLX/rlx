//! In-graph RNG ops: compile, execute, and runtime policy override.

use rlx_ir::{DType, Graph, Op, RngOptions, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn rng_normal_graph(seed_key: u64) -> Graph {
    let mut g = Graph::new("rng_normal");
    let template = g.input("template", Shape::new(&[2, 3], DType::F32));
    let out = g.add_node(
        Op::RngNormal {
            mean: 0.1,
            scale: 2.0,
            key: seed_key,
            op_seed: Some(7.0),
        },
        vec![template],
        Shape::new(&[2, 3], DType::F32),
    );
    g.set_outputs(vec![out]);
    g
}

#[test]
fn rng_normal_philox_is_deterministic() {
    let g = rng_normal_graph(1);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}

#[test]
fn rng_zero_backend_matches_template_shape() {
    let g = rng_normal_graph(2);
    let opts = CompileOptions::new().rng(RngOptions::zero());
    let mut exe = Session::new(Device::Cpu).compile_with(g, &opts);
    let template = vec![1f32; 6];
    let out = exe.run(&[("template", &template)]).remove(0);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn set_rng_changes_output_without_recompile() {
    let g = rng_normal_graph(3);
    let opts = CompileOptions::new().rng(RngOptions::philox(1));
    let mut exe = Session::new(Device::Cpu).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let philox = exe.run(&[("template", &template)]).remove(0);
    exe.set_rng(RngOptions::zero());
    let zero = exe.run(&[("template", &template)]).remove(0);
    assert!(zero.iter().all(|&v| v == 0.0));
    exe.set_rng(RngOptions::philox(1));
    let philox_again = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(philox, philox_again);
    assert_ne!(philox, zero);
}

#[test]
fn rng_backend_switch_via_compile_options() {
    let g = rng_normal_graph(4);
    let template = vec![0f32; 6];
    let mut ort = Session::new(Device::Cpu)
        .compile_with(g.clone(), &CompileOptions::new().rng(RngOptions::ort(7)));
    let mut philox = Session::new(Device::Cpu)
        .compile_with(g, &CompileOptions::new().rng(RngOptions::philox(7)));
    let a = ort.run(&[("template", &template)]).remove(0);
    let b = philox.run(&[("template", &template)]).remove(0);
    assert_ne!(a, b);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn rng_normal_philox_is_deterministic_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let g = rng_normal_graph(5);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Metal).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}

#[cfg(feature = "gpu")]
#[test]
fn rng_normal_philox_is_deterministic_wgpu() {
    if !rlx_runtime::is_available(Device::Gpu) {
        return;
    }
    let g = rng_normal_graph(7);
    let opts = CompileOptions::new().rng(RngOptions::philox(99));
    let mut exe = Session::new(Device::Gpu).compile_with(g, &opts);
    let template = vec![0f32; 6];
    let a = exe.run(&[("template", &template)]).remove(0);
    let b = exe.run(&[("template", &template)]).remove(0);
    assert_eq!(a, b);
    assert_ne!(a, template);
}
