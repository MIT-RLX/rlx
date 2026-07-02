// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Fused resize prologue region vs unfused primitive chain.

#![cfg(feature = "cpu")]
#![allow(dead_code)]

use rlx_fusion::pass::Pass;
use rlx_ir::logical_kernel::KernelDispatchPolicy;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::{
    FusionOptions, FusionTarget, fusion_passes_for_supported, run_passes, supported_for_target,
};
#[cfg(any(
    feature = "metal",
    feature = "gpu",
    feature = "cuda",
    feature = "mlx",
    feature = "rocm",
    feature = "tpu",
))]
use rlx_runtime::is_available;
use rlx_runtime::stages::pipeline_for;
use rlx_runtime::{CompileOptions, Device, Session};
use std::sync::Mutex;

static FK_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Serialize FK env overrides; safe under `cargo test` parallelism.
fn with_fk_batch_single_kernel<R>(f: impl FnOnce() -> R) -> R {
    let _g = FK_ENV_LOCK.lock().unwrap();
    rlx_ir::env::set("RLX_FK_BATCH_SINGLE_KERNEL", "1");
    let out = f();
    rlx_ir::env::unset("RLX_FK_BATCH_SINGLE_KERNEL");
    out
}

fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    Shape::new(&[n, c, h, w], DType::F32)
}

fn build_batch_relu_graph() -> Graph {
    let mut g = Graph::new("fk_batch");
    let batch = g.input("batch", nchw(2, 3, 8, 8));
    let n0 = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: 1,
        },
        vec![batch],
        nchw(1, 3, 8, 8),
    );
    let n1 = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 1,
            len: 1,
        },
        vec![batch],
        nchw(1, 3, 8, 8),
    );
    let chain = vec![rlx_ir::op::ChainStep::Activation(
        Activation::Relu,
        rlx_ir::op::ChainOperand::Input(0),
    )];
    let r0 = g.add_node(
        Op::ElementwiseRegion {
            chain: chain.clone(),
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: rlx_ir::RegionPrologue::None,
            prologue_input: 0,
        },
        vec![n0],
        nchw(1, 3, 8, 8),
    );
    let r1 = g.add_node(
        Op::ElementwiseRegion {
            chain,
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: rlx_ir::RegionPrologue::None,
            prologue_input: 0,
        },
        vec![n1],
        nchw(1, 3, 8, 8),
    );
    let out = g.add_node(Op::Concat { axis: 0 }, vec![r0, r1], nchw(2, 3, 8, 8));
    g.set_outputs(vec![out]);
    g
}

fn build_resize_chain_graph() -> Graph {
    let mut g = Graph::new("fk_chain");
    let x = g.input("x", nchw(1, 3, 8, 8));
    let a = g.input("a", nchw(1, 3, 16, 16));
    let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
    let r = g.activation(Activation::Relu, up, nchw(1, 3, 16, 16));
    let s = g.binary(BinaryOp::Add, r, a, nchw(1, 3, 16, 16));
    let out = g.binary(BinaryOp::Mul, s, a, nchw(1, 3, 16, 16));
    g.set_outputs(vec![out]);
    g
}

fn fuse_prologue_chain(g: Graph, target: FusionTarget) -> Graph {
    let opts = FusionOptions {
        unfuse_elementwise_regions: false,
        decompose_fusion_regions: false,
        ..Default::default()
    };
    let passes = fusion_passes_for_supported(supported_for_target(target), opts, target);
    run_passes(g, &passes, false)
}

fn compile_opts_native_batch(target: FusionTarget) -> CompileOptions {
    let mut opts = CompileOptions::new()
        .fusion_target(target)
        .fusion_opts(FusionOptions {
            skip_fusion: true,
            unfuse_elementwise_regions: false,
            decompose_fusion_regions: false,
            native_fk_regions: true,
            ..FusionOptions::default()
        });
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    opts
}

/// Production-style compile: run the full fusion pipeline with native FKL regions kept.
fn compile_opts_session_native_fk(target: FusionTarget) -> CompileOptions {
    let mut opts = CompileOptions::new().fusion_target(target);
    opts.fusion_opts.native_fk_regions = true;
    opts.fusion_opts.decompose_fusion_regions = false;
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    opts
}

fn fuse_native_batch(g: Graph, target: FusionTarget) -> Graph {
    let opts = FusionOptions {
        native_fk_regions: true,
        decompose_fusion_regions: false,
        ..Default::default()
    };
    let passes = fusion_passes_for_supported(supported_for_target(target), opts, target);
    run_passes(g, &passes, false)
}

fn compile_opts_fused(target: FusionTarget) -> CompileOptions {
    let mut opts = CompileOptions::new()
        .fusion_target(target)
        .fusion_opts(FusionOptions {
            skip_fusion: true,
            unfuse_elementwise_regions: false,
            decompose_fusion_regions: false,
            ..FusionOptions::default()
        });
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    opts
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} len");
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max < tol, "{label}: max_abs_diff={max} tol={tol}");
}

fn run_on(device: Device, g: Graph, opts: &CompileOptions, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let mut c = Session::new(device).compile_with(g, opts);
    c.run(inputs).into_iter().next().expect("one output")
}

#[test]
fn fk_prologue_chain_fusion_ir() {
    let g = fuse_prologue_chain(build_resize_chain_graph(), FusionTarget::Metal);
    assert!(
        !g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ResizeNearest2x))
    );
    let region = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        .expect("region");
    assert_eq!(region.inputs.len(), 2);
    if let Op::ElementwiseRegion {
        prologue,
        num_inputs,
        ..
    } = &region.op
    {
        assert_eq!(*prologue, rlx_ir::RegionPrologue::ResizeNearest2x);
        assert_eq!(*num_inputs, 2);
    } else {
        panic!("expected elementwise region");
    }
}

#[cfg(feature = "metal")]
#[test]
fn fk_prologue_chain_matches_primitives_on_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_prologue_chain_matches_primitives_on_metal (unavailable)");
        return;
    }
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..3 * 8 * 8).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a: Vec<f32> = (0..3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = fuse_prologue_chain(g, FusionTarget::Metal);
    let fus_out = run_on(
        Device::Metal,
        fused,
        &compile_opts_fused(FusionTarget::Metal),
        inputs,
    );
    assert_close(&ref_out, &fus_out, 1e-4, "metal fused vs cpu primitives");
}

#[cfg(feature = "metal")]
#[test]
fn fk_prologue_session_pipeline_keeps_region() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_prologue_session_pipeline_keeps_region (unavailable)");
        return;
    }
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..3 * 8 * 8).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a: Vec<f32> = (0..3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut opts = CompileOptions::new().fusion_target(FusionTarget::Metal);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;

    let pipe = pipeline_for(Device::Metal, &opts);
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g.clone(), &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ElementwiseRegion { .. })),
        "session pipeline should retain ElementwiseRegion"
    );
    assert!(
        !fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ResizeNearest2x)),
        "resize should be folded into prologue"
    );

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let mut compiled = Session::new(Device::Metal).compile_with(g, &opts);
    let out = compiled.run(inputs).into_iter().next().expect("one output");
    assert_close(&ref_out, &out, 1e-4, "session fused vs cpu primitives");
}

#[cfg(feature = "gpu")]
#[test]
fn fk_prologue_chain_matches_primitives_on_wgpu() {
    if !is_available(Device::Vulkan) && !is_available(Device::WebGpu) {
        eprintln!("skip fk_prologue_chain_matches_primitives_on_wgpu (unavailable)");
        return;
    }
    let device = if is_available(Device::Vulkan) {
        Device::Vulkan
    } else {
        Device::WebGpu
    };
    let target = FusionTarget::Wgpu;
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..3 * 8 * 8).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a: Vec<f32> = (0..3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = fuse_prologue_chain(g, target);
    let mut opts = compile_opts_fused(target);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    let fus_out = run_on(device, fused, &opts, inputs);
    assert_close(&ref_out, &fus_out, 1e-4, "wgpu fused vs cpu primitives");
}

#[cfg(feature = "cuda")]
#[test]
fn fk_prologue_chain_matches_primitives_on_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_prologue_chain_matches_primitives_on_cuda (unavailable)");
        return;
    }
    let target = FusionTarget::Cuda;
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let a: Vec<f32> = (0..1 * 3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = fuse_prologue_chain(g, target);
    let fus_out = run_on(Device::Cuda, fused, &compile_opts_fused(target), inputs);
    assert_close(&ref_out, &fus_out, 1e-4, "cuda fused vs cpu primitives");
}

#[cfg(feature = "cuda")]
#[test]
fn fk_prologue_session_pipeline_keeps_region_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_prologue_session_pipeline_keeps_region_cuda (unavailable)");
        return;
    }
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let a: Vec<f32> = (0..1 * 3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut opts = CompileOptions::new().fusion_target(FusionTarget::Cuda);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let mut compiled = Session::new(Device::Cuda).compile_with(g, &opts);
    let out = compiled.run(inputs).into_iter().next().expect("one output");
    assert_close(&ref_out, &out, 1e-4, "cuda session fused vs cpu primitives");
}

#[test]
fn fk_batch_region_fusion_ir() {
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let out = run_passes(g, &[&FuseBatchPreprocess], false);
    assert!(
        out.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
}

#[cfg(feature = "metal")]
#[test]
fn fk_batch_region_matches_primitives_on_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_batch_region_matches_primitives_on_metal (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(
        Device::Metal,
        fused,
        &compile_opts_native_batch(FusionTarget::Metal),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "metal batch region vs cpu primitives",
    );
}

#[cfg(feature = "metal")]
#[test]
fn fk_batch_single_launch_matches_primitives_on_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_batch_single_launch_matches_primitives_on_metal (unavailable)");
        return;
    }
    with_fk_batch_single_kernel(|| {
        use rlx_fusion::fk_fusion::FuseBatchPreprocess;
        let g = build_batch_relu_graph();
        let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let inputs = &[("batch", batch.as_slice())];
        let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
        let fused = run_passes(g, &[&FuseBatchPreprocess], false);
        let fus_out = run_on(
            Device::Metal,
            fused,
            &compile_opts_native_batch(FusionTarget::Metal),
            inputs,
        );
        assert_close(
            &ref_out,
            &fus_out,
            1e-4,
            "metal batch single-launch vs cpu primitives",
        );
    });
}

#[cfg(feature = "gpu")]
#[test]
fn fk_batch_region_matches_primitives_on_wgpu() {
    if !is_available(Device::Vulkan) && !is_available(Device::WebGpu) {
        eprintln!("skip fk_batch_region_matches_primitives_on_wgpu (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let device = if is_available(Device::Vulkan) {
        Device::Vulkan
    } else {
        Device::WebGpu
    };
    let target = FusionTarget::Wgpu;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(device, fused, &compile_opts_native_batch(target), inputs);
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "wgpu batch region vs cpu primitives",
    );
}

#[cfg(feature = "gpu")]
#[test]
fn fk_batch_single_launch_matches_primitives_on_wgpu() {
    if !is_available(Device::Vulkan) && !is_available(Device::WebGpu) {
        eprintln!("skip fk_batch_single_launch_matches_primitives_on_wgpu (unavailable)");
        return;
    }
    with_fk_batch_single_kernel(|| {
        use rlx_fusion::fk_fusion::FuseBatchPreprocess;
        let device = if is_available(Device::Vulkan) {
            Device::Vulkan
        } else {
            Device::WebGpu
        };
        let target = FusionTarget::Wgpu;
        let g = build_batch_relu_graph();
        let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let inputs = &[("batch", batch.as_slice())];
        let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
        let fused = run_passes(g, &[&FuseBatchPreprocess], false);
        let fus_out = run_on(device, fused, &compile_opts_native_batch(target), inputs);
        assert_close(
            &ref_out,
            &fus_out,
            1e-4,
            "wgpu batch single-launch vs cpu primitives",
        );
    });
}

#[cfg(feature = "mlx")]
#[test]
fn fk_batch_region_matches_primitives_on_mlx() {
    if !is_available(Device::Mlx) {
        eprintln!("skip fk_batch_region_matches_primitives_on_mlx (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(
        Device::Mlx,
        fused,
        &compile_opts_native_batch(FusionTarget::Mlx),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "mlx batch region vs cpu primitives",
    );
}

#[cfg(feature = "metal")]
#[test]
fn fk_batch_session_pipeline_keeps_native_region() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_batch_session_pipeline_keeps_native_region (unavailable)");
        return;
    }
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];

    let opts = compile_opts_session_native_fk(FusionTarget::Metal);
    let pipe = pipeline_for(Device::Metal, &opts);
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g.clone(), &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "session pipeline should fuse to BatchElementwiseRegion"
    );
    assert!(
        !fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Concat { .. })),
        "concat should be folded into batch region"
    );

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let out = run_on(Device::Metal, g, &opts, inputs);
    assert_close(&ref_out, &out, 1e-4, "metal session native batch vs cpu");
}

#[cfg(feature = "mlx")]
#[test]
fn fk_batch_session_pipeline_keeps_native_region_mlx() {
    if !is_available(Device::Mlx) {
        eprintln!("skip fk_batch_session_pipeline_keeps_native_region_mlx (unavailable)");
        return;
    }
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];

    let opts = compile_opts_session_native_fk(FusionTarget::Mlx);
    let fused_ir = fuse_native_batch(g.clone(), FusionTarget::Mlx);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );

    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let out = run_on(Device::Mlx, g, &opts, inputs);
    assert_close(&ref_out, &out, 1e-4, "mlx session native batch vs cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn fk_batch_single_launch_matches_primitives_on_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_batch_single_launch_matches_primitives_on_cuda (unavailable)");
        return;
    }
    with_fk_batch_single_kernel(|| {
        use rlx_fusion::fk_fusion::FuseBatchPreprocess;
        let g = build_batch_relu_graph();
        let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let inputs = &[("batch", batch.as_slice())];
        let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
        let fused = run_passes(g, &[&FuseBatchPreprocess], false);
        let fus_out = run_on(
            Device::Cuda,
            fused,
            &compile_opts_native_batch(FusionTarget::Cuda),
            inputs,
        );
        assert_close(
            &ref_out,
            &fus_out,
            1e-4,
            "cuda batch single-launch vs cpu primitives",
        );
    });
}

#[cfg(feature = "cuda")]
#[test]
fn fk_batch_region_matches_primitives_on_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_batch_region_matches_primitives_on_cuda (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(
        Device::Cuda,
        fused,
        &compile_opts_native_batch(FusionTarget::Cuda),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "cuda batch region vs cpu primitives",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn fk_batch_session_pipeline_keeps_native_region_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_batch_session_pipeline_keeps_native_region_cuda (unavailable)");
        return;
    }
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let opts = compile_opts_session_native_fk(FusionTarget::Cuda);
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let out = run_on(Device::Cuda, g, &opts, inputs);
    assert_close(&ref_out, &out, 1e-4, "cuda session native batch vs cpu");
}

#[test]
fn fk_batch_session_default_keeps_batch_region() {
    let g = build_batch_relu_graph();
    let pipe = pipeline_for(Device::Metal, &CompileOptions::new());
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g, &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "default Metal session pipeline keeps BatchElementwiseRegion (native_fk_defaults)"
    );
}

#[cfg(feature = "metal")]
#[test]
fn fk_batch_session_default_matches_cpu_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_batch_session_default_matches_cpu_metal (unavailable)");
        return;
    }
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let out = run_on(
        Device::Metal,
        g,
        &CompileOptions::new().fusion_target(FusionTarget::Metal),
        inputs,
    );
    assert_close(
        &ref_out,
        &out,
        1e-4,
        "metal default session batch region vs cpu primitives",
    );
}

#[test]
fn fk_batch_session_default_keeps_batch_region_tpu() {
    let g = build_batch_relu_graph();
    let pipe = pipeline_for(Device::Tpu, &CompileOptions::new());
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g, &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "default TPU session pipeline keeps BatchElementwiseRegion (native_fk_defaults)"
    );
}

#[test]
fn fk_batch_session_pipeline_keeps_native_region_tpu_ir() {
    let g = build_batch_relu_graph();
    let opts = compile_opts_session_native_fk(FusionTarget::Tpu);
    let pipe = pipeline_for(Device::Tpu, &opts);
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g, &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "TPU session pipeline should keep BatchElementwiseRegion when native_fk_regions"
    );
}

#[test]
fn fk_primitive_batch_session_keeps_native_region_tpu_ir() {
    use rlx_fusion::fk_graphs::batch_narrow_relu_primitive_graph;
    let g = batch_narrow_relu_primitive_graph("tpu_prim", 2, 3, 4, 4);
    let opts = compile_opts_session_native_fk(FusionTarget::Tpu);
    let pipe = pipeline_for(Device::Tpu, &opts);
    let passes = fusion_passes_for_supported(
        pipe.supported_ops
            .unwrap_or_else(|| supported_for_target(pipe.target)),
        pipe.opts,
        pipe.target,
    );
    let fused_ir = run_passes(g, &passes, false);
    assert!(
        fused_ir
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "TPU session should fuse primitive narrow+relu+concat to BatchElementwiseRegion"
    );
}

#[cfg(all(feature = "cpu", feature = "tpu"))]
#[test]
fn fk_batch_region_matches_primitives_on_tpu() {
    if !is_available(Device::Tpu) {
        eprintln!("skip fk_batch_region_matches_primitives_on_tpu (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(
        Device::Tpu,
        fused,
        &compile_opts_native_batch(FusionTarget::Tpu),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "tpu batch region vs cpu primitives",
    );
}

#[cfg(feature = "gpu")]
#[test]
fn fk_batch_session_pipeline_keeps_native_region_wgpu() {
    if !is_available(Device::Vulkan) && !is_available(Device::WebGpu) {
        eprintln!("skip fk_batch_session_pipeline_keeps_native_region_wgpu (unavailable)");
        return;
    }
    let device = if is_available(Device::Vulkan) {
        Device::Vulkan
    } else {
        Device::WebGpu
    };
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let opts = compile_opts_session_native_fk(FusionTarget::Wgpu);
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let out = run_on(device, g, &opts, inputs);
    assert_close(&ref_out, &out, 1e-4, "wgpu session native batch vs cpu");
}

fn build_resize_prologue_secondary_input_graph() -> Graph {
    use rlx_fusion::fk_fusion::FuseRegionPrologue;
    use rlx_fusion::fusion::MarkElementwiseRegions;
    let mut g = Graph::new("fk_prologue_slot1");
    let x = g.input("x", nchw(1, 3, 8, 8));
    let a = g.input("a", nchw(1, 3, 16, 16));
    let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
    let r = g.add_node(
        Op::Activation(Activation::Relu),
        vec![up],
        nchw(1, 3, 16, 16),
    );
    let out = g.add_node(Op::Binary(BinaryOp::Add), vec![a, r], nchw(1, 3, 16, 16));
    g.set_outputs(vec![out]);
    let g = MarkElementwiseRegions.run(g);
    FuseRegionPrologue.run(g)
}

#[cfg(feature = "metal")]
#[test]
fn fk_prologue_resize_on_input_one_matches_cpu() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_prologue_resize_on_input_one_matches_cpu (unavailable)");
        return;
    }
    let g = build_resize_prologue_secondary_input_graph();
    let x: Vec<f32> = (0..3 * 8 * 8).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a: Vec<f32> = (0..3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut g_ref = Graph::new("fk_ref");
    let xr = g_ref.input("x", nchw(1, 3, 8, 8));
    let ar = g_ref.input("a", nchw(1, 3, 16, 16));
    let up = g_ref.add_node(Op::ResizeNearest2x, vec![xr], nchw(1, 3, 16, 16));
    let r = g_ref.activation(Activation::Relu, up, nchw(1, 3, 16, 16));
    let out = g_ref.binary(BinaryOp::Add, ar, r, nchw(1, 3, 16, 16));
    g_ref.set_outputs(vec![out]);

    let ref_out = run_on(Device::Cpu, g_ref, &CompileOptions::new(), inputs);
    let mut opts = compile_opts_fused(FusionTarget::Metal);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    let fus_out = run_on(Device::Metal, g, &opts, inputs);
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "metal prologue slot1 vs cpu primitives",
    );
}

#[cfg(feature = "gpu")]
#[test]
fn fk_prologue_resize_on_input_one_matches_cpu_wgpu() {
    if !is_available(Device::Vulkan) && !is_available(Device::WebGpu) {
        eprintln!("skip fk_prologue_resize_on_input_one_matches_cpu_wgpu (unavailable)");
        return;
    }
    let device = if is_available(Device::Vulkan) {
        Device::Vulkan
    } else {
        Device::WebGpu
    };
    let g = build_resize_prologue_secondary_input_graph();
    let x: Vec<f32> = (0..3 * 8 * 8).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let a: Vec<f32> = (0..3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut g_ref = Graph::new("fk_ref");
    let xr = g_ref.input("x", nchw(1, 3, 8, 8));
    let ar = g_ref.input("a", nchw(1, 3, 16, 16));
    let up = g_ref.add_node(Op::ResizeNearest2x, vec![xr], nchw(1, 3, 16, 16));
    let r = g_ref.add_node(
        Op::Activation(Activation::Relu),
        vec![up],
        nchw(1, 3, 16, 16),
    );
    let out = g_ref.add_node(Op::Binary(BinaryOp::Add), vec![ar, r], nchw(1, 3, 16, 16));
    g_ref.set_outputs(vec![out]);

    let ref_out = run_on(Device::Cpu, g_ref, &CompileOptions::new(), inputs);
    let mut opts = compile_opts_fused(FusionTarget::Wgpu);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    let fus_out = run_on(device, g, &opts, inputs);
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "wgpu prologue slot1 vs cpu primitives",
    );
}

#[cfg(feature = "cuda")]
#[test]
fn fk_prologue_resize_on_input_one_matches_cpu_cuda() {
    if !is_available(Device::Cuda) {
        eprintln!("skip fk_prologue_resize_on_input_one_matches_cpu_cuda (unavailable)");
        return;
    }
    let g = build_resize_prologue_secondary_input_graph();
    let x: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let a: Vec<f32> = (0..1 * 3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut g_ref = Graph::new("fk_ref");
    let xr = g_ref.input("x", nchw(1, 3, 8, 8));
    let ar = g_ref.input("a", nchw(1, 3, 16, 16));
    let up = g_ref.add_node(Op::ResizeNearest2x, vec![xr], nchw(1, 3, 16, 16));
    let r = g_ref.add_node(
        Op::Activation(Activation::Relu),
        vec![up],
        nchw(1, 3, 16, 16),
    );
    let out = g_ref.add_node(Op::Binary(BinaryOp::Add), vec![ar, r], nchw(1, 3, 16, 16));
    g_ref.set_outputs(vec![out]);

    let ref_out = run_on(Device::Cpu, g_ref, &CompileOptions::new(), inputs);
    let mut opts = compile_opts_fused(FusionTarget::Cuda);
    opts.kernel_dispatch.policy = KernelDispatchPolicy::ForceNative;
    let fus_out = run_on(Device::Cuda, g, &opts, inputs);
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "cuda prologue slot1 vs cpu primitives",
    );
}

#[cfg(feature = "mlx")]
#[test]
fn fk_prologue_resize_on_input_one_matches_cpu_mlx() {
    if !is_available(Device::Mlx) {
        eprintln!("skip fk_prologue_resize_on_input_one_matches_cpu_mlx (unavailable)");
        return;
    }
    let g = build_resize_prologue_secondary_input_graph();
    let x: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let a: Vec<f32> = (0..1 * 3 * 16 * 16).map(|i| (i as f32) * 0.001).collect();
    let inputs = &[("x", x.as_slice()), ("a", a.as_slice())];

    let mut g_ref = Graph::new("fk_ref");
    let xr = g_ref.input("x", nchw(1, 3, 8, 8));
    let ar = g_ref.input("a", nchw(1, 3, 16, 16));
    let up = g_ref.add_node(Op::ResizeNearest2x, vec![xr], nchw(1, 3, 16, 16));
    let r = g_ref.add_node(
        Op::Activation(Activation::Relu),
        vec![up],
        nchw(1, 3, 16, 16),
    );
    let out = g_ref.add_node(Op::Binary(BinaryOp::Add), vec![ar, r], nchw(1, 3, 16, 16));
    g_ref.set_outputs(vec![out]);

    let ref_out = run_on(Device::Cpu, g_ref, &CompileOptions::new(), inputs);
    let fus_out = run_on(
        Device::Mlx,
        g,
        &compile_opts_fused(FusionTarget::Mlx),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "mlx prologue slot1 vs cpu primitives",
    );
}

#[test]
fn fk_batch_fusion_four_slices_ir() {
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let mut g = Graph::new("fk_batch4");
    let batch = g.input("batch", nchw(4, 3, 8, 8));
    let chain = vec![rlx_ir::op::ChainStep::Activation(
        Activation::Relu,
        rlx_ir::op::ChainOperand::Input(0),
    )];
    let mut slices = Vec::new();
    for i in 0..4 {
        let sl = g.add_node(
            Op::Narrow {
                axis: 0,
                start: i,
                len: 1,
            },
            vec![batch],
            nchw(1, 3, 8, 8),
        );
        slices.push(g.add_node(
            Op::ElementwiseRegion {
                chain: chain.clone(),
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: rlx_ir::RegionPrologue::None,
                prologue_input: 0,
            },
            vec![sl],
            nchw(1, 3, 8, 8),
        ));
    }
    let out = g.add_node(Op::Concat { axis: 0 }, slices, nchw(4, 3, 8, 8));
    g.set_outputs(vec![out]);
    let fused = FuseBatchPreprocess.run(g);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
}

#[test]
fn fk_native_batch_fusion_ir() {
    let g = fuse_native_batch(build_batch_relu_graph(), FusionTarget::Metal);
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
    assert!(!g.nodes().iter().any(|n| matches!(n.op, Op::Concat { .. })));
}

#[test]
fn fused_prologue_region_unfused_before_autodiff() {
    use rlx_autodiff::prepare_graph_for_ad;
    let fused = fuse_prologue_chain(build_resize_chain_graph(), FusionTarget::Cpu);
    assert!(
        fused.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::ElementwiseRegion {
                    prologue: rlx_ir::RegionPrologue::ResizeNearest2x,
                    ..
                }
            )
        }),
        "expected fused prologue region before AD prep"
    );
    let prep = prepare_graph_for_ad(fused);
    assert!(
        !prep.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::ElementwiseRegion {
                    prologue: rlx_ir::RegionPrologue::ResizeNearest2x,
                    ..
                }
            )
        }),
        "autodiff prep should decompose prologue regions to primitives"
    );
}

#[test]
fn fused_batch_region_decomposed_before_autodiff() {
    use rlx_autodiff::prepare_graph_for_ad;
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = run_passes(build_batch_relu_graph(), &[&FuseBatchPreprocess], false);
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
    let prep = prepare_graph_for_ad(g);
    assert!(
        !prep
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. })),
        "autodiff prep should decompose BatchElementwiseRegion"
    );
}

#[cfg(feature = "metal")]
#[test]
fn fk_primitive_batch_session_matches_cpu_metal() {
    if !is_available(Device::Metal) {
        eprintln!("skip fk_primitive_batch_session_matches_cpu_metal (unavailable)");
        return;
    }
    use rlx_fusion::fk_graphs::batch_narrow_relu_primitive_graph;
    let g = batch_narrow_relu_primitive_graph("fk_prim_sess", 2, 3, 8, 8);
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let opts = compile_opts_session_native_fk(FusionTarget::Metal);
    let out = run_on(Device::Metal, g, &opts, inputs);
    assert_close(&ref_out, &out, 1e-4, "metal primitive batch session vs cpu");
}

#[cfg(feature = "rocm")]
#[test]
fn fk_batch_single_launch_matches_primitives_on_rocm() {
    if !is_available(Device::Rocm) {
        eprintln!("skip fk_batch_single_launch_matches_primitives_on_rocm (unavailable)");
        return;
    }
    with_fk_batch_single_kernel(|| {
        use rlx_fusion::fk_fusion::FuseBatchPreprocess;
        let g = build_batch_relu_graph();
        let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let inputs = &[("batch", batch.as_slice())];
        let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
        let fused = run_passes(g, &[&FuseBatchPreprocess], false);
        let fus_out = run_on(
            Device::Rocm,
            fused,
            &compile_opts_native_batch(FusionTarget::Rocm),
            inputs,
        );
        assert_close(
            &ref_out,
            &fus_out,
            1e-4,
            "rocm batch single-launch vs cpu primitives",
        );
    });
}

#[test]
fn fk_primitive_batch_fuses_via_mark_batch_slice() {
    use rlx_fusion::fk_graphs::batch_narrow_relu_primitive_graph;
    let g = batch_narrow_relu_primitive_graph("fk_prim", 2, 3, 8, 8);
    let fused = fuse_native_batch(g, FusionTarget::Metal);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
    );
}

#[cfg(feature = "rocm")]
#[test]
fn fk_batch_region_matches_primitives_on_rocm() {
    if !is_available(Device::Rocm) {
        eprintln!("skip fk_batch_region_matches_primitives_on_rocm (unavailable)");
        return;
    }
    use rlx_fusion::fk_fusion::FuseBatchPreprocess;
    let g = build_batch_relu_graph();
    let batch: Vec<f32> = (0..2 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.3)
        .collect();
    let inputs = &[("batch", batch.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fused = run_passes(g, &[&FuseBatchPreprocess], false);
    let fus_out = run_on(
        Device::Rocm,
        fused,
        &compile_opts_native_batch(FusionTarget::Rocm),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "rocm batch region vs cpu primitives",
    );
}

#[cfg(feature = "rocm")]
#[test]
fn fk_prologue_resize_relu_matches_cpu_rocm() {
    if !is_available(Device::Rocm) {
        eprintln!("skip fk_prologue_resize_relu_matches_cpu_rocm (unavailable)");
        return;
    }
    let g = build_resize_chain_graph();
    let x: Vec<f32> = (0..1 * 3 * 8 * 8)
        .map(|i| (i as f32) * 0.01 - 0.5)
        .collect();
    let inputs = &[("x", x.as_slice())];
    let ref_out = run_on(Device::Cpu, g.clone(), &CompileOptions::new(), inputs);
    let fus_out = run_on(
        Device::Rocm,
        g,
        &compile_opts_fused(FusionTarget::Rocm),
        inputs,
    );
    assert_close(
        &ref_out,
        &fus_out,
        1e-4,
        "rocm prologue region vs cpu primitives",
    );
}
