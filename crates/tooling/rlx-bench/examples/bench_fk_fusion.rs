// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Benchmark FKL-style region fusion: separate kernels vs fused prologue region.
//
// Run (release, after thermal gate):
//   RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion
//   RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features metal
//   RLX_FK_BATCH_SINGLE_KERNEL=1 RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features cuda,metal,gpu
//   RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bench --release --example bench_fk_fusion --features tpu

use rlx_driver::Device;
use rlx_fusion::fk_graphs::{
    batch_narrow_relu_primitive_graph, batch_narrow_relu_regions_graph, nchw,
    resize_relu_region_graph,
};
use rlx_ir::infer::GraphExt;
use rlx_ir::{Graph, Op, Shape, Tick};
use rlx_opt::{
    FusionOptions, FusionTarget, fusion_passes_for_supported, run_passes, supported_for_target,
};
#[cfg(any(feature = "metal", feature = "rocm", feature = "tpu"))]
use rlx_runtime::is_available;
use rlx_runtime::stages::pipeline_for;
use rlx_runtime::{CompileOptions, Session};

#[derive(Clone, Copy)]
struct BenchCfg {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

impl BenchCfg {
    fn num_in(&self) -> usize {
        self.n * self.c * self.h * self.w
    }
    fn num_out(&self) -> usize {
        self.n * self.c * self.h * 2 * self.w * 2
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Separate,
    SplitRegion,
    FklPrologue,
    /// Hand-built `ElementwiseRegion` with resize prologue (what FuseRegionPrologue targets).
    FklPrologueDirect,
    ChainSeparate,
    ChainFkl,
    /// Full session pipeline: fusion passes + default unfuse (GPU keeps prologue regions).
    SessionDefault,
    /// Primitive narrow+relu+concat; production `CompileOptions` (keeps `BatchElementwiseRegion` by default on GPU-class + TPU).
    SessionDefaultBatch,
    /// Same primitive batch graph; `native_fk_regions` keeps `BatchElementwiseRegion`.
    SessionNativeBatch,
    /// Primitive narrow+relu+concat; `MarkBatchSliceRegions` + native FKL on GPU.
    SessionPrimitiveNativeBatch,
}

fn bench_nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
    nchw(n, c, h, w)
}

fn build_raw(graph_name: &str, cfg: BenchCfg, variant: Variant) -> Graph {
    let mut g = Graph::new(graph_name);
    let x = g.input("x", bench_nchw(cfg.n, cfg.c, cfg.h, cfg.w));
    let up = g.add_node(
        Op::ResizeNearest2x,
        vec![x],
        bench_nchw(cfg.n, cfg.c, cfg.h * 2, cfg.w * 2),
    );
    let out = match variant {
        Variant::Separate => g.relu(up),
        Variant::ChainSeparate => {
            let r = g.relu(up);
            let a = g.input("a", bench_nchw(cfg.n, cfg.c, cfg.h * 2, cfg.w * 2));
            let s = g.add(r, a);
            g.mul(s, a)
        }
        _ => unreachable!("raw build only for separate variants"),
    };
    g.set_outputs(vec![out]);
    g
}

fn count_ops(g: &Graph) -> String {
    use std::collections::BTreeMap;
    let mut m = BTreeMap::new();
    for n in g.nodes() {
        *m.entry(format!("{:?}", n.op)).or_insert(0usize) += 1;
    }
    m.into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_batch_explicit_regions(cfg: BenchCfg) -> Graph {
    batch_narrow_relu_regions_graph("fk_batch_regions", cfg.n, cfg.c, cfg.h, cfg.w)
}

fn build_batch_primitive(cfg: BenchCfg) -> Graph {
    batch_narrow_relu_primitive_graph("fk_batch_prim", cfg.n, cfg.c, cfg.h, cfg.w)
}

fn build_fkl_prologue_direct(cfg: BenchCfg) -> Graph {
    resize_relu_region_graph("fk_direct", cfg.n, cfg.c, cfg.h, cfg.w)
}

fn build_fused(cfg: BenchCfg, target: FusionTarget, variant: Variant) -> Graph {
    if matches!(variant, Variant::FklPrologueDirect) {
        return build_fkl_prologue_direct(cfg);
    }
    if matches!(
        variant,
        Variant::SessionDefaultBatch | Variant::SessionPrimitiveNativeBatch
    ) {
        return build_batch_primitive(cfg);
    }
    if variant == Variant::SessionNativeBatch {
        return build_batch_explicit_regions(cfg);
    }
    if matches!(
        variant,
        Variant::ChainSeparate | Variant::Separate | Variant::SessionDefault
    ) {
        return build_raw(
            "fk_bench",
            cfg,
            if variant == Variant::SessionDefault {
                Variant::Separate
            } else {
                variant
            },
        );
    }
    let opts = FusionOptions {
        unfuse_elementwise_regions: false,
        decompose_fusion_regions: matches!(variant, Variant::SplitRegion),
        ..Default::default()
    };
    let supported = supported_for_target(target);
    let passes = fusion_passes_for_supported(supported, opts, target);

    if matches!(variant, Variant::ChainFkl) {
        let mut g2 = Graph::new("fk_chain");
        let x = g2.input("x", bench_nchw(cfg.n, cfg.c, cfg.h, cfg.w));
        let up = g2.add_node(
            Op::ResizeNearest2x,
            vec![x],
            bench_nchw(cfg.n, cfg.c, cfg.h * 2, cfg.w * 2),
        );
        let r = g2.relu(up);
        let a = g2.input("a", bench_nchw(cfg.n, cfg.c, cfg.h * 2, cfg.w * 2));
        let s = g2.add(r, a);
        let out = g2.mul(s, a);
        g2.set_outputs(vec![out]);
        return run_passes(g2, &passes, false);
    }

    let g = build_raw("fk_bench", cfg, Variant::Separate);
    run_passes(g, &passes, false)
}

fn compile_opts_for_bench(target: FusionTarget, variant: Variant) -> CompileOptions {
    if matches!(
        variant,
        Variant::SessionNativeBatch | Variant::SessionPrimitiveNativeBatch
    ) {
        let mut opts = CompileOptions::new().fusion_target(target);
        opts.fusion_opts.native_fk_regions = true;
        opts.fusion_opts.decompose_fusion_regions = false;
        return opts;
    }
    if matches!(
        variant,
        Variant::SessionDefault | Variant::SessionDefaultBatch
    ) {
        return CompileOptions::new().fusion_target(target);
    }
    CompileOptions::new()
        .fusion_target(target)
        .fusion_opts(FusionOptions {
            skip_fusion: true,
            unfuse_elementwise_regions: false,
            decompose_fusion_regions: false,
            ..FusionOptions::default()
        })
}

fn time_run(
    device: Device,
    target: FusionTarget,
    variant: Variant,
    graph: Graph,
    inputs: &[(&str, &[f32])],
    warmup: usize,
    runs: usize,
) -> (u64, u64) {
    let mut compiled =
        Session::new(device).compile_with(graph, &compile_opts_for_bench(target, variant));
    for _ in 0..warmup {
        let _ = compiled.run(inputs);
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Tick::now();
        let _ = compiled.run(inputs);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let mean = (samples.iter().map(|&v| v as u128).sum::<u128>() / samples.len() as u128) as u64;
    (median, mean)
}

fn variant_label(v: Variant) -> &'static str {
    match v {
        Variant::Separate => "separate_resize+relu",
        Variant::SplitRegion => "split_resize+region",
        Variant::FklPrologue => "fkl_prologue_region",
        Variant::FklPrologueDirect => "fkl_prologue_direct",
        Variant::ChainSeparate => "separate_resize+chain3",
        Variant::ChainFkl => "fkl_prologue_chain3",
        Variant::SessionDefault => "session_default_pipeline",
        Variant::SessionDefaultBatch => "session_default_batch",
        Variant::SessionNativeBatch => "session_native_batch",
        Variant::SessionPrimitiveNativeBatch => "session_primitive_native_batch",
    }
}

fn count_ops_after_pipeline(device: Device, graph: Graph, opts: &CompileOptions) -> String {
    let pipe = pipeline_for(device, opts);
    let supported = pipe
        .supported_ops
        .unwrap_or_else(|| supported_for_target(pipe.target));
    let passes = fusion_passes_for_supported(supported, pipe.opts, pipe.target);
    count_ops(&run_passes(graph, &passes, false))
}

fn fusion_target(device: Device) -> FusionTarget {
    match device {
        Device::Metal => FusionTarget::Metal,
        Device::Cuda => FusionTarget::Cuda,
        Device::Rocm => FusionTarget::Rocm,
        Device::Gpu | Device::Vulkan | Device::WebGpu => FusionTarget::Wgpu,
        Device::Mlx => FusionTarget::Mlx,
        Device::Tpu => FusionTarget::Tpu,
        _ => FusionTarget::Cpu,
    }
}

fn devices() -> Vec<(&'static str, Device)> {
    [
        Some(("cpu", Device::Cpu)),
        #[cfg(feature = "metal")]
        is_available(Device::Metal).then_some(("metal", Device::Metal)),
        #[cfg(feature = "rocm")]
        is_available(Device::Rocm).then_some(("rocm", Device::Rocm)),
        #[cfg(feature = "tpu")]
        is_available(Device::Tpu).then_some(("tpu", Device::Tpu)),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn fk_bench_ops_device(devs: &[(&'static str, Device)]) -> Device {
    devs.iter()
        .find(|(_, d)| matches!(d, Device::Metal))
        .map(|(_, d)| *d)
        .unwrap_or(devs[0].1)
}

fn print_fk_bench_ops(cfg: BenchCfg, devs: &[(&'static str, Device)]) {
    let device = fk_bench_ops_device(devs);
    let label = devs
        .iter()
        .find(|(_, d)| *d == device)
        .map(|(l, _)| *l)
        .unwrap_or("device");
    let target = fusion_target(device);
    println!(
        "\n=== FK_BENCH_OPS NCHW {}x{}x{}x{} ({label}) ===",
        cfg.n, cfg.c, cfg.h, cfg.w
    );
    for &v in &[
        Variant::Separate,
        Variant::FklPrologue,
        Variant::ChainFkl,
        Variant::SessionDefault,
        Variant::SessionDefaultBatch,
        Variant::SessionNativeBatch,
        Variant::SessionPrimitiveNativeBatch,
    ] {
        if matches!(
            v,
            Variant::SessionDefaultBatch
                | Variant::SessionNativeBatch
                | Variant::SessionPrimitiveNativeBatch
        ) && cfg.n < 2
        {
            continue;
        }
        let g = build_fused(cfg, target, v);
        let ops = match v {
            Variant::SessionDefault
            | Variant::SessionDefaultBatch
            | Variant::SessionNativeBatch
            | Variant::SessionPrimitiveNativeBatch => {
                let opts = compile_opts_for_bench(target, v);
                count_ops_after_pipeline(device, g, &opts)
            }
            _ => count_ops(&g),
        };
        println!("  {:28} {}", variant_label(v), ops);
    }
}

fn main() {
    let warmup = 5;
    let runs = 30;
    let sizes: [BenchCfg; 2] = [
        BenchCfg {
            n: 1,
            c: 64,
            h: 56,
            w: 56,
        },
        BenchCfg {
            n: 2,
            c: 64,
            h: 112,
            w: 112,
        },
    ];

    let devs = devices();
    if std::env::var("FK_BENCH_OPS").ok().as_deref() == Some("1") {
        for cfg in sizes {
            print_fk_bench_ops(cfg, &devs);
        }
        return;
    }

    for cfg in sizes {
        println!(
            "\n=== NCHW {}x{}x{}x{} -> 2x ({} out elems) warmup={warmup} runs={runs} ===",
            cfg.n,
            cfg.c,
            cfg.h,
            cfg.w,
            cfg.num_out()
        );
        println!(
            "Devices: {:?}",
            devs.iter().map(|(l, _)| *l).collect::<Vec<_>>()
        );
        let variants = [
            Variant::Separate,
            Variant::SplitRegion,
            Variant::FklPrologue,
            Variant::FklPrologueDirect,
            Variant::SessionDefault,
            Variant::SessionDefaultBatch,
            Variant::SessionNativeBatch,
            Variant::SessionPrimitiveNativeBatch,
            Variant::ChainSeparate,
            Variant::ChainFkl,
        ];

        for &(label, device) in &devs {
            let target = fusion_target(device);
            println!("## {label} ({target:?})");
            let mut baselines: Vec<(Variant, u64)> = Vec::new();
            for &v in &variants {
                if matches!(
                    v,
                    Variant::SessionDefaultBatch
                        | Variant::SessionNativeBatch
                        | Variant::SessionPrimitiveNativeBatch
                ) && cfg.n < 2
                {
                    println!("  {:28} SKIP (batch dim n={} < 2)", variant_label(v), cfg.n);
                    continue;
                }
                let graph = build_fused(cfg, target, v);
                let mut inputs = vec![("x", vec![0.5f32; cfg.num_in()])];
                if matches!(
                    v,
                    Variant::SessionDefaultBatch | Variant::SessionNativeBatch
                ) {
                    inputs = vec![("batch", vec![0.5f32; cfg.num_in()])];
                }
                if matches!(v, Variant::ChainSeparate | Variant::ChainFkl) {
                    inputs.push(("a", vec![0.25f32; cfg.num_out()]));
                }
                let input_refs: Vec<(&str, &[f32])> =
                    inputs.iter().map(|(n, d)| (*n, d.as_slice())).collect();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    time_run(device, target, v, graph, &input_refs, warmup, runs)
                }));
                match result {
                    Ok((med, mean)) => {
                        let us = med as f64 / 1000.0;
                        println!(
                            "  {:28} median={us:8.2} us  mean={:.2} us",
                            variant_label(v),
                            mean as f64 / 1000.0
                        );
                        baselines.push((v, med));
                    }
                    Err(_) => {
                        println!("  {:28} FAILED", variant_label(v));
                    }
                }
            }
            let baseline = |v: Variant| baselines.iter().find(|(bv, _)| *bv == v).map(|(_, t)| *t);
            if let Some(sep) = baseline(Variant::Separate) {
                for (v, t) in &baselines {
                    if *v == Variant::Separate {
                        continue;
                    }
                    let ref_med = match v {
                        Variant::ChainSeparate | Variant::ChainFkl => {
                            baseline(Variant::ChainSeparate).unwrap_or(sep)
                        }
                        Variant::SessionDefaultBatch | Variant::SessionNativeBatch => {
                            baseline(Variant::SessionDefaultBatch)
                                .or(Some(sep))
                                .unwrap_or(sep)
                        }
                        _ => sep,
                    };
                    let speedup = ref_med as f64 / *t as f64;
                    let pct = (speedup - 1.0) * 100.0;
                    let dir = if speedup > 1.02 {
                        "faster"
                    } else if speedup < 0.98 {
                        "slower"
                    } else {
                        "~same"
                    };
                    let ref_label = if matches!(v, Variant::ChainSeparate | Variant::ChainFkl) {
                        "vs chain sep"
                    } else if matches!(
                        v,
                        Variant::SessionDefaultBatch | Variant::SessionNativeBatch
                    ) {
                        "vs session default batch"
                    } else {
                        "vs resize+relu"
                    };
                    println!(
                        "    {ref_label}: {:28} {dir} ({pct:+.1}%, {speedup:.2}x)",
                        variant_label(*v)
                    );
                }
            }
            println!();
        }
    }
}
