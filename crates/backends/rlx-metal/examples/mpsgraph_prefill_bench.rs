// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prefill-scale MPSGraph compile/run bench: N stacked `FusedAttentionBlock`s.
//!
//! A single fused block does not answer whether making MPSGraph compilation
//! synchronous (`waitForCompilationCompletion = YES`, the fix for the ~2%
//! `MPSGraphExecutable` init crash) costs anything once a real prefill compiles
//! *dozens* of blocks back to back. This stacks `--layers` of them at qwen3-ish
//! dims and reports compile and run wall time separately, so a per-graph cost
//! that is invisible at N=1 would show up as a slope in N.
//!
//!   cargo run --release -p rlx-metal --example mpsgraph_prefill_bench -- [layers] [seq]

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
fn main() {
    use rlx_ir::{DType, Graph, Op, Shape};
    use rlx_runtime::{Device, Session};
    use std::time::Instant;

    let a: Vec<String> = std::env::args().collect();
    let layers: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(28);
    let seq: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(256);
    // qwen3-0.6B-ish: 16 heads x 64 = 1024 hidden.
    let (b, nh, dh) = (1usize, 16usize, 64usize);
    let inner = nh * dh;

    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // One graph, `layers` fused blocks chained through the residual stream —
    // the shape a prefill actually compiles.
    let mut g = Graph::new("prefill_bench");
    let mut h = g.input("h", Shape::new(&[b, seq, inner], DType::F32));
    let mask = g.input("mask", Shape::new(&[b, nh, seq, seq], DType::F32));
    for i in 0..layers {
        let qkv_w = g.param(
            format!("qkv_w{i}"),
            Shape::new(&[inner, 3 * inner], DType::F32),
        );
        let out_w = g.param(format!("out_w{i}"), Shape::new(&[inner, inner], DType::F32));
        h = g.add_node(
            Op::FusedAttentionBlock {
                num_heads: nh,
                head_dim: dh,
                has_bias: false,
                has_rope: false,
            },
            vec![h, qkv_w, out_w, mask],
            Shape::new(&[b, seq, inner], DType::F32),
        );
    }
    g.set_outputs(vec![h]);

    let hv = vec![0.01f32; b * seq * inner];
    let mv = vec![0.0f32; b * nh * seq * seq];
    let qkv = vec![0.005f32; inner * 3 * inner];
    let outw = vec![0.005f32; inner * inner];

    let t0 = Instant::now();
    let mut c = Session::new(Device::Metal).compile(g);
    for i in 0..layers {
        c.set_param(&format!("qkv_w{i}"), &qkv);
        c.set_param(&format!("out_w{i}"), &outw);
    }
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

    // First run pays any deferred specialization; time it separately.
    let t1 = Instant::now();
    let first = c.run(&[("h", &hv), ("mask", &mv)]).remove(0);
    let first_ms = t1.elapsed().as_secs_f64() * 1e3;

    let iters = 5;
    let t2 = Instant::now();
    for _ in 0..iters {
        let _ = c.run(&[("h", &hv), ("mask", &mv)]);
    }
    let steady_ms = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;

    println!(
        "layers={layers} seq={seq} hidden={inner}  compile={compile_ms:.1}ms  \
         first_run={first_ms:.1}ms  steady={steady_ms:.1}ms  out[0]={:.5}",
        first.first().copied().unwrap_or(f32::NAN)
    );
}
