// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// RLX — versatile ML compiler + runtime.
//
// 2nd-order AD through decomposed backward ops — compile + execute timing.
//
// ```sh
// cargo run -p rlx-bench --release --example bench_higher_order_decompose --features cuda,gpu
// ./rig.sh bench-higher-order-decompose both
// ```

use rlx_driver::Device;
use rlx_ir::dynamic::{bind_graph, sym};
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::shape::Dim;
use rlx_ir::{DType, DimBinding, Graph, Op, Shape, Tick};
use rlx_opt::nth_order_grad;
use rlx_runtime::CompileCache;
use std::collections::HashMap;

fn parse_usize(flag: &str, args: &[String], default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median_ns(mut samples: Vec<u64>) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn devices() -> Vec<(&'static str, Device)> {
    let out = vec![("cpu", Device::Cpu)];
    #[cfg(feature = "gpu")]
    if is_available(Device::Gpu) {
        out.push(("wgpu", Device::Gpu));
    }
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        out.push(("cuda", Device::Cuda));
    }
    out
}

fn bind_dynamic_conv(g: Graph, batch: usize) -> Graph {
    let spatial_out = 4 * 4;
    bind_graph(
        &g,
        &DimBinding::from_pairs(&[(sym::BATCH, batch), (sym::ROWS, batch * spatial_out)]),
    )
}

fn build_dynamic_conv_loss() -> Graph {
    let f = DType::F32;
    let nchw = [
        Dim::Dynamic(sym::BATCH),
        Dim::Static(1),
        Dim::Static(4),
        Dim::Static(4),
    ];
    let mut g = Graph::new("dyn_conv_bench");
    let x = g.input("x", Shape::from_dims(&nchw, f));
    let w = g.input("w", Shape::new(&[1, 1, 3, 3], f));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_conv_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("conv_bench");
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
    let w = g.input("w", Shape::new(&[1, 1, 3, 3], f));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn build_scan_loss(length: u32) -> Graph {
    let f = DType::F32;
    let n = 2usize;
    let carry = Shape::new(&[n], f);
    let mut body = Graph::new("scan_body_bench");
    let bc = body.input("carry", carry.clone());
    let bx = body.input("x_t", carry.clone());
    let by = body.binary(BinaryOp::Add, bc, bx, carry.clone());
    body.set_outputs(vec![by]);
    let mut g = Graph::new("scan_bench");
    let init = g.input("init", carry.clone());
    let xs = g.input("xs", Shape::new(&[length as usize, n], f));
    let y = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        Shape::new(&[length as usize, n], f),
    );
    let loss = g.reduce(y, ReduceOp::Sum, vec![0, 1], false, Shape::scalar(f));
    g.set_outputs(vec![loss]);
    g
}

fn gen_x(batch: usize) -> Vec<u8> {
    let n = batch * 16;
    (0..n)
        .flat_map(|i| (0.05 * i as f32 - 0.2).to_le_bytes())
        .collect()
}

fn gen_w() -> Vec<u8> {
    (0..9)
        .flat_map(|i| (0.1 * (i as f32 + 1.0)).to_le_bytes())
        .collect()
}

struct CaseSpec {
    name: &'static str,
    build: fn() -> (Graph, Vec<(&'static str, Vec<u8>, DType)>),
}

fn case_specs() -> Vec<CaseSpec> {
    vec![
        CaseSpec {
            name: "dynamic_conv_w_2nd",
            build: || {
                let hg = bind_dynamic_conv(nth_order_grad(&build_dynamic_conv_loss(), "w", 2), 2);
                (
                    hg,
                    vec![("x", gen_x(2), DType::F32), ("w", gen_w(), DType::F32)],
                )
            },
        },
        CaseSpec {
            name: "dynamic_conv_x_2nd",
            build: || {
                let hg = bind_dynamic_conv(nth_order_grad(&build_dynamic_conv_loss(), "x", 2), 2);
                (
                    hg,
                    vec![("x", gen_x(2), DType::F32), ("w", gen_w(), DType::F32)],
                )
            },
        },
        CaseSpec {
            name: "conv2d_w_2nd",
            build: || {
                let hg = nth_order_grad(&build_conv_loss(), "w", 2);
                (
                    hg,
                    vec![("x", gen_x(1), DType::F32), ("w", gen_w(), DType::F32)],
                )
            },
        },
        CaseSpec {
            name: "scan_xs_2nd",
            build: || {
                let (init_b, xs_b) = gen_scan_inputs(3);
                (
                    nth_order_grad(&build_scan_loss(3), "xs", 2),
                    vec![("init", init_b, DType::F32), ("xs", xs_b, DType::F32)],
                )
            },
        },
        CaseSpec {
            name: "scan_long_xs_2nd",
            build: || {
                let (init_b, xs_b) = gen_scan_inputs(130);
                (
                    nth_order_grad(&build_scan_loss(130), "xs", 2),
                    vec![("init", init_b, DType::F32), ("xs", xs_b, DType::F32)],
                )
            },
        },
    ]
}

fn gen_scan_inputs(length: usize) -> (Vec<u8>, Vec<u8>) {
    let init: Vec<u8> = [0.1_f32, -0.2_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let xs: Vec<u8> = (0..length * 2)
        .map(|i| 0.01 * (i as f32).sin())
        .flat_map(f32::to_le_bytes)
        .collect();
    (init, xs)
}

struct Case {
    name: &'static str,
    hg: Graph,
    inputs: Vec<(&'static str, Vec<u8>, DType)>,
}

fn cases() -> Vec<Case> {
    case_specs()
        .into_iter()
        .map(|spec| {
            let (hg, inputs) = (spec.build)();
            Case {
                name: spec.name,
                hg,
                inputs,
            }
        })
        .collect()
}

fn bench_device(
    cache: &mut CompileCache,
    key: u64,
    hg: &Graph,
    inputs: &[(&str, &[u8], DType)],
    warmup: usize,
    runs: usize,
) -> (u64, u64, u64) {
    let t0 = Tick::now();
    let _ = cache.get_or_compile(key, || hg.clone());
    let compile_ns = Tick::now().elapsed_ns(t0);

    let t0 = Tick::now();
    let compiled = cache.get_or_compile(key, || hg.clone());
    let cache_hit_ns = Tick::now().elapsed_ns(t0);

    for _ in 0..warmup {
        let _ = compiled.run_typed(inputs);
    }

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Tick::now();
        let _ = compiled.run_typed(inputs);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    (compile_ns, cache_hit_ns, median_ns(samples))
}

fn prewarm_cuda() {
    #[cfg(feature = "cuda")]
    if is_available(Device::Cuda) {
        let forward = build_conv_loss();
        let hg = nth_order_grad(&forward, "w", 2);
        let x_b = gen_x(1);
        let w_b = gen_w();
        let inputs = [
            ("x", x_b.as_slice(), DType::F32),
            ("w", w_b.as_slice(), DType::F32),
        ];
        let mut ex = Session::new(Device::Cuda).compile(hg);
        let _ = ex.run_typed(&inputs);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let warmup = parse_usize("--warmup", &args, 3);
    let runs = parse_usize("--runs", &args, 50);
    let devs = devices();

    println!("# higher_order_decompose bench — 2nd-order AD, decomposed backward ops");
    println!(
        "# platform: {} / {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "# devices: {}",
        devs.iter().map(|(l, _)| *l).collect::<Vec<_>>().join(", ")
    );
    println!("# runs={runs}, warmup={warmup}");
    println!("# compile via CompileCache (cache miss + hit columns)");
    if std::env::var("RLX_CUDA_COMPILE_MODE").is_ok_and(|v| v.eq_ignore_ascii_case("aot")) {
        println!("# RLX_CUDA_COMPILE_MODE=aot (NVRTC prewarm at compile)");
    }
    if std::env::var("RLX_CUDA_EXEC_MODE").is_ok_and(|v| v.eq_ignore_ascii_case("graph")) {
        println!("# RLX_CUDA_EXEC_MODE=graph (CUDA Graph replay after 1st run)");
    }
    println!();

    prewarm_cuda();

    let t0 = Tick::now();
    let all_cases = cases();
    println!(
        "# AD graph build (all cases): {:.1} ms\n",
        Tick::now().elapsed_ns(t0) as f64 / 1e6
    );

    let mut caches: HashMap<Device, CompileCache> = devs
        .iter()
        .map(|(_, device)| (*device, CompileCache::new(*device, all_cases.len().max(8))))
        .collect();

    for (case_idx, case) in all_cases.into_iter().enumerate() {
        let input_refs: Vec<(&str, &[u8], DType)> = case
            .inputs
            .iter()
            .map(|(n, b, dt)| (*n, b.as_slice(), *dt))
            .collect();

        println!("## {}\n", case.name);
        println!("| device | compile µs | cache hit µs | exec median µs |");
        println!("|--------|---:|---:|---:|");
        for &(label, device) in &devs {
            let cache = caches.get_mut(&device).expect("compile cache");
            let (compile_ns, cache_hit_ns, exec_ns) =
                bench_device(cache, case_idx as u64, &case.hg, &input_refs, warmup, runs);
            println!(
                "| {label} | {comp:.1} | {hit:.1} | {exec:.1} |",
                comp = compile_ns as f64 / 1e3,
                hit = cache_hit_ns as f64 / 1e3,
                exec = exec_ns as f64 / 1e3,
            );
        }
        println!();
    }
}
