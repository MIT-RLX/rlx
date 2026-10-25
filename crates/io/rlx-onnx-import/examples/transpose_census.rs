// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Transpose census: how many transposes an imported model has, and how many
// `SimplifyTransposes` removes.
//
//   cargo run -q -p rlx-onnx-import --example transpose_census -- [--run] [--device D] a.onnx …
//
// `--run` also executes each model before and after the pass (imported
// weights, deterministic inputs) and reports the largest output difference,
// the median time of each, and the median per-round after/before ratio.
// Runs alternate between the two graphs (`CENSUS_ROUNDS`, default 21 rounds),
// so a loaded machine slows both alike. Run `just throttle` first.

use std::collections::HashMap;
use std::path::PathBuf;

use rlx_compile::SimplifyTransposes;
use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, OpKind};
use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

fn transposes(g: &Graph) -> usize {
    g.nodes()
        .iter()
        .filter(|n| n.op.kind() == OpKind::Transpose)
        .count()
}

type Params = HashMap<String, Vec<f32>>;

fn import(path: &std::path::Path, seq: usize) -> Result<(Graph, Params), String> {
    let mut named = HashMap::new();
    for k in [
        "sequence_length",
        "total_sequence_length",
        "encoder_sequence_length",
    ] {
        named.insert(k.to_string(), seq);
    }
    named.insert("batch_size".to_string(), 1);
    named.insert("past_sequence_length".to_string(), 0);
    let opts = ImportOptions {
        sequence_length: seq,
        named_lengths: named,
        strict: false,
        ..Default::default()
    };
    let (hir, params, ..) = build_hir_from_onnx_file(path, opts).map_err(|e| e.to_string())?;
    let g = rlx_ir::hir_to_graph(hir).map_err(|e| e.to_string())?;
    Ok((g, params))
}

type Feed = Vec<(String, Vec<f32>)>;

/// `g` compiled on `dev` with its params set, and deterministic inputs.
fn prepare(
    dev: rlx_runtime::Device,
    g: Graph,
    params: &Params,
) -> (rlx_runtime::CompiledGraph, Feed) {
    let inputs: Feed = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            rlx_ir::Op::Input { name } => {
                let len = n.shape.num_elements().unwrap_or(0);
                let float = n.shape.dtype().is_float();
                let v = (0..len)
                    .map(|i| {
                        if float {
                            ((i as f32) * 0.37).sin() * 0.5
                        } else {
                            0.0
                        }
                    })
                    .collect();
                Some((name.clone(), v))
            }
            _ => None,
        })
        .collect();
    let mut c = rlx_runtime::Session::new(dev).compile(g);
    for (k, v) in params {
        c.set_param(k, v);
    }
    (c, inputs)
}

fn run(c: &mut rlx_runtime::CompiledGraph, inputs: &Feed) -> Vec<Vec<f32>> {
    let feed: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_slice()))
        .collect();
    c.run(&feed)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Outputs of both graphs, and the median time (ms) of each plus the median
/// per-round ratio after/before. Runs alternate between the two so background
/// load lands on both sides alike.
fn run_pair(
    dev: rlx_runtime::Device,
    a: (Graph, &Params),
    b: (Graph, &Params),
    rounds: usize,
) -> ([Vec<Vec<f32>>; 2], f64, f64, f64) {
    let (mut ca, fa) = prepare(dev, a.0, a.1);
    let (mut cb, fb) = prepare(dev, b.0, b.1);
    let oa = run(&mut ca, &fa);
    let ob = run(&mut cb, &fb);
    let (mut ta, mut tb, mut ratio) = (vec![], vec![], vec![]);
    for r in 0..rounds {
        let time = |c: &mut rlx_runtime::CompiledGraph, f: &Feed| {
            let t0 = rlx_ir::Tick::now();
            let _ = run(c, f);
            rlx_ir::Tick::now().elapsed_ns(t0) as f64 / 1e6
        };
        // Alternate which goes first, so neither always runs warm.
        let (x, y) = if r % 2 == 0 {
            let x = time(&mut ca, &fa);
            (x, time(&mut cb, &fb))
        } else {
            let y = time(&mut cb, &fb);
            (time(&mut ca, &fa), y)
        };
        ta.push(x);
        tb.push(y);
        ratio.push(y / x);
    }
    ([oa, ob], median(ta), median(tb), median(ratio))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run = args.iter().any(|a| a == "--run");
    let dev: rlx_runtime::Device = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .map(|d| d.parse().unwrap_or_else(|_| panic!("unknown device {d}")))
        .unwrap_or(rlx_runtime::Device::Cpu);
    let dev_arg = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .cloned();
    print!(
        "{:<44} {:>7} {:>10} {:>9}",
        "model", "nodes", "transposes", "after"
    );
    if run {
        print!(
            " {:>10} {:>10} {:>7} {:>10}",
            "ms before", "ms after", "ratio", "max |Δ|"
        );
    }
    println!();
    let files = args
        .iter()
        .filter(|a| *a != "--run" && *a != "--device" && Some(*a) != dev_arg.as_ref());
    for path in files.map(PathBuf::from) {
        let name = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let parent = path
            .parent()
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let label = format!("{parent}/{name}");
        // Importers fix symbolic lengths; models disagree on which they accept.
        std::panic::set_hook(Box::new(|_| {}));
        let mut graph = None;
        let mut last_err = String::new();
        for seq in [16usize, 128, 1] {
            let p = path.clone();
            let r = std::panic::catch_unwind(move || import(&p, seq));
            match r {
                Ok(Ok(g)) => {
                    graph = Some(g);
                    break;
                }
                Ok(Err(e)) => last_err = e,
                Err(p) => {
                    last_err = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default()
                }
            }
        }
        let _ = std::panic::take_hook();
        let Some((graph, params)) = graph else {
            println!(
                "{label:<44} import failed: {}",
                last_err.lines().next().unwrap_or("")
            );
            continue;
        };
        let before = transposes(&graph);
        let n = graph.len();
        let simplified = SimplifyTransposes.run(graph.clone());
        let after = transposes(&simplified);
        print!("{label:<44} {n:>7} {before:>10} {after:>9}");
        if run && after < before {
            let rounds = std::env::var("CENSUS_ROUNDS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(21);
            let ([a, b], ta, tb, ratio) =
                run_pair(dev, (graph, &params), (simplified, &params), rounds);
            let diff = a
                .iter()
                .zip(&b)
                .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
                .fold(0f32, f32::max);
            print!(" {ta:>10.2} {tb:>10.2} {ratio:>7.3} {diff:>10.2e}");
        }
        println!();
    }
}
