// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-infer` — Tier 1 inference-dynamics mining. Two runnable demos:
//!   * attention: `softmax(scores)` fed sink+local structure → per-query peak
//!     mass + per-key received mass → sparse/windowed-attention viability.
//!   * MoE gate: `topk(tokens · Wg)` with skewed gate → per-expert load → drop
//!     cold / prefetch hot experts.
//! These are the distinctly *inference* signals (they only exist at run time),
//! tapped by `inject_infer_stats` and executed on CPU.

use rlx_ir::{DType, Graph, Op, Philox4x32, Shape};
use rlx_opscope::{StatConfig, inject_infer_stats};
use rlx_runtime::{Device, Session};

const T: usize = 16;
const D: usize = 64;
const E: usize = 8;
const K: usize = 2;

fn attention_demo() {
    // scores [T,T] → softmax(-1). Sink (key 0) + local band get large logits.
    let mut g = Graph::new("attn");
    let scores = g.input("scores", Shape::new(&[T, T], DType::F32));
    let _p = g.softmax(scores, -1, Shape::new(&[T, T], DType::F32));
    g.set_outputs(vec![_p]);
    let (gi, specs) = inject_infer_stats(&g, &StatConfig::default());
    let mut c = Session::new(Device::Cpu).compile(gi);

    let mut s = vec![0f32; T * T];
    for i in 0..T {
        for j in 0..T {
            let mut v = 0.0;
            if j == 0 {
                v += 6.0; // attention sink
            }
            if (i as i64 - j as i64).abs() <= 1 {
                v += 6.0; // local window
            }
            s[i * T + j] = v;
        }
    }
    let outs = c.run(&[("scores", &s)]);
    let find = |stat: &str| outs[specs.iter().find(|x| x.stat == stat).unwrap().out_idx].clone();
    let qmax = find("attn_qmax");
    let krecv = find("attn_krecv");

    let mean_peak = qmax.iter().sum::<f32>() / qmax.len() as f32;
    let concentrated = qmax.iter().filter(|&&m| m > 0.3).count();
    // Per-key received mass as fraction of total (T queries).
    let mut keys: Vec<(usize, f32)> = krecv
        .iter()
        .enumerate()
        .map(|(k, &m)| (k, m / T as f32))
        .collect();
    keys.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top4: f32 = keys.iter().take(4).map(|(_, m)| m).sum();

    println!("── Attention (softmax scores) ──");
    println!("  mean per-query peak mass : {mean_peak:.2}");
    println!("  concentrated queries     : {concentrated}/{T} (peak > 0.30)");
    println!(
        "  top-4 keys receive       : {:.0}% of all attention  (sinks {:?})",
        top4 * 100.0,
        keys.iter().take(3).map(|(k, _)| *k).collect::<Vec<_>>()
    );
    println!("  → EXPLOIT: windowed+sink sparse attention — keep ~4 keys/query, skip the rest\n");
}

fn moe_gate_demo() {
    // tokens [T,D] → gate = tok·Wg [T,E] → topk(K). Wg columns are scaled so
    // low-index experts win more → routing skew.
    let mut g = Graph::new("moe_gate");
    let tok = g.input("tok", Shape::new(&[T, D], DType::F32));
    let wg = g.param("Wg", Shape::new(&[D, E], DType::F32));
    let gate = g.matmul(tok, wg, Shape::new(&[T, E], DType::F32));
    let idx = g.add_node(
        Op::TopK { k: K },
        vec![gate],
        Shape::new(&[T, K], DType::F32),
    );
    g.set_outputs(vec![idx]);
    let (gi, specs) = inject_infer_stats(&g, &StatConfig::default());
    let mut c = Session::new(Device::Cpu).compile(gi);

    // Skewed gate weights: column e scaled by (E-e).
    let mut rng = Philox4x32::new(7);
    let mut w = vec![0f32; D * E];
    rng.fill_normal(&mut w);
    for d in 0..D {
        for e in 0..E {
            w[d * E + e] *= (E - e) as f32;
        }
    }
    c.set_param("Wg", &w);
    let mut tokd = vec![0f32; T * D];
    rng.fill_normal(&mut tokd);
    let outs = c.run(&[("tok", &tokd)]);

    let load = &outs[specs
        .iter()
        .find(|x| x.stat == "route_load")
        .unwrap()
        .out_idx];
    let total: f32 = load.iter().sum::<f32>().max(1.0);
    let mut ranked: Vec<(usize, f32)> = load
        .iter()
        .enumerate()
        .map(|(e, &c)| (e, c / total))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cold = ranked.iter().filter(|(_, f)| *f < 0.02).count();

    println!("── MoE routing (top-{K} of {E} experts) ──");
    print!("  per-expert load: ");
    for (e, f) in &ranked {
        print!("e{e}:{:.0}% ", f * 100.0);
    }
    println!();
    println!(
        "  hottest expert : e{} at {:.0}%  (uniform would be {:.0}%)",
        ranked[0].0,
        ranked[0].1 * 100.0,
        100.0 / E as f32
    );
    println!("  cold experts   : {cold}/{E} under 2% load");
    println!("  → EXPLOIT: drop/merge cold experts, prefetch/cache the hot ones\n");
}

fn main() {
    attention_demo();
    moe_gate_demo();
}
