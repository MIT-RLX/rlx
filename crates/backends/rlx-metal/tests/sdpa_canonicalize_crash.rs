// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reduce the MPSGraph `CanonicalizeSDPA` segfault to a minimal graph.
//!
//! `rlx-latte` (hyperbolic attention, built from raw matmuls rather than
//! `Op::Attention`) takes Apple's MPSGraph compiler down with
//!
//! ```text
//! EXC_BAD_ACCESS (SIGSEGV) KERN_INVALID_ADDRESS at 0x2c
//!   CanonicalizeSDPA<false>::matchAndRewrite(mlir::mps::MatMulOp, mlir::PatternRewriter&)
//!   GreedyPatternRewriteDriver::processWorklist()
//!   CommonRuntimeCanonicalizationPass::runOnOperation()
//! ```
//!
//! It is a crash *inside the vendor compiler*, so rlx cannot catch it — it can
//! only avoid handing over the shape. This narrows down which shape that is, so
//! the guard can be as tight as possible.
//!
//! Each case runs in its own process, because a segfault ends the run:
//!
//!     cargo test -p rlx-metal --test sdpa_canonicalize_crash -- --nocapture

use std::process::Command;

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session};

const F32: DType = DType::F32;

/// Two chained batched matmuls with an elementwise op between — the SDPA
/// skeleton, minus the softmax. `k` is the second matmul's right-hand width;
/// LATTE's is one wider than the first matmul's (the Lorentz coordinate).
fn chained(b: usize, s: usize, d: usize, k: usize, softmax: bool) -> Graph {
    let mut g = Graph::new("chained_mm");
    let q = g.input("q", Shape::new(&[b, s, d], F32));
    let kt = g.input("kt", Shape::new(&[b, d, s], F32));
    let v = g.input("v", Shape::new(&[b, s, k], F32));
    let energy = g.mm(q, kt); // [b, s, s]
    let mid: NodeId = if softmax {
        g.add_node(
            Op::Softmax { axis: -1 },
            vec![energy],
            Shape::new(&[b, s, s], F32),
        )
    } else {
        let one = g.input("one", Shape::new(&[b, s, s], F32));
        g.sub(energy, one)
    };
    let out = g.mm(mid, v); // [b, s, k]
    g.set_outputs(vec![out]);
    g
}

/// The shape LATTE actually crashes on: the second matmul's right operand is a
/// `Concat`, not a plain tensor. Bisecting the real graph put the fault exactly
/// on `MatMul(x, Concat{axis:2}(…))`.
fn chained_concat_rhs(b: usize, s: usize, d: usize, k: usize) -> Graph {
    let mut g = Graph::new("chained_mm_concat");
    let q = g.input("q", Shape::new(&[b, s, d], F32));
    let kt = g.input("kt", Shape::new(&[b, d, s], F32));
    let head = g.input("head", Shape::new(&[b, s, 1], F32));
    let tail = g.input("tail", Shape::new(&[b, s, k], F32));
    let energy = g.mm(q, kt); // [b, s, s]
    let v = g.add_node(
        Op::Concat { axis: 2 },
        vec![head, tail],
        Shape::new(&[b, s, k + 1], F32),
    );
    let out = g.mm(energy, v); // [b, s, k+1]
    g.set_outputs(vec![out]);
    g
}

fn run_concat_case(b: usize, s: usize, d: usize, k: usize) {
    let g = chained_concat_rhs(b, s, d, k);
    let mut c = Session::new(Device::Metal).compile(g);
    c.finalize_params();
    let q = vec![0.1f32; b * s * d];
    let kt = vec![0.2f32; b * d * s];
    let head = vec![1.0f32; b * s];
    let tail = vec![0.3f32; b * s * k];
    let out = c
        .run(&[("q", &q), ("kt", &kt), ("head", &head), ("tail", &tail)])
        .remove(0);
    println!(
        "SURVIVED len={} first={:.4}",
        out.len(),
        out.first().copied().unwrap_or(0.0)
    );
}

/// As `chained_concat_rhs`, but `kᵀ` is produced by an explicit `Transpose` —
/// what an SDPA pattern matcher actually looks for (`matmul(q, transpose(k))`).
fn chained_transpose_concat(b: usize, s: usize, d: usize, k: usize) -> Graph {
    let mut g = Graph::new("chained_t_concat");
    let q = g.input("q", Shape::new(&[b, s, d], F32));
    let kk = g.input("k", Shape::new(&[b, s, d], F32));
    let head = g.input("head", Shape::new(&[b, s, 1], F32));
    let tail = g.input("tail", Shape::new(&[b, s, k], F32));
    let kt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![kk],
        Shape::new(&[b, d, s], F32),
    );
    let energy = g.mm(q, kt); // [b, s, s]
    let v = g.add_node(
        Op::Concat { axis: 2 },
        vec![head, tail],
        Shape::new(&[b, s, k + 1], F32),
    );
    let out = g.mm(energy, v);
    g.set_outputs(vec![out]);
    g
}

fn run_transpose_case(b: usize, s: usize, d: usize, k: usize) {
    let g = chained_transpose_concat(b, s, d, k);
    let mut c = Session::new(Device::Metal).compile(g);
    c.finalize_params();
    let q = vec![0.1f32; b * s * d];
    let kk = vec![0.2f32; b * s * d];
    let head = vec![1.0f32; b * s];
    let tail = vec![0.3f32; b * s * k];
    let out = c
        .run(&[("q", &q), ("k", &kk), ("head", &head), ("tail", &tail)])
        .remove(0);
    println!(
        "SURVIVED len={} first={:.4}",
        out.len(),
        out.first().copied().unwrap_or(0.0)
    );
}

/// The real LATTE shape, read off the graph: a **hand-rolled** softmax
/// (`max → sub → exp → sum → div`, not `Op::Softmax`) feeding a matmul whose
/// right operand is a `Concat`. Apple's SDPA canonicalizer pattern-matches the
/// decomposed softmax, so an `Op::Softmax` node never reaches it.
///
/// `decomposed`: write the softmax out longhand instead of using `Op::Softmax`.
/// `concat_v`:   build `v` with a `Concat` (LATTE's Lorentz lift) instead of an input.
fn latte_shaped(b: usize, s: usize, d: usize, k: usize, decomposed: bool, concat_v: bool) -> Graph {
    let mut g = Graph::new("latte_shaped");
    let q = g.input("q", Shape::new(&[b, s, d], F32));
    let kk = g.input("k", Shape::new(&[b, s, d], F32));
    let kt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![kk],
        Shape::new(&[b, d, s], F32),
    );
    let scores = g.mm(q, kt); // [b, s, s]
    let sq = Shape::new(&[b, s, s], F32);
    let keep = Shape::new(&[b, s, 1], F32);
    let attn = if decomposed {
        let m = g.reduce(
            scores,
            rlx_ir::op::ReduceOp::Max,
            vec![2],
            true,
            keep.clone(),
        );
        let z = g.sub(scores, m);
        let e = g.exp(z);
        let sum = g.reduce(e, rlx_ir::op::ReduceOp::Sum, vec![2], true, keep);
        g.div(e, sum)
    } else {
        g.add_node(Op::Softmax { axis: -1 }, vec![scores], sq)
    };
    let (v, vw) = if concat_v {
        let head = g.input("head", Shape::new(&[b, s, 1], F32));
        let tail = g.input("tail", Shape::new(&[b, s, k], F32));
        (
            g.add_node(
                Op::Concat { axis: 2 },
                vec![head, tail],
                Shape::new(&[b, s, k + 1], F32),
            ),
            k + 1,
        )
    } else {
        (g.input("v", Shape::new(&[b, s, k], F32)), k)
    };
    let out = g.mm(attn, v);
    let _ = vw;
    g.set_outputs(vec![out]);
    g
}

fn run_latte_case(b: usize, s: usize, d: usize, k: usize, decomposed: bool, concat_v: bool) {
    let g = latte_shaped(b, s, d, k, decomposed, concat_v);
    let mut c = Session::new(Device::Metal).compile(g);
    c.finalize_params();
    let q = vec![0.1f32; b * s * d];
    let kk = vec![0.2f32; b * s * d];
    let head = vec![1.0f32; b * s];
    let tail = vec![0.3f32; b * s * k];
    let v = vec![0.3f32; b * s * k];
    let mut ins: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &kk)];
    if concat_v {
        ins.push(("head", &head));
        ins.push(("tail", &tail));
    } else {
        ins.push(("v", &v));
    }
    let out = c.run(&ins).remove(0);
    println!(
        "SURVIVED len={} first={:.4}",
        out.len(),
        out.first().copied().unwrap_or(0.0)
    );
}

fn run_case(b: usize, s: usize, d: usize, k: usize, softmax: bool) {
    let g = chained(b, s, d, k, softmax);
    let mut c = Session::new(Device::Metal).compile(g);
    c.finalize_params();
    let q = vec![0.1f32; b * s * d];
    let kt = vec![0.2f32; b * d * s];
    let v = vec![0.3f32; b * s * k];
    let one = vec![0.05f32; b * s * s];
    let mut ins: Vec<(&str, &[f32])> = vec![("q", &q), ("kt", &kt), ("v", &v)];
    if !softmax {
        ins.push(("one", &one));
    }
    let out = c.run(&ins).remove(0);
    println!(
        "SURVIVED len={} first={:.4}",
        out.len(),
        out.first().copied().unwrap_or(0.0)
    );
}

/// Child-process entry. The case comes through the environment, not argv:
/// libtest owns the command line and rejects unknown flags with exit 101, which
/// looks exactly like the crash we are trying to detect.
fn main_child() -> bool {
    let Some(spec) = rlx_ir::env::var("RLX_SDPA_CASE") else {
        return false;
    };
    let n: Vec<&str> = spec.split(',').collect();
    let p = |i: usize| n[i].parse::<usize>().expect("case field");
    if n[4] == "concat" {
        run_concat_case(p(0), p(1), p(2), p(3));
    } else if n[4] == "tpose" {
        run_transpose_case(p(0), p(1), p(2), p(3));
    } else if let Some(rest) = n[4].strip_prefix("latte-") {
        // `latte-<decomposed><concat>` e.g. `latte-11`.
        let f: Vec<char> = rest.chars().collect();
        run_latte_case(p(0), p(1), p(2), p(3), f[0] == '1', f[1] == '1');
    } else {
        run_case(p(0), p(1), p(2), p(3), n[4] == "1");
    }
    true
}

#[test]
fn narrow_down_the_crashing_shape() {
    if main_child() {
        return;
    }
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("metal not available — skipped");
        return;
    }
    let exe = std::env::current_exe().expect("test exe");
    // (b, s, d, k, softmax) — LATTE's is (6, 16, 16, 17, false).
    let cases: &[(usize, usize, usize, usize, &str)] = &[
        (6, 16, 16, 16, "1"), // a normal SDPA skeleton
        (6, 16, 16, 16, "0"), // …without the softmax
        (6, 16, 16, 17, "0"), // …and with the mismatched second width
        (1, 8, 8, 9, "0"),
        // The real one: the second matmul's rhs is a Concat. Bisecting LATTE put
        // the fault on exactly `MatMul(_, Concat{axis:2})`.
        (6, 16, 16, 16, "concat"),
        (1, 8, 8, 8, "concat"),
        (6, 16, 16, 16, "tpose"),
        // The four combinations of (hand-rolled softmax, concat-built v) at
        // LATTE's real shape. Its own is decomposed + concat.
        (2, 16, 16, 16, "latte-00"),
        (2, 16, 16, 16, "latte-01"),
        (2, 16, 16, 16, "latte-10"),
        (2, 16, 16, 16, "latte-11"),
    ];
    for &(b, s, d, k, sm) in cases {
        let out = Command::new(&exe)
            .args(["--nocapture", "--exact", "narrow_down_the_crashing_shape"])
            .env("RLX_SDPA_CASE", format!("{b},{s},{d},{k},{sm}"))
            .output()
            .expect("spawn child");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let signal = !out.status.success() && out.status.code().is_none();
        let verdict = if stdout.contains("SURVIVED") {
            "ok".to_string()
        } else if signal {
            "CRASHED (signal)".to_string()
        } else {
            format!("failed (exit {:?})", out.status.code())
        };
        println!("  b={b} s={s} d={d} k={k:3} mode={sm:6} -> {verdict}");
        if !stdout.contains("SURVIVED") && !signal {
            let err = String::from_utf8_lossy(&out.stderr);
            println!(
                "      {}",
                err.lines().rev().take(2).collect::<Vec<_>>().join(" | ")
            );
        }
    }
}
