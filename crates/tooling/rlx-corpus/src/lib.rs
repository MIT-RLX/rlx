// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A **named corpus** of graphs that every compiler change must clear.
//!
//! rlx has thousands of tests, but they are organized by the crate that owns
//! the code, not by what a compiler change puts at risk. Changing the memory
//! planner or a fusion pass means running everything and reading the wreckage;
//! nothing answers "which *families* did I break".
//!
//! CAKE §5.3 makes that explicit — "more than 400 static and compile cases and
//! 399 GPU correctness cases across roughly 28 families" — and §3.2 gates
//! compiler evolution on it: "Compiler changes are test-gated across the kernel
//! corpus, because a primitive and its analyses must evolve together."
//!
//! # What this is and is not
//!
//! Each [`Case`] is a graph plus the family it belongs to. The corpus runs
//! every case through the device-free gate stack — structural verify, shape
//! verify, representation compatibility, and the memory-plan program-safety /
//! schedule-semantics gates — under **every planner configuration the backends
//! actually use**, because a plan that is safe when pinned is not necessarily
//! safe when slots are reused.
//!
//! It is deliberately *not* a numerical test suite: those live with their
//! backends and need devices. This answers a narrower question that nothing
//! else does — does a compiler change keep every family structurally sound —
//! and it answers it on any machine with no GPU.
//!
//! # Coverage is reported, never implied
//!
//! [`coverage`] returns how many distinct `OpKind`s the corpus touches out of
//! how many exist. That number is currently a *minority* of the op surface,
//! and printing it is the point: a corpus that reports "all green" without
//! saying what it covers reads like completeness it has not earned.

pub mod evolution;
pub mod oracle;
pub mod reference;

use std::collections::BTreeSet;

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_opt::rlx_compile::memory::{
    MemoryPlan, MemoryPlanOptions, plan_memory, plan_memory_aligned, plan_memory_f32_uniform,
    plan_memory_native, plan_memory_with_options,
};
use rlx_opt::rlx_compile::plan_check::check_plan;

const F: DType = DType::F32;

/// One corpus entry: a family label, a case name, and the graph.
pub struct Case {
    pub family: &'static str,
    pub name: &'static str,
    pub graph: Graph,
}

/// Why a case failed, with enough detail to be a repair target.
#[derive(Debug, Clone)]
pub struct CaseFailure {
    pub family: &'static str,
    pub name: &'static str,
    /// Which gate rejected it: `structure`, `shape`, `repr`, or a planner
    /// configuration name for plan findings.
    pub gate: String,
    pub detail: String,
}

/// Per-family rollup plus every failure.
#[derive(Debug, Clone, Default)]
pub struct CorpusReport {
    pub families: Vec<(String, usize, usize)>,
    pub failures: Vec<CaseFailure>,
    pub cases_run: usize,
    pub plans_checked: usize,
}

impl CorpusReport {
    pub fn is_clean(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        let (covered, total) = coverage();
        out.push_str(&format!(
            "corpus: {} case(s) across {} families, {} plan(s) checked\n",
            self.cases_run,
            self.families.len(),
            self.plans_checked
        ));
        out.push_str(&format!(
            "coverage: {covered} of {total} OpKind(s) appear in a corpus case \
             ({:.0}%) — the remainder is UNCOVERED, not passing\n",
            100.0 * covered as f64 / total.max(1) as f64
        ));
        for (fam, ok, n) in &self.families {
            let mark = if ok == n { "ok  " } else { "FAIL" };
            out.push_str(&format!("  {mark} {fam:<22} {ok}/{n}\n"));
        }
        for f in &self.failures {
            out.push_str(&format!(
                "  ! {}::{} [{}] {}\n",
                f.family, f.name, f.gate, f.detail
            ));
        }
        out
    }
}

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims, F)
}

/// A named planner configuration: label plus the function that produces it.
type Planner = (&'static str, fn(&Graph) -> MemoryPlan);

/// Every planner configuration a backend actually uses. A plan that is safe
/// under one is not automatically safe under another: `pin_output_ancestors`
/// alone decides whether slot reuse happens at all.
fn planners() -> Vec<Planner> {
    fn in_order(g: &Graph) -> MemoryPlan {
        let mut o = MemoryPlanOptions::inference();
        o.pin_output_ancestors = false;
        o.dequant_host_fallback = false;
        plan_memory_with_options(g, 64, o)
    }
    fn no_dequant_pin(g: &Graph) -> MemoryPlan {
        let mut o = MemoryPlanOptions::inference();
        o.dequant_host_fallback = false;
        plan_memory_with_options(g, 64, o)
    }
    vec![
        ("default", plan_memory),
        ("aligned256", |g| plan_memory_aligned(g, 256)),
        ("native", |g| plan_memory_native(g, 64)),
        ("f32_uniform", |g| plan_memory_f32_uniform(g, 64)),
        ("in_order", in_order),
        ("no_dequant_pin", no_dequant_pin),
    ]
}

// ── The corpus ──────────────────────────────────────────────────────────────

fn elementwise_cases() -> Vec<Case> {
    let s = shape(&[32, 64]);
    let mut out = Vec::new();

    let mut g = Graph::new("ew_chain");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let a = g.add(x, y);
    let b = g.mul(a, x);
    let c = g.sub(b, a);
    let d = g.div(c, y);
    g.set_outputs(vec![d]);
    out.push(Case {
        family: "elementwise",
        name: "chain",
        graph: g,
    });

    let mut g = Graph::new("ew_broadcast");
    let x = g.input("x", shape(&[32, 64]));
    let y = g.input("y", shape(&[1, 64]));
    let a = g.add(x, y);
    g.set_outputs(vec![a]);
    out.push(Case {
        family: "elementwise",
        name: "broadcast",
        graph: g,
    });

    out
}

fn matmul_cases() -> Vec<Case> {
    let mut out = Vec::new();

    for (label, m, k, n) in [
        ("decode", 1usize, 256usize, 256usize),
        ("prefill", 64, 256, 256),
    ] {
        let mut g = Graph::new("mm");
        let x = g.input("x", shape(&[m, k]));
        let w = g.param("w", shape(&[k, n]));
        let y = g.matmul(x, w, shape(&[m, n]));
        g.set_outputs(vec![y]);
        out.push(Case {
            family: "matmul",
            name: if label == "decode" {
                "decode"
            } else {
                "prefill"
            },
            graph: g,
        });
    }

    // Shared input feeding two matmuls — the shape that makes a planner reuse
    // or pin, depending on configuration.
    let mut g = Graph::new("mm_branch");
    let x = g.input("x", shape(&[32, 128]));
    let w1 = g.param("w1", shape(&[128, 256]));
    let w2 = g.param("w2", shape(&[128, 256]));
    let a = g.matmul(x, w1, shape(&[32, 256]));
    let b = g.matmul(x, w2, shape(&[32, 256]));
    let c = g.mul(a, b);
    g.set_outputs(vec![c]);
    out.push(Case {
        family: "matmul",
        name: "shared_input_branch",
        graph: g,
    });

    out
}

fn norm_cases() -> Vec<Case> {
    let s = shape(&[8, 64]);
    let mut out = Vec::new();
    for (name, op) in [
        (
            "rms",
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
        ),
        (
            "layer",
            Op::LayerNorm {
                axis: -1,
                eps: 1e-5,
            },
        ),
    ] {
        let mut g = Graph::new("norm");
        let x = g.input("x", s.clone());
        let gamma = g.param("gamma", shape(&[64]));
        let beta = g.param("beta", shape(&[64]));
        let y = g.add_node(op, vec![x, gamma, beta], s.clone());
        g.set_outputs(vec![y]);
        out.push(Case {
            family: "norm",
            name,
            graph: g,
        });
    }
    out
}

fn attention_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (name, mask) in [
        ("causal", rlx_ir::op::MaskKind::Causal),
        ("none", rlx_ir::op::MaskKind::None),
    ] {
        let qkv = shape(&[1, 2, 8, 16]);
        let mut g = Graph::new("attn");
        let q = g.input("q", qkv.clone());
        let k = g.input("k", qkv.clone());
        let v = g.input("v", qkv.clone());
        let y = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 16,
                v_head_dim: None,
                mask_kind: mask,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            qkv,
        );
        g.set_outputs(vec![y]);
        out.push(Case {
            family: "attention",
            name,
            graph: g,
        });
    }
    out
}

fn reduce_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (name, op) in [
        ("sum", rlx_ir::op::ReduceOp::Sum),
        ("max", rlx_ir::op::ReduceOp::Max),
        ("mean", rlx_ir::op::ReduceOp::Mean),
    ] {
        let mut g = Graph::new("reduce");
        let x = g.input("x", shape(&[4, 8, 16]));
        let y = g.add_node(
            Op::Reduce {
                op,
                axes: vec![2],
                keep_dim: false,
            },
            vec![x],
            shape(&[4, 8]),
        );
        g.set_outputs(vec![y]);
        out.push(Case {
            family: "reduce",
            name,
            graph: g,
        });
    }
    out
}

fn transformer_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (name, layers, seq) in [("decode_block", 2usize, 1usize), ("prefill_block", 2, 32)] {
        let (d, ff) = (64usize, 128usize);
        let hs = shape(&[seq, d]);
        let mut g = Graph::new("xf");
        let mut h = g.input("x", hs.clone());
        for l in 0..layers {
            let gamma = g.param(format!("l{l}.g").as_str(), shape(&[d]));
            let beta = g.param(format!("l{l}.b").as_str(), shape(&[d]));
            let normed = g.add_node(
                Op::RmsNorm {
                    axis: -1,
                    eps: 1e-6,
                },
                vec![h, gamma, beta],
                hs.clone(),
            );
            let mut attn = normed;
            for p in ["q", "k", "v", "o"] {
                let w = g.param(format!("l{l}.{p}").as_str(), shape(&[d, d]));
                attn = g.matmul(attn, w, hs.clone());
            }
            h = g.add(h, attn);
            let gw = g.param(format!("l{l}.gate").as_str(), shape(&[d, ff]));
            let uw = g.param(format!("l{l}.up").as_str(), shape(&[d, ff]));
            let gate = g.matmul(h, gw, shape(&[seq, ff]));
            let up = g.matmul(h, uw, shape(&[seq, ff]));
            let act = g.mul(gate, up);
            let dw = g.param(format!("l{l}.down").as_str(), shape(&[ff, d]));
            let mlp = g.matmul(act, dw, hs.clone());
            h = g.add(h, mlp);
        }
        g.set_outputs(vec![h]);
        out.push(Case {
            family: "transformer",
            name,
            graph: g,
        });
    }
    out
}

fn view_cases() -> Vec<Case> {
    let mut out = Vec::new();
    // Views alias their parent's storage. The plan gate must fold them onto
    // the root rather than reporting the intended sharing as an overlap.
    let mut g = Graph::new("view");
    let x = g.input("x", shape(&[4, 64]));
    let r = g.add_node(
        Op::Reshape {
            new_shape: vec![256],
        },
        vec![x],
        shape(&[256]),
    );
    let y = g.add_node(
        Op::Reshape {
            new_shape: vec![4, 64],
        },
        vec![r],
        shape(&[4, 64]),
    );
    let z = g.add(y, x);
    g.set_outputs(vec![z]);
    out.push(Case {
        family: "view",
        name: "reshape_roundtrip",
        graph: g,
    });
    out
}

/// The quantized matmul path: 137 `Op::DequantMatMul` uses in downstream model
/// code, and the family where this tree shipped `rocm-gguf-transposed` — an
/// `sgemm(N,N)` against an `[n,k]` GGUF layout, hidden because the only test
/// used `n = 1`. Every case here therefore uses `n > 1`.
///
/// Packed weights are `U8` blobs whose length follows each scheme's block
/// layout; the structural gates check shape, representation and plan safety.
/// Numerical validation is a separate question the oracle answers only where
/// it can, and it does not cover dequantization.
fn quantized_cases() -> Vec<Case> {
    use rlx_ir::quant::QuantScheme;
    let mut out = Vec::new();

    // GGUF super-block schemes take TWO operands (activation + one packed
    // blob); the split-operand schemes take four. The corpus gate caught this
    // when the first attempt built every case with two, which is exactly the
    // arity confusion a structural gate exists to catch.
    //
    // Block SIZE is per-scheme, not a constant: the K-quants pack 256 elements
    // per super-block but Q8_0 packs 32. This used a hardcoded 256 for every
    // scheme, so `q8_0_prefill` allocated an eighth of the bytes its weight
    // needs — invisible while nothing fed the param real data, and a
    // guaranteed out-of-bounds read the moment anything did.

    for (name, scheme, m, k, n) in [
        (
            "q4k_decode",
            QuantScheme::GgufQ4K,
            1usize,
            512usize,
            128usize,
        ),
        ("q4k_prefill", QuantScheme::GgufQ4K, 32, 512, 128),
        ("q6k_decode", QuantScheme::GgufQ6K, 1, 512, 128),
        ("q8_0_prefill", QuantScheme::GgufQ8_0, 16, 512, 128),
    ] {
        // n > 1 on every case: `rocm-gguf-transposed` shipped precisely because
        // the only test used n = 1, where a transposed layout is invisible.
        assert!(n > 1);
        let blocks_per_row = k / scheme.gguf_block_size() as usize;
        let bytes = n * blocks_per_row * scheme.gguf_block_bytes() as usize;
        let mut g = Graph::new("dq_gguf");
        let x = g.input("x", shape(&[m, k]));
        let w = g.param("w_packed", Shape::new(&[bytes.max(1)], DType::U8));
        let y = g.add_node(Op::DequantMatMul { scheme }, vec![x, w], shape(&[m, n]));
        g.set_outputs(vec![y]);
        out.push(Case {
            family: "quantized",
            name,
            graph: g,
        });
    }

    // Two dequant matmuls in sequence: the shape that makes the planner decide
    // whether to pin the intermediate activation (`dequant_host_fallback`).
    let scheme = QuantScheme::GgufQ4K;
    let bpr =
        |k: usize| (k / scheme.gguf_block_size() as usize) * scheme.gguf_block_bytes() as usize;
    let mut g = Graph::new("dq_chain");
    let x = g.input("x", shape(&[8, 512]));
    let w1 = g.param("w1", Shape::new(&[128 * bpr(512)], DType::U8));
    let a = g.add_node(Op::DequantMatMul { scheme }, vec![x, w1], shape(&[8, 128]));
    let pad = g.param("pad", shape(&[8, 128]));
    let a2 = g.add(a, pad);
    let w2 = g.param("w2", Shape::new(&[64 * bpr(512)], DType::U8));
    // Concat to reach the 512-element contraction the next block needs. An
    // earlier draft used `Expand` from 128 to 256 here, which is not a legal
    // broadcast — and every structural gate accepted it, so it reached the CPU
    // backend and panicked on an out-of-bounds read. See
    // `expand_from_a_non_unit_dim_is_not_rejected`.
    let wide = g.add_node(
        Op::Concat { axis: 1 },
        vec![a2, a2, a2, a2],
        shape(&[8, 512]),
    );
    let b = g.add_node(
        Op::DequantMatMul { scheme },
        vec![wide, w2],
        shape(&[8, 64]),
    );
    g.set_outputs(vec![b]);
    out.push(Case {
        family: "quantized",
        name: "dequant_chain",
        graph: g,
    });

    out
}

/// Shape-manipulation ops, ranked by downstream usage: `Concat` (39),
/// `Expand` (33), `Transpose` (24), `Narrow` (20).
///
/// These are where view aliasing lives, so they are the cases most likely to
/// make the plan gate's alias folding wrong in either direction.
fn structural_cases() -> Vec<Case> {
    let mut out = Vec::new();

    let mut g = Graph::new("concat");
    let a = g.input("a", shape(&[4, 32]));
    let b = g.input("b", shape(&[4, 32]));
    let c = g.add_node(Op::Concat { axis: 1 }, vec![a, b], shape(&[4, 64]));
    let d = g.mul(c, c);
    g.set_outputs(vec![d]);
    out.push(Case {
        family: "structural",
        name: "concat_axis1",
        graph: g,
    });

    let mut g = Graph::new("expand");
    let a = g.input("a", shape(&[1, 32]));
    let e = g.add_node(
        Op::Expand {
            target_shape: vec![8, 32],
        },
        vec![a],
        shape(&[8, 32]),
    );
    let b = g.input("b", shape(&[8, 32]));
    let y = g.add(e, b);
    g.set_outputs(vec![y]);
    out.push(Case {
        family: "structural",
        name: "expand_broadcast",
        graph: g,
    });

    let mut g = Graph::new("transpose");
    let a = g.input("a", shape(&[16, 32]));
    let t = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![a],
        shape(&[32, 16]),
    );
    let w = g.param("w", shape(&[16, 8]));
    let y = g.matmul(t, w, shape(&[32, 8]));
    g.set_outputs(vec![y]);
    out.push(Case {
        family: "structural",
        name: "transpose_then_matmul",
        graph: g,
    });

    out
}

/// Every case in the corpus.
pub fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    all.extend(elementwise_cases());
    all.extend(matmul_cases());
    all.extend(norm_cases());
    all.extend(attention_cases());
    all.extend(reduce_cases());
    all.extend(transformer_cases());
    all.extend(view_cases());
    all.extend(quantized_cases());
    all.extend(structural_cases());
    all
}

/// `(covered, total)` distinct `OpKind`s — honest coverage, not a pass rate.
pub fn coverage() -> (usize, usize) {
    // Keyed on the kind's name: `OpKind` is not `Ord`, and the name is what
    // the coverage report has to print anyway.
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for case in cases() {
        for node in case.graph.nodes() {
            seen.insert(format!("{:?}", node.op.kind()));
        }
    }
    (seen.len(), OPKIND_TOTAL)
}

/// Total `OpKind` variants. Pinned by `opkind_total_is_current` so the
/// coverage denominator cannot silently drift as ops are added.
pub const OPKIND_TOTAL: usize = 187;

/// Run every case through the device-free gate stack.
pub fn run() -> CorpusReport {
    let mut report = CorpusReport::default();
    let mut per_family: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

    for case in cases() {
        report.cases_run += 1;
        let entry = per_family.entry(case.family.to_string()).or_insert((0, 0));
        entry.1 += 1;
        let before = report.failures.len();

        let structural = rlx_ir::verify::verify(&case.graph);
        for e in &structural {
            report.failures.push(CaseFailure {
                family: case.family,
                name: case.name,
                gate: "structure".into(),
                detail: e.message.clone(),
            });
        }
        if structural.is_empty() {
            for e in rlx_ir::verify::verify_shapes(&case.graph) {
                report.failures.push(CaseFailure {
                    family: case.family,
                    name: case.name,
                    gate: "shape".into(),
                    detail: e.message.clone(),
                });
            }
            for f in &rlx_ir::repr_check::check_graph(&case.graph).findings {
                report.failures.push(CaseFailure {
                    family: case.family,
                    name: case.name,
                    gate: "repr".into(),
                    detail: format!(
                        "{:?}: expects {} but producer declares {}",
                        f.op, f.expected, f.actual
                    ),
                });
            }
            for (label, plan_fn) in planners() {
                let plan = plan_fn(&case.graph);
                report.plans_checked += 1;
                for f in &check_plan(&case.graph, &plan).findings {
                    report.failures.push(CaseFailure {
                        family: case.family,
                        name: case.name,
                        gate: format!("plan/{label}"),
                        detail: format!("[{}] {}", f.kind.as_str(), f.detail),
                    });
                }
            }
        }

        if report.failures.len() == before {
            entry.0 += 1;
        }
    }

    report.families = per_family
        .into_iter()
        .map(|(k, (ok, n))| (k, ok, n))
        .collect();
    report
}
