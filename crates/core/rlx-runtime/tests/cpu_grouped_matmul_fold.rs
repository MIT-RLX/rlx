// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Transpose(bank, [0,2,1]) → GroupedMatMul` folds into a B-transposed GEMM.
//!
//! `Op::GroupedMatMul` wants its expert bank as `[E, K, N]`, while GGUF stores
//! banks as `[E, N, K]`. Every dense MoE layer therefore emits a rank-3
//! transpose ahead of the GEMM — a full copy of the bank, on every forward. On a
//! paged GLM-5.3-Flash layer that copy measured **99%** of the layer: 446 ms
//! against 370 us for the eight grouped matmuls it fed. Folding it into
//! `sgemm_bt` took the layer from 299 ms to 35 ms.
//!
//! Two things have to hold, and the second is the one that bites:
//!
//!   * the folded result equals the unfolded one, and
//!   * the fold only elides the transpose when EVERY reader can fold. One
//!     transposed bank feeds all `top_k` grouped matmuls of a layer, so a
//!     single-use guard would reject the exact shape this exists for — but a
//!     reader that still needs the materialized tensor must keep it.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const E: usize = 4;
const K: usize = 6;
const N: usize = 5;
const M: usize = 7;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32) * 0.25 - 2.0)
        .collect()
}

/// `out[i] = input[i] @ bank_kn[expert_idx[i]]`, straight from the definition,
/// with the bank given in `[E, N, K]` (GGUF) order.
fn reference(input: &[f32], bank_nk: &[f32], ids: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; M * N];
    for (i, &e) in ids.iter().enumerate() {
        for j in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                // bank_nk[e][j][k] is the [N, K] element.
                acc += input[i * K + k] * bank_nk[e * N * K + j * K + k];
            }
            out[i * N + j] = acc;
        }
    }
    out
}

/// Build `GroupedMatMul(x, Transpose(bank, [0,2,1]), idx)` repeated `fanout`
/// times (summed), optionally also consuming the transpose somewhere the fold
/// cannot reach.
fn run(bank_nk: &[f32], input: &[f32], ids: &[f32], fanout: usize, extra_reader: bool) -> Vec<f32> {
    let mut g = Graph::new("gmm_fold");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let idx = g.input("idx", Shape::new(&[M], DType::F32));
    let bank = g.param("bank", Shape::new(&[E, N, K], DType::F32));
    let bt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![bank],
        Shape::new(&[E, K, N], DType::F32),
    );

    let mut acc: Option<rlx_ir::NodeId> = None;
    for _ in 0..fanout {
        let y = g.add_node(
            Op::GroupedMatMul,
            vec![x, bt, idx],
            Shape::new(&[M, N], DType::F32),
        );
        acc = Some(match acc {
            Some(p) => g.add_node(
                Op::Binary(BinaryOp::Add),
                vec![p, y],
                Shape::new(&[M, N], DType::F32),
            ),
            None => y,
        });
    }
    let mut outputs = vec![acc.expect("fanout >= 1")];
    if extra_reader {
        // A consumer that needs the transpose materialized. If the compiler
        // elides it anyway, this output reads an uninitialized buffer.
        let s = g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![bt, bt],
            Shape::new(&[E, K, N], DType::F32),
        );
        outputs.push(s);
    }
    g.set_outputs(outputs);

    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("bank", bank_nk);
    let outs = c.run(&[("x", input), ("idx", ids)]);
    if extra_reader {
        // Check the second output too: it is the whole point of this case.
        let want_t: Vec<f32> = {
            let mut t = vec![0f32; E * K * N];
            for e in 0..E {
                for n in 0..N {
                    for k in 0..K {
                        t[e * K * N + k * N + n] = bank_nk[e * N * K + n * K + k] * 2.0;
                    }
                }
            }
            t
        };
        assert_eq!(
            outs[1], want_t,
            "a non-foldable reader of the transpose got the wrong tensor — it was \
             elided out from under them"
        );
    }
    outs.into_iter().next().unwrap()
}

#[test]
fn a_folded_bank_transpose_gives_the_same_answer() {
    let bank = ramp(E * N * K, 1);
    let input = ramp(M * K, 2);
    let ids: Vec<usize> = (0..M).map(|i| (i * 3) % E).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&e| e as f32).collect();

    let want = reference(&input, &bank, &ids);
    let got = run(&bank, &input, &ids_f, 1, false);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "element {i}: folded {g} vs reference {w}"
        );
    }
}

/// The shape the fold exists for: one transposed bank feeding every `top_k`
/// grouped matmul of a layer. A single-use guard rejects exactly this.
#[test]
fn a_bank_shared_by_every_top_k_matmul_still_folds() {
    let bank = ramp(E * N * K, 3);
    let input = ramp(M * K, 4);
    let ids: Vec<usize> = (0..M).map(|i| (i * 5) % E).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&e| e as f32).collect();

    let one = reference(&input, &bank, &ids);
    for fanout in [1usize, 2, 8] {
        let got = run(&bank, &input, &ids_f, fanout, false);
        for (i, (g, w)) in got.iter().zip(&one).enumerate() {
            let want = w * fanout as f32;
            assert!(
                (g - want).abs() < 1e-3,
                "fanout {fanout} element {i}: {g} vs {want}"
            );
        }
    }
}

/// A reader that cannot fold must keep the transpose alive.
#[test]
fn a_non_foldable_reader_keeps_the_transpose() {
    let bank = ramp(E * N * K, 5);
    let input = ramp(M * K, 6);
    let ids: Vec<usize> = (0..M).map(|i| (i * 2) % E).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&e| e as f32).collect();

    // `run` asserts the extra reader's output internally.
    let got = run(&bank, &input, &ids_f, 2, true);
    let one = reference(&input, &bank, &ids);
    for (g, w) in got.iter().zip(&one) {
        assert!((g - w * 2.0).abs() < 1e-3, "{g} vs {}", w * 2.0);
    }
}

/// Every token routed to one expert, and every token to a different one: the
/// segmented GEMM's counting sort has to land the same either way.
#[test]
fn folding_is_correct_across_routing_patterns() {
    let bank = ramp(E * N * K, 7);
    let input = ramp(M * K, 8);
    for pattern in [
        vec![0usize; M],
        (0..M).map(|i| i % E).collect::<Vec<_>>(),
        (0..M).map(|i| (E - 1) - (i % E)).collect::<Vec<_>>(),
    ] {
        let ids_f: Vec<f32> = pattern.iter().map(|&e| e as f32).collect();
        let want = reference(&input, &bank, &pattern);
        let got = run(&bank, &input, &ids_f, 1, false);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "routing {pattern:?} element {i}: {g} vs {w}"
            );
        }
    }
}

/// The fold changes accumulation order, so it must be shown not to change
/// accuracy — every dense MoE layer now takes this path.
///
/// Folded and unfolded are compared against each other and against an f64
/// evaluation of the same sums. `sgemm_bt` walks the bank along `K` where
/// `sgemm` walks it along `N`, which is a different (equally valid) order; what
/// has to hold is that neither is systematically worse, and that the gap
/// between them stays at f32 rounding rather than growing with `K`.
///
/// The unfolded path is obtained by giving the transpose a reader that cannot
/// fold, which forces it to be materialized and the GEMM to run untransposed.
#[test]
fn folding_does_not_cost_accuracy() {
    // Long contraction: rounding error grows with K, so this is where the two
    // orders would diverge if either were unsound.
    const KK: usize = 512;

    let mut bank = vec![0f32; E * N * KK];
    let mut input = vec![0f32; M * KK];
    // Deliberately NOT dyadic. Values that are multiples of 1/32 make every
    // product and partial sum exactly representable, so both orders come out
    // bit-identical and the comparison measures nothing — which is what the
    // first version of this test did.
    for (i, v) in bank.iter_mut().enumerate() {
        *v = ((((i * 41) % 97) as f32) - 48.0) / 97.0;
    }
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((((i * 29) % 89) as f32) - 44.0) / 89.0;
    }
    let ids: Vec<usize> = (0..M).map(|i| (i * 3) % E).collect();
    let ids_f: Vec<f32> = ids.iter().map(|&e| e as f32).collect();

    // f64 reference for the same sums.
    let mut exact = vec![0f64; M * N];
    for (i, &e) in ids.iter().enumerate() {
        for j in 0..N {
            let mut acc = 0f64;
            for k in 0..KK {
                acc += input[i * KK + k] as f64 * bank[e * N * KK + j * KK + k] as f64;
            }
            exact[i * N + j] = acc;
        }
    }

    let folded = run_k(&bank, &input, &ids_f, KK, false);
    let unfolded = run_k(&bank, &input, &ids_f, KK, true);

    let err = |v: &[f32]| -> f64 {
        v.iter()
            .zip(&exact)
            .map(|(g, w)| (*g as f64 - w).abs() / w.abs().max(1.0))
            .fold(0.0f64, f64::max)
    };
    let (ef, eu) = (err(&folded), err(&unfolded));
    eprintln!("folded rel err {ef:.3e}, unfolded {eu:.3e}, K={KK}");
    // If both are exactly zero the inputs were dyadic and nothing was rounded,
    // so the comparison below would be vacuous.
    assert!(
        ef > 0.0 || eu > 0.0,
        "neither order lost a bit over a {KK}-term f32 dot product — the test \
         inputs are exactly representable and this measures nothing"
    );
    // f32 over a 512-term dot product: ~sqrt(K) * eps is ~2.7e-6.
    assert!(ef < 1e-4, "folded relative error {ef:.3e}");
    assert!(eu < 1e-4, "unfolded relative error {eu:.3e}");
    assert!(
        ef < eu * 10.0 + 1e-9,
        "folding is {:.1}x less accurate than not folding ({ef:.3e} vs {eu:.3e})",
        ef / eu.max(1e-30)
    );
}

/// [`run`] with a caller-chosen contraction length.
fn run_k(bank_nk: &[f32], input: &[f32], ids: &[f32], k: usize, extra_reader: bool) -> Vec<f32> {
    let mut g = Graph::new("gmm_fold_k");
    let x = g.input("x", Shape::new(&[M, k], DType::F32));
    let idx = g.input("idx", Shape::new(&[M], DType::F32));
    let bank = g.param("bank", Shape::new(&[E, N, k], DType::F32));
    let bt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![bank],
        Shape::new(&[E, k, N], DType::F32),
    );
    let y = g.add_node(
        Op::GroupedMatMul,
        vec![x, bt, idx],
        Shape::new(&[M, N], DType::F32),
    );
    let mut outputs = vec![y];
    if extra_reader {
        outputs.push(g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![bt, bt],
            Shape::new(&[E, k, N], DType::F32),
        ));
    }
    g.set_outputs(outputs);
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("bank", bank_nk);
    c.run(&[("x", input), ("idx", ids)])
        .into_iter()
        .next()
        .unwrap()
}

/// A folded bank transpose must not reserve arena either.
///
/// Eliding the node from the schedule is only half the win: the memory plan is
/// built first, so without teaching it about the fold the arena still holds a
/// full second copy of every expert bank — written by nobody, read by nobody.
/// On GLM-5.3-Flash that is ~1.88 GB per bank and ~5.6 GB per layer, on nodes
/// the cluster planner sized to hold one copy.
///
/// Checked as a ratio against the bank rather than an absolute, so it does not
/// re-baseline every time the fixture changes size.
#[test]
fn a_folded_bank_transpose_reserves_no_arena() {
    const O: usize = 512;
    const I: usize = 256;
    let bank_bytes = E * O * I * 4;

    let build = |extra_reader: bool| {
        let mut g = Graph::new("arena");
        let x = g.input("x", Shape::new(&[M, I], DType::F32));
        let idx = g.input("idx", Shape::new(&[M], DType::F32));
        let bank = g.param("bank", Shape::new(&[E, O, I], DType::F32));
        let bt = g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            vec![bank],
            Shape::new(&[E, I, O], DType::F32),
        );
        let y = g.add_node(
            Op::GroupedMatMul,
            vec![x, bt, idx],
            Shape::new(&[M, O], DType::F32),
        );
        let mut outs = vec![y];
        if extra_reader {
            outs.push(g.add_node(
                Op::Binary(BinaryOp::Add),
                vec![bt, bt],
                Shape::new(&[E, I, O], DType::F32),
            ));
        }
        g.set_outputs(outs);
        rlx_opt::memory::plan_memory_native_in_order(&g, 64).arena_size
    };

    let folded = build(false);
    let materialized = build(true);
    assert!(
        folded < bank_bytes + bank_bytes / 4,
        "the folded graph reserves {folded} bytes for a {bank_bytes}-byte bank — \
         the transposed copy is still being planned"
    );
    assert!(
        materialized >= folded + bank_bytes,
        "a graph that must materialize the transpose reserved {materialized} \
         against the folded {folded}; the planner is eliding a buffer that is \
         still read, which is silent corruption rather than a saving"
    );
}

/// The small-`m` gate must discriminate, and the planner must honour it.
///
/// Metal's transposed-bank kernel is a decode-path GEMV, so a transpose feeding
/// a PREFILL-size grouped matmul there stays materialized and must keep its
/// buffer. CPU's `sgemm_bt` has no such limit. Both planners read the same
/// predicate with different gates, which is the only arrangement where "what the
/// compiler Nops" and "what the planner drops" cannot drift apart.
#[test]
fn the_small_m_gate_decides_which_transposes_a_metal_style_planner_drops() {
    use rlx_opt::memory::{
        MemoryPlanOptions, is_elidable_bank_transpose, is_elidable_bank_transpose_gated,
        plan_memory_with_options,
    };

    const O: usize = 256;
    const I: usize = 128;
    let bank_bytes = E * O * I * 4;

    let build = |rows: usize| {
        let mut g = Graph::new("gate");
        let x = g.input("x", Shape::new(&[rows, I], DType::F32));
        let idx = g.input("idx", Shape::new(&[rows], DType::F32));
        let bank = g.param("bank", Shape::new(&[E, O, I], DType::F32));
        let bt = g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            vec![bank],
            Shape::new(&[E, I, O], DType::F32),
        );
        let y = g.add_node(
            Op::GroupedMatMul,
            vec![x, bt, idx],
            Shape::new(&[rows, O], DType::F32),
        );
        g.set_outputs(vec![y]);
        (g, bt)
    };

    let metal_opts = MemoryPlanOptions {
        elide_bank_transposes: true,
        elide_requires_small_m: true,
        ..MemoryPlanOptions::inference()
    };

    // Decode: the transposed kernel applies, so the buffer goes.
    let (g, bt) = build(1);
    assert!(is_elidable_bank_transpose_gated(&g, g.node(bt), true));
    let decode = plan_memory_with_options(&g, 128, metal_opts).arena_size;

    // Prefill: it does not, so the buffer must stay.
    let (g, bt) = build(64);
    assert!(
        is_elidable_bank_transpose(&g, g.node(bt)),
        "structurally foldable either way"
    );
    assert!(
        !is_elidable_bank_transpose_gated(&g, g.node(bt), true),
        "a prefill-size reader must NOT be treated as elidable by a backend \
         whose transposed kernel is decode-only — it still executes the \
         transpose, into a buffer the planner would have dropped"
    );
    let prefill = plan_memory_with_options(&g, 128, metal_opts).arena_size;

    assert!(
        prefill >= decode + bank_bytes,
        "prefill plan {prefill} against decode {decode}: the transposed bank \
         ({bank_bytes} bytes) is not being kept for the prefill case"
    );
}
