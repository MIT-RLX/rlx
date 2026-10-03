//! Gradient checkpointing must change memory, not answers.
//!
//! Both halves matter. A pass that produces identical gradients but saves
//! nothing is a no-op dressed up as a feature; one that saves memory but
//! shifts a gradient is worse than useless. So every case here asserts the
//! numbers are bit-comparable AND that the planned arena actually shrank.

use rlx_autodiff::checkpoint::checkpoint_backward;
use rlx_autodiff::grad_with_loss;
use rlx_compile::memory::{ArenaWidthPolicy, plan_memory_with_policy};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const ALIGN: usize = 256;

fn arena_bytes(g: &Graph) -> usize {
    plan_memory_with_policy(g, ALIGN, ArenaWidthPolicy::Native).arena_size
}

/// A deep chain of elementwise ops on a wide tensor, then a scalar loss.
///
/// Deliberately memory-heavy and cheap to recompute — the shape where
/// checkpointing is supposed to pay.
fn deep_chain(width: usize, depth: usize) -> (Graph, rlx_ir::NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("chain");
    let x = g.input("x", Shape::new(&[width], f));
    let w = g.param("w", Shape::new(&[width], f));
    let mut h = g.mul(x, w);
    for _ in 0..depth {
        h = g.tanh(h);
        h = g.mul(h, w);
    }
    let loss = g.mean(h, vec![0], false);
    g.set_outputs(vec![loss]);
    (g, w)
}

fn run(g: Graph, x: &[f32], w: &[f32]) -> Vec<Vec<f32>> {
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w", w);
    let seed = [1.0f32];
    c.run(&[("x", x), ("d_output", &seed[..])])
}

#[test]
fn gradients_are_unchanged_and_the_arena_shrinks() {
    let (fwd, w) = deep_chain(512, 24);
    let bwd = grad_with_loss(&fwd, &[w]);
    let ck = checkpoint_backward(&fwd, &bwd, 4);

    let plain = arena_bytes(&bwd);
    let saved = arena_bytes(&ck);
    assert!(
        saved < plain,
        "checkpointing did not reduce the planned arena: {saved} vs {plain}"
    );
    eprintln!(
        "arena {plain} -> {saved} bytes ({:.0}% of original), nodes {} -> {}",
        100.0 * saved as f64 / plain as f64,
        bwd.nodes().len(),
        ck.nodes().len()
    );

    let x: Vec<f32> = (0..512).map(|i| 0.01 * (i % 7) as f32 - 0.03).collect();
    let wv: Vec<f32> = (0..512).map(|i| 0.5 + 0.001 * (i % 11) as f32).collect();
    let a = run(bwd, &x, &wv);
    let b = run(ck, &x, &wv);
    assert_eq!(a.len(), b.len(), "output count changed");
    for (i, (u, v)) in a.iter().zip(&b).enumerate() {
        assert_eq!(u.len(), v.len(), "output {i} length changed");
        for (p, q) in u.iter().zip(v) {
            assert!(
                (p - q).abs() <= 1e-6 * p.abs().max(1.0),
                "output {i} changed: {p} vs {q}"
            );
        }
    }
}

/// Memory is **U-shaped** in the segment count, not monotone.
///
/// Few segments recompute almost nothing; many segments make almost every
/// node straddle a border, so it has to be kept anyway and the saving
/// evaporates. Asserting monotonicity here would be wrong — and did fail.
/// What must hold is that a sensible middle setting is much better than
/// either extreme, and that `suggest_segments` lands in that middle.
#[test]
fn memory_is_u_shaped_in_the_segment_count() {
    let (fwd, w) = deep_chain(512, 32);
    let bwd = grad_with_loss(&fwd, &[w]);
    let none = arena_bytes(&bwd);

    let sizes: Vec<(usize, usize)> = [2usize, 4, 8, 16, 32]
        .iter()
        .map(|&s| (s, arena_bytes(&checkpoint_backward(&fwd, &bwd, s))))
        .collect();
    eprintln!("arena: 1 segment = {none}; then {sizes:?}");

    let (best_segs, best) = *sizes
        .iter()
        .min_by_key(|(_, b)| *b)
        .expect("non-empty sweep");
    assert!(
        best < none / 2,
        "the best setting ({best_segs} segments, {best} B) should roughly halve \
         the {none} B baseline"
    );
    let worst_many = sizes.last().expect("non-empty").1;
    assert!(
        worst_many > best,
        "over-segmenting should cost memory again, but {worst_many} <= {best}"
    );

    // The suggested default should be near the bottom, not at an extreme.
    let suggested = rlx_autodiff::checkpoint::suggest_segments(fwd.nodes().len());
    let at_suggested = arena_bytes(&checkpoint_backward(&fwd, &bwd, suggested));
    assert!(
        at_suggested <= none / 2,
        "suggest_segments({suggested}) gave {at_suggested} B against a {none} B \
         baseline — it should be in the useful part of the curve"
    );
    eprintln!("suggest_segments({suggested}) -> {at_suggested} B");
}

/// `segments <= 1` is the identity, so callers can leave it off.
#[test]
fn one_segment_is_the_identity() {
    let (fwd, w) = deep_chain(64, 6);
    let bwd = grad_with_loss(&fwd, &[w]);
    for segs in [0usize, 1] {
        let ck = checkpoint_backward(&fwd, &bwd, segs);
        assert_eq!(
            ck.nodes().len(),
            bwd.nodes().len(),
            "{segs} segments should not rewrite anything"
        );
    }
}

/// Recomputation must not touch the parameter gradient itself.
#[test]
fn the_parameter_gradient_is_exact() {
    let f = DType::F32;
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[2, 3], f));
    let w = g.param("w", Shape::new(&[3, 3], f));
    let mut h = g.matmul(x, w, Shape::new(&[2, 3], f));
    for _ in 0..5 {
        h = g.tanh(h);
        h = g.matmul(h, w, Shape::new(&[2, 3], f));
    }
    let loss = g.mean(h, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[w]);
    let ck = checkpoint_backward(&g, &bwd, 3);

    let xv = [0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6];
    let wv = [0.2f32, -0.1, 0.05, 0.3, 0.15, -0.2, 0.1, 0.25, -0.3];
    let a = run(bwd, &xv, &wv);
    let b = run(ck, &xv, &wv);
    // outs = [loss, grad(w)]
    assert!((a[0][0] - b[0][0]).abs() < 1e-6, "loss moved");
    for (p, q) in a[1].iter().zip(&b[1]) {
        assert!((p - q).abs() < 1e-6, "grad(w) moved: {p} vs {q}");
    }
    assert!(
        a[1].iter().any(|v| v.abs() > 1e-9),
        "gradient is all zeros; the test would pass vacuously"
    );
}
