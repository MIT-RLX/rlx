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

//! Backward parity for the COMPACT `Op::SelectiveScan` lowering.
//!
//! The autodiff pre-pass lowers `Op::SelectiveScan` into a single
//! `Op::Scan { save_trajectory: true }` surrounded by S-independent
//! elementwise pre/post graphs (see
//! `rlx_fusion::unfuse::rnn::unfuse_selective_scan_scan`). The generic
//! scan VJP then produces `Op::ScanBackward` / `Op::ScanBackwardXs`
//! instead of a ~20·S-node per-timestep unroll.
//!
//! Gates (CPU-first; all must be green):
//!   1. Compact backward grads == legacy per-step-unroll backward grads
//!      (`RLX_SELSCAN_LEGACY_UNROLL=1`) — proves the new VJP equals the
//!      reference backward.
//!   2. Compact analytic grads == central finite-differences of the
//!      scalar reference recurrence.
//!   3. Scan-lowered forward y == native `Op::SelectiveScan` y.
//!   4. At S=1001 the backward graph is compact (`Op::ScanBackward`
//!      present, total nodes < 2000; was ~237k with the unroll).
//!
//! NB: all gates live in ONE `#[test]` fn on purpose — Gate 1 toggles
//! the process-global `RLX_SELSCAN_LEGACY_UNROLL` env var, so splitting
//! into parallel tests would race on it.

#![cfg(feature = "cpu")]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

const F: DType = DType::F32;

/// Deterministic inputs. Δ ∈ (0, 0.5) keeps exp(Δ·A) bounded — the
/// realistic Mamba regime.
fn ssm_inputs(b: usize, s: usize, h: usize, n: usize) -> [Vec<f32>; 5] {
    let nx = b * s * h;
    let na = h * n;
    let nb = b * s * n;
    let x: Vec<f32> = (0..nx).map(|i| 0.1 + 0.05 * ((i % 13) as f32)).collect();
    let delta: Vec<f32> = (0..nx).map(|i| 0.05 + 0.03 * ((i % 7) as f32)).collect();
    let a: Vec<f32> = (0..na).map(|i| -0.5 + 0.1 * ((i % 11) as f32)).collect();
    let bd: Vec<f32> = (0..nb).map(|i| 0.1 + 0.03 * ((i % 9) as f32)).collect();
    let cd: Vec<f32> = (0..nb).map(|i| 0.2 + 0.04 * ((i % 5) as f32)).collect();
    [x, delta, a, bd, cd]
}

/// SelectiveScan forward graph, all five operands as named inputs.
/// Returns `(graph, y_node, wrt=[x,delta,a,b,c])`.
fn build_ssm(b: usize, s: usize, h: usize, n: usize) -> (Graph, NodeId, Vec<NodeId>) {
    let mut g = Graph::new("ssm_bwd");
    let bsh = Shape::new(&[b, s, h], F);
    let hn = Shape::new(&[h, n], F);
    let bsn = Shape::new(&[b, s, n], F);
    let x = g.input("x", bsh.clone());
    let delta = g.input("delta", bsh.clone());
    let a = g.input("a", hn);
    let b_in = g.input("b", bsn.clone());
    let c_in = g.input("c", bsn);
    let y = g.selective_scan(x, delta, a, b_in, c_in, n, bsh);
    (g, y, vec![x, delta, a, b_in, c_in])
}

/// Same graph with a `loss = Σ y` head (scalar `[1]`).
fn build_ssm_loss(b: usize, s: usize, h: usize, n: usize) -> (Graph, Vec<NodeId>) {
    let (mut g, y, wrt) = build_ssm(b, s, h, n);
    let flat = g.reshape(y, vec![(b * s * h) as i64], Shape::new(&[b * s * h], F));
    let loss = g.reduce(flat, ReduceOp::Sum, vec![0], false, Shape::new(&[1], F));
    g.set_outputs(vec![loss]);
    (g, wrt)
}

/// Compile+run a backward graph on CPU → `[loss, dx, ddelta, da, db, dc]`.
fn run_grads(bwd: &Graph, inp: &[Vec<f32>; 5]) -> Vec<Vec<f32>> {
    let mut sess = Session::new(Device::Cpu).compile_with(bwd.clone(), &CompileOptions::new());
    let seed = [1.0f32];
    sess.run(&[
        ("x", inp[0].as_slice()),
        ("delta", inp[1].as_slice()),
        ("a", inp[2].as_slice()),
        ("b", inp[3].as_slice()),
        ("c", inp[4].as_slice()),
        ("d_output", &seed),
    ])
}

/// Scalar reference: `loss = Σ_{b,s,h} y`, computed in f64. Matches
/// `execute_selective_scan_f32` exactly (state resets per batch row).
fn ref_loss(inp: &[Vec<f32>; 5], b: usize, s: usize, h: usize, n: usize) -> f64 {
    let (x, delta, a, bd, cd) = (&inp[0], &inp[1], &inp[2], &inp[3], &inp[4]);
    let mut loss = 0.0f64;
    let mut state = vec![0.0f64; h * n];
    for bi in 0..b {
        for v in state.iter_mut() {
            *v = 0.0;
        }
        for si in 0..s {
            for ci in 0..h {
                let d = delta[bi * s * h + si * h + ci] as f64;
                let xv = x[bi * s * h + si * h + ci] as f64;
                let mut acc = 0.0f64;
                for ni in 0..n {
                    let da = (d * a[ci * n + ni] as f64).exp();
                    state[ci * n + ni] =
                        da * state[ci * n + ni] + d * bd[bi * s * n + si * n + ni] as f64 * xv;
                    acc += cd[bi * s * n + si * n + ni] as f64 * state[ci * n + ni];
                }
                loss += acc;
            }
        }
    }
    loss
}

/// Central finite-difference gradient of `ref_loss` w.r.t. input array
/// `k` (0=x,1=delta,2=a,3=b,4=c).
fn fd_grad(
    inp: &[Vec<f32>; 5],
    k: usize,
    eps: f32,
    b: usize,
    s: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; inp[k].len()];
    for i in 0..inp[k].len() {
        let mut plus = inp.clone();
        let mut minus = inp.clone();
        plus[k][i] += eps;
        minus[k][i] -= eps;
        let lp = ref_loss(&plus, b, s, h, n);
        let lm = ref_loss(&minus, b, s, h, n);
        out[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
    }
    out
}

fn set_legacy(on: bool) {
    // SAFETY: single-threaded within this test; no other thread reads
    // the var while we mutate it (all gates run in one #[test]).
    unsafe {
        if on {
            std::env::set_var("RLX_SELSCAN_LEGACY_UNROLL", "1");
        } else {
            std::env::remove_var("RLX_SELSCAN_LEGACY_UNROLL");
        }
    }
}

#[test]
fn selective_scan_backward_all_gates() {
    let (b, s, h, n) = (1usize, 5, 2, 3);
    let inp = ssm_inputs(b, s, h, n);

    // ── GATE 1: compact backward == legacy-unroll backward ───────────
    set_legacy(false);
    let (g_c, wrt_c) = build_ssm_loss(b, s, h, n);
    let bwd_compact = grad_with_loss(&g_c, &wrt_c);
    assert!(
        bwd_compact
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::ScanBackward { .. })),
        "compact backward must contain Op::ScanBackward"
    );
    let out_compact = run_grads(&bwd_compact, &inp);

    set_legacy(true);
    let (g_l, wrt_l) = build_ssm_loss(b, s, h, n);
    let bwd_legacy = grad_with_loss(&g_l, &wrt_l);
    assert!(
        !bwd_legacy
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::ScanBackward { .. } | Op::Scan { .. })),
        "legacy backward must be a fully unrolled primitive chain (no Scan/ScanBackward)"
    );
    let out_legacy = run_grads(&bwd_legacy, &inp);
    set_legacy(false);

    let names = ["dx", "ddelta", "da", "db", "dc"];
    let mut gate1_max = 0f32;
    // loss must match too.
    let dloss = (out_compact[0][0] - out_legacy[0][0]).abs();
    assert!(dloss <= 1e-5, "GATE1 loss mismatch: {dloss:e}");
    for k in 0..5 {
        let (cc, ll) = (&out_compact[1 + k], &out_legacy[1 + k]);
        assert_eq!(cc.len(), ll.len(), "GATE1 {} length mismatch", names[k]);
        for (i, (a, b)) in cc.iter().zip(ll.iter()).enumerate() {
            let d = (a - b).abs();
            gate1_max = gate1_max.max(d);
            assert!(
                d <= 1e-5,
                "GATE1 {}[{i}] compact={a} legacy={b} |Δ|={d:e} > 1e-5",
                names[k]
            );
        }
    }
    eprintln!("[GATE 1] compact-vs-legacy backward max|Δ| = {gate1_max:e} (≤ 1e-5) ✓");

    // ── GATE 2: compact analytic grads == finite differences ─────────
    let eps = 1e-3f32;
    let mut gate2_max_rel = 0f32;
    for k in 0..5 {
        let fd = fd_grad(&inp, k, eps, b, s, h, n);
        let an = &out_compact[1 + k];
        assert_eq!(fd.len(), an.len());
        for (i, (a, f)) in an.iter().zip(fd.iter()).enumerate() {
            let abs = (a - f).abs();
            let rel = abs / f.abs().max(1e-4);
            gate2_max_rel = gate2_max_rel.max(rel);
            assert!(
                abs < 2e-3 || rel < 1e-2,
                "GATE2 {}[{i}] analytic={a} fd={f} (abs {abs:e}, rel {rel:e})",
                names[k]
            );
        }
    }
    eprintln!("[GATE 2] analytic-vs-finite-diff max rel err = {gate2_max_rel:e} (< 1e-2) ✓");

    // ── GATE 3: scan-lowered forward == native SelectiveScan forward ─
    set_legacy(false);
    let (mut g_native, y_native, _) = build_ssm(b, s, h, n);
    g_native.set_outputs(vec![y_native]);
    let native_y = Session::new(Device::Cpu)
        .compile(g_native.clone())
        .run(&[
            ("x", inp[0].as_slice()),
            ("delta", inp[1].as_slice()),
            ("a", inp[2].as_slice()),
            ("b", inp[3].as_slice()),
            ("c", inp[4].as_slice()),
        ])
        .pop()
        .unwrap();
    // The compact scan-lowered forward (default unfuse path).
    let unfused = rlx_fusion::unfuse_fused_for_autodiff(g_native);
    assert!(
        unfused
            .nodes()
            .iter()
            .filter(|nd| matches!(nd.op, Op::Scan { .. }))
            .count()
            == 1,
        "scan-lowered forward must have exactly one Op::Scan"
    );
    let scan_y = Session::new(Device::Cpu)
        .compile(unfused)
        .run(&[
            ("x", inp[0].as_slice()),
            ("delta", inp[1].as_slice()),
            ("a", inp[2].as_slice()),
            ("b", inp[3].as_slice()),
            ("c", inp[4].as_slice()),
        ])
        .pop()
        .unwrap();
    let mut gate3_max = 0f32;
    for (i, (nv, sv)) in native_y.iter().zip(scan_y.iter()).enumerate() {
        let d = (nv - sv).abs();
        gate3_max = gate3_max.max(d);
        assert!(d <= 1e-5, "GATE3 y[{i}] native={nv} scan={sv} |Δ|={d:e}");
    }
    eprintln!("[GATE 3] scan-vs-native forward max|Δ| = {gate3_max:e} (≤ 1e-5) ✓");

    // ── GATE 4: S=1001 backward is compact ───────────────────────────
    set_legacy(false);
    let (g_long, wrt_long) = build_ssm_loss(1, 1001, 2, 3);
    let bwd_long = grad_with_loss(&g_long, &wrt_long);
    let n_nodes = bwd_long.nodes().len();
    let has_sb = bwd_long
        .nodes()
        .iter()
        .any(|nd| matches!(nd.op, Op::ScanBackward { .. }));
    assert!(has_sb, "GATE4: expected Op::ScanBackward at S=1001");
    assert!(
        n_nodes < 2000,
        "GATE4: backward graph must stay compact at S=1001, got {n_nodes} nodes"
    );
    eprintln!("[GATE 4] S=1001 backward: {n_nodes} nodes, ScanBackward present (was ~237k) ✓");
}
