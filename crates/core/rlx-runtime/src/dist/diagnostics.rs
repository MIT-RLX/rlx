//! Cross-backend divergence diagnostic — separate explainable f32 round-off from
//! a real kernel bug by running one graph on the CPU oracle + every other backend.

use crate::{Device, Session};
use rlx_ir::Graph;

// ════════════════════════════════════════════════════════════════════════════
// Cross-backend divergence diagnostic — "explainable drift vs. real kernel bug".
//
// On a heterogeneous cluster the SAME graph runs on different backends (Metal on
// the Mac, CUDA on the box, CPU elsewhere), and their f32 results differ. Most of
// that is bounded round-off — expected, "explainable". But a divergence too large
// to be rounding is a kernel bug (the class that hid the SoftmaxCrossEntropy and
// RMSNorm backward bugs). This runs one graph on the CPU oracle + every other
// local backend and quantifies the gap per backend so the two can be told apart.
// ════════════════════════════════════════════════════════════════════════════

/// One backend's worst divergence from the CPU oracle for a given graph run.
#[derive(Clone, Debug)]
pub struct BackendDivergence {
    pub device: Device,
    /// Largest `|backend - cpu|` over every output element.
    pub max_abs: f32,
    /// Largest `|backend - cpu| / (|cpu| + 1e-6)` over every output element.
    pub max_rel: f32,
    /// Index of the graph output that carried `max_rel`.
    pub worst_output: usize,
    /// `true` when `max_rel` exceeds the tolerance → likely a kernel bug, not
    /// round-off. Investigate with the per-primitive parity tests.
    pub suspect: bool,
}

/// Worst (max_abs, max_rel, worst_output_index) between an oracle and a candidate
/// run's outputs. Split out so the classification is unit-testable without a GPU.
fn divergence_stats(oracle: &[Vec<f32>], got: &[Vec<f32>]) -> (f32, f32, usize) {
    let (mut max_abs, mut max_rel, mut worst) = (0.0f32, 0.0f32, 0usize);
    for (i, (o, g)) in oracle.iter().zip(got).enumerate() {
        for (a, b) in o.iter().zip(g) {
            let abs = (a - b).abs();
            let rel = abs / (a.abs() + 1e-6);
            if abs > max_abs {
                max_abs = abs;
            }
            if rel > max_rel {
                max_rel = rel;
                worst = i;
            }
        }
    }
    (max_abs, max_rel, worst)
}

/// Run `graph` on the CPU oracle and on **every other backend live on this node**,
/// reporting each one's worst divergence from CPU. Call it on each cluster node to
/// localize where distributed numbers drift — and whether that drift is bounded
/// round-off (`suspect == false`) or a candidate kernel bug (`suspect == true`).
///
/// `params` are the graph's weights (fed via `set_param`), `inputs` the run feeds;
/// `tol_rel` flags a backend whose relative error exceeds it (≈`2e-3` for f32
/// forward graphs; backward graphs accumulate more, try `~1e-2`). Returns one
/// entry per non-CPU backend (empty on a CPU-only build — nothing to compare).
pub fn backend_divergence(
    graph: &Graph,
    params: &[(&str, &[f32])],
    inputs: &[(&str, &[f32])],
    tol_rel: f32,
) -> Vec<BackendDivergence> {
    let run_on = |dev: Device| -> Vec<Vec<f32>> {
        let mut s = Session::new(dev).compile(graph.clone());
        for (n, v) in params {
            s.set_param(n, v);
        }
        s.run(inputs)
    };
    let oracle = run_on(Device::Cpu);
    crate::available_devices()
        .into_iter()
        // CPU is the oracle; ANE inference parity is a separate gated route.
        .filter(|&d| d != Device::Cpu && !matches!(d, Device::Ane))
        .map(|dev| {
            let (max_abs, max_rel, worst_output) = divergence_stats(&oracle, &run_on(dev));
            BackendDivergence {
                device: dev,
                max_abs,
                max_rel,
                worst_output,
                suspect: max_rel > tol_rel,
            }
        })
        .collect()
}

/// Print a [`backend_divergence`] report and return `true` if ANY backend looks
/// suspect (relative error past `tol_rel`) — a convenient one-liner for a CLI /
/// CI gate on each node. A green report means the cluster's cross-backend drift
/// is bounded round-off; a red one names the backend + output to investigate.
pub fn report_backend_divergence(
    label: &str,
    graph: &Graph,
    params: &[(&str, &[f32])],
    inputs: &[(&str, &[f32])],
    tol_rel: f32,
) -> bool {
    let report = backend_divergence(graph, params, inputs, tol_rel);
    if report.is_empty() {
        eprintln!("[{label}] backend divergence: CPU-only node — nothing to compare");
        return false;
    }
    let mut any = false;
    for d in &report {
        any |= d.suspect;
        eprintln!(
            "[{label}] {:>6} vs cpu: max_abs={:.3e} max_rel={:.3e} (output #{}) — {}",
            crate::device_label(d.device),
            d.max_abs,
            d.max_rel,
            d.worst_output,
            if d.suspect {
                "SUSPECT (relative error too large for round-off — likely a kernel bug)"
            } else {
                "round-off (explainable)"
            },
        );
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_stats_flags_worst_output() {
        // output 0: exact; output 1: one element off by 0.5 on a base of 2.0.
        let oracle = vec![vec![1.0f32, 2.0], vec![2.0f32, 4.0]];
        let got = vec![vec![1.0f32, 2.0], vec![2.5f32, 4.0]];
        let (abs, rel, worst) = divergence_stats(&oracle, &got);
        assert!((abs - 0.5).abs() < 1e-6, "abs={abs}");
        assert!((rel - 0.25).abs() < 1e-3, "rel={rel}"); // 0.5 / (2.0 + eps)
        assert_eq!(worst, 1, "worst output should be #1");
        // identical runs → zero divergence.
        let (a0, r0, _) = divergence_stats(&oracle, &oracle);
        assert_eq!((a0, r0), (0.0, 0.0));
    }

    #[test]
    fn backend_divergence_cpu_only_is_empty() {
        // On a CPU-only build there is no other backend to compare — the report is
        // empty (and must not panic / compare CPU against itself).
        use rlx_ir::{DType, Graph, Shape};
        let mut g = Graph::new("d");
        let x = g.input("x", Shape::new(&[2, 2], DType::F32));
        let w = g.param("w", Shape::new(&[2, 2], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
        g.set_outputs(vec![mm]);
        let report = backend_divergence(
            &g,
            &[("w", &[1.0, 0.0, 0.0, 1.0])],
            &[("x", &[1.0, 2.0, 3.0, 4.0])],
            2e-3,
        );
        // CPU is filtered out as the oracle; on a CPU-only test build nothing else
        // is live, so the report is empty. (On a GPU build it would hold entries.)
        assert!(report.iter().all(|d| d.device != Device::Cpu));
    }
}
