// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Topology-aware parallelism planner.
//!
//! Distribution only pays off when the compute it removes outweighs the
//! communication it adds. Reduced to one number, the deciding ratio is
//! **FLOPs of useful work per byte communicated** versus the machine's
//! **compute-to-bandwidth ratio** `R/β`. This module turns measured
//! device throughput + a measured [`Link`] (e.g. from the `dist_node`
//! bench) + a [`Workload`] into a recommended [`Strategy`] — the
//! "topology-aware auto-parallel" decision, made explicit and testable.
//!
//! The models (per optimizer step, `n` ranks):
//!   * **Data-parallel** — one gradient all-reduce/step; a ring moves
//!     `2·(n-1)/n · 4·params` bytes per rank. Compute is `6·params·tokens`
//!     FLOPs (fwd+bwd). DP is worth it when comm ≪ compute.
//!   * **Tensor-parallel** — two activation all-reduces *per layer*; comm
//!     scales with `tokens·d_model·n_layers`, so it needs a fast link.
//!   * **Pipeline** — one activation hand-off per stage boundary; cheap
//!     comm, the right tool for a slow link or a model too big for one
//!     device.
//!
//! Everything is `f64`, no allocation in the hot path, no backend deps —
//! the same decision logic a runtime auto-planner or a CLI would call.

/// A point-to-point link between two ranks.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    /// Effective payload bandwidth in bytes/second (e.g. the large-message
    /// asymptote from the collective bench, not the NIC line rate).
    pub bandwidth_bytes_per_s: f64,
    /// Per-collective latency floor in seconds (e.g. measured barrier RTT).
    pub latency_s: f64,
}

impl Link {
    /// Time to all-reduce `bytes` over a ring of `n` ranks:
    /// `latency + 2·(n-1)/n · bytes / bandwidth`.
    pub fn all_reduce_time(&self, bytes: f64, n: u32) -> f64 {
        let n = n.max(1) as f64;
        let moved = if n <= 1.0 {
            0.0
        } else {
            2.0 * (n - 1.0) / n * bytes
        };
        self.latency_s + moved / self.bandwidth_bytes_per_s
    }

    /// Time to send `bytes` once (pipeline hand-off): `latency + bytes/bw`.
    pub fn send_time(&self, bytes: f64) -> f64 {
        self.latency_s + bytes / self.bandwidth_bytes_per_s
    }
}

/// One participating accelerator.
#[derive(Clone, Copy, Debug)]
pub struct Device {
    /// Sustained throughput in FLOP/s for the workload's dtype.
    pub flops_per_s: f64,
    /// Usable memory in bytes (for the fits-on-one-device check).
    pub mem_bytes: u64,
}

/// The model + step shape we are planning for.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// Trainable/!inference parameter count.
    pub params: u64,
    /// Tokens processed per optimizer step **across all ranks**.
    pub tokens_per_step: u64,
    pub d_model: u32,
    pub n_layers: u32,
    /// Bytes per parameter resident on device (weights + optimizer state).
    /// 4 for fp32 inference; ~16 for fp32 Adam training.
    pub bytes_per_param: u32,
}

/// Parallelism strategies the planner can recommend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Keep it on one device — comm would cost more than it saves.
    Single,
    /// Shard the batch; all-reduce gradients once per step.
    DataParallel,
    /// Shard each layer; all-reduce activations within every layer.
    TensorParallel,
    /// Shard by layer ranges; hand activations across stage boundaries.
    Pipeline,
}

/// A recommendation with the numbers behind it.
#[derive(Clone, Debug)]
pub struct Plan {
    pub strategy: Strategy,
    pub ranks: u32,
    /// Predicted end-to-end speedup vs. one device (×). `< 1` means slower.
    pub speedup: f64,
    /// Whether the model even fits on a single device.
    pub fits_on_one: bool,
    pub rationale: String,
}

/// Machine compute-to-bandwidth ratio in FLOP/byte — the bar a scheme's
/// useful-FLOP-per-byte must clear to be worth distributing.
pub fn compute_per_byte(dev: &Device, link: &Link) -> f64 {
    dev.flops_per_s / link.bandwidth_bytes_per_s
}

/// Per-rank work fractions for a **heterogeneous** mesh, proportional to each
/// device's throughput — give the faster accelerator more of the batch (DP)
/// or more layers (PP). Sums to 1. This is the compute analog of exo's
/// memory-weighted partitioning, and the reason a CPU + a GPU + an NPU can
/// share one job without the slowest dragging the rest to its own share.
pub fn capacity_weights(devices: &[Device]) -> Vec<f64> {
    let total: f64 = devices.iter().map(|d| d.flops_per_s).sum();
    if total <= 0.0 || devices.is_empty() {
        let n = devices.len().max(1) as f64;
        return vec![1.0 / n; devices.len()];
    }
    devices.iter().map(|d| d.flops_per_s / total).collect()
}

/// Recommend a parallelism strategy. `devices` are the per-rank accelerators
/// (assumed homogeneous for the speedup model; the slowest bounds compute).
pub fn recommend(devices: &[Device], link: Link, w: Workload) -> Plan {
    let n = devices.len().max(1) as u32;
    let slow = devices
        .iter()
        .fold(f64::INFINITY, |a, d| a.min(d.flops_per_s));
    let dev = Device {
        flops_per_s: slow,
        mem_bytes: devices.iter().map(|d| d.mem_bytes).min().unwrap_or(0),
    };

    let model_bytes = w.params * w.bytes_per_param as u64;
    let fits_on_one = model_bytes <= dev.mem_bytes || dev.mem_bytes == 0;

    if n <= 1 {
        return Plan {
            strategy: Strategy::Single,
            ranks: 1,
            speedup: 1.0,
            fits_on_one,
            rationale: "single device — nothing to distribute".into(),
        };
    }

    // ── Data-parallel model ──
    // compute/step (fwd+bwd ≈ 6·P·T), then halved per rank under DP.
    let compute_step = 6.0 * w.params as f64 * w.tokens_per_step as f64 / dev.flops_per_s;
    let compute_step_per_rank = compute_step / n as f64;
    let grad_bytes = 4.0 * w.params as f64; // gradients are fp32
    let dp_comm = link.all_reduce_time(grad_bytes, n);
    // Overlap hides comm under compute up to the compute time available.
    let dp_exposed = (dp_comm - compute_step_per_rank).max(0.0);
    let dp_time = compute_step_per_rank + dp_exposed;
    let dp_speedup = compute_step / dp_time;

    // ── Tensor-parallel model ──
    // 2 activation all-reduces per layer; activation = tokens·d_model fp32.
    let act_bytes = 4.0 * w.tokens_per_step as f64 * w.d_model as f64;
    let tp_comm = 2.0 * w.n_layers as f64 * link.all_reduce_time(act_bytes, n);
    let tp_time = compute_step / n as f64 + tp_comm;
    let tp_speedup = compute_step / tp_time;

    // ── Pipeline model ──
    // One activation hand-off per stage boundary (n-1 of them), per step.
    let pp_comm = (n as f64 - 1.0) * link.send_time(act_bytes);
    let pp_time = compute_step / n as f64 + pp_comm;
    let pp_speedup = compute_step / pp_time;

    // If it doesn't fit on one device, distributing is mandatory: pick the
    // fastest of the splitting strategies (pipeline usually wins on a slow
    // link because its comm is smallest).
    if !fits_on_one {
        let (strategy, speedup) = best_of([
            (Strategy::Pipeline, pp_speedup),
            (Strategy::TensorParallel, tp_speedup),
            (Strategy::DataParallel, dp_speedup),
        ]);
        return Plan {
            strategy,
            ranks: n,
            speedup,
            fits_on_one,
            rationale: format!(
                "model needs {:.1} GB > {:.1} GB/device — must split; \
                 {strategy:?} is fastest (DP {dp_speedup:.2}×, TP {tp_speedup:.2}×, \
                 PP {pp_speedup:.2}×)",
                model_bytes as f64 / 1e9,
                dev.mem_bytes as f64 / 1e9,
            ),
        };
    }

    // Fits on one device: only distribute if some strategy beats 1× by a
    // worthwhile margin (overhead/complexity isn't free). Pipeline is left
    // out here — for a model that fits it's a *capacity* tool, and its
    // idealized compute/n speedup needs micro-batching to fill the pipe, so
    // recommending it for raw speed would be dishonest. DP/TP it is.
    let _ = pp_speedup;
    let (strategy, speedup) = best_of([
        (Strategy::DataParallel, dp_speedup),
        (Strategy::TensorParallel, tp_speedup),
        (Strategy::TensorParallel, tp_speedup),
    ]);
    if speedup > 1.15 {
        Plan {
            strategy,
            ranks: n,
            speedup,
            fits_on_one,
            rationale: format!(
                "{strategy:?} clears the bar: {speedup:.2}× (DP {dp_speedup:.2}×, \
                 TP {tp_speedup:.2}×, PP {pp_speedup:.2}×; {:.0e} FLOP/byte machine ratio)",
                compute_per_byte(&dev, &link),
            ),
        }
    } else {
        Plan {
            strategy: Strategy::Single,
            ranks: 1,
            speedup: 1.0,
            fits_on_one,
            rationale: format!(
                "stay single — best split only {speedup:.2}× over this link \
                 (need >{:.0e} FLOP/byte of work per sync; DP {dp_speedup:.2}×, \
                 TP {tp_speedup:.2}×, PP {pp_speedup:.2}×)",
                compute_per_byte(&dev, &link),
            ),
        }
    }
}

fn best_of(cands: [(Strategy, f64); 3]) -> (Strategy, f64) {
    cands
        .into_iter()
        .fold((Strategy::Single, f64::NEG_INFINITY), |best, c| {
            if c.1 > best.1 { c } else { best }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ~NVIDIA GPU class, fp32 sustained.
    fn gpu() -> Device {
        Device {
            flops_per_s: 15e12,
            mem_bytes: 16u64 << 30,
        }
    }
    // The WiFi link the dist_node bench measured (ring): ~80 MB/s, ~1.7 ms.
    fn wifi() -> Link {
        Link {
            bandwidth_bytes_per_s: 80e6,
            latency_s: 1.7e-3,
        }
    }
    // A Thunderbolt/RDMA-class link: ~2 GB/s, ~50 µs.
    fn fast() -> Link {
        Link {
            bandwidth_bytes_per_s: 2e9,
            latency_s: 50e-6,
        }
    }

    #[test]
    fn small_batch_over_wifi_stays_single() {
        // 0.6B params, only 4k tokens/step → gradient sync dwarfs compute.
        let p = recommend(
            &[gpu(), gpu()],
            wifi(),
            Workload {
                params: 600_000_000,
                tokens_per_step: 4096,
                d_model: 1024,
                n_layers: 28,
                bytes_per_param: 16,
            },
        );
        assert_eq!(p.strategy, Strategy::Single, "{}", p.rationale);
        assert!(p.speedup <= 1.0 + 1e-9);
    }

    #[test]
    fn huge_batch_over_wifi_enables_data_parallel() {
        // Same model, but ~2M tokens/step amortizes the gradient sync.
        let p = recommend(
            &[gpu(), gpu()],
            wifi(),
            Workload {
                params: 600_000_000,
                tokens_per_step: 2_000_000,
                d_model: 1024,
                n_layers: 28,
                bytes_per_param: 16,
            },
        );
        assert_eq!(p.strategy, Strategy::DataParallel, "{}", p.rationale);
        assert!(p.speedup > 1.3, "speedup {} ({})", p.speedup, p.rationale);
    }

    #[test]
    fn fast_link_enables_distribution() {
        // On a Thunderbolt/RDMA-class link a modest batch already clears the
        // 1.15× bar (DP, with TP now affordable too), where WiFi could not.
        let p = recommend(
            &[gpu(), gpu()],
            fast(),
            Workload {
                params: 600_000_000,
                tokens_per_step: 8192,
                d_model: 1024,
                n_layers: 28,
                bytes_per_param: 16,
            },
        );
        assert_ne!(p.strategy, Strategy::Single, "{}", p.rationale);
        assert!(p.speedup > 1.15, "{}", p.rationale);
    }

    #[test]
    fn oversized_model_must_split_even_if_slow() {
        // A model far bigger than one device's memory → must distribute;
        // pipeline's tiny comm wins on the slow link.
        let p = recommend(
            &[gpu(), gpu()],
            wifi(),
            Workload {
                params: 30_000_000_000, // 30B × 2B (fp16 infer) = 60 GB ≫ 16 GB
                tokens_per_step: 2048,
                d_model: 6144,
                n_layers: 64,
                bytes_per_param: 2,
            },
        );
        assert!(!p.fits_on_one);
        assert_eq!(p.strategy, Strategy::Pipeline, "{}", p.rationale);
    }

    #[test]
    fn machine_ratio_is_flops_over_bandwidth() {
        let r = compute_per_byte(&gpu(), &wifi());
        assert!((r - 15e12 / 80e6).abs() < 1.0);
    }

    #[test]
    fn capacity_weights_favor_the_faster_device() {
        let fast = Device {
            flops_per_s: 30e12,
            mem_bytes: 0,
        };
        let slow = Device {
            flops_per_s: 10e12,
            mem_bytes: 0,
        };
        let w = capacity_weights(&[fast, slow]);
        assert!((w[0] - 0.75).abs() < 1e-9, "{w:?}");
        assert!((w[1] - 0.25).abs() < 1e-9, "{w:?}");
        // Heterogeneous mesh: a slowest-bounds-throughput recommendation must
        // still use the SLOW device for its DP compute model.
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}
