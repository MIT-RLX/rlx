// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Hardware-aware placement** — split a layer stack into contiguous per-node
//! stages that (a) FIT each node's RAM/VRAM budget and (b) balance the pipeline.
//! This replaces hand-tuned `--layers 0:15` splits: give it the model's resident
//! cost per layer and the probed [`NodeCaps`], get an assignment back.

use super::caps::NodeCaps;
use super::config::{NodeConfig, PlacementPolicy};
use anyhow::{Result, bail};
use std::ops::Range;

/// Resident-memory + compute cost model for a layer-stack model. Bytes are the
/// ACTUAL per-node resident footprint (packed weights + kept scales), so the
/// planner's budgets are meaningful.
#[derive(Debug, Clone)]
pub struct ModelCost {
    pub n_layers: usize,
    /// Average resident bytes per transformer layer.
    pub per_layer_bytes: u64,
    /// Extra bytes on the FIRST stage (token embedding).
    pub embed_bytes: u64,
    /// Extra bytes on the LAST stage (final norm + LM head).
    pub head_bytes: u64,
    /// Relative FLOPs per layer (for the throughput policy; 1.0 = uniform).
    pub per_layer_flops: f64,
}

impl ModelCost {
    pub fn total_bytes(&self) -> u64 {
        self.embed_bytes + self.head_bytes + self.per_layer_bytes * self.n_layers as u64
    }
}

/// One node's stage.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub addr: String,
    pub ssh: Option<String>,
    pub layers: Range<usize>,
    pub first: bool,
    pub last: bool,
    /// Estimated resident bytes for this stage (weights + embed/head).
    pub est_bytes: u64,
    /// The node's stage RAM budget the planner used.
    pub budget_bytes: u64,
    /// Primary device label for the monitor.
    pub device: String,
}

/// Fraction of a resident GPU's reported memory usable for one stage's largest
/// single allocation — the rest is driver reserve + arena/weight-buffer slack.
const GPU_ALLOC_SAFETY: f64 = 0.85;

/// A node's usable stage-RAM budget. `max_ram_gb`, when set, is the EXPLICIT
/// budget (the operator's control — overrides a pessimistic `ram_avail`, e.g.
/// macOS reporting reclaimable cache as used); otherwise the probed available
/// RAM is used. A *resident* discrete-GPU primary further caps at its VRAM; a
/// *paged* GPU (CUDA managed memory, or Apple/iGPU unified memory) migrates from
/// host RAM and so stays bounded by RAM. The reserve headroom is subtracted last.
fn budget_bytes(caps: &NodeCaps, cfg: &NodeConfig, reserve: u64) -> u64 {
    let mut b = match cfg.max_ram_gb {
        Some(gb) => ((gb * 1e9) as u64).min(caps.ram_total),
        None => caps.ram_avail,
    };
    let dev = cfg.primary_device();
    // CUDA runs on managed (paged) memory here — the stage migrates over PCIe from
    // host RAM, so it is bounded by RAM, not VRAM. Other discrete GPUs are resident.
    let paged = matches!(dev, rlx_runtime::Device::Cuda);
    if dev != rlx_runtime::Device::Cpu && !paged {
        let vram = caps.accel_mem();
        if vram > 0 && vram < caps.ram_total {
            // A single VkDeviceMemory (or discrete cudaMalloc) can't consume the
            // *whole* reported ceiling: the driver reserves some, the mapped
            // activation arena + weight buffer are separate allocations, and
            // alignment adds slack. Cap the stage at 85% of the GPU's memory so
            // its largest single allocation actually succeeds instead of OOM-ing
            // right at the ceiling (an amdgpu APU's GTT is especially tight).
            b = b.min((vram as f64 * GPU_ALLOC_SAFETY) as u64);
        }
    }
    b.saturating_sub(reserve)
}

/// Max whole layers a node can hold given its byte budget and the per-stage
/// overhead it must also carry (embed on the first node, head on the last).
fn layer_cap(budget: u64, per_layer: u64, overhead: u64) -> usize {
    if per_layer == 0 {
        return usize::MAX;
    }
    (budget.saturating_sub(overhead) / per_layer) as usize
}

/// Plan contiguous stages. `nodes` is in pipeline order (first node runs the
/// embedding, last runs the head). Returns one [`Assignment`] per node.
pub fn plan_placement(
    model: &ModelCost,
    nodes: &[(NodeCaps, NodeConfig)],
    policy: PlacementPolicy,
    reserve_bytes: u64,
) -> Result<Vec<Assignment>> {
    let k = nodes.len();
    if k == 0 {
        bail!("no nodes");
    }
    // Manual: honour each node's explicit range.
    if policy == PlacementPolicy::Manual {
        let mut out = Vec::new();
        for (i, (caps, cfg)) in nodes.iter().enumerate() {
            let r = cfg.manual_range().ok_or_else(|| anyhow::anyhow!("node {} has no `layers` for manual policy", cfg.addr))?;
            out.push(mk_assignment(model, caps, cfg, r, i == 0, i == k - 1, reserve_bytes));
        }
        return Ok(out);
    }

    let budgets: Vec<u64> = nodes.iter().map(|(c, cfg)| budget_bytes(c, cfg, reserve_bytes)).collect();
    // Per-node capacity in layers (first pays embed, last pays head).
    let caps_layers: Vec<usize> = (0..k)
        .map(|i| {
            let overhead = if i == 0 { model.embed_bytes } else { 0 } + if i == k - 1 { model.head_bytes } else { 0 };
            layer_cap(budgets[i], model.per_layer_bytes, overhead)
        })
        .collect();
    let total_cap: usize = caps_layers.iter().copied().map(|c| c.min(model.n_layers)).sum();
    if total_cap < model.n_layers {
        bail!(
            "model ({} layers, {:.1} GB) does not fit the cluster (capacity {} layers). \
             Lower precision, add a node, raise max_ram, or reduce reserve.",
            model.n_layers,
            model.total_bytes() as f64 / 1e9,
            total_cap
        );
    }

    // Weight each node by budget (ram_balanced) or throughput (gflops).
    let weight: Vec<f64> = match policy {
        PlacementPolicy::Throughput => nodes.iter().map(|(c, _)| c.gflops.max(1.0) * model.per_layer_flops.max(0.001)).collect(),
        _ => budgets.iter().map(|&b| b as f64).collect(),
    };
    let wsum: f64 = weight.iter().sum();

    // Proportional target, clamped to per-node layer cap, remainder water-filled.
    let mut counts: Vec<usize> = (0..k).map(|i| ((model.n_layers as f64 * weight[i] / wsum).round() as usize).min(caps_layers[i])).collect();
    let mut assigned: usize = counts.iter().sum();
    // Add under-target: give leftover layers to nodes with remaining capacity,
    // most spare first. Remove over-target similarly.
    while assigned < model.n_layers {
        let i = (0..k).filter(|&i| counts[i] < caps_layers[i]).max_by_key(|&i| caps_layers[i] - counts[i]);
        match i {
            Some(i) => {
                counts[i] += 1;
                assigned += 1;
            }
            None => bail!("placement: could not distribute remaining layers"),
        }
    }
    while assigned > model.n_layers {
        let i = (0..k).filter(|&i| counts[i] > 0).max_by_key(|&i| counts[i]);
        match i {
            Some(i) => {
                counts[i] -= 1;
                assigned -= 1;
            }
            None => break,
        }
    }

    let mut out = Vec::with_capacity(k);
    let mut start = 0usize;
    for (i, ((caps, cfg), &cnt)) in nodes.iter().zip(&counts).enumerate() {
        let r = start..start + cnt;
        start += cnt;
        out.push(mk_assignment(model, caps, cfg, r, i == 0, i == k - 1, reserve_bytes));
    }
    Ok(out)
}

fn mk_assignment(
    model: &ModelCost,
    caps: &NodeCaps,
    cfg: &NodeConfig,
    layers: Range<usize>,
    first: bool,
    last: bool,
    reserve: u64,
) -> Assignment {
    let n = (layers.end - layers.start) as u64;
    let est = n * model.per_layer_bytes + if first { model.embed_bytes } else { 0 } + if last { model.head_bytes } else { 0 };
    Assignment {
        addr: cfg.addr.clone(),
        ssh: cfg.ssh.clone(),
        layers,
        first,
        last,
        est_bytes: est,
        budget_bytes: budget_bytes(caps, cfg, reserve),
        device: rlx_runtime::device_label(cfg.primary_device()).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(addr: &str, ram_gb: f64, gflops: f64) -> NodeCaps {
        NodeCaps {
            addr: addr.into(),
            os: "linux".into(),
            cores: 8,
            ram_total: (ram_gb * 1e9) as u64,
            ram_avail: (ram_gb * 1e9) as u64,
            disk_free: 500_000_000_000,
            devices: vec![],
            gflops,
            io_mbps: 0.0,
        }
    }
    fn node(addr: &str) -> NodeConfig {
        NodeConfig {
            addr: addr.into(),
            ssh: None,
            ckpt_dir: "/x".into(),
            device: "cpu".into(),
            precision: "bf16".into(),
            kv_cache: Default::default(),
            rng_seed: None,
            max_ram_gb: None,
            layers: None,
        }
    }

    #[test]
    fn ram_balanced_fits_and_covers_all_layers() {
        // 43 layers @ ~3GB each ≈ 129GB across 3 uneven nodes.
        let model = ModelCost { n_layers: 43, per_layer_bytes: 3_000_000_000, embed_bytes: 200_000_000, head_bytes: 200_000_000, per_layer_flops: 1.0 };
        let nodes = vec![
            (caps("a", 60.0, 100.0), node("a")),
            (caps("b", 55.0, 90.0), node("b")),
            (caps("c", 44.0, 60.0), node("c")),
        ];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::RamBalanced, 5_000_000_000).unwrap();
        // Contiguous + complete cover of 0..43.
        assert_eq!(plan[0].layers.start, 0);
        assert_eq!(plan.last().unwrap().layers.end, 43);
        for w in plan.windows(2) {
            assert_eq!(w[0].layers.end, w[1].layers.start);
        }
        // Every stage within budget.
        for a in &plan {
            assert!(a.est_bytes <= a.budget_bytes, "{} over budget: {} > {}", a.addr, a.est_bytes, a.budget_bytes);
        }
    }

    #[test]
    fn rejects_when_too_big() {
        let model = ModelCost { n_layers: 100, per_layer_bytes: 5_000_000_000, embed_bytes: 0, head_bytes: 0, per_layer_flops: 1.0 };
        let nodes = vec![(caps("a", 40.0, 100.0), node("a"))];
        assert!(plan_placement(&model, &nodes, PlacementPolicy::RamBalanced, 5_000_000_000).is_err());
    }
}
