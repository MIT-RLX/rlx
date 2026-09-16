// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Hardware-aware placement** — split a layer stack into contiguous per-node
//! stages that (a) FIT each node's RAM/VRAM budget and (b) balance the pipeline.
//! This replaces hand-tuned `--layers 0:15` splits: give it the model's resident
//! cost per layer and the probed [`NodeCaps`], get an assignment back.

use super::caps::NodeCaps;
use super::config::{NodeConfig, Objective, Optimize, PlacementPolicy, UnknownKv};
use super::cost::ModelCost;
use anyhow::{Result, bail};
use std::ops::Range;

/// One node's stage.
#[derive(Debug, Clone)]
pub struct Assignment {
    pub addr: String,
    pub ssh: Option<String>,
    pub layers: Range<usize>,
    pub first: bool,
    pub last: bool,
    /// Estimated resident bytes for this stage (weights + embed/head),
    /// **excluding** routed experts when those page from disk.
    pub est_bytes: u64,
    /// The node's stage RAM budget the planner used.
    pub budget_bytes: u64,
    /// Routed-expert bytes this stage must keep on disk (0 for a dense model,
    /// or when the experts fit in RAM alongside the resident weights).
    pub est_disk_bytes: u64,
    /// True when this stage's experts page from disk rather than sitting in RAM.
    /// Paged stages are bounded by [`NodeCaps::io_mbps`], not by GFLOP/s.
    pub experts_paged: bool,
    /// Predicted seconds per token for this stage: compute time plus, when
    /// paged, the time to stream in the experts each token touches. The planner
    /// balances the pipeline by minimizing the largest of these.
    pub est_stage_secs: f64,
    /// KV-cache bytes THIS stage will hold at the objective's context — exact
    /// for its actual layer range, not the stack average.
    pub est_kv_bytes: u64,
    /// Primary device label for the monitor.
    pub device: String,
    /// Active / standby / draining, from [`NodeConfig::role`].
    pub role: super::config::NodeRole,
    /// For a standby: the `addr` of the active node whose layers it mirrors.
    pub backs: Option<String>,
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

/// Fraction of free disk a stage may fill with expert banks. Leaves room for
/// the OS, logs and the checkpoint itself if it is not already on that disk.
const DISK_SAFETY: f64 = 0.90;

/// Whether this node can keep a stage's experts in RAM rather than paging them.
///
/// This is the branch that decides whether a stage is compute-bound or
/// IO-bound, so it is worth being explicit: experts stay resident only if the
/// RAM budget covers the dense weights AND the full expert banks for the same
/// layers. Otherwise they live on disk and stream in per token.
fn experts_fit_in_ram(budget: u64, model: &ModelCost, layers: u64, overhead: u64) -> bool {
    let need = layers * (model.per_layer_bytes + model.per_layer_expert_bytes) + overhead;
    need <= budget
}

/// Max whole layers bounded by free disk for the routed expert banks.
/// `usize::MAX` for a dense model (no expert bytes to store).
fn disk_cap(caps: &NodeCaps, model: &ModelCost) -> usize {
    if model.per_layer_expert_bytes == 0 {
        return usize::MAX;
    }
    let usable = (caps.disk_free as f64 * DISK_SAFETY) as u64;
    (usable / model.per_layer_expert_bytes) as usize
}

/// Settle on a KV profile, or explain why we cannot.
///
/// Order: an explicit `kv_bytes_per_layer_token` wins, then whatever the
/// checkpoint yielded, then [`UnknownKv`].
pub fn resolve_kv(model: &ModelCost, objective: &Objective) -> Result<super::cost::KvProfile> {
    use super::cost::KvProfile;
    if let Some(b) = objective.kv_bytes_per_layer_token {
        return Ok(KvProfile::declared(b));
    }
    if model.kv.is_known() {
        return Ok(model.kv.clone());
    }
    match objective.on_unknown_kv {
        // A declared zero: known to be nothing, as opposed to unknown.
        UnknownKv::Ignore => Ok(KvProfile::declared(0)),
        UnknownKv::Assume => {
            // MHA worst case at model width: every layer caching full keys and
            // values, `2 × hidden × elem` per token.
            //
            // This needs the model's REAL width. Estimating it from byte counts
            // — `sqrt(per_layer_bytes / 12)`, on the premise that a transformer
            // layer is ~12·hidden² parameters at >=1 byte each — is not a
            // conservative fallback: the premise inverts under quantization. At
            // 2 bits per weight a layer is ~3·hidden² bytes, so the estimate
            // returns hidden/2 and the "pessimistic" bound comes out at half the
            // real cache. A bound that can under-reserve is worse than no bound,
            // so without a width this fails.
            let hidden = model.hidden_size as u64;
            if hidden == 0 {
                bail!(
                    "`on_unknown_kv = \"assume\"` needs the model width to bound the \
                     cache, and the checkpoint did not yield one (no rank-1 norm \
                     inside a block). Estimating it from byte counts under-reserves \
                     for quantized weights, so set \
                     `[placement.objective] kv_bytes_per_layer_token = <bytes>` \
                     instead."
                );
            }
            let elem = super::cost::kv_elem_bytes(objective.kv_dtype.as_deref().unwrap_or("f32"));
            Ok(KvProfile::declared(2 * hidden * elem))
        }
        UnknownKv::Fail => bail!(
            "KV cache size is unknown, so the planner cannot reserve for it at \
             context {}. A plan that ignores the cache fits on paper and dies \
             when it fills. Fix by one of:\n  \
             • set `[placement.objective] kv_bytes_per_layer_token = <bytes>`\n  \
             • set `ModelCost::kv` from the model crate (it knows its head dims)\n  \
             • `on_unknown_kv = \"assume\"` for a pessimistic MHA-width bound\n  \
             • `on_unknown_kv = \"ignore\"` if there is genuinely no cache",
            objective.context
        ),
    }
}

/// A node's layer capacity, and whether reaching it means paging experts.
///
/// A node has two ways to carry a layer, and the better one wins:
///
/// * **experts resident** — RAM holds dense weights *and* expert banks. No disk,
///   no paging, bounded purely by RAM/VRAM.
/// * **experts paged** — RAM holds only the dense weights; the banks live on
///   disk. Bounded by RAM for the dense part *and* by free disk for the banks.
///
/// Taking the max matters in both directions: a node with 1 TB of RAM and a
/// small disk should not be limited by the disk it will never use, and a node
/// with modest RAM and a big NVMe should not be limited to what its RAM can
/// hold outright. For a dense model the two coincide.
fn node_capacity(caps: &NodeCaps, model: &ModelCost, budget: u64, overhead: u64) -> (usize, bool) {
    let resident = layer_cap(
        budget,
        model.per_layer_bytes + model.per_layer_expert_bytes,
        overhead,
    );
    if model.per_layer_expert_bytes == 0 {
        return (resident, false);
    }
    let paged = layer_cap(budget, model.per_layer_bytes, overhead).min(disk_cap(caps, model));
    if paged > resident {
        (paged, true)
    } else {
        (resident, false)
    }
}

/// Predicted seconds per token for `layers` on this node.
///
/// `gflops` is the throughput of the device the stage will actually run on —
/// see [`NodeCaps::gflops_for`], and note that it is not always the node's
/// fastest device. Compute time comes from that measured throughput; when the experts
/// page, the stage additionally waits on disk for the bytes each token touches.
/// A node with a fast NVMe therefore carries more MoE layers than one with the
/// same RAM and a spinning disk — which is the whole point of costing IO.
fn stage_secs(caps: &NodeCaps, gflops: f64, model: &ModelCost, layers: u64, paged: bool) -> f64 {
    let gflops = gflops.max(1.0);
    // Both terms are seconds: `per_layer_flops` is a real FLOP count (see
    // `decode_flops_per_layer`) and `gflops` a measured rate, so their quotient
    // is a time that can be added to the disk wait below.
    let compute = layers as f64 * model.per_layer_flops / (gflops * 1e9);
    if !paged || model.per_layer_expert_active_bytes == 0 {
        return compute;
    }
    let bps = (caps.io_mbps.max(1.0)) * 1e6;
    compute + (layers as f64 * model.per_layer_expert_active_bytes as f64) / bps
}

/// Move layers between stages until the pipeline's critical path stops
/// improving.
///
/// A pipeline runs at the speed of its slowest stage, so the number that matters
/// is `max(est_stage_secs)` — not how close each node is to its proportional
/// share. The proportional split cannot target that directly, because a node's
/// per-layer cost depends on how many layers it ends up with: cross the point
/// where the expert banks no longer fit in RAM and the stage switches from
/// resident to paged, and its per-layer time jumps by the disk term. The weights
/// are computed before that is known, so they can be badly wrong in exactly the
/// interesting case.
///
/// This is a steepest-descent pass over the real objective: repeatedly take one
/// layer from the critical node and give it to whichever node leaves the best
/// sorted-descending time vector. Each accepted move strictly lowers that vector
/// lexicographically, and the state space is finite, so it terminates; the
/// iteration cap is a belt-and-braces bound, not the usual exit.
///
/// Costs come from [`Costing::secs_for`], the same function that will report the
/// plan, so what is optimised is what is shown.
fn balance_critical_path(
    cost: &Costing<'_>,
    nodes: &[(NodeCaps, NodeConfig)],
    caps_layers: &[usize],
    counts: &mut [usize],
    optimize: Optimize,
) {
    let k = counts.len();
    if k < 2 {
        return;
    }
    let secs = |i: usize, n: usize| {
        if n == 0 {
            return 0.0;
        }
        let (caps, cfg) = &nodes[i];
        cost.secs_for(caps, cfg, n, i == 0, i == k - 1)
    };
    // Lexicographic on times sorted worst-first: strictly better than comparing
    // the max alone, which stalls as soon as two nodes tie for slowest.
    let key = |c: &[usize]| {
        let mut v: Vec<f64> = (0..k).map(|i| secs(i, c[i])).collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        v
    };
    let better = |a: &[f64], b: &[f64]| -> bool {
        // `a` improves on `b` by more than rounding noise.
        for (x, y) in a.iter().zip(b) {
            if (x - y).abs() > 1e-9 * y.abs().max(1e-6) {
                return x < y;
            }
        }
        false
    };

    let mut cur = key(counts);
    for _ in 0..(counts.iter().sum::<usize>() + 1) * k {
        // Donor: the stage on the critical path. Anyone else is not the problem.
        let Some(from) = (0..k).filter(|&i| counts[i] > 0).max_by(|&a, &b| {
            secs(a, counts[a])
                .partial_cmp(&secs(b, counts[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return;
        };
        let mut best: Option<(usize, Vec<f64>)> = None;
        for to in 0..k {
            if to == from || counts[to] >= caps_layers[to] {
                continue;
            }
            // Optimising for latency means the fewest hops; waking an unused
            // node to shave a stage would trade the thing being optimised for.
            if optimize == Optimize::Latency && counts[to] == 0 {
                continue;
            }
            counts[from] -= 1;
            counts[to] += 1;
            let cand = key(counts);
            counts[from] += 1;
            counts[to] -= 1;
            if better(&cand, &cur) && best.as_ref().is_none_or(|(_, b)| better(&cand, b)) {
                best = Some((to, cand));
            }
        }
        match best {
            Some((to, k2)) => {
                counts[from] -= 1;
                counts[to] += 1;
                cur = k2;
            }
            None => return,
        }
    }
}

/// Plan contiguous stages. `nodes` is in pipeline order (first node runs the
/// embedding, last runs the head). Returns one [`Assignment`] per node.
///
/// Equivalent to [`plan_placement_with`] under the default [`Objective`].
pub fn plan_placement(
    model: &ModelCost,
    nodes: &[(NodeCaps, NodeConfig)],
    policy: PlacementPolicy,
    reserve_bytes: u64,
) -> Result<Vec<Assignment>> {
    plan_placement_with(model, nodes, policy, reserve_bytes, &Objective::default())
}

/// Plan contiguous stages under an explicit [`Objective`].
///
/// The objective shapes two things the policy alone cannot express:
///
/// * **KV is reserved before weights.** `objective.context` decides how much of
///   each node's RAM the cache will eventually want, and that comes off the
///   budget first. Fitting the weights and discovering the cache does not fit is
///   a failure mode worth designing out.
/// * **[`Optimize::Latency`] uses as few nodes as it can.** Each stage boundary
///   is a network hop on the critical path, so latency packs nodes full and
///   leaves the rest idle, while [`Optimize::Throughput`] spreads to even out
///   stage times and [`Optimize::Memory`] spreads to maximise free RAM.
pub fn plan_placement_with(
    model: &ModelCost,
    nodes: &[(NodeCaps, NodeConfig)],
    policy: PlacementPolicy,
    reserve_bytes: u64,
    objective: &Objective,
) -> Result<Vec<Assignment>> {
    let k = nodes.len();
    if k == 0 {
        bail!("no nodes");
    }
    // Manual: honour each node's explicit range.
    if policy == PlacementPolicy::Manual {
        let cost = Costing {
            model,
            kv: &resolve_kv(model, objective)?,
            context: objective.context,
            reserve: reserve_bytes,
        };
        let mut out = Vec::new();
        for (i, (caps, cfg)) in nodes.iter().enumerate() {
            let r = cfg.manual_range().ok_or_else(|| {
                anyhow::anyhow!("node {} has no `layers` for manual policy", cfg.addr)
            })?;
            out.push(cost.assign(caps, cfg, r, i == 0, i == k - 1));
        }
        return Ok(out);
    }

    // KV first: reserve what the cache will grow into at the target context, so
    // the weight budget is what is genuinely left over. Charged per node at its
    // fair share of layers, since the true split is not known yet.
    let kv = resolve_kv(model, objective)?;
    // Budget for the WORST contiguous window of a node's likely size, not the
    // average one. On a hybrid stack the layers that cache are not evenly
    // spread, so the average fits on paper and OOMs on whichever node draws the
    // attention-heavy slice.
    let fair = (model.n_layers).div_ceil(k).max(1);
    let kv_reserve = kv.max_window_bytes(fair, objective.context);
    let budgets: Vec<u64> = nodes
        .iter()
        .map(|(c, cfg)| budget_bytes(c, cfg, reserve_bytes).saturating_sub(kv_reserve))
        .collect();
    // Per-node capacity in layers (first pays embed, last pays head).
    let modes: Vec<(usize, bool)> = (0..k)
        .map(|i| {
            let overhead = if i == 0 { model.embed_bytes } else { 0 }
                + if i == k - 1 { model.head_bytes } else { 0 };
            node_capacity(&nodes[i].0, model, budgets[i], overhead)
        })
        .collect();
    let caps_layers: Vec<usize> = modes.iter().map(|&(c, _)| c).collect();
    let total_cap: usize = caps_layers
        .iter()
        .copied()
        .map(|c| c.min(model.n_layers))
        .sum();
    if total_cap < model.n_layers {
        // A node in paged mode has capacity == min(ram_for_dense, disk_cap), so
        // "disk was the binding constraint" is `disk_cap == capacity`, not
        // `disk_cap < capacity` — the latter can never fire.
        let disk_bound = model.per_layer_expert_bytes > 0
            && (0..k).any(|i| modes[i].1 && disk_cap(&nodes[i].0, model) <= caps_layers[i]);
        bail!(
            "model ({} layers, {:.1} GB total / {:.1} GB resident) does not fit the \
             cluster (capacity {} layers).{} Lower precision, add a node, raise \
             max_ram, or reduce reserve.",
            model.n_layers,
            model.total_bytes() as f64 / 1e9,
            model.total_resident_bytes() as f64 / 1e9,
            total_cap,
            if disk_bound {
                " At least one node is bounded by FREE DISK for the routed expert \
                 banks, not by RAM — free space or move the checkpoint."
            } else {
                ""
            }
        );
    }

    // Weight each node by budget (ram_balanced), raw throughput (throughput), or
    // predicted per-layer time (auto).
    let weight: Vec<f64> = match policy {
        PlacementPolicy::Throughput => nodes
            .iter()
            .map(|(c, cfg)| {
                c.gflops_for(cfg.primary_device()).max(1.0) * model.per_layer_flops.max(0.001)
            })
            .collect(),
        // Auto: share ∝ 1 / (time for one layer here), so the node that
        // processes a layer fastest gets the most layers — and "fastest"
        // accounts for expert paging, so a fast-NVMe node outranks an equally
        // fast one on a slow disk when the experts stream.
        PlacementPolicy::Auto => (0..k)
            .map(|i| {
                let (caps, cfg) = &nodes[i];
                let t = stage_secs(
                    caps,
                    caps.gflops_for(cfg.primary_device()),
                    model,
                    1,
                    modes[i].1,
                );
                if t > 0.0 { 1.0 / t } else { 1.0 }
            })
            .collect(),
        _ => budgets.iter().map(|&b| b as f64).collect(),
    };
    // Latency: use the fewest nodes that hold the model. Walk the nodes in
    // pipeline order, filling each to capacity, and drop the rest from the
    // weighting entirely — an unused stage is a hop not taken.
    let weight: Vec<f64> = if objective.optimize == Optimize::Latency {
        let mut left = model.n_layers;
        (0..k)
            .map(|i| {
                if left == 0 {
                    return 0.0;
                }
                let take = caps_layers[i].min(left);
                left -= take;
                take as f64
            })
            .collect()
    } else {
        weight
    };
    let wsum: f64 = weight.iter().sum::<f64>().max(f64::MIN_POSITIVE);

    let cost = Costing {
        model,
        kv: &kv,
        context: objective.context,
        reserve: reserve_bytes,
    };

    // Proportional target, clamped to per-node layer cap, remainder water-filled.
    let mut counts: Vec<usize> = (0..k)
        .map(|i| ((model.n_layers as f64 * weight[i] / wsum).round() as usize).min(caps_layers[i]))
        .collect();
    let mut assigned: usize = counts.iter().sum();
    // Add under-target: give leftover layers to nodes with remaining capacity,
    // most spare first. Remove over-target similarly.
    while assigned < model.n_layers {
        let i = (0..k)
            .filter(|&i| counts[i] < caps_layers[i])
            .max_by_key(|&i| caps_layers[i] - counts[i]);
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

    // The proportional split is only as good as the weights, and the weights are
    // per-node guesses made before the split exists: `weight[i]` costs node i as
    // if it ran in its capacity-maximising mode, but the split it produces may
    // put few enough layers there for the experts to sit in RAM instead. That
    // node is then costed as slow, given few layers, and runs fast — while the
    // pipeline waits on whoever got the rest. Finish on the real objective.
    //
    // Only under `Auto`. The other policies are literal on purpose: `Throughput`
    // ranks by raw GFLOP/s, `RamBalanced` by budget, and a user who picks one is
    // asking for that rule rather than for the planner's judgement.
    if policy == PlacementPolicy::Auto {
        balance_critical_path(&cost, nodes, &caps_layers, &mut counts, objective.optimize);
    }

    let mut out = Vec::with_capacity(k);
    let mut start = 0usize;
    for (i, ((caps, cfg), &cnt)) in nodes.iter().zip(&counts).enumerate() {
        let r = start..start + cnt;
        start += cnt;
        out.push(cost.assign(caps, cfg, r, i == 0, i == k - 1));
    }
    Ok(out)
}

/// Everything a stage is costed against: the model, the resolved cache, and the
/// two cluster-wide knobs.
///
/// These four always travel together and none of them varies per node, so they
/// belong to the planner rather than to each call. Threading them individually
/// had grown the stage-costing function to nine parameters behind an
/// `#[allow(too_many_arguments)]` — the lint pointing at a missing struct.
struct Costing<'a> {
    model: &'a ModelCost,
    kv: &'a super::cost::KvProfile,
    context: usize,
    reserve: u64,
}

impl Costing<'_> {
    /// Whether `n` layers here have to page their expert banks off disk.
    ///
    /// Single source of truth for the residency mode: the planner asks it while
    /// choosing a split and [`Self::assign`] asks it when reporting one, so the
    /// two cannot disagree about whether a stage is paged.
    fn paged_for(
        &self,
        caps: &NodeCaps,
        cfg: &NodeConfig,
        n: u64,
        first: bool,
        last: bool,
    ) -> bool {
        let overhead = if first { self.model.embed_bytes } else { 0 }
            + if last { self.model.head_bytes } else { 0 };
        let budget = budget_bytes(caps, cfg, self.reserve);
        self.model.per_layer_expert_bytes > 0
            && !experts_fit_in_ram(budget, self.model, n, overhead)
    }

    /// Predicted seconds per token for `n` layers here, in the mode that count
    /// actually implies.
    fn secs_for(
        &self,
        caps: &NodeCaps,
        cfg: &NodeConfig,
        n: usize,
        first: bool,
        last: bool,
    ) -> f64 {
        let paged = self.paged_for(caps, cfg, n as u64, first, last);
        stage_secs(
            caps,
            caps.gflops_for(cfg.primary_device()),
            self.model,
            n as u64,
            paged,
        )
    }

    /// Cost one node's stage.
    fn assign(
        &self,
        caps: &NodeCaps,
        cfg: &NodeConfig,
        layers: Range<usize>,
        first: bool,
        last: bool,
    ) -> Assignment {
        let (model, reserve) = (self.model, self.reserve);

        let layers_for_kv = layers.clone();
        let n = (layers.end - layers.start) as u64;
        let overhead =
            if first { model.embed_bytes } else { 0 } + if last { model.head_bytes } else { 0 };
        let budget = budget_bytes(caps, cfg, reserve);
        let paged = self.paged_for(caps, cfg, n, first, last);
        let resident_experts = !paged && model.per_layer_expert_bytes > 0;
        let est = n * model.per_layer_bytes
            + overhead
            + if resident_experts {
                n * model.per_layer_expert_bytes
            } else {
                0
            };
        Assignment {
            addr: cfg.addr.clone(),
            ssh: cfg.ssh.clone(),
            layers,
            first,
            last,
            est_bytes: est,
            budget_bytes: budget,
            est_disk_bytes: if paged {
                n * model.per_layer_expert_bytes
            } else {
                0
            },
            experts_paged: paged,
            est_stage_secs: stage_secs(
                caps,
                caps.gflops_for(cfg.primary_device()),
                model,
                n,
                paged,
            ),
            est_kv_bytes: self.kv.bytes_for(layers_for_kv, self.context),
            device: rlx_runtime::device_label(cfg.primary_device()).to_string(),
            role: cfg.role,
            backs: cfg.standby_for.clone(),
        }
    }
}

/// Mirror every standby onto the active stage it backs.
///
/// A standby is planned with the **same layer range** as its principal, so it
/// holds the same weights and promoting it is a pointer swap rather than a cold
/// load — for a 90 GB stage, seconds instead of minutes.
///
/// `standby_for` names the principal explicitly. Left unset, a standby backs the
/// largest active stage that is not already covered, then the next largest: with
/// one spare you protect the most expensive stage to rebuild, and with several
/// you spread rather than piling them all on one node.
pub fn assign_standbys(
    model: &ModelCost,
    nodes: &[(NodeCaps, NodeConfig)],
    active: &[Assignment],
    reserve_bytes: u64,
    kv: &super::cost::KvProfile,
    context: usize,
) -> Vec<Assignment> {
    use super::config::NodeRole;
    let cost = Costing {
        model,
        kv,
        context,
        reserve: reserve_bytes,
    };
    let mut covered: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for (caps, cfg) in nodes {
        if cfg.role != NodeRole::Standby {
            continue;
        }
        let principal = match &cfg.standby_for {
            Some(addr) => active.iter().find(|a| &a.addr == addr),
            None => active
                .iter()
                .filter(|a| !covered.contains(&a.addr))
                .max_by_key(|a| a.layers.len())
                .or_else(|| active.iter().max_by_key(|a| a.layers.len())),
        };
        let Some(p) = principal else { continue };
        covered.push(p.addr.clone());
        let mut a = cost.assign(caps, cfg, p.layers.clone(), p.first, p.last);
        a.backs = Some(p.addr.clone());
        out.push(a);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::cost::KvProfile;
    use super::*;

    fn caps(addr: &str, ram_gb: f64, gflops: f64) -> NodeCaps {
        caps_full(addr, ram_gb, gflops, 500.0, 0.0)
    }
    fn caps_full(addr: &str, ram_gb: f64, gflops: f64, disk_gb: f64, io_mbps: f64) -> NodeCaps {
        NodeCaps {
            addr: addr.into(),
            os: "linux".into(),
            cores: 8,
            ram_total: (ram_gb * 1e9) as u64,
            ram_avail: (ram_gb * 1e9) as u64,
            disk_free: (disk_gb * 1e9) as u64,
            devices: vec![],
            gflops,
            io_mbps,
        }
    }
    /// An MoE whose expert banks dwarf the dense weights — the case `Auto` is for.
    fn moe_model(n_layers: usize) -> ModelCost {
        ModelCost {
            n_layers,
            per_layer_bytes: 200_000_000,
            per_layer_expert_bytes: 2_000_000_000,
            per_layer_expert_active_bytes: 56_000_000,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        }
    }
    fn node(addr: &str) -> NodeConfig {
        NodeConfig {
            addr: addr.into(),
            ssh: None,
            ckpt_dir: "/x".into(),
            device: "cpu".parse().unwrap(),
            precision: "bf16".parse().unwrap(),
            kv_cache: Default::default(),
            rng_seed: None,
            max_ram_gb: None,
            layers: None,
            role: Default::default(),
            standby_for: None,
        }
    }

    #[test]
    fn ram_balanced_fits_and_covers_all_layers() {
        // 43 layers @ ~3GB each ≈ 129GB across 3 uneven nodes.
        let model = ModelCost {
            n_layers: 43,
            per_layer_bytes: 3_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 200_000_000,
            head_bytes: 200_000_000,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        };
        let nodes = vec![
            (caps("a", 60.0, 100.0), node("a")),
            (caps("b", 55.0, 90.0), node("b")),
            (caps("c", 44.0, 60.0), node("c")),
        ];
        let plan =
            plan_placement(&model, &nodes, PlacementPolicy::RamBalanced, 5_000_000_000).unwrap();
        // Contiguous + complete cover of 0..43.
        assert_eq!(plan[0].layers.start, 0);
        assert_eq!(plan.last().unwrap().layers.end, 43);
        for w in plan.windows(2) {
            assert_eq!(w[0].layers.end, w[1].layers.start);
        }
        // Every stage within budget.
        for a in &plan {
            assert!(
                a.est_bytes <= a.budget_bytes,
                "{} over budget: {} > {}",
                a.addr,
                a.est_bytes,
                a.budget_bytes
            );
        }
    }

    /// An MoE node's capacity comes from disk, not RAM: 20 layers × 2 GB of
    /// banks needs 40 GB of disk but only 4 GB of resident weights. Costing the
    /// banks as resident would say this cluster cannot run the model.
    #[test]
    fn experts_are_budgeted_to_disk_not_ram() {
        let model = moe_model(20);
        // 16 GB RAM: nowhere near 20×2.2 GB resident, but plenty for 20×0.2 GB.
        let nodes = vec![(caps_full("a", 16.0, 100.0, 500.0, 2000.0), node("a"))];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        assert_eq!(plan[0].layers, 0..20);
        assert!(plan[0].experts_paged, "experts should page from disk");
        assert_eq!(plan[0].est_disk_bytes, 20 * 2_000_000_000);
        // Resident estimate excludes the paged banks.
        assert_eq!(plan[0].est_bytes, 20 * 200_000_000);
    }

    /// With RAM to spare the same model keeps its experts resident — no disk, and
    /// the stage is compute-bound rather than IO-bound.
    #[test]
    fn experts_stay_resident_when_ram_allows() {
        let model = moe_model(4);
        let nodes = vec![(caps_full("a", 64.0, 100.0, 500.0, 2000.0), node("a"))];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        assert!(!plan[0].experts_paged);
        assert_eq!(plan[0].est_disk_bytes, 0);
        assert_eq!(plan[0].est_bytes, 4 * 2_200_000_000);
    }

    /// The point of costing IO: two nodes identical but for disk speed should
    /// NOT get an equal share when the experts stream.
    #[test]
    fn auto_gives_more_layers_to_the_faster_disk() {
        let model = moe_model(24);
        let nodes = vec![
            (caps_full("nvme", 16.0, 100.0, 500.0, 6000.0), node("nvme")),
            (caps_full("sata", 16.0, 100.0, 500.0, 500.0), node("sata")),
        ];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        let (fast, slow) = (plan[0].layers.len(), plan[1].layers.len());
        assert!(
            fast > slow,
            "fast-NVMe node got {fast} layers, slow-disk node {slow}"
        );
        assert_eq!(fast + slow, 24);
        // RamBalanced cannot see the difference — it splits them evenly.
        let rb =
            plan_placement(&model, &nodes, PlacementPolicy::RamBalanced, 2_000_000_000).unwrap();
        assert_eq!(rb[0].layers.len(), rb[1].layers.len());
    }

    /// `Auto` must return a plan no single-layer move can improve.
    ///
    /// The proportional split cannot target the critical path directly: a node's
    /// per-layer cost depends on how many layers it gets, because crossing the
    /// point where the expert banks stop fitting in RAM flips the WHOLE stage
    /// from resident to paged and adds the disk term to every layer at once. The
    /// weights are computed before the split exists, so they are guesses; this
    /// pins that the planner finishes on the real objective.
    ///
    /// Note what this does NOT claim: that stage times come out even. A node on
    /// a 3x slower disk genuinely earns a small resident stage and nothing more,
    /// and an "even" split there would be worse for everyone.
    #[test]
    fn auto_returns_a_plan_no_single_move_improves() {
        let model = moe_model(46);
        for nodes in [
            vec![
                (caps_full("a", 34.0, 400.0, 500.0, 700.0), node("a")),
                (caps_full("b", 34.0, 400.0, 500.0, 2200.0), node("b")),
            ],
            vec![
                (caps_full("a", 24.0, 400.0, 500.0, 1500.0), node("a")),
                (caps_full("b", 48.0, 200.0, 500.0, 1500.0), node("b")),
                (caps_full("c", 34.0, 900.0, 500.0, 900.0), node("c")),
            ],
        ] {
            let plan =
                plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
            let counts: Vec<usize> = plan.iter().map(|a| a.layers.len()).collect();
            assert_eq!(counts.iter().sum::<usize>(), 46, "every layer placed");

            let k = nodes.len();
            let secs = |c: &[usize]| -> Vec<f64> {
                let mut v: Vec<f64> = (0..k)
                    .map(|i| {
                        let n = c[i];
                        if n == 0 {
                            return 0.0;
                        }
                        // Same rule `assign` uses: mode follows the count.
                        let budget = budget_bytes(&nodes[i].0, &nodes[i].1, 2_000_000_000);
                        let paged = !experts_fit_in_ram(budget, &model, n as u64, 0);
                        let g = nodes[i].0.gflops_for(nodes[i].1.primary_device());
                        stage_secs(&nodes[i].0, g, &model, n as u64, paged)
                    })
                    .collect();
                v.sort_by(|a, b| b.partial_cmp(a).unwrap());
                v
            };
            let base = secs(&counts);
            for from in 0..k {
                if counts[from] == 0 {
                    continue;
                }
                for to in 0..k {
                    if to == from {
                        continue;
                    }
                    let mut c = counts.clone();
                    c[from] -= 1;
                    c[to] += 1;
                    let cand = secs(&c);
                    assert!(
                        cand >= base,
                        "moving a layer {} -> {} improves the critical path \
                         {base:?} -> {cand:?}; the planner left it on the table \
                         (counts {counts:?})",
                        nodes[from].1.addr,
                        nodes[to].1.addr,
                    );
                }
            }
        }
    }

    fn benched(label: &str, gflops: f64) -> super::super::caps::DeviceInfo {
        super::super::caps::DeviceInfo {
            device: label.into(),
            name: label.into(),
            mem_bytes: 0,
            unified: true,
            gflops,
            available: true,
        }
    }

    /// A stage must be costed by the device it RUNS on, not by the node's
    /// fastest one. On an M4 Pro the CPU's AMX benches ~1100 GFLOP/s against
    /// Metal's ~540; a node pinned to Metal and priced on the headline figure
    /// is credited with twice the throughput it will deliver, and the planner
    /// hands it twice the layers it can carry.
    #[test]
    fn a_node_is_costed_by_the_device_its_stage_uses() {
        let model = moe_model(20);
        let mut fast_cpu_slow_gpu = caps_full("a", 64.0, 1100.0, 500.0, 1500.0);
        fast_cpu_slow_gpu.devices = vec![benched("cpu", 1100.0), benched("metal", 540.0)];
        let plain = caps_full("b", 64.0, 1100.0, 500.0, 1500.0);

        let mut on_metal = node("a");
        on_metal.device = "metal".parse().unwrap();
        let nodes = vec![(fast_cpu_slow_gpu, on_metal), (plain, node("b"))];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        assert!(
            plan[0].layers.len() < plan[1].layers.len(),
            "the Metal node runs at half the CPU node's rate but got {} layers \
             against {} — it was priced on its fastest device, not its stage's",
            plan[0].layers.len(),
            plan[1].layers.len(),
        );
    }

    /// Identical machines get identical stages — no drift from the rebalance.
    #[test]
    fn auto_splits_identical_nodes_evenly() {
        let model = moe_model(46);
        let nodes = vec![
            (caps_full("a", 34.0, 400.0, 500.0, 1500.0), node("a")),
            (caps_full("b", 34.0, 400.0, 500.0, 1500.0), node("b")),
        ];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        assert_eq!(plan[0].layers.len(), 23);
        assert_eq!(plan[1].layers.len(), 23);
    }

    /// The mode a plan is COSTED in has to be the mode it reports. These used to
    /// be computed by two different rules, so a stage could be sized as paged
    /// and then displayed as resident.
    #[test]
    fn reported_mode_matches_the_costed_mode() {
        let model = moe_model(46);
        let nodes = vec![
            (caps_full("a", 34.0, 400.0, 500.0, 700.0), node("a")),
            (caps_full("b", 34.0, 400.0, 500.0, 2200.0), node("b")),
        ];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        for a in &plan {
            // A paged stage pays disk for its experts; a resident one holds them.
            if a.experts_paged {
                assert!(a.est_disk_bytes > 0, "paged stage with no disk cost: {a:?}");
            } else {
                assert_eq!(a.est_disk_bytes, 0, "resident stage charged disk: {a:?}");
                assert!(
                    a.est_bytes >= a.layers.len() as u64 * model.per_layer_expert_bytes,
                    "resident stage does not account for its experts: {a:?}"
                );
            }
        }
    }

    /// A node with fast disk but not enough of it is capped by free space, and
    /// the error says so rather than blaming RAM.
    #[test]
    fn disk_bound_cluster_reports_disk() {
        let model = moe_model(40);
        // RAM (16 GB) covers the dense weights for all 40 layers but nowhere near
        // the banks, so the node pages — and 20 GB of free disk holds only 9
        // layers' worth. Disk, not RAM, is what stops this cluster.
        let nodes = vec![(caps_full("a", 16.0, 100.0, 20.0, 6000.0), node("a"))];
        let err = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("FREE DISK"), "{err}");
    }

    /// A dense model must behave exactly as before — no disk, no paging.
    #[test]
    fn dense_models_are_unaffected_by_the_expert_path() {
        let model = ModelCost {
            n_layers: 8,
            per_layer_bytes: 1_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        };
        let nodes = vec![(caps("a", 32.0, 100.0), node("a"))];
        let plan = plan_placement(&model, &nodes, PlacementPolicy::Auto, 2_000_000_000).unwrap();
        assert_eq!(plan[0].layers, 0..8);
        assert!(!plan[0].experts_paged);
        assert_eq!(plan[0].est_disk_bytes, 0);
    }

    fn obj(o: Optimize) -> Objective {
        Objective {
            optimize: o,
            ..Default::default()
        }
    }

    /// Latency pays a network hop per stage boundary, so it should use the
    /// FEWEST nodes that hold the model — not spread for its own sake.
    #[test]
    fn latency_uses_as_few_nodes_as_will_fit() {
        let model = ModelCost {
            n_layers: 12,
            per_layer_bytes: 1_000_000_000,
            ..moe_model(12)
        };
        let model = ModelCost {
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            ..model
        };
        // Each node holds all 12 layers on its own.
        let nodes = vec![
            (caps("a", 32.0, 100.0), node("a")),
            (caps("b", 32.0, 100.0), node("b")),
            (caps("c", 32.0, 100.0), node("c")),
        ];
        let lat = plan_placement_with(
            &model,
            &nodes,
            PlacementPolicy::Auto,
            2_000_000_000,
            &obj(Optimize::Latency),
        )
        .unwrap();
        assert_eq!(lat[0].layers.len(), 12, "first node should take everything");
        assert!(
            lat[1..].iter().all(|a| a.layers.is_empty()),
            "latency should leave the other nodes idle: {:?}",
            lat.iter().map(|a| a.layers.len()).collect::<Vec<_>>()
        );

        // Throughput spreads instead, so no single stage is the bottleneck.
        let tp = plan_placement_with(
            &model,
            &nodes,
            PlacementPolicy::Auto,
            2_000_000_000,
            &obj(Optimize::Throughput),
        )
        .unwrap();
        assert!(tp.iter().all(|a| !a.layers.is_empty()));
    }

    /// KV must be reserved BEFORE weights: the same cluster that fits at a short
    /// context has to refuse a long one rather than OOM at run time.
    #[test]
    fn kv_cache_is_reserved_before_weights() {
        let model = ModelCost {
            n_layers: 8,
            per_layer_bytes: 1_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            // 1 MB per layer per token: 8 layers × 4 k = 32 GB at 4 k context.
            kv: KvProfile::declared(1_000_000),
            params: 0,
            hidden_size: 0,
        };
        let nodes = vec![(caps("a", 16.0, 100.0), node("a"))];
        let short = Objective {
            context: 512,
            ..Default::default()
        };
        let long = Objective {
            context: 32_768,
            ..Default::default()
        };
        assert!(
            plan_placement_with(&model, &nodes, PlacementPolicy::Auto, 1_000_000_000, &short)
                .is_ok(),
            "8 GB weights + 4 GB KV fits 16 GB"
        );
        assert!(
            plan_placement_with(&model, &nodes, PlacementPolicy::Auto, 1_000_000_000, &long)
                .is_err(),
            "at 32k the KV cache alone exceeds the node"
        );
    }

    /// A standby mirrors its principal's layer range exactly — that is what makes
    /// promotion a pointer swap rather than a reload.
    #[test]
    fn standby_mirrors_the_stage_it_backs() {
        let model = ModelCost {
            n_layers: 8,
            per_layer_bytes: 1_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        };
        let mut spare = node("spare");
        spare.role = super::super::config::NodeRole::Standby;
        spare.standby_for = Some("b".into());
        let active = vec![
            (caps("a", 32.0, 100.0), node("a")),
            (caps("b", 32.0, 100.0), node("b")),
        ];
        let plan = plan_placement(&model, &active, PlacementPolicy::Auto, 1_000_000_000).unwrap();
        let mut all = active.clone();
        all.push((caps("spare", 32.0, 100.0), spare));
        let sb = assign_standbys(
            &model,
            &all,
            &plan,
            1_000_000_000,
            &KvProfile::declared(0),
            4096,
        );
        assert_eq!(sb.len(), 1);
        assert_eq!(sb[0].backs.as_deref(), Some("b"));
        let b = plan.iter().find(|a| a.addr == "b").unwrap();
        assert_eq!(sb[0].layers, b.layers, "standby must hold the same layers");
        assert_eq!(sb[0].est_bytes, b.est_bytes);
    }

    /// With no explicit principal, a spare covers the largest stage — the one
    /// that would cost the most to rebuild.
    #[test]
    fn unassigned_standby_backs_the_largest_stage() {
        let model = ModelCost {
            n_layers: 12,
            per_layer_bytes: 1_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        };
        let active = vec![
            (caps("big", 64.0, 100.0), node("big")),
            (caps("small", 16.0, 100.0), node("small")),
        ];
        let plan = plan_placement(&model, &active, PlacementPolicy::Auto, 1_000_000_000).unwrap();
        let biggest = plan
            .iter()
            .max_by_key(|a| a.layers.len())
            .unwrap()
            .addr
            .clone();
        let mut spare = node("spare");
        spare.role = super::super::config::NodeRole::Standby;
        let mut all = active.clone();
        all.push((caps("spare", 64.0, 100.0), spare));
        let sb = assign_standbys(
            &model,
            &all,
            &plan,
            1_000_000_000,
            &KvProfile::declared(0),
            4096,
        );
        assert_eq!(sb[0].backs.as_deref(), Some(biggest.as_str()));
    }

    fn unknown_kv_model() -> ModelCost {
        ModelCost {
            n_layers: 8,
            per_layer_bytes: 1_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::default(), // Unknown
            params: 0,
            hidden_size: 0,
        }
    }

    /// Unknown cache size must stop the plan, not quietly reserve nothing — and
    /// the message has to say how to move forward.
    #[test]
    fn unknown_kv_fails_with_the_ways_out() {
        let nodes = vec![(caps("a", 32.0, 100.0), node("a"))];
        let err = plan_placement_with(
            &unknown_kv_model(),
            &nodes,
            PlacementPolicy::Auto,
            1_000_000_000,
            &Objective::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("KV cache size is unknown"), "{err}");
        assert!(err.contains("kv_bytes_per_layer_token"), "{err}");
        assert!(err.contains("assume"), "{err}");
        assert!(err.contains("ignore"), "{err}");
    }

    /// The bound must be built on the model's real width, at the configured
    /// cache dtype — not on a byte-count estimate.
    #[test]
    fn assume_uses_the_real_width_and_dtype() {
        let m = ModelCost {
            hidden_size: 4096,
            ..unknown_kv_model()
        };
        let f32_obj = Objective {
            on_unknown_kv: UnknownKv::Assume,
            ..Default::default()
        };
        let kv = resolve_kv(&m, &f32_obj).unwrap();
        assert_eq!(kv.bytes_per_layer_token(), 2 * 4096 * 4);

        let f16_obj = Objective {
            kv_dtype: Some("f16".into()),
            ..f32_obj.clone()
        };
        assert_eq!(
            resolve_kv(&m, &f16_obj).unwrap().bytes_per_layer_token(),
            2 * 4096 * 2
        );
    }

    /// Without a width there is nothing to be conservative about. The old
    /// byte-count fallback returned `hidden/2` for a 2-bit checkpoint — HALF the
    /// real cache — so a bound that can under-reserve is refused outright.
    #[test]
    fn assume_refuses_rather_than_inventing_a_width() {
        // A 2-bit-per-weight layer: ~3·hidden² bytes, where the old estimate
        // `sqrt(bytes/12)` yields hidden/2.
        let hidden = 4096u64;
        let m = ModelCost {
            hidden_size: 0,
            per_layer_bytes: 3 * hidden * hidden,
            ..unknown_kv_model()
        };
        let old_estimate = ((m.per_layer_bytes as f64 / 12.0).sqrt()) as u64;
        assert_eq!(
            old_estimate,
            hidden / 2,
            "the old fallback halved the width"
        );

        let err = resolve_kv(
            &m,
            &Objective {
                on_unknown_kv: UnknownKv::Assume,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("needs the model width"), "{err}");
        assert!(err.contains("kv_bytes_per_layer_token"), "{err}");
    }

    /// `assume` reserves a pessimistic bound, so it must consume budget — a plan
    /// that fits under it fits in practice.
    #[test]
    fn assume_reserves_more_than_ignore() {
        let nodes = vec![(caps("a", 32.0, 100.0), node("a"))];
        let mk = |u| Objective {
            on_unknown_kv: u,
            context: 8192,
            ..Default::default()
        };
        let ignored = plan_placement_with(
            &ModelCost {
                hidden_size: 4096,
                ..unknown_kv_model()
            },
            &nodes,
            PlacementPolicy::Auto,
            1_000_000_000,
            &mk(UnknownKv::Ignore),
        )
        .unwrap();
        assert_eq!(ignored[0].layers, 0..8);
        // The same cluster under the pessimistic bound cannot take all 8.
        let assumed = plan_placement_with(
            &ModelCost {
                hidden_size: 4096,
                ..unknown_kv_model()
            },
            &nodes,
            PlacementPolicy::Auto,
            1_000_000_000,
            &mk(UnknownKv::Assume),
        );
        match assumed {
            Ok(p) => assert!(
                p[0].layers.len() <= 8,
                "assume must not reserve less than ignore"
            ),
            Err(e) => assert!(e.to_string().contains("does not fit"), "{e}"),
        }
    }

    /// An explicit figure overrides both inference and the unknown policy.
    #[test]
    fn declared_kv_overrides_unknown() {
        let nodes = vec![(caps("a", 32.0, 100.0), node("a"))];
        let obj = Objective {
            kv_bytes_per_layer_token: Some(1024),
            context: 1024,
            ..Default::default()
        };
        let p = plan_placement_with(
            &unknown_kv_model(),
            &nodes,
            PlacementPolicy::Auto,
            1_000_000_000,
            &obj,
        )
        .unwrap();
        assert_eq!(p[0].layers, 0..8);
    }

    #[test]
    fn rejects_when_too_big() {
        let model = ModelCost {
            n_layers: 100,
            per_layer_bytes: 5_000_000_000,
            per_layer_expert_bytes: 0,
            per_layer_expert_active_bytes: 0,
            embed_bytes: 0,
            head_bytes: 0,
            per_layer_flops: 1.0,
            kv: KvProfile::declared(0),
            params: 0,
            hidden_size: 0,
        };
        let nodes = vec![(caps("a", 40.0, 100.0), node("a"))];
        assert!(
            plan_placement(&model, &nodes, PlacementPolicy::RamBalanced, 5_000_000_000).is_err()
        );
    }
}
