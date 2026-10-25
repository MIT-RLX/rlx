// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Roofline sizing: how long a step takes on a given machine type and node
//! count, and which resource sets it.
//!
//! A model describes one step's work as a [`StepWork`] — FLOPs, memory traffic,
//! routed-expert bytes (which may come from disk), bytes crossing a pipeline cut
//! or an expert all-to-all — for any number of tokens processed together
//! ([`StepModel`]). [`time_step`] turns that into time per resource at a
//! [`NodeSpec`]'s nominal rates scaled by an efficiency, assuming compute,
//! memory, disk and network overlap within a stage:
//!
//! * **Pipeline** — layers split evenly over `n` nodes; `m` micro-batches in
//!   flight. A stage processes `1/n` of a micro-batch's work; throughput is
//!   `tokens_per_mb · min(m, n) / (n · stage)`, a token's latency
//!   `max(n, m) · stage`. One sequence (`m = 1`) gets no faster with more nodes.
//! * **Expert parallel** — the dense part is visited node after node; the
//!   routed-expert part (compute, reads) spreads over every node, plus the
//!   dispatch / combine all-to-all.
//!
//! A model, not a measurement: nominal hardware figures ([`presets`] are
//! approximate public FP32-vector numbers; override any field) and no runtime
//! overheads.

use serde::Serialize;

/// One machine type. Rates are nominal; [`time_step`]'s efficiency scales them.
#[derive(Debug, Clone, Serialize)]
pub struct NodeSpec {
    pub name: String,
    pub ram_gb: f64,
    pub ram_bw_gbs: f64,
    /// Discrete GPU memory (0 for none / unified memory).
    pub vram_gb: f64,
    pub vram_bw_gbs: f64,
    /// Host↔GPU link, GB/s (weights left in RAM stream over it to a discrete GPU).
    pub pcie_gbs: f64,
    /// Unified memory: the GPU computes straight out of RAM.
    pub unified: bool,
    pub disk_gb: f64,
    pub disk_read_gbs: f64,
    pub net_gbps: f64,
    pub cpu_tflops: f64,
    pub gpu_tflops: f64,
    /// Memory kept free for the OS and runtime.
    pub reserve_gb: f64,
}

impl NodeSpec {
    /// Memory weights and state can occupy.
    pub fn usable_gb(&self) -> f64 {
        (self.ram_gb + self.vram_gb - self.reserve_gb).max(0.0)
    }
}

/// Nominal specs of common machines (approximate public figures; FP32 vector
/// throughput — matrix units are faster).
pub fn presets() -> Vec<NodeSpec> {
    #[allow(clippy::too_many_arguments)]
    let n = |name: &str,
             ram: f64,
             ram_bw: f64,
             vram: f64,
             vram_bw: f64,
             unified: bool,
             disk: f64,
             disk_bw: f64,
             net: f64,
             cpu: f64,
             gpu: f64| NodeSpec {
        name: name.into(),
        ram_gb: ram,
        ram_bw_gbs: ram_bw,
        vram_gb: vram,
        vram_bw_gbs: vram_bw,
        pcie_gbs: if vram > 0.0 { 25.0 } else { 0.0 },
        unified,
        disk_gb: disk,
        disk_read_gbs: disk_bw,
        net_gbps: net,
        cpu_tflops: cpu,
        gpu_tflops: gpu,
        reserve_gb: (ram * 0.1).clamp(4.0, 32.0),
    };
    vec![
        n(
            "mac-mini-m4pro-64",
            64.0,
            273.0,
            0.0,
            0.0,
            true,
            2000.0,
            5.0,
            10.0,
            1.0,
            9.0,
        ),
        n(
            "mac-studio-m3ultra-512",
            512.0,
            819.0,
            0.0,
            0.0,
            true,
            8000.0,
            6.0,
            10.0,
            2.5,
            28.0,
        ),
        n(
            "mac-studio-m4max-128",
            128.0,
            546.0,
            0.0,
            0.0,
            true,
            4000.0,
            6.0,
            10.0,
            1.8,
            17.0,
        ),
        n(
            "pc-rtx3080ti-64",
            64.0,
            50.0,
            12.0,
            912.0,
            false,
            2000.0,
            3.5,
            2.5,
            1.0,
            34.0,
        ),
        n(
            "pc-rtx4090-128",
            128.0,
            80.0,
            24.0,
            1008.0,
            false,
            4000.0,
            7.0,
            10.0,
            1.5,
            82.0,
        ),
        n(
            "server-mi100-256",
            256.0,
            200.0,
            32.0,
            1229.0,
            false,
            4000.0,
            6.0,
            25.0,
            2.0,
            23.0,
        ),
        n(
            "server-8xh100-2tb",
            2048.0,
            600.0,
            640.0,
            26800.0,
            false,
            30000.0,
            25.0,
            400.0,
            6.0,
            536.0,
        ),
        n(
            "cpu-epyc-768",
            768.0,
            460.0,
            0.0,
            0.0,
            false,
            8000.0,
            12.0,
            25.0,
            8.0,
            0.0,
        ),
    ]
}

/// The preset called `name`.
pub fn preset(name: &str) -> anyhow::Result<NodeSpec> {
    match presets().into_iter().find(|p| p.name == name) {
        Some(p) => Ok(p),
        None => anyhow::bail!(
            "unknown node preset {name:?}; known: {}",
            presets()
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// How layers spread over nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Parallel {
    /// Each node owns a contiguous layer range; a token visits every node in turn.
    Pipeline,
    /// Dense layers visited in turn; routed experts sharded by id over every
    /// node, with an all-to-all per MoE layer.
    ExpertParallel,
}

/// One step's work for some number of tokens processed together.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StepWork {
    /// FLOPs every node in the token's path shares (dense layers, attention).
    pub dense_flops: f64,
    /// FLOPs of the routed experts (spread over nodes under expert parallelism).
    pub expert_flops: f64,
    /// Memory traffic not tied to routed experts (dense weights, KV / state).
    pub dense_bytes: f64,
    /// Routed-expert bytes the step reads.
    pub expert_bytes: f64,
    /// Fraction of `expert_bytes` served from memory; the rest comes from disk.
    pub expert_in_memory: f64,
    /// Bytes per token crossing one pipeline cut.
    pub boundary_bytes_per_token: f64,
    /// Bytes per token of the expert all-to-all if every expert were remote
    /// (scaled by the remote share, `(n-1)/n`).
    pub alltoall_bytes_per_token: f64,
    /// MoE layers (each costs one all-to-all round trip under expert parallelism).
    pub moe_layers: usize,
}

/// What a model reports to [`time_step`].
pub trait StepModel {
    fn work(&self, tokens: f64) -> StepWork;
}

impl<F: Fn(f64) -> StepWork> StepModel for F {
    fn work(&self, tokens: f64) -> StepWork {
        self(tokens)
    }
}

/// Time per resource for one step, and its totals.
#[derive(Debug, Clone, Serialize)]
pub struct StepTime {
    pub compute_s: f64,
    pub memory_s: f64,
    pub disk_s: f64,
    pub network_s: f64,
    /// Wall time of the step (all micro-batches).
    pub total_s: f64,
    pub tokens_per_s: f64,
    /// Seconds from a token entering to leaving the pipeline.
    pub latency_s: f64,
    pub bottleneck: &'static str,
    pub flops: f64,
    pub memory_bytes: f64,
    pub disk_bytes: f64,
    pub network_bytes: f64,
}

/// Time one step of `tokens` tokens in `micro_batches` groups on `nodes` nodes
/// of type `node`, `resident_bytes_per_node` of weights / state each (decides
/// whether a discrete GPU computes out of VRAM or streams over its link).
#[allow(clippy::too_many_arguments)]
pub fn time_step(
    model: &dyn StepModel,
    tokens: f64,
    micro_batches: f64,
    resident_bytes_per_node: f64,
    node: &NodeSpec,
    nodes: usize,
    parallel: Parallel,
    efficiency: f64,
) -> StepTime {
    const GB: f64 = 1e9;
    let eff = efficiency.clamp(0.01, 1.0);
    let nf = nodes.max(1) as f64;
    let m = micro_batches.clamp(1.0, tokens.max(1.0));
    let mb = tokens / m;
    let flops_rate = (if node.gpu_tflops > 0.0 {
        node.gpu_tflops
    } else {
        node.cpu_tflops
    }) * 1e12
        * eff;
    let fits_vram = node.vram_gb > 0.0 && resident_bytes_per_node <= node.vram_gb * GB;
    let bw = if node.unified || node.vram_gb == 0.0 {
        node.ram_bw_gbs
    } else if fits_vram {
        node.vram_bw_gbs
    } else {
        let in_vram = (node.vram_gb * GB) / resident_bytes_per_node.max(1.0);
        1.0 / (in_vram / node.vram_bw_gbs + (1.0 - in_vram) / node.pcie_gbs.min(node.ram_bw_gbs))
    } * GB
        * eff;
    let disk_rate = (node.disk_read_gbs * GB * eff).max(1.0);
    let net_rate = (node.net_gbps / 8.0 * GB * eff).max(1.0);
    let hop = if nodes > 1 { 2e-4 } else { 0.0 };
    let w_all = model.work(tokens);
    let w_mb = model.work(mb);
    let mem_of = |w: &StepWork| w.dense_bytes + w.expert_bytes * w.expert_in_memory;
    let disk_of = |w: &StepWork| w.expert_bytes * (1.0 - w.expert_in_memory);
    let (
        compute_s,
        memory_s,
        disk_s,
        network_s,
        total_s,
        tokens_per_s,
        latency_s,
        memory_bytes,
        disk_bytes,
        network_bytes,
    ) = match parallel {
        Parallel::Pipeline => {
            let c = (w_mb.dense_flops + w_mb.expert_flops) / nf / flops_rate;
            let mm = mem_of(&w_mb) / nf / bw;
            let d = disk_of(&w_mb) / nf / disk_rate;
            let nt = mb * w_mb.boundary_bytes_per_token / net_rate + hop;
            let stage = c.max(mm).max(d).max(nt).max(1e-12);
            let tps = mb * m.min(nf) / (nf * stage);
            (
                c * nf,
                mm * nf,
                d * nf,
                nt * nf,
                stage * nf.max(m),
                tps,
                stage * nf.max(m),
                mem_of(&w_mb) * m,
                disk_of(&w_mb) * m,
                (nf - 1.0) * tokens * w_all.boundary_bytes_per_token,
            )
        }
        Parallel::ExpertParallel => {
            let w = &w_all;
            let net = tokens * w.alltoall_bytes_per_token * (nf - 1.0) / nf;
            let c_d = w.dense_flops / flops_rate;
            let c_e = w.expert_flops / nf / flops_rate;
            let m_d = w.dense_bytes / bw;
            let m_e = w.expert_bytes * w.expert_in_memory / nf / bw;
            let d = disk_of(w) / nf / disk_rate;
            let nt = net / net_rate + hop * w.moe_layers as f64;
            let step = (c_d.max(m_d) + c_e.max(m_e).max(d) + nt).max(1e-12);
            (
                c_d + c_e,
                m_d + m_e,
                d,
                nt,
                step,
                tokens / step,
                step,
                mem_of(w),
                disk_of(w),
                net,
            )
        }
    };
    let bottleneck = [
        ("compute", compute_s),
        ("memory bandwidth", memory_s),
        ("disk", disk_s),
        ("network", network_s),
    ]
    .into_iter()
    .fold(("compute", f64::MIN), |a, x| if x.1 > a.1 { x } else { a })
    .0;
    StepTime {
        compute_s,
        memory_s,
        disk_s,
        network_s,
        total_s,
        tokens_per_s,
        latency_s,
        bottleneck,
        flops: w_all.dense_flops + w_all.expert_flops,
        memory_bytes,
        disk_bytes,
        network_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 GFLOP and 1 GB of dense traffic per token, nothing else.
    fn dense(tokens: f64) -> StepWork {
        StepWork {
            dense_flops: 1e9 * tokens,
            dense_bytes: 1e9,
            boundary_bytes_per_token: 1e4,
            ..StepWork::default()
        }
    }

    #[test]
    fn pipeline_concurrency_not_node_count_speeds_one_sequence() {
        let node = preset("mac-mini-m4pro-64").unwrap();
        let one = time_step(&dense, 1.0, 1.0, 1e9, &node, 1, Parallel::Pipeline, 1.0);
        let four = time_step(&dense, 1.0, 1.0, 1e9, &node, 4, Parallel::Pipeline, 1.0);
        // Weights stream once per token either way: memory-bound, same latency.
        assert_eq!(one.bottleneck, "memory bandwidth");
        assert!((four.latency_s / one.latency_s - 1.0).abs() < 0.05);
        // Four sequences in flight on four nodes: ~4× the tokens per second.
        let busy = time_step(&dense, 4.0, 4.0, 1e9, &node, 4, Parallel::Pipeline, 1.0);
        assert!(busy.tokens_per_s > 3.5 * four.tokens_per_s);
    }

    #[test]
    fn expert_parallel_spreads_disk_reads() {
        let paged = |tokens: f64| StepWork {
            dense_bytes: 1e8,
            expert_bytes: 2e10,
            expert_in_memory: 0.0,
            dense_flops: 1e9 * tokens,
            alltoall_bytes_per_token: 1e6,
            moe_layers: 10,
            ..StepWork::default()
        };
        let node = preset("mac-mini-m4pro-64").unwrap();
        let a = time_step(
            &paged,
            1.0,
            1.0,
            1e9,
            &node,
            2,
            Parallel::ExpertParallel,
            0.5,
        );
        let b = time_step(
            &paged,
            1.0,
            1.0,
            1e9,
            &node,
            8,
            Parallel::ExpertParallel,
            0.5,
        );
        assert_eq!(a.bottleneck, "disk");
        assert!(b.disk_s < a.disk_s / 3.0);
        assert!(b.tokens_per_s > a.tokens_per_s);
    }
}
