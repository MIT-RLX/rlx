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
// RLX — LLaDA2 group-limited MoE gate (shared CPU reference for all backends).

/// Group-limited TopK (TIDE `LLaDA2MoeGate.group_limited_topk`).
pub fn group_limited_topk(
    scores: &[f32],
    num_tokens: usize,
    num_experts: usize,
    n_group: usize,
    topk_group: usize,
    top_k: usize,
) -> (Vec<f32>, Vec<u32>) {
    let epg = num_experts / n_group;
    let mut probs = Vec::with_capacity(num_tokens * top_k);
    let mut indices = Vec::with_capacity(num_tokens * top_k);
    for t in 0..num_tokens {
        let row = &scores[t * num_experts..(t + 1) * num_experts];
        let mut group_scores = vec![0f32; n_group];
        for g in 0..n_group {
            let base = g * epg;
            let slice = &row[base..base + epg];
            let mut top2 = [f32::NEG_INFINITY; 2];
            for &v in slice {
                if v > top2[0] {
                    top2[1] = top2[0];
                    top2[0] = v;
                } else if v > top2[1] {
                    top2[1] = v;
                }
            }
            group_scores[g] = top2[0] + top2[1];
        }
        let mut group_order: Vec<usize> = (0..n_group).collect();
        group_order.sort_by(|&a, &b| {
            group_scores[b]
                .partial_cmp(&group_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected: std::collections::HashSet<usize> =
            group_order.into_iter().take(topk_group).collect();
        let mut masked = vec![f32::NEG_INFINITY; num_experts];
        for g in selected {
            let base = g * epg;
            masked[base..base + epg].copy_from_slice(&row[base..base + epg]);
        }
        let mut order: Vec<usize> = (0..num_experts).collect();
        order.sort_by(|&a, &b| {
            masked[b]
                .partial_cmp(&masked[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut picked_scores = Vec::with_capacity(top_k);
        let mut picked_idx = Vec::with_capacity(top_k);
        for &ei in order.iter().take(top_k) {
            picked_scores.push(row[ei]);
            picked_idx.push(ei as u32);
        }
        let sum: f32 = picked_scores.iter().sum::<f32>() + 1e-20;
        let scale = if top_k > 1 { 1.0 / sum } else { 1.0 };
        for (p, &ei) in picked_scores.iter().zip(&picked_idx) {
            probs.push(p * scale);
            indices.push(ei);
        }
    }
    (probs, indices)
}

#[derive(Clone, Copy)]
pub struct GateAttrs {
    pub n_group: u32,
    pub topk_group: u32,
    pub top_k: u32,
    pub routed_scaling: f32,
    pub num_experts: u32,
}

impl GateAttrs {
    pub fn from_bytes(attrs: &[u8]) -> Self {
        if attrs.len() >= 20 {
            let n_group = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
            let topk_group = u32::from_le_bytes(attrs[4..8].try_into().unwrap());
            let top_k = u32::from_le_bytes(attrs[8..12].try_into().unwrap());
            let routed_scaling = f32::from_le_bytes(attrs[12..16].try_into().unwrap());
            let num_experts = u32::from_le_bytes(attrs[16..20].try_into().unwrap());
            GateAttrs {
                n_group,
                topk_group,
                top_k,
                routed_scaling,
                num_experts,
            }
        } else {
            GateAttrs {
                n_group: 8,
                topk_group: 4,
                top_k: 8,
                routed_scaling: 2.5,
                num_experts: 256,
            }
        }
    }
}

/// Run the gate inside a contiguous f32 arena (CUDA/ROCm/WGPU host segments).
pub fn execute_gate_in_f32_arena(
    host: &mut [f32],
    sig_f32_off: usize,
    route_f32_off: usize,
    out_f32_off: usize,
    n_elems: usize,
    attrs: &[u8],
) -> Result<(), String> {
    let a = GateAttrs::from_bytes(attrs);
    let e = a.num_experts as usize;
    let k = a.top_k as usize;
    let rows = n_elems / e.max(1);
    let out_end = out_f32_off + rows * k * 2;
    let sig = host[sig_f32_off..sig_f32_off + n_elems].to_vec();
    let route = host[route_f32_off..route_f32_off + n_elems].to_vec();
    let out = &mut host[out_f32_off..out_end];
    execute_gate_f32(&sig, &route, out, attrs)
}

/// Execute gate: inputs = [sigmoid scores, routing scores]; output = [idx, weights] packed.
pub fn execute_gate_f32(
    scores_sigmoid: &[f32],
    scores_route: &[f32],
    out: &mut [f32],
    attrs: &[u8],
) -> Result<(), String> {
    let a = GateAttrs::from_bytes(attrs);
    let rows = scores_sigmoid.len() / a.num_experts as usize;
    let e = a.num_experts as usize;
    let k = a.top_k as usize;
    if scores_route.len() != scores_sigmoid.len() {
        return Err("gate: sigmoid and routing score lengths differ".into());
    }
    if out.len() != rows * k * 2 {
        return Err(format!("output len {} != rows*k*2", out.len()));
    }
    let (_, idx) = group_limited_topk(
        scores_route,
        rows,
        e,
        a.n_group as usize,
        a.topk_group as usize,
        k,
    );
    for t in 0..rows {
        let row_sig = &scores_sigmoid[t * e..(t + 1) * e];
        let mut picked = Vec::with_capacity(k);
        for ki in 0..k {
            let ei = idx[t * k + ki] as usize;
            picked.push(row_sig[ei]);
        }
        let sum: f32 = picked.iter().sum::<f32>() + 1e-20;
        let norm = if k > 1 { 1.0 / sum } else { 1.0 };
        for ki in 0..k {
            out[t * k * 2 + ki] = idx[t * k + ki] as f32;
            out[t * k * 2 + k + ki] = picked[ki] * norm * a.routed_scaling;
        }
    }
    Ok(())
}
