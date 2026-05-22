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

//! Per-expert F32 weight slabs for MoE offload (TIDE-style migration source).

use crate::ExpertPool;
use std::sync::Arc;

/// One expert projection stack in GroupedMatMul layout `[num_experts, k, n]`.
#[derive(Debug, Clone)]
pub struct ExpertStackF32 {
    pub num_experts: usize,
    pub k: usize,
    pub n: usize,
    pub data: Arc<[f32]>,
}

impl ExpertStackF32 {
    pub fn new(data: Vec<f32>, num_experts: usize, k: usize, n: usize) -> Self {
        assert_eq!(data.len(), num_experts * k * n);
        Self {
            num_experts,
            k,
            n,
            data: Arc::from(data),
        }
    }

    pub fn expert_stride(&self) -> usize {
        self.k * self.n
    }

    pub fn expert_slice(&self, e: usize) -> &[f32] {
        let stride = self.expert_stride();
        let start = e * stride;
        &self.data[start..start + stride]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Gate / up / down expert stacks for one decoder layer.
#[derive(Debug, Clone)]
pub struct LayerMoeWeights {
    pub layer_index: usize,
    pub gate: ExpertStackF32,
    pub up: ExpertStackF32,
    pub down: ExpertStackF32,
}

/// Host-side expert weights for all MoE layers (migration source of truth).
#[derive(Debug, Clone)]
pub struct MoeExpertStore {
    pub layers: Vec<LayerMoeWeights>,
}

impl MoeExpertStore {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Apply captured TopK indices to per-layer pools (TIDE refresh).
    pub fn refresh_pools(
        &self,
        pools: &mut [ExpertPool],
        captured: &[Vec<u32>],
        decode_step: usize,
        is_prefill_block: bool,
    ) -> bool {
        let n = self.layers.len().min(pools.len()).min(captured.len());
        if n == 0 {
            return false;
        }
        let refresh = pools[0].should_refresh(
            crate::MoEExecMode::Reuse,
            decode_step,
            is_prefill_block,
        );
        if !refresh {
            return false;
        }
        for i in 0..n {
            pools[i].refresh_from_indices(&captured[i]);
        }
        true
    }

    /// Push full host stacks into compiled params (lossless; refreshes arena bytes).
    pub fn apply_to_compiled(&self, compiled: &mut crate::CompiledGraph) {
        for layer in &self.layers {
            let il = layer.layer_index;
            compiled.set_param(
                &format!("blk.{il}.ffn_gate_exps.weight"),
                layer.gate.as_slice(),
            );
            compiled.set_param(
                &format!("blk.{il}.ffn_up_exps.weight"),
                layer.up.as_slice(),
            );
            compiled.set_param(
                &format!("blk.{il}.ffn_down_exps.weight"),
                layer.down.as_slice(),
            );
        }
    }
}

