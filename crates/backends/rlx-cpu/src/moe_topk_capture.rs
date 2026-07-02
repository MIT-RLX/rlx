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

//! Capture MoE router [`Op::TopK`] outputs during CPU forward (TIDE refresh input).

use std::sync::{Arc, Mutex};

/// Shared capture buffer — one entry per MoE router TopK in schedule order.
#[derive(Debug)]
pub struct MoeTopkCapture {
    pub num_experts: usize,
    layers: Mutex<Vec<Vec<u32>>>,
}

impl MoeTopkCapture {
    pub fn new(num_experts: usize) -> Arc<Self> {
        Arc::new(Self {
            num_experts,
            layers: Mutex::new(Vec::new()),
        })
    }

    pub fn clear(&self) {
        self.layers.lock().unwrap().clear();
    }

    /// Record one router TopK output (`outer * k` f32-encoded expert ids).
    pub fn push_topk_f32(&self, data: &[f32], axis_dim: usize) {
        if axis_dim != self.num_experts {
            return;
        }
        let flat: Vec<u32> = data.iter().map(|&v| v as u32).collect();
        self.layers.lock().unwrap().push(flat);
    }

    pub fn take_layers(&self) -> Vec<Vec<u32>> {
        std::mem::take(&mut *self.layers.lock().unwrap())
    }
}
