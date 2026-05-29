// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Named parameter tensors for compiled UMAP graphs.

use std::collections::HashMap;

/// Parameter name → row-major f32 weights.
#[derive(Debug, Clone, Default)]
pub struct WeightStore(pub HashMap<String, Vec<f32>>);

impl WeightStore {
    pub fn apply(&self, exec: &mut rlx_runtime::CompiledGraph) {
        for (name, data) in &self.0 {
            exec.set_param(name, data);
        }
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.0.get(name).map(|v| v.as_slice())
    }

    /// Save encoder weights (`.safetensors` or `.gguf` by extension).
    pub fn save(
        &self,
        path: impl AsRef<std::path::Path>,
        spec: &crate::encoder::mlp::ModelSpec,
    ) -> std::io::Result<()> {
        crate::serialize::save_weights(self, spec, path)
    }

    /// Load encoder weights (safetensors, GGUF, or legacy `.ruama`).
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        crate::serialize::load_weights(path)
    }
}

/// Uniform [0, 1] from an LCG state (21-bit mantissa; safe for any `seed`).
fn unit01(seed: u64) -> f32 {
    const M: u64 = (1 << 21) - 1;
    ((seed >> 11) & M) as f32 / M as f32
}

pub fn init_mat(w: &mut WeightStore, name: &str, rows: usize, cols: usize, seed: &mut u64) {
    let scale = (2.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    let mut v = vec![0.0f32; n];
    for x in &mut v {
        *seed = lcg(*seed);
        let u = unit01(*seed);
        *seed = lcg(*seed);
        let n2 = unit01(*seed);
        *x = (u * 2.0 * std::f32::consts::PI * n2).sin() * scale;
    }
    w.0.insert(name.to_string(), v);
}

pub fn init_vec(w: &mut WeightStore, name: &str, n: usize, seed: &mut u64) {
    let mut v = vec![0.0f32; n];
    for x in &mut v {
        *seed = lcg(*seed);
        *x = 0.01 * (unit01(*seed) - 0.5);
    }
    w.0.insert(name.to_string(), v);
}

fn lcg(s: u64) -> u64 {
    s.wrapping_mul(6364136223846793005).wrapping_add(1)
}
