// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Host-side Adam optimizer (matches fast-umap defaults).

use crate::weights::WeightStore;

#[derive(Debug, Clone)]
pub struct AdamState {
    m: WeightStore,
    v: WeightStore,
    step: u64,
}

impl AdamState {
    pub fn new_like(weights: &WeightStore) -> Self {
        let mut m = WeightStore::default();
        let mut v = WeightStore::default();
        for (name, data) in &weights.0 {
            m.0.insert(name.clone(), vec![0.0; data.len()]);
            v.0.insert(name.clone(), vec![0.0; data.len()]);
        }
        Self { m, v, step: 0 }
    }

    /// Adam step with optional L2 weight decay (`penalty` added to gradient).
    pub fn step(
        &mut self,
        weights: &mut WeightStore,
        grads: &WeightStore,
        lr: f64,
        beta1: f64,
        beta2: f64,
        penalty: f32,
        eps: f64,
    ) {
        self.step += 1;
        let t = self.step as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        let clip_scale = global_grad_clip_scale(grads, 1.0);

        for (name, w) in &mut weights.0 {
            let g = grads.0.get(name).expect("grad for param");
            let m = self.m.0.get_mut(name).unwrap();
            let v = self.v.0.get_mut(name).unwrap();
            for i in 0..w.len() {
                let gi = (g[i] + penalty * w[i]) * clip_scale;
                m[i] = (beta1 * m[i] as f64 + (1.0 - beta1) * gi as f64) as f32;
                v[i] = (beta2 * v[i] as f64 + (1.0 - beta2) * (gi as f64 * gi as f64)) as f32;
                let m_hat = m[i] as f64 / bc1;
                let v_hat = v[i] as f64 / bc2;
                w[i] -= (lr * m_hat / (v_hat.sqrt() + eps)) as f32;
            }
        }
    }
}

/// Global L2 norm clip (scale gradients if norm > `max_norm`).
pub fn global_grad_clip_scale(grads: &WeightStore, max_norm: f32) -> f32 {
    let mut norm_sq = 0.0f32;
    for g in grads.0.values() {
        for gi in g {
            if gi.is_finite() {
                norm_sq += gi * gi;
            }
        }
    }
    let max_sq = max_norm * max_norm;
    if norm_sq > max_sq && norm_sq > 0.0 {
        max_norm / norm_sq.sqrt()
    } else {
        1.0
    }
}
