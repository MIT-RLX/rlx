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

#![cfg(feature = "optim")]

use std::collections::HashMap;

use rlx_optim::{AdamW, Optimizer};
use rlx_umap::optim_adapter::step_weight_store;
use rlx_umap::weights::WeightStore;

#[test]
fn adamw_drives_weight_store_to_target() {
    // Simple convex objective f(w) = 0.5·‖w − w*‖²; gradient is `w − w*`.
    let mut weights = WeightStore::default();
    weights.0.insert("fc.weight".into(), vec![0.0f32; 8 * 4]);
    weights.0.insert("fc.bias".into(), vec![0.0f32; 8]);

    let target_fc: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02 - 0.3).collect();
    let target_b: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();

    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    shapes.insert("fc.weight".into(), vec![8, 4]);
    shapes.insert("fc.bias".into(), vec![8]);

    let mut opt = AdamW::new(0.1).with_weight_decay(0.0);
    for _ in 0..400 {
        let mut grads = WeightStore::default();
        let g_fc: Vec<f32> = weights.0["fc.weight"]
            .iter()
            .zip(&target_fc)
            .map(|(w, t)| w - t)
            .collect();
        let g_b: Vec<f32> = weights.0["fc.bias"]
            .iter()
            .zip(&target_b)
            .map(|(w, t)| w - t)
            .collect();
        grads.0.insert("fc.weight".into(), g_fc);
        grads.0.insert("fc.bias".into(), g_b);
        let stepped = step_weight_store(&mut opt, &mut weights, &grads, &shapes);
        assert_eq!(stepped, 2);
        opt.end_iteration();
    }

    // Both tensors should be close to their target.
    let err_fc: f32 = weights.0["fc.weight"]
        .iter()
        .zip(&target_fc)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    let err_b: f32 = weights.0["fc.bias"]
        .iter()
        .zip(&target_b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(err_fc < 1e-3, "fc.weight residual {err_fc}");
    assert!(err_b < 1e-3, "fc.bias residual {err_b}");
}

#[test]
fn missing_grad_is_skipped() {
    let mut weights = WeightStore::default();
    weights.0.insert("trained".into(), vec![0.0f32; 4]);
    weights.0.insert("frozen".into(), vec![5.0f32; 4]);
    let mut grads = WeightStore::default();
    grads.0.insert("trained".into(), vec![1.0f32; 4]);
    let shapes = HashMap::new();

    let mut opt = AdamW::new(0.1).with_weight_decay(0.0);
    let stepped = step_weight_store(&mut opt, &mut weights, &grads, &shapes);
    assert_eq!(stepped, 1);
    // `frozen` was untouched.
    assert!(weights.0["frozen"].iter().all(|&x| (x - 5.0).abs() < 1e-6));
    // `trained` moved.
    assert!(weights.0["trained"].iter().any(|&x| x.abs() > 1e-6));
}
