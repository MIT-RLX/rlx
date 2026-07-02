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

//! Deterministic fit with fixed data (regression guard).

use rlx_driver::Device;
use rlx_runtime::Session;
use rlx_umap::config::{GraphParams, OptimizationParams, TrainingConfig, UmapConfig};
use rlx_umap::encoder::init_model_weights;
use rlx_umap::encoder::knn::build_knn_edges;
use rlx_umap::encoder::mlp::ModelSpec;
use rlx_umap::model::CompiledUmap;
use rlx_umap::prelude::*;
use rlx_umap::training::train_only;
use rlx_umap::utils::{f64_to_f32, flatten_f64, normalize_data_f64};
use rlx_umap::weights::WeightStore;

fn synth_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| ((i * 17 + j * 3) as f64 * 0.019).sin())
                .collect()
        })
        .collect()
}

fn base_config() -> UmapConfig {
    UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 8,
            verbose: false,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        hidden_sizes: vec![16],
        ..Default::default()
    }
}

#[test]
fn fit_is_deterministic() {
    register();
    let data = synth_data(48, 6);
    let config = base_config();

    let e1 = Umap::new(config.clone())
        .fit(data.clone())
        .embedding()
        .to_vec();
    let e2 = Umap::new(config).fit(data).embedding().to_vec();

    assert_embeddings_match(&e1, &e2);
}

#[test]
fn train_only_is_deterministic() {
    register();
    let data = synth_data(48, 6);
    let config = base_config();
    let (mut flat1, n, d) = flatten_f64(&data);
    let (mut flat2, _, _) = flatten_f64(&data);

    let r1 = train_only(&config, &mut flat1, n, d, Device::Cpu, None);
    let r2 = train_only(&config, &mut flat2, n, d, Device::Cpu, None);
    assert_weights_match(&r1.weights, &r2.weights);
}

/// Regression: repeated `CompiledGraph::run` on the same executable must
/// not read stale arena slots left over from the previous pass.
#[test]
fn repeat_same_step_on_one_compile_is_deterministic() {
    register();
    let data = synth_data(48, 6);
    let (mut flat, n, d) = flatten_f64(&data);
    normalize_data_f64(&mut flat, n, d);
    let x = f64_to_f32(&flat);
    let config = base_config();
    let train_cfg = TrainingConfig::from_umap_config(&config);
    let spec = ModelSpec::from_config(&config, n, d);
    let metric = config.graph.metric.clone();
    let edges = build_knn_edges(&x, n, d, 5, &metric, Device::Cpu);
    let n_pos = edges.len();
    let n_neg = (n_pos * train_cfg.neg_sample_rate).max(n);

    let mut head = Vec::with_capacity(n_pos + n_neg);
    let mut tail = Vec::with_capacity(n_pos + n_neg);
    for &(h, t) in &edges {
        head.push(h as f32);
        tail.push(t as f32);
    }
    for i in 0..n_neg {
        head.push((i % n) as f32);
        tail.push(((i + 1) % n) as f32);
    }

    let session = Session::new(Device::Cpu);
    let mut compiled = CompiledUmap::compile(&session, &spec, n_pos, n_neg);
    let weights = init_model_weights(&spec, 9999);
    compiled.set_weights(&weights);

    let ka = [train_cfg.kernel_a];
    let kb = [train_cfg.kernel_b];
    let rep = [train_cfg.repulsion_strength];
    let inputs = [
        ("x", x.as_slice()),
        ("edge_h", head.as_slice()),
        ("edge_t", tail.as_slice()),
        ("kernel_a", ka.as_slice()),
        ("kernel_b", kb.as_slice()),
        ("repulsion", rep.as_slice()),
        ("d_output", &[1.0f32][..]),
    ];

    let o1 = compiled.train.run(&inputs);
    let o2 = compiled.train.run(&inputs);
    for (a, b) in o1.iter().zip(&o2) {
        for (x, y) in a.iter().zip(b) {
            assert_eq!(x.to_bits(), y.to_bits(), "repeat step mismatch {x} vs {y}");
        }
    }
}

fn assert_embeddings_match(e1: &[Vec<f64>], e2: &[Vec<f64>]) {
    assert_eq!(e1.len(), e2.len());
    for (a, b) in e1.iter().zip(e2) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            let tol = 1e-3 * x.abs().max(y.abs()).max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "non-deterministic fit: {x} vs {y} (tol {tol})"
            );
        }
    }
}

fn assert_weights_match(w1: &WeightStore, w2: &WeightStore) {
    let mut max_diff = 0.0f32;
    for name in w1.0.keys() {
        for (a, b) in w1.get(name).unwrap().iter().zip(w2.get(name).unwrap()) {
            max_diff = max_diff.max((*a - *b).abs());
        }
    }
    assert!(max_diff <= 1e-6, "weight mismatch max_diff={max_diff}");
}
