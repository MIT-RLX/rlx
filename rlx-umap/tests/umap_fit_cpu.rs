//! Smoke test: full `Umap::fit` on CPU.

use rlx_driver::Device;
use rlx_runtime::Session;
use rlx_umap::encoder::mlp::{ModelSpec, init_model_weights};
use rlx_umap::model::CompiledUmap;
use rlx_umap::prelude::*;

#[test]
fn forward_init_bounded() {
    register();
    let n = 64;
    let d = 8;
    let spec = ModelSpec {
        n,
        input_dim: d,
        output_dim: 2,
        hidden: vec![32],
    };
    let weights = init_model_weights(&spec, 42);
    let mut compiled = CompiledUmap::compile(&Session::new(Device::Cpu), &spec, 10, 10);
    compiled.set_weights(&weights);
    let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.01).sin()).collect();
    let emb = compiled.forward_embedding(&x);
    let max_abs = emb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs.is_finite() && max_abs < 20.0,
        "forward at init should be bounded, got max_abs={max_abs}"
    );
}

#[test]
fn umap_fit_small_cpu() {
    register();
    let n = 64;
    let d = 8;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| (i * j) as f64 * 0.01).collect())
        .collect();

    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 5,
            verbose: false,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        ..Default::default()
    };

    let umap = Umap::new(config);
    let fitted = umap.fit(data);
    assert_eq!(fitted.embedding().len(), n);
    assert_eq!(fitted.embedding()[0].len(), 2);

    let max_abs = fitted
        .embedding()
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs.is_finite() && max_abs < 50.0,
        "embedding exploded: max_abs={max_abs}"
    );
}

#[test]
fn umap_save_load_roundtrip() {
    register();
    let n = 32;
    let d = 6;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| (i + j) as f64 * 0.05).collect())
        .collect();
    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let data_for_fit = data.clone();
    let fitted = Umap::new(config.clone()).fit(data_for_fit);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    fitted.save(&path).unwrap();

    let mut loaded = FittedUmap::load(&path, rlx_driver::Device::Cpu).unwrap();
    assert_eq!(loaded.weights().0.len(), fitted.weights().0.len());
    assert_eq!(loaded.config().n_components, 2);

    let proj = loaded.transform(vec![data[0].clone()]);
    let orig = fitted.embedding()[0].clone();
    for (a, b) in orig.iter().zip(&proj[0]) {
        assert!((a - b).abs() < 1e-3, "reload transform drift: {a} vs {b}");
    }

    let w_dir = tempfile::tempdir().unwrap();
    let w_only = w_dir.path().join("w.safetensors");
    fitted.save_weights(&w_only).unwrap();
    let w = WeightStore::load(&w_only).unwrap();

    let gguf_path = dir.path().join("model.gguf");
    fitted.save(&gguf_path).unwrap();
    let mut loaded_gguf = FittedUmap::load(&gguf_path, rlx_driver::Device::Cpu).unwrap();
    let proj_gguf = loaded_gguf.transform(vec![data[0].clone()]);
    assert!((proj_gguf[0][0] - orig[0]).abs() < 1e-3);
    assert_eq!(w.0.len(), fitted.weights().0.len());
}
