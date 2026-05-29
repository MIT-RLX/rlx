//! `transform` uses training normalization statistics.

use rlx_umap::prelude::*;

#[test]
fn transform_matches_training_embedding() {
    register();
    let train: Vec<Vec<f64>> = (0..32)
        .map(|i| vec![i as f64 * 0.1, (i as f64 * 0.07).sin()])
        .collect();
    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 10,
            ..Default::default()
        },
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        hidden_sizes: vec![16],
        ..Default::default()
    };
    let mut fitted = Umap::new(config).fit(train.clone());
    let emb0 = fitted.embedding()[0].clone();
    let proj = fitted.transform(vec![train[0].clone()]);
    assert_eq!(proj.len(), 1);
    for (a, b) in emb0.iter().zip(&proj[0]) {
        assert!(
            (a - b).abs() < 1e-3,
            "transform should match in-sample forward: emb0={a} proj={b}"
        );
    }
}
