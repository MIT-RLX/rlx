//! Deterministic fit with fixed data (regression guard).

use rlx_umap::prelude::*;

fn synth_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| ((i * 17 + j * 3) as f64 * 0.019).sin())
                .collect()
        })
        .collect()
}

#[test]
fn fit_is_deterministic() {
    register();
    let data = synth_data(48, 6);
    let config = UmapConfig {
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
    };

    let e1 = Umap::new(config.clone())
        .fit(data.clone())
        .embedding()
        .to_vec();
    let e2 = Umap::new(config).fit(data).embedding().to_vec();

    assert_eq!(e1.len(), e2.len());
    for (a, b) in e1.iter().zip(&e2) {
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
