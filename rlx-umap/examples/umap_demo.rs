// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// ```sh
// cargo run -p rlx-umap --release --example umap_demo --features full
// ```

use rlx_umap::prelude::*;

fn main() {
    register();
    let n = 200;
    let d = 16;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| ((i + j) as f64 * 0.13).sin()).collect())
        .collect();

    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 50,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let fitted = Umap::new(config).fit(data);
    println!(
        "embedding[0] = [{:.4}, {:.4}]",
        fitted.embedding()[0][0],
        fitted.embedding()[0][1]
    );
}
