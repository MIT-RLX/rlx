//! Minimal parametric UMAP training demo (RLX autodiff only — no Burn).
//!
//! ```sh
//! cargo run -p rlx-umap --release --example training_demo --features full
//! ```

use rlx_driver::Device;
use rlx_umap::prelude::*;

fn main() {
    register();

    let data = generate_test_data(256, 16, 7);
    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 20,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let fitted = fit_with_progress(config, data, FitOptions::new(Device::Cpu), |p| {
        eprintln!("epoch {}/{} loss={:.6}", p.epoch, p.total_epochs, p.loss);
    });

    let emb = fitted.embedding();
    println!(
        "embedding {} × {} (first point {:?})",
        emb.len(),
        emb[0].len(),
        emb[0]
    );
}
