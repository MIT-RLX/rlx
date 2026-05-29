//! Full `Umap::fit` smoke test on MLX (k-NN hybrid + native MLP backward).

#![cfg(all(feature = "full", feature = "mlx", target_os = "macos"))]

use rlx_driver::Device;
use rlx_runtime::device_ext;
use rlx_umap::prelude::*;

#[test]
fn mlx_parametric_fit_smoke() {
    if !device_ext::is_available(Device::Mlx) {
        eprintln!("skip: MLX not available");
        return;
    }

    register();

    let n = 64;
    let d = 16;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d).map(|j| ((i + j) as f64 * 0.07).sin()).collect())
        .collect();

    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 5,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let fitted = Umap::with_device(config, Device::Mlx).fit(data);
    let emb = fitted.embedding();
    assert_eq!(emb.len(), n);
    assert_eq!(emb[0].len(), 2);
}
