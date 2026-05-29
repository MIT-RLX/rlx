// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// ```sh
// cargo run -p rlx-umap --release --example umap_save_load --features full
// ```

use rlx_driver::Device;
use rlx_umap::model_io::{model_path, model_path_with_ext};
use rlx_umap::prelude::*;

fn main() -> std::io::Result<()> {
    register();

    let data: Vec<Vec<f64>> = generate_test_data(128, 16, 42);
    let config = UmapConfig {
        optimization: OptimizationParams {
            n_epochs: 30,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut fitted = Umap::new(config).fit(data.clone());
    let path = model_path(std::env::temp_dir(), "rlx_umap_demo");
    fitted.save(&path)?;
    println!("saved full model to {}", path.display());

    let gguf_path = model_path_with_ext(std::env::temp_dir(), "rlx_umap_demo", "gguf");
    fitted.save(&gguf_path)?;
    println!("saved GGUF model to {}", gguf_path.display());

    let weights_path = model_path(std::env::temp_dir(), "rlx_umap_weights_only");
    fitted.save_weights(&weights_path)?;
    println!("saved weights-only to {}", weights_path.display());

    let mut loaded = FittedUmap::load(&path, Device::Cpu)?;
    let emb = loaded.transform(vec![data[0].clone()]);
    println!(
        "loaded v4 model — transform[0] = [{:.4}, {:.4}]",
        emb[0][0], emb[0][1]
    );

    let w = WeightStore::load(&weights_path)?;
    println!("loaded {} weight tensors (weights-only file)", w.0.len());

    Ok(())
}
