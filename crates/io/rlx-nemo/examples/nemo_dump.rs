// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dump a `.nemo`'s config + state-dict tensor names/shapes and spot-check
//! a few tensors. Usage: `cargo run -p rlx-nemo --example nemo_dump -- <file.nemo>`.

use std::path::PathBuf;

use rlx_nemo::NemoModel;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: nemo_dump <file.nemo>");

    let model = NemoModel::open(&path)?;
    let cfg = model.config();
    eprintln!(
        "config: sample_rate={:?} d_model={:?} n_layers={:?} n_heads={:?} features={:?}",
        cfg.get_i64_any(&["preprocessor.sample_rate", "sample_rate"]),
        cfg.get_i64("encoder.d_model"),
        cfg.get_i64("encoder.n_layers"),
        cfg.get_i64("encoder.n_heads"),
        cfg.get_i64("preprocessor.features"),
    );
    eprintln!("tokenizers:");
    for t in model.tokenizers() {
        eprintln!("  {} ({} bytes)", t.name, t.bytes.len());
    }

    let names = model.names();
    println!("# {} tensors", names.len());
    for name in &names {
        let shape = model.shape_of(name).unwrap_or(&[]);
        println!("{name}\t{shape:?}");
    }

    // Spot-check: read the first few tensors fully.
    let mut checked = 0;
    for name in names.iter().take(5) {
        let t = model.tensor(name)?;
        let numel: usize = t.shape.iter().product();
        assert_eq!(t.data.len(), numel);
        assert!(t.data.iter().all(|x| x.is_finite()));
        checked += 1;
    }
    eprintln!("spot-checked {checked} tensors: finite, correctly shaped f32");
    Ok(())
}
