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
