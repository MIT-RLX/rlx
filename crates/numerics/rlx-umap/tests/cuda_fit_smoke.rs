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

//! Full `Umap::fit` smoke test on CUDA (`Device::Cuda`).

#![cfg(all(feature = "full", feature = "cuda"))]

use rlx_driver::Device;
use rlx_runtime::device_ext;
use rlx_umap::prelude::*;

#[test]
fn cuda_parametric_fit_smoke() {
    if !device_ext::is_available(Device::Cuda) {
        eprintln!("skip: CUDA not available");
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

    let fitted = Umap::with_device(config, Device::Cuda).fit(data);
    let emb = fitted.embedding();
    assert_eq!(emb.len(), n);
    assert_eq!(emb[0].len(), 2);
}
