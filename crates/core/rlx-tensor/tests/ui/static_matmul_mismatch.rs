// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [2, 4];
        param w: [8, 3];   // inner dims 4 vs 8
        let y = x @ w;
        out y;
    };
}
