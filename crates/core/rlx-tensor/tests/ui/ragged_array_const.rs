// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [2, 2];
        const mask = [[1.0, 0.0], [0.0]] : F32;   // ragged rows
        let y = x * mask;
        out y;
    };
}
