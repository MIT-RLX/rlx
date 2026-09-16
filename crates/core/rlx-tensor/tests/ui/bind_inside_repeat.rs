// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [4, 4];
        repeat 2 {
            bind w;            // `bind` adopts a name — not allowed inside a loop
            let x = x @ w;
        }
        out x;
    };
}
