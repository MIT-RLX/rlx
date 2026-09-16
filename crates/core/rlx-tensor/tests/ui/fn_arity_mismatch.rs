// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        fn block(x, w) { let h = x @ w; }
        input a: [2, 4];
        let o = block(a);   // block takes 2 args
        out o;
    };
}
