// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! On macOS, a PyO3 cdylib needs `-undefined dynamic_lookup` so the
//! Python C-API symbols resolve at import time rather than at link
//! time. Maturin sets this for us, but a plain `cargo build -p pyrlx`
//! would otherwise fail with `_PyBaseObject_Type` undefined. Emit the
//! flag here so direct cargo builds during development just work.

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") || target.contains("apple-ios") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
