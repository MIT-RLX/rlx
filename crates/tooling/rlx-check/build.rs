// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! With the `ble` feature on macOS, link `Info.plist` into the binaries'
//! `__TEXT,__info_plist`: macOS reads `NSBluetoothAlwaysUsageDescription`
//! from there before it hands a process Bluetooth, and a binary outside an
//! app bundle has nowhere else to carry it (without it, no adapter is seen).

fn main() {
    println!("cargo::rerun-if-changed=Info.plist");
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if macos && std::env::var_os("CARGO_FEATURE_BLE").is_some() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets this");
        println!("cargo::rustc-link-arg-bins=-Wl,-sectcreate,__TEXT,__info_plist,{dir}/Info.plist");
    }
}
