// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `remote::http_range` over HTTPS with the pure-Rust TLS provider (`--ignored`):
//! the bytes curl returns for the same ranges of a public Hugging Face file.
#![cfg(feature = "remote")]

const URL: &str = "https://huggingface.co/starself/MIRepNet/resolve/main/MIRepNet.pth";

#[test]
#[ignore = "network: range requests to huggingface.co"]
fn https_ranges_match_curl() {
    // a torch checkpoint is a zip: `PK\x03\x04`
    assert_eq!(
        rlx_pkg::http_range(URL, 0, 3).unwrap(),
        [0x50, 0x4b, 0x03, 0x04]
    );
    let got = rlx_pkg::http_range(URL, 1048570, 1048585).unwrap();
    let want: Vec<u8> = (0..16)
        .map(|i| {
            u8::from_str_radix(&"b2bde110723c43dcd8bd7cefb5bd2c57"[2 * i..2 * i + 2], 16).unwrap()
        })
        .collect();
    assert_eq!(got, want, "`curl -r 1048570-1048585` of the same file");
}
