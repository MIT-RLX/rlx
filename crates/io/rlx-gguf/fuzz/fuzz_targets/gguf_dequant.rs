// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the GGUF dequantizers on fuzzer-controlled tensor bytes.
//!
//! After a successful full parse, walk every tensor and call
//! `dequant_f32`, which routes into the per-scheme block decoders — the
//! bulk of the crate's bit-twiddling: Q4_0/Q4_1/Q5_*/Q8_0, the K-quant
//! family (Q2_K..Q8_K), the I-quants (IQ1..IQ4), and TQ/MX/FV5/I8/I16/I32.
//!
//! `tensor_bytes` bounds every slice against the parsed data segment
//! before a decoder runs, so the output `Vec<f32>` is bounded by the input
//! size — this target exercises the decoders without unbounded allocation.

#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cur = Cursor::new(data);
    let Ok(f) = rlx_gguf::GgufFile::from_reader(&mut cur) else {
        return;
    };
    // Snapshot the names under the immutable borrow, then decode each
    // tensor with the fuzzer-supplied quantized bytes.
    let names: Vec<String> = f.keys().map(str::to_string).collect();
    for name in names {
        let _ = f.dequant_f32(&name);
    }
});
