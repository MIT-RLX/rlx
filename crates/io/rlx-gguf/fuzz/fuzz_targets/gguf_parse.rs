// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz the GGUF header / tensor-table parser on untrusted bytes.
//!
//! Two entry points are driven per input:
//!   * `GgufFile::header_from_bytes(&[u8])` — the in-memory-prefix parse
//!     (magic → version gate → KV-metadata loop → tensor table), no data
//!     slurp.
//!   * `GgufFile::from_reader(&mut Cursor<&[u8]>)` — the full parse, which
//!     additionally computes the alignment padding and slurps the tensor
//!     data segment to EOF.
//!
//! Malformed input must never panic or corrupt memory; the parse `Result`
//! is intentionally discarded — we only want the fuzzer to trip on a crash,
//! not on a rejected file.

#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // In-memory prefix parse: magic + version + KV metadata + tensor table.
    let _ = rlx_gguf::GgufFile::header_from_bytes(data);

    // Full parse over the same bytes, including the tensor-data slurp.
    let mut cur = Cursor::new(data);
    let _ = rlx_gguf::GgufFile::from_reader(&mut cur);
});
