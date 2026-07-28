// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Embedded SPIR-V compute kernels, compiled from `shaders/*.comp` by
//! `build.rs` (naga GLSL → SPIR-V). Each blob is the raw little-endian
//! SPIR-V word stream for one `@compute` entry point named `main`.

include!(concat!(env!("OUT_DIR"), "/shaders_generated.rs"));

/// Look up a kernel's SPIR-V blob by name. Returns the little-endian byte
/// stream; callers convert to `&[u32]` for `vkCreateShaderModule`.
pub fn blob(name: &str) -> Option<&'static [u8]> {
    SHADER_BLOBS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, b)| *b)
}

/// All embedded kernel names (sorted, deterministic — build.rs sorts).
pub fn names() -> impl Iterator<Item = &'static str> {
    SHADER_BLOBS.iter().map(|(n, _)| *n)
}

/// Convert a little-endian SPIR-V byte blob to a word stream. SPIR-V is
/// consumed in host endianness; all RLX targets are little-endian.
pub fn words(blob: &[u8]) -> Vec<u32> {
    assert!(
        blob.len().is_multiple_of(4),
        "rlx-vulkan: SPIR-V blob length {} not a multiple of 4",
        blob.len()
    );
    blob.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
