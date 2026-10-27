// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The warm/cold codec reads and writes standard zstd frames whichever
//! implementation is built (`ruzstd`, or libzstd with `zstd-native`): frames
//! libzstd wrote decode through the crate, and the crate's frames decode with
//! both libzstd and ruzstd.

#![cfg(not(target_arch = "wasm32"))]
use rlx_pkg::{
    decode_zstd_block_at, decode_zstd_blocks, decode_zstd_blocks_parallel, encode_zstd_blocks,
};
use std::io::Read;

/// Weight-like bytes: f32s from an LCG, partly repetitive (compressible).
fn weights(n: usize) -> Vec<u8> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .flat_map(|i| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = if i % 7 == 0 {
                0.0
            } else {
                ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            };
            v.to_le_bytes()
        })
        .collect()
}

/// A `ZBLK` warm blob as `encode_zstd_blocks` lays it out, framed by libzstd.
fn zblk_by_libzstd(raw: &[u8], block: usize, level: i32) -> Vec<u8> {
    let mut out = b"ZBLK".to_vec();
    out.extend_from_slice(&(block as u32).to_le_bytes());
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    out.extend_from_slice(&(raw.len().div_ceil(block) as u32).to_le_bytes());
    for c in raw.chunks(block) {
        let f = zstd::encode_all(c, level).unwrap();
        out.extend_from_slice(&(f.len() as u32).to_le_bytes());
        out.extend_from_slice(&f);
    }
    out
}

/// The frames of a `ZBLK` blob.
fn frames(blob: &[u8]) -> Vec<&[u8]> {
    let n = u32::from_le_bytes(blob[16..20].try_into().unwrap()) as usize;
    let mut pos = 20;
    (0..n)
        .map(|_| {
            let l = u32::from_le_bytes(blob[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + l;
            &blob[pos - l..pos]
        })
        .collect()
}

#[test]
fn libzstd_frames_decode_through_the_crate() {
    let raw = weights(700_000); // 2.8 MB: two full 1 MiB blocks and a remainder
    for level in [1, 3, 19] {
        let blob = zblk_by_libzstd(&raw, 1 << 20, level);
        assert_eq!(decode_zstd_blocks(&blob).unwrap(), raw, "level {level}");
        assert_eq!(
            decode_zstd_blocks_parallel(&blob).unwrap(),
            raw,
            "level {level}"
        );
        assert_eq!(
            decode_zstd_block_at(&blob, 2).unwrap(),
            &raw[2 << 20..],
            "level {level}"
        );
    }
}

#[test]
fn the_crates_frames_decode_with_libzstd_and_ruzstd() {
    let raw = weights(700_000);
    let blob = encode_zstd_blocks(&raw, 1 << 20).unwrap();
    let by_libzstd: Vec<u8> = frames(&blob)
        .iter()
        .flat_map(|f| zstd::decode_all(*f).unwrap())
        .collect();
    let by_ruzstd: Vec<u8> = frames(&blob)
        .iter()
        .flat_map(|f| {
            let mut out = Vec::new();
            ruzstd::decoding::StreamingDecoder::new(*f)
                .unwrap()
                .read_to_end(&mut out)
                .unwrap();
            out
        })
        .collect();
    assert_eq!(by_libzstd, raw);
    assert_eq!(by_ruzstd, raw);
}

/// A block whose frame decodes past its declared length is refused, not
/// returned: the output cap holds with either codec.
#[test]
fn an_oversized_block_is_refused() {
    let raw = weights(10_000);
    let mut blob = zblk_by_libzstd(&raw, 1 << 20, 3);
    // declare one byte less than the frame holds
    blob[8..16].copy_from_slice(&(raw.len() as u64 - 1).to_le_bytes());
    assert!(decode_zstd_blocks(&blob).is_err());
}
