// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Pack low-bit-width signed weights into a byte stream that matches
//! `rlx_cortexm::quant::read_weight` exactly.
//!
//! Layout (same as cortexm / TFLite-Micro):
//!
//! * **8-bit** — `bytes[i] = weights[i] as i8` (no packing).
//! * **4-bit** — two nibbles per byte, low nibble first:
//!   `bytes[i/2]`, with `weights[i]` in the low 4 bits when `i` is even
//!   and the high 4 bits when `i` is odd. Each nibble is two's-complement
//!   signed (range `[-8, 7]`; trainers stick to `[-7, 7]`).
//! * **2-bit** — four crumbs per byte, indexed from the LSB up:
//!   `bytes[i/4]`, with `weights[i]` at bit position `(i % 4) * 2`.
//!   Two's-complement signed (range `[-2, 1]`; trainers use ternary
//!   `{-1, 0, 1}`).
//!
//! These helpers exist mainly so synthetic test layers can hand-build a
//! packed weight tensor without going through the trainer. Production
//! weights are emitted directly by `rlx-cortexm-trainer` in this exact
//! layout, so emitting them here would be redundant.

/// Round up `n` divided by `d`.
#[inline]
const fn div_ceil(n: usize, d: usize) -> usize {
    n.div_ceil(d)
}

/// Pack `weights` at `bits ∈ {2, 4, 8}` bits-per-element into the
/// byte-stream layout `read_weight` expects.
pub fn pack(weights: &[i8], bits: u8) -> Vec<i8> {
    match bits {
        8 => weights.to_vec(),
        4 => {
            let n = div_ceil(weights.len(), 2);
            let mut out = vec![0i8; n];
            for (i, &w) in weights.iter().enumerate() {
                let nib = (w as u8) & 0x0F;
                out[i / 2] = ((out[i / 2] as u8) | (nib << ((i & 1) * 4))) as i8;
            }
            out
        }
        2 => {
            let n = div_ceil(weights.len(), 4);
            let mut out = vec![0i8; n];
            for (i, &w) in weights.iter().enumerate() {
                let crumb = (w as u8) & 0x03;
                out[i / 4] = ((out[i / 4] as u8) | (crumb << ((i & 3) * 2))) as i8;
            }
            out
        }
        _ => panic!("pack: unsupported bits {bits} (must be 2, 4 or 8)"),
    }
}

/// Number of logical weights per packed byte: 1, 2, or 4.
#[inline]
pub const fn weights_per_byte(bits: u8) -> usize {
    (8 / bits) as usize
}

/// Length in bytes of a packed-weight tensor.
#[inline]
pub const fn packed_byte_len(n_weights: usize, bits: u8) -> usize {
    div_ceil(n_weights, weights_per_byte(bits))
}

/// Split an oc-major weight tensor (cortexm layout, packed at `bits`)
/// into `p` byte-packed lanes, with **lane `q` holding all weights for
/// output channels where `oc % p == q`** at the same `bits` packing.
///
/// Each lane's contents (in logical-weight order) are:
/// `for each oc with oc % p == q { for k in 0..inner { weights[oc*inner + k] } }`.
///
/// `inner` is `kh*kw*c_in` for conv2d or `in_features` for dense.
///
/// Used by the P-MAC parallel conv2d kernel: each lane gets its own
/// weight ROM addressed by `(oc/p)*inner + (kh*KW + kw)*C_IN + ic`,
/// and all P ROMs read in parallel every cycle.
pub fn split_weights_by_oc_lane(
    packed: &[i8],
    c_out: usize,
    inner: usize,
    bits: u8,
    p: usize,
) -> Vec<Vec<i8>> {
    use rlx_cortexm::quant::read_weight;
    assert!(p > 0);
    assert_eq!(
        c_out % p,
        0,
        "split_weights_by_oc_lane: c_out ({c_out}) not divisible by p ({p})"
    );
    let mut lanes_logical: Vec<Vec<i8>> = vec![Vec::with_capacity((c_out / p) * inner); p];
    for oc in 0..c_out {
        let lane = oc % p;
        for k in 0..inner {
            let logical_idx = oc * inner + k;
            let v = read_weight(packed, logical_idx, bits);
            lanes_logical[lane].push(v as i8);
        }
    }
    lanes_logical.into_iter().map(|w| pack(&w, bits)).collect()
}

/// Split an oc-indexed table (bias, requant) into `p` lanes by `oc % p`.
///
/// Lane `q` contains entries `[table[q], table[q+p], table[q+2p], ...]`,
/// in oc_block order. Each lane has `c_out / p` entries.
pub fn split_table_by_oc_lane<T: Copy>(table: &[T], p: usize) -> Vec<Vec<T>> {
    assert!(p > 0);
    assert_eq!(
        table.len() % p,
        0,
        "split_table_by_oc_lane: table.len ({}) not divisible by p ({p})",
        table.len()
    );
    let mut lanes: Vec<Vec<T>> = vec![Vec::with_capacity(table.len() / p); p];
    for (oc, &v) in table.iter().enumerate() {
        lanes[oc % p].push(v);
    }
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_cortexm::quant::read_weight;

    /// Round-trip every supported width: pack(weights, bits) → read_weight
    /// must recover the original values (modulo two's-complement masking
    /// of the unused codepoints we don't emit anyway).
    #[test]
    fn pack_roundtrip_8bit() {
        let w: Vec<i8> = (-128..=127)
            .collect::<Vec<_>>()
            .iter()
            .map(|&x| x as i8)
            .collect();
        let packed = pack(&w, 8);
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(read_weight(&packed, i, 8), expected as i32);
        }
    }

    #[test]
    fn pack_roundtrip_4bit() {
        // i4 range we actually emit: [-7, 7]
        let w: Vec<i8> = (-7..=7).collect();
        let packed = pack(&w, 4);
        assert_eq!(packed.len(), packed_byte_len(w.len(), 4));
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(read_weight(&packed, i, 4), expected as i32);
        }
    }

    #[test]
    fn pack_roundtrip_2bit_ternary() {
        // Ternary: weights ∈ {-1, 0, 1}
        let w: Vec<i8> = vec![1, -1, 0, 1, 0, 1, -1, 0, -1, 1];
        let packed = pack(&w, 2);
        assert_eq!(packed.len(), packed_byte_len(w.len(), 2));
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(read_weight(&packed, i, 2), expected as i32);
        }
    }

    #[test]
    fn split_lanes_8bit_roundtrip() {
        // 4 channels × 2 inner = 8 weights. P=2: lane0 holds oc∈{0,2}, lane1 holds {1,3}.
        let w: Vec<i8> = (1i8..=8).collect(); // c_out=4, inner=2
        let packed = pack(&w, 8);
        let lanes = split_weights_by_oc_lane(&packed, 4, 2, 8, 2);
        assert_eq!(lanes.len(), 2);
        // lane0: oc=0 → [1,2], oc=2 → [5,6]
        assert_eq!(read_weight(&lanes[0], 0, 8), 1);
        assert_eq!(read_weight(&lanes[0], 1, 8), 2);
        assert_eq!(read_weight(&lanes[0], 2, 8), 5);
        assert_eq!(read_weight(&lanes[0], 3, 8), 6);
        // lane1: oc=1 → [3,4], oc=3 → [7,8]
        assert_eq!(read_weight(&lanes[1], 0, 8), 3);
        assert_eq!(read_weight(&lanes[1], 1, 8), 4);
        assert_eq!(read_weight(&lanes[1], 2, 8), 7);
        assert_eq!(read_weight(&lanes[1], 3, 8), 8);
    }

    #[test]
    fn split_lanes_2bit_roundtrip() {
        // 4 channels × 4 inner = 16 ternary weights. P=2.
        let w: Vec<i8> = vec![
            1, -1, 0, 1, // oc=0
            0, 1, -1, 0, // oc=1
            -1, 1, 0, -1, // oc=2
            1, 0, 1, -1, // oc=3
        ];
        let packed = pack(&w, 2);
        let lanes = split_weights_by_oc_lane(&packed, 4, 4, 2, 2);
        assert_eq!(lanes.len(), 2);
        // lane0: oc=0 → [1,-1,0,1], oc=2 → [-1,1,0,-1]
        for (i, &expected) in [1, -1, 0, 1, -1, 1, 0, -1].iter().enumerate() {
            assert_eq!(read_weight(&lanes[0], i, 2), expected);
        }
        // lane1: oc=1 → [0,1,-1,0], oc=3 → [1,0,1,-1]
        for (i, &expected) in [0, 1, -1, 0, 1, 0, 1, -1].iter().enumerate() {
            assert_eq!(read_weight(&lanes[1], i, 2), expected);
        }
    }

    #[test]
    fn split_table_partitions_by_oc_mod_p() {
        let bias: Vec<i32> = (10..=17).collect(); // c_out=8
        let lanes = split_table_by_oc_lane(&bias, 4);
        assert_eq!(lanes.len(), 4);
        assert_eq!(lanes[0], vec![10, 14]); // oc 0, 4
        assert_eq!(lanes[1], vec![11, 15]); // oc 1, 5
        assert_eq!(lanes[2], vec![12, 16]); // oc 2, 6
        assert_eq!(lanes[3], vec![13, 17]); // oc 3, 7
    }
}
