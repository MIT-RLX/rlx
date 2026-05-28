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
//! Fused GGUF K-quant dequant + matmul without materializing full F32
//! weights (Tier C.11).
//!
//! Computes `C[m,n] = A[m,k] @ B^T` where `B` is `[n,k]` row-major in
//! packed GGUF layout. One 256-element super-block is dequantized at a
//! time into stack storage and accumulated into `C`.

use rlx_gguf::QK_K;
use rlx_ir::quant::QuantScheme;

pub(crate) fn dequant_block(scheme: QuantScheme, block: &[u8], out: &mut [f32; QK_K]) {
    match scheme {
        QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k_block(block, out),
        QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k_block(block, out),
        QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k_block(block, out),
        QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k_block(block, out),
        QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k_block(block, out),
        QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k_block(block, out),
        QuantScheme::GgufQ4_0 => rlx_gguf::dequant_q4_0_block(block, &mut out[..rlx_gguf::QK4_0]),
        QuantScheme::GgufQ8_0 => rlx_gguf::dequant_q8_0_block(block, &mut out[..rlx_gguf::QK8_0]),
        other => panic!("gguf_matmul: unsupported scheme {other:?}"),
    }
}

/// Fused dequant + `sgemm_bt` — `out` is zeroed then accumulated.
pub fn gguf_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(out.len(), m * n);
    out.fill(0.0);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let total_elems = k * n;
    debug_assert!(
        total_elems.is_multiple_of(block_elems),
        "k*n={total_elems} not aligned to GGUF block {block_elems}"
    );
    let num_blocks = total_elems / block_elems;
    debug_assert_eq!(w_bytes.len(), num_blocks * block_bytes);

    let mut block_f32 = [0f32; QK_K];

    if m == 1 {
        let x_row = x;
        for bi in 0..num_blocks {
            let off = bi * block_bytes;
            dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
            let idx0 = bi * block_elems;
            for t in 0..block_elems {
                let idx = idx0 + t;
                let j = idx / k;
                let p = idx % k;
                out[j] += x_row[p] * block_f32[t];
            }
        }
        return;
    }

    for bi in 0..num_blocks {
        let off = bi * block_bytes;
        dequant_block(scheme, &w_bytes[off..off + block_bytes], &mut block_f32);
        let idx0 = bi * block_elems;
        for t in 0..block_elems {
            let idx = idx0 + t;
            let j = idx / k;
            let p = idx % k;
            let w_val = block_f32[t];
            for mi in 0..m {
                out[mi * n + j] += x[mi * k + p] * w_val;
            }
        }
    }
}

/// Fused GGUF dequant + grouped matmul for MoE expert stacks.
///
/// `w_bytes` holds `num_experts` contiguous packed slabs; expert `e` occupies
/// `[e * slab_bytes .. (e+1) * slab_bytes)` with the same GGML layout as a
/// standalone 2-D K-quant matrix of shape `[n, k]`.
pub fn gguf_grouped_matmul_bt(
    x: &[f32],
    w_bytes: &[u8],
    expert_idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: QuantScheme,
) {
    assert_eq!(x.len(), m * k);
    assert_eq!(expert_idx.len(), m);
    assert_eq!(out.len(), m * n);

    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(w_bytes.len(), num_experts * slab_bytes);

    let (packed_in, original_pos, offsets) =
        grouped_moe_sort_plan(x, expert_idx, m, k, num_experts);

    let mut packed_out = vec![0f32; m * n];
    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let in_start = offsets[e];
        let in_slice = &packed_in[in_start * k..(in_start + count) * k];
        let w_slice = &w_bytes[e * slab_bytes..(e + 1) * slab_bytes];
        let out_slice = &mut packed_out[in_start * n..(in_start + count) * n];
        gguf_matmul_bt(in_slice, w_slice, out_slice, count, k, n, scheme);
    }

    grouped_moe_unpermute_out(&packed_out, &original_pos, out, m, n);
}

/// Dequant an MoE expert stack `[E, K, N]` into GroupedMatMul layout (row-major
/// `[k, n]` slabs per expert). Used by `Op::DequantMoEWeights` and autodiff.
pub fn dequant_moe_weights_to_grouped_f32(
    packed: &[u8],
    out: &mut [f32],
    num_experts: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k * n) / block_elems * block_bytes;
    assert_eq!(packed.len(), num_experts * slab_bytes);
    assert_eq!(out.len(), num_experts * k * n);
    for e in 0..num_experts {
        let slab = &packed[e * slab_bytes..(e + 1) * slab_bytes];
        let deq = match scheme {
            QuantScheme::GgufQ4K => rlx_gguf::dequant_q4_k(slab, k * n),
            QuantScheme::GgufQ5K => rlx_gguf::dequant_q5_k(slab, k * n),
            QuantScheme::GgufQ6K => rlx_gguf::dequant_q6_k(slab, k * n),
            QuantScheme::GgufQ8K => rlx_gguf::dequant_q8_k(slab, k * n),
            QuantScheme::GgufQ2K => rlx_gguf::dequant_q2_k(slab, k * n),
            QuantScheme::GgufQ3K => rlx_gguf::dequant_q3_k(slab, k * n),
            other => panic!("dequant_moe_weights: unsupported scheme {other:?}"),
        }
        .expect("dequant_moe_weights: slab dequant failed");
        let base = e * k * n;
        for i in 0..k {
            for j in 0..n {
                out[base + i * n + j] = deq[j * k + i];
            }
        }
    }
}

/// Counting-sort tokens by expert (shared by host and GPU prep paths).
pub fn grouped_moe_sort_plan(
    x: &[f32],
    expert_idx: &[f32],
    m: usize,
    k: usize,
    num_experts: usize,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut counts = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        debug_assert!(e < num_experts);
        counts[e] += 1;
    }
    let mut offsets = vec![0usize; num_experts + 1];
    for e in 0..num_experts {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut packed_in = vec![0f32; m * k];
    let mut original_pos = vec![0usize; m];
    let mut write_idx = vec![0usize; num_experts];
    for i in 0..m {
        let e = expert_idx[i] as usize;
        let dst_row = offsets[e] + write_idx[e];
        packed_in[dst_row * k..(dst_row + 1) * k].copy_from_slice(&x[i * k..(i + 1) * k]);
        original_pos[dst_row] = i;
        write_idx[e] += 1;
    }
    (packed_in, original_pos, offsets)
}

pub fn grouped_moe_unpermute_out(
    packed_out: &[f32],
    original_pos: &[usize],
    out: &mut [f32],
    m: usize,
    n: usize,
) {
    for packed_idx in 0..m {
        let i = original_pos[packed_idx];
        out[i * n..(i + 1) * n].copy_from_slice(&packed_out[packed_idx * n..(packed_idx + 1) * n]);
    }
}

/// Parallel fused matmul for large `n` decode matvecs (e.g. LM head).
pub fn gguf_matmul_bt_parallel(
    x: &[f32],
    w_bytes: &[u8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scheme: QuantScheme,
) {
    gguf_matmul_bt(x, w_bytes, out, m, k, n, scheme);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_q8k_matches_full_dequant() {
        let k = 256;
        let n = 4;
        let m = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..n {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let w_ref = rlx_gguf::dequant_q8_k(&packed, k * n).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let mut fused = vec![0f32; m * n];
        gguf_matmul_bt(&x, &packed, &mut fused, m, k, n, QuantScheme::GgufQ8K);
        let mut expected = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[r * k + i] * w_ref[c * k + i];
                }
                expected[r * n + c] = acc;
            }
        }
        for i in 0..fused.len() {
            assert!(
                (fused[i] - expected[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                fused[i],
                expected[i]
            );
        }
    }

    #[test]
    fn grouped_q8k_matches_per_expert_reference() {
        let k = 256;
        let n = 4;
        let m = 3;
        let num_experts = 2;
        let scale = 0.5f32;
        let mut packed = Vec::new();
        for _ in 0..(num_experts * n) {
            packed.extend_from_slice(&scale.to_le_bytes());
            for i in 0..QK_K {
                let q = (i as i32 - 128).clamp(-128, 127) as i8;
                packed.push(q as u8);
            }
            for _ in 0..(QK_K / 16) {
                packed.extend_from_slice(&0i16.to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..m * k).map(|i| 0.01 * i as f32).collect();
        let expert_idx = vec![0f32, 1.0, 0.0];
        let mut grouped = vec![0f32; m * n];
        gguf_grouped_matmul_bt(
            &x,
            &packed,
            &expert_idx,
            &mut grouped,
            m,
            k,
            n,
            num_experts,
            QuantScheme::GgufQ8K,
        );
        let slab = (k * n) / QK_K * QuantScheme::GgufQ8K.gguf_block_bytes() as usize;
        let mut expected = vec![0f32; m * n];
        for row in 0..m {
            let e = expert_idx[row] as usize;
            let w_ref = rlx_gguf::dequant_q8_k(&packed[e * slab..(e + 1) * slab], k * n).unwrap();
            for col in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[row * k + i] * w_ref[col * k + i];
                }
                expected[row * n + col] = acc;
            }
        }
        for i in 0..grouped.len() {
            assert!(
                (grouped[i] - expected[i]).abs() < 1e-4,
                "i={i}: {} vs {}",
                grouped[i],
                expected[i]
            );
        }
    }
}
