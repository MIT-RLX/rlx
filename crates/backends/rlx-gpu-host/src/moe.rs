// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared host half of the hot-on-GPU / cold-on-CPU MoE split.
//!
//! Used identically by the CUDA and ROCm backends. The backend runs the *hot*
//! experts on-device over the slot buffer; this reads the *cold* tokens' inputs
//! back from the arena, matmuls them against the host-resident cold expert
//! weights ([`rlx_cpu::moe_split::cold_grouped_matmul`]), and writes the results
//! back into the arena's output — leaving the hot rows (already produced by the
//! device kernel) untouched.
//!
//! Cold experts never occupy device memory, so the caller can size its device
//! slot buffer to only the hot experts — that is where the VRAM saving comes from.
//! This function is generic over [`DeviceArena`], so its logic is fully exercised
//! by the in-memory mock in the tests below without a GPU.

use crate::DeviceArena;

/// Expert-index dtype as stored in the f32 arena. The grouped-matmul step keeps
/// routed expert ids as `f32` (see the CUDA/ROCm `GroupedMatmul` step, which reads
/// them back with `x as u32`); we mirror that so offsets line up.
fn read_expert_idx<A: DeviceArena>(a: &mut A, idx_byte_off: usize, m: usize) -> Vec<usize> {
    let mut raw = vec![0u8; m * 4];
    a.dtoh(idx_byte_off, &mut raw);
    let f: &[f32] = bytemuck::cast_slice(&raw);
    f.iter().map(|&v| v.max(0.0) as usize).collect()
}

/// Compute the cold-expert contribution of one MoE grouped matmul on the host.
///
/// - `m,k,n`: token count, in-features, out-features.
/// - `x_byte_off`: arena byte offset of the input `[m,k]` f32.
/// - `idx_byte_off`: arena byte offset of the routed expert ids (`[m]`, f32-coded).
/// - `out_byte_off`: arena byte offset of the output `[m,n]` f32.
/// - `resident`: per-global-expert flag; `true` = hot (device-resident, skipped here).
/// - `host_weights`: full expert stack `[num_experts, k, n]`, expert-major (the
///   host source of truth; the device only holds the hot subset).
///
/// Returns the number of cold tokens processed (0 fast-paths out).
#[allow(clippy::too_many_arguments)]
pub fn run_moe_cold_experts<A: DeviceArena>(
    a: &mut A,
    m: usize,
    k: usize,
    n: usize,
    x_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
    resident: &[bool],
    host_weights: &[f32],
) -> usize {
    a.sync();
    let idx = read_expert_idx(a, idx_byte_off, m);
    let cold: Vec<(usize, usize)> = idx
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, e)| !resident.get(e).copied().unwrap_or(false))
        .collect();
    if cold.is_empty() {
        return 0;
    }

    // Pull the input block once (m is 1 in decode, seq-len in prefill).
    let mut x_raw = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_raw);
    let x: &[f32] = bytemuck::cast_slice(&x_raw);

    // Cold matmul on the host (writes only cold rows of `out`).
    let mut out = vec![0f32; m * n];
    rlx_cpu::moe_split::cold_grouped_matmul(x, k, n, &cold, host_weights, &mut out);

    // Upload just the cold rows so hot rows (device-produced) are preserved.
    for &(t, _) in &cold {
        let row = &out[t * n..(t + 1) * n];
        a.htod(out_byte_off + t * n * 4, bytemuck::cast_slice(row));
    }
    cold.len()
}

/// Cold-expert contribution where the cold weights are read from the **arena**
/// (device) rather than a host slice. This is the phase-1 *compute*-split: the
/// full expert stack still lives in the arena, but cold experts are computed on
/// the host to avoid the device grouped-matmul over them. (The host-slice
/// [`run_moe_cold_experts`] is the phase-2 *VRAM*-saving path, where cold weights
/// live only in host RAM and never occupy the arena.)
///
/// `w_byte_off` is the arena byte offset of the expert stack `[num_experts,k,n]`;
/// `resident` marks device-computed (hot) experts. Passing an all-`false`
/// `resident` computes **every** token on the host — the correctness fallback for
/// an unsorted batch the device split can't cheaply handle.
#[allow(clippy::too_many_arguments)]
pub fn run_moe_cold_experts_from_arena<A: DeviceArena>(
    a: &mut A,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    idx_byte_off: usize,
    out_byte_off: usize,
    resident: &[bool],
) -> usize {
    a.sync();
    let idx = read_expert_idx(a, idx_byte_off, m);
    let cold: Vec<(usize, usize)> = idx
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, e)| e < num_experts && !resident.get(e).copied().unwrap_or(false))
        .collect();
    if cold.is_empty() {
        return 0;
    }

    let mut x_raw = vec![0u8; m * k * 4];
    a.dtoh(x_byte_off, &mut x_raw);
    let x: &[f32] = bytemuck::cast_slice(&x_raw);

    // Pull only the distinct cold experts' weight slabs from the arena.
    let stride = k * n;
    let mut cold_experts: Vec<usize> = cold.iter().map(|&(_, e)| e).collect();
    cold_experts.sort_unstable();
    cold_experts.dedup();
    // Reassemble a sparse [num_experts,k,n] host buffer holding just the cold slabs
    // (other experts are never indexed by the cold token list, so left zero).
    let mut host_weights = vec![0f32; num_experts * stride];
    let mut slab = vec![0u8; stride * 4];
    for &e in &cold_experts {
        a.dtoh(w_byte_off + e * stride * 4, &mut slab);
        host_weights[e * stride..(e + 1) * stride].copy_from_slice(bytemuck::cast_slice(&slab));
    }

    let mut out = vec![0f32; m * n];
    rlx_cpu::moe_split::cold_grouped_matmul(x, k, n, &cold, &host_weights, &mut out);
    for &(t, _) in &cold {
        let row = &out[t * n..(t + 1) * n];
        a.htod(out_byte_off + t * n * 4, bytemuck::cast_slice(row));
    }
    cold.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory arena so the round-trip logic is testable with no GPU.
    struct VecArena {
        data: Vec<u8>,
    }
    impl VecArena {
        fn new(bytes: usize) -> Self {
            Self {
                data: vec![0u8; bytes],
            }
        }
        fn write_f32(&mut self, byte_off: usize, v: &[f32]) {
            self.data[byte_off..byte_off + v.len() * 4].copy_from_slice(bytemuck::cast_slice(v));
        }
        fn read_f32(&self, byte_off: usize, len: usize) -> Vec<f32> {
            bytemuck::cast_slice(&self.data[byte_off..byte_off + len * 4]).to_vec()
        }
    }
    impl DeviceArena for VecArena {
        fn arena_bytes(&self) -> usize {
            self.data.len()
        }
        fn sync(&mut self) {}
        fn dtoh(&mut self, byte_off: usize, dst: &mut [u8]) {
            dst.copy_from_slice(&self.data[byte_off..byte_off + dst.len()]);
        }
        fn htod(&mut self, byte_off: usize, src: &[u8]) {
            self.data[byte_off..byte_off + src.len()].copy_from_slice(src);
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    #[test]
    fn cold_round_trip_matches_reference_and_preserves_hot_rows() {
        let (m, k, n, ne) = (12usize, 4usize, 3usize, 6usize);
        let mut rng = Lcg(0xABCD);
        let input: Vec<f32> = (0..m * k).map(|_| rng.f()).collect();
        let weights: Vec<f32> = (0..ne * k * n).map(|_| rng.f()).collect();
        let expert_idx: Vec<f32> = (0..m).map(|t| (t % ne) as f32).collect();

        // Experts 0,1,2 are hot (resident); 3,4,5 cold.
        let resident = vec![true, true, true, false, false, false];

        // Arena layout: [input | idx | out].
        let x_off = 0usize;
        let idx_off = x_off + m * k * 4;
        let out_off = idx_off + m * 4;
        let mut arena = VecArena::new(out_off + m * n * 4);
        arena.write_f32(x_off, &input);
        arena.write_f32(idx_off, &expert_idx);
        // Pre-fill out with a sentinel to detect which rows the cold pass writes.
        let sentinel = 12345.0f32;
        arena.write_f32(out_off, &vec![sentinel; m * n]);

        let cold_count = run_moe_cold_experts(
            &mut arena, m, k, n, x_off, idx_off, out_off, &resident, &weights,
        );

        // Half the tokens (experts 3,4,5) are cold.
        assert_eq!(cold_count, m / 2);

        let got = arena.read_f32(out_off, m * n);
        // Reference: full grouped matmul, then compare only the cold rows; hot rows
        // must still be the sentinel (this pass must not touch them).
        let idx_u32: Vec<u32> = expert_idx.iter().map(|&v| v as u32).collect();
        let full =
            rlx_cpu::moe_split::grouped_matmul_reference(&input, &weights, &idx_u32, m, k, n);
        for t in 0..m {
            let e = t % ne;
            let row = &got[t * n..(t + 1) * n];
            if resident[e] {
                assert!(
                    row.iter().all(|&v| v == sentinel),
                    "hot row {t} was overwritten by the cold pass"
                );
            } else {
                assert_eq!(
                    row,
                    &full[t * n..(t + 1) * n],
                    "cold row {t} diverged from the full grouped matmul"
                );
            }
        }
    }

    #[test]
    fn arena_variant_reads_cold_weights_from_arena() {
        // Same as the host-slice test, but weights live in the arena and are pulled
        // back per cold expert. Also checks the all-cold fallback (resident all-false).
        let (m, k, n, ne) = (10usize, 4usize, 3usize, 5usize);
        let mut rng = Lcg(0x5151);
        let input: Vec<f32> = (0..m * k).map(|_| rng.f()).collect();
        let weights: Vec<f32> = (0..ne * k * n).map(|_| rng.f()).collect();
        let expert_idx: Vec<f32> = (0..m).map(|t| (t % ne) as f32).collect();
        let idx_u32: Vec<u32> = expert_idx.iter().map(|&v| v as u32).collect();
        let full =
            rlx_cpu::moe_split::grouped_matmul_reference(&input, &weights, &idx_u32, m, k, n);

        // Arena layout: [input | idx | weights | out].
        let x_off = 0usize;
        let idx_off = x_off + m * k * 4;
        let w_off = idx_off + m * 4;
        let out_off = w_off + ne * k * n * 4;
        let mut arena = VecArena::new(out_off + m * n * 4);
        arena.write_f32(x_off, &input);
        arena.write_f32(idx_off, &expert_idx);
        arena.write_f32(w_off, &weights);

        // Case 1: experts 0,1 hot; 2,3,4 cold.
        let resident = vec![true, true, false, false, false];
        arena.write_f32(out_off, &vec![f32::NAN; m * n]);
        run_moe_cold_experts_from_arena(
            &mut arena, m, k, n, ne, x_off, w_off, idx_off, out_off, &resident,
        );
        let got = arena.read_f32(out_off, m * n);
        for t in 0..m {
            let e = t % ne;
            let row = &got[t * n..(t + 1) * n];
            if resident[e] {
                assert!(row.iter().all(|v| v.is_nan()), "hot row {t} touched");
            } else {
                assert_eq!(row, &full[t * n..(t + 1) * n], "cold row {t} wrong");
            }
        }

        // Case 2: all-cold fallback reproduces the full result for every row.
        let all_cold = vec![false; ne];
        arena.write_f32(out_off, &vec![0.0; m * n]);
        let c = run_moe_cold_experts_from_arena(
            &mut arena, m, k, n, ne, x_off, w_off, idx_off, out_off, &all_cold,
        );
        assert_eq!(c, m);
        assert_eq!(arena.read_f32(out_off, m * n), full);
    }

    #[test]
    fn all_hot_is_a_noop() {
        let (m, k, n, ne) = (4usize, 3usize, 2usize, 4usize);
        let resident = vec![true; ne];
        let x_off = 0usize;
        let idx_off = x_off + m * k * 4;
        let out_off = idx_off + m * 4;
        let mut arena = VecArena::new(out_off + m * n * 4);
        arena.write_f32(x_off, &vec![1.0; m * k]);
        arena.write_f32(
            idx_off,
            &(0..m).map(|t| (t % ne) as f32).collect::<Vec<_>>(),
        );
        arena.write_f32(out_off, &vec![7.0; m * n]);
        let weights = vec![0.5f32; ne * k * n];
        let cold = run_moe_cold_experts(
            &mut arena, m, k, n, x_off, idx_off, out_off, &resident, &weights,
        );
        assert_eq!(cold, 0);
        assert!(arena.read_f32(out_off, m * n).iter().all(|&v| v == 7.0));
    }
}
