// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host compute for the cold half of the hot-on-GPU / cold-on-CPU MoE split.
//!
//! In the split, a MoE layer's *hot* experts are resident in device slots and run
//! on the accelerator; the remaining *cold* experts never occupy device memory and
//! are computed here on the host from the full expert store. A GPU backend calls
//! [`cold_grouped_matmul`] for the cold-token subset while its own kernel handles
//! the hot tokens over the slot buffer, then merges the two into one output.
//!
//! Because MoE residency is *per expert* (an expert is either fully hot or fully
//! cold — never per-token), every token's output row is produced by exactly one
//! sgemm over one expert's weights regardless of the split. So as long as the slot
//! buffer holds an exact copy of each hot expert's weights and the host store holds
//! the cold ones, the split is **byte-identical** to the full-stack grouped matmul.
//! [`grouped_matmul_reference`] + [`grouped_matmul_split_reference`] make that
//! property testable on CPU (see tests), which is how the GPU port is de-risked.

use std::collections::BTreeMap;

/// Group `(token, expert)` pairs by expert, preserving token order within a group.
fn group_by_key(pairs: &[(usize, usize)]) -> BTreeMap<usize, Vec<usize>> {
    let mut by: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(t, e) in pairs {
        by.entry(e).or_default().push(t);
    }
    by
}

/// Compute the grouped-matmul contribution of the **cold** experts on the host.
///
/// - `input`: `[m, k]` row-major token inputs (the full token set; only the rows
///   named in `cold` are read).
/// - `cold`: `(token_index, global_expert)` for every cold token (from the slot
///   cache's route split).
/// - `host_weights`: the full expert stack `[num_experts, k, n]`, expert-major
///   (e.g. `MoeExpertStore` `ExpertStackF32::as_slice`).
/// - `out`: `[m, n]` row-major; **only** cold token rows are written, other rows
///   are left untouched (the hot path fills those).
pub fn cold_grouped_matmul(
    input: &[f32],
    k: usize,
    n: usize,
    cold: &[(usize, usize)],
    host_weights: &[f32],
    out: &mut [f32],
) {
    if cold.is_empty() {
        return;
    }
    let stride = k * n;
    for (e, tokens) in group_by_key(cold) {
        let w = &host_weights[e * stride..(e + 1) * stride];
        let cnt = tokens.len();
        // Pack this expert's cold-token rows contiguously.
        let mut packed_in = vec![0f32; cnt * k];
        for (r, &t) in tokens.iter().enumerate() {
            packed_in[r * k..(r + 1) * k].copy_from_slice(&input[t * k..(t + 1) * k]);
        }
        let mut packed_out = vec![0f32; cnt * n];
        crate::blas::sgemm(&packed_in, w, &mut packed_out, cnt, k, n);
        // Scatter back to original token rows.
        for (r, &t) in tokens.iter().enumerate() {
            out[t * n..(t + 1) * n].copy_from_slice(&packed_out[r * n..(r + 1) * n]);
        }
    }
}

/// Reference full grouped matmul (no split): one sgemm per expert over its tokens.
/// This is the oracle the split must match byte-for-byte.
///
/// `weights_all`: `[num_experts, k, n]`, `expert_idx`: one global expert id per token.
pub fn grouped_matmul_reference(
    input: &[f32],
    weights_all: &[f32],
    expert_idx: &[u32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    let cold: Vec<(usize, usize)> = expert_idx
        .iter()
        .take(m)
        .enumerate()
        .map(|(t, &e)| (t, e as usize))
        .collect();
    // Reuse the same per-expert sgemm path as the cold helper (every expert "cold").
    cold_grouped_matmul(input, k, n, &cold, weights_all, &mut out);
    out
}

/// Execute a grouped matmul the way the hot/cold split does, for parity testing and
/// as the CPU backend's split path: hot tokens run over the device `slot_weights`
/// (`[num_slots, k, n]`) with a per-token slot remap; cold tokens run over
/// `host_weights` (`[num_experts, k, n]`).
///
/// - `slot_idx`: per-token device slot, or `cold_sentinel` for cold tokens.
/// - `cold`: `(token, global_expert)` for cold tokens.
pub fn grouped_matmul_split_reference(
    input: &[f32],
    slot_weights: &[f32],
    host_weights: &[f32],
    slot_idx: &[u32],
    cold: &[(usize, usize)],
    m: usize,
    k: usize,
    n: usize,
    cold_sentinel: u32,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    let stride = k * n;

    // Hot tokens: group by slot, one sgemm per slot over the device slot buffer.
    let hot: Vec<(usize, usize)> = slot_idx
        .iter()
        .take(m)
        .enumerate()
        .filter(|&(_, &s)| s != cold_sentinel)
        .map(|(t, &s)| (t, s as usize))
        .collect();
    for (slot, tokens) in group_by_key(&hot) {
        let w = &slot_weights[slot * stride..(slot + 1) * stride];
        let cnt = tokens.len();
        let mut packed_in = vec![0f32; cnt * k];
        for (r, &t) in tokens.iter().enumerate() {
            packed_in[r * k..(r + 1) * k].copy_from_slice(&input[t * k..(t + 1) * k]);
        }
        let mut packed_out = vec![0f32; cnt * n];
        crate::blas::sgemm(&packed_in, w, &mut packed_out, cnt, k, n);
        for (r, &t) in tokens.iter().enumerate() {
            out[t * n..(t + 1) * n].copy_from_slice(&packed_out[r * n..(r + 1) * n]);
        }
    }

    // Cold tokens: host store.
    cold_grouped_matmul(input, k, n, cold, host_weights, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small deterministic LCG so the test needs no rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    /// Build the slot buffer + per-token route for a chosen resident set, then
    /// assert the split output is byte-identical to the full grouped matmul.
    fn check_parity(num_experts: usize, resident: &[usize], m: usize, k: usize, n: usize) {
        let mut rng = Lcg(0x1234_5678 ^ (num_experts as u64));
        let input: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let weights_all: Vec<f32> = (0..num_experts * k * n).map(|_| rng.f32()).collect();
        // Round-robin token→expert assignment so every expert gets some tokens.
        let expert_idx: Vec<u32> = (0..m).map(|t| (t % num_experts) as u32).collect();

        let baseline = grouped_matmul_reference(&input, &weights_all, &expert_idx, m, k, n);

        // Slot buffer: copy each resident expert's slab into a slot.
        let stride = k * n;
        let mut expert_to_slot = vec![None; num_experts];
        let mut slot_weights = vec![0f32; resident.len() * stride];
        for (slot, &e) in resident.iter().enumerate() {
            expert_to_slot[e] = Some(slot);
            slot_weights[slot * stride..(slot + 1) * stride]
                .copy_from_slice(&weights_all[e * stride..(e + 1) * stride]);
        }
        // Route split.
        let sentinel = u32::MAX;
        let mut slot_idx = Vec::with_capacity(m);
        let mut cold = Vec::new();
        for (t, &e) in expert_idx.iter().enumerate() {
            match expert_to_slot[e as usize] {
                Some(s) => slot_idx.push(s as u32),
                None => {
                    slot_idx.push(sentinel);
                    cold.push((t, e as usize));
                }
            }
        }

        let split = grouped_matmul_split_reference(
            &input,
            &slot_weights,
            &weights_all,
            &slot_idx,
            &cold,
            m,
            k,
            n,
            sentinel,
        );

        assert_eq!(
            baseline, split,
            "hot/cold split must be byte-identical to full grouped matmul \
             (num_experts={num_experts}, resident={resident:?})"
        );
    }

    #[test]
    fn split_matches_full_all_resident() {
        // Every expert hot → equivalent to full path.
        check_parity(8, &[0, 1, 2, 3, 4, 5, 6, 7], 40, 5, 6);
    }

    #[test]
    fn split_matches_full_all_cold() {
        // No experts resident → everything on the host path.
        check_parity(8, &[], 40, 5, 6);
    }

    #[test]
    fn split_matches_full_partial() {
        // A realistic hot subset with non-identity slot mapping.
        check_parity(8, &[7, 2, 5], 41, 4, 7);
        check_parity(16, &[1, 3, 9, 15, 4], 100, 8, 3);
    }

    #[test]
    fn cold_helper_writes_only_cold_rows() {
        // Sentinel-fill the output; ensure hot rows stay untouched by the cold pass.
        let (m, k, n, ne) = (6usize, 3usize, 2usize, 4usize);
        let mut rng = Lcg(99);
        let input: Vec<f32> = (0..m * k).map(|_| rng.f32()).collect();
        let w: Vec<f32> = (0..ne * k * n).map(|_| rng.f32()).collect();
        let mut out = vec![f32::NAN; m * n];
        // Only tokens 1 and 4 are cold (experts 2 and 3).
        let cold = vec![(1usize, 2usize), (4usize, 3usize)];
        cold_grouped_matmul(&input, k, n, &cold, &w, &mut out);
        for t in 0..m {
            let touched = out[t * n..(t + 1) * n].iter().all(|v| !v.is_nan());
            let is_cold = t == 1 || t == 4;
            assert_eq!(
                touched, is_cold,
                "row {t} touched={touched}, expected cold={is_cold}"
            );
        }
    }
}
