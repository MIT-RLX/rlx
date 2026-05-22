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

//! Tests for the HIP-CPU validation path. Only compiled with
//! `cargo test -p rlx-cuda --features hip-cpu-validate`.
//!
//! These tests numerically validate that `.cu` kernels produce correct
//! output when executed on CPU via HIP-CPU. They run anywhere C++17
//! compiles — Mac, Linux, Windows, Docker — without an NVIDIA driver.
//!
//! Same kernel sources as the GPU path; only the dispatch route changes.
//! A passing test here means the kernel logic is correct under HIP-CPU's
//! interpretation; it does *not* prove the kernel will produce the same
//! bits under NVCC + real CUDA. That requires a real-GPU run via the
//! main bench (`bench_6way_parity`).
//!
//! Coverage: at least one test per kernel family across all 32 launch
//! entry points. Tests are intentionally tiny — the goal is to exercise
//! the FFI wiring + kernel logic on representative shapes, not to
//! benchmark.

#![cfg(feature = "hip-cpu-validate")]

use rlx_cuda::cpu_dispatch::*;

fn approx(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "idx {i}: got {a}, want {e} (tol {tol})"
        );
    }
}

// ── element-wise ────────────────────────────────────────────────────

#[test]
fn binary_add_matches_reference() {
    let mut arena = vec![
        1.0, 2.0, 3.0, 4.0, // a
        10.0, 20.0, 30.0, 40.0, // b
        0.0, 0.0, 0.0, 0.0, // c
    ];
    run_binary(&mut arena, 4, 0, 4, 8, /*Add*/ 0);
    assert_eq!(&arena[8..12], &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn binary_mul_matches_reference() {
    let mut arena = vec![
        2.0, 3.0, 4.0, 5.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0,
    ];
    run_binary(&mut arena, 4, 0, 4, 8, /*Mul*/ 2);
    assert_eq!(&arena[8..12], &[20.0, 30.0, 40.0, 50.0]);
}

#[test]
fn binary_sub_matches_reference() {
    let mut arena = vec![
        10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
    ];
    run_binary(&mut arena, 4, 0, 4, 8, /*Sub*/ 1);
    assert_eq!(&arena[8..12], &[9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn unary_relu_clamps_negatives() {
    let mut arena = vec![-2.0, -0.5, 0.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    run_unary(&mut arena, 5, 0, 5, /*relu*/ 0);
    assert_eq!(&arena[5..10], &[0.0, 0.0, 0.0, 1.5, 3.0]);
}

#[test]
fn unary_silu_matches_reference() {
    let mut arena = vec![0.0, 1.0, 0.0, 0.0];
    run_unary(&mut arena, 2, 0, 2, /*silu*/ 10);
    // silu(0) = 0; silu(1) = 1 / (1 + e^-1) ≈ 0.731059
    approx(&arena[2..4], &[0.0, 0.7310585], 1e-5);
}

#[test]
fn copy_propagates_input() {
    let mut arena = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    run_copy(&mut arena, 4, 0, 4);
    assert_eq!(&arena[4..8], &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn compare_lt_matches_reference() {
    let mut arena = vec![1.0, 5.0, 3.0, 7.0, 2.0, 4.0, 3.0, 8.0, 0.0, 0.0, 0.0, 0.0];
    run_compare(&mut arena, 4, 0, 4, 8, /*Lt*/ 2);
    assert_eq!(&arena[8..12], &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn where_select_picks_by_cond() {
    let mut arena = vec![
        1.0, 0.0, 1.0, 0.0, // cond
        100.0, 200.0, 300.0, 400.0, // x
        10.0, 20.0, 30.0, 40.0, // y
        0.0, 0.0, 0.0, 0.0, // out
    ];
    run_where_select(&mut arena, 4, 0, 4, 8, 12);
    assert_eq!(&arena[12..16], &[100.0, 20.0, 300.0, 40.0]);
}

// ── matmul family ───────────────────────────────────────────────────

#[test]
fn matmul_2x3x2_matches_reference() {
    // A[2,3] · B[3,2]
    let mut arena = vec![
        // A
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // B
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // C (4 elements)
        0.0, 0.0, 0.0, 0.0,
    ];
    run_matmul(
        &mut arena, 2, 3, 2, /*ao*/ 0, /*bo*/ 6, /*co*/ 12, /*batch*/ 1,
    );
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    approx(&arena[12..16], &[58.0, 64.0, 139.0, 154.0], 1e-4);
}

#[test]
fn grouped_matmul_two_experts() {
    // M=2, K=2, N=2, num_experts=2.
    // input[2,2] · expert_weight[2,2,2] picked by idx[2].
    let mut arena = vec![
        // input [2,2]
        1.0, 2.0, 3.0, 4.0, // weights [2,2,2] (expert 0, then expert 1)
        1.0, 0.0, 0.0, 1.0, // expert 0 = identity
        2.0, 0.0, 0.0, 2.0, // expert 1 = 2*identity
        // idx [2]
        0.0, 1.0, // output [2,2]
        0.0, 0.0, 0.0, 0.0,
    ];
    run_grouped_matmul(
        &mut arena, 2, 2, 2, 2, /*io*/ 0, /*wo*/ 4, /*idx_o*/ 12, /*oo*/ 14,
    );
    // row 0 → expert 0 → [1, 2]; row 1 → expert 1 → [6, 8]
    approx(&arena[14..18], &[1.0, 2.0, 6.0, 8.0], 1e-4);
}

#[test]
fn dequant_matmul_int8_block_matches() {
    // Tiny scheme=0 (Int8Block) test. m=1, k=4, n=1, block_size=4.
    // x = [1,1,1,1]; w (i8) = [2,3,4,5] packed in one f32 word; scale = 1.
    // Expected: 1*2 + 1*3 + 1*4 + 1*5 = 14.
    let mut arena = vec![
        // x [1,4]
        1.0,
        1.0,
        1.0,
        1.0,
        // w packed (4 bytes 2,3,4,5 → little-endian)
        f32::from_bits(0x05_04_03_02),
        // scale [1,1] (one block, one column)
        1.0,
        // zp (unused for scheme 0)
        0.0,
        // out [1,1]
        0.0,
    ];
    run_dequant_matmul(
        &mut arena, 1, 4, 1, 4, /*Int8Block*/ 0, /*xo*/ 0, /*wo*/ 4, /*sco*/ 5,
        /*zo*/ 6, /*oo*/ 7,
    );
    approx(&arena[7..8], &[14.0], 1e-4);
}

// ── reductions ──────────────────────────────────────────────────────

#[test]
fn reduce_sum_along_inner() {
    let mut arena = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0];
    run_reduce(
        &mut arena, /*outer*/ 2, /*inner*/ 3, 0, 6, /*Sum*/ 0,
    );
    approx(&arena[6..8], &[6.0, 15.0], 1e-5);
}

#[test]
fn reduce_max_along_inner() {
    let mut arena = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 0.0, 0.0];
    run_reduce(&mut arena, 2, 3, 0, 6, /*Max*/ 2);
    approx(&arena[6..8], &[5.0, 9.0], 1e-5);
}

#[test]
fn softmax_normalizes_to_unit_sum() {
    let mut arena = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
    run_softmax(&mut arena, 1, 3, 0, 3);
    let row = &arena[3..6];
    let sum: f32 = row.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    approx(row, &[0.09003057, 0.24472847, 0.66524096], 1e-5);
}

#[test]
fn layernorm_zero_mean_unit_var() {
    // op=0 (LayerNorm). gamma=1, beta=0.
    let mut arena = vec![
        // x
        1.0, 2.0, 3.0, 4.0, // gamma
        1.0, 1.0, 1.0, 1.0, // beta
        0.0, 0.0, 0.0, 0.0, // out
        0.0, 0.0, 0.0, 0.0,
    ];
    run_layernorm(
        &mut arena, /*outer*/ 1, /*inner*/ 4, /*io*/ 0, /*oo*/ 12,
        /*go*/ 4, /*beta_o*/ 8, 1e-5, /*op*/ 0,
    );
    let out = &arena[12..16];
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4, "mean was {mean}");
    let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
    assert!((var - 1.0).abs() < 1e-3, "var was {var}");
}

#[test]
fn rmsnorm_scales_by_inv_rms() {
    // op=1 (RmsNorm). gamma=1.
    let mut arena = vec![
        2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // beta unused
        0.0, 0.0, 0.0, 0.0,
    ];
    run_layernorm(&mut arena, 1, 4, 0, 12, 4, 8, 1e-5, /*RmsNorm*/ 1);
    // rms = sqrt(4/4) = 1; out = x / 1 = x.
    approx(&arena[12..16], &[2.0, 0.0, 0.0, 0.0], 1e-3);
}

#[test]
fn fused_residual_ln_matches_reference() {
    // x + r → LN with gamma=1 beta=0 (no bias).
    let mut arena = vec![
        // x
        1.0, 2.0, 3.0, 4.0, // residual
        0.0, 0.0, 0.0, 0.0, // bias unused
        0.0, 0.0, 0.0, 0.0, // gamma
        1.0, 1.0, 1.0, 1.0, // beta
        0.0, 0.0, 0.0, 0.0, // out
        0.0, 0.0, 0.0, 0.0,
    ];
    run_fused_residual_ln(
        &mut arena, 1, 4, /*io*/ 0, /*ro*/ 4, /*bias_o*/ 8, /*go*/ 12,
        /*beta_o*/ 16, /*oo*/ 20, 1e-5, /*has_bias*/ 0,
    );
    let out = &arena[20..24];
    let mean: f32 = out.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-4, "mean was {mean}");
}

#[test]
fn cumsum_inclusive_matches() {
    let mut arena = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    run_cumsum(&mut arena, 1, 4, 0, 4, /*exclusive*/ 0);
    approx(&arena[4..8], &[1.0, 3.0, 6.0, 10.0], 1e-5);
}

#[test]
fn argmax_picks_largest_index() {
    let mut arena = vec![0.1, 0.5, 0.2, 0.9, 0.3, 0.0];
    run_argmax(&mut arena, 1, 5, 0, 5);
    assert_eq!(arena[5] as u32, 3);
}

#[test]
fn topk_picks_two_largest() {
    let mut arena = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.0, 0.0];
    run_topk(&mut arena, 1, 5, /*k*/ 2, 0, 5);
    // Top 2: indices 1 (0.9), then 3 (0.7).
    assert_eq!(arena[5] as u32, 1);
    assert_eq!(arena[6] as u32, 3);
}

// ── shape ops ───────────────────────────────────────────────────────

#[test]
fn gather_pulls_rows_by_index() {
    // input [vocab=3, dim=2], idx [n_idx=2] → out [n_idx=2, dim=2]
    let mut arena = vec![
        // input
        10.0, 11.0, 20.0, 21.0, 30.0, 31.0, // idx
        2.0, 0.0, // out
        0.0, 0.0, 0.0, 0.0,
    ];
    run_gather(
        &mut arena, /*n_out*/ 4, /*n_idx*/ 2, /*dim*/ 2, /*vocab*/ 3,
        /*io*/ 0, /*idx_o*/ 6, /*oo*/ 8,
    );
    approx(&arena[8..12], &[30.0, 31.0, 10.0, 11.0], 1e-5);
}

#[test]
fn narrow_extracts_axis_slice() {
    // Input shape [outer=1, axis_in=4, inner=1] → take 2 starting at 1.
    let mut arena = vec![10.0, 20.0, 30.0, 40.0, 0.0, 0.0];
    run_narrow(
        &mut arena, /*total*/ 2, /*outer*/ 1, /*inner*/ 1, /*axis_in*/ 4,
        /*axis_out*/ 2, /*start*/ 1, /*io*/ 0, /*oo*/ 4,
    );
    approx(&arena[4..6], &[20.0, 30.0], 1e-5);
}

#[test]
fn concat_writes_into_output_slot() {
    // Concat two [1,2,1] inputs along axis -> output [1,4,1].
    // Test the second-input dispatch (start=2).
    let mut arena = vec![
        // input piece (axis_in=2)
        7.0, 8.0, // output [1,4,1] (pre-zeroed)
        0.0, 0.0, 0.0, 0.0,
    ];
    run_concat(
        &mut arena, /*total*/ 2, /*outer*/ 1, /*inner*/ 1, /*axis_in*/ 2,
        /*axis_out*/ 4, /*start*/ 2, /*io*/ 0, /*oo*/ 2,
    );
    approx(&arena[2..6], &[0.0, 0.0, 7.0, 8.0], 1e-5);
}

#[test]
fn transpose_2d_swaps_axes() {
    // 2×3 → 3×2 transpose. perm = [1, 0].
    // meta layout (rank=2): [out_dims[0..2], in_strides_for_out[0..2]]
    // out_dims = [3, 2]; in_strides = [3, 1] for input of shape [2, 3];
    // permuted: in_strides_for_out[0] = in_stride[1] = 1
    //           in_strides_for_out[1] = in_stride[0] = 3
    let mut arena = vec![
        // input [2,3]
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // output [3,2]
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let meta = vec![3u32, 2, 1, 3];
    run_transpose(
        &mut arena, /*rank*/ 2, /*out_total*/ 6, /*io*/ 0, /*oo*/ 6, &meta,
    );
    approx(&arena[6..12], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
}

#[test]
fn expand_broadcasts_axis() {
    // Input [1,3] → output [2,3]. Stride for broadcasted leading axis = 0.
    let mut arena = vec![
        // input [1,3]
        7.0, 8.0, 9.0, // output [2,3]
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let meta = vec![2u32, 3, 0, 1];
    run_expand(
        &mut arena, /*rank*/ 2, /*out_total*/ 6, /*io*/ 0, /*oo*/ 3, &meta,
    );
    approx(&arena[3..9], &[7.0, 8.0, 9.0, 7.0, 8.0, 9.0], 1e-5);
}

// ── attention / rope ────────────────────────────────────────────────

#[test]
fn attention_no_mask_check() {
    // batch=1, heads=1, seq_q=2, seq_k=2, head_dim=2.
    // Q = K = V = identity-ish; expect output equal to weighted V.
    let mut arena = vec![
        // Q [1,1,2,2]
        1.0, 0.0, 0.0, 1.0, // K [1,1,2,2]
        1.0, 0.0, 0.0, 1.0, // V [1,1,2,2]
        10.0, 20.0, 30.0, 40.0, // out [1,1,2,2]
        0.0, 0.0, 0.0, 0.0,
    ];
    run_attention(
        &mut arena, 1, 1, 2, 2, 2, /*qo*/ 0, /*ko*/ 4, /*vo*/ 8, /*oo*/ 12,
        /*mask_o*/ 0, /*mask_kind None*/ 0, /*scale*/ 1.0, /*window*/ 0,
    );
    // check: outputs are finite and differ from initial 0.
    for v in &arena[12..16] {
        assert!(v.is_finite() && *v != 0.0, "got {v}");
    }
}

#[test]
fn rope_rotates_in_pairs() {
    // n_total=2, seq=1, head_dim=2, half=1, last_dim=2.
    // cos=[1.0], sin=[0.0] → rope is identity.
    let mut arena = vec![
        // input [1,1,2]
        3.0, 4.0, // cos [1,1]
        1.0, // sin [1,1]
        0.0, // output
        0.0, 0.0,
    ];
    run_rope(
        &mut arena, /*n_total*/ 2, /*seq*/ 1, /*head_dim*/ 2, /*half*/ 1,
        /*io*/ 0, /*co*/ 2, /*so*/ 3, /*oo*/ 4, /*last_dim*/ 2,
    );
    approx(&arena[4..6], &[3.0, 4.0], 1e-5);
}

// ── scatter / sample / scan ─────────────────────────────────────────

#[test]
fn scatter_add_accumulates() {
    // out[3, trailing=1] ← upd[2, 1] @ idx[2]={0, 0}
    let mut arena = vec![
        // upd
        5.0, 7.0, // idx
        0.0, 0.0, // out [3] (will be zeroed by phase 0)
        9.0, 9.0, 9.0,
    ];
    run_scatter_add(
        &mut arena, /*oo*/ 4, /*total*/ 3, /*upd_o*/ 0, /*idx_o*/ 2,
        /*n_upd*/ 2, /*trailing*/ 1, /*out_dim*/ 3,
    );
    approx(&arena[4..7], &[12.0, 0.0, 0.0], 1e-5);
}

#[test]
fn sample_greedy_with_temperature_one() {
    // Greedy = top_k=1, top_p=1.0, temp=1.0. Should always pick argmax.
    let mut arena = vec![0.1, 0.2, 0.9, 0.3, 0.0];
    run_sample(
        &mut arena, /*outer*/ 1, /*inner*/ 4, /*io*/ 0, /*oo*/ 4,
        /*top_k*/ 1, /*top_p*/ 1.0, /*temp*/ 1.0, /*seed*/ 42,
    );
    assert_eq!(arena[4] as u32, 2);
}

#[test]
fn selective_scan_runs_clean() {
    // batch=1, seq=1, hidden=1, state_size=1.
    // x=1, dt=1, A=0 → da = e^0 = 1; B=1 → state = 1*1*1 = 1; C=1 → out = 1.
    let mut arena = vec![
        // x [1,1,1]
        1.0, // dt [1,1,1]
        1.0, // A [hidden, state_size] = [1,1]
        0.0, // B [batch, seq, state_size] = [1,1,1]
        1.0, // C [batch, seq, state_size] = [1,1,1]
        1.0, // out [1,1,1]
        0.0,
    ];
    run_selective_scan(
        &mut arena, 1, 1, 1, 1, /*xo*/ 0, /*dt_o*/ 1, /*ao*/ 2, /*bo*/ 3,
        /*co*/ 4, /*oo*/ 5,
    );
    approx(&arena[5..6], &[1.0], 1e-4);
}

// ── pool / conv ─────────────────────────────────────────────────────

#[test]
fn pool1d_max_picks_per_window() {
    // n=1 c=1 l=4 → kernel 2 stride 2 → l_out=2.
    let mut arena = vec![1.0, 5.0, 3.0, 2.0, 0.0, 0.0];
    run_pool1d(
        &mut arena, 1, 1, 4, 2, /*kl*/ 2, /*sl*/ 2, /*pl*/ 0, /*Max*/ 0,
        /*io*/ 0, /*oo*/ 4,
    );
    approx(&arena[4..6], &[5.0, 3.0], 1e-5);
}

#[test]
fn pool2d_max_picks_per_window() {
    // 2x2 input, kernel 2x2 stride 1 → 1x1 output.
    let mut arena = vec![1.0, 2.0, 3.0, 4.0, 0.0];
    run_pool2d(
        &mut arena, 1, 1, 2, 2, 1, 1, /*kh*/ 2, /*kw*/ 2, /*sh*/ 1, /*sw*/ 1,
        /*ph*/ 0, /*pw*/ 0, /*Max*/ 0, /*io*/ 0, /*oo*/ 4,
    );
    approx(&arena[4..5], &[4.0], 1e-5);
}

#[test]
fn pool3d_max_picks_per_window() {
    // n=1 c=1 d=2 h=1 w=1 → kernel 2x1x1 → out 1x1x1.
    let mut arena = vec![2.0, 7.0, 0.0];
    run_pool3d(
        &mut arena, 1, 1, 2, 1, 1, 1, 1, 1, /*kd*/ 2, /*kh*/ 1, /*kw*/ 1,
        /*sd*/ 1, /*sh*/ 1, /*sw*/ 1, /*pd*/ 0, /*ph*/ 0, /*pw*/ 0,
        /*Max*/ 0, /*io*/ 0, /*oo*/ 2,
    );
    approx(&arena[2..3], &[7.0], 1e-5);
}

#[test]
fn conv1d_identity_kernel_passes_through() {
    // n=1, c_in=1, c_out=1, l=3, kernel=1 (identity weight).
    let mut arena = vec![
        // input [1,1,3]
        1.0, 2.0, 3.0, // weight [1,1,1]
        1.0, // output [1,1,3]
        0.0, 0.0, 0.0,
    ];
    run_conv1d(
        &mut arena, 1, 1, 1, 3, 3, /*kl*/ 1, /*sl*/ 1, /*pl*/ 0, /*dl*/ 1,
        /*groups*/ 1, /*io*/ 0, /*wo*/ 3, /*oo*/ 4,
    );
    approx(&arena[4..7], &[1.0, 2.0, 3.0], 1e-5);
}

#[test]
fn conv2d_2x2_kernel_sum() {
    // n=1, c=1, h=2, w=2; kernel 2x2 of all ones; valid → 1x1.
    let mut arena = vec![
        // input
        1.0, 2.0, 3.0, 4.0, // weight [1,1,2,2]
        1.0, 1.0, 1.0, 1.0, // output [1,1,1,1]
        0.0,
    ];
    run_conv2d(
        &mut arena, 1, 1, 1, 2, 2, 1, 1, /*kh*/ 2, /*kw*/ 2, /*sh*/ 1,
        /*sw*/ 1, /*ph*/ 0, /*pw*/ 0, /*dh*/ 1, /*dw*/ 1,
        /*groups*/ 1, /*io*/ 0, /*wo*/ 4, /*oo*/ 8,
    );
    approx(&arena[8..9], &[10.0], 1e-5);
}

#[test]
fn fused_binary_unary_add_then_relu() {
    // out[i] = relu(a[i] + b[i])
    let mut arena = vec![
        // a
        -3.0, 2.0, -1.0, 4.0, // b
        1.0, -5.0, 0.5, -10.0, // out
        0.0, 0.0, 0.0, 0.0,
    ];
    run_fused_binary_unary(
        &mut arena, 4, /*a_off*/ 0, /*b_off*/ 4, /*out*/ 8, /*Add*/ 0,
        /*relu*/ 0,
    );
    // a+b = [-2, -3, -0.5, -6]; relu = [0, 0, 0, 0]
    approx(&arena[8..12], &[0.0, 0.0, 0.0, 0.0], 1e-5);
}

#[test]
fn fused_binary_unary_mul_then_silu() {
    // out[i] = silu(a[i] * b[i])
    let mut arena = vec![2.0, 3.0, 0.0, 1.0, 0.0, 0.0];
    run_fused_binary_unary(&mut arena, 2, 0, 2, 4, /*Mul*/ 2, /*silu*/ 10);
    // a*b = [0, 3]; silu(0) = 0; silu(3) = 3 / (1 + e^-3) ≈ 2.857722
    approx(&arena[4..6], &[0.0, 2.857722], 1e-4);
}

#[test]
fn conv3d_identity_check() {
    // n=c_in=c_out=d=h=w=k=1 → identity passes through.
    let mut arena = vec![
        // input
        7.0, // weight
        1.0, // out
        0.0,
    ];
    run_conv3d(
        &mut arena, 1, 1, 1, 1, 1, 1, 1, 1, 1, /*kd*/ 1, /*kh*/ 1, /*kw*/ 1,
        /*sd*/ 1, /*sh*/ 1, /*sw*/ 1, /*pd*/ 0, /*ph*/ 0, /*pw*/ 0,
        /*dd*/ 1, /*dh*/ 1, /*dw*/ 1, /*groups*/ 1, /*io*/ 0,
        /*wo*/ 1, /*oo*/ 2,
    );
    approx(&arena[2..3], &[7.0], 1e-5);
}

#[test]
fn elementwise_region_relu_add_mul_matches_reference() {
    // PLAN L2: chain `relu(x + a) * b` over 6 elements. Three steps,
    // three inputs (x, a, b at offsets 0, 6, 12; output at 18).
    let xs = [-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5];
    let a_ = [0.5f32; 6];
    let b_ = [2.0f32; 6];
    let mut arena: Vec<f32> = Vec::with_capacity(24);
    arena.extend_from_slice(&xs);
    arena.extend_from_slice(&a_);
    arena.extend_from_slice(&b_);
    arena.extend(std::iter::repeat(0.0f32).take(6));

    // meta layout: [input_offs[0..16], chain[0..128]]
    let mut meta: Vec<u32> = Vec::with_capacity(144);
    // input_offs: x@0, a@6, b@12, then padding (16 entries total).
    let mut input_offs = [0u32; 16];
    input_offs[0] = 0;
    input_offs[1] = 6;
    input_offs[2] = 12;
    meta.extend_from_slice(&input_offs);
    // 128-word chain buffer, written as 3 steps × 4 words then zero-fill.
    let mut chain = [0u32; 128];
    // Operand encoding bit 31 = src kind: 0=Input, 1=Step.
    let inp = |i: u32| i & 0x7FFF_FFFFu32;
    let stp = |i: u32| 0x8000_0000u32 | (i & 0x7FFF_FFFFu32);
    // step 0: Binary Add (op_kind=2, op_sub=0) of Input(0) + Input(1)
    chain[0] = 2;
    chain[1] = 0;
    chain[2] = inp(0);
    chain[3] = inp(1);
    // step 1: Activation Relu (op_kind=0, op_sub=3) of Step(0)
    chain[4] = 0;
    chain[5] = 3;
    chain[6] = stp(0);
    chain[7] = 0;
    // step 2: Binary Mul (op_kind=2, op_sub=2) of Step(1) * Input(2)
    chain[8] = 2;
    chain[9] = 2;
    chain[10] = stp(1);
    chain[11] = inp(2);
    meta.extend_from_slice(&chain);

    let input_modulus = [0u32; 16]; // no broadcast inputs in this test
    run_elementwise_region(
        &mut arena,
        /*len=*/ 6,
        /*num_inputs=*/ 3,
        /*num_steps=*/ 3,
        /*dst_off=*/ 18,
        &meta,
        /*scalar_input_mask=*/ 0,
        &input_modulus,
    );

    let want: Vec<f32> = xs.iter().map(|x| (x + 0.5).max(0.0) * 2.0).collect();
    approx(&arena[18..24], &want, 1e-5);
}
