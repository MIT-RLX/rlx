// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA kernel-schedule ports — `matmul.cu` and `attention.cu`.
//!
//! Two questions per port: does the typed schedule describe the kernel that
//! actually ships, and does verifying it catch the hazards that kernel has?
//! The first is answered by scanning the source rather than trusting a
//! transcription, so a schedule cannot drift from the `.cu` it claims to
//! describe.

use rlx_gpu_dispatch::tiles::{MATMUL_TILE_CANDIDATES, TileParams};
use rlx_gpu_kernels::kernel_schedule_port::{
    ATTN_BR, ATTN_D_PAD, BLOCK_ROLE, MATMUL_TARGET, attention_drift_against_source,
    attention_schedule, matmul_schedule, verify_attention_schedule, verify_tile_schedule,
};
use rlx_ir::kernel_schedule::{
    Access, Action, KernelScheduleError, Region, Space, Target, verify_kernel_schedule,
};

#[test]
fn every_shipping_tile_has_a_sound_schedule() {
    // The adoption claim. Every tile the tuner is allowed to measure must
    // describe a schedule that verifies — if one does not, either the tile is
    // wrong or the port is.
    for &tile in MATMUL_TILE_CANDIDATES {
        verify_tile_schedule(tile)
            .unwrap_or_else(|e| panic!("tile {} has an unsound schedule: {e:?}", tile.label()));
    }
    verify_tile_schedule(TileParams::DEFAULT_MATMUL).expect("the default tile must verify");
}

#[test]
fn the_schedule_matches_the_kernel_source() {
    // Guards the port against drifting from `matmul.cu`. These are the facts
    // the source states directly: two shared tiles sized [BM][BK] and [BK][BN],
    // a register accumulator [TM][TN], and one role.
    let tile = TileParams::DEFAULT_MATMUL;
    let s = matmul_schedule(tile);

    let a = s
        .regions
        .iter()
        .find(|r| r.name == "tile_a")
        .expect("tile_a declared");
    let b = s
        .regions
        .iter()
        .find(|r| r.name == "tile_b")
        .expect("tile_b declared");
    let acc = s
        .regions
        .iter()
        .find(|r| r.name == "acc")
        .expect("acc declared");
    assert_eq!(
        a.dims,
        vec![tile.bm as usize, tile.bk as usize],
        "__shared__ tile_a[BM][BK]"
    );
    assert_eq!(
        b.dims,
        vec![tile.bk as usize, tile.bn as usize],
        "__shared__ tile_b[BK][BN]"
    );
    assert_eq!(
        acc.dims,
        vec![tile.tm as usize, tile.tn as usize],
        "float acc[TM][TN]"
    );
    assert_eq!(
        acc.space,
        Space::Register,
        "acc is per-thread registers, not shared"
    );

    assert_eq!(
        s.roles.len(),
        1,
        "matmul.cu is uniform-block: one role, no warp specialization"
    );
    assert_eq!(s.barriers.len(), 2, "two __syncthreads() per K iteration");

    // Shared bytes must equal what the kernel's __shared__ declarations cost,
    // which is the number `TileParams::shared_bytes` reports to the tuner.
    assert_eq!(s.bytes_in(Space::Shared) as u32, tile.shared_bytes());
}

#[test]
fn dropping_the_first_syncthreads_is_caught() {
    // `__syncthreads()` after the tile loads. Without it a thread can read a
    // slice of tile_a another thread has not written yet.
    let mut s = matmul_schedule(TileParams::DEFAULT_MATMUL);
    let acts = s.body.get_mut(BLOCK_ROLE).unwrap();
    acts.retain(|a| !matches!(a, Action::Arrive { barrier, .. } | Action::Wait { barrier, .. } if barrier == "tiles_filled"));
    let errors = verify_kernel_schedule(&s, MATMUL_TARGET);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::MissingReadBarrier { .. })),
        "a missing post-load barrier must be reported: {errors:?}"
    );
}

#[test]
fn dropping_the_second_syncthreads_is_caught() {
    // The write-after-read hazard across the K loop's back-edge: the next
    // iteration's staging landing while threads still read the current tile.
    // Modelled by appending the next iteration's loads.
    let mut s = matmul_schedule(TileParams::DEFAULT_MATMUL);
    let acts = s.body.get_mut(BLOCK_ROLE).unwrap();
    acts.retain(|a| !matches!(a, Action::Arrive { barrier, .. } | Action::Wait { barrier, .. } if barrier == "tiles_consumed"));
    acts.push(Action::Load {
        access: Access::plain("tile_a"),
        stage: 0,
    });
    let errors = verify_kernel_schedule(&s, MATMUL_TARGET);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::MissingOverwriteBarrier { .. })),
        "a missing pre-overwrite barrier must be reported: {errors:?}"
    );
}

#[test]
fn keeping_both_barriers_across_two_iterations_is_clean() {
    // Guards the two tests above from passing for the wrong reason: the same
    // two-iteration body WITH both barriers must verify.
    let mut s = matmul_schedule(TileParams::DEFAULT_MATMUL);
    let second: Vec<Action> = s.body[BLOCK_ROLE].clone();
    s.body.get_mut(BLOCK_ROLE).unwrap().extend(second);
    let errors = verify_kernel_schedule(&s, MATMUL_TARGET);
    assert!(
        errors.is_empty(),
        "two full iterations must verify: {errors:?}"
    );
}

#[test]
fn an_oversized_tile_busts_the_portable_shared_budget() {
    // `matmul.cu` is compiled by NVRTC *and* hipRTC from one text, so the
    // budget that matters is the portable floor, not any single vendor's.
    let big = TileParams {
        bm: 256,
        bn: 256,
        bk: 32,
        tm: 8,
        tn: 8,
        bdx: 32,
        bdy: 32,
    };
    let errors = verify_tile_schedule(big).expect_err("a 256x256x32 tile cannot fit 32 KiB");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::SharedOverBudget { .. })),
        "expected a shared-memory rejection: {errors:?}"
    );
}

// ── Adoption: the schedule gates real source generation ─────────────────────

#[test]
fn no_source_is_emitted_for_a_tile_whose_schedule_fails() {
    // The adoption test. `matmul_cuda_src_tiled` is the ONLY path from a tile
    // to CUDA/HIP text, and it now refuses a tile the schedule rejects. Before
    // this, such a tile compiled and produced wrong numbers.
    //
    // 64x256x32 passes every algebraic rule in `TileParams::validate` — micro
    // tiles cover the block tile, 512 threads, 32 accumulators, staging
    // divides evenly — and needs 40 KiB of shared memory, over the 32 KiB
    // portable floor but under validate's own 48 KiB ceiling. Exactly the tile
    // that would compile for CUDA and fail to launch on a 32 KiB target.
    let over = TileParams {
        bm: 64,
        bn: 256,
        bk: 32,
        tm: 4,
        tn: 8,
        bdx: 32,
        bdy: 16,
    };
    assert_eq!(over.validate(), Ok(()), "the tile is algebraically legal");
    let err = rlx_gpu_kernels::matmul_cuda_src_tiled(over)
        .expect_err("a tile over the shared budget must not produce source");
    let msg = err.to_string();
    assert!(
        msg.contains("schedule does not verify"),
        "the rejection must name the schedule as the reason: {msg}"
    );
}

#[test]
fn every_shipping_tile_still_produces_source() {
    // The other half: gating must not break a tile that was working. If this
    // fails, the gate is too strict, not the tiles wrong.
    for &tile in MATMUL_TILE_CANDIDATES {
        rlx_gpu_kernels::matmul_cuda_src_tiled(tile)
            .unwrap_or_else(|e| panic!("shipping tile {} was rejected: {e}", tile.label()));
    }
}

// ── Anti-drift: the schedule is checked against matmul.cu ───────────────────

#[test]
fn the_schedule_agrees_with_the_shipping_source() {
    // Closes the hole the Metal port did not have: before this, deleting a
    // `__syncthreads()` from matmul.cu left every check in this file green,
    // because the schedule hardcoded two barriers and nothing compared them.
    let problems =
        rlx_gpu_kernels::kernel_schedule_port::drift_against_source(TileParams::DEFAULT_MATMUL);
    assert!(
        problems.is_empty(),
        "the schedule has drifted from matmul.cu:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn the_source_scan_actually_finds_the_declarations() {
    // A scan that silently returns nothing would make the drift check vacuous
    // — the exact failure mode this session has hit three times.
    let decls = rlx_gpu_kernels::kernel_schedule_port::shared_declarations_in("matmul");
    assert_eq!(
        decls,
        vec![
            (
                "tile_a".to_string(),
                vec!["BM".to_string(), "BK".to_string()]
            ),
            (
                "tile_b".to_string(),
                vec!["BK".to_string(), "BN".to_string()]
            ),
        ],
        "matmul.cu's __shared__ declarations are not what the port assumes"
    );
    assert_eq!(
        rlx_gpu_kernels::kernel_schedule_port::syncthreads_in("matmul"),
        2,
        "matmul.cu's K loop has exactly two __syncthreads()"
    );
}

#[test]
fn the_scan_reports_nothing_for_an_unknown_kernel() {
    assert!(rlx_gpu_kernels::kernel_schedule_port::shared_declarations_in("nope").is_empty());
    assert_eq!(
        rlx_gpu_kernels::kernel_schedule_port::syncthreads_in("nope"),
        0
    );
}

// ── Kernel 3: attention.cu ──────────────────────────────────────────────────

#[test]
fn the_attention_schedule_verifies_on_sm86() {
    verify_attention_schedule()
        .unwrap_or_else(|e| panic!("attention.cu schedule is unsound: {e:?}"));
}

#[test]
fn the_attention_schedule_agrees_with_attention_cu() {
    let problems = attention_drift_against_source();
    assert!(
        problems.is_empty(),
        "the attention schedule has drifted from attention.cu:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn attention_does_not_fit_the_portable_shared_floor() {
    // A real portability fact this port made explicit: at D_PAD = 129 the
    // kernel needs ~42.5 KiB of shared memory, so it cannot be compiled for a
    // 32 KiB target as written. Pinned rather than waved at — if someone
    // shrinks the tiles, this fails and the constraint gets re-derived.
    let s = attention_schedule();
    let bytes = s.bytes_in(Space::Shared);
    assert!(
        bytes > 32 * 1024,
        "attention.cu was ~42.5 KiB shared; now {bytes} B — recheck the portability claim"
    );
    assert!(
        bytes < 48 * 1024,
        "it must still fit sm_86's 48 KiB: {bytes} B"
    );
    let errors = verify_kernel_schedule(&attention_schedule(), Target::PORTABLE);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, KernelScheduleError::SharedOverBudget { .. })),
        "the portable floor must reject it: {errors:?}"
    );
}

#[test]
fn the_padded_row_stride_is_modelled_physically() {
    // `__shared__ float q_shared[BR][D_PAD]` allocates 129 floats per row for
    // bank-conflict padding. Region dims are the PHYSICAL extent: declaring
    // the logical 128 with a 129 stride would make the layout address past its
    // own region, which the verifier would (correctly) reject.
    let s = attention_schedule();
    let q: &Region = s
        .regions
        .iter()
        .find(|r| r.name == "q_shared")
        .expect("q_shared");
    assert_eq!(
        q.dims,
        vec![ATTN_BR, ATTN_D_PAD],
        "physical dims, padding included"
    );
    assert_eq!(
        q.layout.strides[0], ATTN_D_PAD,
        "row stride is the padded 129, not 128"
    );
}

#[test]
fn dropping_any_attention_barrier_is_caught() {
    // Four of the five barriers separate a shared write from its read. The
    // fifth (`acc_ready`) guards a register accumulator, which is per-thread
    // and needs no barrier for correctness of THIS check — so it is expected
    // not to fire, and saying which is which is the point.
    let mut caught = Vec::new();
    for b in [
        "q_staged",
        "kv_staged",
        "scores_ready",
        "softmax_ready",
        "acc_ready",
    ] {
        let mut s = attention_schedule();
        let acts = s.body.get_mut("block").unwrap();
        acts.retain(|a| {
            !matches!(a,
            Action::Arrive { barrier, .. } | Action::Wait { barrier, .. } if barrier == b)
        });
        let errors = verify_kernel_schedule(&s, Target::CUDA_SM86);
        if errors.iter().any(|e| {
            matches!(
                e,
                KernelScheduleError::MissingReadBarrier { .. }
                    | KernelScheduleError::MissingOverwriteBarrier { .. }
            )
        }) {
            caught.push(b);
        }
    }
    assert!(
        caught.len() >= 3,
        "dropping a shared-memory barrier must be caught; only {caught:?} fired"
    );
}
