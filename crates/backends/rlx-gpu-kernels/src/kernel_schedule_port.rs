// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`matmul.cu`, described as a typed schedule.**
//!
//! This is the first real kernel in the tree expressed in
//! [`rlx_ir::kernel_schedule`], and it is the point of that module existing: until a
//! shipping kernel is described by a `KernelSchedule`, the verifier only ever sees
//! fixtures written to satisfy it.
//!
//! CAKE Appendix A.5 calls this port-driven expansion — "each port either
//! succeeds, validating the abstraction, or reveals a gap, triggering a
//! proposal to extend the IR or its lowering." This port did both.
//!
//! # What the port revealed
//!
//! `matmul.cu` is a **uniform-block** kernel: every thread stages a slice of
//! `tile_a`/`tile_b` into shared memory, then every thread computes from the
//! whole tile. There is no warp specialization, so there is exactly one role.
//!
//! The schedule verifier's original synchronization check was cross-role — a
//! producer role handing a buffer to a consumer role. On a one-role kernel it
//! is **vacuous**: it never fires, no matter how the barriers are arranged.
//! The hazard that actually exists here is intra-role, and it is what the two
//! `__syncthreads()` per K iteration buy:
//!
//! ```text
//!   load tile_a, tile_b     // write shared
//!   __syncthreads();        // (1) writes visible before the reads
//!   compute acc from tiles  // read shared
//!   __syncthreads();        // (2) reads done before the next overwrite
//! ```
//!
//! Dropping (1) is a read-before-write race; dropping (2) is
//! write-after-read across the loop back-edge. Both produce a kernel that is
//! right most of the time. `MissingReadBarrier` and `MissingOverwriteBarrier`
//! were added to `rlx_ir::kernel_schedule` because of this port.

use rlx_gpu_dispatch::tiles::TileParams;
use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, KernelSchedule, KernelScheduleError, Layout, Region, Role,
    Space, Target, verify_kernel_schedule,
};

/// The single role in `matmul.cu`: the whole thread block.
pub const BLOCK_ROLE: &str = "block";

/// Describe `matmul.cu` under `tile` as a typed schedule.
///
/// The `stages` field is 1: this kernel double-buffers nothing — it reuses one
/// pair of shared tiles across K iterations, which is exactly why it needs the
/// second barrier.
pub fn matmul_schedule(tile: TileParams) -> KernelSchedule {
    let (bm, bn, bk) = (tile.bm as usize, tile.bn as usize, tile.bk as usize);
    let (tm, tn) = (tile.tm as usize, tile.tn as usize);

    let mut s = KernelSchedule::new(format!("matmul_{}", tile.label()));
    s.stages = 1;
    // `mma.sync` is not used — this is a scalar FMA inner product — so the
    // only capability required is plain shared memory, which every target has.
    s.requires = vec![];

    s.regions = vec![
        // `__shared__ float tile_a[BM][BK];`
        Region {
            name: "tile_a".into(),
            space: Space::Shared,
            dims: vec![bm, bk],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[bm, bk]),
        },
        // `__shared__ float tile_b[BK][BN];`
        Region {
            name: "tile_b".into(),
            space: Space::Shared,
            dims: vec![bk, bn],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[bk, bn]),
        },
        // `float acc[TM][TN];` — per-thread registers.
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![tm, tn],
            dtype: DType::F32,
            stages: 1,
            layout: Layout::row_major(&[tm, tn]),
        },
    ];

    // One role. `BLOCK_DIM_X * BLOCK_DIM_Y` threads, i.e. threads/32 warps.
    let warps = tile.threads().div_ceil(32);
    s.roles = vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..warps).collect(),
    }];

    // Two full-block barriers per K iteration. Producers and consumers are the
    // same role because `__syncthreads()` synchronizes a block with itself.
    s.barriers = vec![
        Barrier {
            name: "tiles_filled".into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        },
        Barrier {
            name: "tiles_consumed".into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        },
    ];

    // One K iteration, in program order.
    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            Action::Load {
                access: Access::plain("tile_a"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("tile_b"),
                stage: 0,
            },
            Action::Arrive {
                barrier: "tiles_filled".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "tiles_filled".into(),
                stage: 0,
            },
            Action::Compute {
                reads: vec![Access::plain("tile_a"), Access::plain("tile_b")],
                writes: vec![Access::plain("acc")],
                stage: 0,
                via: None,
            },
            Action::Arrive {
                barrier: "tiles_consumed".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "tiles_consumed".into(),
                stage: 0,
            },
            Action::Store {
                access: Access::plain("acc"),
                stage: 0,
            },
        ],
    );
    s
}

/// The target a tile is checked against.
///
/// `Target::PORTABLE` deliberately: `matmul.cu` is compiled by both NVRTC and
/// hipRTC from the same text, so a tile that only fits one vendor's shared
/// budget is a tile that silently stops being portable.
pub const MATMUL_TARGET: Target = Target::PORTABLE;

/// Verify a tile's schedule. `Ok(())` when it is sound.
pub fn verify_tile_schedule(tile: TileParams) -> Result<(), Vec<KernelScheduleError>> {
    let errors = verify_kernel_schedule(&matmul_schedule(tile), MATMUL_TARGET);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Features `matmul.cu` needs. Empty — kept as a function so a future
/// tensor-core variant declares its requirement here rather than in prose.
pub fn matmul_features() -> Vec<Feature> {
    vec![]
}

// ── Anti-drift: read the structure out of the shipping source ───────────────
//
// Without this the port is a description that merely happens to be right
// today. The Metal port reads its `simdgroup_load` strides out of the MSL;
// this is the CUDA equivalent, and it closes the same hole: before it,
// deleting a `__syncthreads()` from `matmul.cu` left every check green.

/// The body of `kernel_name` inside `src`.
fn kernel_body_in(src: &'static str, kernel_name: &str) -> Option<&'static str> {
    // Match on `<space><name>(` rather than `void <name>(`: a kernel may carry
    // an attribute between the two, e.g.
    // `extern "C" __global__ void __launch_bounds__(THREADS) attention(`.
    // Assuming the name follows `void` made the scan silently find nothing for
    // attention.cu — which the empty-result guard reported as a failure rather
    // than a pass, which is the only reason it was noticed.
    let start = src.find(&format!(" {kernel_name}("))?;
    let rest = &src[start + 1..];
    let end = rest
        .find("\nextern \"C\" __global__")
        .map_or(src.len(), |o| start + 1 + o);
    Some(&src[start..end])
}

/// `__shared__` declarations in `kernel_name`, as `(name, [dim exprs])`.
///
/// Dimensions come back as the source's own text (`"BM"`, `"BK"`) because they
/// are macros, not literals — which is the point: the check is that the
/// schedule's regions correspond to the declarations the kernel really makes,
/// including which tile parameter sizes which axis.
pub fn shared_declarations_in(kernel_name: &str) -> Vec<(String, Vec<String>)> {
    shared_declarations_of(crate::MATMUL_CU, kernel_name)
}

/// As [`shared_declarations_in`], over an explicit source.
pub fn shared_declarations_of(src: &'static str, kernel_name: &str) -> Vec<(String, Vec<String>)> {
    let Some(body) = kernel_body_in(src, kernel_name) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        let Some(rest) = t.strip_prefix("__shared__ ") else {
            continue;
        };
        // `float tile_a[BM][BK];`
        let Some(name_start) = rest.find(' ') else {
            continue;
        };
        let after_type = rest[name_start + 1..].trim();
        let Some(br) = after_type.find('[') else {
            continue;
        };
        let name = after_type[..br].trim().to_string();
        let dims: Vec<String> = after_type[br..]
            .trim_end_matches(';')
            .split(']')
            .filter_map(|seg| seg.strip_prefix('['))
            .map(|d| d.trim().to_string())
            .collect();
        if !name.is_empty() && !dims.is_empty() {
            out.push((name, dims));
        }
    }
    out
}

/// How many `__syncthreads()` calls `kernel_name` makes.
pub fn syncthreads_in(kernel_name: &str) -> usize {
    syncthreads_of(crate::MATMUL_CU, kernel_name)
}

/// As [`syncthreads_in`], over an explicit source.
pub fn syncthreads_of(src: &'static str, kernel_name: &str) -> usize {
    kernel_body_in(src, kernel_name).map_or(0, |b| b.matches("__syncthreads()").count())
}

/// Check a schedule against the source it claims to describe.
///
/// One implementation for every port: the two earlier copies differed only in
/// which source and schedule they named, and drifted in their messages while
/// doing so.
///
/// `dim_expectations` pins which macro sizes which axis, so swapping
/// `tile_a[BM][BK]` to `[BK][BM]` is caught. Regions absent from it are
/// checked for existence only — appropriate where the dimensions are literals
/// the schedule already mirrors.
pub fn drift_between(
    src: &'static str,
    source_name: &str,
    kernel_name: &str,
    sched: &KernelSchedule,
    dim_expectations: &[(&str, &[&str])],
) -> Vec<String> {
    let mut problems = Vec::new();

    let decls = shared_declarations_of(src, kernel_name);
    if decls.is_empty() {
        // A silent empty scan is how a drift check becomes decoration. This
        // has already happened once, for a kernel declared
        // `void __launch_bounds__(T) attention(`.
        problems.push(format!(
            "no __shared__ declarations found in {source_name} — the scan is broken or the \
             kernel changed shape; either way the port is unverified"
        ));
        return problems;
    }

    let declared: Vec<&str> = sched
        .regions
        .iter()
        .filter(|r| r.space == Space::Shared)
        .map(|r| r.name.as_str())
        .collect();
    for (name, _) in &decls {
        if !declared.contains(&name.as_str()) {
            problems.push(format!(
                "{source_name} declares __shared__ {name}, the schedule does not"
            ));
        }
    }
    for name in &declared {
        if !decls.iter().any(|(n, _)| n == name) {
            problems.push(format!(
                "the schedule declares shared region {name}, {source_name} does not"
            ));
        }
    }
    for (name, dims) in &decls {
        if let Some((_, want)) = dim_expectations.iter().find(|(n, _)| n == name)
            && dims.as_slice() != *want
        {
            problems.push(format!(
                "{source_name} declares {name}{dims:?} but the schedule is built for {want:?}"
            ));
        }
    }

    let syncs = syncthreads_of(src, kernel_name);
    if syncs != sched.barriers.len() {
        problems.push(format!(
            "{source_name} calls __syncthreads() {syncs} time(s) but the schedule declares {} \
             barrier(s) — a dropped barrier is a race the schedule would not model",
            sched.barriers.len()
        ));
    }
    problems
}

/// [`drift_between`] for `matmul.cu`.
pub fn drift_against_source(tile: TileParams) -> Vec<String> {
    drift_between(
        crate::MATMUL_CU,
        "matmul.cu",
        "matmul",
        &matmul_schedule(tile),
        &[("tile_a", &["BM", "BK"]), ("tile_b", &["BK", "BN"])],
    )
}

/// [`drift_between`] for `attention.cu`.
pub fn attention_drift_against_source() -> Vec<String> {
    drift_between(
        crate::ATTENTION_CU,
        "attention.cu",
        "attention",
        &attention_schedule(),
        &[
            ("q_shared", &["BR", "D_PAD"]),
            ("k_tile", &["BC", "D_PAD"]),
            ("v_tile", &["BC", "D_PAD"]),
            ("scores", &["BR", "BC"]),
        ],
    )
}

// ── Kernel 3: attention.cu ──────────────────────────────────────────────────

/// `attention.cu`'s compile-time parameters, as the source `#define`s them.
pub const ATTN_BR: usize = 16;
pub const ATTN_BC: usize = 32;
pub const ATTN_MAX_HEAD_DIM: usize = 128;
/// `#define D_PAD (MAX_HEAD_DIM + 1)` — the `+1` is bank-conflict padding, so
/// the PHYSICAL row stride is 129 while the logical head dim is at most 128.
pub const ATTN_D_PAD: usize = ATTN_MAX_HEAD_DIM + 1;
pub const ATTN_WARPS_PER_Q: usize = 8;
pub const ATTN_THREADS: usize = ATTN_BR * ATTN_WARPS_PER_Q;

/// `attention.cu` as a typed schedule.
///
/// Structurally unlike the two GEMM ports, which is why it was worth doing:
///
/// * **Padded rows.** `__shared__ float q_shared[BR][D_PAD]` allocates 129
///   floats per row to break bank conflicts. Region dims are therefore the
///   PHYSICAL extent — the alternative (logical dims with a 129 stride) makes
///   the declared layout address past its own region, which is the check
///   working, not a modelling choice worth fighting.
/// * **Loop-carried state.** `row_max`, `row_sum` and `rescale` are online
///   softmax accumulators that persist across KV tiles, unlike the GEMM tiles
///   which are overwritten every iteration.
/// * **Four barriers per iteration**, not two.
pub fn attention_schedule() -> KernelSchedule {
    let mut s = KernelSchedule::new("attention");
    s.stages = 1;
    s.requires = vec![];

    let shared = |name: &str, dims: Vec<usize>| Region {
        name: name.into(),
        space: Space::Shared,
        dims: dims.clone(),
        dtype: rlx_ir::DType::F32,
        stages: 1,
        layout: Layout::row_major(&dims),
    };
    s.regions = vec![
        shared("q_shared", vec![ATTN_BR, ATTN_D_PAD]),
        shared("k_tile", vec![ATTN_BC, ATTN_D_PAD]),
        shared("v_tile", vec![ATTN_BC, ATTN_D_PAD]),
        shared("scores", vec![ATTN_BR, ATTN_BC]),
        shared("row_max", vec![ATTN_BR]),
        shared("row_sum", vec![ATTN_BR]),
        shared("rescale", vec![ATTN_BR]),
        Region {
            name: "acc".into(),
            space: Space::Register,
            dims: vec![ATTN_MAX_HEAD_DIM / ATTN_WARPS_PER_Q],
            dtype: rlx_ir::DType::F32,
            stages: 1,
            layout: Layout::row_major(&[ATTN_MAX_HEAD_DIM / ATTN_WARPS_PER_Q]),
        },
    ];

    s.roles = vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..(ATTN_THREADS / 32) as u32).collect(),
    }];

    // Five `__syncthreads()`: one after the Q prologue, four per KV tile.
    for n in [
        "q_staged",
        "kv_staged",
        "scores_ready",
        "softmax_ready",
        "acc_ready",
    ] {
        s.barriers.push(Barrier {
            name: n.into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        });
    }

    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            // Prologue: stage Q once for the whole block.
            Action::Load {
                access: Access::plain("q_shared"),
                stage: 0,
            },
            Action::Arrive {
                barrier: "q_staged".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "q_staged".into(),
                stage: 0,
            },
            // One KV tile.
            Action::Load {
                access: Access::plain("k_tile"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("v_tile"),
                stage: 0,
            },
            Action::Arrive {
                barrier: "kv_staged".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "kv_staged".into(),
                stage: 0,
            },
            // scores = Q . K^T
            Action::Compute {
                reads: vec![Access::plain("q_shared"), Access::plain("k_tile")],
                writes: vec![Access::plain("scores")],
                stage: 0,
                via: None,
            },
            Action::Arrive {
                barrier: "scores_ready".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "scores_ready".into(),
                stage: 0,
            },
            // Online softmax: update the running max/sum and the rescale factor.
            Action::Compute {
                reads: vec![
                    Access::plain("scores"),
                    Access::plain("row_max"),
                    Access::plain("row_sum"),
                ],
                writes: vec![
                    Access::plain("row_max"),
                    Access::plain("row_sum"),
                    Access::plain("rescale"),
                ],
                stage: 0,
                via: None,
            },
            Action::Arrive {
                barrier: "softmax_ready".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "softmax_ready".into(),
                stage: 0,
            },
            // acc = rescale * acc + P . V
            Action::Compute {
                reads: vec![
                    Access::plain("scores"),
                    Access::plain("v_tile"),
                    Access::plain("rescale"),
                ],
                writes: vec![Access::plain("acc")],
                stage: 0,
                via: None,
            },
            Action::Arrive {
                barrier: "acc_ready".into(),
                stage: 0,
            },
            Action::Wait {
                barrier: "acc_ready".into(),
                stage: 0,
            },
            Action::Store {
                access: Access::plain("acc"),
                stage: 0,
            },
        ],
    );
    s
}

/// Verify `attention.cu`'s schedule.
///
/// Checked against `Target::CUDA_SM86` rather than the portable floor: at
/// `D_PAD = 129` the shared footprint is ~43 KiB, which does not fit the
/// 32 KiB floor. That is a real portability fact about this kernel, not a
/// modelling artifact — see `attention_does_not_fit_the_portable_floor`.
pub fn verify_attention_schedule() -> Result<(), Vec<KernelScheduleError>> {
    let e = verify_kernel_schedule(&attention_schedule(), Target::CUDA_SM86);
    if e.is_empty() { Ok(()) } else { Err(e) }
}
