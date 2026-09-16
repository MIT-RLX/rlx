// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`hgemm_simd_4x4`, described as a typed schedule — with the strides read
//! out of the MSL rather than retyped.**
//!
//! The CUDA port (`rlx_gpu_kernels::kernel_schedule_port`) is a hand
//! transcription: nothing ties it to `matmul.cu`, so deleting a
//! `__syncthreads()` from that file leaves every check passing. This port does
//! not repeat that mistake for the one value most likely to be wrong.
//!
//! Metal's cooperative-matrix load takes its layout as an *argument*:
//!
//! ```text
//!   threadgroup half A_tg[32 * 32];
//!   simdgroup_load(a, &A_tg[sg_row * 8 * 32 + k_inner], 32);
//!                                                       ^^ elements_per_row
//! ```
//!
//! That literal is the row stride. If it disagrees with how the buffer is
//! actually indexed, the kernel **compiles cleanly and reads the wrong
//! elements** — the same failure shape as `rocm-gguf-transposed`, where the
//! shapes agreed and the addressing did not. rlx has 51 `simdgroup_load` sites
//! and nothing checks any of them.
//!
//! [`simdgroup_strides_in`](crate::kernel_schedule_port::simdgroup_strides_in) parses
//! those literals straight out of
//! [`crate::kernels::RLX_KERNELS_MSL`], so the check compares the schedule
//! against the shipping source rather than against a copy of it.

use rlx_ir::DType;
use rlx_ir::kernel_schedule::{
    Access, Action, Barrier, Feature, Instruction, KernelSchedule, KernelScheduleError, Layout,
    Region, Role, Space, Target, verify_kernel_schedule,
};

/// The single role: the whole threadgroup. `hgemm_simd_4x4` has 16 simdgroups
/// (`sg_row = sgid / 4`, `sg_col = sgid % 4`) of 32 threads each.
pub const BLOCK_ROLE: &str = "threadgroup";

/// Apple GPU limits. `simdgroup_half8x8` fixes the cooperative tile at 8x8x8.
pub const METAL_TARGET: Target = Target::METAL_APPLE;

/// Tile edge of `simdgroup_half8x8`.
pub const SIMDGROUP_TILE: u32 = 8;

/// The threadgroup tile `hgemm_simd_4x4` stages: `threadgroup half A_tg[32*32]`.
pub const TG_TILE: usize = 32;

/// `hgemm_simd_4x4` as a typed schedule.
pub fn hgemm_simd_4x4_schedule() -> KernelSchedule {
    let mut s = KernelSchedule::new("hgemm_simd_4x4");
    s.stages = 1;
    s.requires = vec![Feature::CoopMatrix {
        m: SIMDGROUP_TILE,
        n: SIMDGROUP_TILE,
        k: SIMDGROUP_TILE,
    }];

    // `threadgroup half A_tg[32 * 32]` / `B_tg[32 * 32]`, both indexed
    // `[row * 32 + col]` — hence a row-major layout with row stride 32.
    for name in ["A_tg", "B_tg"] {
        s.regions.push(Region {
            name: name.into(),
            space: Space::Shared,
            dims: vec![TG_TILE, TG_TILE],
            dtype: DType::F16,
            stages: 1,
            layout: Layout::row_major(&[TG_TILE, TG_TILE]),
        });
    }
    // `simdgroup_half8x8 c` — the per-simdgroup accumulator.
    s.regions.push(Region {
        name: "acc".into(),
        space: Space::Register,
        dims: vec![SIMDGROUP_TILE as usize, SIMDGROUP_TILE as usize],
        dtype: DType::F16,
        stages: 1,
        layout: Layout::row_major(&[SIMDGROUP_TILE as usize, SIMDGROUP_TILE as usize]),
    });

    // 16 simdgroups x 32 threads = 512 threads = 16 warps.
    s.roles = vec![Role {
        name: BLOCK_ROLE.into(),
        warps: (0..16).collect(),
    }];

    // Two `threadgroup_barrier(mem_flags::mem_threadgroup)` per K step, the
    // same shape as the CUDA kernel's two `__syncthreads()`.
    for name in ["tiles_filled", "tiles_consumed"] {
        s.barriers.push(Barrier {
            name: name.into(),
            producers: vec![BLOCK_ROLE.into()],
            consumers: vec![BLOCK_ROLE.into()],
            count: 1,
        });
    }

    s.body.insert(
        BLOCK_ROLE.into(),
        vec![
            Action::Load {
                access: Access::plain("A_tg"),
                stage: 0,
            },
            Action::Load {
                access: Access::plain("B_tg"),
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
            // simdgroup_load x2 + simdgroup_multiply_accumulate.
            Action::Compute {
                reads: vec![Access::plain("A_tg"), Access::plain("B_tg")],
                writes: vec![Access::plain("acc")],
                stage: 0,
                via: Some(Instruction::CoopMatrix {
                    m: SIMDGROUP_TILE,
                    n: SIMDGROUP_TILE,
                    k: SIMDGROUP_TILE,
                }),
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

/// Verify the schedule against Apple GPU limits.
pub fn verify() -> Result<(), Vec<KernelScheduleError>> {
    let e = verify_kernel_schedule(&hgemm_simd_4x4_schedule(), METAL_TARGET);
    if e.is_empty() { Ok(()) } else { Err(e) }
}

/// Every `elements_per_row` literal passed to `simdgroup_load`/`simdgroup_store`
/// inside `kernel void <name>`, read from the shipping MSL.
///
/// Deliberately a text scan and not a parser: the goal is to notice when the
/// source and the schedule disagree, and a scan that occasionally declines to
/// find a call is a weaker check, not a wrong one. It returns what it found so
/// a caller can require a minimum count rather than trust an empty result.
pub fn simdgroup_strides_in(kernel_name: &str) -> Vec<usize> {
    let src = crate::kernels::RLX_KERNELS_MSL;
    let Some(start) = src.find(&format!("kernel void {kernel_name}(")) else {
        return Vec::new();
    };
    // End at the next `kernel void`, or the end of the source.
    let rest = &src[start + 1..];
    let end = rest
        .find("\nkernel void ")
        .map_or(src.len(), |o| start + 1 + o);
    let body = &src[start..end];

    let mut out = Vec::new();
    for call in ["simdgroup_load(", "simdgroup_store("] {
        let mut at = 0usize;
        while let Some(i) = body[at..].find(call) {
            let open = at + i + call.len();
            // Walk to the matching close paren, tracking nesting so an index
            // expression like `&A_tg[f(x)]` does not end the argument list.
            let (mut depth, mut j) = (1i32, open);
            let bytes = body.as_bytes();
            while j < body.len() && depth > 0 {
                match bytes[j] {
                    b'(' | b'[' => depth += 1,
                    b')' | b']' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let args = &body[open..j.saturating_sub(1)];
            // Last top-level argument is `elements_per_row`.
            if let Some(last) = split_top_level(args).last()
                && let Ok(v) = last.trim().parse::<usize>()
            {
                out.push(v);
            }
            at = open;
        }
    }
    out
}

/// Split on commas that are not inside brackets or parens.
fn split_top_level(args: &str) -> Vec<&str> {
    let (mut depth, mut start) = (0i32, 0usize);
    let mut parts = Vec::new();
    for (i, c) in args.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&args[start..]);
    parts
}
