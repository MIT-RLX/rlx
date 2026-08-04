// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct ARM SME2 f32 GEMM microkernel (Apple M4+).
//!
//! This is the *documented* matrix-hardware path Apple opened up on the M4:
//! ARM's Scalable Matrix Extension. Rust has no stable SME intrinsics yet, so
//! the microkernel is hand-written in `global_asm!`.
//!
//! # 4-tile register blocking (the perf idea)
//!
//! `FMOPA ZA.s, p/m, p/m, Zn.s, Zm.s` computes the **outer product**
//! `ZA[i,j] += Zn[i]·Zm[j]`. With SVL=512b (16 f32 lanes) one ZA tile is
//! `16×16`. SME2 has **four** f32 ZA tiles (ZA0..ZA3), so the kernel computes a
//! `32×32` output block per invocation as a 2×2 grid of tiles:
//!
//! ```text
//!   load a0=A[0:16,k]  a1=A[16:32,k]   b0=B[k,0:16]  b1=B[k,16:32]
//!   ZA0 += a0⊗b0   ZA1 += a0⊗b1
//!   ZA2 += a1⊗b0   ZA3 += a1⊗b1
//! ```
//!
//! Four FMOPAs per four loads = 4:1 compute:load (register blocking), so the
//! streaming pipeline stays fed instead of stalling on loads (the naive
//! single-tile kernel was load-bound). Tiles are packed contiguous in Rust so
//! the asm only does unit-stride `ld1w`. The kernel runs a whole 32-row **panel**
//! (all n-tiles) in ONE streaming session — `smstart` once, loop the tiles
//! internally, `smstop` once — so the streaming-mode transition is amortized
//! across the panel rather than paid per 32×32 block. Panels are then spread
//! across Rayon workers, each streaming a shared pre-packed B.
//!
//! # Reality check
//!
//! Even blocked+threaded this will not beat Accelerate for dense f32 (Accelerate
//! IS the vendor-tuned AMX/SME path — see [`super::dense`]); it exists as the
//! substrate for the int8/bf16 SME kernels (where there is no vendor matmul)
//! and for the regimes where a lean kernel avoids vendor per-call overhead. It
//! stays opt-in ([[feedback_perf_is_north_star]]).
//!
//! # Portability
//!
//! Hardcodes the 16-lane (512-bit SVL) geometry; [`is_available`] refuses to
//! dispatch unless the chip reports FEAT_SME2 + f32 outer-product + `svl==64`,
//! so the asm is only ever reached on matching hardware.

use super::detect;

/// f32 lanes per ZA tile dimension at 512-bit SVL.
const TILE: usize = 16;
/// Output block edge = 2 tiles (2×2 grid of ZA tiles).
const BLK: usize = 2 * TILE; // 32

unsafe extern "C" {
    /// Compute a whole 32-row × `n_tiles·32`-col output panel in ONE streaming
    /// session: `smstart` once, loop every 32×32 n-tile internally (zero ZA →
    /// K-loop FMOPA → store), `smstop` once — amortizing the streaming-mode
    /// transition (its per-tile cost dominated the naive kernel). `a_packed` is
    /// `k*32` f32 (shared across all n-tiles); `b_packed` is `n_tiles*k*32` f32
    /// (per-tile panels back-to-back); `c_out` is `32 × (n_tiles*32)` f32
    /// row-major scratch. Only valid when [`is_available`] holds.
    fn rlx_sme_fmopa_panel(
        a_packed: *const f32,
        b_packed: *const f32,
        c_out: *mut f32,
        k: usize,
        n_tiles: usize,
    );
}

core::arch::global_asm!(
    r#"
    .arch armv9-a+sme2
    .p2align 2
    .globl _rlx_sme_fmopa_panel
_rlx_sme_fmopa_panel:
    // x0=a_packed  x1=b_packed  x2=c_out  x3=k  x4=n_tiles
    smstart
    ptrue   p0.b
    lsl     x13, x4, #7            // c row stride bytes = n_tiles*32*4
    lsl     x15, x3, #7            // b per-tile stride bytes = k*32*4
    mov     x5, #0                 // tn = 0
    mov     x6, x1                 // b tile base
    mov     x14, #0                // c column byte offset = tn*128
1:  cmp     x5, x4                 // for tn in 0..n_tiles
    b.ge    4f
    zero    {{za}}
    mov     x7, x0                 // a working ptr (reset per tile)
    mov     x9, x6                 // b working ptr
    cbz     x3, 3f                 // k == 0 -> skip accumulate
    mov     x8, x3                 // k counter
2:  ld1w    {{z0.s}}, p0/z, [x7]
    ld1w    {{z1.s}}, p0/z, [x7, #1, mul vl]
    ld1w    {{z2.s}}, p0/z, [x9]
    ld1w    {{z3.s}}, p0/z, [x9, #1, mul vl]
    fmopa   za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa   za1.s, p0/m, p0/m, z0.s, z3.s
    fmopa   za2.s, p0/m, p0/m, z1.s, z2.s
    fmopa   za3.s, p0/m, p0/m, z1.s, z3.s
    add     x7, x7, #128
    add     x9, x9, #128
    subs    x8, x8, #1
    b.ne    2b
3:  mov     w12, #0                // store 32×32 tile at column offset x14
    mov     x11, #0                // row byte offset = i*rowstride
5:  mova    z4.s, p0/m, za0h.s[w12, 0]
    mova    z5.s, p0/m, za1h.s[w12, 0]
    mova    z6.s, p0/m, za2h.s[w12, 0]
    mova    z7.s, p0/m, za3h.s[w12, 0]
    add     x16, x2, x11           // c_out + i*rowstride
    add     x16, x16, x14          // + column offset
    add     x17, x16, x13, lsl #4  // + 16 rows (bottom half)
    st1w    {{z4.s}}, p0, [x16]
    st1w    {{z5.s}}, p0, [x16, #1, mul vl]
    st1w    {{z6.s}}, p0, [x17]
    st1w    {{z7.s}}, p0, [x17, #1, mul vl]
    add     x11, x11, x13
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    5b
    add     x6, x6, x15            // next b tile
    add     x14, x14, #128         // next c column (+32 f32)
    add     x5, x5, #1
    b       1b
4:  smstop
    ret
"#
);

/// True when a direct SME2 f32 GEMM can run: compiled in (`amx-sme`) and the
/// CPU reports FEAT_SME2 + f32 outer-product with the 512-bit SVL the kernel
/// hardcodes.
pub fn is_available() -> bool {
    detect::has_sme2() && detect::sme_f32f32() && detect::svl_bytes() == TILE * 4
}

/// Whether `sgemm_auto` should route through the SME kernel. Requires SME2
/// ([`is_available`]) **and** `RLX_CPU_SME=1`. Opt-in is deliberate: this does
/// not beat Accelerate for dense f32 ([[feedback_perf_is_north_star]]).
pub fn dispatch_enabled() -> bool {
    is_available()
        && matches!(
            std::env::var("RLX_CPU_SME").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
}

/// Whether the int8 **W8A8** SME path should be used in the quant executor.
/// Requires int8 SME ([`is_available_i8`]) + `RLX_CPU_SME_W8A8=1`. Opt-in is
/// mandatory: it quantizes activations to int8 (a lossier W8A8 mode than the
/// f32-activation oracle), so it must never be silently substituted
/// ([[feedback_perf_is_north_star]]).
pub fn w8a8_dispatch_enabled() -> bool {
    is_available_i8()
        && matches!(
            std::env::var("RLX_CPU_SME_W8A8").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
}

/// Cost-model crossover guard for the SME paths. The kernels pay a fixed cost
/// per call (tile packing + one `smstart`/`smstop` streaming-mode transition per
/// 32×32 block), so below a size threshold the scalar/NEON path wins even on
/// M4+. This gates the *already opt-in* fast paths so enabling them can never
/// regress tiny matmuls — a fast path that isn't fast for this shape falls back
/// ([[feedback_perf_is_north_star]]). Thresholds from `sme_lowprec_throughput_report`.
pub fn worth_sme(m: usize, k: usize, n: usize) -> bool {
    // Need enough K to amortize the mode switch, an output big enough that
    // packing overhead is a small fraction of the work, AND enough rows (`m`)
    // that the B-repack + MOPA's 32-row block are amortized. The `m >= 16` floor
    // is critical: at small `m` (esp. GEMV, m=1) the MOPA path is catastrophic
    // (repacks all of B, no m-parallelism, wastes 31/32 rows) — measured ~240ms
    // vs ~1.7ms for Accelerate at 1×4096×4096. Small-m/GEMV uses [`qgemv_i8`].
    m >= 16 && k >= 64 && n >= 32 && m.saturating_mul(n) >= 2048
}

/// Pack a 32-wide `A` row-panel: `dst[p*32 + i] = A[m0+i, p]`, rows `≥mt` zeroed.
#[inline]
fn pack_a_panel(a: &[f32], m0: usize, mt: usize, k: usize, dst: &mut [f32]) {
    for p in 0..k {
        let d = &mut dst[p * BLK..p * BLK + BLK];
        for (i, slot) in d.iter_mut().enumerate().take(BLK) {
            *slot = if i < mt { a[(m0 + i) * k + p] } else { 0.0 };
        }
    }
}

/// `C[m,n] = A[m,k] · B[k,n]`, all row-major f32, on the SME unit (overwrites
/// `C`). Caller must ensure [`is_available`]. Pre-packs B once, blocks the
/// output into 32×32 tiles, and spreads row-panels across Rayon workers.
pub fn sme_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    let m_tiles = m.div_ceil(BLK);
    let n_tiles = n.div_ceil(BLK);

    // Pre-pack B: b_pack_all[tn*k*32 + p*32 + j] = B[p, tn*32 + j], zero-padded.
    let mut b_pack_all = vec![0f32; n_tiles * k * BLK];
    for tn in 0..n_tiles {
        let n0 = tn * BLK;
        let nt = BLK.min(n - n0);
        let base = tn * k * BLK;
        for p in 0..k {
            let src = &b[p * n + n0..p * n + n0 + nt];
            b_pack_all[base + p * BLK..base + p * BLK + nt].copy_from_slice(src);
        }
    }

    let c_ptr = c.as_mut_ptr() as usize;
    let scratch_cols = n_tiles * BLK;
    // One 32-row panel per call — the panel kernel does all n-tiles in a single
    // streaming session, writing a padded 32×scratch_cols scratch we then copy
    // the valid `n` columns from. Reads shared `a`/`b_pack_all`; writes disjoint
    // C rows via raw ptr (m-panels never overlap → sound in parallel).
    let run_panel = |tm: usize, a_pack: &mut [f32], c_scratch: &mut [f32]| {
        let m0 = tm * BLK;
        let mt = BLK.min(m - m0);
        pack_a_panel(a, m0, mt, k, a_pack);
        // SAFETY: is_available() guarantees the SME2 geometry; a_pack=k*32,
        // b_pack_all=n_tiles*k*32, c_scratch=32*scratch_cols.
        unsafe {
            rlx_sme_fmopa_panel(
                a_pack.as_ptr(),
                b_pack_all.as_ptr(),
                c_scratch.as_mut_ptr(),
                k,
                n_tiles,
            );
        }
        // SAFETY: disjoint C rows [m0, m0+mt); first `n` scratch cols are valid.
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        for i in 0..mt {
            c[(m0 + i) * n..(m0 + i) * n + n]
                .copy_from_slice(&c_scratch[i * scratch_cols..i * scratch_cols + n]);
        }
    };

    let threads = crate::pool::num_threads();
    if threads > 1 && m_tiles > 1 {
        crate::pool::par_for(m_tiles, 1, &|off, cnt| {
            let mut a_pack = vec![0f32; k * BLK];
            let mut c_scratch = vec![0f32; BLK * scratch_cols];
            for tm in off..off + cnt {
                run_panel(tm, &mut a_pack, &mut c_scratch);
            }
        });
    } else {
        let mut a_pack = vec![0f32; k * BLK];
        let mut c_scratch = vec![0f32; BLK * scratch_cols];
        for tm in 0..m_tiles {
            run_panel(tm, &mut a_pack, &mut c_scratch);
        }
    }
}

// ── int8 SME GEMM (SMOPA int8→int32) ────────────────────────────────
//
// This is the path BNNSMatMul refused (it is float-only). SME2's `SMOPA`
// computes a *widening* outer product: each 32-bit ZA element accumulates a
// 4-element dot product, `ZA.s[i,j] += Σ_{d<4} Zn.b[4i+d]·Zm.b[4j+d]`. So one
// SMOPA reduces 4 K-values for a 16×16 int32 tile — 4× the K-throughput of the
// f32 FMOPA per instruction. We use the same 2×2 (32×32) 4-ZA-tile blocking.
//
// Packing groups K by 4: a byte vector holds, per 32-bit container `i`, the 4
// consecutive K-values of row `i`. Layout per K-group is `[a0 64B | a1 64B]`
// (rows 0–15 then 16–31), each `dst[i*4+d] = A[row_i, 4g+d]`, zero-padded.

/// K-values fused per SMOPA (the widening factor).
const KG: usize = 4;

unsafe extern "C" {
    /// int8 twin of [`rlx_sme_fmopa_32x32`]: `c_out[32×32] (i32) = Σ Xq⊗Wq`
    /// via `SMOPA`. `a_packed`/`b_packed` are `k_groups*128` i8 (see packing
    /// above); `c_out` is `32*32` i32. `k_groups = ceil(k/4)`.
    fn rlx_sme_smopa_32x32_i8(
        a_packed: *const i8,
        b_packed: *const i8,
        c_out: *mut i32,
        k_groups: usize,
    );
}

core::arch::global_asm!(
    r#"
    .arch armv9-a+sme2
    .p2align 2
    .globl _rlx_sme_smopa_32x32_i8
_rlx_sme_smopa_32x32_i8:
    // x0=a_packed(i8)  x1=b_packed(i8)  x2=c_out(i32)  x3=k_groups
    smstart
    zero    {{za}}
    ptrue   p0.b
    cbz     x3, 2f
    mov     x4, x3
1:  ld1b    {{z0.b}}, p0/z, [x0]              // a0 = rows 0:16  (16×4 bytes)
    ld1b    {{z1.b}}, p0/z, [x0, #1, mul vl]  // a1 = rows 16:32
    ld1b    {{z2.b}}, p0/z, [x1]              // b0 = cols 0:16
    ld1b    {{z3.b}}, p0/z, [x1, #1, mul vl]  // b1 = cols 16:32
    smopa   za0.s, p0/m, p0/m, z0.b, z2.b
    smopa   za1.s, p0/m, p0/m, z0.b, z3.b
    smopa   za2.s, p0/m, p0/m, z1.b, z2.b
    smopa   za3.s, p0/m, p0/m, z1.b, z3.b
    add     x0, x0, #128
    add     x1, x1, #128
    subs    x4, x4, #1
    b.ne    1b
2:  mov     w12, #0
    mov     x8, #0
3:  mova    z4.s, p0/m, za0h.s[w12, 0]
    mova    z5.s, p0/m, za1h.s[w12, 0]
    mova    z6.s, p0/m, za2h.s[w12, 0]
    mova    z7.s, p0/m, za3h.s[w12, 0]
    add     x6, x2, x8
    add     x7, x6, #2048
    st1w    {{z4.s}}, p0, [x6]
    st1w    {{z5.s}}, p0, [x6, #1, mul vl]
    st1w    {{z6.s}}, p0, [x7]
    st1w    {{z7.s}}, p0, [x7, #1, mul vl]
    add     x8, x8, #128
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    3b
    smstop
    ret
"#
);

/// True when the int8 SME GEMM can run: compiled in + FEAT_SME2 + int8→int32
/// outer-product + 512-bit SVL.
pub fn is_available_i8() -> bool {
    detect::has_sme2() && detect::sme_i8i32() && detect::svl_bytes() == TILE * 4
}

/// Pack a 32-wide `A` row-panel for SMOPA: per K-group, `[rows0:16 | rows16:32]`
/// with `dst[g*128 + blk*64 + i*4 + d] = A[m0 + blk*16 + i, 4g+d]`, zero-padded.
#[inline]
fn pack_a_panel_i8(a: &[i8], m0: usize, mt: usize, k: usize, k_groups: usize, dst: &mut [i8]) {
    for g in 0..k_groups {
        for blk in 0..2 {
            for i in 0..TILE {
                let local = blk * TILE + i;
                for d in 0..KG {
                    let kk = g * KG + d;
                    dst[g * 128 + blk * 64 + i * 4 + d] = if local < mt && kk < k {
                        a[(m0 + local) * k + kk]
                    } else {
                        0
                    };
                }
            }
        }
    }
}

/// Pack a 32-wide `B` col-panel for SMOPA: `dst[g*128 + blk*64 + j*4 + d] =
/// B[4g+d, n0 + blk*16 + j]`, zero-padded.
#[inline]
fn pack_b_panel_i8(
    b: &[i8],
    n0: usize,
    nt: usize,
    k: usize,
    n: usize,
    k_groups: usize,
    dst: &mut [i8],
) {
    for g in 0..k_groups {
        for blk in 0..2 {
            for j in 0..TILE {
                let col = blk * TILE + j;
                for d in 0..KG {
                    let kk = g * KG + d;
                    dst[g * 128 + blk * 64 + j * 4 + d] = if col < nt && kk < k {
                        b[kk * n + n0 + col]
                    } else {
                        0
                    };
                }
            }
        }
    }
}

/// Tiled int8 SMOPA GEMM core. Computes the int32 dot products on the SME unit,
/// then writes `out[(m0+i)*n + j] = dequant(j, int32)` — the closure maps the
/// global output column `j` and the raw int32 accumulator to f32, so both
/// per-tensor and per-column (per-output-channel) dequant reuse one kernel.
/// Caller must ensure [`is_available_i8`]. Pre-packs B once, threads row-panels.
fn qmatmul_i8_tiled(
    xq: &[i8],
    wq: &[i8],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    dequant: &(dyn Fn(usize, i32) -> f32 + Sync),
) {
    debug_assert_eq!(xq.len(), m * k);
    debug_assert_eq!(wq.len(), k * n);
    debug_assert_eq!(out.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    let k_groups = k.div_ceil(KG);
    let m_tiles = m.div_ceil(BLK);
    let n_tiles = n.div_ceil(BLK);

    let mut b_pack_all = vec![0i8; n_tiles * k_groups * 128];
    for tn in 0..n_tiles {
        let n0 = tn * BLK;
        let nt = BLK.min(n - n0);
        let base = tn * k_groups * 128;
        pack_b_panel_i8(
            wq,
            n0,
            nt,
            k,
            n,
            k_groups,
            &mut b_pack_all[base..base + k_groups * 128],
        );
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let run_panel = |tm: usize, a_pack: &mut [i8], c_tile: &mut [i32]| {
        let m0 = tm * BLK;
        let mt = BLK.min(m - m0);
        pack_a_panel_i8(xq, m0, mt, k, k_groups, a_pack);
        for tn in 0..n_tiles {
            let n0 = tn * BLK;
            let nt = BLK.min(n - n0);
            let b_pack = &b_pack_all[tn * k_groups * 128..tn * k_groups * 128 + k_groups * 128];
            // SAFETY: is_available_i8() guarantees the SME2 int8 geometry;
            // buffers sized k_groups*128 (packs) and 32*32 (c_tile).
            unsafe {
                rlx_sme_smopa_32x32_i8(
                    a_pack.as_ptr(),
                    b_pack.as_ptr(),
                    c_tile.as_mut_ptr(),
                    k_groups,
                );
            }
            // SAFETY: disjoint C rows [m0,m0+mt) × cols [n0,n0+nt).
            let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, m * n) };
            for i in 0..mt {
                let row = &mut out[(m0 + i) * n + n0..(m0 + i) * n + n0 + nt];
                for (jj, slot) in row.iter_mut().enumerate() {
                    *slot = dequant(n0 + jj, c_tile[i * BLK + jj]);
                }
            }
        }
    };

    let threads = crate::pool::num_threads();
    if threads > 1 && m_tiles > 1 {
        crate::pool::par_for(m_tiles, 1, &|off, cnt| {
            let mut a_pack = vec![0i8; k_groups * 128];
            let mut c_tile = vec![0i32; BLK * BLK];
            for tm in off..off + cnt {
                run_panel(tm, &mut a_pack, &mut c_tile);
            }
        });
    } else {
        let mut a_pack = vec![0i8; k_groups * 128];
        let mut c_tile = vec![0i32; BLK * BLK];
        for tm in 0..m_tiles {
            run_panel(tm, &mut a_pack, &mut c_tile);
        }
    }
}

/// W8A8 int8 GEMM, symmetric **per-tensor** quant:
/// `C[m,n] = x_scale·w_scale · (Xq·Wq)`. `xq`=`[m,k]`, `wq`=`[k,n]` row-major i8.
pub fn sme_qmatmul_i8(
    xq: &[i8],
    x_scale: f32,
    wq: &[i8],
    w_scale: f32,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let scale = x_scale * w_scale;
    qmatmul_i8_tiled(xq, wq, out, m, k, n, &move |_j, v| scale * v as f32);
}

/// W8A8 int8 GEMM, symmetric **per-column** (per-output-channel) weight quant:
/// `C[i,j] = x_scale·w_col_scale[j] · Σ_p Xq[i,p]·Wq[p,j]`. This is the common
/// LLM weight-quant layout (one scale per output channel). `w_col_scale` has
/// length `n`.
pub fn sme_qmatmul_i8_percol(
    xq: &[i8],
    x_scale: f32,
    wq: &[i8],
    w_col_scale: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(w_col_scale.len(), n);
    qmatmul_i8_tiled(xq, wq, out, m, k, n, &move |j, v| {
        x_scale * w_col_scale[j] * v as f32
    });
}

/// int8 **GEMV** (decode, m=1): `out[j] = x_scale·w_col_scale[j] · Σ_p xq[p]·Wq[p,j]`.
///
/// Deliberately NOT the MOPA kernel — matrix-vector is memory-bound (read Wq
/// once), and MOPA wastes 31/32 rows + repacks Wq per call, which measured
/// ~140× slower than this at 1×4096×4096. Instead this reads `Wq` row-major and
/// accumulates a cache-resident `i32[n]` (the auto-vectorized AXPY inner loop
/// becomes NEON widening MACs), threaded over N so `Wq` is streamed once total.
/// Because int8 is ¼ the bytes of f32, this beats even Accelerate's f32 GEMV on
/// bandwidth. `wq` is `[k,n]` row-major i8; `w_col_scale` has length `n`.
pub fn qgemv_i8(
    xq: &[i8],
    x_scale: f32,
    wq: &[i8],
    w_col_scale: &[f32],
    out: &mut [f32],
    k: usize,
    n: usize,
) {
    debug_assert_eq!(xq.len(), k);
    debug_assert_eq!(wq.len(), k * n);
    debug_assert_eq!(w_col_scale.len(), n);
    debug_assert_eq!(out.len(), n);
    if n == 0 {
        return;
    }
    let out_ptr = out.as_mut_ptr() as usize;
    // A 64-column tile (one 64-byte cache line per weight row) with 16 int32x4
    // accumulators held across the whole K reduction, widening i8→i32 MAC.
    // NOTE: with row-major `[k,n]` weights the per-row reads for a column tile
    // stride by `n`, so this is ultimately latency-bound (~9× faster than the
    // scalar path but not memory-bandwidth-bound). A truly bandwidth-bound int8
    // GEMV needs `[n,k]` weights (contiguous per output channel — the GGUF `bt`
    // layout); `DequantMatMul` stores `[k,n]`, so that's a separate path.
    let compute_block = |j0: usize, bn: usize| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, n) };
        let mut col = j0;
        let end = j0 + bn;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            while col + 64 <= end {
                let mut a = [vdupq_n_s32(0); 16];
                let wbase = wq.as_ptr().add(col);
                for p in 0..k {
                    let xkv = vdupq_n_s32(*xq.get_unchecked(p) as i32);
                    let row = wbase.add(p * n);
                    for c in 0..4 {
                        let w = vld1q_s8(row.add(c * 16));
                        let wlo = vmovl_s8(vget_low_s8(w));
                        let whi = vmovl_s8(vget_high_s8(w));
                        let b = c * 4;
                        a[b] = vmlaq_s32(a[b], vmovl_s16(vget_low_s16(wlo)), xkv);
                        a[b + 1] = vmlaq_s32(a[b + 1], vmovl_high_s16(wlo), xkv);
                        a[b + 2] = vmlaq_s32(a[b + 2], vmovl_s16(vget_low_s16(whi)), xkv);
                        a[b + 3] = vmlaq_s32(a[b + 3], vmovl_high_s16(whi), xkv);
                    }
                }
                let mut acc = [0i32; 64];
                for (i, v) in a.iter().enumerate() {
                    vst1q_s32(acc.as_mut_ptr().add(i * 4), *v);
                }
                for (d, &av) in acc.iter().enumerate() {
                    *out.get_unchecked_mut(col + d) =
                        x_scale * *w_col_scale.get_unchecked(col + d) * av as f32;
                }
                col += 64;
            }
        }
        while col < end {
            let mut a = 0i32;
            for p in 0..k {
                a += xq[p] as i32 * wq[p * n + col] as i32;
            }
            out[col] = x_scale * w_col_scale[col] * a as f32;
            col += 1;
        }
    };
    let threads = crate::pool::num_threads();
    let target = n.div_ceil(threads.max(1)).clamp(64, 1024);
    let n_blocks = n.div_ceil(target);
    if threads > 1 && n_blocks > 1 {
        crate::pool::par_for(n_blocks, 1, &|off, cnt| {
            for b in off..off + cnt {
                let j0 = b * target;
                compute_block(j0, target.min(n - j0));
            }
        });
    } else {
        let mut j0 = 0;
        while j0 < n {
            let bn = target.min(n - j0);
            compute_block(j0, bn);
            j0 += bn;
        }
    }
}

unsafe extern "C" {
    /// `Σ_{p<k} x[p]·w[p]` for two contiguous i8 vectors, via NEON `SDOT`
    /// (4-way i8→i32 dot, 1 instruction per 16 MACs). `SDOT` has no stable Rust
    /// intrinsic yet, so it lives in `global_asm!`. Two accumulators hide latency.
    fn rlx_sdot_i8(x: *const i8, w: *const i8, k: usize) -> i32;
}

core::arch::global_asm!(
    r#"
    .arch armv8.2-a+dotprod
    .p2align 2
    .globl _rlx_sdot_i8
_rlx_sdot_i8:
    // x0=x  x1=w  x2=k (MUST be a positive multiple of 16) -> w0.
    // The K remainder is handled in Rust, so this stays a clean vector loop.
    movi    v0.4s, #0
    movi    v16.4s, #0
    cmp     x2, #32
    b.lt    2f
1:  ld1     {{v1.16b}}, [x0], #16
    ld1     {{v2.16b}}, [x1], #16
    ld1     {{v3.16b}}, [x0], #16
    ld1     {{v4.16b}}, [x1], #16
    sdot    v0.4s, v1.16b, v2.16b
    sdot    v16.4s, v3.16b, v4.16b
    sub     x2, x2, #32
    cmp     x2, #32
    b.ge    1b
2:  cbz     x2, 3f
    ld1     {{v1.16b}}, [x0], #16
    ld1     {{v2.16b}}, [x1], #16
    sdot    v0.4s, v1.16b, v2.16b
3:  add     v0.4s, v0.4s, v16.4s
    addv    s0, v0.4s
    fmov    w0, s0
    ret
"#
);

/// int8 **GEMV for `[n,k]` weights** (per-output-channel contiguous — the GGUF
/// `bt` layout that real quantized models use): `out[j] = x_scale·w_row_scale[j]
/// · Σ_p xq[p]·Wq[j,p]`, `Wq` is `[n,k]` row-major i8, `w_row_scale` length `n`.
///
/// Unlike the `[k,n]` GEMV ([`qgemv_i8`], latency-bound on strided column
/// reads), each output channel here reads a **contiguous** `Wq[j,:]` vector, so
/// this is a series of dot products (NEON `SDOT`, 4-way int8→int32) with a
/// register-scalar accumulator — sequential weight streaming, no accumulator
/// round-trips. Threaded over output channels, it reads `Wq` once → memory
/// bandwidth bound, and since int8 is ¼ the bytes of f32 it beats Accelerate's
/// f32 GEMV. This is the fast-decode path the `[k,n]` layout couldn't provide.
pub fn qgemv_i8_nk(
    xq: &[i8],
    x_scale: f32,
    wq: &[i8],
    w_row_scale: &[f32],
    out: &mut [f32],
    k: usize,
    n: usize,
) {
    debug_assert_eq!(xq.len(), k);
    debug_assert_eq!(wq.len(), n * k);
    debug_assert_eq!(w_row_scale.len(), n);
    debug_assert_eq!(out.len(), n);
    if n == 0 {
        return;
    }
    let out_ptr = out.as_mut_ptr() as usize;
    // Whole-vector dot of two contiguous i8 slices → i32. The SDOT asm handles
    // the K-16-multiple prefix; the remainder is a safe Rust tail.
    #[inline(always)]
    fn dot_i8(x: &[i8], w: &[i8], k: usize) -> i32 {
        let mut s;
        let k16 = k & !15;
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: k16 is a multiple of 16 ≤ k, so the asm reads k16 valid bytes.
            s = if k16 > 0 {
                unsafe { rlx_sdot_i8(x.as_ptr(), w.as_ptr(), k16) }
            } else {
                0
            };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            s = 0i32;
            for p in 0..k16 {
                s += x[p] as i32 * w[p] as i32;
            }
        }
        for p in k16..k {
            s += x[p] as i32 * w[p] as i32;
        }
        s
    }
    let compute = |j0: usize, jn: usize| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, n) };
        for j in j0..j0 + jn {
            let s = dot_i8(xq, &wq[j * k..j * k + k], k);
            out[j] = x_scale * w_row_scale[j] * s as f32;
        }
    };
    let threads = crate::pool::num_threads();
    if threads > 1 && n >= 2 * threads {
        let chunk = n.div_ceil(threads);
        crate::pool::par_for(threads, 1, &|off, cnt| {
            for t in off..off + cnt {
                let j0 = t * chunk;
                if j0 < n {
                    compute(j0, chunk.min(n - j0));
                }
            }
        });
    } else {
        compute(0, n);
    }
}

/// W8A8 decode GEMV over **GGUF Q8_0** weights (the real quantized wire format:
/// `[n,k]` int8 in 34-byte blocks = `[f16 scale][32 int8]`, `n_blocks = k/32`).
/// `out[j] = Σ_b (x_bscale[b]·w_bscale) · SDOT(xq_block[b], Wq[j,b])`.
///
/// This applies the fast `[n,k]` SDOT path to real GGUF weights: activations are
/// quantized per 32-block **once** (reused across all `n` output rows), then each
/// output row is a per-block SDOT × (both block scales). Q8_0's per-block scale
/// is folded per block (the MOPA path can't do that). `w_bytes` is
/// `n·n_blocks·34` bytes; `x` is f32 `[k]`; `out` is f32 `[n]`.
pub fn qgemv_q8_0(w_bytes: &[u8], x: &[f32], out: &mut [f32], k: usize, n: usize) {
    const QK: usize = 32; // Q8_0 block elements
    const BB: usize = 2 + QK; // bytes per block: f16 scale + 32 i8
    debug_assert_eq!(k % QK, 0, "Q8_0 GEMV requires k % 32 == 0");
    debug_assert_eq!(out.len(), n);
    let n_blocks = k / QK;
    debug_assert!(w_bytes.len() >= n * n_blocks * BB);
    if n == 0 {
        return;
    }
    // Quantize x once, per 32-block (symmetric), shared across all output rows.
    let mut xq = vec![0i8; n_blocks * QK];
    let mut xs = vec![0f32; n_blocks];
    for b in 0..n_blocks {
        let (codes, scale) = quantize_i8_symmetric(&x[b * QK..b * QK + QK]);
        xq[b * QK..b * QK + QK].copy_from_slice(&codes);
        xs[b] = scale;
    }

    let out_ptr = out.as_mut_ptr() as usize;
    let wptr = w_bytes.as_ptr() as usize;
    let compute = |j0: usize, jn: usize| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, n) };
        for j in j0..j0 + jn {
            let mut acc = 0f32;
            for b in 0..n_blocks {
                let blk = (j * n_blocks + b) * BB;
                // f16 block scale (little-endian) + 32 packed int8.
                let ws = unsafe {
                    let lo = *(wptr as *const u8).add(blk) as u16;
                    let hi = *(wptr as *const u8).add(blk + 1) as u16;
                    half::f16::from_bits(lo | (hi << 8)).to_f32()
                };
                let wq = unsafe { (wptr as *const i8).add(blk + 2) };
                // SAFETY: block holds 32 valid i8; k16=32 multiple of 16.
                let dot = {
                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe { rlx_sdot_i8(xq.as_ptr().add(b * QK), wq, QK) }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        let mut s = 0i32;
                        for i in 0..QK {
                            s += xq[b * QK + i] as i32 * unsafe { *wq.add(i) } as i32;
                        }
                        s
                    }
                };
                acc += dot as f32 * xs[b] * ws;
            }
            out[j] = acc;
        }
    };
    let threads = crate::pool::num_threads();
    if threads > 1 && n >= 2 * threads {
        let chunk = n.div_ceil(threads);
        crate::pool::par_for(threads, 1, &|off, cnt| {
            for t in off..off + cnt {
                let j0 = t * chunk;
                if j0 < n {
                    compute(j0, chunk.min(n - j0));
                }
            }
        });
    } else {
        compute(0, n);
    }
}

// ── bf16 SME GEMM (BFMOPA bf16→f32) ─────────────────────────────────
//
// `BFMOPA` is the bf16 widening outer product: `ZA.s[i,j] += Σ_{d<2}
// Zn.bf16[2i+d]·Zm.bf16[2j+d]` — a 2-way dot into f32 accumulators, so K is
// grouped by 2. Half the operand bytes of f32 with f32-accumulate accuracy;
// complements BNNS bf16 (this is our own kernel, no vendor call). Same 2×2
// 32×32 4-ZA-tile blocking; per K-group layout `[a0 32×bf16 | a1 32×bf16]`.

/// K-values fused per BFMOPA.
const KG_BF: usize = 2;

unsafe extern "C" {
    /// bf16 twin of [`rlx_sme_fmopa_32x32`]: `c_out[32×32] (f32) = Σ A⊗B` via
    /// `BFMOPA`. `a_packed`/`b_packed` are `k_groups*64` bf16 bit-patterns;
    /// `c_out` is `32*32` f32. `k_groups = ceil(k/2)`.
    fn rlx_sme_bfmopa_32x32(
        a_packed: *const u16,
        b_packed: *const u16,
        c_out: *mut f32,
        k_groups: usize,
    );
}

core::arch::global_asm!(
    r#"
    .arch armv9-a+sme2
    .p2align 2
    .globl _rlx_sme_bfmopa_32x32
_rlx_sme_bfmopa_32x32:
    // x0=a_packed(bf16)  x1=b_packed(bf16)  x2=c_out(f32)  x3=k_groups
    smstart
    zero    {{za}}
    ptrue   p0.b
    cbz     x3, 2f
    mov     x4, x3
1:  ld1h    {{z0.h}}, p0/z, [x0]              // a0 = rows 0:16  (16×2 bf16)
    ld1h    {{z1.h}}, p0/z, [x0, #1, mul vl]  // a1 = rows 16:32
    ld1h    {{z2.h}}, p0/z, [x1]              // b0 = cols 0:16
    ld1h    {{z3.h}}, p0/z, [x1, #1, mul vl]  // b1 = cols 16:32
    bfmopa  za0.s, p0/m, p0/m, z0.h, z2.h
    bfmopa  za1.s, p0/m, p0/m, z0.h, z3.h
    bfmopa  za2.s, p0/m, p0/m, z1.h, z2.h
    bfmopa  za3.s, p0/m, p0/m, z1.h, z3.h
    add     x0, x0, #128                      // += 32 bf16 ×2 blocks = 128 bytes
    add     x1, x1, #128
    subs    x4, x4, #1
    b.ne    1b
2:  mov     w12, #0
    mov     x8, #0
3:  mova    z4.s, p0/m, za0h.s[w12, 0]
    mova    z5.s, p0/m, za1h.s[w12, 0]
    mova    z6.s, p0/m, za2h.s[w12, 0]
    mova    z7.s, p0/m, za3h.s[w12, 0]
    add     x6, x2, x8
    add     x7, x6, #2048
    st1w    {{z4.s}}, p0, [x6]
    st1w    {{z5.s}}, p0, [x6, #1, mul vl]
    st1w    {{z6.s}}, p0, [x7]
    st1w    {{z7.s}}, p0, [x7, #1, mul vl]
    add     x8, x8, #128
    add     w12, w12, #1
    cmp     w12, #16
    b.lt    3b
    smstop
    ret
"#
);

/// True when the bf16 SME GEMM can run: compiled in + FEAT_SME2 + bf16→f32
/// outer-product + 512-bit SVL.
pub fn is_available_bf16() -> bool {
    detect::has_sme2() && detect::sme_b16f32() && detect::svl_bytes() == TILE * 4
}

/// Whether `sgemm_auto` should route through the native SME bf16 kernel.
/// Requires bf16 SME ([`is_available_bf16`]) + `RLX_CPU_SME_BF16=1`. Opt-in:
/// f32→bf16 downcast is lossy ([[feedback_perf_is_north_star]]).
pub fn bf16_dispatch_enabled() -> bool {
    is_available_bf16()
        && matches!(
            std::env::var("RLX_CPU_SME_BF16").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
}

/// Pack a 32-wide `A` row-panel to bf16 for BFMOPA:
/// `dst[g*64 + blk*32 + i*2 + d] = bf16(A[m0 + blk*16 + i, 2g+d])`, zero-padded.
#[inline]
fn pack_a_panel_bf16(a: &[f32], m0: usize, mt: usize, k: usize, k_groups: usize, dst: &mut [u16]) {
    for g in 0..k_groups {
        for blk in 0..2 {
            for i in 0..TILE {
                let local = blk * TILE + i;
                for d in 0..KG_BF {
                    let kk = g * KG_BF + d;
                    dst[g * 64 + blk * 32 + i * 2 + d] = if local < mt && kk < k {
                        half::bf16::from_f32(a[(m0 + local) * k + kk]).to_bits()
                    } else {
                        0
                    };
                }
            }
        }
    }
}

/// Pack a 32-wide `B` col-panel to bf16 for BFMOPA.
#[inline]
fn pack_b_panel_bf16(
    b: &[f32],
    n0: usize,
    nt: usize,
    k: usize,
    n: usize,
    k_groups: usize,
    dst: &mut [u16],
) {
    for g in 0..k_groups {
        for blk in 0..2 {
            for j in 0..TILE {
                let col = blk * TILE + j;
                for d in 0..KG_BF {
                    let kk = g * KG_BF + d;
                    dst[g * 64 + blk * 32 + j * 2 + d] = if col < nt && kk < k {
                        half::bf16::from_f32(b[kk * n + n0 + col]).to_bits()
                    } else {
                        0
                    };
                }
            }
        }
    }
}

/// Low-precision `C[m,n] = A·B` on the SME unit: f32 operands are downcast to
/// bf16 in-pack, accumulated in f32 (overwrites `C`). Caller must ensure
/// [`is_available_bf16`]. Pre-packs B once, blocks 32×32, threads row-panels.
pub fn sme_sgemm_bf16(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    let k_groups = k.div_ceil(KG_BF);
    let m_tiles = m.div_ceil(BLK);
    let n_tiles = n.div_ceil(BLK);

    let mut b_pack_all = vec![0u16; n_tiles * k_groups * 64];
    for tn in 0..n_tiles {
        let n0 = tn * BLK;
        let nt = BLK.min(n - n0);
        let base = tn * k_groups * 64;
        pack_b_panel_bf16(
            b,
            n0,
            nt,
            k,
            n,
            k_groups,
            &mut b_pack_all[base..base + k_groups * 64],
        );
    }

    let c_ptr = c.as_mut_ptr() as usize;
    let run_panel = |tm: usize, a_pack: &mut [u16], c_tile: &mut [f32]| {
        let m0 = tm * BLK;
        let mt = BLK.min(m - m0);
        pack_a_panel_bf16(a, m0, mt, k, k_groups, a_pack);
        for tn in 0..n_tiles {
            let n0 = tn * BLK;
            let nt = BLK.min(n - n0);
            let b_pack = &b_pack_all[tn * k_groups * 64..tn * k_groups * 64 + k_groups * 64];
            // SAFETY: is_available_bf16() guarantees the geometry; buffers sized
            // k_groups*64 (packs) and 32*32 (c_tile).
            unsafe {
                rlx_sme_bfmopa_32x32(
                    a_pack.as_ptr(),
                    b_pack.as_ptr(),
                    c_tile.as_mut_ptr(),
                    k_groups,
                );
            }
            let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
            for i in 0..mt {
                c[(m0 + i) * n + n0..(m0 + i) * n + n0 + nt]
                    .copy_from_slice(&c_tile[i * BLK..i * BLK + nt]);
            }
        }
    };

    let threads = crate::pool::num_threads();
    if threads > 1 && m_tiles > 1 {
        crate::pool::par_for(m_tiles, 1, &|off, cnt| {
            let mut a_pack = vec![0u16; k_groups * 64];
            let mut c_tile = vec![0f32; BLK * BLK];
            for tm in off..off + cnt {
                run_panel(tm, &mut a_pack, &mut c_tile);
            }
        });
    } else {
        let mut a_pack = vec![0u16; k_groups * 64];
        let mut c_tile = vec![0f32; BLK * BLK];
        for tm in 0..m_tiles {
            run_panel(tm, &mut a_pack, &mut c_tile);
        }
    }
}

/// Symmetric per-tensor int8 quantization: `scale = max|v|/127`,
/// `code = round(v/scale)` clamped to `[-127,127]`; `scale=1` for all-zero.
pub fn quantize_i8_symmetric(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().fold(0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    (
        v.iter()
            .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
            .collect(),
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn fill(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn sme_matches_naive_across_shapes() {
        if !is_available() {
            eprintln!("SME2 not available on this host; skipping");
            return;
        }
        // Exercise edge padding on both axes + multi-tile + k=0.
        let shapes = [
            (1, 1, 1),
            (16, 16, 16),
            (32, 32, 32),
            (17, 33, 15),
            (33, 65, 31),
            (64, 128, 96),
            (100, 100, 100),
            (5, 5, 100),
        ];
        for (m, k, n) in shapes {
            let a = fill(m * k, 0x1234 + (m * k) as u64);
            let b = fill(k * n, 0x9abc + (k * n) as u64);
            let mut c = vec![0f32; m * n];
            sme_sgemm(&a, &b, &mut c, m, k, n);
            let want = naive(&a, &b, m, k, n);
            let max_abs = c
                .iter()
                .zip(&want)
                .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
            let tol = 1e-3 * (k as f32).sqrt().max(1.0);
            assert!(
                max_abs <= tol,
                "shape {m}x{k}x{n}: max diff {max_abs} > {tol}"
            );
        }
    }

    fn fill_i8(len: usize, seed: u64) -> Vec<i8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 32) as i32 % 255 - 127) as i8
            })
            .collect()
    }

    fn naive_i32(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a[i * k + p] as i32 * b[p * n + j] as i32;
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    /// The int8 SMOPA kernel is EXACT integer arithmetic — with scale 1.0 the
    /// f32 output must equal the int32 dot product bit-for-bit (k kept small so
    /// sums stay < 2^24 where f32 is exact). Covers K not a multiple of 4 and
    /// edge tiles.
    #[test]
    fn sme_i8_exact_integer() {
        if !is_available_i8() {
            eprintln!("SME2 int8 not available; skipping");
            return;
        }
        for (m, k, n) in [
            (1, 1, 1),
            (16, 4, 16),
            (17, 7, 15),
            (32, 32, 32),
            (33, 70, 31),
            (48, 100, 96),
        ] {
            let a = fill_i8(m * k, 0x51 + (m * k) as u64);
            let b = fill_i8(k * n, 0x73 + (k * n) as u64);
            let mut out = vec![0f32; m * n];
            sme_qmatmul_i8(&a, 1.0, &b, 1.0, &mut out, m, k, n);
            let want = naive_i32(&a, &b, m, k, n);
            for (idx, (g, w)) in out.iter().zip(&want).enumerate() {
                assert_eq!(*g, *w as f32, "shape {m}x{k}x{n} idx {idx}: {g} != {w}");
            }
        }
    }

    /// W8A8: quantize a random f32 problem, run int8 SME, compare vs the f32
    /// product. Not bit-exact (that's the point) but must track closely.
    #[test]
    fn sme_i8_w8a8_tracks_f32() {
        if !is_available_i8() {
            return;
        }
        for (m, k, n) in [(8, 64, 8), (32, 256, 48), (64, 512, 64)] {
            let x = fill(m * k, 3 + (m * k) as u64);
            let w = fill(k * n, 5 + (k * n) as u64);
            let (xq, sx) = quantize_i8_symmetric(&x);
            let (wq, sw) = quantize_i8_symmetric(&w);
            let mut got = vec![0f32; m * n];
            sme_qmatmul_i8(&xq, sx, &wq, sw, &mut got, m, k, n);
            let want = naive(&x, &w, m, k, n);
            let (mut dot, mut ng, mut nw, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (g, t) in got.iter().zip(&want) {
                dot += (*g as f64) * (*t as f64);
                ng += (*g as f64).powi(2);
                nw += (*t as f64).powi(2);
                num += ((*g - *t) as f64).powi(2);
                den += (*t as f64).powi(2);
            }
            let cos = dot / (ng.sqrt() * nw.sqrt());
            let rel = (num / den).sqrt();
            eprintln!("SME W8A8 {m}x{k}x{n}: cosine={cos:.5} rel={rel:.4}");
            assert!(cos > 0.99, "cosine {cos} too low");
            assert!(rel < 0.06, "rel err {rel} too high");
        }
    }

    /// Throughput of the SME low-precision kernels vs the scalar int8 oracle
    /// (`dequant_matmul_int8` — the only other CPU int8 matmul; BNNS has none)
    /// and vs Accelerate f32 as a reference ceiling.
    ///   cargo test -p rlx-cpu --features amx-sme sme_lowprec_throughput_report \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn sme_lowprec_throughput_report() {
        use std::time::Instant;
        if !is_available_i8() {
            return;
        }
        let gflops = |m: usize, k: usize, n: usize, iters: usize, f: &mut dyn FnMut()| -> f64 {
            for _ in 0..2 {
                f();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            2.0 * m as f64 * n as f64 * k as f64 * iters as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        eprintln!(
            "{:>16}  {:>12}  {:>12}  {:>11}  {:>10}",
            "shape", "scalar-i8", "SME-i8", "SME-bf16", "Accel-f32"
        );
        // NB: the scalar int8 oracle is a pure reference (no SIMD/threads) and is
        // ~1000× slower than the SME path, so keep shapes modest or it dominates
        // wall time.
        for (m, k, n) in [
            (256usize, 256usize, 256usize),
            (512, 512, 512),
            (1024, 1024, 1024),
        ] {
            let xf = fill(m * k, 1 + (m * k) as u64);
            let wf = fill(k * n, 2 + (k * n) as u64);
            let (xq, sx) = quantize_i8_symmetric(&xf);
            let (wq, sw) = quantize_i8_symmetric(&wf);
            let mut out = vec![0f32; m * n];
            let iters =
                ((6e8 / (2.0 * m as f64 * k as f64 * n as f64)).ceil() as usize).clamp(3, 100);

            // Scalar int8 oracle: single-block (per-column), symmetric. scales
            // is [n_blocks=1, n], so one scale per output column.
            let scales_col = vec![sw; n];
            let scalar = gflops(m, k, n, iters, &mut || {
                crate::thunk::dequant_matmul_int8(
                    &xf,
                    &wq,
                    &scales_col,
                    &[],
                    &mut out,
                    m,
                    k,
                    n,
                    k,
                    false,
                )
            });
            let sme_i8 = gflops(m, k, n, iters, &mut || {
                sme_qmatmul_i8(&xq, sx, &wq, sw, &mut out, m, k, n);
            });
            let sme_bf = if is_available_bf16() {
                gflops(m, k, n, iters, &mut || {
                    sme_sgemm_bf16(&xf, &wf, &mut out, m, k, n)
                })
            } else {
                0.0
            };
            let accel = gflops(m, k, n, iters, &mut || {
                crate::blas::sgemm(&xf, &wf, &mut out, m, k, n)
            });
            eprintln!(
                "{:>16}  {:>12.1}  {:>12.1}  {:>11.1}  {:>10.1}",
                format!("{m}x{k}x{n}"),
                scalar,
                sme_i8,
                sme_bf,
                accel
            );
        }
    }

    /// K-sweep probe (task: does K-cache-blocking help?). Fixed m=n, growing K.
    /// If GF/s rises or holds with K the kernel is compute-bound and K-blocking
    /// is the wrong lever; a drop at large K would signal a cache cliff worth
    /// blocking. Run: cargo test ... sme_kblock_probe -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn sme_kblock_probe() {
        use std::time::Instant;
        if !is_available() {
            return;
        }
        let gf = |m: usize, k: usize, n: usize, f: &mut dyn FnMut()| -> f64 {
            for _ in 0..2 {
                f();
            }
            let it = 5;
            let t0 = Instant::now();
            for _ in 0..it {
                f();
            }
            2.0 * m as f64 * n as f64 * k as f64 * it as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        let (m, n) = (1024usize, 1024usize);
        eprintln!("{:>10}  {:>10}  {:>10}", "K", "f32 GF/s", "int8 GF/s");
        for k in [256usize, 1024, 4096, 8192] {
            let a = fill(m * k, 1 + k as u64);
            let b = fill(k * n, 2 + k as u64);
            let (aq, sa) = quantize_i8_symmetric(&a);
            let (bq, sb) = quantize_i8_symmetric(&b);
            let mut c = vec![0f32; m * n];
            let gf_f32 = if is_available() {
                gf(m, k, n, &mut || sme_sgemm(&a, &b, &mut c, m, k, n))
            } else {
                0.0
            };
            let gf_i8 = if is_available_i8() {
                gf(m, k, n, &mut || {
                    sme_qmatmul_i8(&aq, sa, &bq, sb, &mut c, m, k, n)
                })
            } else {
                0.0
            };
            eprintln!("{k:>10}  {gf_f32:>10.1}  {gf_i8:>10.1}");
        }
    }

    /// `qgemv_i8` is exact integer (scale 1.0) and needs no SME — runs anywhere.
    #[test]
    fn qgemv_i8_exact() {
        for (k, n) in [(1, 1), (7, 15), (64, 32), (100, 257), (512, 300)] {
            let x = fill_i8(k, 0x11 + (k * n) as u64);
            let w = fill_i8(k * n, 0x22 + (k * n) as u64);
            let scales = vec![1.0f32; n];
            let mut out = vec![0f32; n];
            qgemv_i8(&x, 1.0, &w, &scales, &mut out, k, n);
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += x[p] as i32 * w[p * n + j] as i32;
                }
                assert_eq!(out[j], acc as f32, "k{k} n{n} j{j}");
            }
        }
    }

    /// Q8_0 SDOT GEMV vs an f32 reference built from the SAME dequantized Q8_0
    /// weights (so the only extra error is activation quant). Uses the real
    /// `rlx_gguf` Q8_0 quantize/dequant to build the wire-format bytes.
    #[test]
    fn qgemv_q8_0_tracks_f32() {
        use rlx_gguf::dequant_q8_0;
        use rlx_gguf::quantize::quantize_q8_0;
        let (k, n) = (256usize, 64usize); // k multiple of 32
        let x = fill(k, 7);
        let mut wbytes: Vec<u8> = Vec::new();
        let mut wref = vec![0f32; n * k];
        for j in 0..n {
            let wrow = fill(k, 100 + j as u64);
            let q = quantize_q8_0(&wrow).unwrap();
            let dq = dequant_q8_0(&q, k).unwrap();
            wref[j * k..j * k + k].copy_from_slice(&dq);
            wbytes.extend_from_slice(&q);
        }
        let mut out = vec![0f32; n];
        qgemv_q8_0(&wbytes, &x, &mut out, k, n);
        let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
        for j in 0..n {
            let mut r = 0f32;
            for p in 0..k {
                r += x[p] * wref[j * k + p];
            }
            dot += out[j] as f64 * r as f64;
            ng += (out[j] as f64).powi(2);
            nw += (r as f64).powi(2);
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        eprintln!("Q8_0 SDOT GEMV k={k} n={n}: cosine={cos:.5}");
        assert!(cos > 0.999, "Q8_0 SDOT GEMV cosine {cos} too low");
    }

    /// `rlx_sdot_i8` contract: k MUST be a positive multiple of 16 (the K
    /// remainder is handled by the `dot_i8` wrapper, exercised via the
    /// `qgemv_*_exact` tests). Verify the vector kernel on valid inputs.
    #[test]
    fn sdot_i8_debug() {
        for k in [16usize, 32, 48, 64, 96, 256, 1024] {
            let x = fill_i8(k, k as u64);
            let w = fill_i8(k, (k + 1) as u64);
            let got = unsafe { super::rlx_sdot_i8(x.as_ptr(), w.as_ptr(), k) };
            let mut exp = 0i32;
            for p in 0..k {
                exp += x[p] as i32 * w[p] as i32;
            }
            assert_eq!(got, exp, "k={k}");
        }
    }

    /// `qgemv_i8_nk` ([n,k] layout, SDOT): exact integer at scale 1.0. Includes K
    /// not a multiple of 16/32 to exercise the tail. Runs anywhere.
    #[test]
    fn qgemv_i8_nk_exact() {
        for (k, n) in [(1, 1), (7, 15), (63, 40), (100, 257), (512, 300), (37, 9)] {
            let x = fill_i8(k, 0x31 + (k * n) as u64);
            let w = fill_i8(n * k, 0x42 + (k * n) as u64); // [n, k]
            let scales = vec![1.0f32; n];
            let mut out = vec![0f32; n];
            qgemv_i8_nk(&x, 1.0, &w, &scales, &mut out, k, n);
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += x[p] as i32 * w[j * k + p] as i32;
                }
                assert_eq!(out[j], acc as f32, "k{k} n{n} j{j}");
            }
        }
    }

    /// Decode-shaped (m=1 GEMV) throughput: is the int8 SMOPA kernel compute-
    /// bound (MOPA wastes 31/32 rows at m=1) or bandwidth-bound (fine)? Compares
    /// int8 SME, scalar int8, and Accelerate f32 GEMV. Also isolates the B-repack
    /// cost, which is the real decode lever (weights are constant across tokens).
    ///   cargo test -p rlx-cpu --features amx-sme sme_gemv_report -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn sme_gemv_report() {
        use std::time::Instant;
        if !is_available_i8() {
            return;
        }
        let bench = |bytes_moved: f64, iters: usize, f: &mut dyn FnMut()| -> (f64, f64) {
            for _ in 0..3 {
                f();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            let s = t0.elapsed().as_secs_f64() / iters as f64;
            (s * 1e6, bytes_moved / s / 1e9) // (µs/call, GB/s of weight read)
        };
        eprintln!(
            "{:>14}  {:>16}  {:>20}  {:>13}",
            "shape (1×k×n)", "[k,n] qgemv µs", "[n,k] SDOT µs (GB/s)", "Accel-f32 µs"
        );
        for (k, n) in [(4096usize, 4096usize), (4096, 11008), (2048, 2048)] {
            let m = 1;
            let xf = fill(m * k, 1 + (k * n) as u64);
            let wf = fill(k * n, 2 + (k * n) as u64);
            let (xq, sx) = quantize_i8_symmetric(&xf);
            let (wq_kn, sw) = quantize_i8_symmetric(&wf); // [k,n]
            let mut wq_nk = vec![0i8; n * k]; // same weights transposed to [n,k]
            for p in 0..k {
                for j in 0..n {
                    wq_nk[j * k + p] = wq_kn[p * n + j];
                }
            }
            let scales_col = vec![sw; n];
            let mut out = vec![0f32; m * n];
            let iters = 50;
            let wbytes = (k * n) as f64; // int8 weight bytes
            let (kn_us, _) = bench(wbytes, iters, &mut || {
                qgemv_i8(&xq, sx, &wq_kn, &scales_col, &mut out, k, n)
            });
            let (nk_us, nk_gbs) = bench(wbytes, iters, &mut || {
                qgemv_i8_nk(&xq, sx, &wq_nk, &scales_col, &mut out, k, n)
            });
            let (acc_us, _) = bench((k * n * 4) as f64, iters, &mut || {
                crate::blas::sgemm(&xf, &wf, &mut out, m, k, n)
            });
            eprintln!(
                "{:>14}  {:>16.1}  {:>12.1} ({:>4.0}GB/s)  {:>13.1}",
                format!("1x{k}x{n}"),
                kn_us,
                nk_us,
                nk_gbs,
                acc_us
            );
        }
    }

    /// bf16 BFMOPA: f32→bf16 downcast, f32 accumulate. Lossy but must track the
    /// f32 product closely (bf16 has 8 mantissa bits).
    #[test]
    fn sme_bf16_tracks_f32() {
        if !is_available_bf16() {
            eprintln!("SME2 bf16 not available; skipping");
            return;
        }
        for (m, k, n) in [(1, 1, 1), (17, 33, 15), (32, 256, 48), (64, 512, 64)] {
            let a = fill(m * k, 3 + (m * k) as u64);
            let b = fill(k * n, 5 + (k * n) as u64);
            let mut got = vec![0f32; m * n];
            sme_sgemm_bf16(&a, &b, &mut got, m, k, n);
            let want = naive(&a, &b, m, k, n);
            let (mut dot, mut ng, mut nw, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for (g, t) in got.iter().zip(&want) {
                dot += (*g as f64) * (*t as f64);
                ng += (*g as f64).powi(2);
                nw += (*t as f64).powi(2);
                num += ((*g - *t) as f64).powi(2);
                den += (*t as f64).powi(2);
            }
            if den > 0.0 {
                let cos = dot / (ng.sqrt() * nw.sqrt());
                let rel = (num / den).sqrt();
                eprintln!("SME bf16 {m}x{k}x{n}: cosine={cos:.5} rel={rel:.4}");
                assert!(cos > 0.999, "cosine {cos} too low");
                assert!(rel < 0.05, "rel err {rel} too high");
            }
        }
    }
}
