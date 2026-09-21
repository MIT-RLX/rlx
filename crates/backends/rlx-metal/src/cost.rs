// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal cost model — analytical kernel selection for GPU.
//!
//! Mirrors rlx-cpu/src/cost.rs. Centralizes all dispatch decisions so
//! kernel selection is data-driven (hardware specs + matrix dims) rather
//! than scattered hardcoded thresholds.

use crate::device::metal_device;
use std::sync::OnceLock;

// `AppleGpuFamily` lives in `crate::occupancy`, which is NOT gated on
// `rlx_metal_host`. The enum is pure string matching with no Metal dependency,
// and keeping it behind the host gate meant a device-free consumer (a cost
// prediction on CI, say) could not name a chip at all.
pub use crate::occupancy::AppleGpuFamily;

/// Variant picked by the cost model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgemmVariant {
    /// MPSMatrixMultiplication — Apple's per-chip-tuned matmul. Wins for
    /// large matmuls (M·K·N above ~16 MFLOPs) where the ~5–20µs objc
    /// bridging cost amortizes against compute time.
    Mps,
    /// 32×32 output per threadgroup; 16 simdgroups cooperate via threadgroup memory.
    /// Best throughput for our hand-rolled path. Requires M%32==K%32==N%32==0.
    Simd4x4,
    /// 64×64 output per threadgroup; 8 simdgroups, each an 8×64 strip (8 accumulators).
    /// ~1.8× Simd4x4 and beats MPS on TALL / short-K aligned shapes (measured).
    /// Requires M%64==0 && N%64==0 && K%8==0 and enough row-tiles for occupancy.
    Simd64,
    /// Split-K 64×64 tile for FAT-K / small-MN (dW=xᵀ·dq): grid adds a Ksplits z-axis
    /// so few output tiles still fill the GPU; partials hardware-atomic-add into a
    /// pre-zeroed C. Beats MPS ~1.5× on the dW shape. Requires 64-align + K%(S*8)==0.
    Simd64SplitK,
    /// 8×8 output per threadgroup. Requires M%8==K%8==N%8==0.
    Simd,
    /// simdgroup tensor units with bounds-checked partial-tile load/store.
    SimdPadded,
    /// Threadgroup-memory-tiled scalar fp32 (16x16 tiles).
    Tiled,
    /// One thread per output element; for very small dims.
    Naive,
}

/// Bridge to the shared dispatch table's variant enum.
///
/// Two enums rather than one because the table crate must stay dependency-free
/// and backend-agnostic; a test below pins the mapping total in both directions
/// so a new variant on either side cannot silently map to the wrong one.
pub(crate) fn to_dispatch(v: SgemmVariant) -> rlx_gpu_dispatch::dispatch::MetalSgemm {
    use rlx_gpu_dispatch::dispatch::MetalSgemm as D;
    match v {
        SgemmVariant::Mps => D::Mps,
        SgemmVariant::Simd4x4 => D::Simd4x4,
        SgemmVariant::Simd64 => D::Simd64,
        SgemmVariant::Simd64SplitK => D::Simd64SplitK,
        SgemmVariant::Simd => D::Simd,
        SgemmVariant::SimdPadded => D::SimdPadded,
        SgemmVariant::Tiled => D::Tiled,
        SgemmVariant::Naive => D::Naive,
    }
}

pub(crate) fn from_dispatch(v: rlx_gpu_dispatch::dispatch::MetalSgemm) -> SgemmVariant {
    use rlx_gpu_dispatch::dispatch::MetalSgemm as D;
    match v {
        D::Mps => SgemmVariant::Mps,
        D::Simd4x4 => SgemmVariant::Simd4x4,
        D::Simd64 => SgemmVariant::Simd64,
        D::Simd64SplitK => SgemmVariant::Simd64SplitK,
        D::Simd => SgemmVariant::Simd,
        D::SimdPadded => SgemmVariant::SimdPadded,
        D::Tiled => SgemmVariant::Tiled,
        D::Naive => SgemmVariant::Naive,
    }
}

/// Split count for `Simd64SplitK`: the largest `S ∈ {32,16,8,4}` with `k%(S*8)==0`
/// and total threadgroups `(m/64)*(n/64)*S ≤ 256` (caps per-output atomic
/// contention while filling the GPU). Returns 0 when no useful split exists.
pub(crate) fn pick_ksplits(m: usize, k: usize, n: usize) -> u32 {
    if !m.is_multiple_of(64) || !n.is_multiple_of(64) {
        return 0;
    }
    let tiles = (m / 64) * (n / 64);
    for s in [32usize, 16, 8, 4] {
        if k.is_multiple_of(s * 8) && tiles * s <= 256 {
            return s as u32;
        }
    }
    0
}

/// Metal hardware model — built once at startup from device properties.
pub struct MetalHwModel {
    pub gpu_family: AppleGpuFamily,
    pub gpu_name: String,
    /// Effective fp32 throughput for simdgroup_matrix sgemm (GFLOP/s).
    pub sgemm_simd_flops: f64,
    /// Effective throughput for 32×32 tiled simdgroup matmul (GFLOP/s).
    pub sgemm_simd_4x4_flops: f64,
    /// Effective throughput for padded simdgroup variant (GFLOP/s).
    pub sgemm_padded_flops: f64,
    /// Effective throughput for scalar tiled fp32 (GFLOP/s).
    pub sgemm_tiled_flops: f64,
    /// Effective GFLOP/s of the causal attention thunk.
    ///
    /// Measured, not derived from sgemm. Attention runs at roughly a third of a
    /// pure-GEMM kernel's rate on Apple silicon because softmax, masking and
    /// materializing the S x S scores are real work no FLOP count sees —
    /// modelling it as "sgemm times a fudge factor" is how this term ended up
    /// 5x wrong.
    pub attention_flops: f64,
    /// Per-kernel dispatch overhead (ns).
    pub dispatch_overhead_ns: f64,
    /// Per-command-buffer commit + wait_until_completed (ns).
    pub roundtrip_overhead_ns: f64,
    /// Threadgroup memory budget per group (bytes).
    pub threadgroup_mem_bytes: usize,
    /// Has unified memory (zero-copy CPU↔GPU).
    pub unified_memory: bool,
    /// Minimum M·K·N (FLOP/2 ≈ MAC count) above which routing through
    /// MPSMatrixMultiplication wins despite per-call objc bridging cost.
    /// Below this we use our in-encoder MSL kernels.
    pub mps_threshold_flop: u64,
}

impl MetalHwModel {
    fn detect() -> Self {
        let dev = metal_device();
        let (name, unified) = match dev {
            Some(d) => (d.name.clone(), d.has_unified_memory),
            None => ("unknown".to_string(), false),
        };
        let family = AppleGpuFamily::from_name(&name);

        // Tier 1 — compile-time platform defaults (per Apple GPU family).
        // These are last-resort fallbacks when calibration cache + measurement
        // are both unavailable.
        let (simd_flops, padded_flops, tiled_flops) = match family {
            AppleGpuFamily::M4 => (600e9, 350e9, 100e9),
            AppleGpuFamily::M3 => (500e9, 300e9, 90e9),
            AppleGpuFamily::M2 => (400e9, 240e9, 75e9),
            AppleGpuFamily::M1Pro => (350e9, 200e9, 65e9),
            AppleGpuFamily::M1 => (200e9, 110e9, 40e9),
            AppleGpuFamily::Unknown => (300e9, 180e9, 60e9),
        };
        let mut simd_4x4_flops = simd_flops * 3.5;
        // Attention default: ~1/3 of the simd4x4 rate. Measured on an M4 Pro at
        // 437-722 GFLOP/s against a 2152 GFLOP/s simd4x4 ceiling (20-34%), on a
        // contended GPU — so the true fraction is at least this. A default, not
        // a calibration; `Calibration::measure` overwrites it with the real
        // number for the attached device.
        let mut attention_flops = simd_4x4_flops / 3.0;
        let mut simd_flops = simd_flops;
        let mut padded_flops = padded_flops;
        let mut tiled_flops = tiled_flops;
        let mut roundtrip_ns = 800_000.0_f64;

        // Tier 2 — only load if a calibration cache file already exists.
        // We never measure at startup — that's done by `cargo run --example
        // metal_calibrate` (or via `Calibration::measure()` directly).
        // The cache file is keyed by GPU registry ID, so it's portable across runs.
        let dev_id = dev.map(|d| d.registry_id).unwrap_or(0);
        if let Some(cal) = crate::calibrate::Calibration::load(dev_id) {
            simd_4x4_flops = cal.sgemm_simd_4x4_flops;
            // A cache written before this field existed deserialises it as 0.0;
            // keep the arch default rather than dividing by zero.
            if cal.attention_flops > 0.0 {
                attention_flops = cal.attention_flops;
            }
            simd_flops = cal.sgemm_simd_flops;
            padded_flops = cal.sgemm_padded_flops;
            tiled_flops = cal.sgemm_tiled_flops;
            roundtrip_ns = cal.roundtrip_overhead_ns;
        }

        // MPS pays ~5–20µs objc/encoder overhead; we want compute to be at
        // least ~5× that to net win. With our 32×32 simd kernel running near
        // 1 TFLOPS, that's M·K·N ≥ ~25M FLOPs (≈ 256×256×768). Use 16M as a
        // conservative cutoff; tune with RLX_MPS_THRESHOLD_FLOP env var.
        let mps_threshold_flop = rlx_ir::env::var("RLX_MPS_THRESHOLD_FLOP")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(16_000_000);

        Self {
            gpu_family: family,
            gpu_name: name,
            sgemm_simd_flops: simd_flops,
            sgemm_simd_4x4_flops: simd_4x4_flops,
            sgemm_padded_flops: padded_flops,
            sgemm_tiled_flops: tiled_flops,
            attention_flops,
            dispatch_overhead_ns: 8_000.0,
            roundtrip_overhead_ns: roundtrip_ns,
            threadgroup_mem_bytes: 32 * 1024,
            unified_memory: unified,
            mps_threshold_flop,
        }
    }

    // ── Dispatch decisions ──────────────────────────────────────────

    /// Split count for `Simd64SplitK` (0 = don't split). Largest S in {32,16,8,4}
    /// with `k % (S*8) == 0` and total threadgroups `(m/64)*(n/64)*S <= 256` (caps
    /// per-output atomic contention while filling the GPU). See `pick_ksplits`.
    pub fn ksplits(&self, m: usize, k: usize, n: usize) -> u32 {
        pick_ksplits(m, k, n)
    }

    /// Pick the best sgemm variant for these dimensions.
    /// Higher-throughput variants have stricter alignment requirements.
    ///
    /// Three layers, in increasing authority:
    ///
    /// 1. [`Self::pick_sgemm_default`] — the hand-written cascade below, which
    ///    is what this function has always been.
    /// 2. A **measured override** from the shared dispatch table, keyed by
    ///    `(gpu family, matmul, shape bucket)`. Accepted only if the variant is
    ///    also eligible for this shape and this device — see
    ///    [`Self::sgemm_eligible`]. The table tunes; it never relaxes a
    ///    correctness constraint.
    /// 3. `RLX_METAL_SGEMM_VARIANT` — an operator pinning a variant for an A/B,
    ///    which outranks everything.
    pub fn pick_sgemm(&self, m: usize, k: usize, n: usize) -> SgemmVariant {
        let default = self.pick_sgemm_default(m, k, n);
        if let Some(forced) = sgemm_variant_override() {
            // Held to the SAME rule as the tuning cache below: an operator
            // pinning a variant outranks the cost model, but not a correctness
            // constraint. `Simd4x4` at m%32!=0 writes past the end of C, and an
            // A/B is not a licence to corrupt memory.
            if self.sgemm_eligible(forced, m, k, n) {
                return forced;
            }
            // Loud, not silent. A pin that quietly does nothing turns an A/B
            // into a comparison of the default against itself — which is
            // exactly how a bogus "both paths are identical" result gets made.
            warn_once(&format!(
                "rlx-metal: RLX_METAL_SGEMM_VARIANT pinned {forced:?}, but it is NOT eligible \n\
                 at m={m} k={k} n={n} (alignment/occupancy) — running {default:?} instead.\n\
                 Any A/B at this shape is comparing the default against itself."
            ));
            return default;
        }
        let Some(want) =
            rlx_gpu_dispatch::dispatch::resolve_matmul(crate::tuning::gpu_arch(), m, k, n)
                .as_metal_sgemm()
        else {
            return default;
        };
        let want = from_dispatch(want);
        if want == default {
            return default;
        }
        if self.sgemm_eligible(want, m, k, n) {
            want
        } else {
            // A stale cache (tuned before an alignment rule tightened, or copied
            // from another machine) must not be able to route a shape into a
            // kernel that would overrun C. Fall back and say so.
            if rlx_ir::env::flag("RLX_VERBOSE") {
                eprintln!(
                    "rlx-metal: tuning cache picks {want:?} for m={m} k={k} n={n} \
                     but it is not eligible there — using {default:?}"
                );
            }
            default
        }
    }

    /// Can this variant legally run at this shape on this device?
    ///
    /// Two kinds of constraint, both hard:
    ///
    /// * **Shape** — alignment the kernel assumes. `Simd4x4` writes a full 32×32
    ///   tile with no bottom-row mask, so at `m % 32 != 0` it overruns C and
    ///   corrupts the next tensor in the arena (this actually happened: Gemma 4
    ///   E2B prefill at m=16 wrote 32×2048 floats into a 16×2048 buffer). Shared
    ///   with the table via `MetalSgemm::shape_eligible` so the rule has one
    ///   home rather than one copy per caller.
    /// * **Device** — MPS has to be present; `Simd64SplitK` needs a usable split.
    pub fn sgemm_eligible(&self, v: SgemmVariant, m: usize, k: usize, n: usize) -> bool {
        if !to_dispatch(v).shape_eligible(m, k, n) {
            return false;
        }
        match v {
            SgemmVariant::Mps => crate::mps_blas::mps_supports_matmul(),
            SgemmVariant::Simd64SplitK => pick_ksplits(m, k, n) >= 4,
            _ => true,
        }
    }

    /// Does the 64x64 big-tile kernel have enough output tiles to fill the GPU?
    ///
    /// `Simd64`'s measured ~1.8x over `Simd4x4` holds only where there are
    /// threadgroups enough to hide latency. Below this, fat-K/small-MN shapes
    /// like `dW = xᵀ·dq` leave most of the machine idle, which is why the
    /// cascade declines the variant there — and why the cost model must not go
    /// on crediting it the speedup.
    fn simd64_has_occupancy(m: usize, n: usize) -> bool {
        (m / 64) * (n / 64) >= 32
    }

    /// Modelled throughput (FLOP/s) of one variant at one shape.
    ///
    /// Shared so the cost model, the cascade's justification and any test all
    /// read the same numbers. Restating them was how `Simd64` came to be scored
    /// 1.8x at shapes the cascade deliberately refuses it: the constant is
    /// conditional on occupancy, and a copy of it dropped the condition.
    pub fn sgemm_variant_flops(&self, v: SgemmVariant, m: usize, _k: usize, n: usize) -> f64 {
        match v {
            // MPS hits roughly 1.5-2.5x our hand-rolled simd_4x4 throughput
            // on M3/M4 once it's past the bridging-cost threshold.
            SgemmVariant::Mps => self.sgemm_simd_4x4_flops * 2.0,
            SgemmVariant::Simd64 => {
                if Self::simd64_has_occupancy(m, n) {
                    // ~1.8x Simd4x4 on the tall aligned shapes it is gated to.
                    self.sgemm_simd_4x4_flops * 1.8
                } else {
                    // Occupancy-starved: no big-tile advantage left. Modelled at
                    // parity rather than as a penalty, which is the conservative
                    // reading — it says "no reason to prefer it", which is what
                    // the cascade's gate encodes, without inventing a slowdown
                    // nobody measured.
                    self.sgemm_simd_4x4_flops
                }
            }
            // Fat-K split beats MPS ~1.5x (~3x Simd4x4) on the dW shape (measured).
            SgemmVariant::Simd64SplitK => self.sgemm_simd_4x4_flops * 3.0,
            SgemmVariant::Simd4x4 => self.sgemm_simd_4x4_flops,
            SgemmVariant::Simd => self.sgemm_simd_flops,
            SgemmVariant::SimdPadded => self.sgemm_padded_flops,
            SgemmVariant::Tiled => self.sgemm_tiled_flops,
            SgemmVariant::Naive => self.sgemm_tiled_flops * 0.3,
        }
    }

    /// The hand-written cascade — the compile-time default the table overrides.
    pub fn pick_sgemm_default(&self, m: usize, k: usize, n: usize) -> SgemmVariant {
        let aligned_8 = m.is_multiple_of(8) && k.is_multiple_of(8) && n.is_multiple_of(8);

        // MPS pays encoder end + objc bridging per call. Past the threshold,
        // the measured chip-tuned path wins for the large prefill and large-
        // decode matmuls we care about here, so use it by default unless the
        // caller disables it. `RLX_METAL_SGEMM_MPS=1` still forces it on for
        // smaller shapes, and `RLX_DISABLE_MPS=1` remains the opt-out.
        let mps_enabled = rlx_ir::env::flag("RLX_METAL_SGEMM_MPS");
        let mps_disabled = rlx_ir::env::var("RLX_DISABLE_MPS")
            .map(|v| v == "1")
            .unwrap_or(false);
        let flop = (m as u64) * (k as u64) * (n as u64);
        if !mps_disabled
            && crate::mps_blas::mps_supports_matmul()
            && (mps_enabled || flop >= self.mps_threshold_flop)
        {
            return SgemmVariant::Mps;
        }

        // 64×64 big-tile: ~1.8× Simd4x4 (and beats MPS) on TALL aligned shapes
        // (measured — MPS's async-copy pipeline can't amortize at short K).
        // Default-on for qualifying shapes; the occupancy gate ((m/64)*(n/64)>=32)
        // excludes fat-K/small-MN like dW = xᵀ·dq (m=192) where too few threadgroups
        // underutilize the GPU. No partial-tile handling → strict 64/8 alignment.
        // Opt out: RLX_METAL_NO_SGEMM64. Measured: transformer forward ~11% faster,
        // bit-exact. See cse_and_backward_timing.
        if !rlx_ir::env::flag("RLX_METAL_NO_SGEMM64")
            && m.is_multiple_of(64)
            && n.is_multiple_of(64)
            && k.is_multiple_of(8)
            && Self::simd64_has_occupancy(m, n)
        {
            return SgemmVariant::Simd64;
        }

        // Split-K big-tile for FAT-K / small-MN (too few output tiles for Simd64,
        // but K large enough to parallelize). Opt-in (RLX_METAL_SGEMM_SPLITK) while
        // the atomic-accumulate path is validated. Only when a good split exists.
        if rlx_ir::env::flag("RLX_METAL_SGEMM_SPLITK")
            && m.is_multiple_of(64)
            && n.is_multiple_of(64)
            && !Self::simd64_has_occupancy(m, n)
            && pick_ksplits(m, k, n) >= 4
        {
            return SgemmVariant::Simd64SplitK;
        }

        // Tiny-n m=1 GEMVs (e.g. the fused GDN ssm_alpha/beta [K → 2·n_v_heads]
        // projections, n=32): no fast GEMV kernel applies (splitk/kpart need
        // n>=64) and the fallthrough (Naive/SimdPadded) is occupancy-starved —
        // 1 threadgroup, serial K-loop → ~0.33 ms each on qwen3.5. MPS is
        // ~5-10× faster for these. Measured +25% on qwen3.5-0.8B decode.
        if m == 1 && n < 64 && !mps_disabled && crate::mps_blas::mps_supports_matmul() {
            return SgemmVariant::Mps;
        }

        if k.is_multiple_of(32) && n.is_multiple_of(32) && m.is_multiple_of(32) {
            // simd4x4 dispatches an integer number of 32×32 tiles. The MSL
            // kernel writes 32 rows × 32 cols per threadgroup unconditionally
            // — when m is NOT a multiple of 32 the last threadgroup
            // overflows C past row m-1, stomping whatever tensor follows in
            // the arena (verified by the all-zeros Q output on Gemma 4 E2B
            // prefill bucket=16: m=16, k=1536, n=2048 → 32×2048 = 65536
            // floats written into a 16×2048 = 32768-float C buffer, the
            // next 32768 floats of arena got corrupted). Fall back to Simd
            // for sub-32 m until the kernel learns to mask the bottom rows.
            SgemmVariant::Simd4x4
        } else if m < 32 {
            // Decode / small-batch: Naive is correct but ~3× slower than
            // simdgroup on Zonos CFG (m=2). Prefer SimdPadded for large
            // projections; keep Naive for tiny dims or when
            // RLX_METAL_SGEMM_PRECISE=1 (accumulator-parity debug).
            // Fall THROUGH to the general cascade when the large-projection
            // gate misses, rather than dropping straight to `Naive`.
            //
            // The old `else { Naive }` fired whenever any of `k >= 256`,
            // `n >= 256` or `k % 8 == 0` failed — including on shapes where
            // `Simd`, `SimdPadded` or `Tiled` were all eligible and the cost
            // model scores them up to 13.5x cheaper. A sweep found 108 such
            // shapes (`the_cascade_never_picks_a_variant_its_own_cost_model_beats`);
            // 16x128x3072 is the worst, where wide `n` clears the "large
            // projection" intent but `k = 128` trips the `k >= 256` gate.
            //
            // These are the same conditions the `m >= 32` path already uses, so
            // this routes small-m shapes through the branch that was always
            // there — it does not enable any variant for a shape its own
            // eligibility rules reject. `Simd4x4` stays excluded because the
            // outer `m % 32` test already failed, which is the row-overrun
            // constraint documented above.
            let precise = rlx_ir::env::flag("RLX_METAL_SGEMM_PRECISE");
            if precise {
                // Accumulator-parity debug: keep the reference path.
                SgemmVariant::Naive
            } else if (2..32).contains(&m)
                && !m.is_multiple_of(8)
                && k == n
                && k >= 256
                && !mps_disabled
                && !rlx_ir::env::flag("RLX_METAL_NO_SMALL_M_MPS")
                && crate::mps_blas::mps_supports_matmul()
            {
                // Small, 8-UNALIGNED m against a large SQUARE operator.
                // `aligned_8` below needs `m % 8 == 0`, so `Simd` is out and
                // these land on `SimdPadded`, which pads m up to a simdgroup
                // and wastes most of the tile. MPS wins despite its bridging
                // cost — same reasoning as the `m == 1 && n < 64` case above,
                // one rung up in n.
                //
                // `k == n` is the discriminator, and it is not arbitrary: a
                // square operator is a rotation/basis change, never a
                // transformer projection (those are rectangular). Concretely
                // this is `prism.hadamard`'s rotation on Ternary Bonsai 2 —
                // the `[-1, 1024]` block view makes m = width/1024, i.e. 5, 6
                // or 17, against k = n = 1024, ~190 times a token. Without the
                // `k == n` guard the rule also caught rectangular shapes in
                // Qwen3.8-27B and cost it 1.6%.
                //
                // MEASURED on M4 Pro, arms alternated inside each round
                // (median over ~79 decode steps, 3 rounds): Bonsai-2 124.2 ->
                // 117.6 ms/token, won every round. Deliberately excludes
                // m == 1 — the blanket `RLX_METAL_SGEMM_MPS=1`, which does
                // catch m == 1, measured 1.4% SLOWER on Qwen3.8-27B-Q3_K_S
                // (228.0 -> 231.3) and a wash on Bonsai-1, so this is a real
                // shape class and not a general "MPS is better" result.
                // Opt out: RLX_METAL_NO_SMALL_M_MPS=1.
                if rlx_ir::env::flag("RLX_METAL_SMALL_M_MPS_TRACE") {
                    warn_once(&format!("[small_m_mps] m={m} k={k} n={n}"));
                }
                SgemmVariant::Mps
            } else if aligned_8 && m >= 8 && n >= 8 {
                // `Simd` goes FIRST, exactly as in the `m >= 32` arm below.
                //
                // It used to sit behind a `k >= 256 && n >= 256` rule that
                // returned `SimdPadded`, which shadowed it on every fully
                // 8-aligned large projection — so the shape paid for padding it
                // did not need. Caught by
                // `the_cascade_never_picks_a_variant_its_own_cost_model_beats`
                // at 8x512x3072 (5 shapes in the sweep).
                //
                // MEASURED on M4 Pro, not just modelled: pinning each variant
                // via `RLX_METAL_SGEMM_VARIANT` and alternating the two arms
                // inside each timing round, `Simd` won all six moved shapes in
                // both of two runs — 1.38-1.51x at m=8, 1.03-1.12x at m=16,
                // 1.21-1.22x at m=24 (min-of-21, both arms checked against a
                // CPU reference). The flat ~1.7x the cost model assumes on
                // every arch overstates the margin at m=16, but never gets the
                // sign wrong. Interleaving mattered: running one arm fully and
                // then the other put thermal/load drift on the variant axis and
                // produced a nonsense ordering (8x2048 "faster" than 8x1024 on
                // the same kernel).
                //
                // That rule was written when this arm's `else` was `Naive`, so
                // its job was escaping `Naive`, not beating `Simd`. Once the
                // arm gained the general fallthrough it was both wrong and
                // redundant: `k >= 256 && n >= 256 && k % 8 == 0` implies
                // `k % 8 == 0 && n >= 8`, so the `SimdPadded` rule below
                // already covers every shape it used to claim.
                SgemmVariant::Simd
            } else if k.is_multiple_of(8) && n >= 8 {
                SgemmVariant::SimdPadded
            } else if m >= 16 && n >= 16 {
                SgemmVariant::Tiled
            } else {
                SgemmVariant::Naive
            }
        } else if aligned_8 && m >= 8 && n >= 8 {
            SgemmVariant::Simd
        } else if k.is_multiple_of(8) && n >= 8 && m >= 1 {
            SgemmVariant::SimdPadded
        } else if m >= 16 && n >= 16 {
            SgemmVariant::Tiled
        } else {
            SgemmVariant::Naive
        }
    }

    /// Estimate execution time in nanoseconds for an sgemm of given dims.
    pub fn sgemm_cost_ns(&self, m: usize, k: usize, n: usize) -> f64 {
        let flops = 2.0 * m as f64 * k as f64 * n as f64;
        let throughput = self.sgemm_variant_flops(self.pick_sgemm(m, k, n), m, k, n);
        // FLOPs / (FLOP/s) is SECONDS. The `* 1e9` is what makes the name true.
        //
        // Without it this returned ~0.001 for a 1024^3 GEMM and then added
        // `dispatch_overhead_ns` (~18_000), so the compute term was swamped by a
        // constant and every estimate was effectively shape-blind. Nothing
        // consumed it — `estimate_transformer_forward_ns` is the only caller and
        // has none of its own — so this was a landmine rather than a live
        // defect, which is exactly the kind that survives.
        let compute_ns = flops / throughput * 1e9;
        compute_ns + self.dispatch_overhead_ns
    }

    /// Should we fuse matmul + bias + activation into a single kernel?
    /// Yes — saves dispatch overhead. Only skip if any kernel is unsupported.
    pub fn prefer_fused_matmul_bias(&self, _m: usize, _k: usize, _n: usize) -> bool {
        // Always fuse — fused kernels never lose compared to separate calls.
        true
    }

    /// Can the entire transformer layer's intermediates fit in threadgroup memory?
    /// If yes, a monolithic FusedTransformerLayer shader is viable.
    pub fn fits_threadgroup_mem(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        intermediate: usize,
    ) -> bool {
        // Per-row stack: hidden * 4 bytes (residual) + 3*hidden*4 (qkv) + intermediate*4 (ffn)
        // Per row × batch×seq rows
        let m = batch * seq;
        let bytes = m * (hidden + 3 * hidden + hidden + intermediate) * 4;
        bytes <= self.threadgroup_mem_bytes
    }

    /// Estimate total forward time for a transformer of given shape.
    /// Used to predict batch-size crossover where Metal beats CPU.
    pub fn estimate_transformer_forward_ns(
        &self,
        batch: usize,
        seq: usize,
        hidden: usize,
        intermediate: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> f64 {
        let m = batch * seq;
        // `num_heads` is genuinely not needed: attention's FLOP count depends on
        // heads only through `num_heads * head_dim == hidden`, which is already
        // a parameter. Kept in the signature for callers.
        let _ = num_heads;

        // Per layer: QKV proj + out proj + FC1 + FC2 + element-wise ops
        let qkv = self.sgemm_cost_ns(m, hidden, 3 * hidden);
        let out = self.sgemm_cost_ns(m, hidden, hidden);
        let fc1 = self.sgemm_cost_ns(m, hidden, intermediate);
        let fc2 = self.sgemm_cost_ns(m, intermediate, hidden);
        // Attention. Three things were wrong here and each moved the answer by
        // orders of magnitude:
        //
        //  * the FLOP count omitted `batch` and the factor 4. QK^T and PV are
        //    each S^2*D MACs per head = 4*S^2*D FLOPs per head; summed over
        //    heads (H*D == hidden) the layer total is 4*batch*seq^2*hidden.
        //  * it divided by `sgemm_simd_flops` — the 8x8 variant, which
        //    calibrates at 61-66 GFLOP/s on an M4 Pro — instead of
        //    `sgemm_simd_4x4_flops`, the path the cascade actually picks, at
        //    2152-2617 GFLOP/s. A ~33x error.
        //  * it inherited the seconds-vs-nanoseconds bug above.
        //
        // Attention gets its OWN measured throughput rather than sgemm scaled by
        // a fudge factor. The fudge factor was 0.06 and the truth is ~0.3 — a 5x
        // error, because it had been derived from a measurement that timed the
        // whole run of an attention-only graph and so was dominated by Q/K/V
        // upload rather than by the kernel. `Calibration::measure` now times the
        // attention thunk itself.
        let attn_flops = 4.0 * batch as f64 * (seq * seq) as f64 * hidden as f64;
        let attn = attn_flops / self.attention_flops * 1e9 + self.dispatch_overhead_ns;
        // Element-wise + LN: dominated by dispatch overhead at small sizes.
        let elem = 4.0 * self.dispatch_overhead_ns;

        let per_layer = qkv + out + fc1 + fc2 + attn + elem;
        per_layer * num_layers as f64 + self.roundtrip_overhead_ns
    }
}

/// Force a specific sgemm kernel for A/B tuning.
///
/// Env vars consulted (in order; first match wins):
///
/// - **`RLX_METAL_SGEMM_VARIANT`** — explicit variant by name. Accepts
///   see [`SGEMM_PIN_NAMES`] for every accepted spelling. An unrecognized name
///   is reported on stderr and ignored, rather than silently running the
///   default. The pin outranks the cost model and the tuning cache, but NOT
///   [`MetalHwModel::sgemm_eligible`] — a pin that would overrun `C` is refused
///   and reported.
/// - **`RLX_METAL_PRECISE`** — when set to `1` / `true`, forces the
///   scalar fp32 `naive` variant for every matmul. Apple Silicon's
///   `simdgroup_float8x8` tensor units use reduced-precision internal
///   accumulators (~fp16 class), which is fine for production
///   inference but produces ~1e-1 absolute error vs CPU on small
///   parity tests. Set this for precision-critical work; leave unset
///   for production where the 10–100× throughput of the SIMD path
///   wins.
/// Every accepted spelling, and the variant it selects.
///
/// `Simd64` and `Simd64SplitK` were missing here, which made the two variants
/// carrying explicit measured claims ("beats MPS on TALL / short-K",
/// "beats MPS ~1.5x on the dW shape") the only two that could not be A/B'd
/// against MPS. `sgemm_pin_names_cover_every_variant` keeps that from
/// recurring.
pub(crate) const SGEMM_PIN_NAMES: &[(&str, SgemmVariant)] = &[
    ("mps", SgemmVariant::Mps),
    ("simd4x4", SgemmVariant::Simd4x4),
    ("simd_4x4", SgemmVariant::Simd4x4),
    ("4x4", SgemmVariant::Simd4x4),
    ("simd64", SgemmVariant::Simd64),
    ("simd_64", SgemmVariant::Simd64),
    ("64x64", SgemmVariant::Simd64),
    ("simd64splitk", SgemmVariant::Simd64SplitK),
    ("simd64_splitk", SgemmVariant::Simd64SplitK),
    ("splitk", SgemmVariant::Simd64SplitK),
    ("simd", SgemmVariant::Simd),
    ("simd8", SgemmVariant::Simd),
    ("simd_8", SgemmVariant::Simd),
    ("padded", SgemmVariant::SimdPadded),
    ("simd_padded", SgemmVariant::SimdPadded),
    ("simdpadded", SgemmVariant::SimdPadded),
    ("tiled", SgemmVariant::Tiled),
    ("naive", SgemmVariant::Naive),
];

/// Print `msg` to stderr the first time it is seen.
///
/// `pick_sgemm` runs per matmul, so an unconditional `eprintln!` would bury the
/// warning in its own repetitions. Only reached when the env var is set AND
/// something is wrong with it, so the lock is never touched on a normal run.
fn warn_once(msg: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut g) = seen.lock()
        && g.insert(msg.to_string())
    {
        eprintln!("{msg}");
    }
}

pub(crate) fn sgemm_variant_override() -> Option<SgemmVariant> {
    if let Some(raw) = rlx_ir::env::var("RLX_METAL_SGEMM_VARIANT") {
        let key = raw.to_ascii_lowercase();
        if let Some((_, v)) = SGEMM_PIN_NAMES.iter().find(|(name, _)| *name == key) {
            return Some(*v);
        }
        // A typo used to fall through to the default in silence, so the
        // operator measured the default twice and read it as "no difference".
        warn_once(&format!(
            "rlx-metal: RLX_METAL_SGEMM_VARIANT={raw:?} is not a known variant — IGNORED, \n\
             running the default. Accepted: {}",
            SGEMM_PIN_NAMES
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    if let Some(raw) = rlx_ir::env::var("RLX_METAL_PRECISE") {
        match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => return Some(SgemmVariant::Naive),
            _ => {}
        }
    }
    None
}

/// Global hardware model singleton.
pub fn hw_model() -> &'static MetalHwModel {
    static MODEL: OnceLock<MetalHwModel> = OnceLock::new();
    MODEL.get_or_init(MetalHwModel::detect)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose numbers are known, so cost assertions are about the
    /// arithmetic rather than about whatever this machine calibrated to.
    fn model_for_cost_tests() -> MetalHwModel {
        let mut m = MetalHwModel::detect();
        // ALL four throughputs, not just the fast one. `sgemm_cost_ns` routes
        // through `pick_sgemm`, so leaving `tiled` at whatever this machine
        // calibrated (22 GFLOP/s here) let the GEMM terms pick a slow variant
        // and dominate — which made an attention-scaling assertion measure the
        // linear GEMM terms instead.
        m.sgemm_simd_4x4_flops = 2_000e9; // ~a real M4 Pro
        m.sgemm_simd_flops = 60e9;
        m.sgemm_padded_flops = 2_000e9;
        m.sgemm_tiled_flops = 2_000e9;
        m.attention_flops = 700e9; // ~1/3 of the sgemm rate, as measured
        m.dispatch_overhead_ns = 20_000.0;
        m
    }

    #[test]
    fn probe_eligibility_of_affected_shapes() {
        with_clean_dispatch(|| {
            let hw = MetalHwModel::detect();
            for (m, k, n) in [(8usize, 16usize, 16usize), (6, 7, 7), (16, 128, 3072)] {
                let elig: Vec<&str> = [
                    (SgemmVariant::Simd, "Simd"),
                    (SgemmVariant::SimdPadded, "SimdPadded"),
                    (SgemmVariant::Tiled, "Tiled"),
                ]
                .into_iter()
                .filter(|(v, _)| hw.sgemm_eligible(*v, m, k, n))
                .map(|(_, s)| s)
                .collect();
                eprintln!(
                    "{m}x{k}x{n}: eligible={elig:?} picked={:?}",
                    hw.pick_sgemm(m, k, n)
                );
            }
        });
    }

    /// **The cascade must never pick a slower variant than one that was
    /// eligible.**
    ///
    /// This is not a style rule, it is the routing question `reference_perf`
    /// can only answer for the nine shapes it measures. Calibration on an M4
    /// Pro puts the variants 130x apart:
    ///
    /// | variant | measured |
    /// |---|---|
    /// | `Simd4x4` | 2116 GFLOP/s |
    /// | `Simd` | 52 |
    /// | `SimdPadded` | 66 |
    /// | `Tiled` | 16 |
    ///
    /// so selecting `Tiled` where `Simd4x4` was eligible is a ~130x loss, and
    /// nothing device-free was checking for it. Sweeps a shape grid and fails
    /// on any selection the model itself scores as beatable.
    ///
    /// Deliberately compares against `sgemm_cost_ns`, i.e. the model's own
    /// opinion — a cascade that disagrees with the cost model is at minimum
    /// inconsistent, whichever one is right.
    #[test]
    fn the_cascade_never_picks_a_variant_its_own_cost_model_beats() {
        with_clean_dispatch(|| {
            let m = MetalHwModel::detect();
            let candidates = [
                SgemmVariant::Simd4x4,
                SgemmVariant::Simd64,
                SgemmVariant::Simd,
                SgemmVariant::SimdPadded,
                SgemmVariant::Tiled,
            ];
            let mut worst: Option<(usize, usize, usize, SgemmVariant, SgemmVariant, f64)> = None;
            let mut count = 0usize;
            let mut by_pick: std::collections::BTreeMap<String, usize> = Default::default();

            for &mm in &[1usize, 2, 8, 16, 32, 64, 128, 256, 512, 1024] {
                for &kk in &[64usize, 128, 512, 1024, 2048] {
                    for &nn in &[64usize, 128, 512, 1024, 3072] {
                        let picked = m.pick_sgemm(mm, kk, nn);
                        // MPS is measured separately and modelled as a multiple
                        // of simd4x4; it is not a fallback and is excluded.
                        if matches!(picked, SgemmVariant::Mps) {
                            continue;
                        }
                        let picked_cost = m.sgemm_cost_ns(mm, kk, nn);
                        for &c in &candidates {
                            if c == picked || !m.sgemm_eligible(c, mm, kk, nn) {
                                continue;
                            }
                            // Cost of the alternative at the same shape, from
                            // the model itself. Restating the throughput table
                            // here is what let this test disagree with the
                            // cascade for real: it credited `Simd64` a flat 1.8x
                            // at shapes whose occupancy the cascade knows the
                            // speedup does not survive, and then reported the
                            // cascade as the thing at fault.
                            let flops = 2.0 * mm as f64 * kk as f64 * nn as f64;
                            let tput = m.sgemm_variant_flops(c, mm, kk, nn);
                            let alt = flops / tput * 1e9 + m.dispatch_overhead_ns;
                            let ratio = picked_cost / alt;
                            // 1.5x of slack: the model is an estimate and the
                            // cascade encodes correctness constraints (row
                            // overrun, alignment) the cost model does not see.
                            if ratio > 1.5 {
                                count += 1;
                                *by_pick.entry(format!("{picked:?} -> {c:?}")).or_default() += 1;
                                if worst.as_ref().is_none_or(|w| ratio > w.5) {
                                    worst = Some((mm, kk, nn, picked, c, ratio));
                                }
                            }
                        }
                    }
                }
            }

            for (pair, n) in &by_pick {
                eprintln!("  {n:4} shape(s): {pair}");
            }
            if let Some((mm, kk, nn, picked, better, ratio)) = worst {
                panic!(
                    "cascade picked {picked:?} at {mm}x{kk}x{nn} where {better:?} was \
                     eligible and the model scores it {ratio:.1}x cheaper ({} such \
                     shape(s) in the sweep)",
                    count
                );
            }
        });
    }

    /// **The units.** `sgemm_cost_ns` returns NANOSECONDS.
    ///
    /// It used to compute `flops / throughput` — which is seconds — and then add
    /// `dispatch_overhead_ns`. A 1024^3 GEMM came out as `0.001 + 20_000`, so the
    /// compute term was 1e9x too small and every estimate collapsed to a
    /// constant. Nothing consumed it, which is why it survived.
    #[test]
    fn sgemm_cost_is_in_nanoseconds_not_seconds() {
        // `pick_sgemm` reads process-global state and cargo runs these on
        // parallel threads; without the lock a sibling test's env flip changes
        // which variant is selected mid-assertion. Passed alone, failed in the
        // suite.
        with_clean_dispatch(|| {
            let m = model_for_cost_tests();
            let ns = m.sgemm_cost_ns(1024, 1024, 1024);
            // 2*1024^3 = 2.15 GFLOP at 2 TFLOP/s = ~1.07 ms = ~1.07e6 ns. Allow a
            // wide band: the point is the ORDER, not the value.
            assert!(
                (1e5..1e8).contains(&ns),
                "1024^3 should cost ~1e6 ns; got {ns:.3} — units regression"
            );
            // And the compute term must actually dominate dispatch overhead at this
            // size, which is the property the bug destroyed.
            assert!(
                ns > 10.0 * m.dispatch_overhead_ns,
                "compute term is being swamped by dispatch overhead again"
            );
        });
    }

    /// Cost must grow with work. Under the units bug every shape returned
    /// ~`dispatch_overhead_ns`, so this would have failed.
    #[test]
    fn sgemm_cost_scales_with_work() {
        // `pick_sgemm` reads process-global state and cargo runs these on
        // parallel threads; without the lock a sibling test's env flip changes
        // which variant is selected mid-assertion. Passed alone, failed in the
        // suite.
        with_clean_dispatch(|| {
            let m = model_for_cost_tests();
            // Subtract the constant so this tests the COMPUTE term. Comparing the
            // totals conflates a real 64x with a fixed 20 us that inflates the small
            // case (19.6x observed) — an assertion that would drift with any
            // overhead change rather than with the arithmetic it is about.
            let small = m.sgemm_cost_ns(256, 256, 256) - m.dispatch_overhead_ns;
            let big = m.sgemm_cost_ns(1024, 1024, 1024) - m.dispatch_overhead_ns;
            let ratio = big / small;
            assert!(
                (60.0..70.0).contains(&ratio),
                "64x the FLOPs should be ~64x the compute: got {ratio:.1}x \
                 ({small:.0} -> {big:.0} ns)"
            );
        });
    }

    /// Attention is O(seq^2), and the estimate must show it.
    ///
    /// The old term also dropped `batch` and a factor of 4, and divided by the
    /// 8x8 variant's throughput (~33x too slow). Doubling seq must roughly
    /// quadruple the attention contribution.
    #[test]
    fn attention_cost_is_quadratic_in_sequence() {
        // `pick_sgemm` reads process-global state and cargo runs these on
        // parallel threads; without the lock a sibling test's env flip changes
        // which variant is selected mid-assertion. Passed alone, failed in the
        // suite.
        with_clean_dispatch(|| {
            let m = model_for_cost_tests();
            // Isolate attention by differencing two layer estimates that differ
            // only in seq, at a hidden size where the GEMM terms stay linear.
            let at = |seq: usize| m.estimate_transformer_forward_ns(1, seq, 1024, 4096, 16, 1);
            let (s1, s2, s4) = (at(512), at(1024), at(2048));
            let d1 = s2 - s1;
            let d2 = s4 - s2;
            assert!(
                d2 > 2.5 * d1,
                "attention should grow superlinearly in seq: deltas {d1:.0} then {d2:.0}"
            );
        });
    }

    /// Batch must be in the attention term. It was silently absent.
    #[test]
    fn attention_cost_accounts_for_batch() {
        // `pick_sgemm` reads process-global state and cargo runs these on
        // parallel threads; without the lock a sibling test's env flip changes
        // which variant is selected mid-assertion. Passed alone, failed in the
        // suite.
        with_clean_dispatch(|| {
            let m = model_for_cost_tests();
            let b1 = m.estimate_transformer_forward_ns(1, 1024, 1024, 4096, 16, 1);
            let b4 = m.estimate_transformer_forward_ns(4, 1024, 1024, 4096, 16, 1);
            assert!(
                b4 > 3.0 * b1,
                "4x the batch should cost ~4x: {b1:.0} -> {b4:.0}"
            );
        });
    }

    /// `pick_sgemm` reads process-global state on two axes — the `RLX_*` env
    /// vars and the shared dispatch-override table — and cargo runs the tests in
    /// this binary on parallel threads. Without this, one test unsetting
    /// `RLX_DISABLE_MPS` flips another test's expected variant mid-run. Same
    /// guard, same reason, as `rlx-cpu`'s `DISPATCH_TEST_LOCK`.
    static SGEMM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Run `f` with exclusive access to that global state, and leave it clean
    /// for the next test whichever way `f` exits.
    fn with_clean_dispatch(f: impl FnOnce()) {
        let _g = SGEMM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        rlx_gpu_dispatch::dispatch::clear_overrides();
        rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        rlx_ir::env::unset("RLX_METAL_SGEMM_PRECISE");
        rlx_ir::env::unset("RLX_DISABLE_MPS");
        f();
        rlx_gpu_dispatch::dispatch::clear_overrides();
        rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        rlx_ir::env::unset("RLX_METAL_SGEMM_PRECISE");
        rlx_ir::env::unset("RLX_DISABLE_MPS");
    }

    #[test]
    fn detects_some_gpu() {
        let hw = hw_model();
        assert!(!hw.gpu_name.is_empty());
        assert!(hw.sgemm_simd_flops > 0.0);
    }

    #[test]
    fn picks_simd_for_aligned() {
        with_clean_dispatch(|| {
            // Force the in-encoder MSL path so the threshold logic doesn't shadow
            // the alignment routing (these dims would otherwise hit Mps).
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
            let hw = MetalHwModel::detect();
            // Fully 64-aligned with enough tiles to saturate the GPU → the 64×64
            // tile path should win before the 32×32 fallback.
            assert_eq!(hw.pick_sgemm(64, 768, 2304), SgemmVariant::Simd64);
            // m=750 is NOT a multiple of 32, so the 32-row simd4x4 tiles would
            // overflow C past row m-1 — fall back to the padded simd kernel.
            assert_eq!(hw.pick_sgemm(750, 768, 2304), SgemmVariant::SimdPadded);
            // Was `Naive`. 8x16x16 is fully 8-aligned and `Simd` is genuinely
            // eligible here (`probe_eligibility_of_affected_shapes`), so the old
            // expectation was encoding the small-m branch's drop-to-Naive
            // defect rather than an intended choice. At this size dispatch
            // dominates either way; the fix matters at shapes like 16x128x3072.
            assert_eq!(hw.pick_sgemm(8, 16, 16), SgemmVariant::Simd);
            // Large k,n decode-style dims use SimdPadded (not Naive).
            assert_eq!(hw.pick_sgemm(6, 768, 2304), SgemmVariant::SimdPadded);
            assert_eq!(hw.pick_sgemm(2, 2048, 2048), SgemmVariant::SimdPadded);
            // Tiny / unaligned stay Naive.
            assert_eq!(hw.pick_sgemm(6, 7, 7), SgemmVariant::Naive);
            rlx_ir::env::set("RLX_METAL_SGEMM_PRECISE", "1");
            assert_eq!(hw.pick_sgemm(6, 768, 2304), SgemmVariant::Naive);
            rlx_ir::env::unset("RLX_METAL_SGEMM_PRECISE");
            rlx_ir::env::unset("RLX_DISABLE_MPS");
        });
    }

    /// The two enums must map onto each other exactly. A variant added on one
    /// side without the other would otherwise silently become a *different*
    /// kernel — the same class of bug as `Op::Rope`'s dropped `style`.
    #[test]
    fn dispatch_variant_mapping_round_trips() {
        for v in [
            SgemmVariant::Mps,
            SgemmVariant::Simd4x4,
            SgemmVariant::Simd64,
            SgemmVariant::Simd64SplitK,
            SgemmVariant::Simd,
            SgemmVariant::SimdPadded,
            SgemmVariant::Tiled,
            SgemmVariant::Naive,
        ] {
            assert_eq!(from_dispatch(to_dispatch(v)), v, "round-trip {v:?}");
        }
        for d in rlx_gpu_dispatch::dispatch::METAL_SGEMM_VARIANTS {
            assert_eq!(to_dispatch(from_dispatch(*d)), *d, "round-trip {d:?}");
        }
    }

    /// An untuned process must behave exactly as before the table existed.
    #[test]
    fn untuned_pick_equals_the_hand_written_cascade() {
        with_clean_dispatch(|| {
            rlx_gpu_dispatch::dispatch::clear_overrides();
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
            let hw = MetalHwModel::detect();
            for (m, k, n) in [
                (1, 1024, 1024),
                (6, 768, 2304),
                (64, 768, 2304),
                (750, 768, 2304),
                (2048, 2048, 2048),
                (8, 16, 16),
                (6, 7, 7),
            ] {
                assert_eq!(
                    hw.pick_sgemm(m, k, n),
                    hw.pick_sgemm_default(m, k, n),
                    "untuned pick drifted at {m}x{k}x{n}"
                );
            }
        });
    }

    /// The load-bearing guard: the table may retune, never relax. `Simd4x4` at
    /// `m % 32 != 0` writes past the end of C — an override naming it there must
    /// be refused and the default kept.
    /// Every `SgemmVariant` must have at least one spelling in
    /// `SGEMM_PIN_NAMES`.
    ///
    /// The exhaustive `match` is the mechanism: adding a variant without adding
    /// a pin name fails to compile here rather than silently producing another
    /// path nobody can A/B. `Simd64` and `Simd64SplitK` were in exactly that
    /// state — the two variants whose doc comments carry measured
    /// "beats MPS" claims were the two that could not be compared against MPS.
    #[test]
    fn sgemm_pin_names_cover_every_variant() {
        let all = [
            SgemmVariant::Mps,
            SgemmVariant::Simd4x4,
            SgemmVariant::Simd64,
            SgemmVariant::Simd64SplitK,
            SgemmVariant::Simd,
            SgemmVariant::SimdPadded,
            SgemmVariant::Tiled,
            SgemmVariant::Naive,
        ];
        // Total-match guard: if a variant is added, this stops compiling.
        for v in all {
            match v {
                SgemmVariant::Mps
                | SgemmVariant::Simd4x4
                | SgemmVariant::Simd64
                | SgemmVariant::Simd64SplitK
                | SgemmVariant::Simd
                | SgemmVariant::SimdPadded
                | SgemmVariant::Tiled
                | SgemmVariant::Naive => {}
            }
            assert!(
                SGEMM_PIN_NAMES.iter().any(|(_, pinned)| *pinned == v),
                "{v:?} has no RLX_METAL_SGEMM_VARIANT spelling, so it cannot be A/B'd"
            );
        }
    }

    /// The registry's `Enum` list and `SGEMM_PIN_NAMES` must agree.
    ///
    /// Two sources of truth for the same thing is how the entry drifted to
    /// `EnvKind::Bool` in the first place, which is what made a typo
    /// indistinguishable from a valid value. This fails if either side gains a
    /// spelling the other lacks.
    #[test]
    fn registry_enum_matches_the_pin_names() {
        let entry = rlx_ir::env_registry::lookup("RLX_METAL_SGEMM_VARIANT")
            .expect("RLX_METAL_SGEMM_VARIANT must be registered");
        let rlx_ir::env_registry::EnvKind::Enum(declared) = entry.kind else {
            panic!(
                "RLX_METAL_SGEMM_VARIANT is declared {:?}; it takes a variant name, so an \
                 unregistered value cannot be told from a valid one",
                entry.kind
            );
        };
        let mut from_code: Vec<&str> = SGEMM_PIN_NAMES.iter().map(|(n, _)| *n).collect();
        let mut from_registry: Vec<&str> = declared.to_vec();
        from_code.sort_unstable();
        from_registry.sort_unstable();
        assert_eq!(
            from_code, from_registry,
            "registry and SGEMM_PIN_NAMES disagree about the accepted spellings"
        );
    }

    #[test]
    fn every_pin_name_parses_to_its_variant() {
        with_clean_dispatch(|| {
            for (name, want) in SGEMM_PIN_NAMES {
                rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", *name);
                assert_eq!(
                    sgemm_variant_override(),
                    Some(*want),
                    "pin name {name:?} did not select {want:?}"
                );
                // Case-insensitivity is documented; check it is real.
                rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", name.to_ascii_uppercase());
                assert_eq!(
                    sgemm_variant_override(),
                    Some(*want),
                    "{name:?} is case-sensitive"
                );
            }
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        });
    }

    #[test]
    fn an_unknown_pin_name_does_not_silently_select_the_default() {
        with_clean_dispatch(|| {
            rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", "simd_64x64_typo");
            // `None` means "no pin" — the caller falls back to the cost model.
            // That is correct behaviour; the point is that it is now REPORTED
            // on stderr rather than happening in silence, which is what turned
            // a typo into a measurement of the default against itself.
            assert_eq!(sgemm_variant_override(), None);
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        });
    }

    /// The env pin outranks the cost model and the tuning cache — but not a
    /// correctness constraint. Mirror of
    /// `table_override_cannot_relax_an_alignment_rule` for the env path, which
    /// had no such guard: `pick_sgemm` returned the pinned variant before
    /// `sgemm_eligible` was ever consulted.
    #[test]
    fn env_pin_cannot_relax_an_alignment_rule() {
        with_clean_dispatch(|| {
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            let hw = MetalHwModel::detect();
            // m=750 is not a multiple of 32, so Simd4x4 would write past C.
            let (m, k, n) = (750usize, 768usize, 2304usize);
            assert!(!hw.sgemm_eligible(SgemmVariant::Simd4x4, m, k, n));
            let want = hw.pick_sgemm_default(m, k, n);

            rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", "simd4x4");
            assert_eq!(
                hw.pick_sgemm(m, k, n),
                want,
                "an ineligible env pin must be refused, not dispatched"
            );
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        });
    }

    /// ...and the pin must still WORK where it is eligible, or the guard above
    /// would be satisfied by ignoring the variable entirely.
    #[test]
    fn env_pin_is_honoured_where_it_is_eligible() {
        with_clean_dispatch(|| {
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            let hw = MetalHwModel::detect();
            let (m, k, n) = (1024usize, 1024usize, 1024usize);
            for v in [
                SgemmVariant::Simd4x4,
                SgemmVariant::Simd64,
                SgemmVariant::Naive,
            ] {
                if !hw.sgemm_eligible(v, m, k, n) {
                    continue;
                }
                let name = SGEMM_PIN_NAMES
                    .iter()
                    .find(|(_, pv)| *pv == v)
                    .map(|(n, _)| *n)
                    .expect("covered by sgemm_pin_names_cover_every_variant");
                rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", name);
                assert_eq!(
                    hw.pick_sgemm(m, k, n),
                    v,
                    "eligible pin {name:?} was not honoured"
                );
            }
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
        });
    }

    #[test]
    fn table_override_cannot_relax_an_alignment_rule() {
        with_clean_dispatch(|| {
            use rlx_gpu_dispatch::dispatch::{Choice, MetalSgemm, Workload};
            rlx_gpu_dispatch::dispatch::clear_overrides();
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            let hw = MetalHwModel::detect();

            // m=750 is not a multiple of 32.
            let (m, k, n) = (750usize, 768usize, 2304usize);
            assert!(!hw.sgemm_eligible(SgemmVariant::Simd4x4, m, k, n));
            let want = hw.pick_sgemm_default(m, k, n);

            let arch = crate::tuning::gpu_arch().clone();
            let w = Workload::Matmul { m, k, n };
            rlx_gpu_dispatch::dispatch::set_override(
                w.key(&arch),
                Choice::MetalSgemm(MetalSgemm::Simd4x4),
            )
            .expect("the table itself accepts it — eligibility is the backend's call");
            assert_eq!(
                hw.pick_sgemm(m, k, n),
                want,
                "an ineligible override must not displace the default"
            );

            rlx_gpu_dispatch::dispatch::clear_overrides();
            rlx_ir::env::unset("RLX_DISABLE_MPS");
        });
    }

    /// …and an ELIGIBLE override must actually take effect, or the table is
    /// decorative.
    #[test]
    fn eligible_table_override_is_honoured() {
        with_clean_dispatch(|| {
            use rlx_gpu_dispatch::dispatch::{Choice, MetalSgemm, Workload};
            rlx_gpu_dispatch::dispatch::clear_overrides();
            rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");
            rlx_ir::env::set("RLX_DISABLE_MPS", "1");
            let hw = MetalHwModel::detect();

            // Fully 64-aligned: the cascade picks Simd64; Tiled is also legal here.
            let (m, k, n) = (64usize, 768usize, 2304usize);
            assert_eq!(hw.pick_sgemm_default(m, k, n), SgemmVariant::Simd64);
            assert!(hw.sgemm_eligible(SgemmVariant::Tiled, m, k, n));

            let arch = crate::tuning::gpu_arch().clone();
            let w = Workload::Matmul { m, k, n };
            rlx_gpu_dispatch::dispatch::set_override(
                w.key(&arch),
                Choice::MetalSgemm(MetalSgemm::Tiled),
            )
            .unwrap();
            assert_eq!(hw.pick_sgemm(m, k, n), SgemmVariant::Tiled);

            rlx_gpu_dispatch::dispatch::clear_overrides();
            rlx_ir::env::unset("RLX_DISABLE_MPS");
        });
    }
}
