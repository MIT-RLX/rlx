// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile shape as a **compile parameter**, with the legality rules in one place.
//!
//! `kernels/matmul.cu` used to hardcode `#define BM 64 / BN 64 / BK 16` — one
//! physical schedule for every shape on every arch. The kernel sources are JIT
//! strings (NVRTC / hipRTC) and the backends' disk caches are already keyed on
//! the source hash, so varying the tile costs nothing structurally: prepend a
//! `#define` block, get a distinct module and a distinct cache slot.
//!
//! What it *does* cost is a correctness obligation. The kernel body indexes its
//! shared-memory staging tiles with expressions that are only well-formed for
//! certain (BM, BN, BK, TM, TN, block-dim) combinations — a bad tile does not
//! fail to compile, it silently reads out of bounds. [`TileParams::validate`]
//! is that gate: every constraint the kernel body relies on, checked on the host
//! before a single byte of source is generated. This mirrors CAKE's
//! pre-compile-gate discipline — reject the illegal schedule with a localized
//! reason rather than discovering it as a wrong number on the GPU.
//!
//! [`TileParams::DEFAULT_MATMUL`] reproduces the historical hardcoded schedule
//! exactly, and `matmul_cuda_src_tiled` returns the untouched default source
//! for it, so the previously-validated path stays byte-identical.

use std::fmt;

/// A physical tile schedule for the shared `matmul` kernel.
///
/// * `bm`×`bn` — the block tile of C computed by one thread block.
/// * `bk` — the inner-K staging depth.
/// * `tm`×`tn` — the register micro-tile each thread accumulates.
/// * `bdx`×`bdy` — the thread-block dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileParams {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
    pub bdx: u32,
    pub bdy: u32,
}

/// Why a candidate tile was rejected. Each variant names the *kernel* invariant
/// it violates, so the message is a repair target rather than a bare "invalid".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileError {
    /// The tile's typed schedule does not verify.
    ///
    /// Carried as a rendered string rather than a `KernelScheduleError` so this
    /// crate stays dependency-free — it holds decisions, not IR. The schedule
    /// itself is built in `rlx-gpu-kernels`, which owns the kernel text the
    /// schedule describes.
    UnsoundSchedule(String),
    /// A dimension was zero.
    Zero(&'static str),
    /// `bdx * bdy` exceeds the per-block thread ceiling.
    TooManyThreads { threads: u32, max: u32 },
    /// `bm != bdy * tm` (or `bn != bdx * tn`) — the block tile and the register
    /// micro-tiles do not cover each other exactly, so the epilogue would leave
    /// rows/columns of C unwritten.
    TileCoverage {
        which: &'static str,
        block: u32,
        threads: u32,
        micro: u32,
    },
    /// The staging tile does not divide evenly across the thread block, so the
    /// scalar loader's `A_PER_THREAD`/`B_PER_THREAD` truncates and leaves part
    /// of the shared tile uninitialized.
    StagingNotDivisible {
        which: &'static str,
        elems: u32,
        threads: u32,
    },
    /// The vectorized float4 loader needs the staging tile to divide evenly
    /// into `4 * THREADS` chunks (`{A,B}_VEC_PER_THREAD` is exact).
    VectorNotDivisible {
        which: &'static str,
        elems: u32,
        threads: u32,
    },
    /// `bk % 4 != 0` or `bn % 4 != 0` — the float4 loaders' `BK/4` / `BN/4`
    /// chunking and the tile-origin alignment argument both need this.
    NotFloat4Aligned { which: &'static str, value: u32 },
    /// Shared memory for both staging tiles exceeds the per-block budget.
    SharedMemory { bytes: u32, max: u32 },
    /// `tm * tn` accumulator registers per thread past a sane ceiling — beyond
    /// this the compiler spills the accumulator to local memory, which costs
    /// more than the tiling saves.
    AccumulatorPressure { regs: u32, max: u32 },
    /// The block's estimated register demand overruns the per-block register
    /// file, so the launch would fail with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
    RegisterFile { needed: u32, max: u32 },
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsoundSchedule(why) => write!(f, "tile's schedule does not verify: {why}"),
            Self::Zero(w) => write!(f, "{w} must be non-zero"),
            Self::TooManyThreads { threads, max } => {
                write!(f, "{threads} threads/block exceeds the {max} ceiling")
            }
            Self::TileCoverage {
                which,
                block,
                threads,
                micro,
            } => write!(
                f,
                "B{which} ({block}) != block-dim ({threads}) * T{which} ({micro}); \
                 the register micro-tiles would not cover the block tile of C"
            ),
            Self::StagingNotDivisible {
                which,
                elems,
                threads,
            } => write!(
                f,
                "{which} staging tile of {elems} elements does not divide across \
                 {threads} threads; part of the shared tile would stay uninitialized"
            ),
            Self::VectorNotDivisible {
                which,
                elems,
                threads,
            } => write!(
                f,
                "{which} staging tile of {elems} elements does not divide into \
                 float4 chunks across {threads} threads (need a multiple of {})",
                threads * 4
            ),
            Self::NotFloat4Aligned { which, value } => write!(
                f,
                "{which} = {value} is not a multiple of 4; the float4 tile loaders \
                 chunk by {which}/4 and rely on it for tile-origin alignment"
            ),
            Self::SharedMemory { bytes, max } => {
                write!(
                    f,
                    "staging tiles need {bytes} B of shared memory (max {max})"
                )
            }
            Self::AccumulatorPressure { regs, max } => write!(
                f,
                "TM*TN = {regs} accumulator registers per thread exceeds {max}; \
                 the accumulator would spill to local memory"
            ),
            Self::RegisterFile { needed, max } => write!(
                f,
                "the block needs an estimated {needed} registers but the per-block \
                 register file holds {max}; the launch would fail with \
                 CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"
            ),
        }
    }
}

impl std::error::Error for TileError {}

/// Per-block thread ceiling on every arch rlx targets (CUDA sm_50+, ROCm gfx9+).
pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
/// Static shared-memory budget assumed for a block. Both CUDA and ROCm allow
/// more via opt-in dynamic smem; the kernel declares its tiles statically, and
/// 48 KiB is the portable static limit.
pub const MAX_SHARED_BYTES: u32 = 48 * 1024;
/// Accumulator-register ceiling per thread. 64 = an 8×8 micro-tile, already at
/// the edge of what fits alongside the loader's live values in a 255-register
/// budget at useful occupancy.
pub const MAX_ACCUM_REGS: u32 = 64;

/// Per-block register file, in 32-bit registers. 64 K on every CUDA arch from
/// Maxwell through Blackwell and on GCN/RDNA.
pub const REGISTER_FILE: u32 = 65_536;

/// Registers a thread needs beyond the ones the tile algebra accounts for:
/// staging-loader temporaries, the `float4` shuffles, loop induction, and the
/// global-address arithmetic that stays live across `__syncthreads()`.
///
/// **This number is measured, not derived.** A `128×128×32, TM=TN=4, 32×32`
/// tile passes every other rule in [`TileParams::validate`] — the algebra is
/// exact, shared memory is 32 KiB of 48 — and still dies at launch on sm_86 with
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, because 1024 threads times its real
/// register count overruns the file. Back-solving from that failure puts the
/// non-algebraic overhead above 40; 44 is the smallest round value that rejects
/// it while admitting every tile observed to launch.
///
/// Treat this as a cheap pre-filter, not the authority. The authority is the
/// driver: `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` on the *compiled* function
/// reports the exact answer for the exact device, and the CUDA backend checks it
/// after every tiled compile. This constant only exists so a hopeless tile does
/// not cost a compile to discover, and so the host-side rule is stated somewhere
/// a tuner can read it.
pub const SCHEDULING_REG_OVERHEAD: u32 = 44;

impl TileParams {
    /// The historical hardcoded schedule: 64×64 block tile, K-depth 16, 4×4
    /// register micro-tile, 16×16 = 256 threads. Kept exact so the default path
    /// generates byte-identical source.
    pub const DEFAULT_MATMUL: Self = Self {
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
        bdx: 16,
        bdy: 16,
    };

    /// Threads per block.
    pub const fn threads(&self) -> u32 {
        self.bdx * self.bdy
    }

    /// Static shared memory the two staging tiles occupy, in bytes.
    pub const fn shared_bytes(&self) -> u32 {
        (self.bm * self.bk + self.bk * self.bn) * 4
    }

    /// Every invariant the kernel body relies on. An `Ok` tile is safe to
    /// compile and run; an `Err` names the violated contract.
    pub fn validate(&self) -> Result<(), TileError> {
        for (name, v) in [
            ("bm", self.bm),
            ("bn", self.bn),
            ("bk", self.bk),
            ("tm", self.tm),
            ("tn", self.tn),
            ("bdx", self.bdx),
            ("bdy", self.bdy),
        ] {
            if v == 0 {
                return Err(TileError::Zero(match name {
                    "bm" => "bm",
                    "bn" => "bn",
                    "bk" => "bk",
                    "tm" => "tm",
                    "tn" => "tn",
                    "bdx" => "bdx",
                    _ => "bdy",
                }));
            }
        }

        let threads = self.threads();
        if threads > MAX_THREADS_PER_BLOCK {
            return Err(TileError::TooManyThreads {
                threads,
                max: MAX_THREADS_PER_BLOCK,
            });
        }

        // The epilogue writes C at `row0 = blockIdx.y*BM + ty*TM`, so the
        // micro-tiles must tile the block tile exactly in both axes.
        if self.bm != self.bdy * self.tm {
            return Err(TileError::TileCoverage {
                which: "M",
                block: self.bm,
                threads: self.bdy,
                micro: self.tm,
            });
        }
        if self.bn != self.bdx * self.tn {
            return Err(TileError::TileCoverage {
                which: "N",
                block: self.bn,
                threads: self.bdx,
                micro: self.tn,
            });
        }

        // float4 chunking of the staging tiles.
        if !self.bk.is_multiple_of(4) {
            return Err(TileError::NotFloat4Aligned {
                which: "bk",
                value: self.bk,
            });
        }
        if !self.bn.is_multiple_of(4) {
            return Err(TileError::NotFloat4Aligned {
                which: "bn",
                value: self.bn,
            });
        }

        // Scalar loaders: A_PER_THREAD / B_PER_THREAD must be exact.
        let a_elems = self.bm * self.bk;
        let b_elems = self.bk * self.bn;
        if !a_elems.is_multiple_of(threads) {
            return Err(TileError::StagingNotDivisible {
                which: "A",
                elems: a_elems,
                threads,
            });
        }
        if !b_elems.is_multiple_of(threads) {
            return Err(TileError::StagingNotDivisible {
                which: "B",
                elems: b_elems,
                threads,
            });
        }

        // Vector loaders: {A,B}_VEC_PER_THREAD must be exact too, else the
        // float4 path would leave a remainder of the tile unwritten while the
        // `full_block && full_k_tile` guard says it is fully covered.
        if !a_elems.is_multiple_of(threads * 4) {
            return Err(TileError::VectorNotDivisible {
                which: "A",
                elems: a_elems,
                threads,
            });
        }
        if !b_elems.is_multiple_of(threads * 4) {
            return Err(TileError::VectorNotDivisible {
                which: "B",
                elems: b_elems,
                threads,
            });
        }

        let smem = self.shared_bytes();
        if smem > MAX_SHARED_BYTES {
            return Err(TileError::SharedMemory {
                bytes: smem,
                max: MAX_SHARED_BYTES,
            });
        }

        let regs = self.tm * self.tn;
        if regs > MAX_ACCUM_REGS {
            return Err(TileError::AccumulatorPressure {
                regs,
                max: MAX_ACCUM_REGS,
            });
        }

        let needed = threads * self.per_thread_regs();
        if needed > REGISTER_FILE {
            return Err(TileError::RegisterFile {
                needed,
                max: REGISTER_FILE,
            });
        }

        Ok(())
    }

    /// Estimated 32-bit registers one thread needs: the `TM×TN` accumulator,
    /// the `TM + TN` operand registers the inner product holds, and
    /// [`SCHEDULING_REG_OVERHEAD`] for everything the tile algebra does not name.
    pub const fn per_thread_regs(&self) -> u32 {
        self.tm * self.tn + self.tm + self.tn + SCHEDULING_REG_OVERHEAD
    }

    /// The `#define` prelude that pins this tile, for prepending to the kernel
    /// source. Empty for the default tile — the source's own `#ifndef` defaults
    /// already are these values, so the default path's source text (and hence
    /// its JIT cache slot) is unchanged.
    pub fn defines(&self) -> String {
        if *self == Self::DEFAULT_MATMUL {
            return String::new();
        }
        format!(
            "#define BM {}\n#define BN {}\n#define BK {}\n\
             #define TM {}\n#define TN {}\n\
             #define BLOCK_DIM_X {}\n#define BLOCK_DIM_Y {}\n",
            self.bm, self.bn, self.bk, self.tm, self.tn, self.bdx, self.bdy
        )
    }

    /// A short stable label for cache keys, dumps, and tuning records.
    pub fn label(&self) -> String {
        format!(
            "{}x{}x{}_t{}x{}_b{}x{}",
            self.bm, self.bn, self.bk, self.tm, self.tn, self.bdx, self.bdy
        )
    }
}

impl Default for TileParams {
    fn default() -> Self {
        Self::DEFAULT_MATMUL
    }
}

/// The tile candidates the tuner is allowed to try, cheapest-to-widest.
///
/// Every entry is `validate`-clean (a unit test asserts it), so a tuner can
/// walk this list without re-deriving the legality algebra. The family is the
/// square `bm == bn`, `tm == tn`, `bdx == bdy` slice — the one where the float4
/// staging divides evenly on both operands — which is what makes a compact
/// enumerable search space possible at all.
pub const MATMUL_TILE_CANDIDATES: &[TileParams] = &[
    // 32×32 block, 8×8 threads (64), K-depth 16 — small-M decode shapes, where
    // a 64×64 tile leaves most of the block masked off.
    TileParams {
        bm: 32,
        bn: 32,
        bk: 16,
        tm: 4,
        tn: 4,
        bdx: 8,
        bdy: 8,
    },
    // 32×32 block, 16×16 threads (256), K-depth 32 — deeper staging at the same
    // output tile; trades smem for fewer K iterations.
    TileParams {
        bm: 32,
        bn: 32,
        bk: 32,
        tm: 2,
        tn: 2,
        bdx: 16,
        bdy: 16,
    },
    // The historical default.
    TileParams::DEFAULT_MATMUL,
    // 64×64 block, 16×16 threads (256), K-depth 32 — the default's output tile
    // with twice the staging depth: half the K iterations, 16 KiB smem.
    TileParams {
        bm: 64,
        bn: 64,
        bk: 32,
        tm: 4,
        tn: 4,
        bdx: 16,
        bdy: 16,
    },
    // 128×128 block, 16×16 threads (256), K-depth 8 — big-GEMM prefill shapes;
    // 8×8 accumulators per thread, 8 KiB smem.
    TileParams {
        bm: 128,
        bn: 128,
        bk: 8,
        tm: 8,
        tn: 8,
        bdx: 16,
        bdy: 16,
    },
    // 128×128 block, 16×16 threads (256), K-depth 16 — the widest tile that
    // still launches. The 32×32-thread (1024) variant of this shape is *not*
    // here: see `rejects_the_tile_that_overran_the_register_file_on_sm_86`.
    TileParams {
        bm: 128,
        bn: 128,
        bk: 16,
        tm: 8,
        tn: 8,
        bdx: 16,
        bdy: 16,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tile_is_legal_and_emits_no_defines() {
        assert_eq!(TileParams::DEFAULT_MATMUL.validate(), Ok(()));
        // Byte-identical source for the default => same JIT cache slot as before
        // the tile became a parameter.
        assert!(TileParams::DEFAULT_MATMUL.defines().is_empty());
    }

    #[test]
    fn every_candidate_is_legal() {
        for t in MATMUL_TILE_CANDIDATES {
            assert_eq!(t.validate(), Ok(()), "illegal candidate {t:?}");
        }
    }

    #[test]
    fn candidates_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for t in MATMUL_TILE_CANDIDATES {
            assert!(seen.insert(*t), "duplicate candidate {t:?}");
        }
    }

    /// The coverage rule is the one whose violation is silent: C would be
    /// partially written, not a crash.
    #[test]
    fn rejects_micro_tile_that_does_not_cover_the_block_tile() {
        // bm stays 64 but the micro-tile shrinks: 16 rows of threads × TM 2 = 32,
        // so the bottom half of every block tile of C is never written.
        let bad = TileParams {
            tm: 2,
            ..TileParams::DEFAULT_MATMUL
        };
        assert!(matches!(
            bad.validate(),
            Err(TileError::TileCoverage { which: "M", .. })
        ));
    }

    #[test]
    fn rejects_staging_tile_that_does_not_divide_across_threads() {
        let bad = TileParams {
            bm: 8,
            bn: 8,
            bk: 4,
            tm: 1,
            tn: 1,
            bdx: 8,
            bdy: 8,
        };
        // A tile = 8*4 = 32 elements over 64 threads → not divisible.
        assert!(matches!(
            bad.validate(),
            Err(TileError::StagingNotDivisible { which: "A", .. })
        ));
    }

    #[test]
    fn rejects_unaligned_bk() {
        let bad = TileParams {
            bk: 15,
            ..TileParams::DEFAULT_MATMUL
        };
        assert!(matches!(
            bad.validate(),
            Err(TileError::NotFloat4Aligned { which: "bk", .. })
        ));
    }

    #[test]
    fn rejects_shared_memory_blowup() {
        // 256×256 block tile, K-depth 64 → (256*64 + 64*256)*4 = 128 KiB.
        let bad = TileParams {
            bm: 256,
            bn: 256,
            bk: 64,
            tm: 8,
            tn: 8,
            bdx: 32,
            bdy: 32,
        };
        assert!(matches!(
            bad.validate(),
            Err(TileError::SharedMemory { .. })
        ));
    }

    #[test]
    fn rejects_accumulator_blowup() {
        let bad = TileParams {
            bm: 256,
            bn: 256,
            bk: 4,
            tm: 16,
            tn: 16,
            bdx: 16,
            bdy: 16,
        };
        assert!(matches!(
            bad.validate(),
            Err(TileError::AccumulatorPressure { regs: 256, .. })
        ));
    }

    /// Regression for a real launch failure. `128×128×32, TM=TN=4, 32×32
    /// threads` satisfies every shape rule — the algebra is exact, staging is
    /// 32 KiB of the 48 KiB budget, accumulators are 16 of 64 — and still died
    /// on the sm_86 rig with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, because 1024
    /// threads × its real register count overruns the 64 K register file. The
    /// shape algebra cannot see register pressure; this rule can.
    #[test]
    fn rejects_the_tile_that_overran_the_register_file_on_sm_86() {
        let observed_failure = TileParams {
            bm: 128,
            bn: 128,
            bk: 32,
            tm: 4,
            tn: 4,
            bdx: 32,
            bdy: 32,
        };
        // It really does pass every other rule.
        assert_eq!(observed_failure.threads(), 1024);
        assert!(observed_failure.shared_bytes() <= MAX_SHARED_BYTES);
        assert!(observed_failure.tm * observed_failure.tn <= MAX_ACCUM_REGS);
        assert_eq!(
            observed_failure.bm,
            observed_failure.bdy * observed_failure.tm
        );
        // …and is still rejected.
        assert!(matches!(
            observed_failure.validate(),
            Err(TileError::RegisterFile { .. })
        ));
        assert!(!MATMUL_TILE_CANDIDATES.contains(&observed_failure));
    }

    /// The register rule must not be so tight that it rejects tiles observed to
    /// launch — the 256-thread 8×8-accumulator tile is the tightest of those.
    #[test]
    fn register_rule_admits_the_widest_launchable_tile() {
        let widest = TileParams {
            bm: 128,
            bn: 128,
            bk: 16,
            tm: 8,
            tn: 8,
            bdx: 16,
            bdy: 16,
        };
        assert_eq!(widest.validate(), Ok(()));
        assert!(widest.threads() * widest.per_thread_regs() <= REGISTER_FILE);
    }

    #[test]
    fn rejects_too_many_threads() {
        let bad = TileParams {
            bm: 64,
            bn: 64,
            bk: 16,
            tm: 1,
            tn: 1,
            bdx: 64,
            bdy: 64,
        };
        assert!(matches!(
            bad.validate(),
            Err(TileError::TooManyThreads { threads: 4096, .. })
        ));
    }

    #[test]
    fn defines_pin_every_macro_the_kernel_guards() {
        let t = MATMUL_TILE_CANDIDATES[0];
        let d = t.defines();
        for m in ["BM", "BN", "BK", "TM", "TN", "BLOCK_DIM_X", "BLOCK_DIM_Y"] {
            assert!(d.contains(&format!("#define {m} ")), "missing #define {m}");
        }
    }

    #[test]
    fn labels_are_unique_per_candidate() {
        let mut seen = std::collections::HashSet::new();
        for t in MATMUL_TILE_CANDIDATES {
            assert!(seen.insert(t.label()), "label collision for {t:?}");
        }
    }
}
