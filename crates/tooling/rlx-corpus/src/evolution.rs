// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The compiler-evolution ledger** — which past defects became rules, and
//! which are still only stories.
//!
//! CAKE §3.2 is the paper's actual thesis: the harness is itself a target of
//! evolution, and "agents use feedback from failed candidates — sanitizer
//! reports, failure cases, correctness mismatches, debugging logs — and distil
//! recurring or high-cost failure modes into new analyses: an opaque runtime
//! crash becomes a verifier rule, a repeated illegal lowering pattern becomes a
//! static check, a systematic misprediction becomes a calibration target."
//!
//! rlx does this — but by hand, one defect at a time, with no record. The
//! knowledge lives in commit messages and in whoever remembers. This module is
//! the missing artifact: a ledger of defects this tree has actually shipped,
//! each paired with the mechanism that would now catch it, or explicitly
//! marked as **ungated**.
//!
//! The ungated entries are the point. They are the evolution backlog, and
//! keeping them visible is what turns "we fixed a bug" into "we grew a check".
//!
//! This is deliberately a hand-maintained ledger and not an inference engine.
//! Deriving "which check catches which bug" automatically would be guessing;
//! recording it costs one line per defect and is auditable.

/// How a past defect is caught today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// A static check rejects it before compilation.
    StaticGate(&'static str),
    /// A test would fail if it returned.
    RegressionTest(&'static str),
    /// A cost model or calibration was corrected.
    Calibration(&'static str),
    /// Nothing would catch a recurrence. The backlog.
    Ungated,
}

impl Disposition {
    pub const fn is_gated(self) -> bool {
        !matches!(self, Self::Ungated)
    }

    pub fn mechanism(self) -> &'static str {
        match self {
            Self::StaticGate(m) | Self::RegressionTest(m) | Self::Calibration(m) => m,
            Self::Ungated => "—",
        }
    }

    pub const fn kind(self) -> &'static str {
        match self {
            Self::StaticGate(_) => "static-gate",
            Self::RegressionTest(_) => "regression-test",
            Self::Calibration(_) => "calibration",
            Self::Ungated => "UNGATED",
        }
    }
}

/// One defect this tree shipped, and what became of it.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Short slug.
    pub defect: &'static str,
    /// What actually went wrong, in one line.
    pub symptom: &'static str,
    pub disposition: Disposition,
}

/// The ledger. Every entry is a defect that really occurred in this tree.
pub const LEDGER: &[Entry] = &[
    Entry {
        defect: "rope-table-stride",
        symptom: "RoPE table stride used head_dim/2 instead of n_rot/2; three backends agreed and were all wrong",
        disposition: Disposition::RegressionTest("rlx-runtime/tests/fd_backward_gate.rs"),
    },
    Entry {
        defect: "rms-norm-backward-1-over-r",
        symptom: "extra 1/r on the input gradient's cross term, in all seven implementations at once",
        disposition: Disposition::RegressionTest("rlx-runtime/tests/fd_backward_gate.rs"),
    },
    Entry {
        defect: "softmax-ce-dloss-broadcast",
        symptom: "d_loss[N] broadcast across the class axis; invisible at N == 1",
        disposition: Disposition::RegressionTest("rlx-runtime/tests/fd_backward_gate.rs"),
    },
    Entry {
        defect: "mlx-rmsnorm-beta-dropped",
        symptom: "MLX read 2 of 3 declared inputs, silently ignoring beta",
        disposition: Disposition::RegressionTest("rlx-runtime/tests/fd_backward_gate.rs"),
    },
    Entry {
        defect: "wgpu-host-cache-flush-order",
        symptom: "deferred H2D flushed in HashSet order; a dead factor overwrote a live scalar (~47% flake)",
        disposition: Disposition::StaticGate("rlx-compile/src/plan_check.rs (live-overlap)"),
    },
    Entry {
        defect: "metal-simd4x4-row-overrun",
        symptom: "32 rows written for a 16-row C, stomping the next tensor in the arena",
        disposition: Disposition::StaticGate("rlx-metal/src/cost.rs::sgemm_eligible"),
    },
    Entry {
        defect: "metal-env-pin-bypassed-eligibility",
        symptom: "RLX_METAL_SGEMM_VARIANT returned before sgemm_eligible ran, so a pin could dispatch an illegal tile",
        disposition: Disposition::RegressionTest(
            "rlx-metal/src/cost.rs::env_pin_cannot_relax_an_alignment_rule",
        ),
    },
    Entry {
        defect: "metal-variant-typo-silent",
        symptom: "an unrecognised RLX_METAL_SGEMM_VARIANT fell through to the default in silence, so an A/B measured one path twice",
        disposition: Disposition::StaticGate("rlx-metal/src/cost.rs::SGEMM_PIN_NAMES + warn_once"),
    },
    Entry {
        defect: "cuda-tile-register-overrun",
        symptom: "a tile passing every algebraic rule died with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        disposition: Disposition::StaticGate("rlx-gpu-dispatch/src/tiles.rs::TileParams::validate"),
    },
    Entry {
        defect: "rocm-int-constant-reinterpret",
        symptom: "i64 constants bit-reinterpreted as denormals; Op::Slice step>1 returned zeros",
        disposition: Disposition::RegressionTest("rlx-runtime/tests/fd_backward_gate.rs"),
    },
    Entry {
        defect: "rocm-gguf-transposed",
        symptom: "sgemm(N,N) against an [n,k] GGUF layout; hidden by an n=1 test",
        disposition: Disposition::Ungated,
    },
    Entry {
        defect: "wgpu-complex-host-fallback",
        symptom: "should_pack_host_op routed complex Binary to a lane-blind CPU path; complex multiply came back lane-wise",
        disposition: Disposition::RegressionTest("rlx-wgpu/tests/complex_parity.rs"),
    },
    Entry {
        defect: "cost-model-uncalibrated-ranking",
        symptom: "a device cost model invented throughput numbers with no hardware behind them",
        disposition: Disposition::Calibration("rlx-gpu-dispatch/src/cost.rs::CostProvenance"),
    },
    Entry {
        defect: "dispatch-bucket-aspect-collapse",
        symptom: "shape bucket keyed on floor(log2(m*k*n)), so decode and prefill of equal volume shared a route",
        disposition: Disposition::RegressionTest(
            "rlx-gpu-dispatch/src/dispatch.rs::equal_work_opposite_aspect_ratios_do_not_share_a_bucket",
        ),
    },
    Entry {
        defect: "tuner-evaluation-leakage",
        symptom: "one measured shape claimed a whole bucket; no held-out shape was ever checked",
        disposition: Disposition::StaticGate(
            "rlx-cuda/examples/tune_dispatch.rs (DOMAIN holdouts)",
        ),
    },
    Entry {
        defect: "env-registry-orphaned-gate",
        symptom: "check-rlx-env-vars existed but was in no aggregate, and rotted to 210 unregistered reads",
        disposition: Disposition::StaticGate("Justfile::check-rlx-env-vars (in ci)"),
    },
    Entry {
        defect: "env-reads-bypassing-shim",
        symptom: "208 RLX_* reads went through std::env, so no test or in-process A/B could set them",
        disposition: Disposition::StaticGate("scripts/gen-rlx-env-vars.py (shim check)"),
    },
    Entry {
        defect: "vacuous-check-empty-fold",
        symptom: "a comparator folded NaN through f64::max and reported perfect agreement on non-finite output",
        disposition: Disposition::StaticGate("rlx-corpus/src/oracle.rs (non-finite pre-check)"),
    },
    Entry {
        defect: "vacuous-check-wrong-iteration",
        symptom: "a placeholder scan iterated dict keys and inspected nothing, reporting 0 of 507",
        disposition: Disposition::RegressionTest(
            "scripts/gen-rlx-env-vars.py::summary_placeholders",
        ),
    },
    Entry {
        defect: "rocm-negative-probe-cached-forever",
        symptom: "rocm_context() cached its FIRST result in a OnceLock, so one probe against a runtime-suspended GPU marked ROCm unavailable for the whole process and skipped a 57-test suite on healthy hardware",
        disposition: Disposition::StaticGate(
            "rlx-rocm/src/device.rs (Mutex, success-only caching)",
        ),
    },
    Entry {
        defect: "schedule-port-drift",
        symptom: "the CUDA kernel schedule was hand-transcribed, so deleting a __syncthreads() from matmul.cu left every schedule check passing",
        disposition: Disposition::StaticGate(
            "rlx-gpu-kernels/src/kernel_schedule_port.rs::drift_against_source",
        ),
    },
    Entry {
        defect: "kv-append-unusable-without-a-fallback",
        symptom: "Op::KvAppend (O(1) single-row KV write) was implemented on metal/cuda/rocm with NO decompose, so a graph containing it could not run on cpu/wgpu/vulkan/mlx at all; portable model crates therefore stayed on Concat everywhere, and Carbon-500M decode on Metal spent 36.9% of GPU time (112 dispatches) in that concat — as much as every matmul combined",
        disposition: Disposition::RegressionTest("rlx-compile/tests/kv_append_portability.rs"),
    },
    Entry {
        defect: "small-m-sgemm-dropped-to-naive",
        symptom: "pick_sgemm's m<32 branch fell straight to Naive whenever its `k>=256 && n>=256` gate missed, even where Simd/SimdPadded/Tiled were all eligible; a shape sweep found 108 such shapes, worst 16x128x3072 where the cost model scores the alternative 13.5x cheaper. Naive calibrates at ~5 GFLOP/s against 2116 for simd4x4",
        disposition: Disposition::RegressionTest(
            "rlx-metal/src/cost.rs::the_cascade_never_picks_a_variant_its_own_cost_model_beats",
        ),
    },
    Entry {
        defect: "calibration-cache-written-under-contention",
        symptom: "metal_calibrate had no contention gate, so a measurement taken while another workload held the GPU was persisted to ~/.cache and silently trusted by every later process; two runs minutes apart gave 706 and 1510 GFLOP/s for the same attention shape, and either would have been saved",
        disposition: Disposition::StaticGate(
            "rlx-metal/examples/metal_calibrate.rs (load-average + GPU-utilization gate on SAVE)",
        ),
    },
    Entry {
        defect: "thunk-profile-mislabelled-as-wall-time",
        symptom: "the per-thunk profile header read `GPU-sync wall time` while the code recorded gpu_cmd_buf_seconds (a device span); the label led me to distrust the right measurement and quote a run-level one that was dominated by Q/K/V upload, producing a 5x-wrong attention figure",
        disposition: Disposition::Ungated,
    },
    Entry {
        defect: "cost-model-compute-term-in-seconds",
        symptom: "MetalHwModel::sgemm_cost_ns computed flops/throughput (SECONDS) and added dispatch_overhead_ns, so a 1024^3 GEMM contributed 0.001 against a ~20000 ns constant — the compute term was 1e9x too small and every estimate was shape-blind; it survived because nothing consumes it",
        disposition: Disposition::RegressionTest(
            "rlx-metal/src/cost.rs::sgemm_cost_is_in_nanoseconds_not_seconds",
        ),
    },
    Entry {
        defect: "attention-cost-used-the-slow-sgemm-variant",
        symptom: "the attention cost term divided by sgemm_simd_flops (the 8x8 variant, 61-66 GFLOP/s measured) instead of sgemm_simd_4x4_flops (the path the cascade picks, 2152-2617), and its FLOP count dropped both `batch` and a factor of 4",
        disposition: Disposition::RegressionTest(
            "rlx-metal/src/cost.rs::attention_cost_is_quadratic_in_sequence + attention_cost_accounts_for_batch",
        ),
    },
    Entry {
        defect: "ab-arms-dispatched-at-one-tile-size",
        symptom: "the Metal A/B dispatched every arm at the shipping tile edge, so a tile=32 kernel got 16x16 threadgroups, accumulated half of K and returned ~half the answer (5.37e2 vs 1.06e3); a tolerance-based check would have called it a rounding difference",
        disposition: Disposition::RegressionTest(
            "rlx-metal/examples/schedule_sgemm_ab.rs (per-arm tile edge + the bit-exactness gate)",
        ),
    },
    Entry {
        defect: "schedule-dtype-ignored-the-precision-param",
        symptom: "AppleKernelParams::precision=f16 emitted `threadgroup half` but the schedule regions stayed F32, so EmitFacts said 2048 threadgroup bytes while the params said 1024 — two statements of one fact disagreeing in the same report header",
        disposition: Disposition::RegressionTest(
            "rlx-metal/src/kernel_schedule_emit.rs::params_drive_the_emitted_kernel_end_to_end",
        ),
    },
    Entry {
        defect: "metal-port-ungated-broke-the-linux-build",
        symptom: "rlx-metal/src/kernel_schedule_port.rs was ungated while the `kernels` module it reads is `cfg(rlx_metal_host)`, so `cargo check -p rlx-metal` failed on Linux; a workspace test run builds every crate, so one ungated Apple module stops every test on the ROCm rig",
        disposition: Disposition::StaticGate(
            "rlx-metal/src/lib.rs (kernel_schedule_port gated on rlx_metal_host)",
        ),
    },
    Entry {
        defect: "glslang-tempdir-shared-across-threads",
        symptom: "the runtime shader compiler keyed its scratch directory on the pid alone, so two concurrent compiles overwrote each other's source and the first to finish deleted the directory the second was reading; it passed when the test ran alone and failed under cargo's parallel threads",
        disposition: Disposition::StaticGate(
            "rlx-vulkan/src/kernel_schedule_emit.rs::compile_spirv_glslang (per-call AtomicU64 suffix)",
        ),
    },
    Entry {
        defect: "reference-register-written-from-memory",
        symptom: "every metal row in the external-reference register said UNMEASURED while rlx-metal/examples/reference_perf.rs (an MPS anchor) was already in the tree; the register was authored from recollection rather than checked against the repository",
        disposition: Disposition::RegressionTest(
            "rlx-corpus/src/reference.rs::a_backend_with_an_anchor_has_at_least_one_measured_row",
        ),
    },
    Entry {
        defect: "mil-reshape-abort-with-no-finding",
        symptom: "a concrete reshape that changes the element count passes every rlx-ir check, returns garbage on CPU, and abort()s the process inside Apple's compiler with no operation name; the only detector ran on an already-compiled model.mil in a temp dir, so it never ran in CI",
        disposition: Disposition::StaticGate(
            "rlx-coreml/src/mil/verify.rs (walked before the compiler; RLX_COREML_VERIFY=strict)",
        ),
    },
    Entry {
        defect: "naga-barrier-passes-spirv-check-and-still-races",
        symptom: "an emitted Vulkan kernel built with naga's bare barrier() carried the right OpControlBarrier operands (AcquireRelease|WorkgroupMemory) and was still wrong at K=4096 on an RTX 3080 Ti while agreeing at small K; the SPIR-V operand assertion is necessary, not sufficient",
        disposition: Disposition::RegressionTest(
            "rlx-vulkan/tests/schedule_emit_parity.rs::every_emitted_schedule_matches_the_shipping_kernel_bit_for_bit",
        ),
    },
    Entry {
        defect: "example-name-collision-across-packages",
        symptom: "two packages both named an example `schedule_matmul_ab`, so cargo's shared target/release/examples/ left only the last-built one under the unsuffixed name and a harness would have measured it under both labels",
        disposition: Disposition::StaticGate(
            "Justfile::check-example-names (in ci) — scripts/check-example-names.py reads cargo metadata; exemptions must state why the two can never share a target/ dir",
        ),
    },
    Entry {
        defect: "ab-arm-order-first-touch-bias",
        symptom: "timing A/B arms back-to-back charged the first arm the fresh buffers' first-touch cost; the control arm (same kernel as the baseline) read 1.24x, and every treatment looked ~20% fast",
        disposition: Disposition::RegressionTest(
            "rlx-metal/examples/schedule_sgemm_ab.rs (warm-all-then-round-robin) + the serial control arm",
        ),
    },
    Entry {
        defect: "gpu-idle-gate-fooled-by-a-training-lull",
        symptom: "a utilization-only idle gate saw 0% during a training job's data-loading phase (util oscillated 100/0/100/0) and would have started a timing sweep that the job then contended",
        disposition: Disposition::StaticGate(
            "scripts/rig-gpu-idle-watch.sh (util AND zero compute procs, re-checked after the run)",
        ),
    },
    Entry {
        defect: "war-rule-blind-to-pipeline-stage",
        symptom: "the write-after-read rule keyed on region NAME, so every rotating buffer was reported as a race and no software pipeline could be expressed; found by porting a multi-stage matmul onto a rule that had only ever seen one-stage kernels",
        disposition: Disposition::RegressionTest(
            "rlx-ir/tests/kernel_schedule_verify.rs::staging_into_a_different_slot_is_not_a_write_after_read",
        ),
    },
    Entry {
        defect: "kernel-scan-missed-attributed-decl",
        symptom: "the source scan matched `void <name>(` and silently found nothing for a kernel declared `void __launch_bounds__(T) attention(`",
        disposition: Disposition::RegressionTest(
            "rlx-gpu-kernels/src/kernel_schedule_port.rs (empty-result guard + space-prefixed match)",
        ),
    },
    Entry {
        defect: "expand-from-non-unit-dim",
        symptom: "Op::Expand from a non-unit dim passed every static gate and panicked inside the CPU backend with an out-of-bounds read",
        disposition: Disposition::StaticGate(
            "rlx-ir/src/verify.rs (Op::Expand arm asks expand_shape, the same rule infer_shape dropped at its `.ok()`)",
        ),
    },
    Entry {
        defect: "metal-mpsgraph-init-segv",
        symptom: "~2% SIGSEGV per MPSGraphExecutable init, reproducing solo; a production risk on the Metal fast path",
        // Gates that the MITIGATION is in force, not that the crash is absent:
        // the fault stopped reproducing on macOS 26.4.1 (0 crashes in ~700
        // inits with the mitigation disabled, vs ~14 predicted), so a soak
        // would pass either way and gate nothing. Building the descriptor is
        // necessarily best-effort, which makes "OS dropped the selector" and
        // "mitigation applied" the same silent code path — that is what the
        // test distinguishes.
        disposition: Disposition::RegressionTest(
            "rlx-metal/tests/mpsgraph_sync_compile.rs (SyncCompile::Applied + the env control arm)",
        ),
    },
    Entry {
        defect: "wgpu-vulkan-dsp-divergence",
        // Re-measured on MoltenVK (macOS 26.4.1, M4 Pro): PartitionedConv,
        // IIR-scan and C64-Mul now have Vulkan cases and pass, and `pad` had NO
        // Vulkan case at all until `pad_parity.rs::pad_vulkan_matches_cpu` was
        // added — it passes too. So on this driver the symptom does not
        // reproduce. Still Ungated because that is one driver of three (the
        // original also reproduced on RADV and NVIDIA, neither reachable from
        // here) and because `eig` and the second-derivative case named in the
        // original report still have no Vulkan coverage to reproduce *or*
        // retire them. Narrowed, not closed.
        symptom: "PartitionedConv / IIR-scan / pad / eig wrong on Vulkan only; cause not established. Re-measured 2026-09-09: PartitionedConv/IIR/C64/pad all pass on MoltenVK; eig + 2nd-deriv still uncovered on Vulkan, other drivers unretested",
        disposition: Disposition::Ungated,
    },
    Entry {
        defect: "arm-linux-no-blas",
        symptom: "build.rs skipped OpenBLAS on non-x86_64, so 86 linalg tests panicked on Pi-class hardware",
        disposition: Disposition::Ungated,
    },
];

/// Rollup of the ledger.
#[derive(Debug, Clone, Copy)]
pub struct EvolutionReport {
    pub total: usize,
    pub gated: usize,
}

impl EvolutionReport {
    pub fn ungated(&self) -> usize {
        self.total - self.gated
    }

    pub fn coverage(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.gated as f64 / self.total as f64
    }
}

pub fn report() -> EvolutionReport {
    EvolutionReport {
        total: LEDGER.len(),
        gated: LEDGER.iter().filter(|e| e.disposition.is_gated()).count(),
    }
}

/// The evolution backlog: defects that would recur unnoticed.
pub fn ungated() -> Vec<&'static Entry> {
    LEDGER
        .iter()
        .filter(|e| !e.disposition.is_gated())
        .collect()
}

/// Render the ledger for the example app.
pub fn render() -> String {
    let r = report();
    let mut out = String::new();
    out.push_str(&format!(
        "compiler-evolution ledger: {} recorded defect(s), {} became a rule ({:.0}%)\n\n",
        r.total,
        r.gated,
        100.0 * r.coverage()
    ));
    for e in LEDGER {
        out.push_str(&format!(
            "  [{:<15}] {:<34} {}\n",
            e.disposition.kind(),
            e.defect,
            e.disposition.mechanism()
        ));
    }
    let un = ungated();
    if !un.is_empty() {
        out.push_str(&format!(
            "\nBACKLOG — {} defect(s) with no mechanism; a recurrence would be silent:\n",
            un.len()
        ));
        for e in un {
            out.push_str(&format!("  {} — {}\n", e.defect, e.symptom));
        }
    }
    out
}
