// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **What each backend's performance is measured against, and whether anyone
//! has measured it.**
//!
//! CAKE Table 4 reports relative performance per kernel against TensorRT-LLM,
//! CUTLASS, DeepGEMM, FlashAttention-4 and FlashInfer, and treats a
//! below-reference entry as a signal to act on rather than a fact to omit.
//!
//! rlx's numbers are almost all rlx-versus-rlx. `tune_dispatch` says so in its
//! own contract — *"the compile-time default tile only; no external library
//! baseline"* — and beating your own fallback 1.97x says nothing about whether
//! that fallback sits at 40% or 95% of what the hardware gives a good
//! implementation.
//!
//! This module is the register of that gap. It is a *ledger*, not a benchmark:
//! it records, per (backend, workload), what the right external reference is
//! and whether a measurement against it exists. That distinction is the whole
//! value — "we have no external baseline for attention on Metal" is an
//! actionable statement, and an aggregate speedup number that quietly omits it
//! is not.
//!
//! # Why a delegating backend is the easy case, not the hard one
//!
//! For CUDA, an external reference means linking cuBLAS or CUTLASS. For MLX,
//! CoreML, TPU and QNN, rlx is *already calling* the vendor's implementation —
//! so the reference is right there, and the honest baseline for those backends
//! is the vendor's own best path rather than rlx's fallback. Those rows should
//! be the cheapest to close, and this table makes it obvious that they are open.
//!
//! # Relationship to [`crate::oracle::Authority`]
//!
//! `Authority` is the same idea for *numerical* validation: it records whether
//! a comparison is against a third party, an independent closed form, or rlx
//! itself. This is its performance twin. Both exist because a suite that
//! reports "1200 tests passing" or "1.97x faster" without saying against what
//! is reporting a number nobody can act on.

/// How far along a reference comparison is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefStatus {
    /// Measured on real hardware, in-tree, and re-runnable. Carries the path.
    Measured(&'static str),
    /// The reference is available on the target and nothing measures it yet.
    /// This is the backlog.
    Unmeasured,
    /// No external reference exists for this backend/workload, with a reason.
    /// Distinct from `Unmeasured`: nothing to do here, and saying so stops the
    /// row from reading as an omission.
    Unavailable(&'static str),
}

impl RefStatus {
    pub const fn is_measured(self) -> bool {
        matches!(self, Self::Measured(_))
    }

    pub const fn kind(self) -> &'static str {
        match self {
            Self::Measured(_) => "measured",
            Self::Unmeasured => "UNMEASURED",
            Self::Unavailable(_) => "unavailable",
        }
    }

    pub const fn detail(self) -> &'static str {
        match self {
            Self::Measured(p) => p,
            Self::Unmeasured => "—",
            Self::Unavailable(why) => why,
        }
    }
}

/// One (backend, workload) row: the external thing rlx should be compared to.
#[derive(Debug, Clone, Copy)]
pub struct Reference {
    pub backend: &'static str,
    pub workload: &'static str,
    /// The external implementation, named. Not "the vendor library" — the
    /// specific one, because that is what someone has to go and link.
    pub reference: &'static str,
    pub status: RefStatus,
}

/// The register.
///
/// Every row is a comparison rlx *should* be able to make. `Unmeasured` rows
/// are the backlog and are meant to outnumber the measured ones today.
pub const REFERENCES: &[Reference] = &[
    // ── Backends that author their own kernels ─────────────────────────
    Reference {
        backend: "cuda",
        workload: "dense f32 GEMM",
        reference: "cuBLAS (already linked)",
        status: RefStatus::Measured("crates/backends/rlx-cuda/examples/reference_perf.rs"),
    },
    Reference {
        backend: "cuda",
        workload: "attention (prefill + decode)",
        reference: "FlashInfer / FlashAttention",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "cuda",
        workload: "quantized / GGUF matmul",
        reference: "llama.cpp CUDA kernels",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "cuda",
        workload: "MoE / grouped GEMM",
        reference: "TensorRT-LLM pre-routed API",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "rocm",
        workload: "dense f32 GEMM",
        reference: "rocBLAS (already linked; RLX_ROCM_NO_VENDOR_GEMM toggles it)",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "metal",
        workload: "dense f32 GEMM",
        reference: "MPSMatrixMultiplication (already linked)",
        status: RefStatus::Measured("crates/backends/rlx-metal/examples/reference_perf.rs"),
    },
    Reference {
        backend: "metal",
        workload: "attention (causal prefill)",
        reference: "MPSGraph scaledDotProductAttention (macOS 14.4+)",
        status: RefStatus::Measured(
            "crates/backends/rlx-metal/examples/attention_reference_perf.rs",
        ),
    },
    Reference {
        backend: "metal",
        workload: "transformer prefill",
        reference: "PyTorch MPS",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "wgpu",
        workload: "dense f32 GEMM",
        reference: "none in-process",
        status: RefStatus::Unavailable(
            "WebGPU exposes no vendor BLAS; the honest anchor is the native \
             backend on the same device",
        ),
    },
    Reference {
        backend: "vulkan",
        workload: "dense f32 GEMM",
        reference: "none in-process",
        status: RefStatus::Unavailable(
            "no vendor BLAS over raw Vulkan compute; anchor against the native \
             CUDA/ROCm backend on the same device instead",
        ),
    },
    Reference {
        backend: "cpu",
        workload: "dense f32 GEMM",
        reference: "Accelerate / OpenBLAS (already linked)",
        status: RefStatus::Unmeasured,
    },
    // ── Delegating backends: the vendor path IS the reference ──────────
    //
    // These should be the cheapest rows to close. rlx already calls the
    // vendor's implementation, so "rlx via MLX" vs "MLX directly" needs no new
    // dependency — only a harness that runs both.
    Reference {
        backend: "mlx",
        workload: "dense GEMM / transformer block",
        reference: "MLX called directly (mlx.core), no rlx graph",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "coreml",
        workload: "transformer block on the ANE",
        reference: "coremltools-converted model of the same graph",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "tpu",
        workload: "dense GEMM / MLP",
        reference: "JAX/XLA running the same HLO",
        status: RefStatus::Unmeasured,
    },
    Reference {
        backend: "qnn",
        workload: "quantized inference",
        reference: "Qualcomm QNN sample runner",
        status: RefStatus::Unmeasured,
    },
];

/// Rollup.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceReport {
    pub total: usize,
    pub measured: usize,
    pub unavailable: usize,
}

impl ReferenceReport {
    /// Rows that could be measured and are not. The backlog.
    pub fn open(&self) -> usize {
        self.total - self.measured - self.unavailable
    }

    /// Fraction of *applicable* rows that have a measurement.
    ///
    /// `Unavailable` rows are excluded from the denominator rather than
    /// counted as failures — otherwise the number punishes backends for a
    /// reference that does not exist, and a metric that cannot reach 100% gets
    /// ignored.
    pub fn coverage(&self) -> f64 {
        let applicable = self.total - self.unavailable;
        if applicable == 0 {
            return 1.0;
        }
        self.measured as f64 / applicable as f64
    }
}

pub fn report() -> ReferenceReport {
    ReferenceReport {
        total: REFERENCES.len(),
        measured: REFERENCES.iter().filter(|r| r.status.is_measured()).count(),
        unavailable: REFERENCES
            .iter()
            .filter(|r| matches!(r.status, RefStatus::Unavailable(_)))
            .count(),
    }
}

/// Rows with a reachable reference and no measurement.
pub fn backlog() -> Vec<&'static Reference> {
    REFERENCES
        .iter()
        .filter(|r| matches!(r.status, RefStatus::Unmeasured))
        .collect()
}

/// Render the register.
pub fn render() -> String {
    let r = report();
    let mut out = format!(
        "external performance references: {}/{} applicable rows measured ({:.0}%), \
         {} unavailable\n\n",
        r.measured,
        r.total - r.unavailable,
        100.0 * r.coverage(),
        r.unavailable
    );
    for e in REFERENCES {
        out.push_str(&format!(
            "  [{:<11}] {:<8} {:<32} {}\n",
            e.status.kind(),
            e.backend,
            e.workload,
            e.reference
        ));
    }
    let open = backlog();
    if !open.is_empty() {
        out.push_str(&format!(
            "\nBACKLOG — {} comparison(s) that are possible today and have not been made.\n\
             Until they are, every rlx speedup is measured against rlx.\n",
            open.len()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workspace root, from this crate's manifest dir.
    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("crates/<group>/<crate> is three levels below the root")
            .to_path_buf()
    }

    /// A `Measured` row must point at a file that EXISTS.
    ///
    /// Both directions of drift are real and this register hit one immediately:
    /// `metal / dense f32 GEMM` was first written as `Unmeasured` even though
    /// `rlx-metal/examples/reference_perf.rs` was already in the tree, because
    /// the row was authored from memory instead of checked. A register whose
    /// rows are not verified against the repository is worse than none — it
    /// launders a guess as an audit.
    #[test]
    fn measured_rows_point_at_a_file_that_exists() {
        let root = workspace_root();
        for r in REFERENCES {
            if let RefStatus::Measured(path) = r.status {
                assert!(
                    path.ends_with(".rs"),
                    "{}/{}: {path:?} is prose, not a pointer",
                    r.backend,
                    r.workload
                );
                let full = root.join(path);
                assert!(
                    full.is_file(),
                    "{}/{}: {} does not exist",
                    r.backend,
                    r.workload,
                    full.display()
                );
            }
        }
    }

    /// The inverse check: a backend with a `reference_perf` example in the tree
    /// must have **at least one** measured row.
    ///
    /// Deliberately per-backend, not per-workload. The first version of this
    /// test asserted per row and immediately false-fired on
    /// `cuda / attention`: `rlx-cuda/examples/reference_perf.rs` exists but
    /// anchors dense GEMM only, so that row is genuinely open. A gate that
    /// flags legitimate backlog items is one people learn to ignore, which
    /// costs more than the drift it catches.
    ///
    /// What it does catch is the error that actually happened: `metal` had an
    /// MPS anchor in the tree and every metal row said UNMEASURED, because the
    /// rows were written from memory instead of checked.
    #[test]
    fn a_backend_with_an_anchor_has_at_least_one_measured_row() {
        let root = workspace_root();
        let backends: std::collections::BTreeSet<&str> =
            REFERENCES.iter().map(|r| r.backend).collect();
        for b in backends {
            let anchor = root
                .join("crates/backends")
                .join(format!("rlx-{b}"))
                .join("examples/reference_perf.rs");
            if !anchor.is_file() {
                continue;
            }
            assert!(
                REFERENCES
                    .iter()
                    .any(|r| r.backend == b && r.status.is_measured()),
                "{} exists but every `{b}` row is unmeasured — check the register against \
                 the repository rather than from memory",
                anchor.display()
            );
        }
    }

    /// An `Unavailable` row must say why. "No reference" with no reason is
    /// indistinguishable from "nobody looked".
    #[test]
    fn unavailable_rows_carry_a_reason() {
        for r in REFERENCES {
            if let RefStatus::Unavailable(why) = r.status {
                assert!(
                    why.len() > 20,
                    "{}/{}: unavailable with no explanation",
                    r.backend,
                    r.workload
                );
            }
        }
    }

    /// The backlog must be visible in the rendered report. A register that
    /// prints only its wins reads as coverage it has not earned — the same
    /// failure `coverage_is_reported_and_not_overstated` guards for the corpus.
    #[test]
    fn the_register_reports_the_backlog_rather_than_hiding_it() {
        let text = render();
        assert!(text.contains("BACKLOG"), "backlog not surfaced:\n{text}");
        assert!(text.contains("UNMEASURED"));
        let r = report();
        assert!(
            r.open() > 0,
            "if this ever hits zero, delete the assertion deliberately"
        );
    }

    /// Delegating backends must appear. They are the whole reason this register
    /// is cheap to act on: rlx already calls the vendor implementation, so the
    /// baseline needs no new dependency.
    #[test]
    fn delegating_backends_are_registered() {
        for b in ["mlx", "coreml", "tpu"] {
            assert!(
                REFERENCES.iter().any(|r| r.backend == b),
                "{b} delegates its kernels and has no reference row"
            );
        }
    }

    /// The two registers must agree on which backends delegate.
    ///
    /// `oracle::external_authority_for` decides whether a backend's runtime is
    /// an independent *numerical* authority; this table decides whether it is a
    /// *performance* reference. They are the same fact seen twice, and letting
    /// them drift would leave one of the two quietly wrong.
    #[test]
    fn the_perf_register_agrees_with_the_numerical_authority() {
        use crate::oracle::external_authority_for;
        for b in ["mlx", "coreml", "tpu", "qnn"] {
            assert!(
                external_authority_for(b).is_some(),
                "{b} is registered as a vendor perf reference but not as an external                  numerical authority"
            );
        }
        for b in ["cuda", "rocm", "metal", "wgpu", "vulkan", "cpu"] {
            assert!(
                external_authority_for(b).is_none(),
                "{b} runs rlx's own kernels; calling its output an external authority                  would be a category error"
            );
        }
    }
}
