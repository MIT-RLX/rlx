// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "op-gates")]
// Without `op-gates` there is no `op_scan` and no `sample_ops`, and an
// integration test compiles regardless of features — an ungated one breaks the
// default build for everyone.

//! Declared arity vs. what backends actually do with operands.
//!
//! `Op::arity` is a hand-written table. Every other gate around it asks whether
//! it is *self-consistent*; none can tell whether the numbers match reality,
//! because the real consumer is a backend doing `node.inputs[k]`. A wrong count
//! surfaces much later as a rejected graph or a dropped operand.
//!
//! This found `Op::If` declared `Exact(1)` while `sccp` built it with a capture
//! and both `rlx-unfuse` and MLX read `inputs[1..]` — so `verify` rejected every
//! `If` that captured anything.
//!
//! Parsing lives in [`rlx_check::op_scan`]; this file is policy.

use rlx_check::op_scan::{self, ClaimKind};
use rlx_ir::{OpKind, sample_ops};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Sites that exceed the declared arity **on purpose**, each with the reason.
/// An allowlist nobody re-checks becomes the next stale table, so
/// `no_stale_allowlist_entries` fails when an entry stops matching.
///
/// `(path fragment, op, needed operand count, why)`
const KNOWN: &[(&str, &str, usize, &str)] = &[
    // `Op::Conv` is `Exact(2)`, and `verify` rejects a 3-input `Conv`, so these
    // optional-bias reads are unreachable. Kept correct rather than deleted so
    // they cannot silently drop a bias if that contract ever changes.
    (
        "rlx-vulkan/src/backend.rs",
        "Conv",
        3,
        "unreachable: Op::Conv is 2-input by contract",
    ),
    (
        "rlx-wgpu/src/backend/compile/lower.rs",
        "Conv",
        3,
        "unreachable: Op::Conv is 2-input by contract",
    ),
    // Metal and wgpu each fold `FusedConvBiasAct` into a bias-carrying `Conv3d`
    // *after* verification, so the form is real but never reaches `verify`. The
    // IR contract stays `Exact(2)` deliberately: CPU's `compile_conv3d` ignores
    // a third operand, so widening it globally would let the bias be dropped
    // silently on any backend without the fold.
    (
        "rlx-metal/src/thunk/compile.rs",
        "Conv3d",
        3,
        "backend-private post-verify fold from FusedConvBiasAct",
    ),
    (
        "rlx-wgpu/src/backend/compile/lower.rs",
        "Conv3d",
        3,
        "backend-private post-verify fold from FusedConvBiasAct",
    ),
];

/// Anti-vacuity floors, pinned under what the tree has today. Every earlier
/// version of this scan passed while finding nothing, so a degenerate scan has
/// to fail loudly rather than report a clean tree.
const MIN_ARMS: usize = 1_000;
const MIN_INDEX_CLAIMS: usize = 200;
const MIN_LENGTH_CLAIMS: usize = 5;

struct Finding {
    file: String,
    line: usize,
    ops: Vec<String>,
    needs: usize,
    allowed: usize,
}

#[derive(Default)]
struct Stats {
    arms: usize,
    index_claims: usize,
    length_claims: usize,
}

fn backends_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backends")
}

fn rs_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// One scan of every backend source, shared by both tests so they cannot drift
/// apart the way two copies of the loop did.
fn survey() -> (Vec<Finding>, Stats) {
    let by_name: BTreeMap<String, OpKind> = sample_ops::ALL_KINDS
        .iter()
        .map(|&k| (format!("{k:?}"), k))
        .collect();
    let root = backends_dir();
    let mut files = Vec::new();
    rs_files(&root, &mut files);
    assert!(files.len() > 100, "expected to find backend sources");

    let mut findings = Vec::new();
    let mut stats = Stats::default();
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        stats.arms += op_scan::arm_count(&src);
        for claim in op_scan::scan(&src) {
            match claim.kind {
                ClaimKind::Index(_) => stats.index_claims += 1,
                ClaimKind::Length(_) => stats.length_claims += 1,
            }
            // A shared arm (`Op::A | Op::B =>`) is satisfied if ANY of its ops
            // permits the claim — the body branches on which one it got.
            let kinds: Vec<OpKind> = claim
                .ops
                .iter()
                .filter_map(|n| by_name.get(n))
                .copied()
                .collect();
            if kinds.is_empty() {
                continue;
            }
            let mut allowed = Some(0usize);
            for k in &kinds {
                match sample_ops::max_operands(*k) {
                    // Count carried in a field: no static bound to exceed.
                    None => {
                        allowed = None;
                        break;
                    }
                    Some(m) => allowed = allowed.map(|a| a.max(m)),
                }
            }
            let Some(allowed) = allowed else { continue };
            if claim.needs > allowed {
                findings.push(Finding {
                    file: path
                        .strip_prefix(&root)
                        .unwrap_or(path)
                        .display()
                        .to_string(),
                    line: claim.line,
                    ops: claim.ops,
                    needs: claim.needs,
                    allowed,
                });
            }
        }
    }
    (findings, stats)
}

fn is_known(f: &Finding) -> bool {
    f.ops.len() == 1
        && KNOWN
            .iter()
            .any(|(p, o, n, _)| f.file.contains(p) && f.ops[0] == *o && f.needs == *n)
}

#[test]
fn backends_never_index_past_declared_arity() {
    let (findings, stats) = survey();

    // Guard the guard, per claim KIND: a scan that silently stopped seeing one
    // form would still pass on the strength of the other.
    assert!(
        stats.arms > MIN_ARMS,
        "scan degenerated: {} arms",
        stats.arms
    );
    assert!(
        stats.index_claims > MIN_INDEX_CLAIMS,
        "no indexed operand reads attributed ({})",
        stats.index_claims
    );
    assert!(
        stats.length_claims > MIN_LENGTH_CLAIMS,
        "no node.inputs.len() claims attributed ({})",
        stats.length_claims
    );

    let unexpected: Vec<String> = findings
        .iter()
        .filter(|f| !is_known(f))
        .map(|f| {
            format!(
                "{}:{}: arm [{}] needs {} operand(s) but the widest declared arity is {}",
                f.file,
                f.line,
                f.ops.join(" | "),
                f.needs,
                f.allowed
            )
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "{} backend claim(s) beyond the declared arity:\n  {}",
        unexpected.len(),
        unexpected.join("\n  ")
    );
}

#[test]
fn no_stale_allowlist_entries() {
    let (findings, _) = survey();
    let stale: Vec<&str> = KNOWN
        .iter()
        .filter(|(p, o, n, _)| {
            !findings
                .iter()
                .any(|f| f.file.contains(p) && f.ops.len() == 1 && f.ops[0] == *o && f.needs == *n)
        })
        .map(|(p, _, _, _)| *p)
        .collect();
    assert!(
        stale.is_empty(),
        "allowlist entries no longer match any claim (remove them): {stale:?}"
    );
}
