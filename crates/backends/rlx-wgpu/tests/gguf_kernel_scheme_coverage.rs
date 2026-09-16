// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every GGUF scheme registered in the shared id table must have a branch in
//! the WGSL `dequant_gguf` shader — or be explicitly host-routed.
//!
//! The shader is an `if (params.scheme_id == N) { … return; }` chain with
//! **no default arm**. An id that reaches it without a matching branch falls
//! off the end, leaves the dequant scratch untouched, and the weights read as
//! **zeros** — the model degrades silently instead of failing. `Q2_0` (25)
//! sat in the table with no WGSL branch for exactly this reason, which is how
//! `Ternary-Bonsai` and `Pestle-27B` would have "run" on wgpu.
//!
//! Two directions matter and both are checked:
//!   * claimed-but-not-branched → silent zeros (the dangerous one);
//!   * branched-but-not-claimed → correct, just needlessly host-routed.

use std::collections::BTreeSet;

const NEEDLE: &str = "params.scheme_id == ";

/// Every `params.scheme_id == <N>u` the shader actually tests.
fn branches_in_shader() -> BTreeSet<u32> {
    let src = rlx_wgpu::kernels::DEQUANT_GGUF_WGSL;
    let mut ids = BTreeSet::new();
    for (i, _) in src.match_indices(NEEDLE) {
        let tail = &src[i + NEEDLE.len()..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(id) = digits.parse::<u32>() {
            ids.insert(id);
        }
    }
    ids
}

/// Ids in the shared table, via the public inverse of the id mapping.
fn registered_ids() -> BTreeSet<u32> {
    (0u32..=64)
        .filter(|id| rlx_ir::quant::QuantScheme::from_gpu_dequant_scheme_id(*id).is_some())
        .collect()
}

#[test]
fn wgsl_branch_set_matches_kernel_supports_scheme() {
    let in_shader = branches_in_shader();
    assert!(
        !in_shader.is_empty(),
        "parsed no `{NEEDLE}Nu` branches — the shader or this parser changed shape"
    );
    let claimed: BTreeSet<u32> = (0u32..=64)
        .filter(|id| rlx_wgpu::gguf_gpu::kernel_supports_scheme(*id))
        .collect();

    let claimed_not_branched: Vec<u32> = claimed.difference(&in_shader).copied().collect();
    assert!(
        claimed_not_branched.is_empty(),
        "kernel_supports_scheme claims {claimed_not_branched:?} but the WGSL \
         shader has no branch for them — these dispatch to the kernel, match \
         nothing, and read back as zeros"
    );

    let branched_not_claimed: Vec<u32> = in_shader.difference(&claimed).copied().collect();
    assert!(
        branched_not_claimed.is_empty(),
        "the WGSL shader branches on {branched_not_claimed:?} but \
         kernel_supports_scheme host-routes them — correct, but slower than \
         it needs to be; add them to the list"
    );
}

/// No registered scheme may be left unhandled. Today every one of them has a
/// WGSL branch; if that ever stops being true, the scheme must be listed
/// here so it is host-routed on purpose rather than silently mis-decoded.
#[test]
fn every_registered_scheme_is_branched_or_deliberately_host_routed() {
    /// `(id, why)` — schemes with no WGSL kernel, host-routed by choice.
    const HOST_ONLY: &[(u32, &str)] = &[];

    let in_shader = branches_in_shader();
    let registered = registered_ids();
    assert!(
        registered.len() >= 29,
        "expected the full shared id table, walked {}",
        registered.len()
    );

    for id in &registered {
        let branched = in_shader.contains(id);
        let host_only = HOST_ONLY.iter().any(|(h, _)| h == id);
        let scheme = rlx_ir::quant::QuantScheme::from_gpu_dequant_scheme_id(*id).unwrap();
        assert!(
            branched || host_only,
            "scheme {scheme:?} (id {id}) has no WGSL branch and is not listed \
             as host-only — this is the configuration that returns zeros"
        );
        assert!(
            !(branched && host_only),
            "scheme {scheme:?} (id {id}) is both branched and listed host-only"
        );
    }
}
