// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The ND-indexing launches must pass as many arguments as the kernels take.**
//!
//! `launch_kernel!` builds a `Vec<*mut c_void>` and hands it to HIP; cudarc's
//! builder does the same. Neither checks the count against the kernel's own
//! `__global__` signature, and a mismatch is not a launch error — the kernel
//! reads whatever follows in the parameter buffer. That has shipped before
//! (`gguf_gpu::launch_dequant_gguf`), which is why `declared_param_count` exists
//! and why `RLX_GPU_VALIDATE_PARAMS=1` checks it at dispatch.
//!
//! But that check only runs when someone sets the flag *and* has a device. These
//! five kernels are launched from two backends whose rigs are frequently
//! unavailable, so the arity is pinned here instead: device-free, always run.
//!
//! Add or remove a kernel parameter and this test fails, pointing at the launch
//! sites in `rlx-cuda/src/backend/run.rs` and `rlx-rocm/src/backend/run.rs` that
//! have to change with it.

use rlx_gpu_kernels::{INDEXING_ND_CU, declared_param_count};

/// `(entry point, parameter count)` — kept in the same order as the `.cu`.
const SIGNATURES: &[(&str, usize)] = &[
    // arena, n, data_off, idx_off, dst_off, k, slice, tuples_per_batch,
    // batch_stride, meta
    ("gather_nd_f32", 10),
    // arena, n, data_off, idx_off, dst_off, data_len, rank, axis, axis_dim, meta
    ("gather_elements_f32", 10),
    // arena, n, upd_off, idx_off, dst_off, dst_len, rank, axis, reduction, meta
    ("scatter_elements_f32", 10),
    // arena, n, idx_off, upd_off, dst_off, dst_len, k, slice, reduction, meta
    ("scatter_nd_reduce_f32", 10),
    // arena, n, src_off, dst_off, src_len, do_copy, do_sanitize
    ("copy_sanitize_f32", 7),
];

#[test]
fn every_indexing_kernel_has_the_arity_its_launch_sites_assume() {
    for &(entry, expected) in SIGNATURES {
        let found = declared_param_count(INDEXING_ND_CU, entry)
            .unwrap_or_else(|| panic!("`{entry}` not found in indexing_nd.cu"));
        assert_eq!(
            found, expected,
            "`{entry}` takes {found} parameters but the CUDA/ROCm launch sites pass {expected}"
        );
    }
}

#[test]
fn the_entry_points_are_all_present() {
    // A renamed entry point compiles fine and fails only at first launch, on a
    // device — i.e. exactly where it is most expensive to find out.
    for &(entry, _) in SIGNATURES {
        assert!(
            INDEXING_ND_CU.contains(entry),
            "`{entry}` is registered in a kernel cache but missing from the source"
        );
    }
}
