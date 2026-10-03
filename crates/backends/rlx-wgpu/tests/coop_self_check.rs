// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cooperative-matrix self-check (`src/coop_probe.rs`) must (a) pass for
//! every coop kernel this adapter can actually build, and (b) be capable of
//! failing — a probe that always says yes protects nothing.

use rlx_wgpu::coop_probe::{CoopKernel, verified};

fn dev() -> Option<&'static rlx_wgpu::device::WgpuDevice> {
    rlx_wgpu::device::wgpu_device()
}

#[test]
fn every_buildable_coop_kernel_passes_its_self_check() {
    let Some(d) = dev() else {
        eprintln!("no wgpu adapter, skipping");
        return;
    };
    if !d
        .device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        eprintln!("adapter has no cooperative-matrix support, skipping");
        return;
    }
    // Which kernels this adapter would actually DISPATCH — see the selection in
    // `backend/run.rs`. The others are compiled here but never chosen, and they
    // are written against the other backend's convention, so they are reported
    // rather than asserted: `coopLoad`/`coopLoadT` do not mean the same thing on
    // Metal and Vulkan (naga's one `row_major` bool becomes Metal's
    // `transpose_matrix` but SPIR-V's `RowMajorKHR`/`ColumnMajorKHR`).
    let reachable: &[CoopKernel] = if d.backend == wgpu::Backend::Metal {
        &[CoopKernel::F32Metal, CoopKernel::Coop16]
    } else {
        &[
            CoopKernel::F32Portable,
            CoopKernel::Coop16,
            CoopKernel::F16Vk {
                widen: false,
                f32acc: false,
            },
            CoopKernel::F16Vk {
                widen: true,
                f32acc: false,
            },
            CoopKernel::F16Vk {
                widen: false,
                f32acc: true,
            },
            CoopKernel::F16Vk {
                widen: true,
                f32acc: true,
            },
        ]
    };
    let mut checked = 0;
    for &which in reachable {
        if !rlx_wgpu::coop_probe::buildable(&d.device, which) {
            continue;
        }
        checked += 1;
        let ok = verified(&d.device, &d.queue, which);
        eprintln!("{which:?}: {}", if ok { "PASS" } else { "FAIL" });
        assert!(
            ok,
            "{which:?} does not compute a·b on this device — it is reachable from \
             `derive_matmul_compute` on this backend, so shipping it means wrong numbers"
        );
    }
    eprintln!("checked {checked} coop kernel(s)");
}

#[test]
fn the_probe_can_fail() {
    let Some(d) = dev() else {
        eprintln!("no wgpu adapter, skipping");
        return;
    };
    if !d
        .device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        eprintln!("adapter has no cooperative-matrix support, skipping");
        return;
    }
    // Same probe, run against a shader that is the shipped `matmul_coop_f32`
    // with every `coopLoadT`/`coopStoreT` put back to the non-T form — i.e. the
    // bug as it shipped. If this passes, the probe is not measuring anything.
    let broken = rlx_wgpu::kernels::MATMUL_COOP_F32_WGSL
        .replace("coopLoadT<", "coopLoad<")
        .replace("coopStoreT(", "coopStore(");
    assert_ne!(
        broken,
        rlx_wgpu::kernels::MATMUL_COOP_F32_WGSL,
        "the shipped kernel no longer uses the T forms; update this test"
    );
    let ok = rlx_wgpu::coop_probe::probe_source(
        &d.device,
        &d.queue,
        CoopKernel::F32Metal,
        &broken,
        "matmul_coop_f32",
    );
    eprintln!("non-T (as-shipped-broken) kernel probe result: {ok:?}");
    assert_eq!(
        ok,
        Some(false),
        "the probe accepted a kernel that computes b·a — it would not have \
         caught the original bug"
    );
}
