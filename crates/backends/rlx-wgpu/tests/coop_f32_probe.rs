// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probe whether the host adapter can compile + instantiate the
//! pure-f32 cooperative-matrix matmul kernel.
//!
//! We don't run it — we just trigger pipeline creation, which is where
//! WGSL→MSL/SPIR-V translation happens. If naga can't lower
//! `coop_mat8x8<f32>` on this device, the pipeline build panics here.

#[test]
fn probe_coop_f32_kernel_compiles() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    let f = dev.adapter.features();
    eprintln!("Adapter: {} ({:?})", dev.name, dev.backend);
    let coop = f.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    eprintln!("EXPERIMENTAL_COOPERATIVE_MATRIX: {coop}");
    if !coop {
        eprintln!("no coop matrix feature, skipping");
        return;
    }

    let k = rlx_wgpu::kernels::matmul_coop_f32_active_kernel(&dev.device);
    match k {
        Some(_) => eprintln!("matmul_coop_f32 active kernel compiled OK"),
        None => eprintln!("no active coop f32 kernel for this backend"),
    }
    if f.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        && dev.backend == wgpu::Backend::Vulkan
    {
        let portable = rlx_wgpu::kernels::matmul_coop_f32_portable_kernel(&dev.device);
        eprintln!(
            "matmul_coop_f32_portable: {}",
            if portable.is_some() { "OK" } else { "None" }
        );
    }
}
