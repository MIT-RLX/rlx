// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probe for EXPERIMENTAL_COOPERATIVE_MATRIX support on the host adapter.
//! Skips silently if no wgpu adapter is available.

#[test]
fn probe_cooperative_matrix_support() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    let adapter_feats = dev.adapter.features();
    eprintln!("Adapter: {} ({:?})", dev.name, dev.backend);
    eprintln!(
        "SHADER_F16: {}",
        adapter_feats.contains(wgpu::Features::SHADER_F16)
    );
    eprintln!(
        "EXPERIMENTAL_COOPERATIVE_MATRIX: {}",
        adapter_feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    );
    for (i, p) in dev
        .adapter
        .cooperative_matrix_properties()
        .iter()
        .enumerate()
    {
        eprintln!(
            "  coop[{i}] {}x{}x{} AB={:?} CR={:?}{}",
            p.m_size,
            p.n_size,
            p.k_size,
            p.ab_type,
            p.cr_type,
            if p.saturating_accumulation {
                " sat"
            } else {
                ""
            }
        );
    }
    eprintln!(
        "coop_f32_8x8_supported: {}",
        rlx_wgpu::device::coop_f32_8x8_supported()
    );
    eprintln!(
        "coop_f16_16x16_supported: {}",
        rlx_wgpu::device::coop_f16_16x16_supported()
    );
    eprintln!(
        "coop_f16_16x16_f32_acc_supported: {}",
        rlx_wgpu::device::coop_f16_16x16_f32_acc_supported()
    );
    eprintln!(
        "coop_discrete_backend: {}",
        rlx_wgpu::device::coop_discrete_backend()
    );
    if rlx_wgpu::device::coop_discrete_backend() {
        let k = rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&dev.device);
        eprintln!(
            "matmul_coop_f16_vulkan: {}",
            if k.is_some() { "compiled OK" } else { "None" }
        );
        eprintln!(
            "coop_f16_vk_f32acc: {}",
            rlx_wgpu::kernels::coop_f16_vk_f32acc_available(&dev.device)
        );
        let qkv = rlx_wgpu::kernels::matmul_qkv_coop_f16_vk_kernel(&dev.device);
        eprintln!(
            "matmul_qkv_coop_f16_vk: {}",
            if qkv.is_some() { "compiled OK" } else { "None" }
        );
    }
}
