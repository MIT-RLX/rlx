// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Confirms whether the host adapter advertises SUBGROUP for compute.
#[test]
fn probe_subgroup_support() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    let f = dev.adapter.features();
    eprintln!("Adapter: {} ({:?})", dev.name, dev.backend);
    eprintln!("SUBGROUP:        {}", f.contains(wgpu::Features::SUBGROUP));
    eprintln!(
        "SUBGROUP_BARRIER:{}",
        f.contains(wgpu::Features::SUBGROUP_BARRIER)
    );
}
