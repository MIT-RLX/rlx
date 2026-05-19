// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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

    let k = rlx_wgpu::kernels::matmul_coop_f32_kernel(&dev.device);
    match k {
        Some(_) => eprintln!("matmul_coop_f32 compiled OK"),
        None => panic!("matmul_coop_f32 returned None despite coop matrix feature"),
    }
}
