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
}
