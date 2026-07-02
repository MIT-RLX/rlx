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
