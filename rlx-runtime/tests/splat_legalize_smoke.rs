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
//! `Op::GaussianSplatRender` legalizes on every RLX backend compiled into `rlx-runtime`.

mod splat_common;
use splat_common::ParityFixture;

#[test]
fn gaussian_splat_legalizes_on_registered_backends() {
    rlx_splat::register();
    let fixture = ParityFixture::tiny();
    let graph = fixture.build_graph();

    let devices = rlx_runtime::registered_devices();
    assert!(
        devices.contains(&rlx_runtime::Device::Cpu),
        "CPU backend must be registered for rlx-splat tests"
    );

    for device in devices {
        rlx_runtime::legalize_graph_for_device(graph.clone(), device).unwrap_or_else(|e| {
            panic!("legalize failed on {device:?}: {e}");
        });
    }
}
