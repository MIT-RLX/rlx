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
mod splat_common;
use splat_common::ParityFixture;

use rlx_runtime::{Device, Session};
use rlx_splat::{COSINE_DISTANCE_STRICT, assert_parity};

#[test]
fn backward_positions_grad_parity_on_available_devices() {
    rlx_splat::register();
    let fixture = ParityFixture::tiny();
    let graph = fixture.build_backward_graph();
    let reference = fixture.cpu_reference_positions_grad();
    let inputs = fixture.backward_session_inputs();

    let mut cpu_compiled = Session::new(Device::Cpu).compile(graph.clone());
    let cpu_out = cpu_compiled.run(&inputs);
    assert_parity(&cpu_out[0], &reference, 1e-5, COSINE_DISTANCE_STRICT)
        .expect("CPU backward vs host reference");

    for device in rlx_runtime::registered_devices() {
        if device == Device::Cpu {
            continue;
        }
        if !rlx_runtime::is_available(device) {
            eprintln!("skip {device:?}: not available");
            continue;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut compiled = Session::new(device).compile(graph.clone());
            let outs = compiled.run(&inputs);
            assert_eq!(outs[0].len(), reference.len());
            assert!(
                outs[0].iter().all(|v| v.is_finite()),
                "{device:?}: non-finite grad"
            );
            assert_parity(&outs[0], &reference, 1e-5, COSINE_DISTANCE_STRICT)
                .unwrap_or_else(|e| panic!("{device:?} backward parity: {e}"));
        }));
        if result.is_err() {
            eprintln!("skip {device:?}: backward panicked");
        }
    }
}
