#![cfg(feature = "io")]
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

use std::path::PathBuf;

use rlx_fdm::{Network, fdm, io::from_json_str};

#[test]
fn load_arch_json_and_solve() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/arch.json");
    let s = std::fs::read_to_string(&path).expect("arch.json");
    let mut net = from_json_str(&s).expect("parse");
    net.anchor_nodes(&[0, 10]);
    net.q.fill(-1.0);
    net.loads_on_free([0.0, 0.0, -0.2]);
    let eq = fdm(&net).expect("fdm");
    assert!(eq.max_free_residual_norm(&net.is_support) < 1e-8);
}
