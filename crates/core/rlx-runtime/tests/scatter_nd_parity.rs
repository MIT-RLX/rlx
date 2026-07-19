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

//! `Op::ScatterNd` parity — CPU reference vs host-delegating GPU backends.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, ScatterNdReduction, Shape};
use rlx_runtime::{Device, Session};

fn scatter_nd_graph(reduction: ScatterNdReduction) -> Graph {
    let mut g = Graph::new("scatter_nd");
    let data = g.input("data", Shape::new(&[4, 4], DType::F32));
    // F32 indices so f32-uniform GPU arenas match without I64 widen.
    let indices = g.input("indices", Shape::new(&[2, 1], DType::F32));
    let updates = g.input("updates", Shape::new(&[2, 4], DType::F32));
    let y = g.scatter_nd(data, indices, updates, reduction);
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, reduction: ScatterNdReduction) -> Vec<f32> {
    let data = [1.0f32; 16];
    let indices = [0.0f32, 2.0];
    let updates = [0.0f32, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0];
    Session::new(device)
        .compile(scatter_nd_graph(reduction))
        .run(&[
            ("data", &data[..]),
            ("indices", &indices[..]),
            ("updates", &updates[..]),
        ])
        .pop()
        .unwrap()
}

#[test]
fn scatter_nd_none_cpu() {
    let out = run(Device::Cpu, ScatterNdReduction::None);
    assert_eq!(&out[0..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&out[4..8], &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&out[8..12], &[2.0, 2.0, 2.0, 2.0]);
    assert_eq!(&out[12..16], &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn scatter_nd_add_cpu() {
    let mut g = Graph::new("snd_add");
    let data = g.input("data", Shape::new(&[2], DType::F32));
    let indices = g.input("indices", Shape::new(&[2, 1], DType::F32));
    let updates = g.input("updates", Shape::new(&[2], DType::F32));
    let y = g.scatter_nd(data, indices, updates, ScatterNdReduction::Add);
    g.set_outputs(vec![y]);
    let out = Session::new(Device::Cpu)
        .compile(g)
        .run(&[
            ("data", &[1.0f32, 1.0][..]),
            ("indices", &[0.0f32, 0.0][..]),
            ("updates", &[3.0f32, 4.0][..]),
        ])
        .pop()
        .unwrap();
    assert_eq!(out, vec![8.0, 1.0]);
}

#[cfg(feature = "metal")]
#[test]
fn scatter_nd_none_metal_matches_cpu() {
    let cpu = run(Device::Cpu, ScatterNdReduction::None);
    let metal = run(Device::Metal, ScatterNdReduction::None);
    assert_eq!(cpu, metal);
}
