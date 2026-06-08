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

// RLX — scalar literal execution parity on CPU (and Metal when built).

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
#[cfg(feature = "metal")]
use rlx_runtime::is_available;
use rlx_runtime::{Device, Session};

fn f64_bytes(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run_f32_scale(device: Device, factor: f32) -> Vec<f32> {
    let mut g = Graph::new("scale");
    let x = g.input("x", Shape::new(&[3], DType::F32));
    let scale = g.constant(factor as f64, DType::F32);
    let y = g.mul(x, scale);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(device).compile(g);
    let out = compiled.run(&[("x", &[1.0f32, 2.0, 3.0])]);
    out[0].to_vec()
}

#[test]
fn cpu_mul_by_graphext_constant() {
    let got = run_f32_scale(Device::Cpu, 2.0);
    assert_eq!(got, vec![2.0, 4.0, 6.0]);
}

#[test]
fn cpu_add_and_div_by_constants() {
    let mut g = Graph::new("add_div");
    let x = g.input("x", Shape::new(&[2], DType::F32));
    let one = g.constant(1.0, DType::F32);
    let two = g.constant(2.0, DType::F32);
    let sum = g.add(x, one);
    let y = g.div(sum, two);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let out = compiled.run(&[("x", &[1.0f32, 3.0])]);
    assert_eq!(out[0].to_vec(), vec![1.0, 2.0]);
}

#[test]
fn cpu_f64_constant_broadcasts() {
    let mut g = Graph::new("f64");
    let x = g.input("x", Shape::new(&[2], DType::F64));
    let half = g.constant(0.5, DType::F64);
    let y = g.sub(x, half);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let payload = f64_bytes(&[1.0, 2.0]);
    let out = compiled.run_typed(&[("x", &payload, DType::F64)]);
    let got: Vec<f64> = out[0]
        .0
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!((got[0] - 0.5).abs() < 1e-12);
    assert!((got[1] - 1.5).abs() < 1e-12);
}

#[cfg(feature = "metal")]
#[test]
fn metal_mul_by_graphext_constant_matches_cpu() {
    if !is_available(Device::Metal) {
        return;
    }
    let cpu = run_f32_scale(Device::Cpu, 3.0);
    let metal = run_f32_scale(Device::Metal, 3.0);
    assert_eq!(cpu.len(), metal.len());
    for (a, b) in cpu.iter().zip(metal.iter()) {
        assert!((a - b).abs() < 1e-4, "cpu={a} metal={b}");
    }
}
