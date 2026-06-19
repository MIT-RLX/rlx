// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! MLIP force+energy loss helper end-to-end on CPU.

#![cfg(feature = "cpu")]

use rlx_autodiff::{ForceEnergyLossWeights, build_force_energy_loss};
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_out(b: &[u8]) -> f32 {
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

fn gaussian_energy_graph(n: usize, d: usize) -> Graph {
    let mut g = Graph::new("gauss_energy");
    let pos = g.input("positions", Shape::new(&[n, d], DType::F32));
    let sq = g.binary(BinaryOp::Mul, pos, pos, Shape::new(&[n, d], DType::F32));
    let e = g.reduce(
        sq,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![e]);
    g
}

#[test]
fn mlip_loss_zero_at_perfect_targets() {
    let n = 2;
    let d = 3;
    let energy_g = gaussian_energy_graph(n, d);

    let loss_g = build_force_energy_loss(
        &energy_g,
        "positions",
        "force_ref",
        "energy_ref",
        ForceEnergyLossWeights::default(),
    );

    let pos_data = vec![1.0f32, 0.0, -1.0, 0.5, 0.5, 0.0];
    let energy: f32 = pos_data.iter().map(|x| x * x).sum();
    let force_ref_data: Vec<f32> = pos_data.iter().map(|x| -2.0 * x).collect();

    let mut c = Session::new(Device::Cpu).compile(loss_g);
    let outs = c.run_typed(&[
        ("positions", &f32_bytes(&pos_data), DType::F32),
        ("force_ref", &f32_bytes(&force_ref_data), DType::F32),
        ("energy_ref", &f32_bytes(&[energy]), DType::F32),
    ]);
    let loss = f32_out(&outs[0].0);
    assert!(
        loss.abs() < 1e-5,
        "perfect force+energy targets should give ~0 loss, got {loss}"
    );
}

#[test]
fn mlip_loss_increases_with_bad_force_ref() {
    let energy_g = gaussian_energy_graph(1, 2);
    let loss_g = build_force_energy_loss(
        &energy_g,
        "positions",
        "force_ref",
        "energy_ref",
        ForceEnergyLossWeights {
            force: 1.0,
            energy: 0.0,
        },
    );

    let pos = vec![1.0f32, 2.0];
    let good_force = vec![-2.0f32, -4.0];
    let bad_force = vec![0.0f32, 0.0];

    let mut c = Session::new(Device::Cpu).compile(loss_g);
    let good = f32_out(
        &c.run_typed(&[
            ("positions", &f32_bytes(&pos), DType::F32),
            ("force_ref", &f32_bytes(&good_force), DType::F32),
            ("energy_ref", &f32_bytes(&[0.0]), DType::F32),
        ])[0]
            .0,
    );
    let bad = f32_out(
        &c.run_typed(&[
            ("positions", &f32_bytes(&pos), DType::F32),
            ("force_ref", &f32_bytes(&bad_force), DType::F32),
            ("energy_ref", &f32_bytes(&[0.0]), DType::F32),
        ])[0]
            .0,
    );
    assert!(
        bad > good,
        "bad force ref should increase loss: good={good} bad={bad}"
    );
}
