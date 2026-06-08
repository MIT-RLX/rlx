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

//! Autodiff VJP for `Op::LogMel` on CPU.

#![cfg(feature = "cpu")]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad;
use rlx_runtime::{Device, Session};

fn mel_filters(n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    (0..n_mels * n_bins)
        .map(|i| (i % 5) as f32 * 0.05 + 0.02)
        .collect()
}

#[test]
fn log_mel_vjp_matches_finite_diff() {
    let batch = 1;
    let n_fft = 32;
    let n_mels = 4;
    let n_bins = n_fft / 2 + 1;

    let mut g = Graph::new("log_mel_vjp");
    let spec = g.input("spec", Shape::new(&[batch, n_fft * 2], DType::F32));
    let filt = g.param("filters", Shape::new(&[n_mels, n_bins], DType::F32));
    let mel = g.log_mel(spec, filt);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![mel],
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![loss]);

    let filters = mel_filters(n_fft, n_mels);
    let mut spec_val = vec![0f32; batch * n_fft * 2];
    for k in 0..n_bins {
        spec_val[k] = 0.2 * (k as f32 + 1.0);
        spec_val[n_fft + k] = -0.1 * k as f32;
    }

    let mut exec = Session::new(Device::Cpu).compile(g.clone());
    exec.set_param("filters", &filters);
    let loss_val = exec.run(&[("spec", &spec_val)]).remove(0)[0];

    let backward = grad(&g, &[spec]);
    let mut bwd_exec = Session::new(Device::Cpu).compile(backward);
    bwd_exec.set_param("filters", &filters);
    let d_output = vec![1.0f32];
    let grad = bwd_exec
        .run(&[("spec", &spec_val), ("d_output", &d_output)])
        .remove(0);

    let eps = 1e-3;
    for i in 0..spec_val.len() {
        let mut plus = spec_val.clone();
        let mut minus = spec_val.clone();
        plus[i] += eps;
        minus[i] -= eps;
        let lp = exec.run(&[("spec", &plus)]).remove(0)[0];
        let lm = exec.run(&[("spec", &minus)]).remove(0)[0];
        let fd = (lp - lm) / (2.0 * eps);
        let ad = grad[i];
        assert!(
            (fd - ad).abs() < 5e-2,
            "bin {i}: fd={fd} ad={ad} loss={loss_val} spec={}",
            spec_val[i]
        );
    }
}
