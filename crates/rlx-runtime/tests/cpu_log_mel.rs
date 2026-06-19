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

//! `Op::LogMel` end-to-end tests on CPU.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn mel_filterbank(n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        for k in 0..n_bins {
            fb[m * n_bins + k] = if k == m % n_bins { 1.0 } else { 0.0 };
        }
    }
    fb
}

#[test]
fn log_mel_block_layout_cpu() {
    let batch = 2;
    let n_fft = 64;
    let n_mels = 8;
    let n_bins = n_fft / 2 + 1;

    let mut g = Graph::new("log_mel_test");
    let spec_in = g.input("spec", Shape::new(&[batch, n_fft * 2], DType::F32));
    let filt = g.param("filters", Shape::new(&[n_mels, n_bins], DType::F32));
    let mel = g.log_mel(spec_in, filt);
    g.set_outputs(vec![mel]);

    let mut exec = Session::new(Device::Cpu).compile(g);
    let filters = mel_filterbank(n_fft, n_mels);
    exec.set_param("filters", &filters);

    // One bin hot: re=1, im=0 at k=3 → power=1 at that bin.
    let mut spec = vec![0f32; batch * n_fft * 2];
    spec[3] = 1.0;
    spec[n_fft * 2 + 3] = 1.0;

    let out = exec.run(&[("spec", &spec)]).remove(0);
    assert_eq!(out.len(), batch * n_mels);
    assert!(out[3 % n_mels] > out[0]);
}

#[test]
fn log_mel_backward_cpu() {
    let batch = 1;
    let n_fft = 32;
    let n_mels = 4;
    let n_bins = n_fft / 2 + 1;

    let mut g = Graph::new("log_mel_bwd");
    let spec = g.input("spec", Shape::new(&[batch, n_fft * 2], DType::F32));
    let filt = g.param("filters", Shape::new(&[n_mels, n_bins], DType::F32));
    let dy = g.input("dy", Shape::new(&[batch, n_mels], DType::F32));
    let dspec = g.log_mel_backward(spec, filt, dy);
    g.set_outputs(vec![dspec]);

    let filters: Vec<f32> = (0..n_mels * n_bins)
        .map(|i| (i % 5) as f32 * 0.05 + 0.02)
        .collect();
    let mut spec_val = vec![0f32; batch * n_fft * 2];
    for k in 0..n_bins {
        spec_val[k] = 0.2 * (k as f32 + 1.0);
        spec_val[n_fft + k] = -0.1 * k as f32;
    }
    let dy_val = vec![1.0f32; batch * n_mels];

    let mut exec = Session::new(Device::Cpu).compile(g);
    exec.set_param("filters", &filters);
    let grad = exec.run(&[("spec", &spec_val), ("dy", &dy_val)]).remove(0);
    assert_eq!(grad.len(), spec_val.len());
    assert!(grad[0].abs() < 1.0);
    assert!(grad.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn log_mel_after_fft_cpu() {
    let batch = 1;
    let n_fft = 32;
    let n_mels = 4;
    let n_bins = n_fft / 2 + 1;

    let mut g = Graph::new("fft_log_mel");
    let signal = g.input("signal", Shape::new(&[batch, n_fft], DType::F32));
    let zeros = g.sub(signal, signal);
    let block = g.concat_(vec![signal, zeros], 1);
    let fft_out = g.fft(block, false);
    let flat = g.reshape_(fft_out, vec![batch as i64, (n_fft * 2) as i64]);
    let filt = g.param("filters", Shape::new(&[n_mels, n_bins], DType::F32));
    let mel = g.log_mel(flat, filt);
    g.set_outputs(vec![mel]);

    let mut exec = Session::new(Device::Cpu).compile(g);
    exec.set_param("filters", &mel_filterbank(n_fft, n_mels));

    let signal: Vec<f32> = (0..n_fft).map(|i| (i as f32 * 0.1).sin()).collect();
    let out = exec.run(&[("signal", &signal)]).remove(0);
    assert_eq!(out.len(), n_mels);
    assert!(out.iter().all(|v| v.is_finite()));
}
