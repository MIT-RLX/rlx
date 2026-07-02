// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! STFT: batched (one rfft over [n_frames, frame_len], the new `Graph::stft`)
//! vs the old per-frame pattern (one rfft per frame). Same result; the batched
//! form emits ONE FFT node/dispatch instead of `n_frames`, so the batched FFT
//! kernels (and CPU rayon) can parallelize across frames.
//!
//! ```sh
//! cargo run -p rlx-bench --release --example bench_fft_stft_batch          # CPU
//! cargo run -p rlx-bench --release --example bench_fft_stft_batch --features metal
//! ```

use std::io::Write;

use rlx_driver::Device;
use rlx_ir::{FftNorm, Graph, GraphExt, Tick};
use rlx_runtime::{CompiledGraph, Session};

fn const_signal(g: &mut Graph, t: usize) -> rlx_ir::NodeId {
    let data: Vec<u8> = (0..t)
        .flat_map(|i| ((i as f32 * 0.017).sin()).to_le_bytes())
        .collect();
    g.add_node(
        rlx_ir::Op::Constant { data },
        vec![],
        rlx_ir::Shape::new(&[t], rlx_ir::DType::F32),
    )
}

fn batched_stft(t: usize, frame_len: usize, hop: usize) -> Graph {
    let mut g = Graph::new("stft_batched");
    let x = const_signal(&mut g, t);
    let y = g.stft(x, frame_len, hop, FftNorm::Forward);
    g.set_outputs(vec![y]);
    g
}

fn per_frame_stft(t: usize, frame_len: usize, hop: usize) -> Graph {
    let mut g = Graph::new("stft_per_frame");
    let x = const_signal(&mut g, t);
    let n_frames = 1 + (t - frame_len) / hop;
    let half = frame_len / 2 + 1;
    let mut rows = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame = g.narrow_(x, 0, f * hop, frame_len);
        let (re, im) = g.rfft(frame, FftNorm::Forward);
        let block = g.concat_(vec![re, im], 0);
        rows.push(g.reshape_(block, vec![1, 2 * half as i64]));
    }
    let y = g.concat_(rows, 0);
    g.set_outputs(vec![y]);
    g
}

fn bench(dev: Device, name: &str) {
    let empty: &[(&str, &[f32])] = &[];
    let median = |c: &mut CompiledGraph| -> u64 {
        for _ in 0..3 {
            let _ = c.run(empty);
        }
        let mut s = Vec::with_capacity(15);
        for _ in 0..15 {
            let t0 = Tick::now();
            let _ = c.run(empty);
            s.push(Tick::now().elapsed_ns(t0));
        }
        s.sort_unstable();
        s[s.len() / 2]
    };

    println!("STFT on {name}: per-frame vs batched (Forward norm)");
    println!(
        "  {:>7} {:>6} {:>5} {:>7}  {:>12} {:>12}  {:>8}",
        "t", "frame", "hop", "frames", "per-frame µs", "batched µs", "speedup"
    );
    let _ = std::io::stdout().flush();
    // Whisper (400/160), Nemotron/FunASR (512/160), AEC (1024/512).
    for &(t, frame_len, hop) in &[
        (48_000usize, 400usize, 160usize),
        (48_000, 512, 160),
        (48_000, 1024, 512),
    ] {
        let n_frames = 1 + (t - frame_len) / hop;
        let mut cpf = Session::new(dev).compile(per_frame_stft(t, frame_len, hop));
        let mut cb = Session::new(dev).compile(batched_stft(t, frame_len, hop));
        let pf = median(&mut cpf);
        let b = median(&mut cb);
        println!(
            "  {t:>7} {frame_len:>6} {hop:>5} {n_frames:>7}  {:>10.1}µs {:>10.1}µs  {:>7.2}×",
            pf as f64 / 1000.0,
            b as f64 / 1000.0,
            pf as f64 / b.max(1) as f64,
        );
        let _ = std::io::stdout().flush();
    }
    println!();
}

fn main() {
    bench(Device::Cpu, "cpu");
    #[cfg(feature = "metal")]
    bench(Device::Metal, "metal");
    #[cfg(feature = "gpu")]
    bench(Device::Gpu, "wgpu");
}
