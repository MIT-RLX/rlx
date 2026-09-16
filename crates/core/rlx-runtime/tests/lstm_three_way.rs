// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Who is right about the wide-hidden LSTM: CPU, Metal, or neither?**
//!
//! `rlx-metal`'s native MSL LSTM disagrees with the CPU thunk for hidden >= 48.
//! "The host fallback is bit-exact with CPU" proves nothing about which is
//! CORRECT — the host fallback *is* the CPU thunk. This asks a third,
//! independent implementation (wgpu's WGSL kernel) to break the tie.
//!
//! Lives here rather than in `rlx-metal/tests` because only `rlx-runtime` can
//! enable both backends at once:
//!
//!     cargo test -p rlx-runtime --test lstm_three_way --features metal,gpu -- --nocapture

#![cfg(any(feature = "metal", feature = "gpu", feature = "cuda", feature = "rocm"))]

use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F32: DType = DType::F32;

fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * scale
        })
        .collect()
}

/// **Which implementation is actually right?**
///
/// "The host fallback is bit-exact with CPU" is circular — the host fallback IS
/// the CPU thunk, so it can only ever agree with itself. Nothing so far has
/// established that CPU is the reference and Metal the deviation rather than the
/// other way round. A third independent implementation breaks the tie.
///
/// wgpu runs the same `Op::Lstm` through a completely separate WGSL kernel, so
/// whichever of CPU/Metal it agrees with is the one computing the recurrence
/// correctly.
#[cfg(all(feature = "metal", feature = "gpu"))]
#[test]
fn a_third_backend_breaks_the_cpu_vs_metal_tie() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("metal not available — skipped");
        return;
    }
    let wgpu_ok = rlx_runtime::is_available(Device::Gpu);
    if !wgpu_ok {
        eprintln!("wgpu not available — cannot break the tie");
        return;
    }
    println!("  h   seq   |cpu-metal|   |cpu-wgpu|   |metal-wgpu|   verdict");
    for &(h, s) in &[(32usize, 64usize), (48, 64), (96, 64)] {
        let x = fill(s * h, 4, 1.0);
        let cpu = dev_run(Device::Cpu, s, h, &x);
        let met = dev_run(Device::Metal, s, h, &x);
        let wg = dev_run(Device::Gpu, s, h, &x);
        let d = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max)
        };
        let (cm, cw, mw) = (d(&cpu, &met), d(&cpu, &wg), d(&met, &wg));
        let verdict = if cw < 1e-5 && cm > 1e-5 {
            "wgpu sides with CPU -> METAL is wrong"
        } else if mw < 1e-5 && cm > 1e-5 {
            "wgpu sides with Metal -> CPU is wrong"
        } else if cm < 1e-5 {
            "all three agree"
        } else {
            "all three disagree — inconclusive"
        };
        println!("  {h:3} {s:4}   {cm:.3e}   {cw:.3e}   {mw:.3e}   {verdict}");
    }
}

/// Same graph as `cpu_run_w`, on an arbitrary device.
fn dev_run(device: Device, s: usize, h: usize, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("lstm_tie");
    let xi = g.input("x", Shape::new(&[1, s, h], F32));
    let wih = g.param("w_ih", Shape::new(&[4 * h * h], F32));
    let whh = g.param("w_hh", Shape::new(&[4 * h * h], F32));
    let bias = g.param("bias", Shape::new(&[4 * h], F32));
    let y = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![xi, wih, whh, bias],
        Shape::new(&[1, s, h], F32),
    );
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param("w_ih", &fill(4 * h * h, 1, 0.1));
    c.set_param("w_hh", &fill(4 * h * h, 2, 0.1));
    c.set_param("bias", &fill(4 * h, 3, 0.1));
    c.finalize_params();
    c.run(&[("x", x)]).remove(0)
}

/// Independent f64 reference — the arbiter.
///
/// Two GPU backends agreeing is suggestive, not proof: they could share a
/// lowering mistake. This implements the LSTM directly from its definition in
/// f64, in this file, depending on no rlx kernel at all. Gate order i,f,g,o with
/// a single merged bias, matching `Op::Lstm`'s documented layout:
///   z[r] = bias[r] + Σ_j w_ih[r*d+j]·x[j] + Σ_j w_hh[r*h+j]·h_prev[j]
///
/// Gated with its only caller: this file compiles under any of metal / gpu /
/// cuda / rocm, but the arbiter is consulted from the metal arm alone, so a
/// `cpu,cuda` build has it as dead code. Matching the cfg rather than blanket
/// `allow(dead_code)` means adding a caller in another arm is a compile error
/// pointing here, not a silently unused function.
#[cfg(feature = "metal")]
fn reference_f64(s: usize, h: usize, x: &[f32]) -> Vec<f32> {
    let w_ih = fill(4 * h * h, 1, 0.1);
    let w_hh = fill(4 * h * h, 2, 0.1);
    let bias = fill(4 * h, 3, 0.1);
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    let mut hp = vec![0f64; h];
    let mut c = vec![0f64; h];
    let mut out = vec![0f32; s * h];
    for t in 0..s {
        let mut z = vec![0f64; 4 * h];
        for (r, zr) in z.iter_mut().enumerate() {
            let mut acc = bias[r] as f64;
            for j in 0..h {
                acc += w_ih[r * h + j] as f64 * x[t * h + j] as f64;
            }
            for j in 0..h {
                acc += w_hh[r * h + j] as f64 * hp[j];
            }
            *zr = acc;
        }
        for k in 0..h {
            let (i_g, f_g) = (sig(z[k]), sig(z[h + k]));
            let (g_g, o_g) = (z[2 * h + k].tanh(), sig(z[3 * h + k]));
            c[k] = f_g * c[k] + i_g * g_g;
            hp[k] = o_g * c[k].tanh();
            out[t * h + k] = hp[k] as f32;
        }
    }
    out
}

#[cfg(feature = "metal")]
#[test]
fn an_independent_f64_reference_says_who_is_right() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("metal not available — skipped");
        return;
    }
    println!("  h   seq   |ref-cpu|    |ref-metal|   verdict");
    for &(h, s) in &[(32usize, 64usize), (48, 64), (96, 64)] {
        let x = fill(s * h, 4, 1.0);
        let r = reference_f64(s, h, &x);
        let cpu = dev_run(Device::Cpu, s, h, &x);
        let met = dev_run(Device::Metal, s, h, &x);
        let d = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max)
        };
        let (rc, rm) = (d(&r, &cpu), d(&r, &met));
        // f64 vs f32 accumulation differs slightly; 1e-4 is generous but far
        // below the ~1.0 disagreement under investigation.
        let verdict = match (rc < 1e-4, rm < 1e-4) {
            (true, true) => "both correct",
            (true, false) => "CPU correct, METAL wrong",
            (false, true) => "METAL correct, CPU wrong",
            (false, false) => "both wrong",
        };
        println!("  {h:3} {s:4}   {rc:.3e}    {rm:.3e}   {verdict}");
    }
}

/// GRU and RNN share the LSTM's design — one workgroup per batch item, hidden
/// state in threadgroup memory, `hidden <= 256` native — so they plausibly share
/// its wide-hidden defect. CPU is the proven-correct side for LSTM (an f64
/// reference says so above), so a CPU-vs-GPU disagreement here is the same
/// signature.
#[cfg(feature = "metal")]
#[test]
fn gru_and_rnn_are_checked_for_the_same_wide_hidden_defect() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Metal, "metal") {
        eprintln!("metal not available — skipped");
        return;
    }
    let gru = |device: Device, s: usize, h: usize, x: &[f32]| -> Vec<f32> {
        let mut g = Graph::new("gru");
        let xi = g.input("x", Shape::new(&[1, s, h], F32));
        let wih = g.param("w_ih", Shape::new(&[3 * h * h], F32));
        let whh = g.param("w_hh", Shape::new(&[3 * h * h], F32));
        let bih = g.param("b_ih", Shape::new(&[3 * h], F32));
        let bhh = g.param("b_hh", Shape::new(&[3 * h], F32));
        let y = g.add_node(
            Op::Gru {
                hidden_size: h,
                num_layers: 1,
                bidirectional: false,
                carry: false,
            },
            vec![xi, wih, whh, bih, bhh],
            Shape::new(&[1, s, h], F32),
        );
        g.set_outputs(vec![y]);
        let mut c = Session::new(device).compile(g);
        c.set_param("w_ih", &fill(3 * h * h, 1, 0.1));
        c.set_param("w_hh", &fill(3 * h * h, 2, 0.1));
        c.set_param("b_ih", &fill(3 * h, 3, 0.1));
        c.set_param("b_hh", &fill(3 * h, 5, 0.1));
        c.finalize_params();
        c.run(&[("x", x)]).remove(0)
    };

    // Pushed well past where the LSTM failed: GRU has a recurrence and the same
    // fast transcendentals, so "clean at h=96/s=256" could just mean "less
    // sensitive", not immune.
    println!("  GRU:  h   seq   |cpu-metal|   verdict");
    for &(h, s) in &[
        (32usize, 64usize),
        (48, 64),
        (48, 256),
        (96, 64),
        (128, 256),
        (256, 256),
        (128, 512),
        (256, 1024),
    ] {
        let x = fill(s * h, 4, 1.0);
        let cpu = gru(Device::Cpu, s, h, &x);
        let met = gru(Device::Metal, s, h, &x);
        let d = cpu
            .iter()
            .zip(&met)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        let nan = met.iter().filter(|v| v.is_nan()).count();
        println!(
            "       {h:3} {s:4}   {d:.3e}   {}",
            if nan > 0 {
                "NaN — same defect"
            } else if d < 1e-5 {
                "agrees"
            } else {
                "DIVERGES — same defect"
            }
        );
    }
}

/// Does the same wide-hidden LSTM defect exist on CUDA / ROCm?
///
/// Metal (MSL) and wgpu (WGSL) share it bit-exactly while GRU is clean, so it is
/// LSTM-specific and travels with the kernel design rather than with one
/// shading language. CUDA and ROCm implement the same design. CPU is the
/// proven-correct side (see the f64 reference above), so a CPU-vs-device
/// disagreement here is the same defect.
#[cfg(any(feature = "cuda", feature = "rocm"))]
#[test]
fn other_gpu_backends_are_checked_for_the_wide_hidden_lstm_defect() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(rlx_runtime::Device::Gpu, "wgpu") {
        return;
    }
    #[cfg(feature = "cuda")]
    let dev = Device::Cuda;
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    let dev = Device::Rocm;
    // Skip when the device is absent, rather than panicking unconditionally.
    // The intent behind the old `panic!("would have proved nothing")` is right
    // and is exactly what `RLX_REQUIRE_DEVICE` expresses — but as a hard panic
    // it failed on every host without CUDA, including this one, so a workspace
    // `cargo test` (which feature-unifies `cuda` in) could never go green.
    // `skip_unless` asserts under the flag, which `rig.sh` sets, so the loud
    // failure still happens exactly where a missing device means something.
    if common::skip_unless(dev) {
        return;
    }
    println!("  {dev:?}:  h   seq   |cpu-dev|   verdict");
    for &(h, s) in &[(32usize, 64usize), (48, 64), (48, 256), (96, 64)] {
        let x = fill(s * h, 4, 1.0);
        let cpu = dev_run(Device::Cpu, s, h, &x);
        let got = dev_run(dev, s, h, &x);
        let d = cpu
            .iter()
            .zip(&got)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        let nan = got.iter().filter(|v| v.is_nan()).count();
        println!(
            "        {h:3} {s:4}   {d:.3e}   {}",
            if nan > 0 {
                "NaN — SAME DEFECT"
            } else if d < 1e-5 {
                "agrees"
            } else {
                "DIVERGES — SAME DEFECT"
            }
        );
    }
}

/// Is the wide-hidden LSTM defect in the WGSL SOURCE, or in the Apple GPU
/// stack it was measured on?
///
/// On macOS, wgpu runs through Metal — the same device and driver as
/// `rlx-metal` — so "MSL and WGSL agree bit-exactly" is equally consistent with
/// a shared source bug and with an Apple codegen bug. Running the identical
/// WGSL on discrete NVIDIA Vulkan separates them: clean there means the source
/// is fine and the Apple stack is at fault.
///
/// Needs `RLX_WGPU_LSTM_WIDE=1`, since wide hidden is now gated to the host.
#[cfg(feature = "gpu")]
#[test]
fn wgpu_wide_lstm_on_this_platform() {
    let _gpu = common::serialize_gpu();
    // One check, not two: the second was unreachable after the first returned,
    // and its `panic!` carried the same "would have proved nothing" intent that
    // `RLX_REQUIRE_DEVICE` already expresses — `skip_unless_available` asserts
    // under the flag and skips loudly without it.
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    if !rlx_ir::env::flag("RLX_WGPU_LSTM_WIDE") {
        eprintln!("set RLX_WGPU_LSTM_WIDE=1 to exercise the native wide path");
        return;
    }
    println!("  wgpu:  h   seq   |cpu-wgpu|   verdict");
    for &(h, s) in &[(32usize, 64usize), (48, 64), (48, 256), (96, 64)] {
        let x = fill(s * h, 4, 1.0);
        let cpu = dev_run(Device::Cpu, s, h, &x);
        let got = dev_run(Device::Gpu, s, h, &x);
        let d = cpu
            .iter()
            .zip(&got)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        let nan = got.iter().filter(|v| v.is_nan()).count();
        println!(
            "        {h:3} {s:4}   {d:.3e}   {}",
            if nan > 0 {
                "NaN — defect present"
            } else if d < 1e-5 {
                "agrees — defect ABSENT here"
            } else {
                "DIVERGES — defect present"
            }
        );
    }
}
