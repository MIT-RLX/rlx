// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Double-single (2× f32) gives near-f64 precision on **Metal, which has no
//! `f64`** — validated through the real `rlx_metal::double_single::DwReduce`
//! backend capability. Also demonstrates the fast-math hazard (the same kernel
//! under Metal's default fast-math collapses to nonsense).

#![cfg(target_os = "macos")]

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use rlx_metal::double_single::DwReduce;
use std::ffi::c_void;

/// A stress array: `1e8 + 10000·1 − 1e8` (true = 10000). Each `+1` is below
/// `1e8`'s f32 ulp (=8), so naive f32 loses all of them.
fn ill_conditioned() -> Vec<f32> {
    let mut x = vec![1e8f32];
    x.extend(std::iter::repeat(1.0f32).take(10_000));
    x.push(-1e8f32);
    x
}

#[test]
fn dwreduce_recovers_f64_on_metal() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    let reducer = DwReduce::new(&device).expect("compile precise dw reduce");

    // 1) Ill-conditioned cancellation: naive f32 -> ~0, dw -> 10000.
    let x = ill_conditioned();
    let naive: f32 = x.iter().sum();
    let dw = reducer.sum(&x);
    eprintln!("Metal (no f64) ill-conditioned: naive f32={naive}  DwReduce={dw}");
    assert!((naive as f64).abs() < 1.0, "naive f32 lost it: {naive}");
    assert!((dw - 10_000.0).abs() < 1.0, "DwReduce should recover 10000: {dw}");

    // 2) Harmonic sum of 2M f32 terms — compare to the exact sum of the SAME
    //    f32 values (in f64). dw tracks it; naive f32 accumulates error.
    let n = 2_000_000u32;
    let hx: Vec<f32> = (1..=n).map(|i| 1.0 / i as f32).collect();
    let truth: f64 = hx.iter().map(|&v| v as f64).sum();
    let naive_h: f32 = hx.iter().sum();
    let dw_h = reducer.sum(&hx);
    let e_naive = ((naive_h as f64 - truth) / truth).abs();
    let e_dw = ((dw_h - truth) / truth).abs();
    eprintln!("Metal harmonic 2M:  naive f32 err={e_naive:.2e}  |  DwReduce err={e_dw:.2e}");
    assert!(e_dw < e_naive / 100.0, "dw should be >=100x better than naive f32");
    assert!(e_dw < 1e-6, "dw should be near-f64: {e_dw:e}");
}

/// The double-word kernels MUST be compiled with precise math: under fast-math
/// the compiler reassociates `two_sum`'s error term to zero and the result
/// collapses. Demonstrated by compiling the same source both ways.
#[test]
fn fast_math_breaks_the_error_free_transform() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => return,
    };
    const K: &str = r#"
kernel void dw_sum1(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    DwF32 acc = DwF32{0.0f, 0.0f};
    for (uint i = 0u; i < n; ++i) { acc = dw_add(acc, DwF32{x[i], 0.0f}); }
    out[0] = acc.hi; out[1] = acc.lo;
}"#;
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\n{K}",
        rlxsl::dw::double_single_prelude(rlxsl::Lang::Msl)
    );
    let run = |fast: bool| -> f64 {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(fast);
        let lib = device.new_library_with_source(&src, &opts).unwrap();
        let f = lib.get_function("dw_sum1", None).unwrap();
        let pipe = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let x = ill_conditioned();
        let xbuf = device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let obuf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        let n = x.len() as u32;
        let q = device.new_command_queue();
        let cmd = q.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&xbuf), 0);
        enc.set_buffer(1, Some(&obuf), 0);
        enc.set_bytes(2, 4, &n as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let p = obuf.contents() as *const f32;
        unsafe { *p as f64 + *p.add(1) as f64 }
    };
    let precise = run(false);
    let fast = run(true);
    eprintln!("EFT hazard: precise={precise}  fast-math={fast}  (true=10000)");
    assert!((precise - 10_000.0).abs() < 1.0, "precise must work: {precise}");
    assert!((fast - 10_000.0).abs() > 1.0, "fast-math must break it: {fast}");
}
