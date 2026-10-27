// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx_wgpu::set_fast_math(true)` swaps the correctly-rounded `/` and `sqrt`
//! for the native WGSL operators. Results stay within WGSL's own bounds (2.5
//! ULP for `/`, `sqrt` inherited from `inverseSqrt`), and on Metal — whose fast
//! math rounds about a quarter of quotients and roots differently — they must
//! actually differ from the CPU, or the switch did nothing. The switch is
//! process-wide and kernels are built once, so this file holds one test.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 1 << 14;

fn ulps(a: f32, b: f32) -> u32 {
    if a.to_bits() == b.to_bits() {
        return 0;
    }
    if !a.is_finite() || !b.is_finite() || (a < 0.0) != (b < 0.0) {
        return u32::MAX;
    }
    a.to_bits().abs_diff(b.to_bits())
}

#[test]
fn fast_math_stays_within_wgsl_bounds_and_takes_effect() {
    rlx_wgpu::set_fast_math(true);
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    let s = Shape::new(&[N], DType::F32);
    let mut g = Graph::new("fast_math");
    let a = g.input("a", s.clone());
    let b = g.input("b", s.clone());
    let q = g.binary(BinaryOp::Div, a, b, s.clone());
    let r = g.activation(Activation::Sqrt, b, s);
    g.set_outputs(vec![q, r]);
    let av: Vec<f32> = (0..N)
        .map(|i| ((i as f32 + 0.5) * 0.618_034).fract() * 200.0 - 100.0)
        .collect();
    let bv: Vec<f32> = (0..N)
        .map(|i| ((i as f32 + 0.25) * 0.414_214).fract() * 50.0 + 0.01)
        .collect();
    let run = |d: Device| {
        Session::new(d)
            .compile(g.clone())
            .run(&[("a", &av), ("b", &bv)])
    };
    let (gpu, cpu) = (run(Device::Gpu), run(Device::Cpu));
    let stats = |j: usize| {
        let d: Vec<u32> = (0..N).map(|i| ulps(gpu[j][i], cpu[j][i])).collect();
        (
            d.iter().filter(|&&u| u > 0).count(),
            d.into_iter().max().unwrap_or(0),
        )
    };
    let ((div_off, div_max), (sqrt_off, sqrt_max)) = (stats(0), stats(1));
    eprintln!(
        "fast math: div {div_off}/{N} off (max {div_max} ulp), sqrt {sqrt_off}/{N} off (max {sqrt_max} ulp)"
    );
    assert!(div_max <= 3, "fast `/` beyond WGSL's 2.5 ulp: {div_max}");
    assert!(sqrt_max <= 3, "fast sqrt beyond a few ulp: {sqrt_max}");
    let metal =
        rlx_wgpu::device::wgpu_device().is_some_and(|d| matches!(d.backend, wgpu::Backend::Metal));
    if metal {
        assert!(
            div_off + sqrt_off > 0,
            "fast math on Metal changed nothing: the switch did not apply"
        );
    }
}
