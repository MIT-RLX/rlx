// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `CompileOptions::half_casts`: a `Cast` to f16 / bf16 keeps the f32 value by
//! default (half graphs execute in f32) and rounds to the format under
//! `HalfCasts::Round` — on every available backend.

use rlx_ir::number_format::FloatFormat;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, HalfCasts, Session};

/// `x (from) → cast(to) → cast(f32)`.
fn round_trip(from: DType, to: DType, n: usize) -> Graph {
    let mut g = Graph::new("half_cast");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let src = if from == DType::F32 {
        x
    } else {
        g.add_node(Op::Cast { to: from }, vec![x], Shape::new(&[n], from))
    };
    let h = g.add_node(Op::Cast { to }, vec![src], Shape::new(&[n], to));
    let y = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![h],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn half_casts_widen_by_default_and_round_on_request() {
    // Below f16's and bf16's mantissa step, a large value, and one past f16's range.
    let xs = [1.0f32 + 2f32.powi(-12), -3.140625, 70_000.0, 1.0e-6];
    let devices: Vec<Device> = rlx_runtime::available_devices()
        .into_iter()
        .filter(|d| !matches!(d, Device::Ane | Device::Xdna))
        .collect();
    for dev in devices {
        for (from, to, fmt) in [
            (DType::F32, DType::F16, FloatFormat::F16),
            (DType::F32, DType::BF16, FloatFormat::BF16),
            (DType::F64, DType::F16, FloatFormat::F16),
        ] {
            // A raw f64 tensor needs a backend with f64 storage (wgpu has none).
            if from == DType::F64 && dev != Device::Cpu {
                continue;
            }
            let g = round_trip(from, to, xs.len());
            let base = CompileOptions::baseline();
            let mut widen = Session::new(dev).compile_with(g.clone(), &base);
            let kept = widen.run(&[("x", &xs[..])]).remove(0);
            assert_eq!(
                kept, xs,
                "{dev:?} {from:?}→{to:?}: default keeps the f32 value"
            );

            let round = CompileOptions {
                half_casts: HalfCasts::Round,
                ..CompileOptions::baseline()
            };
            let mut rounded = Session::new(dev).compile_with(g, &round);
            let got = rounded.run(&[("x", &xs[..])]).remove(0);
            let want: Vec<f32> = xs.iter().map(|&v| fmt.round(v)).collect();
            assert_eq!(
                got, want,
                "{dev:?} {from:?}→{to:?}: Round matches FloatFormat::round"
            );
            assert_ne!(got, xs);
        }
    }
}

#[test]
fn half_casts_parse() {
    assert_eq!("round".parse::<HalfCasts>().unwrap(), HalfCasts::Round);
    assert_eq!("widen".parse::<HalfCasts>().unwrap(), HalfCasts::Widen);
    assert!("sometimes".parse::<HalfCasts>().is_err());
    assert_eq!(CompileOptions::baseline().half_casts, HalfCasts::Widen);
}
