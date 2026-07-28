// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// F16/BF16 higher-order AD: native typed graphs (CPU) + widen-at-boundary GPU parity.

#![cfg(all(
    feature = "cpu",
    any(
        feature = "cuda",
        feature = "rocm",
        feature = "gpu",
        all(feature = "metal", target_os = "macos"),
        all(feature = "mlx", target_os = "macos")
    )
))]

use half::{bf16, f16};
use rlx_autodiff::nth_order_grad;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn build_x_cubed(dt: DType) -> Graph {
    let mut g = Graph::new("x3_lp");
    let x = g.input("x", Shape::scalar(dt));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(dt));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(dt));
    g.set_outputs(vec![x3]);
    g
}

fn decode_output(bytes: &[u8], dt: DType) -> f32 {
    match dt {
        DType::F16 => f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        DType::BF16 => bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        other => panic!("unexpected output dtype {other:?}"),
    }
}

fn eval_third(device: Device, forward: &Graph, wrt_dt: DType, x_val: f32) -> f32 {
    let hg = nth_order_grad(forward, "x", 3);
    let x_bytes = match wrt_dt {
        DType::F16 => f16::from_f32(x_val).to_le_bytes().to_vec(),
        DType::BF16 => bf16::from_f32(x_val).to_le_bytes().to_vec(),
        DType::F32 => x_val.to_le_bytes().to_vec(),
        other => panic!("{other:?}"),
    };
    let outs = Session::new(device)
        .compile(hg)
        .run_typed(&[("x", &x_bytes, wrt_dt)]);
    decode_output(&outs[0].0, outs[0].1)
}

fn assert_matches_cpu(
    device: Device,
    forward: Graph,
    wrt_dt: DType,
    x_val: f32,
    tol: f32,
    label: &str,
) {
    if !is_available(device) {
        eprintln!("skip higher_order_low_precision_parity {label} on {device:?} (unavailable)");
        return;
    }
    let cpu = eval_third(Device::Cpu, &forward, wrt_dt, x_val);
    let gpu = eval_third(device, &forward, wrt_dt, x_val);
    assert!(
        (cpu - gpu).abs() < tol,
        "{label} {device:?} {wrt_dt:?}: cpu={cpu} gpu={gpu} tol={tol}"
    );
}

mod cpu_only {
    use super::*;

    #[test]
    fn nth_order_f16_bf16_graphs_build() {
        for dt in [DType::F16, DType::BF16] {
            let g = build_x_cubed(dt);
            let hg = nth_order_grad(&g, "x", 3);
            assert_eq!(hg.node(hg.outputs[0]).shape.dtype(), dt);
            let _ = Session::new(Device::Cpu).compile(hg);
        }
    }

    #[test]
    fn native_f16_third_derivative() {
        let forward = build_x_cubed(DType::F16);
        let got = eval_third(Device::Cpu, &forward, DType::F16, 1.5);
        assert!(
            (got - 6.0).abs() < 0.5,
            "native f16 third deriv at 1.5: {got}"
        );
    }

    #[test]
    fn native_bf16_third_derivative() {
        let forward = build_x_cubed(DType::BF16);
        let got = eval_third(Device::Cpu, &forward, DType::BF16, 1.5);
        assert!(
            (got - 6.0).abs() < 1.0,
            "native bf16 third deriv at 1.5: {got}"
        );
    }

    #[test]
    fn nth_order_f32_third_with_f16_input_widen() {
        let forward = build_x_cubed(DType::F32);
        let got = eval_third(Device::Cpu, &forward, DType::F16, 1.5);
        assert!(
            (got - 6.0).abs() < 1e-2,
            "widened f16 input third deriv: {got}"
        );
    }

    #[test]
    fn nth_order_f32_third_with_bf16_input_widen() {
        let forward = build_x_cubed(DType::F32);
        let got = eval_third(Device::Cpu, &forward, DType::BF16, 1.5);
        assert!(
            (got - 6.0).abs() < 5e-2,
            "widened bf16 input third deriv: {got}"
        );
    }
}

fn gpu_native_f16_third(device: Device) {
    assert_matches_cpu(
        device,
        build_x_cubed(DType::F16),
        DType::F16,
        1.5,
        0.5,
        "native f16 third deriv",
    );
}

fn gpu_native_bf16_third(device: Device) {
    assert_matches_cpu(
        device,
        build_x_cubed(DType::BF16),
        DType::BF16,
        1.5,
        1.0,
        "native bf16 third deriv",
    );
}

fn gpu_f16_widen_third(device: Device) {
    let forward = build_x_cubed(DType::F32);
    assert_matches_cpu(
        device,
        forward,
        DType::F16,
        1.5,
        1e-2,
        "f32 graph f16 I/O widen third deriv",
    );
}

fn gpu_bf16_widen_third(device: Device) {
    let forward = build_x_cubed(DType::F32);
    assert_matches_cpu(
        device,
        forward,
        DType::BF16,
        1.5,
        5e-2,
        "f32 graph bf16 I/O widen third deriv",
    );
}

macro_rules! lp_gpu_suite {
    ($mod_name:ident, $device:expr, $($cfg:meta),+) => {
        $(#[$cfg])*
        mod $mod_name {
            use super::*;
            #[test]
            fn native_f16_third_derivative() {
                gpu_native_f16_third($device);
            }
            #[test]
            fn native_bf16_third_derivative() {
                gpu_native_bf16_third($device);
            }
            #[test]
            fn f16_widen_third_derivative() {
                gpu_f16_widen_third($device);
            }
            #[test]
            fn bf16_widen_third_derivative() {
                gpu_bf16_widen_third($device);
            }
        }
    };
}

lp_gpu_suite!(cuda, Device::Cuda, cfg(feature = "cuda"));
lp_gpu_suite!(rocm, Device::Rocm, cfg(feature = "rocm"));
lp_gpu_suite!(wgpu, Device::Gpu, cfg(feature = "gpu"));
lp_gpu_suite!(
    metal,
    Device::Metal,
    cfg(all(feature = "metal", target_os = "macos"))
);
lp_gpu_suite!(
    mlx,
    Device::Mlx,
    cfg(all(feature = "mlx", target_os = "macos"))
);
