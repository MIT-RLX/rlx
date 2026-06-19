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

//
// CPU vs GPU parity for [`directional_nth_grad`] (ND wrt + direction inputs).
//!
//! ```sh
//! cargo test -p rlx-runtime --features cpu,cuda --test directional_nth_gpu_parity
//! cargo test -p rlx-runtime --features cpu,apple --test directional_nth_gpu_parity
//! ```

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

use rlx_autodiff::directional_nth_grad;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_out(b: &[u8]) -> f32 {
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

fn eval_f32(device: Device, g: Graph, inputs: &[(&str, &[u8], DType)]) -> f32 {
    f32_out(&Session::new(device).compile(g).run_typed(inputs)[0].0)
}

fn assert_matches_cpu(
    device: Device,
    g: Graph,
    inputs: &[(&str, &[u8], DType)],
    tol: f32,
    label: &str,
) {
    if !is_available(device) {
        eprintln!("skip directional_nth_gpu_parity {label} on {device:?} (unavailable)");
        return;
    }
    let cpu = eval_f32(Device::Cpu, g.clone(), inputs);
    let gpu = eval_f32(device, g, inputs);
    assert!(
        (cpu - gpu).abs() < tol,
        "{label} {device:?}: cpu={cpu} gpu={gpu} tol={tol}"
    );
}

fn build_sum_squares(n: usize) -> Graph {
    let mut g = Graph::new("sum_sq");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let xx = g.binary(BinaryOp::Mul, x, x, Shape::new(&[n], DType::F32));
    let f = g.reduce(xx, ReduceOp::Sum, vec![0], false, Shape::scalar(DType::F32));
    g.set_outputs(vec![f]);
    g
}

/// f(x)=sum(x²), x ∈ R^n. Two directions v,v → `<v, H v>` with H=2I.
fn directional_second_sum_squares(device: Device) {
    let n = 4;
    let forward = build_sum_squares(n);
    let hg = directional_nth_grad(&forward, "x", &["a", "b"]);
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.5, -0.25, 1.0, -1.5];
    let x_bytes = f32_bytes(&x_data);
    let v_bytes = f32_bytes(&v);
    let inputs = [
        ("x", x_bytes.as_slice(), DType::F32),
        ("dir_0", v_bytes.as_slice(), DType::F32),
        ("dir_1", v_bytes.as_slice(), DType::F32),
    ];
    assert_matches_cpu(device, hg, &inputs, 1e-3, "sum(x²) directional 2nd");
    let want = 2.0 * v.iter().map(|x| x * x).sum::<f32>();
    let cpu = eval_f32(
        Device::Cpu,
        directional_nth_grad(&forward, "x", &["a", "b"]),
        &inputs,
    );
    assert!(
        (cpu - want).abs() < 1e-4,
        "sum(x²) directional 2nd reference: cpu={cpu} want={want}"
    );
}

/// Scalar cubic: directional order-3 matches scalar nth_order on x³.
fn directional_third_x_cubed(device: Device) {
    let mut forward = Graph::new("x3_dir");
    let x = forward.input("x", Shape::scalar(DType::F32));
    let x2 = forward.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F32));
    let x3 = forward.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F32));
    forward.set_outputs(vec![x3]);

    let x_val = 1.5f32;
    let x_bytes = f32_bytes(&[x_val]);
    let dir = f32_bytes(&[1.0]);
    let inputs = [
        ("x", x_bytes.as_slice(), DType::F32),
        ("dir_0", dir.as_slice(), DType::F32),
        ("dir_1", dir.as_slice(), DType::F32),
        ("dir_2", dir.as_slice(), DType::F32),
    ];
    let hg = directional_nth_grad(&forward, "x", &["u", "v", "w"]);
    assert_matches_cpu(device, hg, &inputs, 1e-3, "x³ directional 3rd");
    let want = 6.0f32;
    let cpu = eval_f32(
        Device::Cpu,
        directional_nth_grad(&forward, "x", &["u", "v", "w"]),
        &inputs,
    );
    assert!(
        (cpu - want).abs() < 1e-3,
        "x³ directional 3rd reference: cpu={cpu} want={want}"
    );
}

macro_rules! directional_parity_suite {
    ($mod_name:ident, $device:expr, $($cfg:meta),+) => {
        $(#[$cfg])*
        mod $mod_name {
            use super::*;
            #[test]
            fn sum_squares_directional_second() {
                directional_second_sum_squares($device);
            }
            #[test]
            fn x_cubed_directional_third() {
                directional_third_x_cubed($device);
            }
        }
    };
}

directional_parity_suite!(cuda, Device::Cuda, cfg(feature = "cuda"));
directional_parity_suite!(rocm, Device::Rocm, cfg(feature = "rocm"));
directional_parity_suite!(wgpu, Device::Gpu, cfg(feature = "gpu"));
directional_parity_suite!(
    metal,
    Device::Metal,
    cfg(all(feature = "metal", target_os = "macos"))
);
directional_parity_suite!(
    mlx,
    Device::Mlx,
    cfg(all(feature = "mlx", target_os = "macos"))
);
