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

//! Precision API basic test — verifies Session::new_with_precision()
//! plumbs through to backends. Doesn't yet require all kernels to have
//! F16 implementations (Metal F16 currently falls back to F32).

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Precision, Session};

    let build = || {
        let mut g = Graph::new("basic");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let bias = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(op::Activation::Gelu, bias, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let b_data = vec![0.5, -0.5, 0.0];
    let x_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    for &(label, prec) in &[("F32", Precision::F32), ("F16", Precision::F16)] {
        for &(dev_label, dev) in &[("CPU", Device::Cpu), ("Metal", Device::Metal)] {
            let session = Session::new_with_precision(dev, prec);
            let mut compiled = session.compile(build());
            compiled.set_param("w", &w_data);
            compiled.set_param("b", &b_data);
            let out = compiled.run(&[("x", &x_data)]);
            println!(
                "[{:>3} {:>5}] precision={} device={} output={:?}",
                dev_label,
                label,
                session.precision(),
                session.device(),
                out[0]
            );
        }
    }

    println!();
    println!("Precision API plumbing works — Session::new_with_precision() routes through.");
    println!("Note: F16 Metal currently falls back to F32 (full f16 kernel set is WIP).");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires --features metal on macOS");
}
