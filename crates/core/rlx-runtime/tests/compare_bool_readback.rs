// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `Compare` mask read back through `run()` on every backend. The CPU
//! writes a Bool output one byte per element; reading that slot as f32 strode
//! four elements per value, so on x86 a 64-element `Eq` came back with 23
//! wrong entries (and read past the slot). Sizes cross the SIMD widths.

mod common;

use rlx_ir::op::CmpOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn mask(device: Device, n: usize, cmp: CmpOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("compare_readback");
    let x = g.input("a", Shape::new(&[n], DType::F32));
    let y = g.input("b", Shape::new(&[n], DType::F32));
    let m = g.add_node(Op::Compare(cmp), vec![x, y], Shape::new(&[n], DType::Bool));
    g.set_outputs(vec![m]);
    Session::new(device)
        .compile(g)
        .run(&[("a", a), ("b", b)])
        .pop()
        .unwrap()
}

#[test]
fn compare_mask_reads_back_per_element() {
    let _gpu = common::serialize_gpu();
    for device in rlx_runtime::available_devices() {
        if device == Device::Xdna {
            continue; // covered through its own chain tests; NPU contexts are scarce
        }
        for n in [7usize, 16, 33, 64, 1000] {
            let a: Vec<f32> = (0..n).map(|j| ((j * 7) % 5) as f32).collect();
            let b: Vec<f32> = (0..n).map(|j| (j % 3) as f32).collect();
            for cmp in [CmpOp::Eq, CmpOp::Lt, CmpOp::Ge] {
                let got = mask(device, n, cmp, &a, &b);
                assert_eq!(got.len(), n, "{device:?} {cmp:?} n={n}: length");
                let bad: Vec<usize> = (0..n)
                    .filter(|&i| {
                        let want = match cmp {
                            CmpOp::Eq => a[i] == b[i],
                            CmpOp::Lt => a[i] < b[i],
                            _ => a[i] >= b[i],
                        };
                        (got[i] != 0.0) != want
                    })
                    .collect();
                assert!(bad.is_empty(), "{device:?} {cmp:?} n={n}: wrong at {bad:?}");
            }
        }
    }
}
