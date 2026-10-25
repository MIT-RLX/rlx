// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cpu")]

//! I32 / I64 tensors on CPU mean the same integers whichever way they come
//! in: as their own bytes (`run_typed`) or as f32 values (`run`), and through
//! ops whose kernels read integers (casts) as well as ops that compute in f32
//! (arithmetic, compare, gather indices) or have no integer arm (reduce).
//! Before, the two kernel families read each other's bits: a cast after an
//! integer multiply gave `1065353216` for `1`, token ids fed as integers all
//! selected row 0, and an I64 fed through `run` was half a value.

use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// One small graph per op: `x` is `[1, 2, 3, 4]` of `dt`; the output as f64.
fn case(name: &str, dt: DType, typed: bool) -> Vec<f64> {
    let i = |d: &[usize]| Shape::new(d, dt);
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new(name);
    let x = g.input("x", i(&[4]));
    let y = match name {
        "add" => g.add_node(Op::Binary(BinaryOp::Add), vec![x, x], i(&[4])),
        "mul_then_cast" => {
            let m = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, x], i(&[4]));
            g.add_node(Op::Cast { to: DType::F32 }, vec![m], f(&[4]))
        }
        "cast" => g.add_node(Op::Cast { to: DType::F32 }, vec![x], f(&[4])),
        "compare" => {
            let c = g.add_node(
                Op::Compare(CmpOp::Lt),
                vec![x, x],
                Shape::new(&[4], DType::Bool),
            );
            g.add_node(Op::Cast { to: DType::F32 }, vec![c], f(&[4]))
        }
        "gather" => {
            let t = g.param("t", f(&[5, 1]));
            g.gather(t, x, 0, f(&[4, 1]))
        }
        "sum" => g.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![0],
                keep_dim: false,
            },
            vec![x],
            i(&[]),
        ),
        "narrow" => g.add_node(
            Op::Narrow {
                axis: 0,
                start: 1,
                len: 2,
            },
            vec![x],
            i(&[2]),
        ),
        _ => x,
    };
    g.set_outputs(vec![y]);
    let out_dt = g.node(y).shape.dtype();
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("t", &[10.0, 11.0, 12.0, 13.0, 14.0]);
    if !typed {
        return c
            .run(&[("x", &[1.0, 2.0, 3.0, 4.0])])
            .remove(0)
            .iter()
            .map(|v| *v as f64)
            .collect();
    }
    let bytes: Vec<u8> = match dt {
        DType::I64 => [1i64, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::U32 => [1u32, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => [1i32, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    };
    let (out, dt_out) = c.run_typed(&[("x", &bytes, dt)]).remove(0);
    assert_eq!(dt_out, out_dt);
    match dt_out {
        DType::I64 => out
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b) as f64)
            .collect(),
        DType::I32 => out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f64)
            .collect(),
        DType::U32 => out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b) as f64)
            .collect(),
        _ => out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect(),
    }
}

#[test]
fn integers_mean_the_same_through_every_kind_of_kernel_and_both_feeds() {
    let want: &[(&str, &[f64])] = &[
        ("identity", &[1.0, 2.0, 3.0, 4.0]),
        ("add", &[2.0, 4.0, 6.0, 8.0]),
        ("mul_then_cast", &[1.0, 4.0, 9.0, 16.0]),
        ("cast", &[1.0, 2.0, 3.0, 4.0]),
        ("compare", &[0.0, 0.0, 0.0, 0.0]),
        ("gather", &[11.0, 12.0, 13.0, 14.0]),
        ("sum", &[10.0]),
        ("narrow", &[2.0, 3.0]),
    ];
    let mut wrong = Vec::new();
    for dt in [DType::I32, DType::U32, DType::I64] {
        for typed in [true, false] {
            for (name, w) in want {
                let got = case(name, dt, typed);
                if got != *w {
                    wrong.push(format!(
                        "{dt:?} {} {name}: {got:?}, want {w:?}",
                        if typed { "run_typed" } else { "run" }
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
