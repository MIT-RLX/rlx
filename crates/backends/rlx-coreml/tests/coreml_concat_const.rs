// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! CoreML folds a rank ≥ 3 concat of compile-time values wrongly (the first
//! rows, then zeros), so such a concat is lowered at rank 2. These shapes all
//! came back wrong before that.
#![cfg(all(target_vendor = "apple", not(target_os = "watchos")))]

use rlx_coreml::{ComputeUnits, CoremlExecutable};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};

fn constant(g: &mut Graph, v: &[f32], dims: &[usize]) -> rlx_ir::NodeId {
    let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], Shape::new(dims, DType::F32))
}

fn ramp(n: usize, step: f32) -> Vec<f32> {
    (0..n).map(|i| step * (1.0 + i as f32)).collect()
}

/// Row-major concat on `axis`.
fn expect(a: &[f32], b: &[f32], da: &[usize], db: &[usize], axis: usize) -> Vec<f32> {
    let (ia, ib): (usize, usize) = (da[axis..].iter().product(), db[axis..].iter().product());
    a.chunks(ia)
        .zip(b.chunks(ib))
        .flat_map(|(x, y)| x.iter().chain(y).copied())
        .collect()
}

fn out_dims(da: &[usize], db: &[usize], axis: usize) -> Vec<usize> {
    let mut d = da.to_vec();
    d[axis] += db[axis];
    d
}

#[test]
fn constants_concat_at_any_rank() {
    let cases: [(&[usize], &[usize], usize); 7] = [
        (&[1, 12, 8], &[1, 12, 8], 2),
        (&[2, 4, 3, 8], &[2, 4, 3, 8], 3),
        (&[1, 1, 2, 8], &[1, 1, 2, 8], 3),
        (&[3, 8, 8], &[3, 8, 8], 2),
        (&[6, 7, 5], &[6, 7, 8], 2),
        (&[1, 2, 7, 8], &[1, 2, 4, 8], 2),
        (&[1, 2, 4, 2, 6], &[1, 2, 4, 2, 7], 4),
    ];
    for (da, db, axis) in cases {
        let (a, b) = (
            ramp(da.iter().product(), 1.0),
            ramp(db.iter().product(), -1.0),
        );
        for units in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
            let mut g = Graph::new("cat");
            let (x, y) = (constant(&mut g, &a, da), constant(&mut g, &b, db));
            let o = g.add_node(
                Op::Concat { axis },
                vec![x, y],
                Shape::new(&out_dims(da, db, axis), DType::F32),
            );
            g.set_outputs(vec![o]);
            let got = CoremlExecutable::compile_with_units(g, units)
                .run(&[])
                .expect("run")
                .remove(0);
            assert_eq!(
                got,
                expect(&a, &b, da, db, axis),
                "{da:?} + {db:?} on axis {axis}, {units:?}"
            );
        }
    }
}

#[test]
fn computed_constants_concat_too() {
    // the operands are no `Constant` nodes, but all compile-time: cat(x, x − x)
    let d = [2, 4, 3, 8];
    let a = ramp(d.iter().product(), 1.0);
    let mut g = Graph::new("chain");
    let x = constant(&mut g, &a, &d);
    let s = Shape::new(&d, DType::F32);
    let z = g.add_node(Op::Binary(BinaryOp::Sub), vec![x, x], s);
    let o = g.add_node(
        Op::Concat { axis: 3 },
        vec![x, z],
        Shape::new(&[2, 4, 3, 16], DType::F32),
    );
    g.set_outputs(vec![o]);
    let got = CoremlExecutable::compile_with_units(g, ComputeUnits::CpuOnly)
        .run(&[])
        .expect("run")
        .remove(0);
    assert_eq!(got, expect(&a, &vec![0.0; a.len()], &d, &d, 3));
}

#[test]
fn a_runtime_operand_stays_right() {
    let d = [2, 4, 3, 8];
    let (a, b) = (
        ramp(d.iter().product(), 1.0),
        ramp(d.iter().product(), -1.0),
    );
    let mut g = Graph::new("mixed");
    let x = constant(&mut g, &a, &d);
    let y = g.input("y", Shape::new(&d, DType::F32));
    let o = g.add_node(
        Op::Concat { axis: 3 },
        vec![x, y],
        Shape::new(&[2, 4, 3, 16], DType::F32),
    );
    g.set_outputs(vec![o]);
    let got = CoremlExecutable::compile_with_units(g, ComputeUnits::CpuOnly)
        .run(&[("y", &b)])
        .expect("run")
        .remove(0);
    assert_eq!(got, expect(&a, &b, &d, &d, 3));
}

#[test]
fn integer_constants_concat_too() {
    let d = [1usize, 12, 8];
    let n: usize = d.iter().product();
    let (a, b) = (
        (1..=n as i32).collect::<Vec<_>>(),
        (1..=n as i32).map(|v| -v).collect::<Vec<_>>(),
    );
    let mut g = Graph::new("int");
    let mut c = |v: &[i32]| {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.add_node(Op::Constant { data }, vec![], Shape::new(&d, DType::I32))
    };
    let (x, y) = (c(&a), c(&b));
    let o = g.add_node(
        Op::Concat { axis: 2 },
        vec![x, y],
        Shape::new(&[1, 12, 16], DType::I32),
    );
    g.set_outputs(vec![o]);
    let got = CoremlExecutable::compile_with_units(g, ComputeUnits::CpuOnly)
        .run(&[])
        .expect("run")
        .remove(0);
    let (fa, fb): (Vec<f32>, Vec<f32>) = (
        a.iter().map(|&v| v as f32).collect(),
        b.iter().map(|&v| v as f32).collect(),
    );
    assert_eq!(got, expect(&fa, &fb, &d, &d, 2));
}

#[test]
fn bool_constants_concat_too() {
    // MIL has no bool output, so the mask selects between two values
    let d = [1usize, 12, 8];
    let o = [1usize, 12, 16];
    let n: usize = d.iter().product();
    let (a, b): (Vec<bool>, Vec<bool>) = (
        (0..n).map(|i| i % 3 == 0).collect(),
        (0..n).map(|i| i % 5 != 0).collect(),
    );
    let mut g = Graph::new("bool");
    let mut mask = |v: &[bool]| {
        let data = v.iter().map(|&x| x as u8).collect();
        g.add_node(Op::Constant { data }, vec![], Shape::new(&d, DType::Bool))
    };
    let (x, y) = (mask(&a), mask(&b));
    let m = g.add_node(
        Op::Concat { axis: 2 },
        vec![x, y],
        Shape::new(&o, DType::Bool),
    );
    let t = constant(&mut g, &vec![1.0; 2 * n], &o);
    let f = constant(&mut g, &vec![-1.0; 2 * n], &o);
    let w = g.add_node(Op::Where, vec![m, t, f], Shape::new(&o, DType::F32));
    g.set_outputs(vec![w]);
    let got = CoremlExecutable::compile_with_units(g, ComputeUnits::CpuOnly)
        .run(&[])
        .expect("run")
        .remove(0);
    let want: Vec<f32> = a
        .chunks(8)
        .zip(b.chunks(8))
        .flat_map(|(p, q)| p.iter().chain(q))
        .map(|&v| if v { 1.0 } else { -1.0 })
        .collect();
    assert_eq!(got, want);
}

/// No inputs at all, at f16: the constants become hidden inputs, fed on every
/// run. (The program that stalled CoreML's compiler without this, the
/// BGT60TR13C baseband, took over 150 s and now 0.3 s; this small one shows
/// the lifted values arrive right, twice.)
#[test]
fn an_all_constant_f16_program_compiles_and_reruns() {
    let (b, f) = (16usize, 241usize);
    let half = |v: &[f32]| {
        v.iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let (x, y): (Vec<f32>, Vec<f32>) = (
        (0..b).map(|i| 1.0 + i as f32 / 8.0).collect(),
        (0..f).map(|i| 0.5 + i as f32 / 256.0).collect(),
    );
    let mut g = Graph::new("all_const");
    let (sx, sy, so) = (
        Shape::new(&[b, 1], DType::F16),
        Shape::new(&[1, f], DType::F16),
        Shape::new(&[b, f], DType::F16),
    );
    let cx = g.add_node(Op::Constant { data: half(&x) }, vec![], sx);
    let cy = g.add_node(Op::Constant { data: half(&y) }, vec![], sy);
    let mut t = g.add_node(Op::Binary(BinaryOp::Mul), vec![cx, cy], so.clone());
    // multiply and divide by the same y in turn: bounded, f16 holds it
    for i in 0..40 {
        let op = if i % 2 == 0 {
            BinaryOp::Mul
        } else {
            BinaryOp::Div
        };
        let u = g.add_node(Op::Binary(op), vec![t, cy], so.clone());
        t = g.add_node(Op::Binary(BinaryOp::Add), vec![u, cx], so.clone());
    }
    g.set_outputs(vec![t]);
    let mut e = CoremlExecutable::compile(g);
    let t0 = std::time::Instant::now();
    let first = e.run(&[]).expect("run").remove(0);
    assert!(
        t0.elapsed().as_secs() < 60,
        "compile took {:?}",
        t0.elapsed()
    );
    assert_eq!(first.len(), b * f);
    let want: Vec<f32> = x
        .iter()
        .flat_map(|&xi| {
            y.iter().map(move |&yj| {
                let mut t = xi * yj;
                for i in 0..40 {
                    t = if i % 2 == 0 { t * yj } else { t / yj } + xi;
                }
                t
            })
        })
        .collect();
    for (k, (g, w)) in first.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 2e-2 * w.abs(), "[{k}] {g} against {w}");
    }
    assert_eq!(e.run(&[]).expect("rerun").remove(0), first);
}
