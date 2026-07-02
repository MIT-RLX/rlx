#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
//! Diagnostic: isolate whether the VQ Metal mismatch is `argmin` itself or the
//! upstream distance computation (matmul / row-broadcast subtract).

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(Op::Constant { data: bytes }, vec![], Shape::new(dims, DType::F32))
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn run(dev: Device, build: &dyn Fn(&mut Graph) -> Vec<NodeId>) -> Vec<Vec<f32>> {
    let mut g = Graph::new("d");
    let o = build(&mut g);
    g.set_outputs(o);
    Session::new(dev).compile(g).run_typed(&[]).iter().map(|o| f32s(&o.0)).collect()
}

#[test]
fn argmin_isolated() {
    let build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let d = const_f32(g, &[0.0, -80.0, 80.0, 5.0, 5.0, -1.0, 9.0, 2.0, 3.0], &[3, 3]);
        let s = rlx_ir::shape::reduce_shape(g.shape(d), &[1], false).unwrap();
        let idx = g.argmin(d, 1, false, s);
        vec![idx]
    };
    let c = run(Device::Cpu, build);
    let m = run(Device::Metal, build);
    println!("argmin cpu={:?} metal={:?}", c[0], m[0]);
    assert_eq!(c[0], m[0], "argmin isolated mismatch");
}

#[test]
fn vq_dist_isolated() {
    // Reproduce vector_quantize's L2 dist proxy and expose it + argmin.
    let build: &dyn Fn(&mut Graph) -> Vec<NodeId> = &|g| {
        let cb = const_f32(g, &[0.0, 0.0, 10.0, 0.0, 0.0, 10.0], &[3, 2]);
        let x = const_f32(g, &[9.0, 1.0, 1.0, 9.0, 0.2, 0.1], &[3, 2]);
        let cb_t = g.transpose_(cb, vec![1, 0]);
        let cross = g.mm(x, cb_t);
        let two = g.constant(2.0, DType::F32);
        let two_cross = g.mul(cross, two);
        let cb_sq = g.mul(cb, cb);
        let cb_norm = g.sum(cb_sq, vec![1], false);
        let cb_norm_row = g.reshape_(cb_norm, vec![1, 3]);
        let dist = g.sub(cb_norm_row, two_cross);
        let s = rlx_ir::shape::reduce_shape(g.shape(dist), &[1], false).unwrap();
        let idx = g.argmin(dist, 1, false, s);
        vec![dist, idx]
    };
    let c = run(Device::Cpu, build);
    let m = run(Device::Metal, build);
    println!("dist  cpu={:?}\ndist  metal={:?}", c[0], m[0]);
    println!("idx   cpu={:?} metal={:?}", c[1], m[1]);
    assert_eq!(c[0], m[0], "dist tensor mismatch (upstream of argmin)");
    assert_eq!(c[1], m[1], "argmin mismatch");
}

/// Regression for the Metal broadcast bug: the scalar/row/col/1-axis fast
/// paths swapped operands when the *lhs* was the broadcast side, silently
/// computing `rhs OP lhs` for the non-commutative `Sub`/`Div`/`Pow`.
#[test]
fn noncommutative_lhs_broadcast_parity() {
    let cases: &[(&str, &dyn Fn(&mut Graph) -> Vec<NodeId>)] = &[
        ("sub_row_lhs_broadcast", &|g| {
            let a = const_f32(g, &[10.0, 20.0, 30.0, 40.0], &[1, 4]); // lhs broadcasts rows
            let b = const_f32(g, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[3, 4]);
            vec![g.sub(a, b)]
        }),
        ("sub_col_lhs_broadcast", &|g| {
            let a = const_f32(g, &[1.0, 2.0, 3.0], &[3, 1]); // lhs broadcasts cols
            let b = const_f32(g, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[3, 4]);
            vec![g.sub(a, b)]
        }),
        ("div_row_lhs_broadcast", &|g| {
            let a = const_f32(g, &[12.0, 24.0, 36.0, 48.0], &[1, 4]);
            let b = const_f32(g, &(1..13).map(|i| i as f32).collect::<Vec<_>>(), &[3, 4]);
            vec![g.div(a, b)]
        }),
        ("sub_scalar_lhs_broadcast", &|g| {
            let a = g.constant(100.0, DType::F32); // rank-0 scalar lhs
            let b = const_f32(g, &(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[3, 4]);
            vec![g.sub(a, b)]
        }),
    ];
    for (name, build) in cases {
        let c = run(Device::Cpu, *build);
        let m = run(Device::Metal, *build);
        assert_eq!(c.len(), m.len(), "{name}: output count");
        for (a, b) in c[0].iter().zip(m[0].iter()) {
            let rel = (a - b).abs() / a.abs().max(1e-4);
            assert!(rel < 1e-5, "{name}: cpu={a} metal={b} (rel={rel})");
        }
    }
}
