// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_ir::{DType, Dim, NodeId};
use rlx_tensor::{GraphScope, Tensor, ax, graph, graph_with, rg, s, shape};

#[test]
fn mlp_matches_pyrlx_quickstart() {
    let g = graph("mlp", |g| {
        let x = g.input("x", shape![2, 4]);
        let w = g.param("w", shape![4, 3]);
        let b = g.param("b", shape![3]);
        (&x.matmul(&w) + &b).gelu() * 2.0
    });
    assert_eq!(g.name, "mlp");
    assert_eq!(g.outputs.len(), 1);
    assert!(g.len() >= 5);
}

#[test]
fn scalar_promotion_and_left_scalar() {
    let g = graph("arith", |g| {
        let x = g.input("x", shape![4]);
        let two = g.scalar(2.0);
        let one = g.scalar(1.0);
        (&x * &two + &one).relu()
    });
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn graph_with_multiple_outputs() {
    let (g, (a_id, b_id)) = graph_with("pair", |g| {
        let x = g.input("x", shape![2, 2]);
        let w = g.param("w", shape![2, 2]);
        let a = x.matmul(&w);
        let b = a.relu();
        g.set_outputs([&a, &b]);
        (a.id(), b.id())
    });
    assert_eq!(g.outputs.len(), 2);
    assert_eq!(g.outputs[0], a_id);
    assert_eq!(g.outputs[1], b_id);
}

#[test]
fn tensor_shape_and_dtype() {
    let mut scope = GraphScope::new("meta");
    let x = scope.input("x", shape![F32; 2, 3]);
    assert_eq!(x.shape().rank(), 2);
    assert_eq!(x.dtype(), DType::F32);
    assert!(x.is_static_shape());
    drop(x);
    let _g = scope.finish();
}

#[test]
fn dynamic_shape_macro() {
    let s = shape![?, 128];
    assert!(!s.is_static());
    assert_eq!(s.dims()[0], Dim::Dynamic(0));
    assert_eq!(s.dims()[1], Dim::Static(128));
}

#[test]
fn slice_lowers_to_narrow() {
    let g = graph("win", |g| {
        let x = g.input("x", shape![4, 16]);
        x.slice(s![ax(), rg(2, 10)])
    });
    assert_eq!(g.outputs.len(), 1);
    assert!(g.len() >= 2);
}

#[test]
fn concat_is_symbolic() {
    let g = graph("cat", |g| {
        let a = g.input("a", shape![2, 4]);
        let b = g.input("b", shape![2, 4]);
        g.cat(&[&a, &b], 0)
    });
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn tensor_into_node_id() {
    let g = graph("id", |g| {
        let x = g.input("x", shape![4]);
        let id: NodeId = (&x).into();
        assert_eq!(id, x.id());
        x
    });
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn extra_activations_preserve_shape() {
    let mut scope = GraphScope::new("acts");
    let x = scope.input("x", shape![2, 3]);
    for y in [
        x.sigmoid(),
        x.log(),
        x.rsqrt(),
        x.abs(),
        x.sin(),
        x.cos(),
        x.tan(),
        x.atan(),
        x.round(),
    ] {
        assert_eq!(y.shape().dims(), x.shape().dims());
        assert_eq!(y.dtype(), DType::F32);
    }
}

#[test]
fn reductions_drop_axis() {
    let mut scope = GraphScope::new("reduce");
    let x = scope.input("x", shape![2, 4, 8]);
    for y in [x.max([2], false), x.min([2], false), x.prod([2], false)] {
        assert_eq!(y.rank(), 2);
        assert_eq!(y.shape().dims(), &[Dim::Static(2), Dim::Static(4)]);
    }
    let kept = x.max([2], true);
    assert_eq!(
        kept.shape().dims(),
        &[Dim::Static(2), Dim::Static(4), Dim::Static(1)]
    );
}

#[test]
fn comparisons_yield_bool() {
    let mut scope = GraphScope::new("cmp");
    let a = scope.input("a", shape![4]);
    let b = scope.input("b", shape![4]);
    for y in [a.eq(&b), a.ne(&b), a.lt(&b), a.le(&b), a.gt(&b), a.ge(&b)] {
        assert_eq!(y.dtype(), DType::Bool);
        assert_eq!(y.rank(), 1);
    }
}

#[test]
fn clamp_and_minmax_keep_shape() {
    let g = graph("clamp", |g| {
        let x = g.input("x", shape![3, 5]);
        let lo = g.param("lo", shape![3, 5]);
        let c = x.clamp(0.0, 6.0);
        assert_eq!(c.shape().dims(), &[Dim::Static(3), Dim::Static(5)]);
        c.maximum(&lo).minimum(&lo)
    });
    assert_eq!(g.outputs.len(), 1);
}

#[test]
fn shape_manipulation_dims() {
    let x = Tensor::from_vec(vec![0.0; 6], [2, 3]);
    assert_eq!(x.flatten().shape().dims(), &[Dim::Static(6)]);
    assert_eq!(
        x.unsqueeze(0).shape().dims(),
        &[Dim::Static(1), Dim::Static(2), Dim::Static(3)]
    );
    assert_eq!(
        x.unsqueeze(2).shape().dims(),
        &[Dim::Static(2), Dim::Static(3), Dim::Static(1)]
    );
    let y = Tensor::from_vec(vec![0.0; 3], [1, 3, 1]);
    assert_eq!(
        y.squeeze(0).shape().dims(),
        &[Dim::Static(3), Dim::Static(1)]
    );
    assert_eq!(y.squeeze_all().shape().dims(), &[Dim::Static(3)]);
    assert_eq!(
        x.broadcast_to([4, 2, 3]).shape().dims(),
        &[Dim::Static(4), Dim::Static(2), Dim::Static(3)]
    );
}

#[test]
fn cat_and_stack_shapes() {
    use rlx_tensor::{cat, stack};
    let a = Tensor::from_vec(vec![1.0, 2.0], [2]);
    let b = Tensor::from_vec(vec![3.0, 4.0], [2]);
    assert_eq!(cat(&[&a, &b], 0).shape().dims(), &[Dim::Static(4)]);
    assert_eq!(
        stack(&[&a, &b], 0).shape().dims(),
        &[Dim::Static(2), Dim::Static(2)]
    );
}

#[test]
fn clone_and_slice_are_zero_copy() {
    let a = Tensor::from_vec(vec![0.0; 1000], [1000]); // 4000 bytes
    assert_eq!(a.storage_bytes(), 4000);

    // clone: same graph, no extra data.
    let c = a.clone();
    assert!(c.shares_graph(&a));
    assert_eq!(c.storage_bytes(), 4000);

    // slice: a view into the same graph; adds no payload bytes.
    let s = a.slice(s![rg(10, 20)]);
    assert!(s.shares_graph(&a), "slice must share storage");
    assert_eq!(s.storage_bytes(), 4000, "slice must copy zero bytes");
    assert_eq!(s.dims(), vec![10]);

    // reshape is likewise a zero-copy view.
    let r = a.reshape(vec![10_i64, 100]);
    assert!(r.shares_graph(&a));
    assert_eq!(r.storage_bytes(), 4000);
}

#[test]
fn independent_tensors_do_not_share() {
    let a = Tensor::from_vec(vec![1.0, 2.0], [2]);
    let b = Tensor::from_vec(vec![3.0, 4.0], [2]);
    assert!(!a.shares_graph(&b));
    // Combining merges b into a's graph (the one copy), result shares a.
    let c = &a + &b;
    assert!(c.shares_graph(&a));
}

#[test]
fn tensor_clone_is_shallow() {
    let g = graph("clone", |g| {
        let x = g.input("x", shape![4]);
        let y = x.clone();
        &x + &y
    });
    assert_eq!(g.outputs.len(), 1);
}
