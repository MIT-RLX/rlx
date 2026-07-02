// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Higher-order reverse-mode AD — stack `grad_with_loss` with backward
//! decomposition for 2nd/3rd/4th derivatives.

use rlx_ir::{Graph, Shape};

use crate::autodiff::grad_with_loss;
use crate::compose::{
    cse, find_input_by_name, internalize_d_output, output_depends_on_differentiable,
    peel_scalar_expands, zero_derivative_graph,
};
use crate::decompose_backward::{
    contract_grad_with_direction, decompose_backward_for_ad, decompose_backward_ops,
};

/// Opt-in elementwise fusion after higher-order stacking.
pub fn fuse_elementwise(g: Graph) -> Graph {
    use rlx_fusion::Pass;
    rlx_fusion::MarkElementwiseRegions.run(g)
}

/// Options for [`nth_order_grad_with_options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HigherOrderOptions {
    /// Run elementwise fusion after each differentiation layer (default: on).
    pub fuse_elementwise: bool,
}

impl HigherOrderOptions {
    /// Default options: elementwise fusion enabled unless `RLX_HIGHER_ORDER_NO_FUSE=1`.
    pub fn new() -> Self {
        Self {
            fuse_elementwise: !rlx_ir::env::flag("RLX_HIGHER_ORDER_NO_FUSE"),
        }
    }
}

/// Like [`nth_order_grad`] with optional post-layer fusion.
pub fn nth_order_grad_with_options(
    forward: &Graph,
    wrt_name: &str,
    order: usize,
    opts: HigherOrderOptions,
) -> Graph {
    nth_order_grad_inner(forward, wrt_name, order, opts.fuse_elementwise)
}

/// Scalar `wrt`, scalar output: differentiate `order` times.
pub fn nth_order_grad(forward: &Graph, wrt_name: &str, order: usize) -> Graph {
    nth_order_grad_inner(
        forward,
        wrt_name,
        order,
        !rlx_ir::env::flag("RLX_HIGHER_ORDER_NO_FUSE"),
    )
}

fn nth_order_grad_inner(forward: &Graph, wrt_name: &str, order: usize, do_fuse: bool) -> Graph {
    assert_eq!(
        forward.outputs.len(),
        1,
        "nth_order_grad: forward must have exactly one output"
    );
    let wrt = find_input_by_name(forward, wrt_name)
        .unwrap_or_else(|| panic!("nth_order_grad: no Input/Param named '{wrt_name}'"));
    let dtype = forward.node(wrt).shape.dtype();
    let loss = forward.outputs[0];
    if order == 0 {
        let mut g = forward.clone();
        g.set_outputs(vec![loss]);
        return g;
    }
    if !output_depends_on_differentiable(forward, loss, wrt) {
        return zero_derivative_graph(&format!("{}_d{order}_zero", forward.name), wrt_name, dtype);
    }

    let mut g = forward.clone();
    for layer in 0..order {
        let wrt_id = find_input_by_name(&g, wrt_name).expect("wrt input preserved");
        if layer > 0 && !output_depends_on_differentiable(&g, g.outputs[0], wrt_id) {
            return zero_derivative_graph(
                &format!("{}_d{order}_zero", forward.name),
                wrt_name,
                dtype,
            );
        }
        let grad_g = grad_with_loss(&g, &[wrt_id]);
        g = decompose_backward_for_ad(grad_g, 0);
        g = cse(g);
        g = peel_scalar_expands(g);
        if do_fuse {
            g = fuse_elementwise(g);
        }
        g.name = format!("{}_d{}", forward.name, layer + 1);
    }
    g
}

/// ND wrt via per-level direction contraction.
///
/// `directions.len()` is the derivative order. Each direction is exposed as
/// `"dir_<level>"`. Order-2 with the same direction twice yields `<v, H v>`.
pub fn directional_nth_grad(forward: &Graph, wrt_name: &str, directions: &[&str]) -> Graph {
    assert_eq!(
        forward.outputs.len(),
        1,
        "directional_nth_grad: forward must have exactly one output"
    );
    let order = directions.len();
    let wrt = find_input_by_name(forward, wrt_name)
        .unwrap_or_else(|| panic!("directional_nth_grad: no Input/Param named '{wrt_name}'"));
    let dtype = forward.node(wrt).shape.dtype();
    let wrt_shape = forward.node(wrt).shape.clone();
    let loss = forward.outputs[0];
    if order == 0 {
        let mut g = forward.clone();
        g.set_outputs(vec![loss]);
        return g;
    }
    if !output_depends_on_differentiable(forward, loss, wrt) {
        return zero_derivative_graph(
            &format!("{}_dir_d{order}_zero", forward.name),
            wrt_name,
            dtype,
        );
    }

    let mut g = forward.clone();
    let fuse = !rlx_ir::env::flag("RLX_HIGHER_ORDER_NO_FUSE");
    for (level, _dir_name) in directions.iter().enumerate() {
        let wrt_id = find_input_by_name(&g, wrt_name).expect("wrt input preserved");
        if level > 0 && !output_depends_on_differentiable(&g, g.outputs[0], wrt_id) {
            return zero_derivative_graph(
                &format!("{}_dir_d{order}_zero", forward.name),
                wrt_name,
                dtype,
            );
        }
        let grad_g = grad_with_loss(&g, &[wrt_id]);
        let grad_out = grad_g.outputs[1];

        let mut contracted = decompose_backward_ops(grad_g);
        internalize_d_output(&mut contracted);
        let dir_input = contracted.input(
            format!("dir_{level}"),
            if wrt_shape.rank() == 0 {
                Shape::scalar(dtype)
            } else {
                wrt_shape.clone()
            },
        );
        let scalar = contract_grad_with_direction(&mut contracted, grad_out, dir_input);
        contracted.set_outputs(vec![scalar]);
        g = cse(contracted);
        g = peel_scalar_expands(g);
        if fuse {
            g = fuse_elementwise(g);
        }
        g.name = format!("{}_dir_d{}", forward.name, level + 1);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Graph, Op, Shape};

    /// A piecewise-linear op (MaxPool) has a genuinely-zero second
    /// derivative: after the first backward is decomposed, the only path
    /// back to `x` runs through a `Compare` mask, which elementwise fusion
    /// folds into an opaque `ElementwiseRegion`. The higher-order driver
    /// must see through that region and short-circuit to zero rather than
    /// hand a no-gradient graph to reverse mode (which would panic).
    #[test]
    fn maxpool_second_order_is_zero_not_panic() {
        use rlx_ir::op::ReduceOp;
        let f = DType::F32;
        let mut g = Graph::new("pool_ho");
        let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
        let p = g.add_node(
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![x],
            Shape::new(&[1, 1, 2, 2], f),
        );
        let loss = g.reduce(p, ReduceOp::Sum, vec![0, 1, 2, 3], false, Shape::scalar(f));
        g.set_outputs(vec![loss]);
        // Must not panic; second derivative collapses to a zero constant.
        let hg = nth_order_grad(&g, "x", 2);
        assert_eq!(hg.outputs.len(), 1);
        assert!(matches!(hg.node(hg.outputs[0]).op, Op::Constant { .. }));
    }

    #[test]
    fn nth_order_x_cubed_graph_shape() {
        let mut g = Graph::new("x3");
        let x = g.input("x", Shape::scalar(DType::F64));
        let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
        let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F64));
        g.set_outputs(vec![x3]);

        let g3 = nth_order_grad(&g, "x", 3);
        assert_eq!(g3.outputs.len(), 1);
        assert!(find_input_by_name(&g3, "d_output").is_none());
    }

    #[test]
    fn nth_order_f16_bf16_graph_builds() {
        for dt in [DType::F16, DType::BF16] {
            let mut g = Graph::new("x3_lp");
            let x = g.input("x", Shape::scalar(dt));
            let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(dt));
            let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(dt));
            g.set_outputs(vec![x3]);
            let g3 = nth_order_grad(&g, "x", 3);
            assert_eq!(g3.node(g3.outputs[0]).shape.dtype(), dt);
        }
    }

    #[test]
    fn relu_higher_order_builds() {
        use rlx_ir::op::Activation;

        let mut g = Graph::new("relu");
        let x = g.input("x", Shape::scalar(DType::F64));
        let y = g.activation(Activation::Relu, x, Shape::scalar(DType::F64));
        g.set_outputs(vec![y]);

        let g2 = nth_order_grad(&g, "x", 2);
        let g3 = nth_order_grad(&g, "x", 3);
        assert_eq!(g2.outputs.len(), 1);
        assert_eq!(g3.outputs.len(), 1);
        assert!(find_input_by_name(&g3, "d_output").is_none());
    }

    #[test]
    fn directional_scalar_x_cubed_third() {
        use rlx_ir::op::BinaryOp;

        let mut g = Graph::new("x3_dir");
        let x = g.input("x", Shape::scalar(DType::F64));
        let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
        let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F64));
        g.set_outputs(vec![x3]);

        let hg = directional_nth_grad(&g, "x", &["a", "b", "c"]);
        assert_eq!(hg.outputs.len(), 1);
        assert!(find_input_by_name(&hg, "dir_0").is_some());
    }

    #[test]
    fn unreachable_compare_path_short_circuits() {
        use rlx_ir::infer::GraphExt;
        use rlx_ir::op::CmpOp;

        let mut g = Graph::new("cmp");
        let x = g.input("x", Shape::scalar(DType::F64));
        let zero = g.add_node(
            crate::compose::constant_zero(&Shape::scalar(DType::F64)),
            vec![],
            Shape::scalar(DType::F64),
        );
        let cmp = g.add_node(
            Op::Compare(CmpOp::Gt),
            vec![x, zero],
            Shape::scalar(DType::F32),
        );
        let out = g.cast(cmp, DType::F64);
        g.set_outputs(vec![out]);

        let g3 = nth_order_grad(&g, "x", 3);
        assert!(matches!(&g3.node(g3.outputs[0]).op, Op::Constant { .. }));
    }
}
