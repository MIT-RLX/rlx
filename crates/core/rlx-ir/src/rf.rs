// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RF / complex-scalar graph builders — shared by EDA inverse-design crates.
//!
//! All ops are scalar `f32` rank-1 tensors (`Shape::new(&[1], F32)`).

use crate::graph::Graph;
use crate::op::BinaryOp;
use crate::{DType, NodeId, Op, Shape};

/// Scalar `f32` shape `[1]`.
#[inline]
pub fn scalar_f32() -> Shape {
    Shape::new(&[1], DType::F32)
}

/// Insert a literal `f32` constant node, `val` repeated over `shape`.
///
/// [`Op::Constant`] carries the **whole tensor** (`num_elements * 4` bytes).
/// Backends copy `min(data, buffer)` floats and leave the remainder at zero, so
/// a bare 4-byte literal paired with a wider shape fills element 0 and silently
/// zeroes the rest — wrong numbers, not an error. This fills the shape.
///
/// For a value that should *broadcast* rather than be materialized, pass
/// [`scalar_f32`] and let the consuming binary op broadcast it — that keeps the
/// constant 4 bytes instead of `num_elements * 4`.
///
/// # Panics
///
/// If `shape` has a dynamic dimension: the element count is unknown at build
/// time, so the tensor cannot be filled. Use [`scalar_f32`] + broadcast there.
pub fn const_f32(g: &mut Graph, val: f32, shape: Shape) -> NodeId {
    let n = shape.num_elements().unwrap_or_else(|| {
        panic!(
            "rf::const_f32: shape {shape:?} has a dynamic dimension, so the \
             constant cannot be filled — use rf::scalar_f32() and let the \
             consuming op broadcast it"
        )
    });
    let le = val.to_le_bytes();
    let mut data = Vec::with_capacity(n * 4);
    for _ in 0..n {
        data.extend_from_slice(&le);
    }
    g.add_node(Op::Constant { data }, vec![], shape)
}

/// `|z|²` for complex `z = re + j·im`.
pub fn mag2(g: &mut Graph, re: NodeId, im: NodeId, shape: Shape) -> NodeId {
    let re2 = g.binary(BinaryOp::Mul, re, re, shape.clone());
    let im2 = g.binary(BinaryOp::Mul, im, im, shape.clone());
    g.binary(BinaryOp::Add, re2, im2, shape)
}

/// CS + source degeneration: `Z_in = R_in + j·X_in` at `freq_hz`.
///
/// `R_in = (gm·Ls)/Cgs`, `X_in = ω(Lg+Ls) − 1/(ω·Cgs)`.
pub fn cs_degen_z_in(
    g: &mut Graph,
    gm: NodeId,
    cgs: NodeId,
    lg: NodeId,
    ls: NodeId,
    freq_hz: NodeId,
) -> (NodeId, NodeId) {
    let s = scalar_f32();
    let two_pi = const_f32(g, std::f32::consts::TAU, s.clone());
    let omega = g.binary(BinaryOp::Mul, two_pi, freq_hz, s.clone());
    let gm_ls = g.binary(BinaryOp::Mul, gm, ls, s.clone());
    let r_in = g.binary(BinaryOp::Div, gm_ls, cgs, s.clone());
    let lg_plus_ls = g.binary(BinaryOp::Add, lg, ls, s.clone());
    let omega_l = g.binary(BinaryOp::Mul, omega, lg_plus_ls, s.clone());
    let omega_cgs = g.binary(BinaryOp::Mul, omega, cgs, s.clone());
    let one = const_f32(g, 1.0, s.clone());
    let one_over_wc = g.binary(BinaryOp::Div, one, omega_cgs, s.clone());
    let x_in = g.binary(BinaryOp::Sub, omega_l, one_over_wc, s);
    (r_in, x_in)
}

/// `S11 = (Z − Z0)/(Z + Z0)` for `Z = z_re + j·z_im`.
pub fn s11_from_z(g: &mut Graph, z_re: NodeId, z_im: NodeId, z0: f32) -> (NodeId, NodeId) {
    let s = scalar_f32();
    let z0_n = const_f32(g, z0, s.clone());
    let num_re = g.binary(BinaryOp::Sub, z_re, z0_n, s.clone());
    let num_im = z_im;
    let den_re = g.binary(BinaryOp::Add, z_re, z0_n, s.clone());
    let den_im = z_im;
    complex_div(g, num_re, num_im, den_re, den_im, s)
}

/// Complex division `(nr + j·ni) / (dr + j·di)` → `(re, im)`.
///
/// Uses the stable form `(a+jb)/(c+jd) = ((ac+bd) + j(bc−ad)) / (c²+d²)`.
pub fn complex_div(
    g: &mut Graph,
    nr: NodeId,
    ni: NodeId,
    dr: NodeId,
    di: NodeId,
    shape: Shape,
) -> (NodeId, NodeId) {
    let ac = g.binary(BinaryOp::Mul, nr, dr, shape.clone());
    let bd = g.binary(BinaryOp::Mul, ni, di, shape.clone());
    let bc = g.binary(BinaryOp::Mul, ni, dr, shape.clone());
    let ad = g.binary(BinaryOp::Mul, nr, di, shape.clone());

    let num_re = g.binary(BinaryOp::Add, ac, bd, shape.clone());
    let num_im = g.binary(BinaryOp::Sub, bc, ad, shape.clone());

    let c2 = g.binary(BinaryOp::Mul, dr, dr, shape.clone());
    let d2 = g.binary(BinaryOp::Mul, di, di, shape.clone());
    let denom = g.binary(BinaryOp::Add, c2, d2, shape.clone());

    let re = g.binary(BinaryOp::Div, num_re, denom, shape.clone());
    let im = g.binary(BinaryOp::Div, num_im, denom, shape);
    (re, im)
}

/// Find `Op::Param` node id by name.
pub fn find_param_node(g: &Graph, name: &str) -> Option<NodeId> {
    g.nodes().iter().enumerate().find_map(|(i, n)| match &n.op {
        Op::Param { name: pname, .. } if pname == name => Some(NodeId(i as u32)),
        _ => None,
    })
}

/// Resolve param nodes in the same order as `names`.
pub fn find_param_nodes(g: &Graph, names: &[&str]) -> Result<Vec<NodeId>, String> {
    names
        .iter()
        .map(|n| find_param_node(g, n).ok_or_else(|| format!("param not found in graph: {n}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn complex_div_builds_nodes() {
        let mut g = Graph::new("div");
        let s = scalar_f32();
        let nr = const_f32(&mut g, 1.0, s.clone());
        let ni = const_f32(&mut g, 0.0, s.clone());
        let dr = const_f32(&mut g, 2.0, s.clone());
        let di = const_f32(&mut g, 0.0, s.clone());
        let (re, im) = complex_div(&mut g, nr, ni, dr, di, s);
        g.set_outputs(vec![re, im]);
        assert_eq!(g.outputs.len(), 2);
        assert!(g.len() > 4);
    }

    /// `Op::Constant` carries the whole tensor. A 4-byte literal paired with a
    /// wider shape used to fill element 0 and leave the rest at zero — silently
    /// wrong numbers, since backends copy `min(data, buffer)` floats.
    #[test]
    fn const_f32_fills_the_whole_shape() {
        let mut g = Graph::new("fill");
        let id = const_f32(&mut g, 3.0, Shape::new(&[2, 4], DType::F32));
        let Op::Constant { data } = &g.node(id).op else {
            panic!("expected a Constant");
        };
        assert_eq!(data.len(), 2 * 4 * 4, "must carry num_elements * 4 bytes");
        let vals: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![3.0; 8], "every element must be the literal");
    }

    /// The scalar case is unchanged — still 4 bytes, so a broadcast constant
    /// does not grow into a materialized tensor.
    #[test]
    fn const_f32_scalar_stays_four_bytes() {
        let mut g = Graph::new("scalar");
        let id = const_f32(&mut g, 1.5, scalar_f32());
        let Op::Constant { data } = &g.node(id).op else {
            panic!("expected a Constant");
        };
        assert_eq!(data.len(), 4);
    }

    #[test]
    #[should_panic(expected = "dynamic dimension")]
    fn const_f32_rejects_dynamic_shape() {
        let mut g = Graph::new("dyn");
        let shape = Shape::new(&[1], DType::F32)
            .with_dim(0, crate::shape::Dim::Dynamic(crate::dynamic::sym::BATCH));
        let _ = const_f32(&mut g, 1.0, shape);
    }
}
