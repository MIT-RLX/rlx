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

/// Insert a literal `f32` constant node.
pub fn const_f32(g: &mut Graph, val: f32, shape: Shape) -> NodeId {
    g.add_node(
        Op::Constant {
            data: val.to_le_bytes().to_vec(),
        },
        vec![],
        shape,
    )
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
}
