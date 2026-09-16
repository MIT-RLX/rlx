// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One D2Q9 streaming step as an rlx [`Graph`].
//!
//! Streaming on a periodic domain is exactly [`Op::Roll`](rlx_ir::Op::Roll): population `i` at
//! cell `x` came from `x − c_i`, and reading a whole field shifted by `−c_i`
//! *is* a cyclic shift. So the entire gather is nine rolls per stored field, and
//! the reconstruction is elementwise arithmetic on top — no custom kernel, no
//! per-backend work, and it runs wherever `Session` runs.
//!
//! This is also the first workload in the tree with a genuine stencil access
//! pattern. Everything else is matmul-shaped, so the fusion and region machinery
//! has never seen a graph like this.
//!
//! Field layout is `[nworld, ny, nx]`: axis 0 indexes parallel worlds (lanes),
//! axis 1 is `y`, axis 2 is `x`. Worlds never interact, so streaming rolls only
//! the spatial axes and the lane axis rides along untouched — which is exactly
//! what makes a `[nworld]` lane mask (see [`rlx_ir::lanes`]) able to reset a
//! subset of worlds without touching the graph.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::lattice::{CS2, D2Q9_C, D2Q9_W, h2, h3};

/// The six stored moment fields of a D2Q9 state, as graph nodes.
#[derive(Debug, Clone, Copy)]
pub struct MomentNodes {
    /// Density.
    pub rho: NodeId,
    /// x velocity.
    pub ux: NodeId,
    /// y velocity.
    pub uy: NodeId,
    /// `S_xx`.
    pub sxx: NodeId,
    /// `S_yy`.
    pub syy: NodeId,
    /// `S_xy`.
    pub sxy: NodeId,
}

impl MomentNodes {
    /// In the canonical order used by [`FIELD_NAMES`] and the graph outputs.
    pub fn to_array(self) -> [NodeId; 6] {
        [self.rho, self.ux, self.uy, self.sxx, self.syy, self.sxy]
    }
}

/// Input / output names, in the order [`MomentNodes::to_array`] uses.
pub const FIELD_NAMES: [&str; 6] = ["rho", "ux", "uy", "sxx", "syy", "sxy"];

/// Multiply `x` by a compile-time scalar.
fn scale(g: &mut Graph, x: NodeId, k: f64) -> NodeId {
    let c = g.constant(k, DType::F32);
    g.mul(x, c)
}

/// `acc + k * x`, skipping the work entirely when `k` is zero — most of the
/// D2Q9 coefficients are, and emitting `+ 0·x` would triple the node count for
/// nothing.
fn acc_scaled(g: &mut Graph, acc: Option<NodeId>, x: NodeId, k: f64) -> Option<NodeId> {
    if k == 0.0 {
        return acc;
    }
    let term = if k == 1.0 { x } else { scale(g, x, k) };
    Some(match acc {
        Some(a) => g.add(a, term),
        None => term,
    })
}

/// Reconstruct population `i` from moment nodes, following Eq. 29 exactly as
/// [`crate::moment::reconstruct_d2q9`] does on the host.
fn reconstruct_dir(g: &mut Graph, m: &MomentNodes, i: usize) -> NodeId {
    let c = &D2Q9_C[i];
    let inv_cs2 = 1.0 / CS2;
    let inv_2cs4 = 1.0 / (2.0 * CS2 * CS2);
    let inv_2cs6 = 3.0 / (2.0 * CS2 * CS2 * CS2);

    // A_xxy = Sxx*uy + 2*Sxy*ux - 2*ux*ux*uy
    let t1 = g.mul(m.sxx, m.uy);
    let t2 = g.mul(m.sxy, m.ux);
    let t2 = scale(g, t2, 2.0);
    let uxux = g.mul(m.ux, m.ux);
    let t3 = g.mul(uxux, m.uy);
    let t3 = scale(g, t3, 2.0);
    let a_xxy = g.add(t1, t2);
    let a_xxy = g.sub(a_xxy, t3);

    // A_xyy = Syy*ux + 2*Sxy*uy - 2*ux*uy*uy
    let s1 = g.mul(m.syy, m.ux);
    let s2 = g.mul(m.sxy, m.uy);
    let s2 = scale(g, s2, 2.0);
    let uyuy = g.mul(m.uy, m.uy);
    let s3 = g.mul(m.ux, uyuy);
    let s3 = scale(g, s3, 2.0);
    let a_xyy = g.add(s1, s2);
    let a_xyy = g.sub(a_xyy, s3);

    // bracket = 1 + (c·u)/cs² + H²:S/(2cs⁴) + 3·H³:A/(2cs⁶)
    let mut br: Option<NodeId> = None;
    br = acc_scaled(g, br, m.ux, c[0] as f64 * inv_cs2);
    br = acc_scaled(g, br, m.uy, c[1] as f64 * inv_cs2);
    br = acc_scaled(g, br, m.sxx, h2(c, 0, 0) * inv_2cs4);
    br = acc_scaled(g, br, m.syy, h2(c, 1, 1) * inv_2cs4);
    br = acc_scaled(g, br, m.sxy, 2.0 * h2(c, 0, 1) * inv_2cs4);
    br = acc_scaled(g, br, a_xxy, h3(c, 0, 0, 1) * inv_2cs6);
    br = acc_scaled(g, br, a_xyy, h3(c, 0, 1, 1) * inv_2cs6);

    let one = g.constant(1.0, DType::F32);
    let bracket = match br {
        Some(b) => g.add(one, b),
        None => one,
    };
    let rw = scale(g, m.rho, D2Q9_W[i]);
    g.mul(rw, bracket)
}

/// Build a graph computing one **streaming** step: gather every population from
/// its upwind neighbour and re-take the moments.
///
/// Returns `(graph, outputs)` where `outputs` are the six post-streaming fields
/// in [`FIELD_NAMES`] order. Collision is deliberately left to the host
/// ([`crate::moment::collide_d2q9`]) — it is pointwise and adds nothing to the
/// access-pattern story this graph exists to exercise.
pub fn stream_step_graph(nx: usize, ny: usize) -> (Graph, Vec<NodeId>) {
    stream_step_graph_batched(1, nx, ny)
}

/// Batched form: `nworld` independent D2Q9 worlds stepped in one graph.
///
/// Fields are `[nworld, ny, nx]`. This is the shape a parallel-environment
/// workload actually has, and the one the lane-mask reset seam is built for.
pub fn stream_step_graph_batched(nworld: usize, nx: usize, ny: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("lbm_d2q9_stream");
    let shape = Shape::new(&[nworld, ny, nx], DType::F32);
    let inputs = MomentNodes {
        rho: g.input("rho", shape.clone()),
        ux: g.input("ux", shape.clone()),
        uy: g.input("uy", shape.clone()),
        sxx: g.input("sxx", shape.clone()),
        syy: g.input("syy", shape.clone()),
        sxy: g.input("sxy", shape.clone()),
    };

    // Accumulators for Σf, Σf·c, Σf·c⊗c.
    let mut acc_rho: Option<NodeId> = None;
    let mut acc_mx: Option<NodeId> = None;
    let mut acc_my: Option<NodeId> = None;
    let mut acc_pxx: Option<NodeId> = None;
    let mut acc_pyy: Option<NodeId> = None;
    let mut acc_pxy: Option<NodeId> = None;

    for i in 0..9 {
        let c = &D2Q9_C[i];
        // Shift the whole state by +c_i so cell x reads the neighbour at x − c_i.
        // Only the spatial axes roll — axis 0 is the lane axis, and rolling it
        // would leak one world's fluid into the next.
        let shifts = vec![c[1] as i64, c[0] as i64];
        let dims = vec![1usize, 2usize];
        let shifted = MomentNodes {
            rho: g.roll_(inputs.rho, shifts.clone(), dims.clone()),
            ux: g.roll_(inputs.ux, shifts.clone(), dims.clone()),
            uy: g.roll_(inputs.uy, shifts.clone(), dims.clone()),
            sxx: g.roll_(inputs.sxx, shifts.clone(), dims.clone()),
            syy: g.roll_(inputs.syy, shifts.clone(), dims.clone()),
            sxy: g.roll_(inputs.sxy, shifts, dims),
        };
        let f = reconstruct_dir(&mut g, &shifted, i);

        acc_rho = acc_scaled(&mut g, acc_rho, f, 1.0);
        acc_mx = acc_scaled(&mut g, acc_mx, f, c[0] as f64);
        acc_my = acc_scaled(&mut g, acc_my, f, c[1] as f64);
        acc_pxx = acc_scaled(&mut g, acc_pxx, f, (c[0] * c[0]) as f64);
        acc_pyy = acc_scaled(&mut g, acc_pyy, f, (c[1] * c[1]) as f64);
        acc_pxy = acc_scaled(&mut g, acc_pxy, f, (c[0] * c[1]) as f64);
    }

    let zero = g.full(&[nworld, ny, nx], 0.0, DType::F32);
    let rho_n = acc_rho.expect("rest direction always contributes");
    let inv_rho = {
        let one = g.constant(1.0, DType::F32);
        g.div(one, rho_n)
    };
    // A direction group can contribute nothing (e.g. Σf·c_x c_y over the axis
    // directions), in which case the accumulator is a constant-zero field.
    let take = |a: Option<NodeId>| -> NodeId { a.unwrap_or(zero) };

    let mx = take(acc_mx);
    let my = take(acc_my);
    let pxx = take(acc_pxx);
    let pyy = take(acc_pyy);
    let pxy = take(acc_pxy);

    let ux_n = g.mul(mx, inv_rho);
    let uy_n = g.mul(my, inv_rho);
    let cs2c = g.constant(CS2, DType::F32);
    let sxx_n = {
        let t = g.mul(pxx, inv_rho);
        g.sub(t, cs2c)
    };
    let syy_n = {
        let t = g.mul(pyy, inv_rho);
        g.sub(t, cs2c)
    };
    let sxy_n = g.mul(pxy, inv_rho);

    let outs = vec![rho_n, ux_n, uy_n, sxx_n, syy_n, sxy_n];
    g.set_outputs(outs.clone());
    (g, outs)
}
