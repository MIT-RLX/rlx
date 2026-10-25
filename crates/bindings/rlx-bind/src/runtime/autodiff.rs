// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph → graph transforms: `grad`, `jvp`, `hvp`, `vmap`, higher-order grads.
//!
//! Each borrows the forward graph and returns a new one. A `wrt` the loss does
//! not depend on, or an op with no derivative rule, is a `panic!` inside rlx —
//! surfaced as an error here instead of aborting the host.

use rlx_ir::Graph;

use crate::ApiResult;
use crate::kinds::{Arg, nodes};
use crate::panics::catch;

/// A graph whose outputs are `[loss, …d(loss)/d(wrt)]`; it gains a
/// `d_output` input for the upstream gradient (seed it with `[1]`).
pub fn grad(forward: &Graph, wrt: Vec<u32>) -> ApiResult<Graph> {
    let ids = nodes(forward, wrt, Arg("grad", "wrt"))?;
    catch("grad", || rlx_opt::autodiff::grad_with_loss(forward, &ids))
}

/// `[…primals, …tangents]`, with one `tangent_<name>` input per seeded node.
pub fn jvp(forward: &Graph, tangent_for: Vec<u32>) -> ApiResult<Graph> {
    let ids = nodes(forward, tangent_for, Arg("jvp", "tangent_for"))?;
    catch("jvp", || rlx_opt::autodiff_fwd::jvp(forward, &ids))
}

/// Hessian-vector product, forward-over-reverse.
pub fn hvp(forward: &Graph, wrt: Vec<u32>) -> ApiResult<Graph> {
    let ids = nodes(forward, wrt, Arg("hvp", "wrt"))?;
    catch("hvp", || rlx_opt::autodiff_fwd::hvp(forward, &ids))
}

/// Batch the named inputs along a new leading axis of `batch_size`.
pub fn vmap(
    forward: &Graph,
    batched_input_names: Vec<String>,
    batch_size: usize,
) -> ApiResult<Graph> {
    let refs: Vec<&str> = batched_input_names.iter().map(String::as_str).collect();
    catch("vmap", || rlx_opt::vmap::vmap(forward, &refs, batch_size))
}

/// The `order`-th derivative of a scalar function w.r.t. scalar input `wrt_name`.
pub fn nth_order_grad(forward: &Graph, wrt_name: &str, order: usize) -> ApiResult<Graph> {
    catch("nth_order_grad", || {
        rlx_opt::rlx_autodiff::higher_order::nth_order_grad(forward, wrt_name, order)
    })
}

/// The order-`n` derivative contracted against one direction per level, for
/// non-scalar inputs. The result gains inputs `dir_0` … `dir_{order-1}`.
pub fn directional_nth_grad(forward: &Graph, wrt_name: &str, order: usize) -> ApiResult<Graph> {
    // The IR names the direction inputs itself (`dir_<level>`); only the
    // count of this slice matters.
    let dirs = vec!["v"; order];
    catch("directional_nth_grad", || {
        rlx_opt::rlx_autodiff::higher_order::directional_nth_grad(forward, wrt_name, &dirs)
    })
}
