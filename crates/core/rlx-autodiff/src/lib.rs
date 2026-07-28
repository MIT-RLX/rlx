// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JAX-shaped program transforms on RLX MIR: autodiff, JVP/HVP, and vmap.
//!
//! Run [`prepare_graph_for_ad`] (or [`PrepareForAutodiff`]) before the
//! gradient walk when the graph contains fused ops from HIR `Direct`
//! lowering or inference fusion passes.

pub mod activation_deriv;
pub mod autodiff;
pub mod autodiff_fwd;
pub mod compose;
pub mod decompose_backward;
pub mod decompose_backward_kernels;
pub mod fuse_splat;
pub mod higher_order;
pub mod legalize_reduce;
pub mod mlip;
pub mod prepare_ad;
pub mod vmap;

pub use autodiff::{
    GradWithLossOptions, grad, grad_with_loss, grad_with_loss_opts, quantized_weight_bits,
};
pub use autodiff_fwd::{hvp, jvp};
pub use compose::{broadcast_scalar, cse};
pub use decompose_backward::{
    decompose_backward_for_ad, decompose_backward_ops, decompose_backward_ops_except,
    prepare_grad_graph_for_jvp,
};
pub use higher_order::{
    HigherOrderOptions, directional_nth_grad, fuse_elementwise, nth_order_grad,
    nth_order_grad_with_options,
};
pub use mlip::{
    ForceEnergyLossWeights, build_force_energy_loss, grad_subgraph, grad_subgraph_for_jvp,
};
pub use prepare_ad::{
    AutodiffError, MirAutodiffExt, PrepareForAutodiff, grad_with_loss_module, hvp_module,
    jvp_module, nth_order_grad_module, prepare_graph_for_ad, prepare_mir_for_ad,
    prepare_module_for_ad,
};
pub use vmap::vmap;
