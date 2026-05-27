// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! JAX-shaped program transforms on RLX MIR: autodiff, JVP/HVP, and vmap.
//!
//! Run [`prepare_graph_for_ad`] (or [`PrepareForAutodiff`]) before the
//! gradient walk when the graph contains fused ops from HIR `Direct`
//! lowering or inference fusion passes.

pub mod autodiff;
pub mod autodiff_fwd;
pub mod fuse_splat;
pub mod legalize_reduce;
pub mod prepare_ad;
pub mod vmap;

pub use autodiff::{grad, grad_with_loss, quantized_weight_bits};
pub use autodiff_fwd::{hvp, jvp};
pub use prepare_ad::{
    AutodiffError, MirAutodiffExt, PrepareForAutodiff, grad_with_loss_module, jvp_module,
    prepare_graph_for_ad, prepare_mir_for_ad, prepare_module_for_ad,
};
pub use vmap::vmap;
