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

//! Bridge between [`WeightStore`] and any [`rlx_optim::Optimizer`].
//!
//! The bundled [`crate::adam::AdamState`] is the host-side reference
//! Adam used by the parametric UMAP trainer. When you want a
//! different rule — AdamW, Lion, Muon, Sophia, … — you can drive any
//! [`rlx_optim::Optimizer`] over the same [`WeightStore`] / gradient
//! pair via [`step_weight_store`].
//!
//! Matrix-aware optimizers (Adafactor, SOAP, Muon, Kron-PSGD) need
//! the per-parameter shape. Pass it via the `shapes` map keyed by
//! the same name used in the `WeightStore`. Names with no entry in
//! `shapes` are treated as 1-D vectors of length `data.len()`.
//!
//! ```ignore
//! use rlx_optim::AdamW;
//! use rlx_umap::optim_adapter::step_weight_store;
//! use std::collections::HashMap;
//!
//! let mut opt = AdamW::new(3e-4).with_weight_decay(0.1);
//! let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
//! shapes.insert("fc.weight".to_string(), vec![128, 64]);
//! shapes.insert("fc.bias".to_string(),   vec![128]);
//!
//! step_weight_store(&mut opt, &mut weights, &grads, &shapes);
//! opt.end_iteration();
//! ```

use std::collections::HashMap;

use rlx_optim::Optimizer;

use crate::weights::WeightStore;

/// Drive `opt` for every name in `grads` that also appears in
/// `weights`. Missing grads (no entry in `grads.0` for a name held
/// by `weights`) are silently skipped — useful for partial training
/// passes (e.g. freezing the projection head).
///
/// Returns the number of parameters actually stepped.
pub fn step_weight_store<O: Optimizer>(
    opt: &mut O,
    weights: &mut WeightStore,
    grads: &WeightStore,
    shapes: &HashMap<String, Vec<usize>>,
) -> usize {
    let mut count = 0;
    for (name, w) in weights.0.iter_mut() {
        let Some(g) = grads.0.get(name) else { continue };
        if g.len() != w.len() {
            // Skip silently — caller probably has a key collision with
            // a separate non-trainable tensor.
            continue;
        }
        let default_shape = [w.len()];
        let shape: &[usize] = shapes
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&default_shape);
        opt.step(name, shape, w, g);
        count += 1;
    }
    count
}

/// Convenience: drive `opt` with all parameters treated as flat 1-D
/// vectors. Equivalent to `step_weight_store(opt, weights, grads, &HashMap::new())`.
/// Matrix-aware optimizers fall back to their elementwise path in this case.
pub fn step_weight_store_flat<O: Optimizer>(
    opt: &mut O,
    weights: &mut WeightStore,
    grads: &WeightStore,
) -> usize {
    step_weight_store(opt, weights, grads, &HashMap::new())
}
