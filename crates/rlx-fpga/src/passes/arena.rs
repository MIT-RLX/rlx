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

//! Liveness-based arena allocation for activation BRAMs.
//!
//! Sequential pipeline (which is what we have today — every layer runs
//! to completion before the next starts) makes the live-range graph an
//! interval graph that's trivially 2-colorable: at any point only the
//! producer's output and the consumer's input are live. Two ping-pong
//! BRAM slots, sized to the largest activation, suffice.
//!
//! The pass writes:
//! * `Hints.bram_slot_in[i]` — which slot layer `i` reads from
//! * `Hints.bram_slot_out[i]` — which slot layer `i` writes to
//!
//! Slot 0 holds the model input; subsequent layers ping-pong between
//! slots 0 and 1. Elided layers (`hints.elided` set, e.g. by
//! `fuse_conv_relu`) are skipped — the next non-elided layer reads from
//! the previous non-elided layer's output slot. That's the BRAM-saving
//! half of fusion.

use std::collections::BTreeMap;

use crate::model::Model;
use crate::passes::Hints;

/// Run arena allocation. Returns a `slot → bank_factor` map: slots
/// whose consumer needs ic-parallel reads get `bank_factor = P_ic`
/// (typically 4); all other slots get `1` (omitted from the map).
pub fn run(model: &Model, hints: &mut [Hints]) -> BTreeMap<u8, u8> {
    debug_assert_eq!(model.layers.len(), hints.len());
    let mut prev_slot: u8 = 0;
    let mut next_slot: u8 = 1;

    for i in 0..hints.len() {
        if hints[i].elided {
            hints[i].bram_slot_in = Some(prev_slot);
            hints[i].bram_slot_out = Some(prev_slot);
            continue;
        }
        hints[i].bram_slot_in = Some(prev_slot);
        hints[i].bram_slot_out = Some(next_slot);
        std::mem::swap(&mut prev_slot, &mut next_slot);
    }

    // Bank factor per slot: max(consumer.ic_parallelism) over all
    // layers reading the slot. Only entries > 1 are recorded.
    let mut bank: BTreeMap<u8, u8> = BTreeMap::new();
    for h in hints.iter() {
        if h.elided || h.ic_parallelism <= 1 {
            continue;
        }
        if let Some(slot) = h.bram_slot_in {
            let entry = bank.entry(slot).or_insert(1);
            *entry = (*entry).max(h.ic_parallelism as u8);
        }
    }
    let _ = model;
    bank
}

/// How many distinct slots the pass uses. Always 2 for sequential
/// pipelines — exposed for tests / estimator.
pub fn slot_count(hints: &[Hints]) -> usize {
    let mut max_slot: u8 = 0;
    for h in hints {
        if let Some(s) = h.bram_slot_in {
            max_slot = max_slot.max(s);
        }
        if let Some(s) = h.bram_slot_out {
            max_slot = max_slot.max(s);
        }
    }
    max_slot as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;
    use crate::passes::{Hints, fuse_conv_relu};

    #[test]
    fn ping_pong_uses_two_slots() {
        let m = tinyconv_mnist_from_cortexm();
        let mut hints = vec![Hints::default(); m.layers.len()];
        let bank = run(&m, &mut hints);
        assert_eq!(
            slot_count(&hints),
            2,
            "sequential pipeline should fit in 2 ping-pong slots"
        );
        assert!(
            bank.is_empty(),
            "no banked slots when no layer has ic-parallel"
        );
    }

    #[test]
    fn fused_layers_dont_burn_a_slot() {
        let m = tinyconv_mnist_from_cortexm();
        let mut hints = vec![Hints::default(); m.layers.len()];
        fuse_conv_relu::run(&m, &mut hints);
        run(&m, &mut hints);

        for i in 0..hints.len() {
            if hints[i].elided {
                assert_eq!(
                    hints[i].bram_slot_in, hints[i].bram_slot_out,
                    "elided layer {i} should have in==out slot"
                );
            }
        }
        assert_eq!(slot_count(&hints), 2);
    }

    #[test]
    fn ic_parallel_consumer_marks_input_slot_for_banking() {
        let m = tinyconv_mnist_from_cortexm();
        let mut hints = vec![Hints::default(); m.layers.len()];
        // Pretend conv2 (idx 3) is ic-parallel.
        hints[3].ic_parallelism = 4;
        let bank = run(&m, &mut hints);
        // Conv2's input slot should be banked.
        let conv2_in = hints[3].bram_slot_in.unwrap();
        assert_eq!(bank.get(&conv2_in), Some(&4));
    }
}
