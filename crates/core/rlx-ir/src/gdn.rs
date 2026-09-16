// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed-gradient layout for [`crate::Op::GatedDeltaNetBackward`].
//!
//! A graph node has one output, but the backward of a gated delta-net produces
//! five or six gradients of three different shapes. They are returned packed
//! into a single 1-D tensor, and both the VJP rule (which slices them back out)
//! and the backend kernel (which writes them) index it through this type — one
//! source of truth, so the two cannot drift apart.

/// Where each gradient sits in the packed backward output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnBackwardLayout {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub state_size: usize,
    pub gate_per_channel: bool,
    pub carry_state: bool,
}

impl GdnBackwardLayout {
    pub fn new(
        batch: usize,
        seq: usize,
        heads: usize,
        state_size: usize,
        gate_per_channel: bool,
        carry_state: bool,
    ) -> Self {
        Self {
            batch,
            seq,
            heads,
            state_size,
            gate_per_channel,
            carry_state,
        }
    }

    /// Elements in one of `q` / `k` / `v` / `dy`: `[B, S, H, N]`.
    pub fn qkv_elems(&self) -> usize {
        self.batch * self.seq * self.heads * self.state_size
    }

    /// Elements in `beta`: `[B, S, H]`.
    pub fn beta_elems(&self) -> usize {
        self.batch * self.seq * self.heads
    }

    /// Elements in `g`: `[B, S, H, N]` per-channel, else `[B, S, H]`.
    pub fn gate_elems(&self) -> usize {
        if self.gate_per_channel {
            self.qkv_elems()
        } else {
            self.beta_elems()
        }
    }

    /// Elements in the carried state: `[B, H, N, N]`. Zero when not carried.
    pub fn state_elems(&self) -> usize {
        if self.carry_state {
            self.batch * self.heads * self.state_size * self.state_size
        } else {
            0
        }
    }

    pub fn dq_offset(&self) -> usize {
        0
    }
    pub fn dk_offset(&self) -> usize {
        self.qkv_elems()
    }
    pub fn dv_offset(&self) -> usize {
        2 * self.qkv_elems()
    }
    pub fn dg_offset(&self) -> usize {
        3 * self.qkv_elems()
    }
    pub fn dbeta_offset(&self) -> usize {
        self.dg_offset() + self.gate_elems()
    }
    /// Only meaningful when `carry_state`.
    pub fn dstate_offset(&self) -> usize {
        self.dbeta_offset() + self.beta_elems()
    }

    /// Total packed length.
    pub fn total_elems(&self) -> usize {
        self.dstate_offset() + self.state_elems()
    }

    /// `(offset, len)` for each gradient, in input order: q, k, v, g, beta,
    /// and state when carried. The VJP walks this to emit its slices.
    pub fn slices(&self) -> Vec<(usize, usize)> {
        let mut out = vec![
            (self.dq_offset(), self.qkv_elems()),
            (self.dk_offset(), self.qkv_elems()),
            (self.dv_offset(), self.qkv_elems()),
            (self.dg_offset(), self.gate_elems()),
            (self.dbeta_offset(), self.beta_elems()),
        ];
        if self.carry_state {
            out.push((self.dstate_offset(), self.state_elems()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_head_layout_is_contiguous_and_complete() {
        let l = GdnBackwardLayout::new(2, 3, 4, 8, false, false);
        let qkv = 2 * 3 * 4 * 8;
        let bsh = 2 * 3 * 4;
        assert_eq!(l.qkv_elems(), qkv);
        assert_eq!(l.gate_elems(), bsh);
        assert_eq!(l.state_elems(), 0);
        assert_eq!(l.total_elems(), 3 * qkv + bsh + bsh);

        // Slices must tile the packed buffer exactly, with no gap or overlap.
        let mut cursor = 0;
        for (off, len) in l.slices() {
            assert_eq!(off, cursor, "gap or overlap in packed layout");
            cursor += len;
        }
        assert_eq!(cursor, l.total_elems());
    }

    #[test]
    fn per_channel_gate_widens_only_the_gate_slice() {
        let l = GdnBackwardLayout::new(2, 3, 4, 8, true, false);
        assert_eq!(l.gate_elems(), l.qkv_elems());
        let mut cursor = 0;
        for (off, len) in l.slices() {
            assert_eq!(off, cursor);
            cursor += len;
        }
        assert_eq!(cursor, l.total_elems());
    }

    #[test]
    fn carried_state_appends_a_slice() {
        let l = GdnBackwardLayout::new(2, 3, 4, 8, false, true);
        assert_eq!(l.state_elems(), 2 * 4 * 8 * 8);
        assert_eq!(l.slices().len(), 6);
        let mut cursor = 0;
        for (off, len) in l.slices() {
            assert_eq!(off, cursor);
            cursor += len;
        }
        assert_eq!(cursor, l.total_elems());
    }
}
