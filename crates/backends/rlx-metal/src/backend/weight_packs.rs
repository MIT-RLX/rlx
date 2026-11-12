// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight packs: a fused weight stored once, as a param of its own.
//!
//! The QKV and gate/up fusions read one combined weight, `Concat` of the
//! separate weight params. Run as is, that concat is a second resident copy of
//! those weights in the activation arena, private to the executable and
//! rebuilt every run. When every weight in such a concat is read by it alone,
//! the concat is replaced here by a param of its own, the pack, and the
//! separate params leave the graph: [`set_param`](super::MetalExecutable::set_param)
//! of a part writes its slice of the pack. The pack is an ordinary weight —
//! weight buffer, sharing scope — so the fusion costs no memory.
//!
//! Weights read anywhere else as well (a backward graph's `dX = dY·Wᵀ`) keep
//! the concat. `RLX_METAL_NO_WEIGHT_PACKS=1` keeps every concat.

use rlx_ir::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

/// Where a part's values go in its pack, element-wise: row `r` of the part,
/// `inner` long, lands at `r * stride + offset` in the pack.
#[derive(Debug, Clone)]
pub(crate) struct PackPart {
    /// The pack's param name.
    pub pack: String,
    pub rows: usize,
    pub inner: usize,
    pub stride: usize,
    pub offset: usize,
}

impl PackPart {
    /// Elements in the part.
    pub fn len(&self) -> usize {
        self.rows * self.inner
    }
}

pub(crate) fn enabled() -> bool {
    !rlx_ir::env::flag("RLX_METAL_NO_WEIGHT_PACKS")
}

/// Turn each eligible weight concat of `graph` into a pack param, in place,
/// and return the parts by param name. The parts are left without readers,
/// for dead-code elimination.
pub(crate) fn pack_weight_concats(graph: &mut Graph) -> HashMap<String, PackPart> {
    let mut readers: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &i in &node.inputs {
            *readers.entry(i).or_default() += 1;
        }
    }
    for &o in &graph.outputs {
        *readers.entry(o).or_default() += 1;
    }
    let mut parts = HashMap::new();
    let ids: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();
    for id in ids {
        let node = graph.node(id);
        let Op::Concat { axis } = node.op else {
            continue;
        };
        if node.shape.dtype() != DType::F32 || graph.outputs.contains(&id) {
            continue;
        }
        let mut names = Vec::with_capacity(node.inputs.len());
        let mut inners = Vec::with_capacity(node.inputs.len());
        let mut rows = None;
        let ok = node.inputs.iter().all(|&i| {
            let p = graph.node(i);
            let Op::Param { name } = &p.op else {
                return false;
            };
            if !p.shape.is_static() {
                return false;
            }
            let dims: Vec<usize> = p.shape.dims().iter().map(|d| d.unwrap_static()).collect();
            if p.shape.dtype() != DType::F32
                || readers.get(&i) != Some(&1)
                || axis >= dims.len()
                || names.contains(name)
            {
                return false;
            }
            let r: usize = dims[..axis].iter().product();
            if *rows.get_or_insert(r) != r {
                return false;
            }
            names.push(name.clone());
            inners.push(dims[axis..].iter().product::<usize>());
            true
        });
        if !ok || names.len() < 2 {
            continue;
        }
        let rows = rows.unwrap_or(1);
        let stride: usize = inners.iter().sum();
        // The parts' names make the pack's: the same weights packed the same
        // way in another executable are the same pack, for sharing.
        let pack_name = format!("pack@{axis}:{}", names.join("|"));
        let mut offset = 0;
        for (name, inner) in names.into_iter().zip(inners) {
            parts.insert(
                name,
                PackPart {
                    pack: pack_name.clone(),
                    rows,
                    inner,
                    stride,
                    offset,
                },
            );
            offset += inner;
        }
        let n = graph.node_mut(id);
        n.op = Op::Param { name: pack_name };
        n.inputs.clear();
    }
    parts
}

/// Copy the part's values `src` (f32) into the pack at `dst` (f32), at most
/// `part.len()` of them.
///
/// # Safety
/// `dst` must be valid for writes of `part.rows * part.stride` f32s.
pub(crate) unsafe fn scatter(part: &PackPart, src: &[f32], dst: *mut f32) {
    let n = src.len().min(part.len());
    for r in 0..n.div_ceil(part.inner.max(1)) {
        let from = &src[r * part.inner..((r + 1) * part.inner).min(n)];
        unsafe {
            std::ptr::copy_nonoverlapping(
                from.as_ptr(),
                dst.add(r * part.stride + part.offset),
                from.len(),
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{GraphExt, Shape};

    fn qkv(extra_reader: bool) -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("packs");
        let x = g.input("x", Shape::new(&[4, 8], f));
        let wq = g.param("q", Shape::new(&[8, 6], f));
        let wk = g.param("k", Shape::new(&[8, 2], f));
        let w = g.add_node(Op::Concat { axis: 1 }, vec![wq, wk], Shape::new(&[8, 8], f));
        let y = g.matmul(x, w, Shape::new(&[4, 8], f));
        let mut outs = vec![y];
        if extra_reader {
            outs.push(g.transpose_(wq, vec![1, 0]));
        }
        g.set_outputs(outs);
        g
    }

    #[test]
    fn a_concat_of_lone_weights_becomes_a_pack() {
        let mut g = qkv(false);
        let parts = pack_weight_concats(&mut g);
        let (q, k) = (&parts["q"], &parts["k"]);
        assert_eq!((q.rows, q.inner, q.stride, q.offset), (8, 6, 8, 0));
        assert_eq!((k.rows, k.inner, k.stride, k.offset), (8, 2, 8, 6));
        assert_eq!(q.pack, "pack@1:q|k");
        assert!(g.param_id("pack@1:q|k").is_some());
    }

    #[test]
    fn a_weight_read_elsewhere_keeps_its_concat() {
        let mut g = qkv(true);
        assert!(pack_weight_concats(&mut g).is_empty());
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::Concat { .. })));
    }

    #[test]
    fn scatter_interleaves_rows() {
        let mut g = qkv(false);
        let parts = pack_weight_concats(&mut g);
        let mut pack = vec![0f32; 64];
        let q: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let k: Vec<f32> = (0..16).map(|i| 100.0 + i as f32).collect();
        unsafe {
            scatter(&parts["q"], &q, pack.as_mut_ptr());
            scatter(&parts["k"], &k, pack.as_mut_ptr());
        }
        for r in 0..8 {
            assert_eq!(&pack[r * 8..r * 8 + 6], &q[r * 6..r * 6 + 6]);
            assert_eq!(&pack[r * 8 + 6..r * 8 + 8], &k[r * 2..r * 2 + 2]);
        }
    }
}
