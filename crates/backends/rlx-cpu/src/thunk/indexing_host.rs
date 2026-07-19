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

//! Native ScatterNd / ScatterElements / GatherNd / GatherElements host path
//! for GPU backends.
//!
//! CPU already lowers these to [`Thunk::ScatterNd`] (etc.) and runs
//! [`crate::onnx_indexing`] in-place. GPU backends previously wrapped them in
//! [`super::HostOpDesc`], which rebuilds a mini-graph per call and stages every
//! input as `f32` — wrong for `I64` indices and expensive for F5-style graphs
//! with dozens of scatters per step.
//!
//! This module exposes the same native thunks with outer-arena byte offsets so
//! Metal can run them on unified memory and wgpu/CUDA can stage only the
//! touched span.

use super::Thunk;
use super::ops::{exec_gather_elements, exec_gather_nd, exec_scatter_elements, exec_scatter_nd};
use rlx_ir::{Graph, NodeId, Op};

fn shape_u32s(shape: &rlx_ir::Shape) -> Vec<u32> {
    (0..shape.rank())
        .map(|i| shape.dim(i).unwrap_static() as u32)
        .collect()
}

/// Build a native indexing [`Thunk`] from a graph node + outer-arena offsets.
pub fn indexing_thunk_from_node(
    graph: &Graph,
    node: &rlx_ir::Node,
    mut byte_off: impl FnMut(NodeId) -> usize,
) -> Thunk {
    match &node.op {
        Op::ScatterNd { reduction } => {
            let data_shape = &graph.node(node.inputs[0]).shape;
            let indices_shape = &graph.node(node.inputs[1]).shape;
            let updates_shape = &graph.node(node.inputs[2]).shape;
            Thunk::ScatterNd {
                data: byte_off(node.inputs[0]),
                indices: byte_off(node.inputs[1]),
                updates: byte_off(node.inputs[2]),
                dst: byte_off(node.id),
                data_shape: shape_u32s(data_shape),
                indices_shape: shape_u32s(indices_shape),
                data_len: data_shape.num_elements().unwrap_or(0) as u32,
                updates_len: updates_shape.num_elements().unwrap_or(0) as u32,
                indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
                indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
                reduction: *reduction,
            }
        }
        Op::ScatterElements { axis, reduction } => {
            let data_shape = &graph.node(node.inputs[0]).shape;
            let indices_shape = &graph.node(node.inputs[1]).shape;
            let updates_shape = &graph.node(node.inputs[2]).shape;
            Thunk::ScatterElements {
                data: byte_off(node.inputs[0]),
                indices: byte_off(node.inputs[1]),
                updates: byte_off(node.inputs[2]),
                dst: byte_off(node.id),
                data_shape: shape_u32s(data_shape),
                data_len: data_shape.num_elements().unwrap_or(0) as u32,
                updates_len: updates_shape.num_elements().unwrap_or(0) as u32,
                indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
                indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
                axis: *axis,
                reduction: *reduction,
            }
        }
        Op::GatherNd { batch_dims } => {
            let data_shape = &graph.node(node.inputs[0]).shape;
            let indices_shape = &graph.node(node.inputs[1]).shape;
            Thunk::GatherNd {
                data: byte_off(node.inputs[0]),
                indices: byte_off(node.inputs[1]),
                dst: byte_off(node.id),
                data_shape: shape_u32s(data_shape),
                indices_shape: shape_u32s(indices_shape),
                data_len: data_shape.num_elements().unwrap_or(0) as u32,
                indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
                out_len: node.shape.num_elements().unwrap_or(0) as u32,
                indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
                batch_dims: *batch_dims,
            }
        }
        Op::GatherElements { axis } => {
            let data_shape = &graph.node(node.inputs[0]).shape;
            let indices_shape = &graph.node(node.inputs[1]).shape;
            Thunk::GatherElements {
                data: byte_off(node.inputs[0]),
                indices: byte_off(node.inputs[1]),
                dst: byte_off(node.id),
                data_shape: shape_u32s(data_shape),
                indices_shape: shape_u32s(indices_shape),
                data_len: data_shape.num_elements().unwrap_or(0) as u32,
                indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
                out_len: node.shape.num_elements().unwrap_or(0) as u32,
                indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
                axis: *axis,
            }
        }
        _ => panic!(
            "indexing_thunk_from_node: expected ScatterNd/ScatterElements/GatherNd/GatherElements"
        ),
    }
}

fn idx_nbytes(len: u32, indices_i64: u8) -> usize {
    if indices_i64 != 0 {
        len as usize * 8
    } else {
        len as usize * 4
    }
}

/// `(byte_offset, nbytes)` regions touched by an indexing thunk.
pub fn indexing_thunk_regions(thunk: &Thunk) -> Vec<(usize, usize)> {
    match thunk {
        Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            ..
        } => vec![
            (*data, *data_len as usize * 4),
            (*indices, idx_nbytes(*indices_len, *indices_i64)),
            (*updates, *updates_len as usize * 4),
            (*dst, *data_len as usize * 4),
        ],
        Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            ..
        } => vec![
            (*data, *data_len as usize * 4),
            (*indices, idx_nbytes(*indices_len, *indices_i64)),
            (*updates, *updates_len as usize * 4),
            (*dst, *data_len as usize * 4),
        ],
        Thunk::GatherNd {
            data,
            indices,
            dst,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            ..
        } => vec![
            (*data, *data_len as usize * 4),
            (*indices, idx_nbytes(*indices_len, *indices_i64)),
            (*dst, *out_len as usize * 4),
        ],
        Thunk::GatherElements {
            data,
            indices,
            dst,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            ..
        } => vec![
            (*data, *data_len as usize * 4),
            (*indices, idx_nbytes(*indices_len, *indices_i64)),
            (*dst, *out_len as usize * 4),
        ],
        _ => panic!("indexing_thunk_regions: not an indexing thunk"),
    }
}

fn rebase_off(off: usize, lo: usize) -> usize {
    off.checked_sub(lo)
        .expect("indexing host span: offset below span lo")
}

fn rebase_indexing_thunk(thunk: Thunk, lo: usize) -> Thunk {
    match thunk {
        Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            reduction,
        } => Thunk::ScatterNd {
            data: rebase_off(data, lo),
            indices: rebase_off(indices, lo),
            updates: rebase_off(updates, lo),
            dst: rebase_off(dst, lo),
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            reduction,
        },
        Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            axis,
            reduction,
        } => Thunk::ScatterElements {
            data: rebase_off(data, lo),
            indices: rebase_off(indices, lo),
            updates: rebase_off(updates, lo),
            dst: rebase_off(dst, lo),
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            axis,
            reduction,
        },
        Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            batch_dims,
        } => Thunk::GatherNd {
            data: rebase_off(data, lo),
            indices: rebase_off(indices, lo),
            dst: rebase_off(dst, lo),
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            batch_dims,
        },
        Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            axis,
        } => Thunk::GatherElements {
            data: rebase_off(data, lo),
            indices: rebase_off(indices, lo),
            dst: rebase_off(dst, lo),
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            axis,
        },
        _ => panic!("rebase_indexing_thunk: not an indexing thunk"),
    }
}

/// Contiguous host span covering every region an indexing thunk touches, with
/// offsets rebased to the span start (wgpu/CUDA readback mirror of
/// [`super::HostOpSpan`]).
#[derive(Clone)]
pub struct IndexingHostSpan {
    pub lo: usize,
    pub hi: usize,
    pub thunk: Thunk,
}

impl IndexingHostSpan {
    pub fn from_thunk(thunk: Thunk) -> Self {
        let regions = indexing_thunk_regions(&thunk);
        let mut lo = regions[0].0;
        let mut hi = regions[0].0 + regions[0].1;
        for &(off, n) in &regions[1..] {
            lo = lo.min(off);
            hi = hi.max(off + n);
        }
        Self {
            lo,
            hi,
            thunk: rebase_indexing_thunk(thunk, lo),
        }
    }

    pub fn len(&self) -> usize {
        self.hi - self.lo
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Destination `(byte_off, nbytes)` written by an indexing thunk.
pub fn indexing_thunk_dst_region(thunk: &Thunk) -> (usize, usize) {
    match thunk {
        Thunk::ScatterNd { dst, data_len, .. } | Thunk::ScatterElements { dst, data_len, .. } => {
            (*dst, *data_len as usize * 4)
        }
        Thunk::GatherNd { dst, out_len, .. } | Thunk::GatherElements { dst, out_len, .. } => {
            (*dst, *out_len as usize * 4)
        }
        _ => panic!("indexing_thunk_dst_region: not an indexing thunk"),
    }
}

/// Remap outer-arena byte offsets in an indexing thunk (packed host staging).
pub fn remap_indexing_thunk_offsets(thunk: Thunk, mut map: impl FnMut(usize) -> usize) -> Thunk {
    match thunk {
        Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            reduction,
        } => Thunk::ScatterNd {
            data: map(data),
            indices: map(indices),
            updates: map(updates),
            dst: map(dst),
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            reduction,
        },
        Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            axis,
            reduction,
        } => Thunk::ScatterElements {
            data: map(data),
            indices: map(indices),
            updates: map(updates),
            dst: map(dst),
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64,
            axis,
            reduction,
        },
        Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            batch_dims,
        } => Thunk::GatherNd {
            data: map(data),
            indices: map(indices),
            dst: map(dst),
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            batch_dims,
        },
        Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            axis,
        } => Thunk::GatherElements {
            data: map(data),
            indices: map(indices),
            dst: map(dst),
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64,
            axis,
        },
        _ => panic!("remap_indexing_thunk_offsets: not an indexing thunk"),
    }
}

/// Max contiguous span (bytes) before discrete GPUs switch to packed region staging.
///
/// F5 DiT ScatterNd often places `data` near arena start and `dst` near the end
/// — a contiguous mirror would be multi-GiB on CUDA/wgpu; packing keeps PCIe
/// traffic proportional to the touched tensors only.
pub const INDEXING_CONTIGUOUS_SPAN_CAP: usize = 512 * 1024 * 1024;

/// Run a native indexing thunk against a raw byte arena.
///
/// # Safety
/// `base` must span every outer offset referenced by `thunk` (inputs + output),
/// including full `I64` index buffers when `indices_i64 != 0`.
pub unsafe fn execute_indexing_thunk_on_bytes(base: *mut u8, thunk: &Thunk) {
    match thunk {
        Thunk::ScatterNd { .. } => exec_scatter_nd(thunk, base),
        Thunk::ScatterElements { .. } => exec_scatter_elements(thunk, base),
        Thunk::GatherNd { .. } => exec_gather_nd(thunk, base),
        Thunk::GatherElements { .. } => exec_gather_elements(thunk, base),
        _ => panic!("execute_indexing_thunk_on_bytes: not an indexing thunk"),
    }
}

/// Opaque wrapper so GPU backends can store a native indexing [`Thunk`] in
/// `Debug`-derived enums without requiring `Thunk: Debug`.
#[derive(Clone)]
pub struct IndexingThunk(pub Thunk);

impl std::fmt::Debug for IndexingThunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IndexingThunk(..)")
    }
}

impl IndexingThunk {
    pub fn from_node(
        graph: &Graph,
        node: &rlx_ir::Node,
        byte_off: impl FnMut(NodeId) -> usize,
    ) -> Self {
        Self(indexing_thunk_from_node(graph, node, byte_off))
    }

    pub fn inner(&self) -> &Thunk {
        &self.0
    }

    /// For f32-uniform GPU arenas (CUDA / wgpu / ROCm): I64 index tensors are
    /// stored as f32 slots. Force the f32→i64 reader so ScatterNd does not
    /// reinterpret float bits as i64 (that bug drifted F5 DiT vs CPU/Metal).
    pub fn force_indices_f32(mut self) -> Self {
        self.0 = force_indexing_indices_f32(self.0);
        self
    }
}

/// See [`IndexingThunk::force_indices_f32`].
pub fn force_indexing_indices_f32(thunk: Thunk) -> Thunk {
    match thunk {
        Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            reduction,
            ..
        } => Thunk::ScatterNd {
            data,
            indices,
            updates,
            dst,
            data_shape,
            indices_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64: 0,
            reduction,
        },
        Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            axis,
            reduction,
            ..
        } => Thunk::ScatterElements {
            data,
            indices,
            updates,
            dst,
            data_shape,
            data_len,
            updates_len,
            indices_len,
            indices_i64: 0,
            axis,
            reduction,
        },
        Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            batch_dims,
            ..
        } => Thunk::GatherNd {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64: 0,
            batch_dims,
        },
        Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            axis,
            ..
        } => Thunk::GatherElements {
            data,
            indices,
            dst,
            data_shape,
            indices_shape,
            data_len,
            indices_len,
            out_len,
            indices_i64: 0,
            axis,
        },
        other => other,
    }
}

/// Build an indexing thunk from `(graph, node, |nid| byte_offset)`.
#[macro_export]
macro_rules! rlx_indexing_thunk {
    ($graph:expr, $node:expr, $byte_off:expr) => {
        $crate::thunk::IndexingThunk::from_node(&$graph, $node, $byte_off)
    };
}

/// Execute a native indexing thunk against a raw byte arena.
#[macro_export]
macro_rules! rlx_execute_indexing_on_bytes {
    ($base:expr, $thunk:expr) => {
        $crate::thunk::execute_indexing_thunk_on_bytes($base, $thunk)
    };
}

/// D2H → native indexing thunk → H2D staging for discrete-GPU f32 arenas.
///
/// Runs against the full arena with original byte offsets (I64 index buffers
/// stay intact in the byte view of the f32 host mirror).
#[macro_export]
macro_rules! rlx_indexing_stage_d2h {
    (
        arena_size_bytes = $nbytes:expr,
        thunk = $thunk:expr,
        sync = $sync:block,
        dtoh = |$host:ident| $dtoh:block,
        htod = |$host_back:ident| $htod:block $(,)?
    ) => {
        $crate::rlx_arena_stage_d2h! {
            arena_size_bytes = $nbytes,
            sync = $sync,
            dtoh = |$host| $dtoh,
            on_host = |host_mut| {
                unsafe {
                    $crate::thunk::execute_indexing_thunk_on_bytes(
                        host_mut.as_mut_ptr() as *mut u8,
                        $thunk,
                    );
                }
            },
            htod = |$host_back| $htod,
        }
    };
}
