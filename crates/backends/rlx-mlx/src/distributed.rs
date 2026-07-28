// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`MlxTransport`] — a [`rlx_driver::Transport`] backed by MLX's
//! distributed module.
//!
//! `MlxTransport::init` calls `mlx::core::distributed::init`, which
//! selects the best available backend:
//!   - **jaccl** — RDMA over Thunderbolt (macOS SDK ≥ 26.2), the fast
//!     path for a few Apple-Silicon machines on a Thunderbolt mesh;
//!   - **ring** — TCP, the portable fallback;
//!   - **mpi** — when launched under MPI.
//!
//! Because all three sit behind one MLX API, this transport gets
//! Thunderbolt RDMA "for free" where the hardware/SDK allow it, and
//! degrades to TCP otherwise — without any change here.
//!
//! Two surfaces are exposed:
//!   - **Native collectives** ([`MlxTransport::all_sum`],
//!     [`MlxTransport::all_gather`]) — delegate straight to MLX; use
//!     these for tensor-parallel reductions, they are far faster than
//!     the gather-to-root fallback in [`rlx_driver::ProcessGroup`].
//!   - The [`Transport`] trait (two-sided `send_bytes`/`recv_bytes` +
//!     `barrier`) so the same `ProcessGroup` and pipeline-parallel relay
//!     that run over [`rlx_driver::TcpTransport`] run unchanged over MLX.
//!
//! Run programs that use this with MLX's launcher, e.g.
//! `mlx.launch --backend jaccl --hostfile hosts.json <prog>`.

use crate::ffi;
use rlx_driver::{CollectiveError, Transport};
use std::ffi::{CStr, CString};
use std::os::raw::c_int;

fn last_error() -> String {
    unsafe {
        let p = ffi::rlx_mlx_last_error();
        if p.is_null() {
            "(null)".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn terr(ctx: &str) -> CollectiveError {
    CollectiveError::TransportError {
        reason: format!("mlx {ctx}: {}", last_error()),
    }
}

/// A distributed transport delegating to MLX's `jaccl`/`ring`/`mpi`
/// backend. Construct with [`MlxTransport::init`] inside a program
/// started by MLX's launcher.
pub struct MlxTransport {
    rank: u32,
    world: u32,
}

impl MlxTransport {
    /// `true` if MLX was built/launched with a working distributed
    /// backend.
    pub fn is_available() -> bool {
        let _guard = crate::sync::runtime_guard();
        let mut out: c_int = 0;
        let rc = unsafe { ffi::rlx_mlx_dist_is_available(&mut out) };
        rc == ffi::RLX_MLX_OK && out != 0
    }

    /// Initialize the group. `backend` is `"any"` (let MLX choose),
    /// `"jaccl"`, `"ring"`, or `"mpi"`. With `strict = false` and no
    /// backend available, MLX returns a size-1 singleton group and the
    /// collectives become no-ops (handy for running single-node).
    pub fn init(strict: bool, backend: &str) -> Result<Self, CollectiveError> {
        let _guard = crate::sync::runtime_guard();
        let cb = CString::new(backend).map_err(|e| CollectiveError::TransportError {
            reason: e.to_string(),
        })?;
        let mut rank: c_int = 0;
        let mut size: c_int = 0;
        let rc =
            unsafe { ffi::rlx_mlx_dist_init(strict as c_int, cb.as_ptr(), &mut rank, &mut size) };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("init"));
        }
        Ok(Self {
            rank: rank as u32,
            world: size as u32,
        })
    }

    /// In-place native AllReduce(sum). Prefer this over
    /// `ProcessGroup::all_reduce` when the transport is MLX.
    pub fn all_sum(&self, data: &mut [f32]) -> Result<(), CollectiveError> {
        if self.world <= 1 {
            return Ok(());
        }
        let _guard = crate::sync::runtime_guard();
        let mut out = vec![0f32; data.len()];
        let rc =
            unsafe { ffi::rlx_mlx_dist_all_sum_f32(data.as_ptr(), out.as_mut_ptr(), data.len()) };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("all_sum"));
        }
        data.copy_from_slice(&out);
        Ok(())
    }

    /// Native AllGather: returns the concatenation of every rank's
    /// `local`, in rank order (`world_size * local.len()` elements).
    pub fn all_gather(&self, local: &[f32]) -> Result<Vec<f32>, CollectiveError> {
        if self.world <= 1 {
            return Ok(local.to_vec());
        }
        let _guard = crate::sync::runtime_guard();
        let total = local.len() * self.world as usize;
        let mut out = vec![0f32; total];
        let rc = unsafe {
            ffi::rlx_mlx_dist_all_gather_f32(
                local.as_ptr(),
                local.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("all_gather"));
        }
        Ok(out)
    }

    fn raw_send(&self, dst: u32, data: &[f32]) -> Result<(), CollectiveError> {
        let _guard = crate::sync::runtime_guard();
        let rc = unsafe { ffi::rlx_mlx_dist_send_f32(data.as_ptr(), data.len(), dst as c_int) };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("send"));
        }
        Ok(())
    }

    fn raw_recv(&self, src: u32, out: &mut [f32]) -> Result<(), CollectiveError> {
        let _guard = crate::sync::runtime_guard();
        let rc = unsafe { ffi::rlx_mlx_dist_recv_f32(out.as_mut_ptr(), out.len(), src as c_int) };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("recv"));
        }
        Ok(())
    }
}

impl Transport for MlxTransport {
    fn rank(&self) -> u32 {
        self.rank
    }
    fn world_size(&self) -> u32 {
        self.world
    }

    /// Two-sided byte send over MLX's fixed-shape `send`/`recv`.
    ///
    /// MLX's point-to-point needs the receiver to know the element count
    /// up front, so we prefix every message with a 2-element f32 header
    /// carrying the exact byte length (`len/4096`, `len%4096` — both
    /// integers exactly representable in f32), then ship the payload as
    /// `ceil(len/4)` f32 words. The payload words are raw bytes
    /// reinterpreted as f32 (never arithmetic), and jaccl/ring move them
    /// as an opaque byte buffer, so the bit pattern round-trips exactly.
    /// `tag` is ignored: MLX point-to-point is FIFO per peer pair and our
    /// use is sequential per pair.
    fn send_bytes(&self, to: u32, _tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
        let len = bytes.len();
        let hdr = [(len / 4096) as f32, (len % 4096) as f32];
        self.raw_send(to, &hdr)?;
        let nf = len.div_ceil(4);
        if nf > 0 {
            let mut buf = vec![0f32; nf];
            // SAFETY: buf has nf*4 >= len bytes; we copy exactly `len`.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr() as *mut u8, len);
            }
            self.raw_send(to, &buf)?;
        }
        Ok(())
    }

    fn recv_bytes(&self, from: u32, _tag: u32) -> Result<Vec<u8>, CollectiveError> {
        let mut hdr = [0f32; 2];
        self.raw_recv(from, &mut hdr)?;
        let len = (hdr[0] as usize) * 4096 + (hdr[1] as usize);
        let nf = len.div_ceil(4);
        let mut out = vec![0u8; len];
        if nf > 0 {
            let mut buf = vec![0f32; nf];
            self.raw_recv(from, &mut buf)?;
            // SAFETY: buf has nf*4 >= len bytes; we copy exactly `len`.
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr() as *const u8, out.as_mut_ptr(), len);
            }
        }
        Ok(out)
    }

    fn barrier(&self) -> Result<(), CollectiveError> {
        if self.world <= 1 {
            return Ok(());
        }
        let _guard = crate::sync::runtime_guard();
        let rc = unsafe { ffi::rlx_mlx_dist_barrier() };
        if rc != ffi::RLX_MLX_OK {
            return Err(terr("barrier"));
        }
        Ok(())
    }
}

// ── Device-resident in-graph all-reduce (GPU-resident tensor parallel) ──
//
// Registers `collective.all_reduce` as an MLX custom op whose kernel composes
// `mc::distributed::all_sum` on the *lazy device array* (via the
// `rlx_mlx_dist_all_sum_array` shim) — no host copy, no eval. So a
// tensor-parallel layer's all-reduce stays on the GPU and rides MLX's
// jaccl/ring transport, unlike the CPU/Metal host-sync kernels. Pair with
// [`MlxTransport::init`] (which sets up the process group). The CPU
// equivalent (and the IR shape extension) live in `rlx-collectives`; this
// re-registers the same op name with a device kernel.

use crate::array::{Array, check};
use crate::op_registry::{MlxKernel, register_mlx_kernel};
use rlx_ir::Shape as IrShape;
use rlx_ir::op_registry::{OpExtension, register_op};
use std::sync::Arc;

/// Op name, shared with the CPU collective in `rlx-collectives`.
pub const ALL_REDUCE: &str = "collective.all_reduce";
/// All-gather op name, shared with `rlx-collectives`.
pub const ALL_GATHER: &str = "collective.all_gather";
/// Reduce-scatter op name, shared with `rlx-collectives`.
pub const REDUCE_SCATTER: &str = "collective.reduce_scatter";
/// Megatron `f` op name (identity forward), shared with `rlx-collectives`.
pub const COPY_TO_PARALLEL: &str = "collective.copy_to_parallel";
/// Megatron `g` op name (all-reduce forward), shared with `rlx-collectives`.
pub const REDUCE_FROM_PARALLEL: &str = "collective.reduce_from_parallel";

/// `world_size` from a gather/scatter `attrs` blob
/// (`[group_id: u64][world_size: u32]`, as encoded by `rlx-collectives`).
/// Shape inference needs it because these ops scale the axis-0 extent.
fn attrs_world_size(attrs: &[u8]) -> usize {
    u32::from_le_bytes(
        attrs
            .get(8..12)
            .and_then(|b| b.try_into().ok())
            .expect("collective attrs: 8-byte group id + 4-byte world size"),
    ) as usize
}

struct AllReduceExt;
impl OpExtension for AllReduceExt {
    fn name(&self) -> &str {
        ALL_REDUCE
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&IrShape], _attrs: &[u8]) -> IrShape {
        inputs[0].clone()
    }
}

struct AllReduceMlx;
impl MlxKernel for AllReduceMlx {
    fn name(&self) -> &str {
        ALL_REDUCE
    }
    fn execute(
        &self,
        inputs: &[&Array],
        _output_shape: &IrShape,
        attrs: &[u8],
    ) -> Result<Array, crate::array::MlxError> {
        // Device-resident all-reduce on the lazy array — no host round-trip.
        // The reduce kind rides in attrs[8] (0=Sum default), matching
        // rlx-collectives' encode_kind.
        let kind = attrs.get(8).copied().unwrap_or(0) as std::os::raw::c_int;
        let _guard = crate::sync::runtime_guard();
        let mut out: *mut ffi::mlx_array_t = std::ptr::null_mut();
        let rc = unsafe { ffi::rlx_mlx_dist_all_reduce_array(inputs[0].ptr, kind, &mut out) };
        check(rc)?;
        Ok(Array::from_raw(out))
    }
}

// ── Device-resident all-gather / reduce-scatter ──
//
// Same shape rules as the CPU ops in `rlx-collectives` (axis-0 extent scales
// by `world_size`, read from `attrs`), but the kernels run device-resident on
// the lazy MLX array via the `*_array` shims — no host copy. `reduce_scatter`
// composes all_sum + slice inside MLX (see the C++ shim).

struct AllGatherExt;
impl OpExtension for AllGatherExt {
    fn name(&self) -> &str {
        ALL_GATHER
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&IrShape], attrs: &[u8]) -> IrShape {
        let ws = attrs_world_size(attrs);
        let inp = inputs[0];
        let mut dims: Vec<usize> = inp.dims().iter().map(|d| d.unwrap_static()).collect();
        dims[0] *= ws;
        IrShape::new(&dims, inp.dtype())
    }
}

struct AllGatherMlx;
impl MlxKernel for AllGatherMlx {
    fn name(&self) -> &str {
        ALL_GATHER
    }
    fn execute(
        &self,
        inputs: &[&Array],
        _output_shape: &IrShape,
        _attrs: &[u8],
    ) -> Result<Array, crate::array::MlxError> {
        let _guard = crate::sync::runtime_guard();
        let mut out: *mut ffi::mlx_array_t = std::ptr::null_mut();
        let rc = unsafe { ffi::rlx_mlx_dist_all_gather_array(inputs[0].ptr, &mut out) };
        check(rc)?;
        Ok(Array::from_raw(out))
    }
}

struct ReduceScatterExt;
impl OpExtension for ReduceScatterExt {
    fn name(&self) -> &str {
        REDUCE_SCATTER
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&IrShape], attrs: &[u8]) -> IrShape {
        let ws = attrs_world_size(attrs);
        let inp = inputs[0];
        let mut dims: Vec<usize> = inp.dims().iter().map(|d| d.unwrap_static()).collect();
        dims[0] /= ws;
        IrShape::new(&dims, inp.dtype())
    }
}

struct ReduceScatterMlx;
impl MlxKernel for ReduceScatterMlx {
    fn name(&self) -> &str {
        REDUCE_SCATTER
    }
    fn execute(
        &self,
        inputs: &[&Array],
        _output_shape: &IrShape,
        attrs: &[u8],
    ) -> Result<Array, crate::array::MlxError> {
        // reduce_scatter attrs: [group_id: u64][world_size: u32][kind: u8].
        // The kind (0=Sum default) rides at byte 12, matching rlx-collectives.
        let kind = attrs.get(12).copied().unwrap_or(0) as std::os::raw::c_int;
        let _guard = crate::sync::runtime_guard();
        let mut out: *mut ffi::mlx_array_t = std::ptr::null_mut();
        let rc = unsafe { ffi::rlx_mlx_dist_reduce_scatter_array(inputs[0].ptr, kind, &mut out) };
        check(rc)?;
        Ok(Array::from_raw(out))
    }
}

// ── Megatron f / g operators (device-resident) ──
//
// `f` (copy_to_parallel) is the identity forward; `g` (reduce_from_parallel) is
// an all-reduce forward. Their *backward* rules — the reason the two exist —
// live in `rlx-collectives`' IR extension and only matter for training. MLX's
// device-resident path is forward inference, where `f` is a pass-through and
// `g` == all_reduce, so these kernels just cover the forward.

struct CopyToParallelExt;
impl OpExtension for CopyToParallelExt {
    fn name(&self) -> &str {
        COPY_TO_PARALLEL
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&IrShape], _attrs: &[u8]) -> IrShape {
        inputs[0].clone()
    }
}

struct CopyToParallelMlx;
impl MlxKernel for CopyToParallelMlx {
    fn name(&self) -> &str {
        COPY_TO_PARALLEL
    }
    fn execute(
        &self,
        inputs: &[&Array],
        _output_shape: &IrShape,
        _attrs: &[u8],
    ) -> Result<Array, crate::array::MlxError> {
        // Identity forward — pass the lazy array through unchanged.
        inputs[0].clone_handle()
    }
}

struct ReduceFromParallelExt;
impl OpExtension for ReduceFromParallelExt {
    fn name(&self) -> &str {
        REDUCE_FROM_PARALLEL
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&IrShape], _attrs: &[u8]) -> IrShape {
        inputs[0].clone()
    }
}

struct ReduceFromParallelMlx;
impl MlxKernel for ReduceFromParallelMlx {
    fn name(&self) -> &str {
        REDUCE_FROM_PARALLEL
    }
    fn execute(
        &self,
        inputs: &[&Array],
        _output_shape: &IrShape,
        attrs: &[u8],
    ) -> Result<Array, crate::array::MlxError> {
        // g's forward is an all-reduce; the kind rides at attrs[8] (0=Sum).
        let kind = attrs.get(8).copied().unwrap_or(0) as std::os::raw::c_int;
        let _guard = crate::sync::runtime_guard();
        let mut out: *mut ffi::mlx_array_t = std::ptr::null_mut();
        let rc = unsafe { ffi::rlx_mlx_dist_all_reduce_array(inputs[0].ptr, kind, &mut out) };
        check(rc)?;
        Ok(Array::from_raw(out))
    }
}

/// Install the device-resident `collective.*` ops on the MLX backend (IR shape
/// extensions + MLX kernels): `all_reduce`, `all_gather`, `reduce_scatter`, and
/// the Megatron `f`/`g` operators. Call once after [`MlxTransport::init`].
pub fn register_collective() {
    register_op(Arc::new(AllReduceExt));
    register_op(Arc::new(AllGatherExt));
    register_op(Arc::new(ReduceScatterExt));
    register_op(Arc::new(CopyToParallelExt));
    register_op(Arc::new(ReduceFromParallelExt));
    register_mlx_kernel(Arc::new(AllReduceMlx));
    register_mlx_kernel(Arc::new(AllGatherMlx));
    register_mlx_kernel(Arc::new(ReduceScatterMlx));
    register_mlx_kernel(Arc::new(CopyToParallelMlx));
    register_mlx_kernel(Arc::new(ReduceFromParallelMlx));
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests force the final native link against libmlx.a + libjaccl.a
    // and exercise the distributed C ABI. Multi-rank jaccl/ring needs MLX's
    // launcher (and, for jaccl, multiple Thunderbolt-linked machines); here
    // we verify the singleton path that runs without a launcher.

    #[test]
    fn is_available_links_and_runs() {
        // Just calling this resolves the shim + MLX + jaccl symbols.
        let _ = MlxTransport::is_available();
    }

    #[test]
    fn init_singleton_group_without_launcher() {
        // strict=false → MLX returns a size-1 group with no launcher, and
        // collectives degrade to no-ops.
        let t = MlxTransport::init(false, "any").expect("singleton init");
        assert_eq!(t.world_size(), 1);
        assert_eq!(t.rank(), 0);

        let mut data = vec![1.0f32, 2.0, 3.0];
        t.all_sum(&mut data).unwrap();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0],
            "all_sum on singleton is identity"
        );

        let gathered = t.all_gather(&[7.0, 8.0]).unwrap();
        assert_eq!(gathered, vec![7.0, 8.0]);

        // Transport-trait round-trip to self is a no-op-safe path too.
        use rlx_driver::Transport as _;
        assert_eq!(<MlxTransport as rlx_driver::Transport>::world_size(&t), 1);
    }

    #[test]
    fn device_resident_all_reduce_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        // Install the op + init the (singleton) group.
        register_collective();
        let _ = MlxTransport::init(false, "any");

        // Graph: x -> collective.all_reduce -> y, run on the MLX backend.
        // The all_sum executes device-resident (lazy array, no host copy);
        // on a size-1 group it's the identity, so y == x.
        let mut g = Graph::new("device_all_reduce");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.custom_op(ALL_REDUCE, vec![], vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn device_resident_all_reduce_max_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        register_collective();
        let _ = MlxTransport::init(false, "any");

        // attrs: 8-byte group id + kind byte 2 (= Max), matching
        // rlx-collectives' encode_kind. On a size-1 group every reduction is the
        // identity, so this exercises the kind decode + the `all_max` shim path
        // (`rlx_mlx_dist_all_reduce_array`) without needing multiple ranks.
        let mut g = Graph::new("device_all_reduce_max");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let mut attrs = vec![0u8; 8];
        attrs.push(2);
        let y = g.custom_op(ALL_REDUCE, attrs, vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// `[group_id: u64][world_size: u32]` attrs, as `rlx-collectives` encodes
    /// them. Singleton group ⇒ `world_size = 1`.
    fn gather_attrs(world_size: u32) -> Vec<u8> {
        let mut a = 0u64.to_le_bytes().to_vec();
        a.extend_from_slice(&world_size.to_le_bytes());
        a
    }

    #[test]
    fn device_resident_all_gather_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        register_collective();
        let _ = MlxTransport::init(false, "any");

        // On a size-1 group, all_gather is the identity (concatenate one shard).
        let mut g = Graph::new("device_all_gather");
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let y = g.custom_op(ALL_GATHER, gather_attrs(1), vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[5.0f32, 6.0, 7.0])]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn device_resident_reduce_scatter_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        register_collective();
        let _ = MlxTransport::init(false, "any");

        // On a size-1 group, reduce_scatter is the identity (all_sum of one
        // rank, then slice the whole axis-0 range).
        let mut g = Graph::new("device_reduce_scatter");
        let x = g.input("x", Shape::new(&[3], DType::F32));
        let y = g.custom_op(REDUCE_SCATTER, gather_attrs(1), vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[5.0f32, 6.0, 7.0])]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn device_resident_copy_to_parallel_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        register_collective();
        let _ = MlxTransport::init(false, "any");

        // f (copy_to_parallel) is the identity forward — output == input.
        let mut g = Graph::new("device_copy");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.custom_op(COPY_TO_PARALLEL, vec![0u8; 8], vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn device_resident_reduce_from_parallel_runs_on_mlx() {
        use crate::backend::MlxExecutable;
        use rlx_ir::{DType, Graph, Shape};

        register_collective();
        let _ = MlxTransport::init(false, "any");

        // g (reduce_from_parallel) is an all-reduce forward; on a size-1 group
        // it's the identity. attrs = 8-byte group id (kind byte omitted → Sum).
        let mut g = Graph::new("device_g");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.custom_op(REDUCE_FROM_PARALLEL, vec![0u8; 8], vec![x]);
        g.set_outputs(vec![y]);

        let mut exe = MlxExecutable::compile(g);
        let out = exe.run(&[("x", &[1.0f32, 2.0, 3.0, 4.0])]);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }
}
