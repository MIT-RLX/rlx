// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NCCL device-resident collectives for CUDA (WideEP / TP Tier 2).
//!
//! Enabled with `--features nccl`. Mirrors the MLX seam in
//! `rlx-mlx/src/distributed.rs`: register a [`Comm`] under a group id, then
//! `collective.all_reduce` can run on-device instead of
//! `GPU→host→TCP→host→GPU`.
//!
//! Bootstrap: rank 0 creates [`new_nccl_id`], broadcasts [`id_to_bytes`] over
//! any transport, peers reconstruct with [`id_from_bytes`], then each rank
//! calls [`init_and_register`].
//!
//! **Hardware:** needs ≥2 NVIDIA GPUs (or multi-node) with libnccl. Compiles
//! on hosts without NCCL via cudarc dynamic-loading; runtime calls fail
//! cleanly when the library is missing.

use cudarc::driver::{CudaSlice, CudaStream};
use cudarc::nccl::{Comm, Id, ReduceOp, group_end, group_start};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// NCCL `Comm` is a raw pointer wrapper and is not `Send`/`Sync` in cudarc.
/// Communicators are per-rank and used under the registry lock / from the
/// executing backend thread — mark them shareable for the static map.
pub struct NcclComm(Comm);

// SAFETY: NCCL communicators are intended for concurrent use from a single
// process with external synchronization; we only hand out `Arc` clones and
// run collectives under normal CUDA stream ordering.
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

fn comms() -> &'static RwLock<HashMap<u64, Arc<NcclComm>>> {
    static C: OnceLock<RwLock<HashMap<u64, Arc<NcclComm>>>> = OnceLock::new();
    C.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Create a fresh NCCL unique id (rank 0 only).
pub fn new_nccl_id() -> Result<Id, String> {
    match std::panic::catch_unwind(Id::new) {
        Ok(Ok(id)) => Ok(id),
        Ok(Err(e)) => Err(format!("nccl Id::new: {e:?}")),
        Err(_) => Err("nccl Id::new: libnccl not loadable (missing shared library)".into()),
    }
}

/// Serialize an NCCL id for transport bootstrap (128 bytes).
pub fn id_to_bytes(id: &Id) -> [u8; 128] {
    let internal = id.internal();
    let mut out = [0u8; 128];
    for (i, &c) in internal.iter().enumerate() {
        out[i] = c as u8;
    }
    out
}

/// Reconstruct an NCCL id from bootstrap bytes.
pub fn id_from_bytes(bytes: &[u8; 128]) -> Id {
    let mut internal = [0i8; 128];
    for (i, &b) in bytes.iter().enumerate() {
        internal[i] = b as i8;
    }
    Id::uninit(internal)
}

/// Build a communicator for this rank and register it under `group_id`.
pub fn init_and_register(
    group_id: u64,
    stream: Arc<CudaStream>,
    rank: usize,
    world_size: usize,
    id: Id,
) -> Result<Arc<NcclComm>, String> {
    let comm = Comm::from_rank(stream, rank, world_size, id)
        .map_err(|e| format!("nccl Comm::from_rank: {e:?}"))?;
    let arc = Arc::new(NcclComm(comm));
    register_nccl_comm(group_id, arc.clone());
    Ok(arc)
}

/// Publish a communicator for in-graph collectives keyed by `group_id`.
pub fn register_nccl_comm(group_id: u64, comm: Arc<NcclComm>) {
    comms().write().unwrap().insert(group_id, comm);
}

/// Drop the communicator for `group_id`.
pub fn unregister_nccl_comm(group_id: u64) {
    comms().write().unwrap().remove(&group_id);
}

/// Look up a registered communicator.
pub fn lookup_nccl_comm(group_id: u64) -> Option<Arc<NcclComm>> {
    comms().read().unwrap().get(&group_id).cloned()
}

fn group_id_from_attrs(attrs: &[u8]) -> Option<u64> {
    attrs.get(..8)?.try_into().ok().map(u64::from_le_bytes)
}

fn reduce_op_from_attrs(attrs: &[u8]) -> ReduceOp {
    // rlx-collectives all_reduce: [group_id:u64][kind:u8]…
    match attrs.get(8).copied().unwrap_or(0) {
        1 => ReduceOp::Avg, // Mean
        2 => ReduceOp::Max,
        3 => ReduceOp::Min,
        _ => ReduceOp::Sum,
    }
}

/// Device-resident in-place all-reduce on `buff[off..off+len]`.
///
/// Returns `Ok(true)` if an NCCL comm handled it, `Ok(false)` if no comm is
/// registered (caller should host-fallback), `Err` on NCCL failure.
pub fn try_all_reduce_f32(
    buffer: &mut CudaSlice<f32>,
    off: usize,
    len: usize,
    attrs: &[u8],
) -> Result<bool, String> {
    let Some(gid) = group_id_from_attrs(attrs) else {
        return Ok(false);
    };
    let Some(comm) = lookup_nccl_comm(gid) else {
        return Ok(false);
    };
    if len == 0 {
        return Ok(true);
    }
    let op = reduce_op_from_attrs(attrs);
    let mut view = buffer.slice_mut(off..off + len);
    comm.0
        .all_reduce_in_place(&mut view, &op)
        .map_err(|e| format!("nccl all_reduce: {e:?}"))?;
    Ok(true)
}

/// Equal-chunk device all-to-all via NCCL send/recv (WideEP pad path).
///
/// Returns `Ok(true)` if handled, `Ok(false)` if no comm registered.
pub fn try_all_to_all_equal_f32(
    send: &CudaSlice<f32>,
    recv: &mut CudaSlice<f32>,
    attrs: &[u8],
) -> Result<bool, String> {
    let Some(gid) = group_id_from_attrs(attrs) else {
        return Ok(false);
    };
    let Some(comm) = lookup_nccl_comm(gid) else {
        return Ok(false);
    };
    let world = comm.0.world_size();
    let rank = comm.0.rank();
    if send.len() != recv.len() || !send.len().is_multiple_of(world) {
        return Err(format!(
            "nccl all_to_all: len {} not divisible by world {world}",
            send.len()
        ));
    }
    let chunk = send.len() / world;
    if chunk == 0 {
        return Ok(true);
    }

    let stream = comm.0.stream();
    stream
        .memcpy_dtod(
            &send.slice(rank * chunk..(rank + 1) * chunk),
            &mut recv.slice_mut(rank * chunk..(rank + 1) * chunk),
        )
        .map_err(|e| format!("nccl all_to_all self copy: {e:?}"))?;

    group_start().map_err(|e| format!("nccl group_start: {e:?}"))?;
    for peer in 0..world {
        if peer == rank {
            continue;
        }
        let send_view = send.slice(peer * chunk..(peer + 1) * chunk);
        comm.0
            .send(&send_view, peer as i32)
            .map_err(|e| format!("nccl send→{peer}: {e:?}"))?;
    }
    for peer in 0..world {
        if peer == rank {
            continue;
        }
        let mut recv_view = recv.slice_mut(peer * chunk..(peer + 1) * chunk);
        comm.0
            .recv(&mut recv_view, peer as i32)
            .map_err(|e| format!("nccl recv←{peer}: {e:?}"))?;
    }
    group_end().map_err(|e| format!("nccl group_end: {e:?}"))?;
    Ok(true)
}

/// Whether this build includes the `nccl` cargo feature.
pub const NCCL_FEATURE: bool = true;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_bytes_roundtrip() {
        let id = match std::panic::catch_unwind(Id::new) {
            Ok(Ok(id)) => id,
            Ok(Err(_)) | Err(_) => {
                eprintln!("skip id_bytes_roundtrip: libnccl unavailable");
                return;
            }
        };
        let bytes = id_to_bytes(&id);
        let back = id_from_bytes(&bytes);
        assert_eq!(id_to_bytes(&back), bytes);
    }

    #[test]
    fn registry_empty_lookup() {
        assert!(lookup_nccl_comm(0xDEAD_BEEF).is_none());
        unregister_nccl_comm(0xDEAD_BEEF);
    }

    #[test]
    fn feature_flag_is_on() {
        assert!(NCCL_FEATURE);
    }
}
