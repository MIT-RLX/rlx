// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed nodes: `RlxGroup` joins a mesh; point-to-point messages,
//! collectives and the ship-graph protocol. Every call blocks until its peers
//! take part, so run each rank on its own thread, process or machine.

use std::ffi::c_char;

use rlx_bind::runtime::dist::{self, Group};

use crate::ffi::{RlxF32s, arg, borrow, guard, handle, out, release};
use crate::graph::{RlxGraph, graph_mut};

/// This process's membership in a mesh of rlx nodes.
pub struct RlxGroup(Group);

/// `world` free loopback addresses (`"127.0.0.1:port"`) as a JSON array, for
/// running every rank on one machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_loopback_peers(world: u32, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let peers = dist::loopback_peers(world)?;
        out::strings(out, &peers, "rlx_loopback_peers out")
    })
}

/// Join a mesh as `rank` of `world`; blocks until every peer connects.
/// `options_json`: `{"peers": ["host:port", …], "topology": "mesh" | "star",
/// "device": "auto", "discover": {"port", "data_port", "via"}, "heap_mb"}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_connect(
    rank: u32,
    world: u32,
    options_json: *const c_char,
    out: *mut *mut RlxGroup,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_group_connect options")?;
        let g = Group::connect(rank, world, &opts)?;
        out::put(out, handle(RlxGroup(g)), "rlx_group_connect out")
    })
}

/// Join with `RANK` / `WORLD` / `PEERS` / `TOPOLOGY` / `DEVICE` / `DISCOVER`
/// from the environment.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_from_env(out: *mut *mut RlxGroup) -> i32 {
    guard(|| unsafe {
        let g = Group::from_env()?;
        out::put(out, handle(RlxGroup(g)), "rlx_group_from_env out")
    })
}

/// Leave the mesh and free the group (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_free(g: *mut RlxGroup) {
    unsafe { release(g) }
}

/// This node's rank.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_rank(g: *mut RlxGroup, out: *mut u32) -> i32 {
    guard(|| unsafe { out::put(out, borrow(g, "group")?.0.rank(), "rlx_group_rank out") })
}

/// How many ranks the mesh has.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_world(g: *mut RlxGroup, out: *mut u32) -> i32 {
    guard(|| unsafe { out::put(out, borrow(g, "group")?.0.world(), "rlx_group_world out") })
}

/// Whether this is rank 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_is_leader(g: *mut RlxGroup, out: *mut bool) -> i32 {
    guard(|| unsafe {
        out::put(
            out,
            borrow(g, "group")?.0.is_leader(),
            "rlx_group_is_leader out",
        )
    })
}

/// Wait until every rank arrives.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_barrier(g: *mut RlxGroup) -> i32 {
    guard(|| unsafe { borrow(g, "group")?.0.barrier() })
}

/// Send `len` f32 values to rank `to` under `tag`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_send(
    g: *mut RlxGroup,
    to: u32,
    tag: u32,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let data = arg::slice(data, len, "rlx_group_send data")?;
        borrow(g, "group")?.0.send(to, tag, data)
    })
}

/// Receive f32 values from rank `from` under `tag`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_recv(
    g: *mut RlxGroup,
    from: u32,
    tag: u32,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let v = borrow(g, "group")?.0.recv(from, tag)?;
        out::put(out, v.into(), "rlx_group_recv out")
    })
}

/// Reduce `data` across every rank; `op` (`NULL` = `"sum"`) is `"sum"`,
/// `"mean"`, `"max"` or `"min"`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_all_reduce(
    g: *mut RlxGroup,
    data: *const f32,
    len: usize,
    op: *const c_char,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let data = arg::slice(data, len, "rlx_group_all_reduce data")?;
        let op = arg::str_or(op, "sum", "rlx_group_all_reduce op")?;
        let v = borrow(g, "group")?.0.all_reduce(data, &op)?;
        out::put(out, v.into(), "rlx_group_all_reduce out")
    })
}

/// Every rank's `data`, concatenated in rank order.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_all_gather(
    g: *mut RlxGroup,
    data: *const f32,
    len: usize,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let data = arg::slice(data, len, "rlx_group_all_gather data")?;
        let v = borrow(g, "group")?.0.all_gather(data)?;
        out::put(out, v.into(), "rlx_group_all_gather out")
    })
}

/// Rank `root`'s `data`, on every rank.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_broadcast(
    g: *mut RlxGroup,
    root: u32,
    data: *const f32,
    len: usize,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let data = arg::slice(data, len, "rlx_group_broadcast data")?;
        let v = borrow(g, "group")?.0.broadcast(root, data)?;
        out::put(out, v.into(), "rlx_group_broadcast out")
    })
}

/// Coordinator: the capabilities each worker announced, as JSON
/// `[{rank, devices, platform, role}]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_collect_caps(g: *mut RlxGroup, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let caps = borrow(g, "group")?.0.collect_caps()?;
        out::json(out, &caps, "rlx_group_collect_caps out")
    })
}

/// Coordinator: ship a stage of `graph` (borrowed) to worker `rank`.
/// `options_json`: `{input, output, device, weights: {param: uri}, packed:
/// {param: uri}, stage_id}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_ship_stage(
    g: *mut RlxGroup,
    rank: u32,
    graph: *mut RlxGraph,
    options_json: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_group_ship_stage options")?;
        borrow(g, "group")?
            .0
            .ship_stage(rank, graph_mut(graph)?, &opts)
    })
}

/// Send an activation to rank `to`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_send_activation(
    g: *mut RlxGroup,
    to: u32,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let data = arg::slice(data, len, "rlx_group_send_activation data")?;
        borrow(g, "group")?.0.send_activation(to, data)
    })
}

/// Receive an activation from rank `from`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_recv_activation(
    g: *mut RlxGroup,
    from: u32,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let v = borrow(g, "group")?.0.recv_activation(from)?;
        out::put(out, v.into(), "rlx_group_recv_activation out")
    })
}

/// Worker: announce, receive a stage, serve activations until the coordinator
/// finishes (or `{"steps": n}`). Writes `{rank, device, platform, activations}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_serve_worker(
    g: *mut RlxGroup,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_group_serve_worker options")?;
        let report = borrow(g, "group")?.0.serve_worker(&opts)?;
        out::json(out, &report, "rlx_group_serve_worker out")
    })
}

/// Coordinator: ship a training job for backward `graph` (borrowed) to worker
/// `rank`. `spec_json`: `{params: [{name, uri}], data: [{input, uri, elem,
/// shard_start, shard_len}], grad_start, loss_index, seed_input, momentum,
/// lr_per_epoch, batch, device, push_data}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_ship_train(
    g: *mut RlxGroup,
    rank: u32,
    graph: *mut RlxGraph,
    spec_json: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let spec = arg::json(spec_json, "rlx_group_ship_train spec")?;
        borrow(g, "group")?
            .0
            .ship_train(rank, graph_mut(graph)?, &spec)
    })
}

/// Train this rank's shard of a job in lockstep with the workers; writes
/// `{metrics, params}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_run_train(
    g: *mut RlxGroup,
    graph: *mut RlxGraph,
    spec_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let spec = arg::json(spec_json, "rlx_group_run_train spec")?;
        let report = borrow(g, "group")?.0.run_train(graph_mut(graph)?, &spec)?;
        out::json(out, &report, "rlx_group_run_train out")
    })
}

/// Worker: announce, receive a training job, train in lockstep. Writes
/// `{rank, platform, metrics, params}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_group_serve_trainer(
    g: *mut RlxGroup,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_group_serve_trainer options")?;
        let report = borrow(g, "group")?.0.serve_trainer(&opts)?;
        out::json(out, &report, "rlx_group_serve_trainer out")
    })
}
