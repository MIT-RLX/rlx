// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed nodes: `rlx.Group` joins a mesh of rlx nodes — messages,
//! collectives, and the ship-graph protocol (a coordinator ships a stage or a
//! training job; workers run it on their own hardware).
//!
//! ```js
//! const g = rlx.Group.connect(0, 2, { peers: ["10.0.0.1:29500", "10.0.0.2:29500"] });
//! g.allReduce(new Float32Array([1]), "sum");
//! ```
//!
//! Every call blocks until its peers take part. A script is one thread, so it
//! is one rank; the other ranks are other processes or machines.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::dist::{self, Group};

use crate::api::bind::{lift, report, to_json};
use crate::api::graph::graph_ref;
use crate::convert::*;
use crate::handle::{self, CLASS_GROUP};

fn group_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Group> {
    handle::borrow_mut(ctx, this, CLASS_GROUP, "rlx.Group")
}

fn rank_arg(ctx: &mut Context, v: &Value, what: &str) -> JsResult<u32> {
    Ok(to_usize(ctx, v, what)? as u32)
}

fn f32s(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<f32>> {
    to_f32_vec(ctx, v, what)
}

/// `loopbackPeers(world)` → `["127.0.0.1:port", …]`, free ports on this machine.
fn f_loopback_peers(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let world = rank_arg(ctx, arg(args, 0), "loopbackPeers world")?;
    let peers = lift(ctx, dist::loopback_peers(world))?;
    Ok(new_string_array(ctx, &peers))
}

/// `Group.connect(rank, world, {peers, topology, device, discover, heapMb})`.
fn group_connect(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let rank = rank_arg(ctx, arg(args, 0), "Group.connect rank")?;
    let world = rank_arg(ctx, arg(args, 1), "Group.connect world")?;
    let opts = to_json(ctx, arg(args, 2))?;
    let g = lift(ctx, Group::connect(rank, world, &opts))?;
    Ok(handle::wrap(ctx, CLASS_GROUP, g))
}

/// `Group.fromEnv()` — `RANK` / `WORLD` / `PEERS` / … from the environment.
fn group_from_env(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let g = lift(ctx, Group::from_env())?;
    Ok(handle::wrap(ctx, CLASS_GROUP, g))
}

fn group_rank(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(Value::number(group_of(ctx, this)?.rank() as f64))
}

fn group_world(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(Value::number(group_of(ctx, this)?.world() as f64))
}

fn group_is_leader(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(Value::Bool(group_of(ctx, this)?.is_leader()))
}

fn group_barrier(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let r = group_of(ctx, this)?.barrier();
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// `send(to, tag, data)`.
fn group_send(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let to = rank_arg(ctx, arg(args, 0), "send to")?;
    let tag = rank_arg(ctx, arg(args, 1), "send tag")?;
    let data = f32s(ctx, arg(args, 2), "send data")?;
    let r = group_of(ctx, this)?.send(to, tag, &data);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// `recv(from, tag)` → `Float32Array`.
fn group_recv(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let from = rank_arg(ctx, arg(args, 0), "recv from")?;
    let tag = rank_arg(ctx, arg(args, 1), "recv tag")?;
    let r = group_of(ctx, this)?.recv(from, tag);
    let v = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &v))
}

/// `allReduce(data, op = "sum")` → `Float32Array`.
fn group_all_reduce(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let data = f32s(ctx, arg(args, 0), "allReduce data")?;
    let op = to_string_or(ctx, arg(args, 1), "sum")?;
    let r = group_of(ctx, this)?.all_reduce(&data, &op);
    let v = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &v))
}

fn group_all_gather(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let data = f32s(ctx, arg(args, 0), "allGather data")?;
    let r = group_of(ctx, this)?.all_gather(&data);
    let v = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &v))
}

/// `broadcast(root, data)` — other ranks pass a buffer of the root's length.
fn group_broadcast(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let root = rank_arg(ctx, arg(args, 0), "broadcast root")?;
    let data = f32s(ctx, arg(args, 1), "broadcast data")?;
    let r = group_of(ctx, this)?.broadcast(root, &data);
    let v = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &v))
}

fn group_collect_caps(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let r = group_of(ctx, this)?.collect_caps();
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

/// `shipStage(rank, graph, {input, output, device, weights, packed, stageId})`.
fn group_ship_stage(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let rank = rank_arg(ctx, arg(args, 0), "shipStage rank")?;
    let opts = to_json(ctx, arg(args, 2))?;
    let graph = graph_ref(ctx, arg(args, 1))?.clone();
    let r = group_of(ctx, this)?.ship_stage(rank, &graph, &opts);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

fn group_send_activation(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let to = rank_arg(ctx, arg(args, 0), "sendActivation to")?;
    let data = f32s(ctx, arg(args, 1), "sendActivation data")?;
    let r = group_of(ctx, this)?.send_activation(to, &data);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

fn group_recv_activation(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let from = rank_arg(ctx, arg(args, 0), "recvActivation from")?;
    let r = group_of(ctx, this)?.recv_activation(from);
    let v = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &v))
}

/// `serveWorker({steps})` → `{rank, device, platform, activations}`.
fn group_serve_worker(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let opts = to_json(ctx, arg(args, 0))?;
    let r = group_of(ctx, this)?.serve_worker(&opts);
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

/// `shipTrain(rank, backwardGraph, spec)`.
fn group_ship_train(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let rank = rank_arg(ctx, arg(args, 0), "shipTrain rank")?;
    let spec = to_json(ctx, arg(args, 2))?;
    let graph = graph_ref(ctx, arg(args, 1))?.clone();
    let r = group_of(ctx, this)?.ship_train(rank, &graph, &spec);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// `runTrain(backwardGraph, spec)` → `{metrics, params}`.
fn group_run_train(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let spec = to_json(ctx, arg(args, 1))?;
    let graph = graph_ref(ctx, arg(args, 0))?.clone();
    let r = group_of(ctx, this)?.run_train(&graph, &spec);
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

/// `serveTrainer({log})` → `{rank, platform, metrics, params}`.
fn group_serve_trainer(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let opts = to_json(ctx, arg(args, 0))?;
    let r = group_of(ctx, this)?.serve_trainer(&opts);
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "loopbackPeers" => f_loopback_peers, 1;
    };
    // No constructor: a group only comes from `connect` / `fromEnv`.
    js_class! {
        ctx, namespace;
        name: "Group",
        class: CLASS_GROUP,
        methods: {
            "rank" => group_rank, 0;
            "world" => group_world, 0;
            "isLeader" => group_is_leader, 0;
            "barrier" => group_barrier, 0;
            "send" => group_send, 3;
            "recv" => group_recv, 2;
            "allReduce" => group_all_reduce, 2;
            "allGather" => group_all_gather, 1;
            "broadcast" => group_broadcast, 2;
            "collectCaps" => group_collect_caps, 0;
            "shipStage" => group_ship_stage, 3;
            "sendActivation" => group_send_activation, 2;
            "recvActivation" => group_recv_activation, 1;
            "serveWorker" => group_serve_worker, 1;
            "shipTrain" => group_ship_train, 3;
            "runTrain" => group_run_train, 2;
            "serveTrainer" => group_serve_trainer, 1;
        }
    };
    js_statics! {
        ctx, namespace, "Group";
        "connect" => group_connect, 3;
        "fromEnv" => group_from_env, 0;
    };
}
