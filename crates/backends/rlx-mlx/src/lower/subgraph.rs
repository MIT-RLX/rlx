// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leaf/subgraph construction.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use rlx_ir::RegionPrologue;
use rlx_ir::op::{
    Activation, AdaNormKind, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind, ReduceOp,
    RopeStyle, ScaleMode, SteKind, TransformStep,
};
use rlx_ir::shape::{Dim, DimBinding, Shape};
use rlx_ir::{DType, Graph, NodeId, Op};

use crate::array::{Array, MlxError, async_eval, eval};
use crate::ffi::{MlxMask, MlxReduce, MlxUnary};
use crate::ops;

use super::helpers::*;
use super::*;

pub fn expand_leaf_env(
    graph: &Graph,
    mut env: HashMap<NodeId, Array>,
) -> Result<HashMap<NodeId, Array>, MlxError> {
    let mut canon_input: HashMap<String, NodeId> = HashMap::new();
    let mut canon_param: HashMap<String, NodeId> = HashMap::new();
    for (id, key) in compile_leaf_order(graph) {
        match key {
            LeafKey::Input(name) => {
                canon_input.insert(name, id);
            }
            LeafKey::Param(name) => {
                canon_param.insert(name, id);
            }
            LeafKey::Constant => {}
        }
    }

    for node in graph.nodes() {
        if env.contains_key(&node.id) {
            continue;
        }
        match &node.op {
            Op::Input { name } => {
                let canon = *canon_input.get(name).ok_or_else(|| {
                    MlxError(format!("expand_leaf_env: missing canonical input '{name}'"))
                })?;
                let arr = env.get(&canon).ok_or_else(|| {
                    MlxError(format!("expand_leaf_env: canonical input '{name}' unbound"))
                })?;
                env.insert(node.id, arr.clone_handle()?);
            }
            Op::Param { name } => {
                let canon = *canon_param.get(name).ok_or_else(|| {
                    MlxError(format!("expand_leaf_env: missing canonical param '{name}'"))
                })?;
                let arr = env.get(&canon).ok_or_else(|| {
                    MlxError(format!("expand_leaf_env: canonical param '{name}' unbound"))
                })?;
                env.insert(node.id, arr.clone_handle()?);
            }
            Op::Constant { .. } => {
                return Err(MlxError(format!(
                    "expand_leaf_env: constant leaf {:?} not bound",
                    node.id
                )));
            }
            _ => {}
        }
    }
    Ok(env)
}

/// Expand scalar host buffers to match a batched graph leaf when vmap
/// left a shared `[1]` binding but the lifted node is `[B, …]`.
pub(crate) fn broadcast_leaf_data(
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<Vec<f32>, MlxError> {
    let product: usize = shape.iter().product();
    if data.len() == product {
        return Ok(data.to_vec());
    }
    if data.len() == 1 && product > 1 {
        return Ok(vec![data[0]; product]);
    }
    Err(MlxError(format!(
        "leaf '{name}': host len {} != shape {shape:?} product {product}",
        data.len()
    )))
}

/// Build the leaf array for a single node. Prefers typed bytes if a
/// matching name appears in `inputs_typed` / `params_typed`; falls
/// back to the f32 host map. The typed path uses Array::from_bytes
/// for zero-widen F16/BF16 / I32 leaves.
pub fn build_leaf_for(
    graph: &Graph,
    id: NodeId,
    params: &HashMap<String, Vec<f32>>,
    inputs: &HashMap<String, Vec<f32>>,
    params_typed: &HashMap<String, (Vec<u8>, DType)>,
    inputs_typed: &HashMap<String, (Vec<u8>, DType)>,
    gpu_inputs: Option<&HashMap<String, Array>>,
) -> Result<Array, MlxError> {
    let node = graph.node(id);
    let shape: Vec<usize> = node
        .shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let dtype = node.shape.dtype();
    match &node.op {
        Op::Input { name } => {
            if let Some(map) = gpu_inputs {
                if let Some(arr) = map.get(name) {
                    return arr.clone_handle();
                }
            }
            if let Some((bytes, dt)) = inputs_typed.get(name) {
                if *dt != dtype {
                    return Err(MlxError(format!(
                        "typed input '{name}' dtype {dt:?} doesn't match graph's {dtype:?}"
                    )));
                }
                if dtype == DType::C64 {
                    return mlx_c64_leaf_from_bytes(bytes);
                }
                if dtype == DType::C128 {
                    return mlx_c128_leaf_from_bytes(bytes);
                }
                let want: usize = shape.iter().product::<usize>() * dt.size_bytes();
                if bytes.len() != want && std::env::var_os("RLX_DIM_DBG").is_some() {
                    eprintln!(
                        "[leaf] input '{name}' static_shape={shape:?} bytes={} want={want}",
                        bytes.len()
                    );
                }
                return Array::from_bytes(bytes, &shape, dtype);
            }
            let data = inputs
                .get(name)
                .ok_or_else(|| MlxError(format!("missing input '{name}'")))?;
            let data = broadcast_leaf_data(name, data, &shape)?;
            Array::from_f32_slice(&data, &shape, dtype)
        }
        Op::Param { name } => {
            if let Some((bytes, dt)) = params_typed.get(name) {
                if *dt != dtype {
                    return Err(MlxError(format!(
                        "typed param '{name}' dtype {dt:?} doesn't match graph's {dtype:?}"
                    )));
                }
                if dtype == DType::C64 {
                    return mlx_c64_leaf_from_bytes(bytes);
                }
                if dtype == DType::C128 {
                    return mlx_c128_leaf_from_bytes(bytes);
                }
                return Array::from_bytes(bytes, &shape, dtype);
            }
            let data = params
                .get(name)
                .ok_or_else(|| MlxError(format!("missing param '{name}'")))?;
            // Fast path: f32 param whose host buffer already matches
            // the graph-declared shape (no broadcast needed). Wrap as
            // a zero-copy view over the caller-owned `Vec<f32>` — the
            // MlxExecutable mutates the Vec in place on set_param
            // calls, keeping the buffer address stable across runs.
            //
            // SAFETY: `data` lives in `MlxExecutable::params` (a
            // HashMap of Vec<f32>) which outlives the Array. The
            // executable syncs MLX evaluation in `run_internal`
            // before returning, so the Array is no longer referenced
            // by MLX by the time the next `set_param` mutates the
            // buffer.
            // Param zero-copy view: previously default, now opt-in via
            // `RLX_MLX_PARAM_VIEW=1`. Holding an MLX Array as a *view*
            // over a host Vec across multiple `compiled.run()` calls
            // breaks parity on Gemma 4 E2B QAT — same prefill graph
            // re-invoked with a new `input_ids[5]` returned stale
            // logits (the value MLX computed on the previous call,
            // i.e. position-5 of the original prompt) while a freshly
            // compiled executable returned the correct value. The
            // safety comment below ("Array is no longer referenced by
            // MLX by the time the next set_param mutates the buffer")
            // is true for `set_param`-driven mutations but does NOT
            // cover MLX's internal compile-trace cache, which appears
            // to retain a reference to the view past the eval barrier.
            // Default to the safe copying path; callers that have
            // proven their lifetime story can re-enable the view.
            if dtype == DType::F32
                && data.len() == shape.iter().product::<usize>()
                && std::env::var("RLX_MLX_PARAM_VIEW").as_deref() == Ok("1")
            {
                return unsafe { Array::from_f32_slice_view(data, &shape) };
            }
            let data = broadcast_leaf_data(name, data, &shape)?;
            Array::from_f32_slice(&data, &shape, dtype)
        }
        Op::Constant { data } => {
            // Constants are little-endian raw bytes in the node's
            // dtype. Every dtype rlx-ir declares has a native MLX
            // counterpart; from_bytes handles the typed read directly.
            // F32 still goes through the iterator path because that
            // matches the prior behavior bit-for-bit.
            match dtype {
                DType::F32 => {
                    let n = data.len() / 4;
                    let mut buf = Vec::with_capacity(n);
                    for i in 0..n {
                        let bytes = [
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        ];
                        buf.push(f32::from_le_bytes(bytes));
                    }
                    Array::from_f32_slice(&buf, &shape, dtype)
                }
                // C64 has no native MLX dtype — materialize as the
                // f32-backed interleaved (re, im) representation.
                DType::C64 => mlx_c64_leaf_from_bytes(data),
                // C128 → F64-backed interleaved (re, im) f64 representation.
                DType::C128 => mlx_c128_leaf_from_bytes(data),
                _ => Array::from_bytes(data, &shape, dtype),
            }
        }
        other => Err(MlxError(format!("build_leaf called on non-leaf {other:?}"))),
    }
}

/// Lower a sub-graph (then/else branch of `Op::If`, or body/cond of
/// `Op::While`). Captures bind positionally: the i-th `Op::Input` in
/// the sub-graph (in topo order) is bound to `captures[i]`. Params
/// look up in the parent's `params` / `params_typed` by name. Every
/// leaf array gets a fresh `clone_handle` so the parent's ownership
/// is undisturbed.
pub fn lower_subgraph(
    sub: &Graph,
    captures: &[&Array],
    parent_params: &HashMap<String, Vec<f32>>,
    parent_params_typed: &HashMap<String, (Vec<u8>, DType)>,
    rng: rlx_ir::RngOptions,
) -> Result<Vec<Array>, MlxError> {
    let mut sub_env: HashMap<NodeId, Array> = HashMap::with_capacity(sub.nodes().len());
    let mut canon_param: HashMap<String, NodeId> = HashMap::new();

    let mut input_idx = 0;
    for node in sub.nodes() {
        match &node.op {
            Op::Input { name } => {
                if input_idx >= captures.len() {
                    return Err(MlxError(format!(
                        "sub-graph has more Op::Input nodes than parent supplied \
                         captures (input #{input_idx} = {name:?})"
                    )));
                }
                sub_env.insert(node.id, captures[input_idx].clone_handle()?);
                input_idx += 1;
            }
            Op::Param { name } => {
                if let Some(&canon_id) = canon_param.get(name) {
                    let arr = sub_env.get(&canon_id).ok_or_else(|| {
                        MlxError(format!("sub-graph canonical param '{name}' missing"))
                    })?;
                    sub_env.insert(node.id, arr.clone_handle()?);
                    continue;
                }
                let leaf = if let Some((bytes, dt)) = parent_params_typed.get(name) {
                    let shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    Array::from_bytes(bytes, &shape, *dt)?
                } else if let Some(data) = parent_params.get(name) {
                    let shape: Vec<usize> = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| d.unwrap_static())
                        .collect();
                    let dtype = node.shape.dtype();
                    Array::from_f32_slice(data, &shape, dtype)?
                } else {
                    return Err(MlxError(format!(
                        "sub-graph param '{name}' not found in parent's param maps"
                    )));
                };
                canon_param.insert(name.clone(), node.id);
                sub_env.insert(node.id, leaf);
            }
            Op::Constant { data } => {
                let shape: Vec<usize> = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static())
                    .collect();
                let dtype = node.shape.dtype();
                let leaf = match dtype {
                    DType::F32 => {
                        let n = data.len() / 4;
                        let mut buf = Vec::with_capacity(n);
                        for i in 0..n {
                            let bytes = [
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ];
                            buf.push(f32::from_le_bytes(bytes));
                        }
                        Array::from_f32_slice(&buf, &shape, dtype)?
                    }
                    _ => Array::from_bytes(data, &shape, dtype)?,
                };
                sub_env.insert(node.id, leaf);
            }
            _ => {} // non-leaf: handled by lower_with_env
        }
    }

    if input_idx < captures.len() {
        // More captures than the sub-graph used. Not necessarily an
        // error — extra captures may have been provided "in case" —
        // but worth a debug-friendly note. For now silently allow.
    }

    lower_with_env(sub, sub_env, parent_params, parent_params_typed, rng, false)
}
