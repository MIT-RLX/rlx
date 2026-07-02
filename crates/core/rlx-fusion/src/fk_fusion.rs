// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! FKL-inspired transform / prologue / batch fusion passes.

use crate::fusion_fragment::{prologue_for_transform_op, transform_chain_eligible};
use crate::graph_rewrite::Rewriter;
use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::{Graph, NodeId, Op, Shape};
use std::collections::HashMap;

fn consumer_counts(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &input in &node.inputs {
            *consumers.entry(input).or_insert(0) += 1;
        }
    }
    for &out in &graph.outputs {
        *consumers.entry(out).or_insert(0) += 1;
    }
    consumers
}

fn is_prologue_candidate(
    graph: &Graph,
    resize_id: NodeId,
    consumers: &HashMap<NodeId, usize>,
) -> bool {
    if consumers.get(&resize_id).copied() != Some(1) {
        return false;
    }
    let Some(consumer_id) = graph
        .nodes()
        .iter()
        .find(|n| n.inputs.contains(&resize_id))
        .map(|n| n.id)
    else {
        return false;
    };
    let consumer = graph.node(consumer_id);
    if matches!(
        consumer.op,
        Op::ElementwiseRegion {
            prologue: RegionPrologue::None,
            ..
        }
    ) {
        return consumer.inputs.contains(&resize_id);
    }
    // Unary tail (`relu`, same-dtype `cast`, ...) merged by [`FuseRegionPrologue`].
    consumer.inputs.len() == 1
        && consumer.inputs[0] == resize_id
        && unary_region_step_from_op(graph, consumer).is_some()
}

/// Swap `ChainOperand::Input(a)` ? `Input(b)` throughout a region chain.
fn remap_chain_input_slots(chain: &[ChainStep], a: u32, b: u32) -> Vec<ChainStep> {
    let remap = |op: ChainOperand| -> ChainOperand {
        match op {
            ChainOperand::Input(i) if i == a => ChainOperand::Input(b),
            ChainOperand::Input(i) if i == b => ChainOperand::Input(a),
            other => other,
        }
    };
    chain
        .iter()
        .map(|step| match step {
            ChainStep::Activation(act, x) => ChainStep::Activation(*act, remap(*x)),
            ChainStep::Cast(dt, x) => ChainStep::Cast(*dt, remap(*x)),
            ChainStep::Binary(op, l, r) => ChainStep::Binary(*op, remap(*l), remap(*r)),
            ChainStep::Compare(op, l, r) => ChainStep::Compare(*op, remap(*l), remap(*r)),
            ChainStep::Where(c, t, f) => ChainStep::Where(remap(*c), remap(*t), remap(*f)),
        })
        .collect()
}

fn swap_region_input_metadata(
    scalar_input_mask: u32,
    input_modulus: [u32; 16],
    a: usize,
    b: usize,
) -> (u32, [u32; 16]) {
    let mut mask = scalar_input_mask;
    let bit_a = 1u32 << a;
    let bit_b = 1u32 << b;
    let a_set = mask & bit_a != 0;
    let b_set = mask & bit_b != 0;
    mask = (mask & !(bit_a | bit_b))
        | (if a_set { bit_b } else { 0 })
        | (if b_set { bit_a } else { 0 });
    let mut modulus = input_modulus;
    modulus.swap(a, b);
    (mask, modulus)
}

/// Collapse single-consumer transform chains into [`Op::TransformRegion`].
///
/// Skips `ResizeNearest2x ? ElementwiseRegion` (handled by [`FuseRegionPrologue`]).
pub struct MarkTransformRegions;

impl Pass for MarkTransformRegions {
    fn name(&self) -> &str {
        "mark_transform_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        let consumers = consumer_counts(&graph);
        let mut chain_members: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in graph.nodes() {
            if !transform_chain_eligible(&node.op) {
                continue;
            }
            if is_prologue_candidate(&graph, node.id, &consumers) {
                continue;
            }
            let mut members = vec![node.id];
            let mut cur = node.id;
            if consumers.get(&node.id).copied().unwrap_or(0) == 1 {
                while let Some(next_node) = graph.nodes().iter().find(|n| {
                    n.inputs.len() == 1 && n.inputs[0] == cur && transform_chain_eligible(&n.op)
                }) {
                    if consumers.get(&next_node.id).copied().unwrap_or(0) != 1 {
                        break;
                    }
                    if is_prologue_candidate(&graph, next_node.id, &consumers) {
                        break;
                    }
                    members.push(next_node.id);
                    cur = next_node.id;
                }
            }
            let tail = *members.last().unwrap();
            chain_members.insert(tail, members);
        }

        if chain_members.is_empty() {
            return graph;
        }

        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for (tail, members) in &chain_members {
            for &m in members.iter().filter(|id| **id != *tail) {
                fused_away.insert(m, ());
            }
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some(members) = chain_members.get(&node.id) {
                let external_input = graph.node(members[0]).inputs[0];
                rw.ensure_mapped(&graph, std::slice::from_ref(&external_input));
                let steps: Vec<TransformStep> = (0..members.len())
                    .map(|_| TransformStep::ResizeNearest2x(ChainOperand::Input(0)))
                    .collect();
                let tail_shape = graph.node(*members.last().unwrap()).shape.clone();
                let fused = rw.add_fused(
                    Op::TransformRegion {
                        steps,
                        num_inputs: 1,
                    },
                    std::slice::from_ref(&external_input),
                    tail_shape,
                );
                for m in members {
                    rw.replace(*m, fused);
                }
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

/// Fuse `ResizeNearest2x ? ElementwiseRegion` (or unary `Activation` / `Cast`) into a
/// closed region with prologue.
pub struct FuseRegionPrologue;

/// Unary ops that can close a resize prologue region (one input, same shape as output).
fn unary_region_step_from_op(graph: &Graph, node: &rlx_ir::Node) -> Option<ChainStep> {
    match &node.op {
        Op::Activation(a) => Some(ChainStep::Activation(*a, ChainOperand::Input(0))),
        Op::Cast { to } => {
            let in_dt = graph.shape(node.inputs[0]).dtype();
            if *to == in_dt {
                Some(ChainStep::Cast(*to, ChainOperand::Input(0)))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_fuse_resize_into_region(
    rw: &mut Rewriter,
    graph: &Graph,
    consumers: &HashMap<NodeId, usize>,
    fused_resize: &mut HashMap<NodeId, ()>,
    resize_id: NodeId,
    resize_slot: usize,
    chain: Vec<ChainStep>,
    num_inputs: u32,
    scalar_input_mask: u32,
    input_modulus: [u32; 16],
    out_shape: Shape,
    region_inputs: &[NodeId],
    consumer_id: NodeId,
) -> bool {
    if fused_resize.contains_key(&resize_id) {
        return false;
    }
    if region_inputs.get(resize_slot).copied() != Some(resize_id) {
        return false;
    }
    let resize_node = graph.node(resize_id);
    if !matches!(resize_node.op, Op::ResizeNearest2x) {
        return false;
    }
    if consumers.get(&resize_id).copied() != Some(1) {
        return false;
    }
    let Some(prologue) = prologue_for_transform_op(&resize_node.op) else {
        return false;
    };
    let resize_input = resize_node.inputs[0];
    let mut inputs: Vec<NodeId> = region_inputs.to_vec();
    let mut chain = chain;
    let mut scalar_input_mask = scalar_input_mask;
    let mut input_modulus = input_modulus;
    if resize_slot != 0 {
        inputs.swap(0, resize_slot);
        chain = remap_chain_input_slots(&chain, 0, resize_slot as u32);
        (scalar_input_mask, input_modulus) =
            swap_region_input_metadata(scalar_input_mask, input_modulus, 0, resize_slot);
    }
    inputs[0] = resize_input;
    rw.ensure_mapped(graph, &inputs);
    let fused = rw.add_fused(
        Op::ElementwiseRegion {
            chain,
            num_inputs,
            scalar_input_mask,
            input_modulus,
            prologue,
            prologue_input: 0,
        },
        &inputs,
        out_shape,
    );
    rw.replace(resize_id, fused);
    rw.replace(consumer_id, fused);
    fused_resize.insert(resize_id, ());
    true
}

impl Pass for FuseRegionPrologue {
    fn name(&self) -> &str {
        "fuse_region_prologue"
    }

    fn run(&self, graph: Graph) -> Graph {
        let consumers = consumer_counts(&graph);
        let mut rw = Rewriter::new(&graph.name);
        let mut fused_resize: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if fused_resize.contains_key(&node.id) {
                continue;
            }
            if matches!(node.op, Op::ResizeNearest2x)
                && is_prologue_candidate(&graph, node.id, &consumers)
            {
                continue;
            }
            if let Op::ElementwiseRegion {
                chain,
                num_inputs,
                scalar_input_mask,
                input_modulus,
                prologue,
                prologue_input: _,
            } = &node.op
            {
                if *prologue != RegionPrologue::None {
                    rw.copy_node(node);
                    continue;
                }
                if node.inputs.is_empty() {
                    rw.copy_node(node);
                    continue;
                }
                if let Some((resize_slot, resize_id)) =
                    node.inputs.iter().enumerate().find_map(|(i, &id)| {
                        if matches!(graph.node(id).op, Op::ResizeNearest2x)
                            && consumers.get(&id).copied() == Some(1)
                        {
                            Some((i, id))
                        } else {
                            None
                        }
                    })
                    && try_fuse_resize_into_region(
                        &mut rw,
                        &graph,
                        &consumers,
                        &mut fused_resize,
                        resize_id,
                        resize_slot,
                        chain.clone(),
                        *num_inputs,
                        *scalar_input_mask,
                        *input_modulus,
                        node.shape.clone(),
                        &node.inputs,
                        node.id,
                    )
                {
                    continue;
                }
                rw.copy_node(node);
                continue;
            }
            if node.inputs.len() == 1 {
                let resize_id = node.inputs[0];
                if let Some(step) = unary_region_step_from_op(&graph, node) {
                    if try_fuse_resize_into_region(
                        &mut rw,
                        &graph,
                        &consumers,
                        &mut fused_resize,
                        resize_id,
                        0,
                        vec![step],
                        1,
                        0,
                        [0; 16],
                        node.shape.clone(),
                        std::slice::from_ref(&resize_id),
                        node.id,
                    ) {
                        continue;
                    }
                }
            }
            if fused_resize.contains_key(&node.id) {
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

#[derive(Hash, PartialEq, Eq)]
struct RegionSignature {
    chain_len: usize,
    num_inputs: u32,
    scalar_input_mask: u32,
    prologue: RegionPrologue,
    modulus: [u32; 16],
}

fn region_signature(op: &Op) -> Option<RegionSignature> {
    let Op::ElementwiseRegion {
        chain,
        num_inputs,
        scalar_input_mask,
        input_modulus,
        prologue,
        prologue_input: _,
    } = op
    else {
        return None;
    };
    Some(RegionSignature {
        chain_len: chain.len(),
        num_inputs: *num_inputs,
        scalar_input_mask: *scalar_input_mask,
        prologue: *prologue,
        modulus: *input_modulus,
    })
}

fn narrow_batch_slice(graph: &Graph, input_id: NodeId) -> Option<(NodeId, usize)> {
    let node = graph.node(input_id);
    let Op::Narrow {
        axis,
        start,
        len: _,
    } = node.op
    else {
        return None;
    };
    if axis != 0 {
        return None;
    }
    if node.inputs.len() != 1 {
        return None;
    }
    Some((node.inputs[0], start))
}

fn shared_concat_consumer(graph: &Graph, region_ids: &[NodeId], axis: usize) -> Option<NodeId> {
    let mut consumer: Option<NodeId> = None;
    for &rid in region_ids {
        let users: Vec<_> = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&rid))
            .map(|n| n.id)
            .collect();
        if users.len() != 1 {
            return None;
        }
        match consumer {
            None => consumer = Some(users[0]),
            Some(c) if c == users[0] => {}
            _ => return None,
        }
    }
    let cid = consumer?;
    let concat = graph.node(cid);
    let Op::Concat { axis: a } = concat.op else {
        return None;
    };
    if a != axis {
        return None;
    }
    if concat.inputs.len() != region_ids.len() {
        return None;
    }
    if !region_ids.iter().all(|rid| concat.inputs.contains(rid)) {
        return None;
    }
    Some(cid)
}

/// Collapse `Narrow` ? unary (`relu`, same-dtype `cast`) batch slices into
/// [`Op::ElementwiseRegion`] so [`FuseBatchPreprocess`] can merge them.
pub struct MarkBatchSliceRegions;

impl Pass for MarkBatchSliceRegions {
    fn name(&self) -> &str {
        "mark_batch_slice_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        let consumers = consumer_counts(&graph);
        let mut rewrites: Vec<(NodeId, NodeId, Vec<ChainStep>)> = Vec::new();

        for node in graph.nodes() {
            let Op::Concat { axis } = node.op else {
                continue;
            };
            if axis != 0 || node.inputs.len() < 2 {
                continue;
            }
            let mut batch_parent: Option<NodeId> = None;
            let mut slice_plan: Vec<(NodeId, NodeId, ChainStep)> = Vec::new();
            let mut template: Option<ChainStep> = None;
            for &inp in &node.inputs {
                let unary = graph.node(inp);
                if matches!(unary.op, Op::ElementwiseRegion { .. }) {
                    slice_plan.clear();
                    break;
                }
                let step = match unary_region_step_from_op(&graph, unary) {
                    Some(s) => s,
                    None => {
                        slice_plan.clear();
                        break;
                    }
                };
                if consumers.get(&inp).copied() != Some(1) {
                    slice_plan.clear();
                    break;
                }
                if unary.inputs.len() != 1 {
                    slice_plan.clear();
                    break;
                }
                let narrow_id = unary.inputs[0];
                let Some((parent, _start)) = narrow_batch_slice(&graph, narrow_id) else {
                    slice_plan.clear();
                    break;
                };
                match batch_parent {
                    None => batch_parent = Some(parent),
                    Some(p) if p == parent => {}
                    _ => {
                        slice_plan.clear();
                        break;
                    }
                }
                match &template {
                    None => template = Some(step.clone()),
                    Some(t) if chain_step_same(t, &step) => {}
                    _ => {
                        slice_plan.clear();
                        break;
                    }
                }
                slice_plan.push((inp, narrow_id, step));
            }
            if slice_plan.len() < 2 {
                continue;
            }
            let chain = vec![template.unwrap()];
            for (unary_id, narrow_id, _) in slice_plan {
                rewrites.push((unary_id, narrow_id, chain.clone()));
            }
        }

        if rewrites.is_empty() {
            return graph;
        }

        let mut rw = Rewriter::new(&graph.name);
        let rewrite_set: HashMap<NodeId, ()> = rewrites.iter().map(|(u, _, _)| (*u, ())).collect();
        for node in graph.nodes() {
            if let Some((_, narrow_id, chain)) = rewrites.iter().find(|(u, _, _)| *u == node.id) {
                let narrow = graph.node(*narrow_id);
                rw.ensure_mapped(&graph, std::slice::from_ref(narrow_id));
                let mapped_narrow = rw.map(*narrow_id);
                let region = rw.add_fused(
                    Op::ElementwiseRegion {
                        chain: chain.clone(),
                        num_inputs: 1,
                        scalar_input_mask: 0,
                        input_modulus: [0; 16],
                        prologue: RegionPrologue::None,
                        prologue_input: 0,
                    },
                    std::slice::from_ref(&mapped_narrow),
                    narrow.shape.clone(),
                );
                rw.replace(node.id, region);
                continue;
            }
            if rewrite_set.contains_key(&node.id) {
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

fn chain_step_same(a: &ChainStep, b: &ChainStep) -> bool {
    match (a, b) {
        (ChainStep::Activation(x, _), ChainStep::Activation(y, _)) => x == y,
        (ChainStep::Cast(x, _), ChainStep::Cast(y, _)) => x == y,
        _ => false,
    }
}

/// Fuse identical element-wise regions on batch slices (FKL horizontal fusion).
pub struct FuseBatchPreprocess;

impl Pass for FuseBatchPreprocess {
    fn name(&self) -> &str {
        "fuse_batch_preprocess"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut groups: HashMap<RegionSignature, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            if let Some(sig) = region_signature(&node.op) {
                if node.inputs.len() != 1 {
                    continue;
                }
                groups.entry(sig).or_default().push(node.id);
            }
        }

        let mut batch_groups: Vec<(NodeId, Vec<NodeId>, NodeId)> = Vec::new();
        for (_sig, ids) in groups {
            if ids.len() < 2 {
                continue;
            }
            let mut by_parent: HashMap<NodeId, Vec<(NodeId, usize)>> = HashMap::new();
            for id in ids {
                let node = graph.node(id);
                let Some((parent, start)) = narrow_batch_slice(&graph, node.inputs[0]) else {
                    continue;
                };
                by_parent.entry(parent).or_default().push((id, start));
            }
            for (parent, mut entries) in by_parent {
                if entries.len() < 2 {
                    continue;
                }
                entries.sort_by_key(|(_, start)| *start);
                let region_ids: Vec<NodeId> = entries.iter().map(|(id, _)| *id).collect();
                let concat_id = shared_concat_consumer(&graph, &region_ids, 0);
                if let Some(concat) = concat_id {
                    batch_groups.push((parent, region_ids, concat));
                }
            }
        }

        if batch_groups.is_empty() {
            return graph;
        }

        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for (_, region_ids, concat_id) in &batch_groups {
            for &id in region_ids {
                fused_away.insert(id, ());
            }
            fused_away.insert(*concat_id, ());
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                if batch_groups.iter().any(|(_, ids, _)| ids[0] == node.id) {
                    let (parent, region_ids, concat_id) = batch_groups
                        .iter()
                        .find(|(_, ids, _)| ids[0] == node.id)
                        .unwrap();
                    let concat_node = graph.node(*concat_id);
                    let template = graph.node(region_ids[0]);
                    let Op::ElementwiseRegion {
                        chain,
                        num_inputs: _,
                        scalar_input_mask,
                        input_modulus,
                        prologue,
                        prologue_input: _,
                    } = &template.op
                    else {
                        unreachable!();
                    };
                    rw.ensure_mapped(&graph, std::slice::from_ref(parent));
                    let mut narrow_old: Vec<NodeId> = Vec::new();
                    for &rid in region_ids {
                        let narrow_id = graph.node(rid).inputs[0];
                        rw.ensure_mapped(&graph, std::slice::from_ref(&narrow_id));
                        narrow_old.push(narrow_id);
                    }
                    let fused = rw.add_fused(
                        Op::BatchElementwiseRegion {
                            chain: chain.clone(),
                            num_batch_inputs: narrow_old.len() as u32,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                            prologue: *prologue,
                            prologue_input: 0,
                        },
                        &narrow_old,
                        concat_node.shape.clone(),
                    );
                    for &rid in region_ids {
                        rw.replace(rid, fused);
                    }
                    rw.replace(*concat_id, fused);
                }
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

/// Lower FKL-style fused regions to primitive MIR (Rust-only / fallback path).
pub struct DecomposeFusionRegions;

impl Pass for DecomposeFusionRegions {
    fn name(&self) -> &str {
        "decompose_fusion_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        let any = graph.nodes().iter().any(|n| {
            matches!(
                n.op,
                Op::TransformRegion { .. } | Op::BatchElementwiseRegion { .. }
            )
        });
        if !any {
            return graph;
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            match &node.op {
                Op::TransformRegion { steps, .. } => {
                    rw.ensure_mapped(&graph, &node.inputs);
                    let mut cur_inputs: Vec<NodeId> =
                        node.inputs.iter().map(|id| rw.map(*id)).collect();
                    let mut cur_shape = graph.node(node.inputs[0]).shape.clone();
                    for TransformStep::ResizeNearest2x(_) in steps {
                        if cur_shape.rank() == 4 {
                            cur_shape = Shape::new(
                                &[
                                    cur_shape.dim(0).unwrap_static(),
                                    cur_shape.dim(1).unwrap_static(),
                                    cur_shape.dim(2).unwrap_static() * 2,
                                    cur_shape.dim(3).unwrap_static() * 2,
                                ],
                                cur_shape.dtype(),
                            );
                        }
                        let new_id = rw.new_graph.add_node(
                            Op::ResizeNearest2x,
                            cur_inputs.clone(),
                            cur_shape.clone(),
                        );
                        cur_inputs = vec![new_id];
                    }
                    let last = *cur_inputs.last().unwrap();
                    rw.replace(node.id, last);
                }
                Op::BatchElementwiseRegion {
                    chain,
                    num_batch_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input: _,
                } => {
                    rw.ensure_mapped(&graph, &node.inputs);
                    let mapped: Vec<NodeId> = node.inputs.iter().map(|id| rw.map(*id)).collect();
                    if mapped.len() != *num_batch_inputs as usize {
                        rw.copy_node(node);
                        continue;
                    }
                    let mut slice_ids = Vec::with_capacity(mapped.len());
                    for input_id in mapped {
                        let mut region_input = input_id;
                        if *prologue == RegionPrologue::ResizeNearest2x {
                            let in_shape = rw.new_graph.node(region_input).shape.clone();
                            let out_shape = resize_nearest_shape(&in_shape, &node.shape);
                            region_input = rw.new_graph.add_node(
                                Op::ResizeNearest2x,
                                vec![region_input],
                                out_shape,
                            );
                        }
                        let slice = rw.new_graph.add_node(
                            Op::ElementwiseRegion {
                                chain: chain.clone(),
                                num_inputs: 1,
                                scalar_input_mask: *scalar_input_mask,
                                input_modulus: *input_modulus,
                                prologue: RegionPrologue::None,
                                prologue_input: 0,
                            },
                            vec![region_input],
                            Shape::new(
                                &[
                                    1,
                                    node.shape.dim(1).unwrap_static(),
                                    node.shape.dim(2).unwrap_static(),
                                    node.shape.dim(3).unwrap_static(),
                                ],
                                node.shape.dtype(),
                            ),
                        );
                        slice_ids.push(slice);
                    }
                    if slice_ids.is_empty() {
                        rw.copy_node(node);
                        continue;
                    }
                    let concat = rw.new_graph.add_node(
                        Op::Concat { axis: 0 },
                        slice_ids,
                        node.shape.clone(),
                    );
                    rw.replace(node.id, concat);
                }
                Op::ElementwiseRegion {
                    prologue: RegionPrologue::ResizeNearest2x,
                    prologue_input: 0,
                    ..
                } => {
                    rw.copy_node(node);
                }
                _ => {
                    rw.copy_node(node);
                }
            }
        }
        rw.finish(&graph.outputs)
    }
}

fn resize_nearest_shape(in_shape: &Shape, fallback: &Shape) -> Shape {
    if in_shape.rank() == 4 {
        Shape::new(
            &[
                in_shape.dim(0).unwrap_static(),
                in_shape.dim(1).unwrap_static(),
                in_shape.dim(2).unwrap_static() * 2,
                in_shape.dim(3).unwrap_static() * 2,
            ],
            in_shape.dtype(),
        )
    } else {
        fallback.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    fn nchw(n: usize, c: usize, h: usize, w: usize) -> Shape {
        Shape::new(&[n, c, h, w], DType::F32)
    }

    #[test]
    fn fuse_region_prologue_on_resize_elementwise_chain() {
        use crate::fusion::MarkElementwiseRegions;
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let a = g.input("a", nchw(1, 3, 16, 16));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let r = g.add_node(
            Op::Activation(Activation::Relu),
            vec![up],
            nchw(1, 3, 16, 16),
        );
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![r, a], nchw(1, 3, 16, 16));
        let out = g.add_node(Op::Binary(BinaryOp::Mul), vec![s, a], nchw(1, 3, 16, 16));
        g.set_outputs(vec![out]);
        let g = MarkElementwiseRegions.run(g);
        let g = FuseRegionPrologue.run(g);
        assert!(
            !g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
        let region = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .expect("region");
        assert_eq!(region.inputs.len(), 2);
        assert_eq!(g.node(region.inputs[0]).op, Op::Input { name: "x".into() });
        assert_eq!(g.node(region.inputs[1]).op, Op::Input { name: "a".into() });
        if let Op::ElementwiseRegion {
            prologue,
            num_inputs,
            ..
        } = &region.op
        {
            assert_eq!(*prologue, RegionPrologue::ResizeNearest2x);
            assert_eq!(*num_inputs, 2);
        } else {
            panic!("expected elementwise region");
        }
    }

    #[test]
    fn fuse_region_prologue_when_resize_is_not_input_zero() {
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let a = g.input("a", nchw(1, 3, 16, 16));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let chain = vec![
            ChainStep::Activation(Activation::Relu, ChainOperand::Input(1)),
            ChainStep::Binary(BinaryOp::Add, ChainOperand::Input(0), ChainOperand::Step(0)),
        ];
        let region = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 2,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![a, up],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![region]);

        let out = FuseRegionPrologue.run(g);
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
        let region = out
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .expect("region");
        assert_eq!(
            out.node(region.inputs[0]).op,
            Op::Input { name: "x".into() }
        );
        assert_eq!(
            out.node(region.inputs[1]).op,
            Op::Input { name: "a".into() }
        );
        if let Op::ElementwiseRegion { prologue, .. } = &region.op {
            assert_eq!(*prologue, RegionPrologue::ResizeNearest2x);
        } else {
            panic!("expected elementwise region");
        }
    }

    #[test]
    fn fuse_region_prologue_via_pipeline_on_resize_relu() {
        use crate::fusion::MarkElementwiseRegions;
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let out = g.add_node(
            Op::Activation(Activation::Relu),
            vec![up],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![out]);
        let g = MarkElementwiseRegions.run(g);
        let g = FuseRegionPrologue.run(g);
        assert!(
            !g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
        assert!(g.nodes().iter().any(|n| matches!(
            n.op,
            Op::ElementwiseRegion {
                prologue: RegionPrologue::ResizeNearest2x,
                ..
            }
        )));
    }

    #[test]
    fn fuse_region_prologue_from_resize_relu_ops() {
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let out = g.add_node(
            Op::Activation(Activation::Relu),
            vec![up],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![out]);

        let out = FuseRegionPrologue.run(g);
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
        let regions: Vec<_> = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1);
        if let Op::ElementwiseRegion {
            prologue, chain, ..
        } = &regions[0].op
        {
            assert_eq!(*prologue, RegionPrologue::ResizeNearest2x);
            assert_eq!(chain.len(), 1);
        } else {
            panic!("expected elementwise region");
        }
        assert_eq!(
            out.node(regions[0].inputs[0]).op,
            Op::Input { name: "x".into() }
        );
    }

    #[test]
    fn fuse_region_prologue_merges_resize_and_region() {
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let chain = vec![ChainStep::Activation(
            Activation::Relu,
            ChainOperand::Input(0),
        )];
        let region = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![up],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![region]);

        let out = FuseRegionPrologue.run(g);
        let regions: Vec<_> = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1);
        if let Op::ElementwiseRegion { prologue, .. } = &regions[0].op {
            assert_eq!(*prologue, RegionPrologue::ResizeNearest2x);
        } else {
            panic!("expected elementwise region");
        }
        assert_eq!(
            out.node(regions[0].inputs[0]).op,
            Op::Input { name: "x".into() }
        );
    }

    #[test]
    fn decompose_prologue_region_kept_native() {
        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let chain = vec![ChainStep::Activation(
            Activation::Relu,
            ChainOperand::Input(0),
        )];
        let region = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::ResizeNearest2x,
                prologue_input: 0,
            },
            vec![x],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![region]);

        let out = DecomposeFusionRegions.run(g);
        let regions: Vec<_> = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1);
        if let Op::ElementwiseRegion { prologue, .. } = &regions[0].op {
            assert_eq!(*prologue, RegionPrologue::ResizeNearest2x);
        } else {
            panic!("expected elementwise region");
        }
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x))
        );
    }

    #[test]
    fn fuse_batch_preprocess_groups_narrow_slices() {
        let mut g = Graph::new("t");
        let batch = g.input("batch", nchw(4, 3, 8, 8));
        let n0 = g.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: 1,
            },
            vec![batch],
            nchw(1, 3, 8, 8),
        );
        let n1 = g.add_node(
            Op::Narrow {
                axis: 0,
                start: 1,
                len: 1,
            },
            vec![batch],
            nchw(1, 3, 8, 8),
        );
        let chain = vec![ChainStep::Activation(
            Activation::Relu,
            ChainOperand::Input(0),
        )];
        let r0 = g.add_node(
            Op::ElementwiseRegion {
                chain: chain.clone(),
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![n0],
            nchw(1, 3, 8, 8),
        );
        let r1 = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
                prologue: RegionPrologue::None,
                prologue_input: 0,
            },
            vec![n1],
            nchw(1, 3, 8, 8),
        );
        let cat = g.add_node(Op::Concat { axis: 0 }, vec![r0, r1], nchw(2, 3, 8, 8));
        g.set_outputs(vec![cat]);

        let out = FuseBatchPreprocess.run(g);
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Concat { .. }))
        );
    }

    #[test]
    fn mark_batch_slice_regions_from_primitive_relu() {
        let g = crate::fk_graphs::batch_narrow_relu_primitive_graph("t", 2, 3, 8, 8);
        let out =
            crate::pass::run_passes(g, &[&MarkBatchSliceRegions, &FuseBatchPreprocess], false);
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Concat { .. }))
        );
    }

    #[test]
    fn fuse_batch_preprocess_four_slices() {
        let mut g = Graph::new("t");
        let batch = g.input("batch", nchw(4, 3, 8, 8));
        let chain = vec![ChainStep::Activation(
            Activation::Relu,
            ChainOperand::Input(0),
        )];
        let mut slices = Vec::new();
        for i in 0..4 {
            let sl = g.add_node(
                Op::Narrow {
                    axis: 0,
                    start: i,
                    len: 1,
                },
                vec![batch],
                nchw(1, 3, 8, 8),
            );
            slices.push(g.add_node(
                Op::ElementwiseRegion {
                    chain: chain.clone(),
                    num_inputs: 1,
                    scalar_input_mask: 0,
                    input_modulus: [0; 16],
                    prologue: RegionPrologue::None,
                    prologue_input: 0,
                },
                vec![sl],
                nchw(1, 3, 8, 8),
            ));
        }
        let cat = g.add_node(Op::Concat { axis: 0 }, slices, nchw(4, 3, 8, 8));
        g.set_outputs(vec![cat]);

        let out = FuseBatchPreprocess.run(g);
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::BatchElementwiseRegion { .. }))
        );
    }

    #[test]
    fn gpu_unfuse_preserves_prologue_region() {
        use crate::fusion::{MarkElementwiseRegions, UnfuseElementwiseRegions};
        use crate::pass::run_passes;

        let mut g = Graph::new("t");
        let x = g.input("x", nchw(1, 3, 8, 8));
        let up = g.add_node(Op::ResizeNearest2x, vec![x], nchw(1, 3, 16, 16));
        let r = g.add_node(
            Op::Activation(Activation::Relu),
            vec![up],
            nchw(1, 3, 16, 16),
        );
        g.set_outputs(vec![r]);

        let passes: Vec<&dyn crate::pass::Pass> = vec![
            &MarkElementwiseRegions,
            &FuseRegionPrologue,
            &UnfuseElementwiseRegions::FOR_GPU,
        ];
        let out = run_passes(g, &passes, false);
        assert!(
            out.nodes().iter().any(|n| {
                matches!(
                    n.op,
                    Op::ElementwiseRegion {
                        prologue: RegionPrologue::ResizeNearest2x,
                        prologue_input: 0,
                        ..
                    }
                )
            }),
            "GPU unfuse should keep FKL prologue regions"
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ResizeNearest2x)),
            "resize should be folded into prologue"
        );
    }
}
