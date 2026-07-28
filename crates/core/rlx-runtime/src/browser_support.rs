// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Browser graph legality checks — reject transport/host-only paths before
//! compile/run on wasm.

use rlx_driver::Device;
use rlx_ir::{Graph, Op, OpKind};

/// `collective.*` custom op names blocked in the browser (no TCP transport).
pub const COLLECTIVE_OPS: &[&str] = &[
    "collective.all_reduce",
    "collective.all_gather",
    "collective.reduce_scatter",
    "collective.copy_to_parallel",
    "collective.reduce_from_parallel",
    "collective.broadcast",
    "collective.reduce",
    "collective.all_to_all",
    "collective.moe_dispatch",
    "collective.moe_combine",
    "collective.ppermute",
    "collective.send",
];

/// Returns a human-readable reason when `graph` contains browser-forbidden ops.
pub fn collective_block_reason(graph: &Graph) -> Option<String> {
    for node in graph.nodes() {
        if let Op::Custom { name, .. } = &node.op {
            if COLLECTIVE_OPS.contains(&name.as_str()) {
                return Some(format!(
                    "collective op '{name}' unavailable in browser: no TCP transport on wasm32 \
                     (rlx-driver ProcessGroup uses std::net sockets)"
                ));
            }
        }
    }
    None
}

/// True when the graph has no browser-blocked transport/custom ops.
pub fn passes_browser_preflight(graph: &Graph) -> bool {
    collective_block_reason(graph).is_none()
}

/// Pick the highest-priority browser device that can legalize `graph`.
pub fn select_browser_device_for_graph(graph: &Graph) -> Option<Device> {
    for &device in crate::device_ext::BROWSER_DEVICE_PRIORITY {
        if !crate::device_ext::is_available(device) {
            continue;
        }
        if !passes_browser_preflight(graph) {
            return None;
        }
        if crate::device_ext::supports_graph(device, graph) {
            return Some(device);
        }
    }
    None
}

/// First unsupported op kind for a browser device (for diagnostics).
pub fn first_browser_unsupported_op(graph: &Graph, device: Device) -> Option<(usize, OpKind)> {
    if collective_block_reason(graph).is_some() {
        return graph.nodes().iter().find_map(|n| {
            if let Op::Custom { name, .. } = &n.op {
                if COLLECTIVE_OPS.contains(&name.as_str()) {
                    return Some((n.id.0 as usize, OpKind::Custom));
                }
            }
            None
        });
    }
    crate::device_ext::first_unsupported_op(device, graph).map(|(i, op)| (i, op.kind()))
}
