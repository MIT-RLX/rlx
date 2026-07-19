// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Heterogeneous **split** execution — run one graph across several devices.
//!
//! Unlike [`crate::graph_devices::GraphDevices`] (compile the *whole* graph to
//! each device and pick one), this partitions a single graph into per-device
//! **segments** and runs them as a pipeline, materializing the tensors that
//! cross a device boundary on the host between segments. Two uses:
//!
//!   - **Op fallback:** [`DeviceMap::auto_fallback`] places every op the primary
//!     backend can't lower (per [`crate::backend::Backend::supported_ops`]) on a
//!     fallback device (usually CPU), so a graph runs on a GPU that only covers
//!     part of it.
//!   - **Explicit placement:** [`DeviceMap::from_fn`] assigns each node a device
//!     (e.g. compute-heavy convs on the GPU, a cheap head on the CPU).
//!
//! Segments are maximal topological runs of same-device compute nodes, so a
//! graph that's mostly one device produces few segments (few host round-trips).
//! A single-device map yields one segment — identical to a plain
//! [`CompiledGraph`], just via the value-map indirection.

use std::collections::HashMap;

use rlx_driver::Device;
use rlx_ir::{Graph, Node, NodeId, Op, OpKind};

use crate::CompileOptions;
use crate::compiled::CompiledGraph;
use crate::registry::backend_for;
use crate::session::Session;

/// Per-node device assignment for a graph. Indexed by `NodeId`. Source nodes
/// (`Op::Input` / `Op::Param`) are external, so their entry is unused.
#[derive(Debug, Clone)]
pub struct DeviceMap {
    devices: Vec<Device>,
}

fn is_source(op: &Op) -> bool {
    matches!(op, Op::Input { .. } | Op::Param { .. })
}

impl DeviceMap {
    /// Every node on `device` (one segment — the degenerate case).
    pub fn single(graph: &Graph, device: Device) -> Self {
        Self {
            devices: vec![device; graph.len()],
        }
    }

    /// Assign each node a device from `f`.
    pub fn from_fn(graph: &Graph, f: impl Fn(&Node) -> Device) -> Self {
        Self {
            devices: graph.nodes().iter().map(&f).collect(),
        }
    }

    /// Run each op on `primary`, except ops `primary`'s backend can't lower —
    /// those go to `fallback`. Uses [`Backend::supported_ops`]; an empty list
    /// means "accepts everything" (no split). `primary`'s backend must be
    /// registered (its feature enabled), else everything falls back.
    ///
    /// [`Backend::supported_ops`]: crate::backend::Backend::supported_ops
    pub fn auto_fallback(graph: &Graph, primary: Device, fallback: Device) -> Self {
        let supported: Vec<OpKind> = backend_for(primary)
            .map(|b| b.supported_ops().to_vec())
            .unwrap_or_default();
        let accepts = |op: &Op| supported.is_empty() || supported.contains(&op.kind());
        let devices = graph
            .nodes()
            .iter()
            .map(|n| {
                if is_source(&n.op) || accepts(&n.op) {
                    primary
                } else {
                    fallback
                }
            })
            .collect();
        Self { devices }
    }

    #[inline]
    fn get(&self, id: NodeId) -> Device {
        self.devices[id.0 as usize]
    }
}

/// One compiled per-device segment plus the boundary tensors it consumes /
/// produces (named by their original `NodeId`).
struct Segment {
    device: Device,
    compiled: CompiledGraph,
    /// Original NodeIds this segment needs as inputs (boundary values).
    input_ids: Vec<NodeId>,
    /// The subgraph input name for each `input_ids` entry (`"v<id>"`).
    input_names: Vec<String>,
    /// Original NodeIds this segment produces for later segments / outputs,
    /// in the segment subgraph's output order.
    output_ids: Vec<NodeId>,
}

/// A graph compiled across multiple devices. Feed inputs / params by name and
/// [`run`](Self::run); intermediates that cross a device boundary round-trip
/// through host memory.
pub struct HeteroExecutable {
    segments: Vec<Segment>,
    input_name_to_id: HashMap<String, NodeId>,
    param_name_to_id: HashMap<String, NodeId>,
    graph_outputs: Vec<NodeId>,
    params: HashMap<NodeId, Vec<f32>>,
}

impl HeteroExecutable {
    /// Partition `graph` per `map` and compile each segment to its device.
    pub fn compile(graph: &Graph, map: &DeviceMap, options: &CompileOptions) -> Self {
        // 1. Assign each compute node to a segment: a new segment starts
        //    whenever the device changes from the previous compute node (nodes
        //    are in topological order, so a segment only depends on earlier
        //    segments / source nodes).
        let n = graph.len();
        let mut seg_of: Vec<Option<usize>> = vec![None; n];
        let mut seg_device: Vec<Device> = Vec::new();
        let mut cur_dev: Option<Device> = None;
        for node in graph.nodes() {
            if is_source(&node.op) {
                continue;
            }
            let d = map.get(node.id);
            if cur_dev != Some(d) {
                seg_device.push(d);
                cur_dev = Some(d);
            }
            seg_of[node.id.0 as usize] = Some(seg_device.len() - 1);
        }

        // 2. Mark segment outputs: a compute node is an output of its segment
        //    if it's a graph output or consumed by a *different* segment.
        let mut is_seg_output: Vec<bool> = vec![false; n];
        for node in graph.nodes() {
            let Some(ns) = seg_of[node.id.0 as usize] else {
                continue;
            };
            for &inp in &node.inputs {
                if let Some(ps) = seg_of[inp.0 as usize]
                    && ps != ns
                {
                    is_seg_output[inp.0 as usize] = true;
                }
            }
        }
        for &out in &graph.outputs {
            is_seg_output[out.0 as usize] = true;
        }

        // 3. Build + compile each segment's subgraph.
        let mut segments = Vec::with_capacity(seg_device.len());
        for (s, &device) in seg_device.iter().enumerate() {
            let mut sub = Graph::new(format!("{}#seg{s}", graph.name));
            let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
            let mut input_ids: Vec<NodeId> = Vec::new();
            let mut input_names: Vec<String> = Vec::new();

            for node in graph.nodes() {
                if seg_of[node.id.0 as usize] != Some(s) {
                    continue;
                }
                let mut new_inputs = Vec::with_capacity(node.inputs.len());
                for &inp in &node.inputs {
                    let nid = if let Some(&existing) = remap.get(&inp) {
                        existing
                    } else {
                        // Boundary value produced outside this segment → a
                        // subgraph Input fed by the executor.
                        let name = format!("v{}", inp.0);
                        let id = sub.append_node(
                            Op::Input { name: name.clone() },
                            Vec::new(),
                            graph.shape(inp).clone(),
                            Some(name.clone()),
                        );
                        remap.insert(inp, id);
                        input_ids.push(inp);
                        input_names.push(name);
                        id
                    };
                    new_inputs.push(nid);
                }
                let new_id = sub.append_node(
                    node.op.clone(),
                    new_inputs,
                    node.shape.clone(),
                    node.name.clone(),
                );
                remap.insert(node.id, new_id);
            }

            // Outputs in stable node order.
            let mut output_ids: Vec<NodeId> = Vec::new();
            let mut sub_outputs: Vec<NodeId> = Vec::new();
            for node in graph.nodes() {
                if seg_of[node.id.0 as usize] == Some(s) && is_seg_output[node.id.0 as usize] {
                    output_ids.push(node.id);
                    sub_outputs.push(remap[&node.id]);
                }
            }
            sub.set_outputs(sub_outputs);

            let compiled = Session::new(device).compile_with(sub, options);
            segments.push(Segment {
                device,
                compiled,
                input_ids,
                input_names,
                output_ids,
            });
        }

        // 4. Name → id maps for feeding inputs / params, and the final outputs.
        let mut input_name_to_id = HashMap::new();
        let mut param_name_to_id = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_name_to_id.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_name_to_id.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        Self {
            segments,
            input_name_to_id,
            param_name_to_id,
            graph_outputs: graph.outputs.clone(),
            params: HashMap::new(),
        }
    }

    /// [`compile`](Self::compile) with the [`DeviceMap::auto_fallback`] policy —
    /// run `graph` on `primary`, spilling unsupported ops to `fallback`.
    pub fn with_fallback(
        graph: &Graph,
        primary: Device,
        fallback: Device,
        options: &CompileOptions,
    ) -> Self {
        let map = DeviceMap::auto_fallback(graph, primary, fallback);
        Self::compile(graph, &map, options)
    }

    /// Set a parameter value (routed to whichever segment(s) consume it).
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_name_to_id.get(name) {
            self.params.insert(id, data.to_vec());
        }
    }

    /// Run the pipeline: seed inputs + params, execute each segment on its
    /// device (transferring boundary tensors through host memory), return the
    /// graph outputs in declaration order.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut values: HashMap<NodeId, Vec<f32>> = HashMap::new();
        for (name, data) in inputs {
            if let Some(&id) = self.input_name_to_id.get(*name) {
                values.insert(id, data.to_vec());
            }
        }
        for (id, data) in &self.params {
            values.insert(*id, data.clone());
        }

        for seg in &mut self.segments {
            let run_inputs: Vec<(&str, &[f32])> = seg
                .input_ids
                .iter()
                .zip(&seg.input_names)
                .map(|(id, name)| {
                    (
                        name.as_str(),
                        values.get(id).map(Vec::as_slice).unwrap_or(&[]),
                    )
                })
                .collect();
            let outs = seg.compiled.run(&run_inputs);
            for (i, &oid) in seg.output_ids.iter().enumerate() {
                values.insert(oid, outs.get(i).cloned().unwrap_or_default());
            }
        }

        self.graph_outputs
            .iter()
            .map(|id| values.get(id).cloned().unwrap_or_default())
            .collect()
    }

    /// Number of device segments the graph was split into.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// The device of each segment, in execution order.
    pub fn segment_devices(&self) -> Vec<Device> {
        self.segments.iter().map(|s| s.device).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Shape};

    /// Register the CPU backend under a second device so a "heterogeneous"
    /// split can be exercised (and its result checked) on one machine.
    fn alias_cpu_to(device: Device) {
        crate::registry::register_backend(device, || {
            backend_for(Device::Cpu).expect("cpu backend (enable feature `cpu`)")
        });
    }

    // y = relu(x·w + b), then y·2. A few compute nodes to split.
    fn build() -> (Graph, NodeId, NodeId, NodeId) {
        let f = DType::F32;
        let mut g = Graph::new("hetero_test");
        let x = g.input("x", Shape::new(&[2, 3], f));
        let w = g.param("w", Shape::new(&[3, 4], f));
        let b = g.param("b", Shape::new(&[2, 4], f));
        let mm = g.matmul(x, w, Shape::new(&[2, 4], f));
        let z = g.add(mm, b);
        let two = g.add(z, z); // stand-in for a second stage
        g.set_outputs(vec![two]);
        (g, w, b, two)
    }

    fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = (0..6).map(|i| i as f32 * 0.1).collect();
        let w = (0..12).map(|i| (i as f32 * 0.05) - 0.3).collect();
        let b = vec![0.1; 8];
        (x, w, b)
    }

    #[test]
    fn single_segment_matches_plain_compile() {
        let (g, _w, _b, _) = build();
        let (x, w, b) = inputs();
        // Reference: plain CPU compile.
        let mut plain = Session::new(Device::Cpu).compile_with(g.clone(), &CompileOptions::new());
        plain.set_param("w", &w);
        plain.set_param("b", &b);
        let want = plain.run(&[("x", &x)]);

        // Hetero, all CPU → one segment.
        let map = DeviceMap::single(&g, Device::Cpu);
        let mut hx = HeteroExecutable::compile(&g, &map, &CompileOptions::new());
        assert_eq!(hx.num_segments(), 1);
        hx.set_param("w", &w);
        hx.set_param("b", &b);
        let got = hx.run(&[("x", &x)]);
        assert_eq!(got, want);
    }

    #[test]
    fn multi_device_split_matches_single() {
        // Second CPU-backed device so the split really compiles + runs two
        // segments, transferring the boundary tensor through host memory.
        alias_cpu_to(Device::Vulkan);
        let (g, _w, _b, two) = build();
        let (x, w, b) = inputs();

        let mut plain = Session::new(Device::Cpu).compile_with(g.clone(), &CompileOptions::new());
        plain.set_param("w", &w);
        plain.set_param("b", &b);
        let want = plain.run(&[("x", &x)]);

        // Put the last node (`two`) on the aliased device, the rest on CPU →
        // two segments with a CPU→Vulkan boundary.
        let map = DeviceMap::from_fn(&g, |node| {
            if node.id == two {
                Device::Vulkan
            } else {
                Device::Cpu
            }
        });
        let mut hx = HeteroExecutable::compile(&g, &map, &CompileOptions::new());
        assert_eq!(hx.num_segments(), 2, "expected a CPU + aliased split");
        assert_eq!(hx.segment_devices(), vec![Device::Cpu, Device::Vulkan]);
        hx.set_param("w", &w);
        hx.set_param("b", &b);
        let got = hx.run(&[("x", &x)]);
        for (a, e) in got[0].iter().zip(&want[0]) {
            assert!((a - e).abs() < 1e-5, "split mismatch: {a} vs {e}");
        }
    }

    // A genuine CPU + GPU split: the matmul+add prefix runs on the real CUDA
    // device, the final node on CPU — the boundary tensor crosses GPU→host→CPU.
    // Only built/run with `--features cuda` (i.e. on a CUDA machine).
    #[cfg(feature = "cuda")]
    #[test]
    fn real_cpu_cuda_split_matches_single() {
        // The `cuda` feature can be compiled (cudarc dynamic-loading) on a host
        // with no CUDA device (e.g. macOS CI); skip rather than panic there.
        if !crate::device_ext::is_available(Device::Cuda) {
            eprintln!("skipping real_cpu_cuda_split_matches_single: no CUDA device");
            return;
        }
        let (g, _w, _b, two) = build();
        let (x, w, b) = inputs();

        let mut plain = Session::new(Device::Cpu).compile_with(g.clone(), &CompileOptions::new());
        plain.set_param("w", &w);
        plain.set_param("b", &b);
        let want = plain.run(&[("x", &x)]);

        let map = DeviceMap::from_fn(&g, |node| {
            if node.id == two {
                Device::Cpu
            } else {
                Device::Cuda
            }
        });
        let mut hx = HeteroExecutable::compile(&g, &map, &CompileOptions::new());
        assert_eq!(hx.num_segments(), 2);
        assert_eq!(hx.segment_devices(), vec![Device::Cuda, Device::Cpu]);
        hx.set_param("w", &w);
        hx.set_param("b", &b);
        let got = hx.run(&[("x", &x)]);
        for (a, e) in got[0].iter().zip(&want[0]) {
            assert!((a - e).abs() < 1e-4, "cpu+cuda split mismatch: {a} vs {e}");
        }
    }

    #[test]
    fn auto_fallback_all_cpu_is_single_segment() {
        // CPU claims to support everything (empty supported_ops) → no split.
        let (g, _, _, _) = build();
        let map = DeviceMap::auto_fallback(&g, Device::Cpu, Device::Cpu);
        let hx = HeteroExecutable::compile(&g, &map, &CompileOptions::new());
        assert_eq!(hx.num_segments(), 1);
    }
}
