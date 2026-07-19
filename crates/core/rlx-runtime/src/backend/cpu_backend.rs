use super::*;
use rlx_cpu::{arena::Arena, thunk};
use rlx_ir::{DType, NodeId, Op};
use rlx_opt::memory::{self, MemoryPlan};
// Arena typed read/write helpers live in `crate::arena` so every
// backend (CPU, Metal, future CUDA/wgpu/WASM) shares one implementation.
use rlx_driver::arena::{read_typed_to_f32, write_typed_from_f32};

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_cpu::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        static ONNX_KERNELS: std::sync::Once = std::sync::Once::new();
        ONNX_KERNELS.call_once(rlx_cpu::onnx_ref::register_onnx_reference_kernels);
        // Lower Op::If / Op::While to primitives BEFORE legalize
        // so the supported-op check doesn't reject them — the CPU
        // backend has no native sub-graph executor; this rewrite
        // makes If/While invisible to the rest of the pipeline.
        // No-op when neither op is in the graph.
        let graph = rlx_opt::LowerControlFlow.run(graph);
        // Lower f32 SPD-manifold ops (ReEig/LogEig/BiMap/SpdBatchNorm) to the
        // graph-primitive Jacobi eigensolver BEFORE legalize — like If/While,
        // the CPU backend has no f32 SPD kernel (the native ones are f64
        // LAPACK). No-op unless an f32 SPD op is present; f64 SPD nodes are
        // left untouched for the native LAPACK path.
        let graph = rlx_opt::LowerSpectral.run(graph);
        // PLAN L4: legalize against the backend's claimed op set
        // BEFORE running fusion (so the diagnostic points at the
        // user's IR, not at a fused-away node).
        if let Err(errors) = rlx_opt::legalize_for_backend(&graph, rlx_cpu::SUPPORTED_OPS) {
            panic!("{}", rlx_opt::format_legalize_error("cpu", &errors));
        }
        let policy = options.policy.clone();
        let _precision = options.precision;
        let cfg = rlx_cpu::config::RuntimeConfig::global();

        let graph = crate::precompile::precompile_cleanup(graph, options);

        // Run fusion pipeline (HIR/MIR/LIR ideology — fusion is first-class).
        let mut compile_opts = options.clone();
        compile_opts.arena_alignment = cfg.arena_alignment;
        let compile_result = crate::stages::compile_graph_stages_for_backend(
            rlx_driver::Device::Cpu,
            graph,
            &compile_opts,
            rlx_cpu::SUPPORTED_OPS,
        );
        crate::stages::maybe_log_fusion(&compile_result.fusion);
        let fused = compile_result.lir.into_graph();

        // Apply precision policy AFTER fusion — Cast nodes don't disrupt
        // the now-flattened fused ops.
        let fused = match policy {
            Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(fused),
            None => fused,
        };

        let io_manifest = cpu_low_precision::IoDtypeManifest::from_graph(&fused);
        let exec_graph = if cpu_low_precision::needs_f32_exec(&fused) {
            cpu_low_precision::promote_to_f32(fused)
        } else {
            fused
        };

        // Re-plan after precision rewrites (may change dtypes / sizes).
        let plan = memory::plan_memory_aligned(&exec_graph, cfg.arena_alignment);
        if cfg.verbose >= 1 {
            eprintln!(
                "[rlx] arena: {} bytes, {} buffers, alignment: {}",
                plan.arena_size,
                plan.assignments.len(),
                cfg.arena_alignment
            );
        }
        Box::new(build_cpu_executable(
            exec_graph,
            plan,
            io_manifest,
            options.rng,
        ))
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // `Instant` is unimplemented on wasm32 — never touch it there.
        #[cfg(not(target_arch = "wasm32"))]
        let prof = std::env::var_os("RLX_PROFILE_COMPILE").is_some();
        #[cfg(not(target_arch = "wasm32"))]
        let tt = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let alignment = lir.buffers.alignment.max(options.arena_alignment);
        let mut graph = lir.into_graph();
        {
            use rlx_opt::pass::Pass as _;
            graph = rlx_opt::LegalizeBroadcast.run(graph);
        }
        if let Some(p) = options.policy.clone() {
            use rlx_opt::pass::Pass;
            graph = rlx_opt::AutoMixedPrecision::new(p).run(graph);
        }
        let io_manifest = cpu_low_precision::IoDtypeManifest::from_graph(&graph);
        let promote = cpu_low_precision::needs_f32_exec(&graph);
        let exec_graph = if promote {
            cpu_low_precision::promote_to_f32(graph)
        } else {
            graph
        };
        #[cfg(not(target_arch = "wasm32"))]
        let t_prep = tt.map(|t| t.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        let t1 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // LegalizeBroadcast may insert Expand nodes — must replan; the
        // embedded LIR buffer map is from before legalization.
        let plan = memory::plan_memory_aligned(&exec_graph, alignment);
        #[cfg(not(target_arch = "wasm32"))]
        let t_plan = t1.map(|t| t.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        if prof {
            eprintln!(
                "[compile_lir] {} nodes: into_graph+passes={:?} plan_memory={:?}",
                exec_graph.nodes().len(),
                t_prep.unwrap(),
                t_plan.unwrap(),
            );
        }
        let cfg = rlx_cpu::config::RuntimeConfig::global();
        if cfg.verbose >= 1 {
            eprintln!(
                "[rlx] compile_lir: arena {} bytes ({} buffers, alignment {})",
                plan.arena_size,
                plan.assignments.len(),
                alignment,
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        let t2 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let exe = build_cpu_executable(exec_graph, plan, io_manifest, options.rng);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(t2) = t2 {
            eprintln!("[compile_lir] build_thunks={:?}", t2.elapsed());
        }
        Box::new(exe)
    }
}

fn build_cpu_executable(
    graph: Graph,
    plan: MemoryPlan,
    io_manifest: cpu_low_precision::IoDtypeManifest,
    rng: rlx_ir::RngOptions,
) -> CpuExecutable {
    let mut arena = Arena::from_plan(plan);
    let mut input_ids = HashMap::new();
    let mut param_ids = HashMap::new();
    let mut node_dtypes: HashMap<NodeId, DType> = HashMap::new();
    for node in graph.nodes() {
        node_dtypes.insert(node.id, node.shape.dtype());
        match &node.op {
            Op::Input { name } => {
                input_ids.insert(name.clone(), node.id);
            }
            Op::Param { name } => {
                param_ids.insert(name.clone(), node.id);
            }
            _ => {}
        }
    }

    let schedule = thunk::compile_thunks_with_rng(&graph, &arena, rng);

    let mut input_slots = Vec::new();
    for node in graph.nodes() {
        if let Op::Input { name } = &node.op {
            let off = arena.byte_offset(node.id);
            let len = node.shape.num_elements().unwrap_or(0);
            input_slots.push((name.clone(), off, len, node.shape.dtype()));
        }
    }

    let output_slots: Vec<(usize, usize)> = graph
        .outputs
        .iter()
        .map(|&id| {
            let off = arena.byte_offset(id);
            let len = graph.node(id).shape.num_elements().unwrap_or(0);
            (off, len)
        })
        .collect();

    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op
            && arena.has_buffer(node.id)
            && !data.is_empty()
        {
            match node.shape.dtype() {
                // True-width dtypes (their arena slot is sized to the real
                // element width, not f32): copy the raw bytes. I64/I32/U32
                // constants were previously caught by the f32-reinterpret
                // branch below, which read them in 4-byte chunks as f32 —
                // corrupting e.g. the VITS sequence-mask `arange` constant
                // (i64 [0..T-1]) so the downstream i64 `Compare` read garbage.
                // Bool/U8/I8 must also raw-copy: a 1-byte `ConstantOfShape(true)`
                // has `data.len()/4 == 0` in the f32 branch, so the slot stayed
                // zero and Soprano's causal `And(ones, mask)` collapsed → Softmax NaN.
                DType::F64
                | DType::F16
                | DType::BF16
                | DType::I64
                | DType::I32
                | DType::U32
                | DType::Bool
                | DType::U8
                | DType::I8 => {
                    let off = arena.byte_offset(node.id);
                    let buf = arena.raw_buf_mut();
                    let n = buf.len().saturating_sub(off).min(data.len());
                    buf[off..off + n].copy_from_slice(&data[..n]);
                }
                _ => {
                    let buf = arena.slice_mut(node.id);
                    let n_floats = data.len() / 4;
                    let n = buf.len().min(n_floats);
                    for i in 0..n {
                        let bytes = [
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        ];
                        buf[i] = f32::from_le_bytes(bytes);
                    }
                }
            }
        }
    }

    CpuExecutable {
        graph,
        arena,
        input_ids,
        param_ids,
        node_dtypes,
        io_manifest,
        schedule,
        input_slots,
        output_slots,
        handles: HashMap::new(),
        active_extent: None,
        moe_resident: None,
        moe_resident_layers: None,
        moe_topk_capture: None,
        baseline_written: false,
    }
}

#[derive(Clone)]
struct CpuExecutable {
    graph: Graph,
    arena: Arena,
    input_ids: HashMap<String, NodeId>,
    param_ids: HashMap<String, NodeId>,
    /// Per-node arena dtype. Lets set_param/run cast f32 ↔ F16/BF16
    /// when AutoMixedPrecision has rewritten the graph.
    node_dtypes: HashMap<NodeId, DType>,
    /// User-facing boundary dtypes (before f32 promotion for CPU exec).
    io_manifest: cpu_low_precision::IoDtypeManifest,
    schedule: thunk::ThunkSchedule,
    // Pre-resolved: ordered list of (input_name, arena_byte_offset, max_elems, dtype)
    input_slots: Vec<(String, usize, usize, DType)>,
    /// Output (byte_offset, num_elements). dtype is in node_dtypes.
    output_slots: Vec<(usize, usize)>,
    /// Persistent buffer handles (KV-cache, optimizer state, etc.).
    /// Lives outside the arena and survives across run() calls.
    /// On run(): if a handle's name matches a graph input, the
    /// handle's data is used as the input.
    handles: HashMap<String, Vec<f32>>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every thunk in the schedule is in
    /// `Thunk::safe_for_active_extent`, the executor processes only
    /// `actual / upper` of each kernel's work. Otherwise (or when
    /// `None`) runs at the full compiled extent. See PLAN L1.
    active_extent: Option<(usize, usize)>,
    moe_resident: Option<std::sync::Arc<[bool]>>,
    moe_resident_layers: Option<std::sync::Arc<Vec<std::sync::Arc<[bool]>>>>,
    moe_topk_capture: Option<std::sync::Arc<rlx_cpu::moe_topk_capture::MoeTopkCapture>>,
    /// Whether params + constants are already resident in the arena. While
    /// `true`, `restore_arena_baseline` zeros only the scratch buffers instead
    /// of re-zeroing + rewriting every param each run (which is O(params) and
    /// allocates a full params clone — catastrophic for multi-GB models).
    /// `set_param`/`set_param_typed` reset it to `false`.
    baseline_written: bool,
}

unsafe impl Send for CpuExecutable {}

impl CpuExecutable {
    /// Per-node dump (mirror of rlx-wgpu's `RLX_WGPU_DUMP_NODES`) for
    /// cross-backend divergence bisection: each F32 node's max|x| + nonzero
    /// count in topo order. Diff against the wgpu dump to find the first
    /// diverging node. `RLX_CPU_DUMP_FLAT=<i>` also prints that flat element.
    fn dump_nodes_if_requested(&self) {
        if !rlx_ir::env::flag("RLX_CPU_DUMP_NODES") {
            return;
        }
        let limit = rlx_ir::env::parse_or("RLX_CPU_DUMP_NODES_LIMIT", 2000usize);
        let flat_probe = rlx_ir::env::parse_or::<usize>("RLX_CPU_DUMP_FLAT", usize::MAX);
        eprintln!("[rlx-cpu-dump] per-node max |x| (topo order, limit={limit})");
        let buf = self.arena.raw_buf();
        let mut shown = 0usize;
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if !self.arena.has_buffer(node.id) {
                continue;
            }
            if matches!(
                node.op,
                rlx_ir::Op::Input { .. }
                    | rlx_ir::Op::Param { .. }
                    | rlx_ir::Op::Constant { .. }
                    | rlx_ir::Op::Reshape { .. }
                    | rlx_ir::Op::Cast { .. }
            ) {
                continue;
            }
            if self
                .node_dtypes
                .get(&node.id)
                .copied()
                .unwrap_or(DType::F32)
                != DType::F32
            {
                continue;
            }
            let off = self.arena.byte_offset(node.id);
            let n = node.shape.num_elements().unwrap_or(0);
            let data: &[f32] =
                unsafe { std::slice::from_raw_parts(buf.as_ptr().add(off) as *const f32, n) };
            let max = data.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let nz = data.iter().filter(|&&v| v != 0.0).count();
            let flat_s = if flat_probe < data.len() {
                format!(" flat[{flat_probe}]={:.6}", data[flat_probe])
            } else {
                String::new()
            };
            eprintln!(
                "  [{i:>3}] {:?} shape={:?} max={max:.6} nonzero={nz}/{}{flat_s}",
                node.op,
                node.shape.dims(),
                data.len()
            );
            shown += 1;
            if shown >= limit {
                break;
            }
        }
    }

    /// Write a f32 input slice into the arena, casting to the node's dtype.
    fn write_input(&mut self, id: NodeId, data: &[f32]) {
        let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
        let off = self.arena.byte_offset(id);
        let buf = self.arena.raw_buf_mut();
        let elem_size = dtype.size_bytes();
        let max_elems = (buf.len() - off) / elem_size;
        unsafe {
            write_typed_from_f32(buf.as_mut_ptr().add(off), dtype, data, max_elems);
        }
    }

    /// Read a node's arena bytes back as Vec<f32>, casting from its dtype.
    fn read_output(&self, id: NodeId) -> Vec<f32> {
        let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
        let off = self.arena.byte_offset(id);
        let buf = self.arena.raw_buf();
        let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
        unsafe { read_typed_to_f32(buf.as_ptr().add(off), dtype, n_elems) }
    }
}

impl ExecutableGraph for CpuExecutable {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            moe: true,
            typed_io: true,
            active_extent: true,
            ..crate::ExecutableCapabilities::NONE
        }
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(self.clone())
    }
    fn set_param(&mut self, name: &str, data: &[f32]) {
        // Params live solely in the arena (dedicated, never-aliased slots, see
        // the memory planner) — no redundant CPU-side copy is kept, which would
        // double the weight footprint for multi-GB models.
        // Cast f32 → arena dtype when the param has been rewritten to F16/BF16.
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
            let off = self.arena.byte_offset(id);
            let buf = self.arena.raw_buf_mut();
            let elem_size = dtype.size_bytes();
            let max_elems = (buf.len() - off) / elem_size;
            unsafe {
                write_typed_from_f32(buf.as_mut_ptr().add(off), dtype, data, max_elems);
            }
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.restore_arena_baseline();
        // 1. Apply persistent handles first — they act like default inputs.
        //    Explicit `inputs` passed to run() override matching handle names.
        let handle_names: Vec<String> = self.handles.keys().cloned().collect();
        for name in &handle_names {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                let data = self.handles.get(name).cloned().unwrap_or_default();
                self.write_input(id, &data);
            }
        }
        // 2. Explicit per-call inputs override handles.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.write_input(id, data);
            }
        }

        // Active-extent fast-path (PLAN L1): if hinted AND every thunk
        // in the schedule supports it, run scaled. Otherwise fall back
        // to full-extent dispatch — preserves correctness when the
        // schedule contains a thunk that hasn't yet been wired in.
        let active_used = if let Some((actual, upper)) = self.active_extent {
            thunk::execute_thunks_active(&self.schedule, self.arena.raw_buf_mut(), actual, upper)
        } else {
            false
        };
        if !active_used {
            // Execute via pre-compiled thunks (zero per-node dispatch overhead)
            thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
        }

        self.dump_nodes_if_requested();

        // 3. Sync any handle whose name matches a graph OUTPUT —
        //    KV-cache pattern: outputs flow back into the same-named
        //    handle for the next iteration.
        for (idx, &out_id) in self.graph.outputs.iter().enumerate() {
            let name = format!("out{idx}");
            if self.handles.contains_key(&name) {
                let v = self.read_output(out_id);
                self.handles.insert(name, v);
            }
        }

        self.graph
            .outputs
            .iter()
            .map(|&out_id| self.read_output(out_id))
            .collect()
    }

    fn run_raw(&mut self, inputs: &[(&str, &[f32])]) -> Vec<(*const f32, usize)> {
        self.restore_arena_baseline();
        // Copy inputs by name (HashMap lookup), casting to arena dtype.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.write_input(id, data);
            }
        }
        thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
        // Note: pointers are raw arena bytes — for F16 outputs, callers
        // must read 2 bytes/elem, not 4. run() is the safe path for
        // mixed precision; run_raw() is only meaningful for F32.
        self.graph
            .outputs
            .iter()
            .map(|&out_id| {
                let (ptr, len) = self.arena.raw_ptr(out_id);
                (ptr as *const f32, len)
            })
            .collect()
    }

    /// Fastest path: inputs by index (matching input_slots order), zero-copy output.
    /// No HashMap, no name matching, no Vec allocation. Casts f32 input
    /// to F16/BF16 if the input slot's dtype was rewritten.
    fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.restore_arena_baseline();
        let buf = self.arena.raw_buf_mut();
        for (i, &data) in inputs.iter().enumerate() {
            if i < self.input_slots.len() {
                let (_, off, max_len, dtype) = &self.input_slots[i];
                unsafe {
                    write_typed_from_f32(buf.as_mut_ptr().add(*off), *dtype, data, *max_len);
                }
            }
        }
        thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
        &self.output_slots
    }

    fn arena_ptr(&self) -> *const u8 {
        self.arena.raw_buf_mut_ptr()
    }

    fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
        // Persistent buffer: stored separately from arena, survives run().
        // If the name matches a graph input, run() will use this data
        // as the input. If the graph also writes back to this name (via
        // an output binding pattern), read_handle returns the latest.
        self.handles.insert(name.to_string(), data.to_vec());
        true
    }

    fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.handles.get(name).cloned()
    }

    fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.schedule.rng.write().unwrap() = rng;
    }

    fn rng(&self) -> rlx_ir::RngOptions {
        *self.schedule.rng.read().unwrap()
    }

    fn set_moe_resident_experts(&mut self, mask: &[bool]) {
        self.moe_resident_layers = None;
        self.schedule.moe_resident_layers = None;
        self.moe_resident = Some(Arc::from(mask));
        self.schedule.moe_resident = self.moe_resident.clone();
    }

    fn set_moe_resident_experts_per_layer(&mut self, masks: &[&[bool]]) {
        self.moe_resident = None;
        self.schedule.moe_resident = None;
        let layers: Vec<Arc<[bool]>> = masks.iter().map(|m| Arc::from(*m)).collect();
        let arc = Arc::new(layers);
        self.moe_resident_layers = Some(arc.clone());
        self.schedule.moe_resident_layers = Some(arc);
    }

    fn enable_moe_topk_capture(&mut self, num_experts: usize) -> bool {
        let cap = rlx_cpu::moe_topk_capture::MoeTopkCapture::new(num_experts);
        self.moe_topk_capture = Some(cap.clone());
        self.schedule.moe_topk_capture = Some(cap);
        true
    }

    fn take_moe_topk_capture(&mut self) -> Option<Vec<Vec<u32>>> {
        let cap = self.moe_topk_capture.as_ref()?;
        let layers = cap.take_layers();
        if layers.is_empty() {
            None
        } else {
            Some(layers)
        }
    }

    fn take_moe_residency_stats(&mut self) -> Option<crate::MoeResidencyStats> {
        rlx_cpu::moe_residency::take_last_forward_stats()
    }

    /// Typed param upload. F32 / F16 / BF16 go through the existing
    /// widen-to-f32 path (the CPU arena is historically f32 with
    /// optional half-precision rewrite). F64 (and any future
    /// non-widenable dtype) lands directly in the arena as bytes —
    /// the f32 path would lose precision.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if matches!(dtype, DType::F64 | DType::I64 | DType::I32 | DType::U32) {
            self.set_param_bytes(name, data, dtype);
            return;
        }
        // U8 / I8 raw byte tensors: opaque storage for the GGUF
        // K-quant `Op::DequantMatMul` path (weights stay packed
        // in the arena). One arena byte = one element.
        if matches!(dtype, DType::U8 | DType::I8) {
            self.set_param_bytes(name, data, dtype);
            return;
        }
        if dtype == DType::F32 {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            self.set_param(name, s);
        } else {
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.set_param(name, &f32_buf);
        }
    }

    /// Typed run with mixed-dtype inputs/outputs.
    ///
    /// For each input: if its declared graph dtype matches the
    /// caller's bytes, we write directly into the arena (zero
    /// precision loss — F64 stays F64). For F32 with a half-precision
    /// arena rewrite, we widen as before. F16/BF16 callers go
    /// through the existing widen path.
    ///
    /// Outputs are read straight from the arena in the graph node's
    /// declared dtype — F64 outputs come back as 8 bytes/element,
    /// F32 as 4, etc.
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        // Decide: are *all* inputs F64? If so, use the direct-byte
        // path for everything and skip the f32 widening machinery
        // entirely. Mixed dtype graphs (F32 + F64) take the
        // per-input dispatch route below.
        let all_f64 = !inputs.is_empty() && inputs.iter().all(|(_, _, dt)| *dt == DType::F64);

        if all_f64 {
            for (name, data, _) in inputs {
                if let Some(&id) = self.input_ids.get(*name) {
                    if !self.arena.has_buffer(id) {
                        continue;
                    }
                    let off = self.arena.byte_offset(id);
                    let buf = self.arena.raw_buf_mut();
                    let n = data.len();
                    debug_assert!(
                        off + n <= buf.len(),
                        "run_typed: input '{name}' overflows arena slot"
                    );
                    buf[off..off + n].copy_from_slice(data);
                }
            }
            thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
        } else {
            // Mixed-dtype path: dtypes that survive untouched
            // through the f32-aliased arena (F64, I32, I64, U32)
            // go in as bytes; F32 and the half-precision family
            // route through widen-to-f32 + run.
            let mut f32_owned: Vec<(String, Vec<f32>)> = Vec::new();
            for (name, data, dt) in inputs {
                let direct = matches!(
                    *dt,
                    DType::F64 | DType::I32 | DType::I64 | DType::U32 | DType::C64
                );
                if direct {
                    if let Some(&id) = self.input_ids.get(*name) {
                        if !self.arena.has_buffer(id) {
                            continue;
                        }
                        let off = self.arena.byte_offset(id);
                        let buf = self.arena.raw_buf_mut();
                        buf[off..off + data.len()].copy_from_slice(data);
                    }
                } else {
                    let v = super::widen_bytes_to_f32(data, *dt);
                    f32_owned.push((name.to_string(), v));
                }
            }
            for (name, data) in &f32_owned {
                if let Some(&id) = self.input_ids.get(name.as_str()) {
                    if self.arena.has_buffer(id) {
                        self.write_input(id, data);
                    }
                }
            }
            let active_used = if let Some((actual, upper)) = self.active_extent {
                thunk::execute_thunks_active(
                    &self.schedule,
                    self.arena.raw_buf_mut(),
                    actual,
                    upper,
                )
            } else {
                false
            };
            if !active_used {
                thunk::execute_thunks(&self.schedule, self.arena.raw_buf_mut());
            }
        }

        self.dump_nodes_if_requested();

        // Read outputs in declared boundary dtypes.
        self.graph
            .outputs
            .iter()
            .enumerate()
            .map(|(idx, &id)| {
                let exec_dtype = self.graph.node(id).shape.dtype();
                let declared = self.io_manifest.output_dtype(idx, exec_dtype);
                if matches!(
                    exec_dtype,
                    DType::F64
                        | DType::F16
                        | DType::BF16
                        | DType::I32
                        | DType::I64
                        | DType::U32
                        | DType::C64
                ) {
                    let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    let n_bytes = n_elems * exec_dtype.size_bytes();
                    let off = self.arena.byte_offset(id);
                    let bytes = self.arena.raw_buf()[off..off + n_bytes].to_vec();
                    return (bytes, declared);
                }
                let f32_vals = self.read_output(id);
                if declared != exec_dtype {
                    return (super::narrow_f32_to_bytes(&f32_vals, declared), declared);
                }
                let bytes = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, declared)
            })
            .collect()
    }
}

impl CpuExecutable {
    /// Clear ephemeral (scratch) arena slots before each `run()`. Params are
    /// written into their dedicated, never-aliased arena slots by `set_param`
    /// and live for the whole execution, so they are NOT re-zeroed/rewritten
    /// here — only the intermediate buffers (which carry stale data from the
    /// previous pass) are zeroed. Compile-time constants are written once.
    ///
    /// This keeps the per-run cost O(scratch) instead of O(params): a previous
    /// version cloned + rewrote the entire (multi-GB) weight region every run,
    /// which made large models swap-thrash.
    fn restore_arena_baseline(&mut self) {
        // Persistent slots (params + constants) — never zeroed.
        let persistent: std::collections::HashSet<NodeId> = {
            let mut s: std::collections::HashSet<NodeId> =
                self.param_ids.values().copied().collect();
            for node in self.graph.nodes() {
                if matches!(node.op, Op::Constant { .. }) {
                    s.insert(node.id);
                }
            }
            s
        };

        // Write compile-time constants into the arena once (a fresh arena is
        // zero-initialized; params are already resident via set_param).
        if !self.baseline_written {
            let constants: Vec<(NodeId, DType, Vec<u8>)> = self
                .graph
                .nodes()
                .iter()
                .filter_map(|node| {
                    if let Op::Constant { data } = &node.op
                        && self.arena.has_buffer(node.id)
                        && !data.is_empty()
                    {
                        Some((node.id, node.shape.dtype(), data.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            for (id, dtype, data) in constants {
                self.write_constant_to_arena(id, dtype, &data);
            }
            self.baseline_written = true;
        }

        // Zero everything EXCEPT the persistent (param + constant) byte ranges.
        //
        // We zero the *complement* of the persistent ranges rather than each
        // scratch node's exact byte span. That covers inter-slot padding and
        // arena gaps too — a kernel that over-reads its input into adjacent
        // alignment padding (common in SIMD reductions) would otherwise pick up
        // stale bytes from a previous run, since per-node zeroing only clears
        // `num_elements` and leaves the padding dirty. The cost stays O(arena −
        // params): for a 7B the params dominate the arena and are skipped, so
        // the swept region is tiny.
        let mut keep: Vec<(usize, usize)> = self
            .graph
            .nodes()
            .iter()
            .filter_map(|node| {
                let id = node.id;
                if !persistent.contains(&id) || !self.arena.has_buffer(id) {
                    return None;
                }
                let dtype = self.node_dtypes.get(&id).copied().unwrap_or(DType::F32);
                let nbytes = node.shape.num_elements().unwrap_or(0) * dtype.size_bytes();
                let off = self.arena.byte_offset(id);
                Some((off, off + nbytes))
            })
            .collect();
        keep.sort_unstable();

        let buf = self.arena.raw_buf_mut();
        let len = buf.len();
        let mut cursor = 0usize;
        for (start, end) in keep {
            let start = start.min(len);
            if cursor < start {
                buf[cursor..start].fill(0);
            }
            cursor = cursor.max(end.min(len));
        }
        if cursor < len {
            buf[cursor..len].fill(0);
        }
    }

    fn write_constant_to_arena(&mut self, id: NodeId, dtype: DType, data: &[u8]) {
        match dtype {
            DType::F64
            | DType::F16
            | DType::BF16
            | DType::U8
            | DType::I8
            | DType::Bool
            | DType::I64
            | DType::I32
            | DType::U32 => {
                let off = self.arena.byte_offset(id);
                let buf = self.arena.raw_buf_mut();
                let n = buf.len().saturating_sub(off).min(data.len());
                buf[off..off + n].copy_from_slice(&data[..n]);
            }
            _ => {
                let buf = self.arena.slice_mut(id);
                let n_floats = data.len() / 4;
                let n = buf.len().min(n_floats);
                for i in 0..n {
                    let bytes = [
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ];
                    buf[i] = f32::from_le_bytes(bytes);
                }
            }
        }
    }

    /// Direct-byte param upload — copies caller's bytes into the
    /// arena slot for the named param without any dtype conversion.
    /// Used by `set_param_typed` for dtypes that f32-widening would
    /// corrupt (F64). Caller is responsible for matching the param's
    /// declared graph dtype.
    fn set_param_bytes(&mut self, name: &str, data: &[u8], _dtype: rlx_ir::DType) {
        // Byte-backed params also live solely in the arena (no CPU-side copy).
        self.write_param_bytes_to_arena(name, data);
    }

    fn write_param_bytes_to_arena(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            let off = self.arena.byte_offset(id);
            let buf = self.arena.raw_buf_mut();
            debug_assert!(
                off + data.len() <= buf.len(),
                "set_param_bytes: '{name}' would overflow arena slot"
            );
            buf[off..off + data.len()].copy_from_slice(data);
        }
    }
}
