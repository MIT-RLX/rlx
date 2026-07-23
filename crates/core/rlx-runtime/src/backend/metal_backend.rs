use super::*;
use rlx_metal::backend::MetalExecutable;

pub struct MetalBackend;

impl Backend for MetalBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_metal::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        // Same If/While → primitive rewrite as the CPU pipeline
        // (Metal also has no native sub-graph executor wired
        // through its thunk schedule).
        let graph = rlx_opt::LowerControlFlow.run(graph);
        let dispatch = options.kernel_dispatch;
        let graph = rlx_opt::legalize_or_rewrite_for_backend_with_config(
            graph,
            rlx_metal::SUPPORTED_OPS,
            dispatch,
        )
        .unwrap_or_else(|errors| {
            panic!("{}", rlx_opt::format_legalize_error("metal", &errors));
        });
        let graph = apply_scan_device_preference(graph, options);
        let graph = crate::precompile::precompile_cleanup(graph, options);

        // Hand the policy to MetalExecutable so the rewrite runs AFTER
        // its internal fusion passes (avoids breaking pattern matchers).
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
        Box::new(MetalExecutableWrapper {
            inner: MetalExecutable::compile_with_policy(
                graph,
                options.policy.clone(),
                Some(rlx_metal::SUPPORTED_OPS),
                options.rng,
            ),
            io_manifest,
        })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        let mut graph = lir.into_graph();
        graph = rlx_opt::LowerControlFlow.run(graph);
        let dispatch = options.kernel_dispatch;
        let mut graph = rlx_opt::legalize_or_rewrite_for_backend_with_config(
            graph,
            rlx_metal::SUPPORTED_OPS,
            dispatch,
        )
        .unwrap_or_else(|errors| {
            panic!("{}", rlx_opt::format_legalize_error("metal", &errors));
        });
        graph = apply_scan_device_preference(graph, options);
        graph = crate::precompile::precompile_cleanup(graph, options);
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
        Box::new(MetalExecutableWrapper {
            inner: MetalExecutable::compile_from_fused(
                graph,
                options.policy.clone(),
                Some(rlx_metal::SUPPORTED_OPS),
                options.rng,
                options.disable_mpsgraph,
            ),
            io_manifest,
        })
    }
}

struct MetalExecutableWrapper {
    inner: MetalExecutable,
    io_manifest: cpu_low_precision::IoDtypeManifest,
}

unsafe impl Send for MetalExecutableWrapper {}

impl ExecutableGraph for MetalExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
            async_pipeline: true,
            active_extent: true,
            ..crate::ExecutableCapabilities::NONE
        }
    }

    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }

    fn finalize_params(&mut self) {
        self.inner.preload_qmatmul_weights();
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner.run(inputs)
    }
    fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.inner.run_read_outputs(inputs, read_indices)
    }
    fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.inner.bind_gpu_handle(name, data)
    }
    fn has_gpu_handle(&self, name: &str) -> bool {
        self.inner.has_gpu_handle(name)
    }
    fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.set_gpu_handle_feed(handle_name, output_index);
        true
    }
    fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle(name)
    }
    fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.register_kv_row_feed(handle_name, output_index);
        true
    }
    fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) -> bool {
        self.inner.feed_kv_row(src_row, dst_row, row_elems);
        true
    }
    fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner
            .feed_kv_batch_major(dst_row, batch, seq_cap, row_elems);
        true
    }
    fn read_output_row(&self, out_idx: usize, row: usize, row_inner: usize) -> Option<Vec<f32>> {
        Some(self.inner.read_graph_output_row(out_idx, row, row_inner))
    }
    fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.inner.run_slots(inputs)
    }
    fn arena_ptr(&self) -> *const u8 {
        self.inner.arena_ptr()
    }
    fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        self.inner.commit_no_wait(inputs);
    }
    fn sync_pending(&mut self) {
        self.inner.sync_pending();
    }
    fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        self.inner.run_pipelined(input_sets)
    }
    fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.inner.set_active_extent(extent);
    }

    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.inner.set_rng(rng);
    }

    fn rng(&self) -> rlx_ir::RngOptions {
        self.inner.rng()
    }

    /// Typed param upload — accepts F16/BF16 host bytes by widening
    /// to F32 first, then routing through `set_param` (which may
    /// re-narrow into an F16 arena slot under AutoMixedPrecision).
    /// U8/I8 packed weights and native integer/F64 params copy
    /// directly into the arena. F16 host bytes skip the widen only
    /// when the param's storage is already F16 (Zonos half-weights).
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        // Integer control params (duration carry, masks) live as F32 in the
        // Metal arena after `widen_integer_activations_to_f32`. Match
        // `run_typed`: encode values as floats. Raw i64 bytes would be read
        // as denormals (i64 3 → ~4e-45) and zero Kitten ConcatFromSequence.
        if matches!(
            dtype,
            rlx_ir::DType::I32 | rlx_ir::DType::I64 | rlx_ir::DType::U32 | rlx_ir::DType::Bool
        ) {
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.inner.set_param(name, &f32_buf);
            return;
        }
        if matches!(
            dtype,
            rlx_ir::DType::U8 | rlx_ir::DType::I8 | rlx_ir::DType::F64
        ) {
            self.inner.set_param_bytes(name, data);
            return;
        }
        if dtype == rlx_ir::DType::F16 && self.inner.param_storage_is_f16(name) {
            // F16 Linear weights into an F16 slot: copy half-bytes
            // (no f32 widen — keeps Metal Zonos under the 4 GiB cliff).
            self.inner.set_param_bytes(name, data);
            return;
        }
        if dtype == rlx_ir::DType::F32 {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            self.inner.set_param(name, s);
        } else {
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.inner.set_param(name, &f32_buf);
        }
    }

    /// Typed run. Integer inputs (I64 token ids, etc.) are copied
    /// directly into the unified-memory arena; F32/F16/BF16 widen
    /// through the existing host path. Outputs use native arena bytes.
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        self.inner.run_typed(inputs)
    }

    fn copy_params_from(&mut self, src: &dyn ExecutableGraph) -> bool {
        let Some(src_any) = src.executable_as_any() else {
            return false;
        };
        let Some(src_wrap) = src_any.downcast_ref::<MetalExecutableWrapper>() else {
            return false;
        };
        let Some(dst_any) = self.executable_as_any_mut() else {
            return false;
        };
        let Some(dst_wrap) = dst_any.downcast_mut::<MetalExecutableWrapper>() else {
            return false;
        };
        dst_wrap.inner.copy_params_from(&src_wrap.inner)
    }

    fn share_params_from(&mut self, src: &dyn ExecutableGraph) -> bool {
        let Some(src_any) = src.executable_as_any() else {
            return false;
        };
        let Some(src_wrap) = src_any.downcast_ref::<MetalExecutableWrapper>() else {
            return false;
        };
        let Some(dst_any) = self.executable_as_any_mut() else {
            return false;
        };
        let Some(dst_wrap) = dst_any.downcast_mut::<MetalExecutableWrapper>() else {
            return false;
        };
        dst_wrap.inner.share_weights_from(&src_wrap.inner)
    }

    fn executable_as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn executable_as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(MetalExecutableWrapper {
            inner: self.inner.clone_for_cache(),
            io_manifest: self.io_manifest.clone(),
        })
    }
}
