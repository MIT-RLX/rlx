// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_mlx::MlxExecutable;

pub struct MlxBackend;

impl Backend for MlxBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_mlx::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let compile_result = crate::stages::compile_graph_stages_for_backend(
            rlx_driver::Device::Mlx,
            graph,
            options,
            rlx_mlx::SUPPORTED_OPS,
        );
        crate::stages::maybe_log_fusion(&compile_result.fusion);
        self.compile_lir(compile_result.lir, options)
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        let mut graph = lir.into_graph();
        graph = rlx_opt::LowerControlFlow.run(graph);
        graph = rlx_opt::rlx_autodiff::decompose_backward_ops_except(graph, rlx_mlx::SUPPORTED_OPS);
        let graph = prepare_fused_graph(graph, options, rlx_mlx::SUPPORTED_OPS, "mlx");
        Box::new(build_mlx_executable(graph, options.rng))
    }
}

fn build_mlx_executable(graph: Graph, rng: rlx_ir::RngOptions) -> MlxExecutableWrapper {
    let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
    // Integer control params (Kitten duration carry, masks) must live as F32
    // so `set_param_typed` can widen values — raw i64 bytes are read as
    // denormals by f32 Where/Expand and break alignment (same as Metal).
    let graph = cpu_low_precision::widen_integer_control_params_to_f32(graph);
    let mode = mlx_mode_from_env();
    let mut exe = MlxExecutable::compile_from_fused_with_rng(graph, mode, rng);
    if mode == rlx_mlx::lower::MlxMode::Compiled {
        if let Err(e) = exe.warm_compile() {
            eprintln!(
                "[rlx-runtime] MLX warm_compile failed ({e}); first run will pay the trace cost"
            );
        }
    }
    MlxExecutableWrapper {
        inner: exe,
        io_manifest,
    }
}

fn mlx_mode_from_env() -> rlx_mlx::lower::MlxMode {
    match rlx_ir::env::var("RLX_MLX_MODE").as_deref() {
        Some(s) if s.eq_ignore_ascii_case("eager") => rlx_mlx::lower::MlxMode::Eager,
        Some(s) if s.eq_ignore_ascii_case("lazy") => rlx_mlx::lower::MlxMode::Lazy,
        Some(s) if s.eq_ignore_ascii_case("compiled") => rlx_mlx::lower::MlxMode::Compiled,
        _ => rlx_mlx::lower::MlxMode::Compiled,
    }
}

struct MlxExecutableWrapper {
    inner: MlxExecutable,
    io_manifest: cpu_low_precision::IoDtypeManifest,
}

unsafe impl Send for MlxExecutableWrapper {}

impl ExecutableGraph for MlxExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            persistent_handles: true,
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
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner.run(inputs)
    }
    fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.inner
            .run_read_outputs(inputs, read_indices)
            .unwrap_or_else(|e| panic!("MLX run_read_outputs failed: {e}"))
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
    fn bind_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.inner.bind_handle(name, data)
    }
    fn read_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_handle(name)
    }
    fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        self.inner.bind_gpu_handle(name, data).is_ok()
    }
    fn has_gpu_handle(&self, name: &str) -> bool {
        self.inner.has_gpu_handle(name)
    }
    fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.set_gpu_handle_feed(handle_name, output_index);
        true
    }
    fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) -> bool {
        self.inner.register_kv_row_feed(handle_name, output_index);
        true
    }
    fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) -> bool {
        self.inner.feed_kv_row(src_row, dst_row, row_elems).is_ok()
    }
    fn feed_kv_batch_major(
        &mut self,
        dst_row: usize,
        batch: usize,
        seq_cap: usize,
        row_elems: usize,
    ) -> bool {
        self.inner
            .feed_kv_batch_major(dst_row, batch, seq_cap, row_elems)
            .is_ok()
    }
    fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle(name).ok()
    }
    fn read_output_row(&self, out_idx: usize, row: usize, row_inner: usize) -> Option<Vec<f32>> {
        self.inner.read_output_row(out_idx, row, row_inner)
    }
    fn run_feed_gpu_handle(
        &mut self,
        inputs: &[(&str, &[f32])],
        handle_name: &str,
        output_index: usize,
    ) -> Option<Vec<f32>> {
        self.inner
            .run_feed_gpu(inputs, handle_name, output_index)
            .ok()
    }
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        // Integer control params live as F32 after
        // `widen_integer_control_params_to_f32`. Match Metal/wgpu: encode
        // values as floats. Raw i64 bytes would fail MLX's typed-param
        // dtype check (I64 vs graph F32) or be read as denormals.
        if matches!(
            dtype,
            rlx_ir::DType::I32 | rlx_ir::DType::I64 | rlx_ir::DType::U32 | rlx_ir::DType::Bool
        ) {
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.inner.set_param(name, &f32_buf);
            return;
        }
        self.inner.set_param_typed(name, data, dtype);
    }
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        self.inner.run_typed(inputs)
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

    fn copy_params_from(&mut self, src: &dyn ExecutableGraph) -> bool {
        let Some(src_any) = src.executable_as_any() else {
            return false;
        };
        let Some(src_wrap) = src_any.downcast_ref::<MlxExecutableWrapper>() else {
            return false;
        };
        let Some(dst_any) = self.executable_as_any_mut() else {
            return false;
        };
        let Some(dst_wrap) = dst_any.downcast_mut::<MlxExecutableWrapper>() else {
            return false;
        };
        dst_wrap.inner.copy_params_from(&src_wrap.inner)
    }

    fn executable_as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn executable_as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(MlxExecutableWrapper {
            inner: self.inner.clone_for_cache(),
            io_manifest: self.io_manifest.clone(),
        })
    }
}
