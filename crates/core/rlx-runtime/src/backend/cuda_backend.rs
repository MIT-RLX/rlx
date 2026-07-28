// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_cuda::backend::CudaExecutable;

pub struct CudaBackend;

impl Backend for CudaBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_cuda::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        // Decompose FusedSwiGLU / FAB / etc. before legalization (CudaExecutable
        // unfuses again; this pass is idempotent).
        let graph = rlx_cuda::unfuse::unfuse(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, rlx_cuda::SUPPORTED_OPS)
            .unwrap_or_else(|errors| {
                panic!("{}", rlx_opt::format_legalize_error("cuda", &errors));
            });
        let graph = apply_scan_device_preference(graph, options);
        let graph = crate::precompile::precompile_cleanup(graph, options);
        // Mid-axis broadcasts (EEG patch embed) before elementwise fusion.
        let graph = rlx_opt::LegalizeBroadcast.run(graph);
        // Backend-aware fusion via the shared compile pipeline.
        let compile_result = crate::stages::compile_graph_stages_for_backend(
            rlx_driver::Device::Cuda,
            graph,
            options,
            rlx_cuda::SUPPORTED_OPS,
        );
        crate::stages::maybe_log_fusion(&compile_result.fusion);
        let graph = compile_result.lir.into_graph();
        let graph = match options.policy.clone() {
            Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
            None => graph,
        };
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
        Box::new(CudaExecutableWrapper {
            inner: CudaExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        let graph = rlx_opt::LegalizeBroadcast.run(lir.into_graph());
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(prepare_fused_graph(
            rlx_cuda::unfuse::unfuse(graph),
            options,
            rlx_cuda::SUPPORTED_OPS,
            "cuda",
        ));
        Box::new(CudaExecutableWrapper {
            inner: CudaExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }
}

struct CudaExecutableWrapper {
    inner: CudaExecutable,
    io_manifest: cpu_low_precision::IoDtypeManifest,
}

// CudaExecutable owns CudaContext + CudaSlice handles; cudarc claims
// they're Send (CudaContext is Arc-wrapped, CudaSlice is logically
// a device pointer + length). The Backend trait requires Send for
// the executable; we honor that here.
unsafe impl Send for CudaExecutableWrapper {}

impl ExecutableGraph for CudaExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            gpu_handles: true,
            kv_resident: true,
            typed_io: true,
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
    fn read_output_row(&self, out_idx: usize, row: usize, row_inner: usize) -> Option<Vec<f32>> {
        self.inner.read_output_row(out_idx, row, row_inner)
    }
    fn read_gpu_handle_row(&self, name: &str, row: usize, row_inner: usize) -> Option<Vec<f32>> {
        self.inner.read_gpu_handle_row(name, row, row_inner)
    }
    fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        self.inner.prepare_resident_gpu_handle(name)
    }
    fn stage_bound_gpu_handles_to_arena(&mut self) {
        self.inner.stage_bound_gpu_handles_to_arena();
    }
    fn seed_resident_kv_prefix_from(
        &mut self,
        src: &dyn ExecutableGraph,
        prefix_tokens: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        let Some(dst_exe) = self.cuda_executable_for_kv_seed() else {
            return false;
        };
        let Some(src_exe) = src.cuda_executable_for_kv_seed_ref() else {
            return false;
        };
        dst_exe.seed_resident_kv_prefix_from(
            src_exe,
            prefix_tokens,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }
    fn copy_resident_kv_rows_from(
        &mut self,
        src: &dyn ExecutableGraph,
        from_row: usize,
        to_row: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        let Some(dst_exe) = self.cuda_executable_for_kv_seed() else {
            return false;
        };
        let Some(src_exe) = src.cuda_executable_for_kv_seed_ref() else {
            return false;
        };
        dst_exe.copy_resident_kv_rows_from(
            src_exe,
            from_row,
            to_row,
            outgoing_upper,
            kv_dim,
            n_layers,
        )
    }
    fn cuda_executable_for_kv_seed(&mut self) -> Option<&mut rlx_cuda::backend::CudaExecutable> {
        Some(&mut self.inner)
    }
    fn cuda_executable_for_kv_seed_ref(&self) -> Option<&rlx_cuda::backend::CudaExecutable> {
        Some(&self.inner)
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

    fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.inner.run_slots(inputs)
    }

    fn arena_ptr(&self) -> *const u8 {
        self.inner.arena_ptr()
    }

    /// Typed param upload. F16/BF16 Linear weights are registered in the
    /// half-arena (`set_param_half`) so `cublasGemmEx` matches Metal's
    /// native F16 matmul and CPU f16 thunks — widening-only upload was
    /// the root of F5 DiT chained ODE drift (cos≈0.87 vs Metal's 1.0).
    /// The f32 slot is also filled as a fallback for non-matmul readers.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if matches!(dtype, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
            self.inner.set_param_bytes(name, data);
            return;
        }
        if dtype == rlx_ir::DType::F32 {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            self.inner.set_param(name, s);
            return;
        }
        if matches!(dtype, rlx_ir::DType::F16 | rlx_ir::DType::BF16) {
            let bits: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let half_dt = match dtype {
                rlx_ir::DType::BF16 => rlx_cuda::arena::HalfDtype::Bf16,
                _ => rlx_cuda::arena::HalfDtype::F16,
            };
            self.inner.set_param_half(name, half_dt, &bits);
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.inner.set_param(name, &f32_buf);
            return;
        }
        let f32_buf = super::widen_bytes_to_f32(data, dtype);
        self.inner.set_param(name, &f32_buf);
    }

    /// Typed run — widen each typed input to F32, run, then narrow
    /// each output back to its declared graph dtype.
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (name, data, dt) in inputs {
            let v = super::widen_bytes_to_f32(data, *dt);
            owned.push((name.to_string(), v));
        }
        let refs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let dtypes = super::declared_output_dtypes(&self.io_manifest, self.inner.output_dtypes());
        let outs = self.inner.run(&refs);
        outs.into_iter()
            .zip(
                dtypes
                    .into_iter()
                    .chain(std::iter::repeat(rlx_ir::DType::F32)),
            )
            .map(|(v, dt)| (super::narrow_f32_to_bytes(&v, dt), dt))
            .collect()
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(CudaExecutableWrapper {
            inner: self.inner.clone_for_cache(),
            io_manifest: self.io_manifest.clone(),
        })
    }
}
