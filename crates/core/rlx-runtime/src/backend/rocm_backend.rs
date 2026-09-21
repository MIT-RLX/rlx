// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_rocm::backend::RocmExecutable;

pub struct RocmBackend;

/// Everything `compile` does to a graph before `rlx-rocm` plans and lowers it.
///
/// Extracted so the arena size can be **asked** rather than estimated. Each of
/// these passes changes how much memory the plan needs — `unfuse` alone is
/// where `Op::SelectiveScan` becomes a trajectory-saving `Op::Scan`, which for
/// a Mamba backward is the single largest term — so a caller that plans the
/// raw graph is sizing something the backend never sees. Predicting from the
/// raw graph under-counted by 1.70x on a 12-layer Mamba training graph
/// (6.12 GB predicted vs 10.39 GB actual), enough to choose a batch that then
/// failed to compile.
///
/// `log_fusion` is the one behavioural difference: the planning path stays
/// quiet so a size probe does not spam the fusion report.
pub(crate) fn prepare_rocm_exec_graph(
    graph: Graph,
    options: &CompileOptions,
    log_fusion: bool,
) -> (Graph, cpu_low_precision::IoDtypeManifest) {
    use rlx_opt::pass::Pass as _;
    let graph = rlx_rocm::unfuse::unfuse(graph);
    let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, rlx_rocm::SUPPORTED_OPS)
        .unwrap_or_else(|errors| {
            panic!("{}", rlx_opt::format_legalize_error("rocm", &errors));
        });
    let graph = apply_scan_device_preference(graph, options);
    let graph = crate::precompile::precompile_cleanup(graph, options);
    let graph = rlx_opt::LegalizeBroadcast.run(graph);
    let compile_result = crate::stages::compile_graph_stages_for_backend(
        rlx_driver::Device::Rocm,
        graph,
        options,
        rlx_rocm::SUPPORTED_OPS,
    );
    if log_fusion {
        crate::stages::maybe_log_fusion(&compile_result.fusion);
    }
    let graph = compile_result.lir.into_graph();
    let graph = match options.policy.clone() {
        Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
        None => graph,
    };
    // Non-trailing `Op::Reduce` is lowered inside `rlx_rocm`'s own compile
    // entry (mirroring rlx-cuda), so every caller gets it — not just this
    // wrapper.
    cpu_low_precision::prepare_f32_exec_graph(graph)
}

impl Backend for RocmBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_rocm::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let (graph, io_manifest) = prepare_rocm_exec_graph(graph, options, true);
        Box::new(RocmExecutableWrapper {
            inner: RocmExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let fused = prepare_fused_graph(
            rlx_rocm::unfuse::unfuse(lir.into_graph()),
            options,
            rlx_rocm::SUPPORTED_OPS,
            "rocm",
        );
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(fused);
        Box::new(RocmExecutableWrapper {
            inner: RocmExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }
}

struct RocmExecutableWrapper {
    inner: RocmExecutable,
    io_manifest: cpu_low_precision::IoDtypeManifest,
}

// Same Send-claim shape as CudaExecutableWrapper. RocmExecutable
// owns Arc<RocmContext> + HipBuffer handles; the HipRuntime bundle
// is internally thread-safe per AMD's documentation.
unsafe impl Send for RocmExecutableWrapper {}

impl ExecutableGraph for RocmExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            moe: true,
            gpu_handles: true,
            typed_io: true,
            active_extent: true,
            kv_resident: true,
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
    fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.inner.run_slots(inputs)
    }
    fn arena_ptr(&self) -> *const u8 {
        self.inner.arena_ptr()
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

    fn set_moe_resident_experts(&mut self, mask: &[bool]) {
        self.inner.set_moe_resident_experts(mask);
    }

    fn set_moe_resident_experts_per_layer(&mut self, masks: &[&[bool]]) {
        self.inner.set_moe_resident_experts_per_layer(masks);
    }

    /// Typed param upload. F16/BF16 Linear weights go through
    /// `set_param_half` (hipBLAS GemmEx) to match Metal/CPU f16 matmul;
    /// f32 slot filled as fallback for non-matmul readers.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if matches!(dtype, rlx_ir::DType::U8 | rlx_ir::DType::I8) {
            self.inner.set_param_bytes(name, data);
            return;
        }
        if dtype == rlx_ir::DType::F32 {
            let s = rlx_ir::bytes::decode_le::<f32>(data);
            self.inner.set_param(name, s.as_ref());
            return;
        }
        if matches!(dtype, rlx_ir::DType::F16 | rlx_ir::DType::BF16) {
            let bits: Vec<u16> = data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let half_dt = match dtype {
                rlx_ir::DType::BF16 => rlx_rocm::arena::HalfDtype::Bf16,
                _ => rlx_rocm::arena::HalfDtype::F16,
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
        Box::new(RocmExecutableWrapper {
            inner: self.inner.clone_for_cache(),
            io_manifest: self.io_manifest.clone(),
        })
    }
}
