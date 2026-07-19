use super::*;
use rlx_tpu::TpuExecutable;

pub struct TpuBackend;

impl Backend for TpuBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_tpu::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let graph =
            rlx_opt::rlx_autodiff::decompose_backward_ops_except(graph, rlx_tpu::SUPPORTED_OPS);
        let graph = rlx_opt::legalize_or_rewrite_for_backend_with_config(
            graph,
            rlx_tpu::SUPPORTED_OPS,
            options.kernel_dispatch,
        )
        .unwrap_or_else(|errors| {
            panic!("{}", rlx_opt::format_legalize_error("tpu", &errors));
        });
        // The TPU's IR-side pass pipeline (DCE, ConstFold,
        // FuseResidualLN, FuseMatMulBiasAct, LegalizeBroadcast,
        // MarkElementwiseRegions) lives inside
        // `TpuExecutable::compile` so the same passes run whether
        // a caller goes through Session or invokes the executable
        // directly. We only do backend-cross-cutting work here:
        // legalization (must precede the pipeline so we panic
        // early on unsupported ops) and AutoMixedPrecision.
        //
        // Default policy on TPU is `AutoMixedBf16`: BF16 is the
        // native compute dtype on TPU silicon and recent GPUs,
        // and XLA's CPU plugin handles it natively too. Callers
        // can opt out by passing an explicit `PrecisionPolicy`
        // (e.g. `AlwaysF32` for accuracy debugging or
        // `AlwaysF16` to match a CUDA workload's choice).
        use rlx_opt::pass::Pass as _;
        let policy = options
            .policy
            .clone()
            .unwrap_or(rlx_opt::PrecisionPolicy::AutoMixedBf16);
        let graph = rlx_opt::AutoMixedPrecision::new(policy).run(graph);
        let _ = options.dce;
        let _ = options.constant_folding;
        Box::new(TpuExecutableWrapper {
            inner: TpuExecutable::compile_rng_with_param_bytes(
                graph,
                options.rng,
                options.quant_param_bindings.as_ref(),
            ),
        })
    }
}

struct TpuExecutableWrapper {
    inner: TpuExecutable,
}

// PJRT clients + buffers are documented as thread-safe per the
// upstream C API. Same Send-claim shape as CudaExecutableWrapper /
// RocmExecutableWrapper.
unsafe impl Send for TpuExecutableWrapper {}

impl ExecutableGraph for TpuExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            typed_io: true,
            ..crate::ExecutableCapabilities::NONE
        }
    }

    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }
    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner.run(inputs)
    }

    /// Typed param upload — widens F16/BF16/etc. host bytes to
    /// f32 today. Once the HLO emitter speaks bf16 natively
    /// (which TPUs prefer over f16), the typed path will hand
    /// the original bytes straight through `Buffer_FromHostBuffer`.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        if dtype == rlx_ir::DType::F32 {
            let n = data.len() / 4;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
            self.inner.set_param(name, s);
        } else {
            let f32_buf = super::widen_bytes_to_f32(data, dtype);
            self.inner.set_param(name, &f32_buf);
        }
    }

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
        let dtypes = self.inner.output_dtypes();
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
        Box::new(TpuExecutableWrapper {
            inner: self.inner.clone_for_cache(),
        })
    }
}
