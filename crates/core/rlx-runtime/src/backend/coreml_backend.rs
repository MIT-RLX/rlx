use super::*;
use crate::Precision;
use rlx_coreml::{CoremlExecutable, default_lower_options};

pub struct CoremlBackend;

impl Backend for CoremlBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        // Under `training` the claim includes the native backward kernels so
        // the fusion pipeline preserves them for the dedicated MIL arm instead
        // of decomposing them first (decompose-route backward ops stay out, so
        // the pipeline still turns those into primitives). See
        // `rlx_coreml::SUPPORTED_OPS_TRAINING`.
        #[cfg(feature = "training")]
        {
            &rlx_coreml::SUPPORTED_OPS_TRAINING
        }
        #[cfg(not(feature = "training"))]
        {
            rlx_coreml::SUPPORTED_OPS
        }
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // `supported_ops()` already includes the native backward kernels under
        // `training`, so the rewrite keeps them intact for their MIL arm while
        // decomposing the decompose-route backward ops to primitives.
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, self.supported_ops())
            .unwrap_or_else(|errors| {
                panic!("{}", rlx_opt::format_legalize_error("coreml", &errors));
            });
        let graph = apply_scan_device_preference(graph, options);
        // Automatic Floating Point: apply the mixed-precision policy if set.
        // AMP keeps boundaries (input/param/const/output) in F32 and casts at
        // f16 boundaries, so CoreML lowers a valid mixed f16/f32 ML Program
        // (per-node dtypes; consts stay F32 → no storage/type mismatch). This
        // is the precise way to get f16 compute — unlike a blanket
        // `float_dtype = F16` flip, which mis-sizes float consts.
        let (graph, mut lower_opts) = match options.policy.clone() {
            Some(policy) => {
                use rlx_opt::pass::Pass as _;
                let g = rlx_opt::AutoMixedPrecision::new(policy).run(graph);
                let opts = default_lower_options(&g);
                (g, opts)
            }
            None => {
                let mut opts = default_lower_options(&graph);
                // Legacy blanket-F16 path (no AMP policy): kept for inference.
                if options.precision == Precision::F16 {
                    opts.float_dtype = rlx_ir::DType::F16;
                }
                (graph, opts)
            }
        };
        if let Some(binding) = &options.dim_binding {
            let _ = binding;
            lower_opts.flexible_inputs = false;
        }
        Box::new(CoremlExecutableWrapper {
            inner: CoremlExecutable::compile_with_lower_opts(graph, lower_opts),
        })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // No LIR arena path for CoreML; reconstruct the graph and go
        // through the normal compile.
        self.compile(lir.into_graph(), options)
    }
}

struct CoremlExecutableWrapper {
    inner: CoremlExecutable,
}

// The loaded MLModel is owned exclusively and accessed via &mut self.
unsafe impl Send for CoremlExecutableWrapper {}

impl ExecutableGraph for CoremlExecutableWrapper {
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

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        // GGUF-quantized weights arrive as raw bytes; the lowering
        // host-dequantizes them when baking the model.
        self.inner.set_param_typed(name, data, dtype);
    }

    fn finalize_params(&mut self) {
        self.inner
            .finalize()
            .unwrap_or_else(|e| panic!("CoreML finalize failed: {e}"));
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner
            .run(inputs)
            .unwrap_or_else(|e| panic!("CoreML run failed: {e}"))
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(CoremlExecutableWrapper {
            inner: self.inner.clone_for_cache(),
        })
    }

    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        use rlx_ir::DType;
        // The CoreML model is promoted to an f32 flow (`promote_int_to_f32`), so
        // widen integer/bool inputs (e.g. `phone_ids`) to f32 on the host surface.
        let owned: Vec<(String, Vec<f32>)> = inputs
            .iter()
            .map(|(name, data, dt)| {
                let v: Vec<f32> = match dt {
                    DType::I64 => data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::I32 => data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                        .collect(),
                    DType::U8 | DType::Bool => data.iter().map(|&b| b as f32).collect(),
                    _ => super::widen_bytes_to_f32(data, *dt),
                };
                (name.to_string(), v)
            })
            .collect();
        let refs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        self.run(&refs)
            .into_iter()
            .map(|v| {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (bytes, DType::F32)
            })
            .collect()
    }
}
