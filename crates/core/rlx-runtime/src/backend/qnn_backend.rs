use super::*;
use rlx_qnn::runtime::QnnExecutable;

pub struct QnnBackend;

impl Backend for QnnBackend {
    fn supported_ops(&self) -> &'static [rlx_ir::OpKind] {
        rlx_qnn::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, _options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // Legalize / rewrite unsupported fused forms, then decompose claimed
        // `FusedAttentionBlock` to the primitive chain the FFI runtime lowers
        // (MatMul → Narrow → Reshape/Transpose → [Rope] → Attention → …).
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, self.supported_ops())
            .unwrap_or_else(|errors| {
                panic!("{}", rlx_opt::format_legalize_error("qnn", &errors));
            });
        let graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        let exec = QnnExecutable::compile_graph(&graph)
            .unwrap_or_else(|e| panic!("rlx-qnn compile failed: {e}"));
        Box::new(QnnExecutableWrapper { inner: exec })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // No LIR arena path for QNN; reconstruct the graph and go through the
        // normal compile (legalize + FAB unfuse + FFI lower).
        self.compile(lir.into_graph(), options)
    }
}

struct QnnExecutableWrapper {
    inner: QnnExecutable,
}

impl ExecutableGraph for QnnExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities::NONE
    }

    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }

    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match dtype {
            rlx_ir::DType::F32 => {
                if data.len() % 4 != 0 {
                    panic!(
                        "set_param_typed F32: data length {} not a multiple of 4",
                        data.len()
                    );
                }
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                self.inner.set_param(name, &floats);
            }
            rlx_ir::DType::U8 | rlx_ir::DType::I8 => {
                self.inner
                    .set_param_bytes(name, data, dtype)
                    .unwrap_or_else(|e| panic!("rlx-qnn set_param_typed: {e}"));
            }
            other => panic!("rlx-qnn set_param_typed: unsupported dtype {other:?}"),
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.inner
            .run(inputs)
            .unwrap_or_else(|e| panic!("rlx-qnn run failed: {e}"))
    }
}
