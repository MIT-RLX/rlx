use super::*;
use rlx_ir::OpKind;
use rlx_oneapi::backend::OneApiExecutable;

pub struct OneApiBackend;

impl Backend for OneApiBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        rlx_oneapi::backend::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // `OneApiExecutable::compile_rng` runs the legalize/rewrite pass
        // itself (decomposing DotGeneral / Fma / fused ops down to the
        // native primitive set), so hand it the graph directly.
        Box::new(OneApiExecutableWrapper {
            inner: OneApiExecutable::compile_rng_with_options(
                graph,
                options.rng,
                options.scan_unroll_max_length,
            ),
        })
    }
}

struct OneApiExecutableWrapper {
    inner: OneApiExecutable,
}

unsafe impl Send for OneApiExecutableWrapper {}

impl ExecutableGraph for OneApiExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
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

    fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.inner.set_active_extent(extent);
    }

    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.inner.set_rng(rng);
    }

    fn rng(&self) -> rlx_ir::RngOptions {
        self.inner.rng()
    }

    /// The oneAPI arena is f32-uniform: widen F16/BF16/int params to f32.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match dtype {
            rlx_ir::DType::U8 | rlx_ir::DType::I8 => self.inner.set_param_bytes(name, data),
            rlx_ir::DType::F32 => {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            }
            other => {
                let f = super::widen_bytes_to_f32(data, other);
                self.inner.set_param(name, &f);
            }
        }
    }

    /// Widen typed inputs to f32, run, then narrow each output back to its
    /// declared dtype (byte-identical with native-dtype backends).
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (name, data, dt) in inputs {
            let v = if *dt == rlx_ir::DType::F32 {
                let n = data.len() / 4;
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }.to_vec()
            } else {
                super::widen_bytes_to_f32(data, *dt)
            };
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
        Box::new(OneApiExecutableWrapper {
            inner: self.inner.clone_for_cache(),
        })
    }
}
