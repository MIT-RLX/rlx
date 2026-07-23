use super::*;
use rlx_ir::OpKind;
use rlx_wgpu::backend::WgpuExecutable;

pub struct WgpuBackend;

impl Backend for WgpuBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        rlx_wgpu::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        let graph = rlx_opt::LowerControlFlow.run(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, rlx_wgpu::SUPPORTED_OPS)
            .unwrap_or_else(|errors| {
                panic!("{}", rlx_opt::format_legalize_error("wgpu", &errors));
            });
        let graph = apply_scan_device_preference(graph, options);
        let graph = crate::precompile::precompile_cleanup(graph, options);
        // Materialize mid-axis broadcasts before MarkElementwiseRegions:
        // wgpu Binary/region kernels only handle trailing/scalar broadcast
        // via modulus; EEG patch embed uses [1,C,1,D] + [1,C,P,D].
        let graph = rlx_opt::LegalizeBroadcast.run(graph);
        // ORDER MATTERS: targeted-pattern fusions run BEFORE the
        // catch-all `MarkElementwiseRegions`. Otherwise the region
        // pass swallows the Add / Activation nodes into chains and
        // FuseMatMulBiasAct / FuseResidualLN fail to match the
        // narrower patterns they look for. (Metal pipeline at line
        // ~377 already orders these correctly; wgpu was inverted
        // and silently shipped 13 unfused LayerNorms per BERT
        // forward where 12 should have been FusedResidualLN.)
        let compile_result = crate::stages::compile_graph_stages_for_backend(
            rlx_driver::Device::Gpu,
            graph,
            options,
            rlx_wgpu::SUPPORTED_OPS,
        );
        crate::stages::maybe_log_fusion(&compile_result.fusion);
        let graph = compile_result.lir.into_graph();
        let graph = match options.policy.clone() {
            Some(p) => rlx_opt::AutoMixedPrecision::new(p).run(graph),
            None => graph,
        };
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
        Box::new(WgpuExecutableWrapper {
            inner: WgpuExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        use rlx_opt::pass::Pass as _;
        // LIR may already contain fused ElementwiseRegions; legalize
        // broadcasts on the unfused graph shape before backend prep.
        let graph = rlx_opt::LegalizeBroadcast.run(lir.into_graph());
        let graph = prepare_fused_graph(graph, options, rlx_wgpu::SUPPORTED_OPS, "wgpu");
        let (graph, io_manifest) = cpu_low_precision::prepare_f32_exec_graph(graph);
        Box::new(WgpuExecutableWrapper {
            inner: WgpuExecutable::compile_rng(graph, options.rng),
            io_manifest,
        })
    }
}

struct WgpuExecutableWrapper {
    inner: WgpuExecutable,
    io_manifest: cpu_low_precision::IoDtypeManifest,
}

unsafe impl Send for WgpuExecutableWrapper {}

impl ExecutableGraph for WgpuExecutableWrapper {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            gpu_handles: true,
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
    fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        self.inner.prepare_resident_gpu_handle(name)
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

    /// Typed param upload: the wgpu arena is f32-uniform, so widen F16/BF16 and
    /// small-integer/bool params (I64/I32/U32/Bool — e.g. `input_ids`, duration
    /// carry, alignment frame count) to F32 at the host boundary. U8/I8 packed
    /// quant weights keep their raw bytes.
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match dtype {
            rlx_ir::DType::U8 | rlx_ir::DType::I8 => {
                self.inner.set_param_bytes(name, data);
            }
            rlx_ir::DType::F32 => {
                let n = data.len() / 4;
                let f32_slice =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, f32_slice);
            }
            rlx_ir::DType::F16
            | rlx_ir::DType::BF16
            | rlx_ir::DType::F64
            | rlx_ir::DType::I64
            | rlx_ir::DType::I32
            | rlx_ir::DType::U32
            | rlx_ir::DType::Bool => {
                let f32 = super::widen_bytes_to_f32(data, dtype);
                self.inner.set_param(name, &f32);
            }
            other => panic!(
                "rlx-wgpu set_param_typed: dtype {other:?} unsupported \
                             (f32-uniform arena; widen small ints/floats to F32)"
            ),
        }
    }

    /// Typed run: widen each typed input to F32, run, then narrow each
    /// output back to its declared dtype.
    fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], rlx_ir::DType)],
    ) -> Vec<(Vec<u8>, rlx_ir::DType)> {
        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (name, data, dt) in inputs {
            let v: Vec<f32> = match *dt {
                rlx_ir::DType::F32 => {
                    let n = data.len() / 4;
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }.to_vec()
                }
                rlx_ir::DType::F16 => {
                    let n = data.len() / 2;
                    let s =
                        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n) };
                    s.iter().map(|h| h.to_f32()).collect()
                }
                rlx_ir::DType::BF16 => {
                    let n = data.len() / 2;
                    let s = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n)
                    };
                    s.iter().map(|h| h.to_f32()).collect()
                }
                // Integer/bool inputs (e.g. embedding indices `phone_ids`) are
                // widened to f32, matching the f32-arena convention shared with
                // the CPU backend (Gather etc. operate on f32-encoded indices).
                rlx_ir::DType::I64 => {
                    let n = data.len() / 8;
                    let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n) };
                    s.iter().map(|&x| x as f32).collect()
                }
                rlx_ir::DType::I32 => {
                    let n = data.len() / 4;
                    let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n) };
                    s.iter().map(|&x| x as f32).collect()
                }
                rlx_ir::DType::U8 | rlx_ir::DType::I8 | rlx_ir::DType::Bool => {
                    data.iter().map(|&b| b as f32).collect()
                }
                other => {
                    panic!("rlx-wgpu run_typed: input '{name}' dtype {other:?} unsupported")
                }
            };
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
            .map(|(v, dt)| (narrow_to_dtype(&v, dt), dt))
            .collect()
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(WgpuExecutableWrapper {
            inner: self.inner.clone_for_cache(),
            io_manifest: self.io_manifest.clone(),
        })
    }

    #[cfg(target_arch = "wasm32")]
    fn wgpu_requires_host(&self) -> bool {
        self.inner.requires_host_execution()
    }

    #[cfg(target_arch = "wasm32")]
    fn wgpu_run_async<'a>(
        &'a mut self,
        inputs: &'a [(&'a str, &'a [f32])],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, String>> + 'a>>
    {
        if self.inner.requires_host_execution() {
            return Box::pin(async move {
                Err(
                    "WebGPU wasm graph contains host-executed ops (GGUF dequant, LSTM, \
                     collectives, FFT-host, …) — use WebGL fallback or CPU"
                        .into(),
                )
            });
        }
        Box::pin(async move { Ok(self.inner.run_async(inputs).await) })
    }
}

/// Cast every element of a wgpu f32 output buffer down to the
/// declared output dtype, returning the corresponding byte stream.
/// The arena keeps every value as f32; declared output dtypes
/// (Bool, I8, I32, F16, ...) require an exit-time narrowing to be
/// byte-identical with backends that store the native dtype.
fn narrow_to_dtype(v: &[f32], dt: rlx_ir::DType) -> Vec<u8> {
    use rlx_ir::DType;
    match dt {
        DType::F32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            bytes
        }
        DType::F16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&half::f16::from_f32(x).to_le_bytes());
            }
            bytes
        }
        DType::BF16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&half::bf16::from_f32(x).to_le_bytes());
            }
            bytes
        }
        DType::F64 => {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&(x as f64).to_le_bytes());
            }
            bytes
        }
        DType::I8 => v.iter().map(|&x| x as i8 as u8).collect(),
        DType::U8 => v.iter().map(|&x| x as u8).collect(),
        DType::I16 => {
            let mut bytes = Vec::with_capacity(v.len() * 2);
            for &x in v {
                bytes.extend_from_slice(&(x as i16).to_le_bytes());
            }
            bytes
        }
        DType::I32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&(x as i32).to_le_bytes());
            }
            bytes
        }
        DType::U32 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&(x as u32).to_le_bytes());
            }
            bytes
        }
        DType::I64 => {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&(x as i64).to_le_bytes());
            }
            bytes
        }
        DType::Bool => v
            .iter()
            .map(|&x| if x != 0.0 { 1u8 } else { 0u8 })
            .collect(),
        // C64 (complex f32 pair) — the wgpu backend's f32 arena
        // doesn't synthesize complex outputs today; this branch
        // only fires if a graph somehow asks for a C64 output and
        // the backend lowered it as 2N real floats. We pass the
        // raw f32 stream straight through; downstream code that
        // wants complex semantics is responsible for re-pairing.
        DType::C64 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for &x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            bytes
        }
        // C128 (complex f64 pair) — like C64, the f32-uniform wgpu arena
        // never synthesizes complex outputs (C128 casts/ops are rejected
        // at lowering). Defensive only: widen the raw f32 lanes to f64 so
        // the byte width matches the declared 16 B/elem C128 layout.
        DType::C128 => {
            let mut bytes = Vec::with_capacity(v.len() * 8);
            for &x in v {
                bytes.extend_from_slice(&(x as f64).to_le_bytes());
            }
            bytes
        }
    }
}
