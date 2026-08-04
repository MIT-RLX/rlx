// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_ir::OpKind;
use rlx_vulkan::backend::VulkanExecutable;

pub struct VulkanBackend;

impl Backend for VulkanBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        rlx_vulkan::backend::SUPPORTED_OPS
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        // `VulkanExecutable::compile_rng` runs the legalize/rewrite pass
        // (decomposing DotGeneral / Fma / fused ops / non-last reduce down
        // to the native primitive set) itself, so we can hand it the graph
        // directly — no fusion pre-pass that would emit ops it can't lower.
        Box::new(VulkanExecutableWrapper {
            inner: VulkanExecutable::compile_rng_with_options(
                graph,
                options.rng,
                options.scan_unroll_max_length,
            ),
        })
    }
}

struct VulkanExecutableWrapper {
    inner: VulkanExecutable,
}

unsafe impl Send for VulkanExecutableWrapper {}

impl ExecutableGraph for VulkanExecutableWrapper {
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

    fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.inner.set_active_extent(extent);
    }

    fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        self.inner.set_rng(rng);
    }

    fn rng(&self) -> rlx_ir::RngOptions {
        self.inner.rng()
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

    /// The Vulkan arena is f32-uniform: widen F16/BF16/int params to f32. The
    /// exception is a bf16 **matmul weight** — kept PACKED (raw bf16 bytes, 2
    /// bytes/elem) in the arena and unpacked in the `matmul_bf16` shader, so its
    /// bytes go through untouched (halves the weight the GPU streams).
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match dtype {
            rlx_ir::DType::U8 | rlx_ir::DType::I8 => self.inner.set_param_bytes(name, data),
            rlx_ir::DType::F32 => {
                let n = data.len() / 4;
                let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
                self.inner.set_param(name, s);
            }
            rlx_ir::DType::BF16 if self.inner.is_packed_bf16_param(name) => {
                self.inner.set_param_bytes(name, data);
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
        Box::new(VulkanExecutableWrapper {
            inner: self.inner.clone_for_cache(),
        })
    }
}
