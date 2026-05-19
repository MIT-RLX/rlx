// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `TpuExecutable` — per-graph runtime bound to a PJRT client.
//!
//! Lifecycle:
//!   1. `compile(graph)` — run rlx-opt's unfuse pass, lower the graph
//!      to an HLO module via `lower::lower_graph`, then call
//!      `PJRT_Client_Compile` to produce a `PJRT_LoadedExecutable*`.
//!   2. `set_param(name, data)` — stash host f32 data; uploaded to
//!      the device on the first `run`.
//!   3. `run(inputs)` — upload any not-yet-uploaded params + the
//!      inputs via `Buffer_FromHostBuffer`, call
//!      `PJRT_LoadedExecutable_Execute`, drain outputs via
//!      `Buffer_ToHostBuffer`, return them in graph-output order.
//!
//! Param buffers are cached after first upload; subsequent runs only
//! re-upload the inputs. The executable is destroyed in `Drop`.

use std::collections::HashMap;
use std::ffi::c_void;

use rlx_ir::{DType, Graph};

use crate::device::tpu_context;
use crate::libtpu::{
    PJRT_BUFFER_TYPE_BF16, PJRT_BUFFER_TYPE_F16, PJRT_BUFFER_TYPE_F32, PJRT_BUFFER_TYPE_F64,
    PJRT_BUFFER_TYPE_PRED, PJRT_BUFFER_TYPE_S8, PJRT_BUFFER_TYPE_S16, PJRT_BUFFER_TYPE_S32,
    PJRT_BUFFER_TYPE_S64, PJRT_BUFFER_TYPE_U8, PJRT_BUFFER_TYPE_U32, PJRT_Buffer_Destroy_Args,
    PJRT_Buffer_ToHostBuffer_Args, PJRT_Client_BufferFromHostBuffer_Args, PJRT_Client_Compile_Args,
    PJRT_ExecuteOptions, PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL,
    PJRT_LoadedExecutable_Destroy_Args, PJRT_LoadedExecutable_Execute_Args,
    PJRT_PROGRAM_FORMAT_HLO, PJRT_Program, PjrtBuffer, PjrtLoadedExecutable, error_to_string,
    event_await,
};
use crate::lower::{HloModule, lower_graph};

/// Compiled-once, run-many TPU executable.
pub struct TpuExecutable {
    /// Compiled HLO module + I/O metadata. Owned because we re-read
    /// the param/input layout on every `run` — the IR is small, this
    /// is cheap.
    module: HloModule,

    /// Owning host copies of every parameter buffer.
    params: HashMap<String, Vec<u8>>,
    param_dtypes: HashMap<String, DType>,

    /// PJRT executable handle. NULL if compile failed (we panic
    /// before constructing in that case, so reading this in non-Drop
    /// paths is always safe — but keep the option pattern for Drop).
    executable: *mut PjrtLoadedExecutable,

    /// Lazily-uploaded device-resident parameter buffers, keyed by
    /// the parameter index in the HLO program. None until the first
    /// `run` populates it.
    param_buffers: Vec<*mut PjrtBuffer>,
    /// True after the first run uploaded everything. Subsequent runs
    /// reuse `param_buffers` and only upload inputs.
    params_uploaded: bool,
}

// PJRT executables + buffers are documented as thread-safe by the
// upstream C API.
unsafe impl Send for TpuExecutable {}

impl TpuExecutable {
    /// Compile `graph` for the active TPU device.
    pub fn compile(graph: Graph) -> Self {
        let ctx = tpu_context().unwrap_or_else(|| {
            panic!(
                "rlx-tpu: no PJRT runtime available. \
                 libtpu.so / libpjrt_c_cpu.so could not be loaded. Set \
                 LIBTPU_PATH to a plugin .so location, or install the \
                 libtpu Python package on a GCP TPU VM. Mac and \
                 non-GCP hosts have no TPU support — use Device::Cpu, \
                 Device::Mlx, or Device::Cuda there instead."
            )
        });

        // ── IR-level optimization for HLO emission ────────────────
        //
        // Run a minimal rlx-opt pipeline before lowering. XLA does
        // its own aggressive fusion + layout selection downstream, so
        // we keep this short — only passes that strictly reduce work
        // for the lowering walker or shrink the emitted module:
        //
        //   * DCE + ConstantFolding — remove unused / fold compile-
        //     time-known scalars; smaller graph → smaller HLO.
        //   * FuseResidualLN / FuseMatMulBiasAct — collapse common
        //     transformer building blocks into the tier-2 fused ops
        //     that rlx-tpu lowers directly. One HLO subgraph instead
        //     of three primitives, and we own the decomposition rather
        //     than relying on XLA's pattern matcher to recognize it.
        //   * LegalizeBroadcast — HLO requires explicit
        //     `broadcast_in_dim` shapes (no implicit numpy-style
        //     broadcasts), so canonicalize ahead of emission.
        //   * MarkElementwiseRegions — fold maximal elementwise chains
        //     into a single `Op::ElementwiseRegion`. Our lowering
        //     walks the chain inline (one HLO subgraph), so this
        //     trades many round-trip materializations for a single
        //     primitive chain.
        use rlx_opt::pass::Pass as _;
        let graph = rlx_opt::DeadCodeElimination.run(graph);
        let graph = rlx_opt::ConstantFolding.run(graph);
        let graph = rlx_opt::FuseResidualLN.run(graph);
        let graph = rlx_opt::FuseMatMulBiasAct.run(graph);
        let graph = rlx_opt::LegalizeBroadcast.run(graph);
        let graph = rlx_opt::MarkElementwiseRegions.run(graph);

        // Normalize composed ops via the local unfuse pass.
        // FusedSwiGLU / FusedAttentionBlock / FusedTransformerLayer /
        // LoraMatMul / If / While are decomposed back to primitives
        // for HLO emission. FusedMatMulBiasAct and FusedResidualLN
        // are NOT unfused — they're tier-2 fused ops that have their
        // own dedicated lowering paths in lower.rs.
        let graph = crate::unfuse::unfuse(graph);

        let module = lower_graph(&graph);

        // Optional HLO dump for inspection. RLX_TPU_HLO_DUMP can be:
        //   * a directory  → write `<dir>/<graph_name>.pb`
        //   * a file path  → write to that exact path
        // The file is the serialized `xla.HloModuleProto` (pre-XLA-
        // optimization, i.e. exactly what we send to
        // `PJRT_Client_Compile`). Inspect via:
        //
        //   from jax.lib import xla_extension
        //   m = xla_extension.HloModule.from_serialized_hlo_module_proto(
        //       open("graph.pb", "rb").read())
        //   print(m.to_string())
        if let Ok(dump_path) = std::env::var("RLX_TPU_HLO_DUMP") {
            let p = std::path::Path::new(&dump_path);
            let target: std::path::PathBuf = if p.is_dir() {
                p.join(format!("{}.pb", graph.name))
            } else {
                p.to_path_buf()
            };
            if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&target, &module.bytes) {
                Ok(()) => eprintln!(
                    "[rlx-tpu] wrote HLO dump ({} bytes) → {}",
                    module.bytes.len(),
                    target.display()
                ),
                Err(e) => eprintln!("[rlx-tpu] HLO dump to {} failed: {}", target.display(), e),
            }
        }

        // Compile via PJRT_Client_Compile. Format = "hlo" — bytes are
        // an HloModuleProto. We must pass a non-empty CompileOptions
        // proto: XLA's default leaves replica_count=0, which trips a
        // CHECK in DeviceAssignment::DeviceAssignment(). Minimal
        // proto:
        //   CompileOptionsProto {
        //     executable_build_options: {  // field 3, message
        //       num_replicas: 1,           // field 4, int64
        //       num_partitions: 1,         // field 5, int64
        //     }
        //   }
        // Hand-encoded:
        //   0x1a 0x04            -- field 3, length-delim, len=4
        //     0x20 0x01          -- field 4, varint, value=1
        //     0x28 0x01          -- field 5, varint, value=1
        const COMPILE_OPTIONS: [u8; 6] = [0x1a, 0x04, 0x20, 0x01, 0x28, 0x01];
        let format = PJRT_PROGRAM_FORMAT_HLO;
        let mut program = PJRT_Program {
            struct_size: std::mem::size_of::<PJRT_Program>(),
            extension_start: std::ptr::null_mut(),
            code: module.bytes.as_ptr() as *mut u8,
            code_size: module.bytes.len(),
            format: format.as_ptr(),
            format_size: format.len(),
        };
        let mut args = PJRT_Client_Compile_Args {
            struct_size: std::mem::size_of::<PJRT_Client_Compile_Args>(),
            extension_start: std::ptr::null_mut(),
            client: ctx.client,
            program: &program,
            compile_options: COMPILE_OPTIONS.as_ptr(),
            compile_options_size: COMPILE_OPTIONS.len(),
            executable: std::ptr::null_mut(),
        };
        let err = unsafe { (ctx.runtime.fns.client_compile)(&mut args) };
        if !err.is_null() {
            let msg = unsafe { error_to_string(&ctx.runtime.fns, err) };
            panic!("rlx-tpu: PJRT_Client_Compile failed: {msg}");
        }
        let executable = args.executable;
        if executable.is_null() {
            panic!(
                "rlx-tpu: PJRT_Client_Compile returned NULL executable \
                 without setting an error — plugin contract violation."
            );
        }
        // `program` is read-only during the call; safe to drop now.
        let _ = &mut program;

        let n_params = module.param_names.len();
        Self {
            module,
            params: HashMap::new(),
            param_dtypes: HashMap::new(),
            executable,
            param_buffers: vec![std::ptr::null_mut(); n_params],
            params_uploaded: false,
        }
    }

    /// Stash a parameter's host bytes. Treats `data` as f32 — the
    /// runtime converts non-f32 dtypes via the typed setter before
    /// calling here.
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
                .to_vec();
        self.params.insert(name.to_string(), bytes);
        self.param_dtypes.insert(name.to_string(), DType::F32);
    }

    /// Stash a parameter's host bytes with a non-f32 dtype.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.params.insert(name.to_string(), data.to_vec());
        self.param_dtypes.insert(name.to_string(), dtype);
    }

    /// Execute the graph. Inputs are matched by name to the IR's
    /// `Op::Input` nodes; outputs come back in graph-output order.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let ctx = tpu_context().expect("rlx-tpu: PJRT context vanished");
        let fns = &ctx.runtime.fns;

        // 1. Upload params on the first run.
        if !self.params_uploaded {
            for (i, name) in self.module.param_names.iter().enumerate() {
                let dtype = *self
                    .param_dtypes
                    .get(name)
                    .unwrap_or(&self.module.param_dtypes[i]);
                let dims = self.module.param_shapes[i].clone();
                let bytes = self.params.get(name).unwrap_or_else(|| {
                    panic!(
                        "rlx-tpu: parameter '{name}' was never set; call \
                     set_param before run"
                    )
                });
                let buf = upload_buffer(ctx, bytes, dtype, &dims);
                self.param_buffers[i] = buf;
            }
            self.params_uploaded = true;
        }

        // 2. Upload inputs (every run).
        let mut input_buffers: Vec<*mut PjrtBuffer> =
            vec![std::ptr::null_mut(); self.module.input_names.len()];
        for (i, name) in self.module.input_names.iter().enumerate() {
            let (_, slice) = inputs
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("rlx-tpu: input '{name}' missing from run() arguments"));
            let bytes =
                unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
            let dtype = self.module.input_dtypes[i];
            let dims = self.module.input_shapes[i].clone();
            input_buffers[i] = upload_buffer(ctx, bytes, dtype, &dims);
        }

        // 3. Build the per-device argument list. Single device.
        let mut all_args: Vec<*mut PjrtBuffer> =
            Vec::with_capacity(input_buffers.len() + self.param_buffers.len());
        all_args.extend_from_slice(&input_buffers);
        all_args.extend_from_slice(&self.param_buffers);
        // Outer pointers required by PJRT_LoadedExecutable_Execute_Args.
        let inner_args_ptr = all_args.as_ptr();
        let device_args_ptr = std::ptr::from_ref(&inner_args_ptr).cast::<*const *mut PjrtBuffer>();

        // 4. Output list — one slot per graph output (or 1 if the
        //    entry returns a tuple, since HLO returns a tuple as a
        //    single buffer of tuple shape — but TPU plugin actually
        //    flattens single-tuple outputs across the buffer list).
        //    PJRT contract: we pre-allocate the per-device output
        //    pointer array; the plugin fills the buffer pointers.
        let n_outputs = self.module.output_lens.len();
        let mut output_buffers: Vec<*mut PjrtBuffer> = vec![std::ptr::null_mut(); n_outputs];
        let device_outputs_ptr = output_buffers.as_mut_ptr();
        let device_outputs_outer = std::ptr::from_ref(&device_outputs_ptr);

        let exec_options = PJRT_ExecuteOptions {
            struct_size: std::mem::size_of::<PJRT_ExecuteOptions>(),
            extension_start: std::ptr::null_mut(),
            send_callbacks: std::ptr::null_mut(),
            recv_callbacks: std::ptr::null_mut(),
            num_send_ops: 0,
            num_recv_ops: 0,
            launch_id: 0,
            non_donatable_input_indices: std::ptr::null(),
            num_non_donatable_input_indices: 0,
            context: std::ptr::null_mut(),
            call_location: std::ptr::null(),
            num_tasks: 0,
            task_ids: std::ptr::null_mut(),
            incarnation_ids: std::ptr::null_mut(),
            multi_slice_config: std::ptr::null_mut(),
        };
        let mut exec_args = PJRT_LoadedExecutable_Execute_Args {
            struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Execute_Args>(),
            extension_start: std::ptr::null_mut(),
            executable: self.executable,
            options: &exec_options,
            argument_lists: device_args_ptr,
            num_devices: 1,
            num_args: all_args.len(),
            output_lists: device_outputs_outer,
            device_complete_events: std::ptr::null_mut(),
            execute_device: std::ptr::null_mut(),
        };
        let err = unsafe { (fns.loaded_executable_execute)(&mut exec_args) };
        if !err.is_null() {
            let msg = unsafe { error_to_string(fns, err) };
            // Best-effort cleanup of any input buffers we uploaded.
            for &b in &input_buffers {
                destroy_buffer(ctx, b);
            }
            panic!("rlx-tpu: PJRT_LoadedExecutable_Execute failed: {msg}");
        }

        // 5. Drain outputs. Each output_buffers[i] is a typed
        //    PJRT buffer; we copy to host as f32 (with widening for
        //    non-f32 dtypes).
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(n_outputs);
        for (oi, &buf) in output_buffers.iter().enumerate() {
            let n_elems = self.module.output_lens[oi];
            let dtype = self.module.output_dtypes[oi];
            let host = download_buffer(ctx, buf, n_elems, dtype);
            outputs.push(host);
            destroy_buffer(ctx, buf);
        }

        // 6. Drop input buffers (params stick around for next run).
        for &b in &input_buffers {
            destroy_buffer(ctx, b);
        }

        outputs
    }

    /// Output dtypes in graph-output order.
    pub fn output_dtypes(&self) -> Vec<DType> {
        self.module.output_dtypes.clone()
    }
}

impl Drop for TpuExecutable {
    fn drop(&mut self) {
        // Best-effort cleanup. tpu_context() should still resolve; if
        // not we leak — better than a Drop panic.
        if let Some(ctx) = tpu_context() {
            for &b in &self.param_buffers {
                destroy_buffer(ctx, b);
            }
            if !self.executable.is_null() {
                let mut args = PJRT_LoadedExecutable_Destroy_Args {
                    struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                    extension_start: std::ptr::null_mut(),
                    executable: self.executable,
                };
                let err = unsafe { (ctx.runtime.fns.loaded_executable_destroy)(&mut args) };
                if !err.is_null() {
                    // We're in Drop — eat the error rather than
                    // panicking (which would abort if Drop is called
                    // during unwinding).
                    let _ = unsafe { error_to_string(&ctx.runtime.fns, err) };
                }
                self.executable = std::ptr::null_mut();
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn pjrt_buffer_type(dt: DType) -> i32 {
    match dt {
        DType::F32 => PJRT_BUFFER_TYPE_F32,
        DType::F16 => PJRT_BUFFER_TYPE_F16,
        DType::BF16 => PJRT_BUFFER_TYPE_BF16,
        DType::F64 => PJRT_BUFFER_TYPE_F64,
        DType::I8 => PJRT_BUFFER_TYPE_S8,
        DType::I16 => PJRT_BUFFER_TYPE_S16,
        DType::I32 => PJRT_BUFFER_TYPE_S32,
        DType::I64 => PJRT_BUFFER_TYPE_S64,
        DType::U8 => PJRT_BUFFER_TYPE_U8,
        DType::U32 => PJRT_BUFFER_TYPE_U32,
        DType::Bool => PJRT_BUFFER_TYPE_PRED,
        DType::C64 => panic!("rlx-tpu: DType::C64 (complex) not yet supported"),
    }
}

fn upload_buffer(
    ctx: &crate::device::TpuContext,
    bytes: &[u8],
    dtype: DType,
    dims: &[i64],
) -> *mut PjrtBuffer {
    let fns = &ctx.runtime.fns;

    // Pick the first addressable device. PJRT requires a device or
    // memory pointer for `BufferFromHostBuffer`.
    let device = first_addressable_device(ctx);

    let mut args = PJRT_Client_BufferFromHostBuffer_Args {
        struct_size: std::mem::size_of::<PJRT_Client_BufferFromHostBuffer_Args>(),
        extension_start: std::ptr::null_mut(),
        client: ctx.client,
        data: bytes.as_ptr() as *const c_void,
        type_: pjrt_buffer_type(dtype),
        dims: dims.as_ptr(),
        num_dims: dims.len(),
        byte_strides: std::ptr::null(),
        num_byte_strides: 0,
        host_buffer_semantics: PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL,
        device,
        memory: std::ptr::null_mut(),
        device_layout: std::ptr::null_mut(),
        done_with_host_buffer: std::ptr::null_mut(),
        buffer: std::ptr::null_mut(),
    };
    let err = unsafe { (fns.client_buffer_from_host_buffer)(&mut args) };
    if !err.is_null() {
        let msg = unsafe { error_to_string(fns, err) };
        panic!("rlx-tpu: BufferFromHostBuffer failed: {msg}");
    }
    // Wait for the upload to settle if the plugin gave us an event.
    // For IMMUTABLE_ONLY_DURING_CALL semantics the host buffer stays
    // pinned only for the duration of the call; we wait synchronously
    // before returning so the slice isn't reused under us.
    if !args.done_with_host_buffer.is_null()
        && let Err(e) = unsafe { event_await(fns, args.done_with_host_buffer) }
    {
        panic!("rlx-tpu: host-buffer-done event errored: {e}");
    }
    args.buffer
}

fn destroy_buffer(ctx: &crate::device::TpuContext, buf: *mut PjrtBuffer) {
    if buf.is_null() {
        return;
    }
    let mut args = PJRT_Buffer_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Buffer_Destroy_Args>(),
        extension_start: std::ptr::null_mut(),
        buffer: buf,
    };
    let err = unsafe { (ctx.runtime.fns.buffer_destroy)(&mut args) };
    if !err.is_null() {
        let _ = unsafe { error_to_string(&ctx.runtime.fns, err) };
    }
}

fn download_buffer(
    ctx: &crate::device::TpuContext,
    buf: *mut PjrtBuffer,
    n_elems: usize,
    dtype: DType,
) -> Vec<f32> {
    let fns = &ctx.runtime.fns;
    let elem_bytes = match dtype {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 | DType::I16 => 2,
        DType::I8 | DType::U8 | DType::Bool => 1,
        DType::C64 => panic!("rlx-tpu: DType::C64 (complex) not yet supported"),
    };
    let mut host_buf: Vec<u8> = vec![0u8; n_elems * elem_bytes];
    let mut args = PJRT_Buffer_ToHostBuffer_Args {
        struct_size: std::mem::size_of::<PJRT_Buffer_ToHostBuffer_Args>(),
        extension_start: std::ptr::null_mut(),
        src: buf,
        host_layout: std::ptr::null_mut(),
        dst: host_buf.as_mut_ptr() as *mut c_void,
        dst_size: host_buf.len(),
        event: std::ptr::null_mut(),
    };
    let err = unsafe { (fns.buffer_to_host_buffer)(&mut args) };
    if !err.is_null() {
        let msg = unsafe { error_to_string(fns, err) };
        panic!("rlx-tpu: Buffer_ToHostBuffer failed: {msg}");
    }
    // Wait for the copy to complete.
    if !args.event.is_null()
        && let Err(e) = unsafe { event_await(fns, args.event) }
    {
        panic!("rlx-tpu: Buffer_ToHostBuffer event errored: {e}");
    }

    // Widen to f32 so callers see a uniform slice type.
    widen_to_f32(&host_buf, dtype, n_elems)
}

fn widen_to_f32(bytes: &[u8], dtype: DType, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
                out.push(f32::from_le_bytes(b));
            }
        }
        DType::F64 => {
            for i in 0..n {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                out.push(f64::from_le_bytes(b) as f32);
            }
        }
        DType::F16 => {
            for i in 0..n {
                let mut b = [0u8; 2];
                b.copy_from_slice(&bytes[i * 2..i * 2 + 2]);
                out.push(f16_to_f32(u16::from_le_bytes(b)));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let mut b = [0u8; 2];
                b.copy_from_slice(&bytes[i * 2..i * 2 + 2]);
                let v = u16::from_le_bytes(b);
                let f = f32::from_bits((v as u32) << 16);
                out.push(f);
            }
        }
        DType::I32 => {
            for i in 0..n {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
                out.push(i32::from_le_bytes(b) as f32);
            }
        }
        DType::I64 => {
            for i in 0..n {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                out.push(i64::from_le_bytes(b) as f32);
            }
        }
        DType::I16 => {
            for i in 0..n {
                let mut b = [0u8; 2];
                b.copy_from_slice(&bytes[i * 2..i * 2 + 2]);
                out.push(i16::from_le_bytes(b) as f32);
            }
        }
        DType::U32 => {
            for i in 0..n {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
                out.push(u32::from_le_bytes(b) as f32);
            }
        }
        DType::I8 => {
            for i in 0..n {
                out.push(bytes[i] as i8 as f32);
            }
        }
        DType::U8 => {
            for i in 0..n {
                out.push(bytes[i] as f32);
            }
        }
        DType::Bool => {
            for i in 0..n {
                out.push(if bytes[i] != 0 { 1.0 } else { 0.0 });
            }
        }
        DType::C64 => panic!("rlx-tpu: DType::C64 (complex) not yet supported"),
    }
    out
}

/// Decode IEEE 754 half. `f32::from(half::f16)` would be cleaner but
/// we don't pull `half` as a dep.
fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v >> 15) & 0x1) as u32;
    let exp = ((v >> 10) & 0x1f) as u32;
    let mant = (v & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal — normalize
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((127 - 15 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Read the first addressable device handle from the client.
fn first_addressable_device(ctx: &crate::device::TpuContext) -> *mut crate::libtpu::PjrtDevice {
    use crate::libtpu::PJRT_Client_AddressableDevices_Args;
    let mut args = PJRT_Client_AddressableDevices_Args {
        struct_size: std::mem::size_of::<PJRT_Client_AddressableDevices_Args>(),
        extension_start: std::ptr::null_mut(),
        client: ctx.client,
        addressable_devices: std::ptr::null(),
        num_addressable_devices: 0,
    };
    let err = unsafe { (ctx.runtime.fns.client_addressable_devices)(&mut args) };
    if !err.is_null() {
        let msg = unsafe { error_to_string(&ctx.runtime.fns, err) };
        panic!("rlx-tpu: Client_AddressableDevices failed: {msg}");
    }
    if args.num_addressable_devices == 0 {
        panic!("rlx-tpu: PJRT client reports no addressable devices");
    }
    unsafe { *args.addressable_devices }
}
