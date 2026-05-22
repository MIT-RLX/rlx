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
//! Multi-segment TPU execution: HLO subgraphs interleaved with host splat steps.

use std::collections::HashMap;

use rlx_ir::{DType, Graph, NodeId, Op};

use crate::backend::{compile_pjrt_executable, destroy_buffer, download_buffer, upload_buffer};
use crate::device::tpu_context;
use crate::lower::{HloModule, lower_graph};
use crate::segment::{Segment, plan};
use crate::splat_host::{HostTensors, run_splat_backward, run_splat_render};

pub struct OrchestratedExecutable {
    graph: Graph,
    segments: Vec<CompiledSegment>,
    params: HashMap<String, Vec<u8>>,
    param_dtypes: HashMap<String, DType>,
}

enum CompiledSegment {
    Hlo {
        module: HloModule,
        executable: *mut crate::libtpu::PjrtLoadedExecutable,
        param_buffers: Vec<*mut crate::libtpu::PjrtBuffer>,
        params_uploaded: bool,
        output_orig: Vec<NodeId>,
    },
    SplatRender {
        node: NodeId,
    },
    SplatBackward {
        node: NodeId,
    },
}

impl OrchestratedExecutable {
    pub fn compile(graph: Graph) -> Self {
        let segments = plan(&graph);
        let mut compiled = Vec::new();
        for seg in segments {
            match seg {
                Segment::Hlo {
                    graph: seg_graph,
                    output_orig,
                } => {
                    let module = lower_graph(&seg_graph);
                    let executable = compile_pjrt_executable(&module.bytes);
                    let n_params = module.param_names.len();
                    compiled.push(CompiledSegment::Hlo {
                        module,
                        executable,
                        param_buffers: vec![std::ptr::null_mut(); n_params],
                        params_uploaded: false,
                        output_orig,
                    });
                }
                Segment::SplatRender { node } => {
                    compiled.push(CompiledSegment::SplatRender { node });
                }
                Segment::SplatBackward { node } => {
                    compiled.push(CompiledSegment::SplatBackward { node });
                }
            }
        }
        Self {
            graph,
            segments: compiled,
            params: HashMap::new(),
            param_dtypes: HashMap::new(),
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
                .to_vec();
        self.params.insert(name.to_string(), bytes);
        self.param_dtypes.insert(name.to_string(), DType::F32);
    }

    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.params.insert(name.to_string(), data.to_vec());
        self.param_dtypes.insert(name.to_string(), dtype);
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut env: HostTensors = HashMap::new();
        for n in self.graph.nodes() {
            if let Op::Input { name } = &n.op {
                if let Some((_, data)) = inputs.iter().find(|(n, _)| n == name) {
                    env.insert(n.id, data.to_vec());
                }
            }
        }

        for seg in &mut self.segments {
            match seg {
                CompiledSegment::Hlo {
                    module,
                    executable,
                    param_buffers,
                    params_uploaded,
                    output_orig,
                } => {
                    let outs = run_hlo_segment(
                        module,
                        *executable,
                        param_buffers,
                        params_uploaded,
                        &self.params,
                        &self.param_dtypes,
                        inputs,
                        &env,
                    );
                    for (oid, data) in output_orig.iter().zip(outs) {
                        env.insert(*oid, data);
                    }
                }
                CompiledSegment::SplatRender { node } => {
                    run_splat_render(&self.graph, *node, &mut env);
                }
                CompiledSegment::SplatBackward { node } => {
                    run_splat_backward(&self.graph, *node, &mut env);
                }
            }
        }

        self.graph
            .outputs
            .iter()
            .map(|&id| {
                env.remove(&id)
                    .unwrap_or_else(|| panic!("rlx-tpu: missing output tensor for {id:?}"))
            })
            .collect()
    }

    pub fn output_dtypes(&self) -> Vec<DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }
}

impl Drop for OrchestratedExecutable {
    fn drop(&mut self) {
        if let Some(ctx) = tpu_context() {
            for seg in &mut self.segments {
                if let CompiledSegment::Hlo {
                    param_buffers,
                    executable,
                    ..
                } = seg
                {
                    for b in param_buffers.drain(..) {
                        destroy_buffer(ctx, b);
                    }
                    if !executable.is_null() {
                        use crate::libtpu::{
                            PJRT_LoadedExecutable_Destroy_Args, error_to_string,
                        };
                        let mut args = PJRT_LoadedExecutable_Destroy_Args {
                            struct_size: std::mem::size_of::<PJRT_LoadedExecutable_Destroy_Args>(),
                            extension_start: std::ptr::null_mut(),
                            executable: *executable,
                        };
                        let err = unsafe {
                            (ctx.runtime.fns.loaded_executable_destroy)(&mut args)
                        };
                        let _ = unsafe { error_to_string(&ctx.runtime.fns, err) };
                        *executable = std::ptr::null_mut();
                    }
                }
            }
        }
    }
}

fn run_hlo_segment(
    module: &HloModule,
    executable: *mut crate::libtpu::PjrtLoadedExecutable,
    param_buffers: &mut Vec<*mut crate::libtpu::PjrtBuffer>,
    params_uploaded: &mut bool,
    params: &HashMap<String, Vec<u8>>,
    param_dtypes: &HashMap<String, DType>,
    inputs: &[(&str, &[f32])],
    env: &HostTensors,
) -> Vec<Vec<f32>> {
    let ctx = tpu_context().expect("rlx-tpu: PJRT context vanished");
    let fns = &ctx.runtime.fns;

    if !*params_uploaded {
        for (i, name) in module.param_names.iter().enumerate() {
            let dtype = *param_dtypes
                .get(name)
                .unwrap_or(&module.param_dtypes[i]);
            let dims = module.param_shapes[i].clone();
            let bytes = params.get(name).unwrap_or_else(|| {
                panic!(
                    "rlx-tpu: parameter '{name}' was never set; call set_param before run"
                )
            });
            param_buffers[i] = upload_buffer(ctx, bytes, dtype, &dims);
        }
        *params_uploaded = true;
    }

    let mut input_buffers: Vec<*mut crate::libtpu::PjrtBuffer> =
        vec![std::ptr::null_mut(); module.input_names.len()];
    for (i, name) in module.input_names.iter().enumerate() {
        let bytes: &[u8] = if let Some((_, slice)) = inputs.iter().find(|(n, _)| n == name) {
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
        } else if let Some(prefix) = name.strip_prefix("__bnd_") {
            let orig = NodeId(
                prefix
                    .parse::<u32>()
                    .unwrap_or_else(|_| panic!("rlx-tpu: bad boundary name '{name}'")),
            );
            let data = env
                .get(&orig)
                .unwrap_or_else(|| panic!("rlx-tpu: boundary tensor missing for {orig:?}"));
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
        } else {
            panic!("rlx-tpu: segment input '{name}' missing from run() and env");
        };
        let dtype = module.input_dtypes[i];
        let dims = module.input_shapes[i].clone();
        input_buffers[i] = upload_buffer(ctx, bytes, dtype, &dims);
    }

    let mut all_args: Vec<*mut crate::libtpu::PjrtBuffer> =
        Vec::with_capacity(input_buffers.len() + param_buffers.len());
    all_args.extend_from_slice(&input_buffers);
    all_args.extend_from_slice(param_buffers);
    let inner_args_ptr = all_args.as_ptr();
    let device_args_ptr = std::ptr::from_ref(&inner_args_ptr).cast::<*const *mut crate::libtpu::PjrtBuffer>();

    let n_outputs = module.output_lens.len();
    let mut output_buffers: Vec<*mut crate::libtpu::PjrtBuffer> = vec![std::ptr::null_mut(); n_outputs];
    let device_outputs_ptr = output_buffers.as_mut_ptr();
    let device_outputs_outer = std::ptr::from_ref(&device_outputs_ptr);

    use crate::libtpu::{
        PJRT_ExecuteOptions, PJRT_LoadedExecutable_Execute_Args, error_to_string,
    };
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
        executable,
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
        for &b in &input_buffers {
            destroy_buffer(ctx, b);
        }
        panic!("rlx-tpu: PJRT_LoadedExecutable_Execute failed: {msg}");
    }

    let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(n_outputs);
    for (oi, &buf) in output_buffers.iter().enumerate() {
        let n_elems = module.output_lens[oi];
        let dtype = module.output_dtypes[oi];
        outputs.push(download_buffer(ctx, buf, n_elems, dtype));
        destroy_buffer(ctx, buf);
    }
    for &b in &input_buffers {
        destroy_buffer(ctx, b);
    }
    outputs
}
