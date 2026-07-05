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

//! `compile` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ArgmaxParams, AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams,
    Conv1dParams, Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams,
    DequantMatmulParams, ElementwiseRegionParams, ExpandParams, FmaParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, GruParams, Im2Col2dParams, Kernel, LayerNormBwdParams,
    LayerNormParams, Mamba2Params, MatmulParams, MatmulQkvParams, NarrowConcatParams, Pool1dParams,
    Pool2dParams, Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams,
    RopeParams, SampleParams, ScatterAddParams, SceParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
    argmax_kernel, attention_bwd_kernel, attention_kernel, batch_elementwise_region_kernel,
    binary_kernel, cast_f32_to_f16_kernel, compare_kernel, concat_kernel, conv1d_kernel,
    conv1d_tiled_kernel, conv2d_kernel, conv3d_kernel, copy_kernel, cumsum_backward_kernel,
    cumsum_kernel, dequant_matmul_kernel, elementwise_region_kernel,
    elementwise_region_spatial_kernel, expand_kernel, fma_kernel, fused_residual_ln_kernel,
    fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel, gather_axis_kernel,
    gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel, gather_split_kernel,
    grouped_matmul_kernel, gru_kernel, im2col2d_kernel, layer_norm_backward_gamma_partial_kernel,
    layer_norm_backward_gamma_reduce_kernel, layer_norm_backward_input_kernel, layernorm_kernel,
    mamba2_kernel, matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel,
    pool3d_kernel, reduce_kernel, rms_norm_backward_kernel, rms_norm_backward_param_kernel,
    rnn_kernel, rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_kernel,
    selective_scan_kernel, softmax_cross_entropy_kernel, softmax_kernel, topk_kernel,
    transpose_kernel, umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel,
    welch_peaks_gpu_kernel, where_kernel,
};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    /// Compile against an explicit `DimBinding`. Each `Dim::Dynamic`
    /// in the graph that maps to a symbol in `bindings` is replaced
    /// with `Dim::Static(size)` before the standard compile runs.
    /// Symbols not in the binding stay dynamic — and then `compile`
    /// will panic with the usual diagnostic.
    pub fn compile_with_bindings(graph: Graph, bindings: &DimBinding) -> Self {
        if bindings.is_empty() {
            return Self::compile(graph);
        }
        // Walk the graph and bind every node's shape.
        let mut fresh = Graph::new(&graph.name);
        for node in graph.nodes() {
            let bound = node.shape.bind(bindings);
            fresh.add_node(node.op.clone(), node.inputs.clone(), bound);
        }
        fresh.set_outputs(graph.outputs.clone());
        Self::compile(fresh)
    }

    pub fn compile(graph: Graph) -> Self {
        Self::compile_rng(graph, rlx_ir::RngOptions::default())
    }

    pub fn compile_rng(graph: Graph, rng: rlx_ir::RngOptions) -> Self {
        let rng = std::sync::Arc::new(std::sync::RwLock::new(rng));
        if has_dynamic_dims(&graph) {
            return Self::deferred(graph, rng);
        }
        Self::compile_static_inner(graph, rng)
    }

    pub(crate) fn compile_static_inner(
        graph: Graph,
        rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    ) -> Self {
        let dev = wgpu_device().expect("rlx-wgpu: no compatible adapter found");

        // Decompose composed/fused ops (FusedMatMulBiasAct, LoraMatMul,
        // FusedAttentionBlock, FusedTransformerLayer, ...) into primitive
        // sequences before memory planning so every intermediate gets a
        // regular arena slot. CPU/Metal/MLX lower the fused variants
        // directly with bespoke kernels; we choose simplicity over peak
        // throughput here.
        let graph = crate::unfuse::unfuse(graph);

        // f32-uniform slots + liveness reuse (pairwise `[n,n]` graphs).
        let mut plan = plan_f32_uniform(&graph, 16);
        let dequant_scratch = crate::gguf_gpu::dequant_gguf_scratch_bytes(&graph);
        let dequant_scratch_off = if dequant_scratch > 0 {
            let aligned = plan.arena_size.div_ceil(16) * 16;
            let new_size = aligned + dequant_scratch.div_ceil(16) * 16;
            if (new_size as u64) <= dev.device.limits().max_buffer_size {
                plan.arena_size = new_size;
                aligned
            } else {
                0
            }
        } else {
            0
        };
        // Pre-walk to compute the max scratch any single op needs.
        // Currently only `Op::LayerNormBackwardGamma` uses scratch
        // (`num_workgroups * H * 4` bytes for the partial-sums buffer).
        let base_scratch_bytes = compute_scratch_bytes(&graph);
        // Reserve tail scratch for the im2col `col` matrix only when the opt-in
        // im2col+GEMM conv path is enabled (the default tiled/direct convs need
        // no extra scratch, so the arena stays lean).
        let conv_col_scratch = if rlx_ir::env::flag("RLX_WGPU_CONV_IM2COL") {
            conv_im2col_scratch_bytes(
                &graph,
                plan.arena_size,
                dev.device.limits().max_storage_buffer_binding_size,
            )
        } else {
            0
        };
        let scratch_bytes = base_scratch_bytes.max(conv_col_scratch);
        let mut arena = Arena::from_plan_with_scratch(&dev.device, &plan, scratch_bytes);
        // Override slot lengths with the actual elem*4 byte counts so
        // readback returns the right element count (slots may be
        // padded for alignment).
        for node in graph.nodes() {
            let elems = node.shape.num_elements().unwrap_or(0);
            arena.set_actual_len(node.id, elems * 4);
        }

        // Initialize Constants directly into the arena. The wgpu arena is f32-
        // uniform (4 bytes/elem; integer inputs are widened on upload), so integer
        // and bool constants must be converted to f32 here too — writing their raw
        // bytes would corrupt them (e.g. the VITS sequence-mask `arange` i64 const,
        // 8 bytes/elem, read back as f32 garbage → all-zero mask → dead encoder).
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let widened: Option<Vec<u8>> = match node.shape.dtype() {
                    rlx_ir::DType::I64 => Some(
                        data.chunks_exact(8)
                            .flat_map(|c| {
                                (i64::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                            })
                            .collect(),
                    ),
                    rlx_ir::DType::I32 => Some(
                        data.chunks_exact(4)
                            .flat_map(|c| {
                                (i32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                            })
                            .collect(),
                    ),
                    rlx_ir::DType::U32 => Some(
                        data.chunks_exact(4)
                            .flat_map(|c| {
                                (u32::from_le_bytes(c.try_into().unwrap()) as f32).to_le_bytes()
                            })
                            .collect(),
                    ),
                    rlx_ir::DType::Bool | rlx_ir::DType::U8 => Some(
                        data.iter()
                            .flat_map(|&b| (b as f32).to_le_bytes())
                            .collect(),
                    ),
                    rlx_ir::DType::I8 => Some(
                        data.iter()
                            .flat_map(|&b| ((b as i8) as f32).to_le_bytes())
                            .collect(),
                    ),
                    _ => None,
                };
                let bytes: &[u8] = widened.as_deref().unwrap_or(data);
                let bytes_to_write = bytes.len().min(arena.len_of(node.id));
                dev.queue.write_buffer(
                    &arena.buffer,
                    arena.offset(node.id) as u64,
                    &bytes[..bytes_to_write],
                );
            }
        }

        let mut input_offsets = HashMap::new();
        let mut param_offsets = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_offsets.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_offsets.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        let mm_k = matmul_kernel(&dev.device);
        let mm_w = matmul_wide_kernel(&dev.device);
        let _mm_w_active = matmul_wide_active_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
        let mm_cast = cast_f32_to_f16_kernel(&dev.device);
        let bk = binary_kernel(&dev.device);
        let uk = unary_kernel(&dev.device);
        let ck = compare_kernel(&dev.device);
        let wk = where_kernel(&dev.device);
        let fk = fma_kernel(&dev.device);

        let mut schedule = Vec::new();
        let mut uniforms = Vec::new();
        let mut bind_groups = Vec::new();
        let mut fft_gpu_steps: Vec<crate::fft_dispatch::FftGpuResources> = Vec::new();
        let mut gguf_host_pad: Option<(wgpu::Buffer, wgpu::BindGroup)> = None;
        let mut meta_buffers: Vec<wgpu::Buffer> = Vec::new();
        let mut coop_f16_b_param: HashMap<u32, String> = HashMap::new();
        let mut coop_f16_vk_wide_bind_groups: HashMap<usize, wgpu::BindGroup> = HashMap::new();
        let mm_w_active_compile = matmul_wide_active_kernel(&dev.device);

        let coop_f16_vk_mirror_acts = collect_coop_f16_vk_mirror_activations(&graph, &dev.device);

        // Detect (FusedMatMulBiasAct → Narrow×3) split-QKV pattern. Returns
        // a map parent_node_id → (q_narrow_id, k_narrow_id, v_narrow_id).
        // The matmul_qkv kernel collapses the matmul + 3 narrows into one
        // dispatch by routing each output column to the right Q/K/V sink.
        //
        // CRITICAL: only mark a pattern site for elision when the parent
        // FMB will actually take the MatmulQkv path (which only fires
        // for F32 compute precision). For Coop16/CoopF32-eligible FMBs,
        // those kernels write to the FMB's *own* output slot, NOT the
        // 3 narrow slots — skipping the narrows would leave Q/K/V
        // uninitialized and attention would read garbage. Predict the
        // compute precision the FMB will receive; only skip when F32.
        let mut qkv_split: HashMap<NodeId, (NodeId, NodeId, NodeId)> = HashMap::new();
        for (parent_id, qkv) in detect_split_qkv_pattern(&graph) {
            let parent = graph.node(parent_id);
            // Mirror the lowering's precision derivation. FMB inputs:
            // [a, w, bias]; we need (m, k, n) to query.
            let a_id = parent.inputs[0];
            let b_id = parent.inputs[1];
            let a_dims = graph.node(a_id).shape.dims();
            let b_dims = graph.node(b_id).shape.dims();
            let out_dims = parent.shape.dims();
            let (m, k, n) =
                if a_dims.len() >= 2 && b_dims.len() == 2 && out_dims.len() == a_dims.len() {
                    let leading: usize = a_dims[..a_dims.len() - 2]
                        .iter()
                        .map(|d| d.unwrap_static())
                        .product();
                    let m_inner = a_dims[a_dims.len() - 2].unwrap_static();
                    let k_inner = a_dims[a_dims.len() - 1].unwrap_static();
                    let n_inner = b_dims[1].unwrap_static();
                    ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
                } else if a_dims.len() == 2 && b_dims.len() == 2 {
                    (
                        a_dims[0].unwrap_static() as u32,
                        a_dims[1].unwrap_static() as u32,
                        b_dims[1].unwrap_static() as u32,
                    )
                } else {
                    continue; // unusual shape — let the regular FMB path handle
                };
            let cp = derive_matmul_compute(
                &dev.device,
                &graph,
                &coop_f16_vk_mirror_acts,
                a_id,
                b_id,
                m,
                k,
                n,
            );
            // F32 → matmul_qkv. CoopF32 → matmul_qkv_coop_f32. Both write
            // Q/K/V into the narrow output slots, so the narrows can be
            // elided. Coop16 still falls back to FMB+narrows (kernel
            // would need an f16-acc variant; deferred).
            if cp == MatmulCompute::F32 || cp == MatmulCompute::CoopF32 {
                qkv_split.insert(parent_id, qkv);
            }
        }
        let qkv_skip_narrows: HashSet<NodeId> = qkv_split
            .values()
            .flat_map(|&(q, k, v)| [q, k, v])
            .collect();

        // EEG-DINO / packed QKV: FMB → [B,S,3,H,D] → Narrow×3 (axis 2) → Attention.
        // Match CPU `compile_thunks` fused_strided_attn: read Q/K/V from the
        // packed parent with seq stride 3·H·D instead of materializing narrows.
        let mut packed_bshd_attn: HashMap<NodeId, (NodeId, u32)> = HashMap::new();
        let mut packed_bshd_skip_narrows: HashSet<NodeId> = HashSet::new();
        if !rlx_ir::env::flag("RLX_WGPU_NO_PACKED_BSHD_ATTN") {
            for node in graph.nodes() {
                let Op::Attention { .. } = &node.op else {
                    continue;
                };
                if node.inputs.len() < 3 {
                    continue;
                }
                if let Some((parent, head_width, narrows)) =
                    rlx_ir::detect_packed_bshd_qkv_attention(
                        &graph,
                        node.inputs[0],
                        node.inputs[1],
                        node.inputs[2],
                    )
                {
                    packed_bshd_attn.insert(node.id, (parent, head_width as u32));
                    for narrow in narrows {
                        if rlx_ir::packed_bshd_narrow_elidable(&graph, narrow, node.id) {
                            packed_bshd_skip_narrows.insert(narrow);
                        }
                    }
                }
            }
        }

        // Detect (Add → LayerNorm) where Add has multi-consumer downstream.
        // The standard `FuseResidualLN` pass declines to fuse these (its
        // single-consumer guard forces materializing the sum); we collapse
        // them here at the wgpu lowering level via `Step::FusedResidualLnTee`.
        // Returns:
        //   ln_to_tee: ln_id  → (h, delta, gamma, beta, sum_arena_id)
        //   skip_adds: { add_id }  — these Add nodes are computed by the
        //                            tee step; their normal Step emission
        //                            is suppressed.
        let (ln_to_tee, skip_adds) = detect_residual_ln_tee_pattern(&graph);

        let mut coop_f16_host_activations: Vec<(NodeId, Activation, String)> = Vec::new();

        let emit_uniform = |size: usize| -> wgpu::Buffer {
            dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu uniform"),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        for node in graph.nodes() {
            // Helpers — capture device + arena into closures isn't
            // ergonomic in the loop, so inline the bind-group build
            // when each step is emitted below.
            let elems = node.shape.num_elements().unwrap_or(0) as u32;
            match &node.op {
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => continue,
                Op::MatMul => {
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_shape = graph.node(a_id).shape.dims();
                    let b_shape = graph.node(b_id).shape.dims();
                    let out_shape = node.shape.dims();
                    // Three patterns:
                    //   • 2D×2D                              → batch=1
                    //   • [..,M,K] × [K,N]  (broadcast rhs)  → batch=1, flatten leading into M
                    //   • [..,M,K] × [..,K,N] (matched batch)→ batch=prod(leading), per-batch strides
                    let (m, k, n, batch, a_bs, b_bs, c_bs) = if a_shape.len() == 2
                        && b_shape.len() == 2
                        && out_shape.len() == 2
                    {
                        (
                            a_shape[0].unwrap_static() as u32,
                            a_shape[1].unwrap_static() as u32,
                            b_shape[1].unwrap_static() as u32,
                            1u32,
                            0u32,
                            0u32,
                            0u32,
                        )
                    } else if a_shape.len() >= 2
                        && b_shape.len() == 2
                        && out_shape.len() == a_shape.len()
                    {
                        let leading: usize = a_shape[..a_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .product();
                        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                        let n_inner = b_shape[1].unwrap_static();
                        (
                            (leading * m_inner) as u32,
                            k_inner as u32,
                            n_inner as u32,
                            1u32,
                            0u32,
                            0u32,
                            0u32,
                        )
                    } else if (a_shape.len() >= 3 || b_shape.len() >= 3)
                        && a_shape.len() >= 2
                        && b_shape.len() >= 2
                        && out_shape.len() == a_shape.len().max(b_shape.len())
                    {
                        // Batched: leading PRODUCTS must match. Allows broadcasting a
                        // leading-1 across a rank mismatch, e.g. the audio rel-pos attn
                        // a=[1,H,M,K] (lead 1·H) × b=[H,K,N] (lead H) → b_count=H.
                        let lead_a: usize = a_shape[..a_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .product();
                        let lead_b: usize = b_shape[..b_shape.len() - 2]
                            .iter()
                            .map(|d| d.unwrap_static())
                            .product();
                        if lead_a != lead_b {
                            panic!(
                                "rlx-wgpu MatMul: batched leading-product mismatch \
                                    a={a_shape:?} b={b_shape:?} out={out_shape:?}"
                            );
                        }
                        let b_count: usize = lead_a;
                        let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                        let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                        let n_inner = b_shape[b_shape.len() - 1].unwrap_static();
                        (
                            m_inner as u32,
                            k_inner as u32,
                            n_inner as u32,
                            b_count as u32,
                            (m_inner * k_inner) as u32,
                            (k_inner * n_inner) as u32,
                            (m_inner * n_inner) as u32,
                        )
                    } else {
                        panic!(
                            "rlx-wgpu MatMul: unsupported shapes a={a_shape:?} b={b_shape:?} \
                                out={out_shape:?} (supported: 2D×2D, [..,M,K]×[K,N], [..,M,K]×[..,K,N])"
                        );
                    };
                    let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                    let b_bytes = arena.len_of(b_id) as u64;
                    let mut compute_precision = derive_matmul_compute(
                        &dev.device,
                        &graph,
                        &coop_f16_vk_mirror_acts,
                        a_id,
                        b_id,
                        m,
                        k,
                        n,
                    );
                    if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena.param_fits_f16_mirror(b_id)
                        && !rlx_ir::env::flag("RLX_WGPU_NO_F16_MIRROR")
                    {
                        compute_precision = MatmulCompute::F16;
                    }
                    let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param);
                    let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        node.id,
                        a_id,
                        b_id,
                        b_in_arena,
                    );
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    // Only grow to cover B when B is actually bound through the arena.
                    let expand_ids: &[NodeId] = if b_in_arena {
                        &[node.id, a_id, b_id]
                    } else {
                        &[node.id, a_id]
                    };
                    arena_expand_bind_window(&arena, expand_ids, &mut base, &mut size, max_binding);
                    let mut scratch = arena.scratch_off as u64;
                    if param_anchor {
                        arena_ensure_scratch_in_window(&mut scratch, base, size);
                    }
                    if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                        // The invariant we actually need is that the large param
                        // B is addressable in the bound window. That holds either
                        // via an explicit param anchor OR when the whole arena is
                        // bound (`arena_whole_arena_bind`), in which case
                        // `arena_matmul_bind_window` returns `param_anchor=false`
                        // but B is trivially in `[0, arena.size)`. Keying the
                        // assert on `param_anchor` alone spuriously panicked for
                        // models whose entire arena fits `max_binding`.
                        assert!(
                            arena_tensor_in_window(&arena, b_id, base, size),
                            "rlx-wgpu matmul: large param B {:?} off={} not in window base={base} size={size}",
                            b_id,
                            arena.offset(b_id),
                        );
                    }
                    let a_off_f32 = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off_f32 = if !b_in_arena {
                        // B is read from the f16 shadow buffer (separate binding);
                        // never bind/stage it through the arena window. Use the
                        // global arena word index — build_matmul_bind_group rebases
                        // it into the f16 weight window.
                        (arena.offset(b_id) / 4) as u32
                    } else if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena_tensor_in_window(&arena, b_id, base, size)
                    {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            b_id,
                            &mut base,
                            &mut size,
                        )
                    };
                    maybe_push_coop_f16_vk_casts(
                        &graph,
                        a_id,
                        b_id,
                        &coop_f16_vk_mirror_acts,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut uniforms,
                        &mut bind_groups,
                        &mm_cast,
                        compute_precision,
                        a_off_f32,
                        m,
                        k,
                        batch,
                        b_off_f32,
                        n,
                    );
                    schedule.push(Step::Matmul {
                        m,
                        k,
                        n,
                        batch,
                        a_batch_stride: a_bs,
                        b_batch_stride: b_bs,
                        c_batch_stride: c_bs,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32: arena_local_off_f32(&arena, node.id, base),
                        has_bias: 0,
                        bias_off_f32: 0,
                        act_id: 0xFFFF,
                        b_is_param,
                        compute_precision,
                    });
                    let b_off_global = (arena.offset(b_id) / 4) as u32;
                    let b_off_bind = if b_is_param
                        && matches!(
                            compute_precision,
                            MatmulCompute::Coop16 | MatmulCompute::CoopF16Vk | MatmulCompute::F16
                        ) {
                        b_off_global
                    } else {
                        b_off_f32
                    };
                    register_coop_f16_vk_b_param(
                        &mut coop_f16_b_param,
                        &param_offsets,
                        b_id,
                        b_off_bind,
                        compute_precision,
                    );
                    let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                    let (bg, b_off_adj) = build_matmul_bind_group(
                        &dev.device,
                        mm_k,
                        mm_w,
                        &mm_f16w,
                        &mm_f16c,
                        &mm_coop,
                        &mm_coop_f32,
                        &arena,
                        base,
                        size,
                        &u,
                        b_is_param,
                        compute_precision,
                        k,
                        n,
                        batch,
                        b_off_bind,
                        b_bs,
                    );
                    if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                        *b_off_f32 = b_off_adj;
                    }
                    uniforms.push(u);
                    bind_groups.push(bg);
                    if compute_precision == MatmulCompute::CoopF16Vk {
                        coop_f16_vk_wide_bind_groups.insert(
                            schedule.len() - 1,
                            bind_two_buf0_window(
                                &dev.device,
                                mm_w_active_compile,
                                &arena.buffer,
                                base,
                                size,
                                &uniforms[uniforms.len() - 1],
                            ),
                        );
                    }
                }
                Op::Binary(bop) => {
                    // Skip emit when this Add is consumed by a downstream
                    // FRLTee — the tee step writes the sum to this node's
                    // arena slot directly. Subsequent consumers read the
                    // same slot and find correct data.
                    if skip_adds.contains(&node.id) {
                        continue;
                    }
                    require_equal_shapes(&graph, &node.inputs, "Binary");
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let win_ids = [node.id, a_id, b_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let a_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        a_id,
                        &mut base,
                        &mut size,
                    );
                    let b_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        b_id,
                        &mut base,
                        &mut size,
                    );
                    let p = BinaryParams {
                        n: elems,
                        a_off,
                        b_off,
                        c_off: arena_local_off_f32(&arena, node.id, base),
                        op: binary_op_id(*bop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Binary { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, bk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Compare(cop) => {
                    require_equal_shapes(&graph, &node.inputs, "Compare");
                    let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_src = arena.offset(a_id) as u64;
                    let b_src = arena.offset(b_id) as u64;
                    let a_len = arena.len_of(a_id) as u64;
                    let b_len = arena.len_of(b_id) as u64;
                    let a_in = a_src >= base && a_src + a_len <= base + size;
                    let b_in = b_src >= base && b_src + b_len <= base + size;
                    let a_dst = arena.scratch_off as u64;
                    let a_aligned = a_len.div_ceil(256) * 256;
                    let b_dst = a_dst + a_aligned;
                    if a_dst < base || b_dst + b_len > base + size {
                        base = (arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let a_off = if a_in {
                        arena_local_off_f32(&arena, a_id, base)
                    } else {
                        if a_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Compare staging operand A too large ({a_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: a_src,
                            dst_byte_off: a_dst,
                            bytes: a_len as u32,
                        });
                        ((a_dst.saturating_sub(base)) / 4) as u32
                    };
                    let b_off = if b_in {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        if b_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Compare staging operand B too large ({b_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: b_src,
                            dst_byte_off: b_dst,
                            bytes: b_len as u32,
                        });
                        ((b_dst.saturating_sub(base)) / 4) as u32
                    };
                    let p = BinaryParams {
                        n: elems,
                        a_off,
                        b_off,
                        c_off: arena_local_off_f32(&arena, node.id, base),
                        op: compare_op_id(*cop),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Compare { params: p });
                    let u = emit_uniform(std::mem::size_of::<BinaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, ck, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Activation(act) => {
                    if coop_f16_vk_mirror_acts.contains(&node.id) {
                        let src_name =
                            tensor_host_name(&input_offsets, &param_offsets, node.inputs[0]);
                        coop_f16_host_activations.push((node.id, *act, src_name));
                        continue;
                    }
                    let in_id = node.inputs[0];
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = UnaryParams {
                        n: elems,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        op: activation_op_id(*act),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Unary {
                        params: p,
                        f16_mirror: false,
                    });
                    let u = emit_uniform(std::mem::size_of::<UnaryParams>());
                    let bg = bind_two_buf0_window(&dev.device, uk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Where => {
                    let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let cond_id = node.inputs[0];
                    let x_id = node.inputs[1];
                    let y_id = node.inputs[2];
                    let cond_src = arena.offset(cond_id) as u64;
                    let x_src = arena.offset(x_id) as u64;
                    let y_src = arena.offset(y_id) as u64;
                    let cond_len = arena.len_of(cond_id) as u64;
                    let x_len = arena.len_of(x_id) as u64;
                    let y_len = arena.len_of(y_id) as u64;
                    let cond_in = cond_src >= base && cond_src + cond_len <= base + size;
                    let x_in = x_src >= base && x_src + x_len <= base + size;
                    let y_in = y_src >= base && y_src + y_len <= base + size;
                    let cond_dst = arena.scratch_off as u64;
                    let cond_aligned = cond_len.div_ceil(256) * 256;
                    let x_dst = cond_dst + cond_aligned;
                    let x_aligned = x_len.div_ceil(256) * 256;
                    let y_dst = x_dst + x_aligned;
                    if cond_dst < base || y_dst + y_len > base + size {
                        base = (arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let cond_off = if cond_in {
                        arena_local_off_f32(&arena, cond_id, base)
                    } else {
                        if cond_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging cond too large ({cond_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: cond_src,
                            dst_byte_off: cond_dst,
                            bytes: cond_len as u32,
                        });
                        ((cond_dst.saturating_sub(base)) / 4) as u32
                    };
                    let x_off = if x_in {
                        arena_local_off_f32(&arena, x_id, base)
                    } else {
                        if x_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging x too large ({x_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: x_src,
                            dst_byte_off: x_dst,
                            bytes: x_len as u32,
                        });
                        ((x_dst.saturating_sub(base)) / 4) as u32
                    };
                    let y_off = if y_in {
                        arena_local_off_f32(&arena, y_id, base)
                    } else {
                        if y_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Where staging y too large ({y_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: y_src,
                            dst_byte_off: y_dst,
                            bytes: y_len as u32,
                        });
                        ((y_dst.saturating_sub(base)) / 4) as u32
                    };
                    let p = WhereParams {
                        n: elems,
                        cond_off,
                        x_off,
                        y_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Where { params: p });
                    let u = emit_uniform(std::mem::size_of::<WhereParams>());
                    let bg = bind_two_buf0_window(&dev.device, wk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Fma => {
                    let (mut base, size) = arena_window_for_nodes(&dev.device, &arena, &[node.id]);
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let c_id = node.inputs[2];
                    let a_src = arena.offset(a_id) as u64;
                    let b_src = arena.offset(b_id) as u64;
                    let c_src = arena.offset(c_id) as u64;
                    let a_len = arena.len_of(a_id) as u64;
                    let b_len = arena.len_of(b_id) as u64;
                    let c_len = arena.len_of(c_id) as u64;
                    let a_in = a_src >= base && a_src + a_len <= base + size;
                    let b_in = b_src >= base && b_src + b_len <= base + size;
                    let c_in = c_src >= base && c_src + c_len <= base + size;
                    let a_dst = arena.scratch_off as u64;
                    let a_aligned = a_len.div_ceil(256) * 256;
                    let b_dst = a_dst + a_aligned;
                    let b_aligned = b_len.div_ceil(256) * 256;
                    let c_dst = b_dst + b_aligned;
                    if a_dst < base || c_dst + c_len > base + size {
                        base = (arena.size as u64).saturating_sub(size);
                        base = (base / 256) * 256;
                    }
                    let a_off = if a_in {
                        arena_local_off_f32(&arena, a_id, base)
                    } else {
                        if a_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Fma staging a too large ({a_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: a_src,
                            dst_byte_off: a_dst,
                            bytes: a_len as u32,
                        });
                        ((a_dst.saturating_sub(base)) / 4) as u32
                    };
                    let b_off = if b_in {
                        arena_local_off_f32(&arena, b_id, base)
                    } else {
                        if b_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Fma staging b too large ({b_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: b_src,
                            dst_byte_off: b_dst,
                            bytes: b_len as u32,
                        });
                        ((b_dst.saturating_sub(base)) / 4) as u32
                    };
                    let c_off = if c_in {
                        arena_local_off_f32(&arena, c_id, base)
                    } else {
                        if c_len > 64 * 1024 * 1024 {
                            panic!("rlx-wgpu: Fma staging c too large ({c_len} bytes)");
                        }
                        schedule.push(Step::BufferCopy {
                            src_byte_off: c_src,
                            dst_byte_off: c_dst,
                            bytes: c_len as u32,
                        });
                        ((c_dst.saturating_sub(base)) / 4) as u32
                    };
                    let p = FmaParams {
                        n: elems,
                        a_off,
                        b_off,
                        c_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Fma { params: p });
                    let u = emit_uniform(std::mem::size_of::<FmaParams>());
                    let bg = bind_two_buf0_window(&dev.device, fk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::BatchElementwiseRegion {
                    chain,
                    num_batch_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    let n = *num_batch_inputs as usize;
                    if n == 0 || chain.len() > 32 {
                        panic!(
                            "rlx-wgpu BatchElementwiseRegion: num_batch_inputs={n} steps={}",
                            chain.len()
                        );
                    }
                    let slice_shape = rlx_ir::batch_region_slice_shape(&node.shape);
                    let slice_elems = rlx_ir::batch_region_slice_elems(&node.shape, n)
                        .expect("batch region static shape");
                    let mut win_ids: Vec<NodeId> = vec![node.id];
                    win_ids.extend(node.inputs.iter().copied());
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let chain_enc = rlx_ir::encode_chain_steps(chain);
                    let tail =
                        rlx_ir::encode_prologue_tail(*prologue, &slice_shape, *prologue_input);
                    let base_dst = arena_local_off_f32(&arena, node.id, base);
                    let use_single = rlx_ir::fk_batch_use_single_launch(n, *prologue);
                    if use_single {
                        let mut batch_input_offs = [0u32; 64];
                        for i in 0..n {
                            batch_input_offs[i] = arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                node.inputs[i],
                                &mut base,
                                &mut size,
                            );
                        }
                        let p = BatchElementwiseRegionParams {
                            slice_len: slice_elems,
                            num_batch: n as u32,
                            num_steps: chain.len() as u32,
                            base_dst_off: base_dst,
                            slice_elems,
                            batch_input_offs,
                            chain: chain_enc,
                            scalar_input_mask: *scalar_input_mask,
                            input_modulus: *input_modulus,
                        };
                        schedule.push(Step::BatchElementwiseRegion { params: p });
                        let ek = batch_elementwise_region_kernel(&dev.device);
                        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("rlx-wgpu batch region params"),
                            size: std::mem::size_of::<BatchElementwiseRegionParams>() as u64,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        let bg =
                            bind_two_buf0_window(&dev.device, ek, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let spatial = tail[0] == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW;
                        let ek = if spatial {
                            elementwise_region_spatial_kernel(&dev.device)
                        } else {
                            elementwise_region_kernel(&dev.device)
                        };
                        for i in 0..n {
                            let mut input_offs = [0u32; 16];
                            input_offs[0] = arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                node.inputs[i],
                                &mut base,
                                &mut size,
                            );
                            let p = ElementwiseRegionParams {
                                len: slice_elems,
                                num_inputs: 1,
                                num_steps: chain.len() as u32,
                                dst_off: rlx_ir::batch_region_slice_dst_off_f32(
                                    base_dst,
                                    slice_elems,
                                    i,
                                ),
                                input_offs,
                                chain: chain_enc,
                                scalar_input_mask: *scalar_input_mask,
                                prologue: tail[0],
                                out_n: tail[1],
                                out_c: tail[2],
                                out_h: tail[3],
                                out_w: tail[4],
                                prologue_input: tail[5],
                                input_modulus: *input_modulus,
                            };
                            schedule.push(Step::ElementwiseRegion { params: p });
                            let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("rlx-wgpu batch region params"),
                                size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            });
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ek,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                    }
                }
                Op::ElementwiseRegion {
                    chain,
                    num_inputs,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    prologue_input,
                } => {
                    // PLAN L2 native lowering. Encode the chain into a
                    // fixed-size u32 buffer; one uniform per region.
                    let n = *num_inputs as usize;
                    if n > 16 || chain.len() > 32 {
                        panic!(
                            "rlx-wgpu ElementwiseRegion: chain too large \
                                (inputs={n}, steps={}). Caps: 16 / 32. \
                                Use UnfuseElementwiseRegions to fall back.",
                            chain.len()
                        );
                    }
                    let mut win_ids: Vec<NodeId> = vec![node.id];
                    win_ids.extend(node.inputs.iter().copied());
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let mut input_offs = [0u32; 16];
                    for (i, &id) in node.inputs.iter().enumerate() {
                        input_offs[i] = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            id,
                            &mut base,
                            &mut size,
                        );
                    }
                    let chain_enc = rlx_ir::encode_chain_steps(chain);
                    let tail =
                        rlx_ir::encode_prologue_tail(*prologue, &node.shape, *prologue_input);
                    let p = ElementwiseRegionParams {
                        len: elems,
                        num_inputs: *num_inputs,
                        num_steps: chain.len() as u32,
                        dst_off: arena_local_off_f32(&arena, node.id, base),
                        input_offs,
                        chain: chain_enc,
                        scalar_input_mask: *scalar_input_mask,
                        prologue: tail[0],
                        out_n: tail[1],
                        out_c: tail[2],
                        out_h: tail[3],
                        out_w: tail[4],
                        prologue_input: tail[5],
                        input_modulus: *input_modulus,
                    };
                    schedule.push(Step::ElementwiseRegion { params: p });
                    let ek = if p.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW {
                        elementwise_region_spatial_kernel(&dev.device)
                    } else {
                        elementwise_region_kernel(&dev.device)
                    };
                    // STORAGE (not UNIFORM) — the WGSL params struct
                    // contains `array<u32, N>` arrays whose 4-byte
                    // stride violates uniform's 16-byte stride rule.
                    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu region params"),
                        size: std::mem::size_of::<ElementwiseRegionParams>() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg = bind_two_buf0_window(&dev.device, ek, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Reduce {
                    op: rop,
                    axes,
                    keep_dim: _,
                } => {
                    // Single-axis reduce OR contiguous multi-axis reduce.
                    // The kernel walks the input as `[outer, reduce_dim,
                    // inner]` — for contiguous axes [k..k+m], we set
                    // `reduce_dim = product(dims[k..k+m])`.
                    // Non-contiguous reductions are not yet wired (no
                    // model has hit them); transposing into contiguous
                    // form first is the future fix.
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let mut sorted = axes.clone();
                    sorted.sort_unstable();
                    let contiguous = sorted.windows(2).all(|w| w[1] == w[0] + 1);
                    if !contiguous {
                        panic!(
                            "rlx-wgpu Reduce: non-contiguous axes not yet wired \
                             (got axes={axes:?}, rank={})",
                            in_shape.len()
                        );
                    }
                    let ax_first = sorted[0];
                    let ax_last = *sorted.last().unwrap();
                    let dims_u32: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    let outer: u32 = dims_u32[..ax_first].iter().product();
                    let reduce_dim: u32 = dims_u32[ax_first..=ax_last].iter().product();
                    let inner: u32 = dims_u32[ax_last + 1..].iter().product();
                    let red_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let red_fits = arena_span_bytes(&arena, &red_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &red_ids,
                    );
                    if !red_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = ReduceParams {
                        outer,
                        reduce_dim,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        op: reduce_op_id(*rop),
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::Reduce { params: p });
                    let rk = reduce_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ReduceParams>());
                    let bg = bind_two_buf0_window(&dev.device, rk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Softmax { axis } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let last = (in_shape.len() - 1) as i32;
                    if *axis != -1 && *axis != last {
                        panic!("rlx-wgpu Softmax: only last-axis wired (got axis={axis})");
                    }
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let sm_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let sm_fits = arena_span_bytes(&arena, &sm_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &sm_ids,
                    );
                    if !sm_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let p = SoftmaxParams {
                        outer,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Softmax { params: p });
                    let sk = softmax_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SoftmaxParams>());
                    let bg = bind_two_buf0_window(&dev.device, sk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::SoftmaxCrossEntropy => {
                    // Dense / soft-label cross-entropy: logits [N,C], targets
                    // [N,C] → loss [N]. One thread per row (outer=N, inner=C).
                    // Window must cover both inputs + the [N] output slot;
                    // mirrors the LayerNorm multi-input arena-window dance.
                    let logits_id = node.inputs[0];
                    let targets_id = node.inputs[1];
                    let logits_shape = graph.node(logits_id).shape.dims();
                    let inner = logits_shape[logits_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = logits_shape
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product();
                    let outer = total / inner.max(1);

                    let sce_win = vec![node.id, logits_id, targets_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let sce_fits = arena_span_bytes(&arena, &sce_win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &sce_win,
                    );
                    if !sce_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let logits_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        logits_id,
                        &mut base,
                        &mut size,
                    );
                    let targets_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        targets_id,
                        &mut base,
                        &mut size,
                    );
                    let p = SceParams {
                        outer,
                        inner,
                        logits_off,
                        targets_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::SoftmaxCrossEntropy { params: p });
                    let sk = softmax_cross_entropy_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SceParams>());
                    let bg = bind_two_buf0_window(&dev.device, sk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::LayerNorm { axis: _, eps } | Op::RmsNorm { axis: _, eps } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let is_layer_norm = matches!(&node.op, Op::LayerNorm { .. });

                    // FRLTee fast path: if this LN is the head of a
                    // (multi-consumer Add → LN) pattern, emit one
                    // `Step::FusedResidualLnTee` that writes the sum to
                    // the eliminated Add's arena slot AND the LN result
                    // to this LN's slot. The Add itself is skipped
                    // upstream (`skip_adds`).
                    if is_layer_norm
                        && let Some(&(h_id, delta_id, gamma_id, beta_id, sum_id)) =
                            ln_to_tee.get(&node.id)
                    {
                        let gamma_is_param =
                            tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                        let gamma_bytes = arena.len_of(gamma_id) as u64;
                        let frlt_win: Vec<NodeId> =
                            if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                                vec![gamma_id, node.id, h_id, delta_id, beta_id, sum_id]
                            } else {
                                vec![node.id, h_id, delta_id, gamma_id, beta_id, sum_id]
                            };
                        let mut scratch = arena.scratch_off as u64;
                        let (mut base, mut size, param_anchor) = arena_multi_op_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            &mut schedule,
                            &mut scratch,
                            &frlt_win,
                        );
                        if !param_anchor {
                            base = arena_bind_window_covering_scratch_if_needed(
                                &arena, base, size, scratch,
                            );
                        }
                        let in_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            h_id,
                            &mut base,
                            &mut size,
                        );
                        let residual_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            delta_id,
                            &mut base,
                            &mut size,
                        );
                        let sum_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            sum_id,
                            &mut base,
                            &mut size,
                        );
                        let gamma_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            gamma_id,
                            &mut base,
                            &mut size,
                        );
                        let beta_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            beta_id,
                            &mut base,
                            &mut size,
                        );
                        let p = FusedResidualLnTeeParams {
                            outer,
                            inner,
                            in_off,
                            residual_off,
                            bias_off: 0, // FRLTee currently no-bias only
                            gamma_off,
                            beta_off,
                            sum_off,
                            ln_out_off: arena_local_off_f32(&arena, node.id, base),
                            eps_bits: eps.to_bits(),
                            has_bias: 0,
                            _p0: 0,
                        };
                        schedule.push(Step::FusedResidualLnTee { params: p });
                        let frtk = fused_residual_ln_tee_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<FusedResidualLnTeeParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, frtk, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                        continue;
                    }

                    let gamma_id = node.inputs[1];
                    // beta is the third input for LayerNorm; RmsNorm
                    // ignores it (kernel branch on `op` skips the read).
                    let beta_id = if is_layer_norm && node.inputs.len() >= 3 {
                        node.inputs[2]
                    } else {
                        // Use gamma's offset as a benign placeholder;
                        // the RmsNorm kernel branch never reads it.
                        gamma_id
                    };
                    let gamma_is_param = tensor_is_graph_param(&graph, &param_offsets, gamma_id);
                    let gamma_bytes = arena.len_of(gamma_id) as u64;
                    let ln_win: Vec<NodeId> = if gamma_is_param && gamma_bytes > ARENA_STAGE_CAP {
                        vec![gamma_id, node.id, in_id]
                    } else {
                        let mut v = vec![node.id, in_id];
                        if gamma_is_param {
                            v.push(gamma_id);
                        }
                        if is_layer_norm {
                            v.push(beta_id);
                        }
                        v
                    };
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let ln_fits = arena_span_bytes(&arena, &ln_win) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &ln_win,
                    );
                    if !ln_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let gamma_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        gamma_id,
                        &mut base,
                        &mut size,
                    );
                    let beta_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        beta_id,
                        &mut base,
                        &mut size,
                    );
                    let p = LayerNormParams {
                        outer,
                        inner,
                        in_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        gamma_off,
                        beta_off,
                        eps_bits: eps.to_bits(),
                        op: if is_layer_norm { 0 } else { 1 },
                    };
                    schedule.push(Step::LayerNorm { params: p });
                    let lk = layernorm_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNormParams>());
                    let bg = bind_two_buf0_window(&dev.device, lk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Reshape { .. } => {
                    // No-op: memory planner view-aliased this slot.
                }

                Op::Cast { .. } => {
                    // A same-dtype Cast is view-aliased by the planner (src==dst) →
                    // no-op. A dtype-changing Cast (e.g. Bool→F32 for VITS sequence
                    // masks) gets its own slot; on the f32-uniform arena every value
                    // is already f32-encoded, so the cast is a value-preserving copy.
                    // Without this the output slot keeps its (zero) init — a Bool→F32
                    // mask Cast silently produced all-zeros, killing the encoder.
                    let in_id = node.inputs[0];
                    let src = arena.offset(in_id);
                    let dst = arena.offset(node.id);
                    if src != dst {
                        let bytes = arena.len_of(in_id).min(arena.len_of(node.id));
                        schedule.push(Step::BufferCopy {
                            src_byte_off: src as u64,
                            dst_byte_off: dst as u64,
                            bytes: bytes as u32,
                        });
                    }
                }

                Op::Transpose { perm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let out_shape = node.shape.dims();
                    let rank = perm.len();
                    if rank != in_shape.len() || rank != out_shape.len() {
                        panic!("rlx-wgpu Transpose: rank mismatch");
                    }
                    let in_dims: Vec<u32> =
                        in_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    let out_dims: Vec<u32> =
                        out_shape.iter().map(|d| d.unwrap_static() as u32).collect();
                    // Input cumulative strides (row-major).
                    let mut in_strides = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
                    }
                    // For each *output* axis i, the corresponding input
                    // axis is perm[i] — its stride is in_strides[perm[i]].
                    let strides_for_out: Vec<u32> =
                        (0..rank).map(|i| in_strides[perm[i]]).collect();

                    // Build meta buffer: dims (rank u32s) + strides (rank u32s).
                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu transpose meta"),
                        size: (meta_data.len() * 4).max(4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue
                        .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta_buf);

                    // PLAN L1: precompute "bucket axis stays at out
                    // axis 0" flag from perm. When `perm[0] == 0`,
                    // active-extent scaling of `out_total` is safe.
                    let bucket_outermost = if perm[0] == 0 { 1u32 } else { 0u32 };
                    let tr_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let in_is_param = tensor_is_graph_param(&graph, &param_offsets, in_id);
                    let in_bytes = arena.len_of(in_id) as u64;
                    let (mut base, mut size) = if in_is_param && in_bytes <= max_binding {
                        arena_window_for_nodes(&dev.device, &arena, &[in_id])
                    } else if arena_span_bytes(&arena, &tr_ids) <= max_binding {
                        arena_window_for_nodes(&dev.device, &arena, &tr_ids)
                    } else {
                        arena_window_for_nodes(&dev.device, &arena, &[node.id])
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = TransposeParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off,
                        out_off,
                        bucket_outermost,
                        out_dim_0: out_dims[0],
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Transpose {
                        params: p,
                        meta_idx,
                    });
                    let tk = transpose_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<TransposeParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu transpose bg"),
                        layout: &tk.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &arena.buffer,
                                    offset: base,
                                    size: aligned_bind_size(size, base, arena.buffer.size()),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Narrow { axis, start, len } => {
                    // Part of a split-QKV pattern: the parent FMB has been
                    // (or will be) replaced by Step::MatmulQkv that writes
                    // directly into this narrow's arena slot. Skip the
                    // narrow's own dispatch.
                    if qkv_skip_narrows.contains(&node.id)
                        || packed_bshd_skip_narrows.contains(&node.id)
                    {
                        continue;
                    }
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let outer: u32 = in_shape[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = in_shape[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_in = in_shape[*axis].unwrap_static() as u32;
                    let win_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = NarrowConcatParams {
                        total: elems,
                        outer,
                        inner,
                        axis_in_size: axis_in,
                        axis_out_size: *len as u32,
                        start: *start as u32,
                        in_off,
                        out_off,
                    };
                    schedule.push(Step::Narrow { params: p });
                    let nk = narrow_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                    let bg = bind_two_buf0_window(&dev.device, nk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Concat { axis } => {
                    let out_shape = node.shape.dims();
                    let outer: u32 = out_shape[..*axis]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let inner: u32 = out_shape[*axis + 1..]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let axis_out = out_shape[*axis].unwrap_static() as u32;

                    let all_ids: Vec<NodeId> = std::iter::once(node.id)
                        .chain(node.inputs.iter().copied())
                        .collect();
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits_all = arena_span_bytes(&arena, &all_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &all_ids,
                    );
                    arena_expand_bind_window(&arena, &all_ids, &mut base, &mut size, max_binding);
                    if !fits_all && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let mut start_pos: u32 = 0;
                    for &in_id in &node.inputs {
                        let in_shape = graph.node(in_id).shape.dims();
                        let axis_in = in_shape[*axis].unwrap_static() as u32;
                        let in_total: u32 =
                            in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let _win_ids = [node.id, in_id];
                        let in_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            in_id,
                            &mut base,
                            &mut size,
                        );
                        // Recompute the output offset against the *current* base:
                        // `arena_off_in_bind_window` above may have shifted `base`
                        // to cover this input (large decoder arenas), which would
                        // otherwise leave a pre-loop `out_off` stale → writes land
                        // outside the bound window and the output stays zero.
                        let out_off = arena_local_off_f32(&arena, node.id, base);
                        let p = NarrowConcatParams {
                            total: in_total,
                            outer,
                            inner,
                            axis_in_size: axis_in,
                            axis_out_size: axis_out,
                            start: start_pos,
                            in_off,
                            out_off,
                        };
                        schedule.push(Step::Concat { params: p });
                        let cck = concat_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<NarrowConcatParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, cck, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                        start_pos += axis_in;
                    }
                }

                Op::Attention {
                    num_heads,
                    head_dim,
                    mask_kind,
                    score_scale,
                    attn_logit_softcap: _,
                } => {
                    // v5: rank-4 [B, H, S, D] inputs only. SlidingWindow
                    // synthesizes a Custom mask host-side.
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    // Accept either rank-4 [B, H, S, D] or rank-3 [B*H, S, D]
                    // (the latter is what BERT-flavored builders emit). For
                    // rank-3 we treat the leading dim as `batch * heads`,
                    // setting heads = num_heads from the Op so the kernel's
                    // (b, h) indexing folds back to the right offset.
                    let h = *num_heads as u32;
                    let hd = *head_dim as u32;
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let bhsd = geom.bhsd;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            geom.batch as u32,
                            geom.heads as u32,
                            geom.seq_q as u32,
                            geom.seq_k as u32,
                        ),
                        3 => {
                            // Two rank-3 layouts coexist:
                            //   [B, S, H·D] — transpose-elided layout
                            //   [B·H, S, D] — canonical compacted layout
                            // Distinguish by last-dim: if it equals H·D
                            // (the per-token feature width) it's [B, S, H·D];
                            // otherwise it's [B·H, S, D].
                            let last = q_shape[2].unwrap_static() as u32;
                            if last == h * hd {
                                // [B, S, H·D]: leading = B, seq = S
                                (
                                    q_shape[0].unwrap_static() as u32,
                                    h,
                                    q_shape[1].unwrap_static() as u32,
                                    k_shape[1].unwrap_static() as u32,
                                )
                            } else {
                                // [B·H, S, D]: leading must be divisible by H
                                let leading = q_shape[0].unwrap_static() as u32;
                                if !leading.is_multiple_of(h) {
                                    panic!(
                                        "rlx-wgpu Attention: rank-3 leading dim {leading} \
                                            not divisible by num_heads {h} (and last dim \
                                            {last} ≠ H·D = {})",
                                        h * hd
                                    );
                                }
                                (
                                    leading / h,
                                    h,
                                    q_shape[1].unwrap_static() as u32,
                                    k_shape[1].unwrap_static() as u32,
                                )
                            }
                        }
                        other => panic!(
                            "rlx-wgpu Attention: only rank-3 / rank-4 Q,K,V \
                                         inputs supported (got rank {other})"
                        ),
                    };
                    let scale = score_scale.unwrap_or(1.0_f32 / (hd as f32).sqrt());

                    let (mask_kind_id, mask_buf, window) = match mask_kind {
                        MaskKind::None => (0u32, None, 0u32),
                        MaskKind::Causal => (1u32, None, 0u32),
                        // 2 = binary key-padding mask (Custom: <0.5 → -inf);
                        // 4 = additive bias mask (Bias: score += mask). These
                        // are NOT interchangeable — the encoder's block-diagonal
                        // winmask is additive, so folding it into the binary
                        // path silently corrupts attention.
                        MaskKind::Custom => (2u32, None, 0u32),
                        MaskKind::Bias => (4u32, None, 0u32),
                        MaskKind::SlidingWindow(w) => (3u32, None, *w as u32),
                    };

                    // Mask address strides. For Custom masks, derive from
                    // the mask's IR shape so the kernel can broadcast a
                    // [B, S] padding mask without materializing the full
                    // [B, H, S_q, S_k] expansion. Other mask kinds use
                    // canonical [B, H, S_q, S_k] strides (the kernel's
                    // mask_partial computation is harmless when not read).
                    struct MStrides {
                        b: u32,
                        h: u32,
                        q: u32,
                        k: u32,
                    }
                    let mask_strides = if mask_kind_id == 2u32 || mask_kind_id == 4u32 {
                        let m_dims = graph.node(node.inputs[3]).shape.dims();
                        let dim = |i: usize| m_dims[i].unwrap_static() as u32;
                        match m_dims.len() {
                            2 => MStrides {
                                b: dim(1),
                                h: 0,
                                q: 0,
                                k: 1,
                            },
                            3 => MStrides {
                                b: dim(1) * dim(2),
                                h: 0,
                                q: dim(2),
                                k: 1,
                            },
                            4 => MStrides {
                                b: dim(1) * dim(2) * dim(3),
                                h: dim(2) * dim(3),
                                q: dim(3),
                                k: 1,
                            },
                            _ => MStrides {
                                b: heads * seq_q * seq_k,
                                h: seq_q * seq_k,
                                q: seq_k,
                                k: 1,
                            },
                        }
                    } else {
                        MStrides {
                            b: heads * seq_q * seq_k,
                            h: seq_q * seq_k,
                            q: seq_k,
                            k: 1,
                        }
                    };

                    let stride = |shape: &[rlx_ir::shape::Dim], seq_extent: u32| {
                        rlx_ir::strides_for_shape(shape, heads, hd, seq_extent, bhsd)
                    };
                    let packed_parent = packed_bshd_attn.get(&node.id).copied();
                    // GQA/MQA: the KV-head count is layout-independent — K's
                    // element count over B·S_k·D. Packed QKV shares Q's head
                    // count (uniform strides), so treat it as MHA.
                    let nkv: u32 = if packed_parent.is_some() {
                        heads
                    } else {
                        let k_numel: u32 =
                            k_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        (k_numel / (batch.max(1) * seq_k.max(1) * hd.max(1))).max(1)
                    };
                    let (q_b, q_h, q_s, k_b, k_h, k_s, v_b, v_h, v_s) =
                        if let Some((_parent, head_width)) = packed_parent {
                            let (batch_stride, head_stride, pack_seq) =
                                rlx_ir::packed_bshd_qkv_strides(head_width as usize, hd, seq_q);
                            (
                                batch_stride,
                                head_stride,
                                pack_seq,
                                batch_stride,
                                head_stride,
                                pack_seq,
                                batch_stride,
                                head_stride,
                                pack_seq,
                            )
                        } else {
                            let (qb, qh, qs) = stride(q_shape, seq_q);
                            // K/V carry nkv heads (GQA); strides_for_shape
                            // detects their BSHD/BHSD layout from nkv·D.
                            let (kb, kh, ks) =
                                rlx_ir::strides_for_shape(k_shape, nkv, hd, seq_k, bhsd);
                            let v_shape = graph.node(v_id).shape.dims();
                            let (vb, vh, vs) =
                                rlx_ir::strides_for_shape(v_shape, nkv, hd, seq_k, bhsd);
                            (qb, qh, qs, kb, kh, ks, vb, vh, vs)
                        };
                    let out_shape = node.shape.dims();
                    let (o_b, o_h, o_s) = stride(out_shape, seq_q);
                    let mut attn_ids = if let Some((parent, _)) = packed_parent {
                        vec![node.id, parent]
                    } else {
                        vec![node.id, q_id, k_id, v_id]
                    };
                    if mask_kind_id == 2 {
                        attn_ids.push(node.inputs[3]);
                    }
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &attn_ids,
                    );
                    if !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let (q_off, k_off, v_off) = if let Some((parent, head_width)) = packed_parent {
                        let parent_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            parent,
                            &mut base,
                            &mut size,
                        );
                        (
                            parent_off,
                            parent_off.saturating_add(head_width),
                            parent_off.saturating_add(head_width * 2),
                        )
                    } else {
                        let q_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            q_id,
                            &mut base,
                            &mut size,
                        );
                        let k_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            k_id,
                            &mut base,
                            &mut size,
                        );
                        let v_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            v_id,
                            &mut base,
                            &mut size,
                        );
                        (q_off, k_off, v_off)
                    };
                    let out_byte = arena.offset(node.id) as u64;
                    let out_len = arena.len_of(node.id) as u64;
                    let out_aliases_qkv = arena_tensors_overlap(&arena, node.id, q_id)
                        || arena_tensors_overlap(&arena, node.id, k_id)
                        || arena_tensors_overlap(&arena, node.id, v_id)
                        || packed_parent.is_some_and(|(parent, _)| {
                            arena_tensors_overlap(&arena, node.id, parent)
                        });
                    let mut kernel_out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let mut attn_scratch_copy: Option<(u64, u32)> = None;
                    if out_aliases_qkv && rlx_ir::env::flag("RLX_WGPU_DEBUG_ATTN_ALIAS") {
                        eprintln!(
                            "rlx-wgpu Attention alias: out={:?}@{}+{} q={:?}@{} k={:?}@{} v={:?}@{}",
                            node.id,
                            out_byte,
                            out_len,
                            q_id,
                            arena.offset(q_id),
                            k_id,
                            arena.offset(k_id),
                            v_id,
                            arena.offset(v_id),
                        );
                    }
                    if out_aliases_qkv {
                        let tmp_byte = scratch;
                        let tmp_aligned = out_len.div_ceil(256) * 256;
                        scratch = scratch.saturating_add(tmp_aligned);
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        } else {
                            base = arena_bind_window_covering_scratch_if_needed(
                                &arena, base, size, scratch,
                            );
                        }
                        kernel_out_off = ((tmp_byte.saturating_sub(base)) / 4) as u32;
                        attn_scratch_copy = Some((tmp_byte, out_len as u32));
                    }
                    let mask_off = if mask_kind_id == 2 {
                        arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            node.inputs[3],
                            &mut base,
                            &mut size,
                        )
                    } else {
                        0
                    };
                    let p = AttentionParams {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off,
                        k_off,
                        v_off,
                        out_off: kernel_out_off,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        // Mask strides — derive from the mask's IR shape:
                        //   [B, S]:           (mb=S,        mh=0,    mq=0,   mk=1)
                        //   [B, S_q, S_k]:    (mb=S_q·S_k,  mh=0,    mq=S_k, mk=1)
                        //   [B, H, S_q, S_k]: (mb=H·S_q·S_k mh=S_q·S_k mq=S_k mk=1)
                        // Stride 0 means the kernel broadcasts across that
                        // axis (reads the same element for every value of
                        // the index). Lets us skip the Expand pre-pass that
                        // unfuse used to emit per attention block.
                        seq_q_stride: mask_strides.q,
                        seq_k_stride: mask_strides.k,
                        mask_batch_stride: mask_strides.b,
                        mask_head_stride: mask_strides.h,
                        kv_heads: nkv,
                        _pad_mask_1: 0,
                        _pad_mask_2: 0,
                        q_batch_stride: q_b,
                        q_head_stride: q_h,
                        q_seq_stride: q_s,
                        _pad_q: 0,
                        k_batch_stride: k_b,
                        k_head_stride: k_h,
                        k_seq_stride: k_s,
                        _pad_k: 0,
                        v_batch_stride: v_b,
                        v_head_stride: v_h,
                        v_seq_stride: v_s,
                        _pad_v: 0,
                        o_batch_stride: o_b,
                        o_head_stride: o_h,
                        o_seq_stride: o_s,
                        _pad_o: 0,
                    };
                    let _ = num_heads;
                    schedule.push(Step::Attention {
                        params: p,
                        mask_buf,
                    });
                    if let Some((tmp_byte, bytes)) = attn_scratch_copy {
                        schedule.push(Step::BufferCopy {
                            src_byte_off: tmp_byte,
                            dst_byte_off: out_byte,
                            bytes,
                        });
                    }
                    let ak = attention_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AttentionParams>());
                    let bg = bind_two_buf0_window(&dev.device, ak, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::AttentionBackward {
                    num_heads,
                    head_dim,
                    mask_kind,
                    wrt,
                } => {
                    use rlx_ir::op::AttentionBwdWrt;
                    let q_id = node.inputs[0];
                    let k_id = node.inputs[1];
                    let v_id = node.inputs[2];
                    let dy_id = node.inputs[3];
                    let q_shape = graph.node(q_id).shape.dims();
                    let k_shape = graph.node(k_id).shape.dims();
                    let hd = *head_dim as u32;
                    let q_ir = graph.node(q_id).shape.clone();
                    let k_ir = graph.node(k_id).shape.clone();
                    let geom = rlx_ir::attention_geom(&q_ir, &k_ir, *num_heads, *head_dim);
                    let bhsd = geom.bhsd;
                    let (batch, heads, seq_q, seq_k) = match q_shape.len() {
                        4 => (
                            geom.batch as u32,
                            geom.heads as u32,
                            geom.seq_q as u32,
                            geom.seq_k as u32,
                        ),
                        3 => {
                            let h = q_shape[2].unwrap_static() as u32 / hd;
                            (
                                q_shape[0].unwrap_static() as u32 / h,
                                h,
                                q_shape[1].unwrap_static() as u32,
                                k_shape[1].unwrap_static() as u32,
                            )
                        }
                        other => panic!(
                            "rlx-wgpu AttentionBackward: only rank-3/4 Q,K,V (got rank {other})"
                        ),
                    };
                    let scale = 1.0_f32 / (hd as f32).sqrt();
                    let (mask_kind_id, mask_off, mask_buf, window) = match mask_kind {
                        MaskKind::None => (0u32, 0u32, None, 0u32),
                        MaskKind::Causal => (1u32, 0u32, None, 0u32),
                        MaskKind::Custom => {
                            (2u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32)
                        }
                        MaskKind::Bias => {
                            (4u32, (arena.offset(node.inputs[4]) / 4) as u32, None, 0u32)
                        }
                        MaskKind::SlidingWindow(w) => (3u32, 0u32, None, *w as u32),
                    };
                    struct MStrides {
                        b: u32,
                        h: u32,
                        q: u32,
                        k: u32,
                    }
                    let mask_strides = if mask_kind_id == 2 || mask_kind_id == 4 {
                        let m_dims = graph.node(node.inputs[4]).shape.dims();
                        let dim = |i: usize| m_dims[i].unwrap_static() as u32;
                        match m_dims.len() {
                            2 => MStrides {
                                b: dim(1),
                                h: 0,
                                q: 0,
                                k: 1,
                            },
                            3 => MStrides {
                                b: dim(1) * dim(2),
                                h: 0,
                                q: dim(2),
                                k: 1,
                            },
                            4 => MStrides {
                                b: dim(1) * dim(2) * dim(3),
                                h: dim(2) * dim(3),
                                q: dim(3),
                                k: 1,
                            },
                            _ => MStrides {
                                b: heads * seq_q * seq_k,
                                h: seq_q * seq_k,
                                q: seq_k,
                                k: 1,
                            },
                        }
                    } else {
                        MStrides {
                            b: heads * seq_q * seq_k,
                            h: seq_q * seq_k,
                            q: seq_k,
                            k: 1,
                        }
                    };
                    let stride = |shape: &[rlx_ir::shape::Dim], seq_extent: u32| {
                        rlx_ir::strides_for_shape(shape, heads, hd, seq_extent, bhsd)
                    };
                    let (q_b, q_h, q_s) = stride(q_shape, seq_q);
                    let (k_b, k_h, k_s) = stride(k_shape, seq_k);
                    let v_shape = graph.node(v_id).shape.dims();
                    let (v_b, v_h, v_s) = stride(v_shape, seq_k);
                    let out_shape = node.shape.dims();
                    let out_seq = match wrt {
                        AttentionBwdWrt::Query => seq_q,
                        AttentionBwdWrt::Key | AttentionBwdWrt::Value => seq_k,
                    };
                    let (o_b, o_h, o_s) = stride(out_shape, out_seq);
                    let wrt_id = match wrt {
                        AttentionBwdWrt::Query => 0u32,
                        AttentionBwdWrt::Key => 1u32,
                        AttentionBwdWrt::Value => 2u32,
                    };
                    let p = AttentionBwdParams {
                        batch,
                        heads,
                        seq_q,
                        seq_k,
                        head_dim: hd,
                        q_off: (arena.offset(q_id) / 4) as u32,
                        k_off: (arena.offset(k_id) / 4) as u32,
                        v_off: (arena.offset(v_id) / 4) as u32,
                        dy_off: (arena.offset(dy_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        mask_off,
                        mask_kind: mask_kind_id,
                        scale_bits: scale.to_bits(),
                        window,
                        wrt: wrt_id,
                        seq_q_stride: mask_strides.q,
                        seq_k_stride: mask_strides.k,
                        mask_batch_stride: mask_strides.b,
                        mask_head_stride: mask_strides.h,
                        _pad_mask_0: 0,
                        _pad_mask_1: 0,
                        _pad_mask_2: 0,
                        q_batch_stride: q_b,
                        q_head_stride: q_h,
                        q_seq_stride: q_s,
                        _pad_q: 0,
                        k_batch_stride: k_b,
                        k_head_stride: k_h,
                        k_seq_stride: k_s,
                        _pad_k: 0,
                        v_batch_stride: v_b,
                        v_head_stride: v_h,
                        v_seq_stride: v_s,
                        _pad_v: 0,
                        o_batch_stride: o_b,
                        o_head_stride: o_h,
                        o_seq_stride: o_s,
                        _pad_o: 0,
                    };
                    schedule.push(Step::AttentionBackward {
                        params: p,
                        mask_buf,
                    });
                    let ak = attention_bwd_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<AttentionBwdParams>());
                    let bg = bind_op_output_window(&dev.device, ak, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Rope {
                    head_dim,
                    n_rot,
                    style,
                } => {
                    let x_id = node.inputs[0];
                    let cos_id = node.inputs[1];
                    let sin_id = node.inputs[2];
                    let x_shape = graph.node(x_id).shape.dims();
                    let last = x_shape.last().map(|d| d.unwrap_static()).unwrap_or(0);
                    if !last.is_multiple_of(*head_dim) {
                        panic!(
                            "rlx-wgpu Rope: last_dim ({last}) must be a multiple \
                                of head_dim ({head_dim})"
                        );
                    }
                    if head_dim % 2 != 0 {
                        panic!("rlx-wgpu Rope: head_dim must be even");
                    }
                    let total: u32 = x_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let seq = x_shape[x_shape.len() - 2].unwrap_static() as u32;
                    // PLAN L1: derive batch from total / seq / last_dim
                    // (= product of leading dims). `seq_stride` stays at
                    // full seq for buffer offset math; `seq` becomes the
                    // runtime-scaled loop bound.
                    let batch = total / (seq * last as u32).max(1);
                    let cos_is_param = tensor_is_graph_param(&graph, &param_offsets, cos_id);
                    let cos_bytes = arena.len_of(cos_id) as u64;
                    let rope_win: Vec<NodeId> = if cos_is_param && cos_bytes > ARENA_STAGE_CAP {
                        vec![cos_id, sin_id, node.id, x_id]
                    } else {
                        vec![node.id, x_id, cos_id, sin_id]
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &rope_win,
                    );
                    if !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        x_id,
                        &mut base,
                        &mut size,
                    );
                    let cos_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        cos_id,
                        &mut base,
                        &mut size,
                    );
                    let sin_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        sin_id,
                        &mut base,
                        &mut size,
                    );
                    let p = RopeParams {
                        n_total: total,
                        seq,
                        head_dim: *head_dim as u32,
                        half: (*head_dim / 2) as u32,
                        in_off,
                        cos_off,
                        sin_off,
                        out_off: arena_local_off_f32(&arena, node.id, base),
                        last_dim: last as u32,
                        batch,
                        seq_stride: seq,
                        style: match style {
                            rlx_ir::op::RopeStyle::NeoX => 0,
                            rlx_ir::op::RopeStyle::GptJ => 1,
                        },
                        // Partial rotary: rotate only n_rot dims (Gemma 4 global
                        // layers use n_rot < head_dim). Equals half for full rope.
                        rot_half: (*n_rot / 2) as u32,
                    };
                    schedule.push(Step::Rope { params: p });
                    let rk = rope_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<RopeParams>());
                    let bg = bind_two_buf0_window(&dev.device, rk, &arena.buffer, base, size, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Expand { target_shape } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let in_rank = in_shape.len();
                    let rank = target_shape.len();
                    if in_rank > rank {
                        panic!(
                            "rlx-wgpu Expand: rank mismatch \
                                (in_rank={in_rank}, target_rank={rank})"
                        );
                    }
                    // Implicit leading 1s when input rank < target rank (e.g.
                    // scalar → vector from `LegalizeBroadcast`).
                    let pad = rank.saturating_sub(in_rank);
                    let out_dims: Vec<u32> = target_shape.iter().map(|&d| d as u32).collect();
                    let in_dims: Vec<u32> = (0..rank)
                        .map(|i| {
                            if i < pad {
                                1
                            } else {
                                in_shape[i - pad].unwrap_static() as u32
                            }
                        })
                        .collect();
                    // Cumulative input strides (row-major). When the
                    // input dim is 1 but target dim > 1, that axis
                    // broadcasts → stride = 0.
                    let mut in_strides_row = vec![1u32; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides_row[i] = in_strides_row[i + 1] * in_dims[i + 1];
                    }
                    let strides_for_out: Vec<u32> = (0..rank)
                        .map(|i| {
                            if in_dims[i] == 1 && out_dims[i] != 1 {
                                0
                            } else {
                                in_strides_row[i]
                            }
                        })
                        .collect();

                    let mut meta_data: Vec<u32> = Vec::with_capacity(rank * 2);
                    meta_data.extend_from_slice(&out_dims);
                    meta_data.extend_from_slice(&strides_for_out);
                    let meta_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("rlx-wgpu expand meta"),
                        size: (meta_data.len() * 4).max(4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue
                        .write_buffer(&meta_buf, 0, bytemuck::cast_slice(&meta_data));
                    let meta_idx = meta_buffers.len();
                    meta_buffers.push(meta_buf);

                    // PLAN L1: bucket axis stays at out axis 0 iff the
                    // expand at axis 0 isn't a broadcast (in_dims[0]
                    // matches out_dims[0]). When broadcast at axis 0
                    // (in_dims[0]==1, out_dims[0]>1), the bucket-axis
                    // contract doesn't apply — fall back to full extent.
                    let bucket_outermost = if in_dims[0] == out_dims[0] {
                        1u32
                    } else {
                        0u32
                    };
                    let exp_ids = [node.id, in_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let exp_fits = arena_span_bytes(&arena, &exp_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &exp_ids,
                    );
                    if !exp_fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let p = ExpandParams {
                        rank: rank as u32,
                        out_total: elems,
                        in_off,
                        out_off,
                        bucket_outermost,
                        out_dim_0: out_dims[0],
                        _p2: 0,
                        _p3: 0,
                    };
                    schedule.push(Step::Expand {
                        params: p,
                        meta_idx,
                    });
                    let ek = expand_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<ExpandParams>());
                    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rlx-wgpu expand bg"),
                        layout: &ek.bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &arena.buffer,
                                    offset: base,
                                    size: aligned_bind_size(size, base, arena.buffer.size()),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: u.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: meta_buffers[meta_idx].as_entire_binding(),
                            },
                        ],
                    });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }

                Op::Gather { axis } => {
                    let table_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    let table_is_param = tensor_is_graph_param(&graph, &param_offsets, table_id);
                    let table_bytes = arena.len_of(table_id) as u64;
                    // Split-binding path: on >4 GiB arenas the embedding output and
                    // its index can lie more than one ≤4 GiB binding window away
                    // from the (multi-GiB) table. A single arena binding then can't
                    // cover both — the kernel writes the output OUTSIDE the bound
                    // window (the write is dropped/clamped) and idx-staging would
                    // even overwrite part of the table. Route axis-0 embedding
                    // gathers through a host segment with separate table / idx
                    // windows and a dedicated output buffer (copied back).
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let gather_needs_split = *axis == 0
                        && arena_whole_arena_bind(&arena, max_binding).is_none()
                        && arena_span_bytes(&arena, &[table_id, idx_id, node.id]) > max_binding;
                    if gather_needs_split {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let vocab = table_shape[0].unwrap_static() as u32;
                        let dim: u32 = table_shape[1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let n_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let table_w = arena.len_of(table_id) as u64;
                        assert!(
                            table_w <= max_binding,
                            "rlx-wgpu gather_split: embedding table {table_w} bytes exceeds \
                             max_storage_buffer_binding_size {max_binding}"
                        );
                        schedule.push(Step::GatherSplit {
                            n_out: elems,
                            n_idx,
                            dim,
                            vocab,
                            table_byte_off: arena.offset(table_id) as u64,
                            idx_byte_off: arena.offset(idx_id) as u64,
                            out_byte_off: arena.offset(node.id) as u64,
                        });
                        // Host segment: like DequantMatmulGguf, it builds its own
                        // bind group at exec time, so do NOT push a uniform/bind
                        // group lane (those are indexed by GPU-step count).
                        continue;
                    }
                    let gather_win: Vec<NodeId> = if table_is_param && table_bytes > ARENA_STAGE_CAP
                    {
                        vec![table_id, node.id, idx_id]
                    } else {
                        vec![node.id, idx_id, table_id]
                    };
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, table_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &gather_win,
                    );
                    if !table_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off =
                        if table_anchor && arena_tensor_in_window(&arena, table_id, base, size) {
                            arena_local_off_f32(&arena, table_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                table_id,
                                &mut base,
                                &mut size,
                            )
                        };
                    let idx_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        idx_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_local_off_f32(&arena, node.id, base);
                    if *axis == 0 {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let vocab = table_shape[0].unwrap_static() as u32;
                        let dim: u32 = table_shape[1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let n_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let p = GatherParams {
                            n_out: elems,
                            n_idx,
                            dim,
                            vocab,
                            in_off,
                            idx_off,
                            out_off,
                            _p0: 0,
                        };
                        schedule.push(Step::Gather { params: p });
                        let gk = gather_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, gk, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let table_shape = graph.node(table_id).shape.dims();
                        let idx_shape = graph.node(idx_id).shape.dims();
                        let outer: u32 = table_shape[..*axis]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let trailing: u32 = table_shape[*axis + 1..]
                            .iter()
                            .map(|d| d.unwrap_static() as u32)
                            .product::<u32>()
                            .max(1);
                        let axis_dim = table_shape[*axis].unwrap_static() as u32;
                        let num_idx: u32 =
                            idx_shape.iter().map(|d| d.unwrap_static() as u32).product();
                        let total = outer * num_idx * trailing;
                        let p = GatherAxisParams {
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off: in_off,
                            idx_off,
                            out_off,
                        };
                        schedule.push(Step::GatherAxis { params: p });
                        let gk = gather_axis_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GatherAxisParams>());
                        let bg =
                            bind_two_buf0_window(&dev.device, gk, &arena.buffer, base, size, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }

                Op::FusedMatMulBiasAct { activation } => {
                    // Inputs: [x, w, bias]. We require 2D × 2D or
                    // [..,M,K] × [K,N] (broadcast bias). Bias is shape [N].
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let bias_id = node.inputs[2];
                    let a_shape = graph.node(a_id).shape.dims();
                    let b_shape = graph.node(b_id).shape.dims();
                    let out_shape = node.shape.dims();
                    let (m, k, n) =
                        if a_shape.len() == 2 && b_shape.len() == 2 && out_shape.len() == 2 {
                            (
                                a_shape[0].unwrap_static() as u32,
                                a_shape[1].unwrap_static() as u32,
                                b_shape[1].unwrap_static() as u32,
                            )
                        } else if a_shape.len() >= 2
                            && b_shape.len() == 2
                            && out_shape.len() == a_shape.len()
                        {
                            let leading: usize = a_shape[..a_shape.len() - 2]
                                .iter()
                                .map(|d| d.unwrap_static())
                                .product();
                            let m_inner = a_shape[a_shape.len() - 2].unwrap_static();
                            let k_inner = a_shape[a_shape.len() - 1].unwrap_static();
                            let n_inner = b_shape[1].unwrap_static();
                            ((leading * m_inner) as u32, k_inner as u32, n_inner as u32)
                        } else {
                            panic!(
                                "rlx-wgpu FusedMatMulBiasAct: unsupported shapes \
                                a={a_shape:?} b={b_shape:?}"
                            );
                        };
                    let act_id = match activation {
                        None => 0xFFFFu32,
                        Some(a) => activation_op_id(*a),
                    };
                    let b_is_param = tensor_is_graph_param(&graph, &param_offsets, b_id);
                    let b_bytes = arena.len_of(b_id) as u64;
                    let mut compute_precision = derive_matmul_compute(
                        &dev.device,
                        &graph,
                        &coop_f16_vk_mirror_acts,
                        a_id,
                        b_id,
                        m,
                        k,
                        n,
                    );
                    if b_is_param
                        && b_bytes > ARENA_STAGE_CAP
                        && arena.param_fits_f16_mirror(b_id)
                        && !rlx_ir::env::flag("RLX_WGPU_NO_F16_MIRROR")
                    {
                        compute_precision = MatmulCompute::F16;
                    }

                    // Split-QKV pattern: matmul writes Q/K/V directly into
                    // 3 separate output buffers, eliminating the 3 Narrow
                    // dispatches that would otherwise follow.
                    let mqk_eligible = act_id == 0xFFFFu32
                        && matches!(
                            compute_precision,
                            MatmulCompute::F32 | MatmulCompute::CoopF32 | MatmulCompute::CoopF16Vk
                        );
                    if mqk_eligible && let Some(&(q_id, k_id_n, v_id)) = qkv_split.get(&node.id) {
                        let head_width = n / 3;
                        let qkv_kind = match compute_precision {
                            MatmulCompute::CoopF16Vk => MatmulQkvKind::CoopF16Vk,
                            MatmulCompute::CoopF32 => MatmulQkvKind::CoopF32,
                            _ => MatmulQkvKind::F32,
                        };
                        let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param);
                        let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            q_id,
                            a_id,
                            b_id,
                            b_in_arena,
                        );
                        let mut scratch = arena.scratch_off as u64;
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        }
                        if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                            assert!(
                                param_anchor && arena_tensor_in_window(&arena, b_id, base, size),
                                "rlx-wgpu FusedMatMul QKV: large param B {:?} not in bind window",
                                b_id,
                            );
                        }
                        let a_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            a_id,
                            &mut base,
                            &mut size,
                        );
                        let q_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            q_id,
                            &mut base,
                            &mut size,
                        );
                        let k_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            k_id_n,
                            &mut base,
                            &mut size,
                        );
                        let v_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            v_id,
                            &mut base,
                            &mut size,
                        );
                        let bias_off = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            bias_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_f32 = if !b_in_arena {
                            (arena.offset(b_id) / 4) as u32
                        } else if b_is_param
                            && b_bytes > ARENA_STAGE_CAP
                            && arena_tensor_in_window(&arena, b_id, base, size)
                        {
                            arena_local_off_f32(&arena, b_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                b_id,
                                &mut base,
                                &mut size,
                            )
                        };
                        let b_off_global = (arena.offset(b_id) / 4) as u32;
                        maybe_push_coop_f16_vk_casts(
                            &graph,
                            a_id,
                            b_id,
                            &coop_f16_vk_mirror_acts,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mm_cast,
                            compute_precision,
                            a_off,
                            m,
                            k,
                            1,
                            if qkv_kind == MatmulQkvKind::CoopF16Vk {
                                b_off_global
                            } else {
                                b_off_f32
                            },
                            n,
                        );
                        let p = MatmulQkvParams {
                            m,
                            k,
                            n,
                            a_off,
                            b_off: if qkv_kind == MatmulQkvKind::CoopF16Vk {
                                b_off_global
                            } else {
                                b_off_f32
                            },
                            q_off,
                            k_off,
                            v_off,
                            head_width,
                            has_bias: 1,
                            bias_off,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                        };
                        schedule.push(Step::MatmulQkv {
                            params: p,
                            kind: qkv_kind,
                        });
                        register_coop_f16_vk_b_param(
                            &mut coop_f16_b_param,
                            &param_offsets,
                            b_id,
                            p.b_off,
                            match qkv_kind {
                                MatmulQkvKind::CoopF16Vk => MatmulCompute::CoopF16Vk,
                                MatmulQkvKind::CoopF32 => MatmulCompute::CoopF32,
                                MatmulQkvKind::F32 => MatmulCompute::F32,
                            },
                        );
                        let u = emit_uniform(std::mem::size_of::<MatmulQkvParams>());
                        let bg = match qkv_kind {
                            MatmulQkvKind::CoopF16Vk => {
                                let mqk = matmul_qkv_coop_f16_vk_kernel(&dev.device).expect(
                                    "coop f16 matmul_qkv kernel: feature was checked but missing",
                                );
                                let (bg, b_off_adj) = build_matmul_qkv_coop_f16_vk_bind_group(
                                    &dev.device,
                                    mqk,
                                    &arena,
                                    base,
                                    size,
                                    &u,
                                    k,
                                    n,
                                    p.b_off,
                                );
                                if let Some(Step::MatmulQkv { params, .. }) = schedule.last_mut() {
                                    params.b_off = b_off_adj;
                                }
                                bg
                            }
                            MatmulQkvKind::CoopF32 => bind_two_buf0_window(
                                &dev.device,
                                matmul_qkv_coop_f32_kernel(&dev.device).expect(
                                    "coop matmul_qkv kernel: hardware feature was checked but kernel missing",
                                ),
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            ),
                            MatmulQkvKind::F32 => bind_two_buf0_window(
                                &dev.device,
                                matmul_qkv_kernel(&dev.device),
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            ),
                        };
                        uniforms.push(u);
                        bind_groups.push(bg);
                        if qkv_kind == MatmulQkvKind::CoopF16Vk {
                            coop_f16_vk_wide_bind_groups.insert(
                                schedule.len() - 1,
                                bind_two_buf0_window(
                                    &dev.device,
                                    matmul_qkv_kernel(&dev.device),
                                    &arena.buffer,
                                    base,
                                    size,
                                    &uniforms[uniforms.len() - 1],
                                ),
                            );
                        }
                    } else {
                        let b_in_arena = !matmul_b_from_f16(compute_precision, b_is_param);
                        let (mut base, mut size, param_anchor) = arena_matmul_bind_window(
                            &dev.device,
                            &arena,
                            &graph,
                            &param_offsets,
                            node.id,
                            a_id,
                            b_id,
                            b_in_arena,
                        );
                        let mut scratch = arena.scratch_off as u64;
                        if param_anchor {
                            arena_ensure_scratch_in_window(&mut scratch, base, size);
                        }
                        if b_is_param && b_bytes > ARENA_STAGE_CAP && b_in_arena {
                            assert!(
                                param_anchor && arena_tensor_in_window(&arena, b_id, base, size),
                                "rlx-wgpu FusedMatMul: large param B {:?} not in bind window",
                                b_id,
                            );
                        }
                        let a_off_f32 = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            a_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_f32 = if !b_in_arena {
                            (arena.offset(b_id) / 4) as u32
                        } else if b_is_param
                            && b_bytes > ARENA_STAGE_CAP
                            && arena_tensor_in_window(&arena, b_id, base, size)
                        {
                            arena_local_off_f32(&arena, b_id, base)
                        } else {
                            arena_off_in_bind_window(
                                &graph,
                                &param_offsets,
                                &dev.device,
                                &arena,
                                &mut schedule,
                                &mut scratch,
                                b_id,
                                &mut base,
                                &mut size,
                            )
                        };
                        let bias_off_f32 = arena_off_in_bind_window(
                            &graph,
                            &param_offsets,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut scratch,
                            bias_id,
                            &mut base,
                            &mut size,
                        );
                        let b_off_global = (arena.offset(b_id) / 4) as u32;
                        let b_off_bind = if b_is_param
                            && matches!(
                                compute_precision,
                                MatmulCompute::Coop16
                                    | MatmulCompute::CoopF16Vk
                                    | MatmulCompute::F16
                            ) {
                            b_off_global
                        } else {
                            b_off_f32
                        };
                        maybe_push_coop_f16_vk_casts(
                            &graph,
                            a_id,
                            b_id,
                            &coop_f16_vk_mirror_acts,
                            &dev.device,
                            &arena,
                            &mut schedule,
                            &mut uniforms,
                            &mut bind_groups,
                            &mm_cast,
                            compute_precision,
                            a_off_f32,
                            m,
                            k,
                            1,
                            b_off_bind,
                            n,
                        );
                        schedule.push(Step::Matmul {
                            m,
                            k,
                            n,
                            batch: 1,
                            a_batch_stride: 0,
                            b_batch_stride: 0,
                            c_batch_stride: 0,
                            a_off_f32,
                            b_off_f32,
                            c_off_f32: arena_local_off_f32(&arena, node.id, base),
                            has_bias: 1,
                            bias_off_f32,
                            act_id,
                            b_is_param,
                            compute_precision,
                        });
                        register_coop_f16_vk_b_param(
                            &mut coop_f16_b_param,
                            &param_offsets,
                            b_id,
                            b_off_bind,
                            compute_precision,
                        );
                        let u = emit_uniform(std::mem::size_of::<MatmulParams>());
                        let (bg, b_off_adj) = build_matmul_bind_group(
                            &dev.device,
                            mm_k,
                            mm_w,
                            &mm_f16w,
                            &mm_f16c,
                            &mm_coop,
                            &mm_coop_f32,
                            &arena,
                            base,
                            size,
                            &u,
                            b_is_param,
                            compute_precision,
                            k,
                            n,
                            1,
                            b_off_bind,
                            0,
                        );
                        if let Some(Step::Matmul { b_off_f32, .. }) = schedule.last_mut() {
                            *b_off_f32 = b_off_adj;
                        }
                        uniforms.push(u);
                        bind_groups.push(bg);
                        if compute_precision == MatmulCompute::CoopF16Vk {
                            coop_f16_vk_wide_bind_groups.insert(
                                schedule.len() - 1,
                                bind_two_buf0_window(
                                    &dev.device,
                                    mm_w_active_compile,
                                    &arena.buffer,
                                    base,
                                    size,
                                    &uniforms[uniforms.len() - 1],
                                ),
                            );
                        }
                    }
                }

                Op::DotGeneral { .. } => {
                    // Should be unreachable: DotGeneral is decomposed into
                    // MatMul + Transpose + Reshape by the unfusion pass
                    // before memory planning. If we hit this arm, the
                    // unfusion pass has a gap.
                    panic!(
                        "rlx-wgpu DotGeneral: leaked past unfusion pass — \
                            check unfuse.rs::expand_dot_general for missing patterns"
                    );
                }

                Op::Sample {
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    // Greedy fast-path: temperature == 1.0 with no top_k/top_p
                    // is an argmax — same numeric result, much cheaper kernel.
                    let is_greedy = *top_k == 0
                        && (*top_p - 1.0).abs() < 1e-6
                        && (*temperature - 1.0).abs() < 1e-6;
                    if is_greedy {
                        let p = ArgmaxParams {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                        };
                        schedule.push(Step::Argmax { params: p });
                        let amk = argmax_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<ArgmaxParams>());
                        let bg = bind_op_output_window(&dev.device, amk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let p = SampleParams {
                            outer,
                            inner,
                            in_off: (arena.offset(in_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            top_k: *top_k as u32,
                            top_p_bits: top_p.to_bits(),
                            temp_bits: temperature.to_bits(),
                            seed_lo: *seed as u32,
                            seed_hi: (*seed >> 32) as u32,
                            _p0: 0,
                            _p1: 0,
                            _p2: 0,
                        };
                        schedule.push(Step::Sample { params: p });
                        let sk = sample_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<SampleParams>());
                        let bg = bind_op_output_window(&dev.device, sk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }

                Op::Pool {
                    kind,
                    kernel_size,
                    stride,
                    padding,
                } => {
                    let in_shape = graph.node(node.inputs[0]).shape.dims();
                    let out_shape = node.shape.dims();
                    let op_id: u32 = match kind {
                        ReduceOp::Sum => 0,
                        ReduceOp::Mean => 1,
                        ReduceOp::Max => 2,
                        ReduceOp::Min => 3,
                        ReduceOp::Prod => 4,
                    };
                    match (kernel_size.len(), in_shape.len(), out_shape.len()) {
                        (1, 3, 3) => {
                            let p = Pool1dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                l: in_shape[2].unwrap_static() as u32,
                                l_out: out_shape[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: stride.first().copied().unwrap_or(1) as u32,
                                pl: padding.first().copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                                _p3: 0,
                                _p4: 0,
                                _p5: 0,
                            };
                            schedule.push(Step::Pool1d { params: p });
                            let pk = pool1d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool1dParams>());
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (2, 4, 4) => {
                            let p = Pool2dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                h: in_shape[2].unwrap_static() as u32,
                                w: in_shape[3].unwrap_static() as u32,
                                h_out: out_shape[2].unwrap_static() as u32,
                                w_out: out_shape[3].unwrap_static() as u32,
                                kh: kernel_size[0] as u32,
                                kw: kernel_size[1] as u32,
                                sh: stride.first().copied().unwrap_or(1) as u32,
                                sw: stride.get(1).copied().unwrap_or(1) as u32,
                                ph: padding.first().copied().unwrap_or(0) as u32,
                                pw: padding.get(1).copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::Pool2d { params: p });
                            let pk = pool2d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool2dParams>());
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (3, 5, 5) => {
                            let p = Pool3dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c: in_shape[1].unwrap_static() as u32,
                                d: in_shape[2].unwrap_static() as u32,
                                h: in_shape[3].unwrap_static() as u32,
                                w: in_shape[4].unwrap_static() as u32,
                                d_out: out_shape[2].unwrap_static() as u32,
                                h_out: out_shape[3].unwrap_static() as u32,
                                w_out: out_shape[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: stride.first().copied().unwrap_or(1) as u32,
                                sh: stride.get(1).copied().unwrap_or(1) as u32,
                                sw: stride.get(2).copied().unwrap_or(1) as u32,
                                pd: padding.first().copied().unwrap_or(0) as u32,
                                ph: padding.get(1).copied().unwrap_or(0) as u32,
                                pw: padding.get(2).copied().unwrap_or(0) as u32,
                                op: op_id,
                                in_off: (arena.offset(node.inputs[0]) / 4) as u32,
                                out_off: (arena.offset(node.id) / 4) as u32,
                                _p0: 0,
                                _p1: 0,
                            };
                            schedule.push(Step::Pool3d { params: p });
                            let pk = pool3d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Pool3dParams>());
                            let bg = bind_op_output_window(&dev.device, pk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (k, n, m) => panic!(
                            "rlx-wgpu Pool: kernel-rank {k} with input rank {n} / \
                             output rank {m} not supported (use 1D/2D/3D NCHW)"
                        ),
                    }
                }

                Op::Conv {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let win_ids = [node.id, in_id, w_id];
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    let fits = arena_span_bytes(&arena, &win_ids) <= max_binding;
                    let mut scratch = arena.scratch_off as u64;
                    let (mut base, mut size, param_anchor) = arena_multi_op_window(
                        &dev.device,
                        &arena,
                        &graph,
                        &param_offsets,
                        &mut schedule,
                        &mut scratch,
                        &win_ids,
                    );
                    arena_expand_bind_window(&arena, &win_ids, &mut base, &mut size, max_binding);
                    if !fits && !param_anchor {
                        base = arena_bind_window_covering_scratch_if_needed(
                            &arena, base, size, scratch,
                        );
                    }
                    let in_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        in_id,
                        &mut base,
                        &mut size,
                    );
                    let w_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        w_id,
                        &mut base,
                        &mut size,
                    );
                    let out_off = arena_off_in_bind_window(
                        &graph,
                        &param_offsets,
                        &dev.device,
                        &arena,
                        &mut schedule,
                        &mut scratch,
                        node.id,
                        &mut base,
                        &mut size,
                    );
                    let in_shape = graph.node(in_id).shape.dims();
                    let w_shape = graph.node(w_id).shape.dims();
                    let out_shape = node.shape.dims();
                    let s = |i: usize| stride.get(i).copied().unwrap_or(1) as u32;
                    let p = |i: usize| padding.get(i).copied().unwrap_or(0) as u32;
                    let d = |i: usize| dilation.get(i).copied().unwrap_or(1) as u32;
                    match (
                        kernel_size.len(),
                        in_shape.len(),
                        w_shape.len(),
                        out_shape.len(),
                    ) {
                        (1, 3, 3, 3) => {
                            let p1 = Conv1dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                l: in_shape[2].unwrap_static() as u32,
                                l_out: out_shape[2].unwrap_static() as u32,
                                kl: kernel_size[0] as u32,
                                sl: s(0),
                                pl: p(0),
                                dl: d(0),
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::Conv1d { params: p1 });
                            let ck = conv1d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv1dParams>());
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ck,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (2, 4, 4, 4) => {
                            let h_in = in_shape[2].unwrap_static() as u32;
                            let w_in = in_shape[3].unwrap_static() as u32;
                            // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis
                            // and the length in W (`[N,C,1,L]`, kernel `[k,1]`). The 2D
                            // conv kernel would run the k-tap kernel over the singleton
                            // H axis. `[N,C,1,L]` and `[N,C,L,1]` share row-major layout,
                            // so relabel the length onto H (no data copy) — matching the
                            // CPU/MLX 1D paths and onnxruntime.
                            let one_d = h_in == 1
                                && w_in > 1
                                && kernel_size[0] > 1
                                && kernel_size.get(1).copied().unwrap_or(1) == 1;
                            let (h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw) = if one_d {
                                (
                                    w_in,
                                    1,
                                    out_shape[3].unwrap_static() as u32,
                                    1,
                                    kernel_size[0] as u32,
                                    1,
                                    s(0),
                                    1,
                                    p(0),
                                    0,
                                    d(0),
                                    1,
                                )
                            } else {
                                (
                                    h_in,
                                    w_in,
                                    out_shape[2].unwrap_static() as u32,
                                    out_shape[3].unwrap_static() as u32,
                                    kernel_size[0] as u32,
                                    kernel_size[1] as u32,
                                    s(0),
                                    s(1),
                                    p(0),
                                    p(1),
                                    d(0),
                                    d(1),
                                )
                            };
                            let p2 = Conv2dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                h,
                                w,
                                h_out,
                                w_out,
                                kh,
                                kw,
                                sh,
                                sw,
                                ph,
                                pw,
                                dh,
                                dw,
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                            };
                            // Two accelerated conv paths for the `one_d`
                            // (kw==1) vocoder convs:
                            //   • conv1d_tiled (default): 2D register-blocked
                            //     direct conv — reuses input across the output-
                            //     channel tile. Drop-in for the direct conv
                            //     (same params + bind window); no scratch.
                            //   • im2col + GEMM (opt-in, RLX_WGPU_CONV_IM2COL):
                            //     materialize `col` in scratch, run the tiled
                            //     f32 GEMM. Wins only for GEMM-friendly shapes;
                            //     needs whole-arena bind + reserved scratch.
                            // Everything else falls back to the direct conv.
                            let spatial = (p2.h_out as u64) * (p2.w_out as u64);
                            let k_total = (p2.c_in as u64) * (p2.kh as u64) * (p2.kw as u64);
                            let col_bytes = k_total.saturating_mul(spatial).saturating_mul(4);
                            let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                            let whole = arena_whole_arena_bind(&arena, max_binding);
                            let im2col_opt_in = rlx_ir::env::flag("RLX_WGPU_CONV_IM2COL");
                            let use_im2col = im2col_opt_in
                                && p2.groups == 1
                                && p2.n == 1
                                && (p2.kh as u64) * (p2.kw as u64) >= 2
                                && spatial >= im2col_min_spatial()
                                && k_total >= im2col_min_k()
                                && (p2.c_out as u64) >= im2col_min_cout()
                                && col_bytes <= CONV_IM2COL_MAX_COL_BYTES
                                && col_bytes <= arena.scratch_bytes as u64
                                && whole.is_some();
                            let use_tiled = !use_im2col
                                && one_d
                                && p2.kw == 1
                                && p2.w == 1
                                && p2.w_out == 1
                                && p2.groups == 1
                                && p2.n == 1
                                && spatial >= conv_tiled_min_spatial()
                                && !rlx_ir::env::flag("RLX_WGPU_NO_TILED_CONV");
                            if use_im2col {
                                let (base_w, size_w) = whole.expect("whole-arena bind");
                                // col lives at the reserved scratch tail (free of
                                // live data); consumed immediately by the GEMM.
                                let col_word_off = (arena.scratch_off / 4) as u32;
                                let im_params = Im2Col2dParams {
                                    c_in: p2.c_in,
                                    h: p2.h,
                                    w: p2.w,
                                    h_out: p2.h_out,
                                    w_out: p2.w_out,
                                    kh: p2.kh,
                                    kw: p2.kw,
                                    sh: p2.sh,
                                    sw: p2.sw,
                                    ph: p2.ph,
                                    pw: p2.pw,
                                    dh: p2.dh,
                                    dw: p2.dw,
                                    in_off: (arena.offset(in_id) / 4) as u32,
                                    col_off: col_word_off,
                                    k_total: k_total as u32,
                                    spatial: spatial as u32,
                                    _p0: 0,
                                    _p1: 0,
                                    _p2: 0,
                                };
                                schedule.push(Step::Im2ColGpu { params: im_params });
                                let imk = im2col2d_kernel(&dev.device);
                                let u_im = emit_uniform(std::mem::size_of::<Im2Col2dParams>());
                                // Static params — write once at compile (the
                                // active-extent rewrite pass is a no-op here).
                                dev.queue
                                    .write_buffer(&u_im, 0, bytemuck::bytes_of(&im_params));
                                let bg_im = bind_two_buf0_window(
                                    &dev.device,
                                    imk,
                                    &arena.buffer,
                                    base_w,
                                    size_w,
                                    &u_im,
                                );
                                uniforms.push(u_im);
                                bind_groups.push(bg_im);
                                // GEMM: weight[c_out, K] @ col[K, spatial]
                                //   → out[c_out, spatial]  (NCHW, N==1).
                                let m = p2.c_out;
                                let kk = k_total as u32;
                                let nn2 = spatial as u32;
                                schedule.push(Step::Matmul {
                                    m,
                                    k: kk,
                                    n: nn2,
                                    batch: 1,
                                    a_batch_stride: m.saturating_mul(kk),
                                    b_batch_stride: kk.saturating_mul(nn2),
                                    c_batch_stride: m.saturating_mul(nn2),
                                    a_off_f32: (arena.offset(w_id) / 4) as u32,
                                    b_off_f32: col_word_off,
                                    c_off_f32: (arena.offset(node.id) / 4) as u32,
                                    has_bias: 0,
                                    bias_off_f32: 0,
                                    act_id: 0xFFFF,
                                    b_is_param: false,
                                    compute_precision: MatmulCompute::F32,
                                });
                                let u_mm = emit_uniform(std::mem::size_of::<MatmulParams>());
                                let bg_mm = bind_two_buf0_window(
                                    &dev.device,
                                    mm_k,
                                    &arena.buffer,
                                    base_w,
                                    size_w,
                                    &u_mm,
                                );
                                uniforms.push(u_mm);
                                bind_groups.push(bg_mm);
                            } else {
                                schedule.push(if use_tiled {
                                    Step::Conv2dTiled { params: p2 }
                                } else {
                                    Step::Conv2d { params: p2 }
                                });
                                let ck = if use_tiled {
                                    conv1d_tiled_kernel(&dev.device)
                                } else {
                                    conv2d_kernel(&dev.device)
                                };
                                let u = emit_uniform(std::mem::size_of::<Conv2dParams>());
                                let bg = bind_two_buf0_window(
                                    &dev.device,
                                    ck,
                                    &arena.buffer,
                                    base,
                                    size,
                                    &u,
                                );
                                uniforms.push(u);
                                bind_groups.push(bg);
                            }
                        }
                        (3, 5, 5, 5) => {
                            let p3 = Conv3dParams {
                                n: in_shape[0].unwrap_static() as u32,
                                c_in: in_shape[1].unwrap_static() as u32,
                                c_out: out_shape[1].unwrap_static() as u32,
                                d: in_shape[2].unwrap_static() as u32,
                                h: in_shape[3].unwrap_static() as u32,
                                w: in_shape[4].unwrap_static() as u32,
                                d_out: out_shape[2].unwrap_static() as u32,
                                h_out: out_shape[3].unwrap_static() as u32,
                                w_out: out_shape[4].unwrap_static() as u32,
                                kd: kernel_size[0] as u32,
                                kh: kernel_size[1] as u32,
                                kw: kernel_size[2] as u32,
                                sd: s(0),
                                sh: s(1),
                                sw: s(2),
                                pd: p(0),
                                ph: p(1),
                                pw: p(2),
                                dd: d(0),
                                dh: d(1),
                                dw: d(2),
                                groups: *groups as u32,
                                in_off,
                                w_off,
                                out_off,
                                _p0: 0,
                            };
                            schedule.push(Step::Conv3d { params: p3 });
                            let ck = conv3d_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<Conv3dParams>());
                            let bg = bind_two_buf0_window(
                                &dev.device,
                                ck,
                                &arena.buffer,
                                base,
                                size,
                                &u,
                            );
                            uniforms.push(u);
                            bind_groups.push(bg);
                        }
                        (k, ni, wi, mi) => panic!(
                            "rlx-wgpu Conv: rank kernel={k} in={ni} weight={wi} out={mi} \
                             not supported (use 1D/2D/3D NCHW)"
                        ),
                    }
                }

                Op::Im2Col {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    if kernel_size.len() != 2 || x_shape.rank() != 4 {
                        panic!("rlx-wgpu Im2Col: 2D NCHW only");
                    }
                    let n = match x_shape.dim(0) {
                        rlx_ir::shape::Dim::Static(v) => v as u32,
                        _ => 0,
                    };
                    let c_in = x_shape.dim(1).unwrap_static() as u32;
                    let h = x_shape.dim(2).unwrap_static() as u32;
                    let w = x_shape.dim(3).unwrap_static() as u32;
                    let kh = kernel_size[0] as u32;
                    let kw = kernel_size[1] as u32;
                    let sh = stride.first().copied().unwrap_or(1) as u32;
                    let sw = stride.get(1).copied().unwrap_or(1) as u32;
                    let ph = padding.first().copied().unwrap_or(0) as u32;
                    let pw = padding.get(1).copied().unwrap_or(0) as u32;
                    let dh = dilation.first().copied().unwrap_or(1) as u32;
                    let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
                    let h_out = rlx_ir::shape::conv2d_spatial_output(
                        h as usize,
                        kh as usize,
                        sh as usize,
                        ph as usize,
                        dh as usize,
                    ) as u32;
                    let w_out = rlx_ir::shape::conv2d_spatial_output(
                        w as usize,
                        kw as usize,
                        sw as usize,
                        pw as usize,
                        dw_dil as usize,
                    ) as u32;
                    schedule.push(Step::Im2ColHost {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        col_byte_off: arena.offset(node.id) as u32,
                        n,
                        c_in,
                        h,
                        w,
                        h_out,
                        w_out,
                        kh,
                        kw,
                        sh,
                        sw,
                        ph,
                        pw,
                        dh,
                        dw_dil,
                    });
                }

                Op::Cumsum { axis, exclusive } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.dims();
                    let last = (in_shape.len() - 1) as i32;
                    if *axis != -1 && *axis != last {
                        panic!("rlx-wgpu Cumsum: only last-axis wired (got axis={axis})");
                    }
                    let inner = in_shape[in_shape.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_shape.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = CumsumParams {
                        outer,
                        inner,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::Cumsum { params: p });
                    let ck2 = cumsum_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<CumsumParams>());
                    let bg = bind_op_output_window(&dev.device, ck2, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Fft { inverse, norm } => {
                    let in_id = node.inputs[0];
                    let in_shape = graph.node(in_id).shape.clone();
                    let meta = rlx_ir::fft::fft_meta(&in_shape);
                    let dtype = in_shape.dtype();
                    // The wgpu arena is f32-uniform (4 bytes/elem — see the Constant
                    // upload path). f64/c64 tensors get half-sized slots and are
                    // silently truncated to garbage; the FFT-host fallback can't
                    // read them back either. Reject cleanly instead (like MLX/ANE,
                    // which don't do f64 on the GPU). Use f32 for GPU FFT.
                    assert!(
                        matches!(dtype, rlx_ir::DType::F32),
                        "rlx-wgpu Op::Fft: only f32 is supported on the wgpu backend \
                         (the arena is f32-uniform); got {dtype:?}. Run f64/c64 FFT on CPU."
                    );
                    let use_gpu = rlx_ir::fft::gpu_fft_native_eligible(dtype, meta.n_complex)
                        && meta.n_complex >= 2;
                    let scale = norm.output_scale(meta.n_complex, *inverse) as f32;
                    if use_gpu {
                        schedule.push(Step::FftGpu {
                            src_off: (arena.offset(in_id) / 4) as u32,
                            dst_off: (arena.offset(node.id) / 4) as u32,
                            outer: meta.outer as u32,
                            n: meta.n_complex as u32,
                            inverse: if *inverse { 1 } else { 0 },
                            norm_scale: scale,
                        });
                        fft_gpu_steps.push(crate::fft_dispatch::FftGpuResources::new(
                            &dev.device,
                            &arena.buffer,
                        ));
                    } else {
                        schedule.push(Step::FftHost {
                            src_byte_off: arena.offset(in_id) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            outer: meta.outer as u32,
                            n_complex: meta.n_complex as u32,
                            inverse: *inverse,
                            norm_tag: norm.tag(),
                            dtype_tag: fft_dtype_tag(dtype),
                        });
                    }
                }
                Op::WelchPeaks { k, n_segments } => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let meta = rlx_ir::audio::welch_peaks_meta(&spec_shape, *k, *n_segments)
                        .unwrap_or_else(|e| panic!("Op::WelchPeaks: {e}"));
                    let use_gpu = rlx_ir::audio::welch_peaks_gpu_native_eligible(
                        &spec_shape,
                        *k,
                        *n_segments,
                    )
                    .unwrap_or(false);
                    if use_gpu {
                        let p = WelchPeaksGpuParams {
                            spec_off: (arena.offset(node.inputs[0]) / 4) as u32,
                            dst_off: (arena.offset(node.id) / 4) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                            n_bins: meta.n_bins as u32,
                            _p0: 0,
                            _p1: 0,
                        };
                        schedule.push(Step::WelchPeaksGpu { params: p });
                        let wk = welch_peaks_gpu_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<WelchPeaksGpuParams>());
                        let bg = bind_op_output_window(&dev.device, wk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        schedule.push(Step::WelchPeaksHost {
                            spec_byte_off: arena.offset(node.inputs[0]) as u32,
                            dst_byte_off: arena.offset(node.id) as u32,
                            welch_batch: meta.welch_batch as u32,
                            n_fft: meta.n_fft as u32,
                            n_segments: meta.n_segments as u32,
                            k: meta.k as u32,
                        });
                    }
                }
                Op::LogMel => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMel: {e}"));
                    schedule.push(Step::LogMelHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::LogMelBackward => {
                    let spec_shape = graph.node(node.inputs[0]).shape.clone();
                    let filt_shape = graph.node(node.inputs[1]).shape.clone();
                    let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                        .unwrap_or_else(|e| panic!("Op::LogMelBackward: {e}"));
                    schedule.push(Step::LogMelBackwardHost {
                        spec_byte_off: arena.offset(node.inputs[0]) as u32,
                        filt_byte_off: arena.offset(node.inputs[1]) as u32,
                        dy_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: meta.outer as u32,
                        n_fft: meta.n_fft as u32,
                        n_bins: meta.n_bins as u32,
                        n_mels: meta.n_mels as u32,
                    });
                }
                Op::SelectiveScan { state_size } => {
                    if *state_size > 256 {
                        panic!(
                            "rlx-wgpu SelectiveScan: state_size {} exceeds compile-time \
                                cap of 256 (kernel uses fixed-size private array)",
                            state_size
                        );
                    }
                    let x_id = node.inputs[0];
                    let dt_id = node.inputs[1];
                    let a_id = node.inputs[2];
                    let b_id = node.inputs[3];
                    let c_id = node.inputs[4];
                    let in_dims = graph.node(x_id).shape.dims();
                    let seq = in_dims[1].unwrap_static() as u32;
                    let p = SelectiveScanParams {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq,
                        hidden: in_dims[2].unwrap_static() as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        delta_off: (arena.offset(dt_id) / 4) as u32,
                        a_off: (arena.offset(a_id) / 4) as u32,
                        b_off: (arena.offset(b_id) / 4) as u32,
                        c_off: (arena.offset(c_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        // PLAN L1: full-extent stride; safe under
                        // active-extent scaling of params.seq.
                        seq_stride: seq,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                        _p4: 0,
                        _p5: 0,
                    };
                    schedule.push(Step::SelectiveScan { params: p });
                    let ssk = selective_scan_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<SelectiveScanParams>());
                    let bg = bind_op_output_window(&dev.device, ssk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Mamba2 {
                    head_dim,
                    state_size,
                } => {
                    if *state_size > 256 {
                        panic!(
                            "rlx-wgpu Mamba2: state_size {} exceeds compile-time cap of 256",
                            state_size
                        );
                    }
                    let x_id = node.inputs[0];
                    let in_dims = graph.node(x_id).shape.dims(); // [B,S,H,P]
                    let seq = in_dims[1].unwrap_static() as u32;
                    let p = Mamba2Params {
                        batch: in_dims[0].unwrap_static() as u32,
                        seq,
                        heads: in_dims[2].unwrap_static() as u32,
                        head_dim: *head_dim as u32,
                        state_size: *state_size as u32,
                        x_off: (arena.offset(x_id) / 4) as u32,
                        dt_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        a_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        b_off: (arena.offset(node.inputs[3]) / 4) as u32,
                        c_off: (arena.offset(node.inputs[4]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        seq_stride: seq,
                        _p1: 0,
                        _p2: 0,
                        _p3: 0,
                        _p4: 0,
                    };
                    schedule.push(Step::Mamba2 { params: p });
                    let mk = mamba2_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<Mamba2Params>());
                    let bg = bind_op_output_window(&dev.device, mk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::Gru {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_id = node.inputs[0];
                    let in_dims = graph.node(x_id).shape.dims(); // [B,S,In]
                    let batch = in_dims[0].unwrap_static() as u32;
                    let seq = in_dims[1].unwrap_static() as u32;
                    let input_size = in_dims[2].unwrap_static() as u32;
                    let hidden = *hidden_size as u32;
                    let simple = *num_layers == 1 && !*bidirectional && !*carry;
                    if simple && hidden <= 256 {
                        let p = GruParams {
                            batch,
                            seq,
                            input_size,
                            hidden,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            wih_off: (arena.offset(node.inputs[1]) / 4) as u32,
                            whh_off: (arena.offset(node.inputs[2]) / 4) as u32,
                            bih_off: (arena.offset(node.inputs[3]) / 4) as u32,
                            bhh_off: (arena.offset(node.inputs[4]) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            seq_stride: seq,
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                            _p5: 0,
                        };
                        schedule.push(Step::Gru { params: p });
                        let gk = gru_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<GruParams>());
                        let bg = bind_op_output_window(&dev.device, gk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let h0 = if *carry {
                            arena.offset(node.inputs[5]) as u32
                        } else {
                            0
                        };
                        schedule.push(Step::GruHost {
                            x: arena.offset(x_id) as u32,
                            w_ih: arena.offset(node.inputs[1]) as u32,
                            w_hh: arena.offset(node.inputs[2]) as u32,
                            b_ih: arena.offset(node.inputs[3]) as u32,
                            b_hh: arena.offset(node.inputs[4]) as u32,
                            h0,
                            dst: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            carry: *carry,
                        });
                    }
                }
                Op::Rnn {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    let x_id = node.inputs[0];
                    let in_dims = graph.node(x_id).shape.dims();
                    let batch = in_dims[0].unwrap_static() as u32;
                    let seq = in_dims[1].unwrap_static() as u32;
                    let input_size = in_dims[2].unwrap_static() as u32;
                    let hidden = *hidden_size as u32;
                    let simple = *num_layers == 1 && !*bidirectional && !*carry;
                    if simple && hidden <= 256 {
                        let p = RnnParams {
                            batch,
                            seq,
                            input_size,
                            hidden,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            wih_off: (arena.offset(node.inputs[1]) / 4) as u32,
                            whh_off: (arena.offset(node.inputs[2]) / 4) as u32,
                            bias_off: (arena.offset(node.inputs[3]) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            seq_stride: seq,
                            relu: u32::from(*relu),
                            _p1: 0,
                            _p2: 0,
                            _p3: 0,
                            _p4: 0,
                            _p5: 0,
                        };
                        schedule.push(Step::Rnn { params: p });
                        let rk = rnn_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<RnnParams>());
                        let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    } else {
                        let h0 = if *carry {
                            arena.offset(node.inputs[4]) as u32
                        } else {
                            0
                        };
                        schedule.push(Step::RnnHost {
                            x: arena.offset(x_id) as u32,
                            w_ih: arena.offset(node.inputs[1]) as u32,
                            w_hh: arena.offset(node.inputs[2]) as u32,
                            bias: arena.offset(node.inputs[3]) as u32,
                            h0,
                            dst: arena.offset(node.id) as u32,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers: *num_layers as u32,
                            bidirectional: *bidirectional,
                            carry: *carry,
                            relu: *relu,
                        });
                    }
                }
                Op::GatedDeltaNet {
                    state_size,
                    carry_state,
                } => {
                    if *state_size > rlx_cpu::gdn::GDN_MAX_STATE {
                        panic!(
                            "rlx-wgpu GatedDeltaNet: state_size {state_size} > {}",
                            rlx_cpu::gdn::GDN_MAX_STATE
                        );
                    }
                    let q_id = node.inputs[0];
                    let q_shape = &graph.node(q_id).shape;
                    let state_off = if *carry_state {
                        arena.offset(node.inputs[5])
                    } else {
                        0
                    };
                    schedule.push(Step::GatedDeltaNet {
                        q_byte_off: arena.offset(q_id) as u32,
                        k_byte_off: arena.offset(node.inputs[1]) as u32,
                        v_byte_off: arena.offset(node.inputs[2]) as u32,
                        g_byte_off: arena.offset(node.inputs[3]) as u32,
                        beta_byte_off: arena.offset(node.inputs[4]) as u32,
                        state_byte_off: state_off as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: q_shape.dim(0).unwrap_static() as u32,
                        seq: q_shape.dim(1).unwrap_static() as u32,
                        heads: q_shape.dim(2).unwrap_static() as u32,
                        state_size: *state_size as u32,
                        use_carry: *carry_state,
                    });
                    if gguf_host_pad.is_none() {
                        let bk = binary_kernel(&dev.device);
                        let u = emit_uniform(256);
                        gguf_host_pad = Some((
                            u.clone(),
                            bind_op_output_window(&dev.device, bk, &arena, node.id, &u),
                        ));
                    }
                    let (u, bg) = gguf_host_pad.as_ref().unwrap();
                    uniforms.push(u.clone());
                    bind_groups.push(bg.clone());
                }
                Op::Lstm {
                    hidden_size,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let (h0, c0) = if *carry {
                        (
                            arena.offset(node.inputs[4]) as u32,
                            arena.offset(node.inputs[5]) as u32,
                        )
                    } else {
                        (0u32, 0u32)
                    };
                    schedule.push(Step::Lstm {
                        x_byte_off: arena.offset(node.inputs[0]) as u32,
                        w_ih_byte_off: arena.offset(node.inputs[1]) as u32,
                        w_hh_byte_off: arena.offset(node.inputs[2]) as u32,
                        bias_byte_off: arena.offset(node.inputs[3]) as u32,
                        h0_byte_off: h0,
                        c0_byte_off: c0,
                        dst_byte_off: arena.offset(node.id) as u32,
                        batch: x_shape.dim(0).unwrap_static() as u32,
                        seq: x_shape.dim(1).unwrap_static() as u32,
                        input_size: x_shape.dim(2).unwrap_static() as u32,
                        hidden: *hidden_size as u32,
                        num_layers: *num_layers as u32,
                        bidirectional: *bidirectional,
                        carry: *carry,
                    });
                    // Host step — keep schedule/uniform/bind_group lanes aligned.
                    if gguf_host_pad.is_none() {
                        let bk = binary_kernel(&dev.device);
                        let u = emit_uniform(256);
                        gguf_host_pad = Some((
                            u.clone(),
                            bind_op_output_window(&dev.device, bk, &arena, node.id, &u),
                        ));
                    }
                    let (u, bg) = gguf_host_pad.as_ref().unwrap();
                    uniforms.push(u.clone());
                    bind_groups.push(bg.clone());
                }
                Op::ConvTranspose2d {
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    output_padding: _,
                    groups,
                } => {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let out_shape = &node.shape;
                    schedule.push(Step::ConvTranspose2d {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        weight_byte_off: arena.offset(node.inputs[1]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c_in: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w_in: in_shape.dim(3).unwrap_static() as u32,
                        c_out: out_shape.dim(1).unwrap_static() as u32,
                        h_out: out_shape.dim(2).unwrap_static() as u32,
                        w_out: out_shape.dim(3).unwrap_static() as u32,
                        kh: kernel_size[0] as u32,
                        kw: kernel_size[1] as u32,
                        sh: stride[0] as u32,
                        sw: stride[1] as u32,
                        ph: padding[0] as u32,
                        pw: padding[1] as u32,
                        dh: dilation[0] as u32,
                        dw: dilation[1] as u32,
                        groups: *groups as u32,
                    });
                    // Host step: schedule-only, like `MsDeformAttnHost`/`UmapKnnHost`.
                    // The uniform/bind_group lanes are indexed by GPU-step count
                    // (`gpu_ui`), so host steps must NOT push to them.
                }
                Op::GroupNorm { num_groups, eps } => {
                    // NCHW: x [n,c,h,w], gamma/beta [c]. Host step (schedule-only).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::GroupNormHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        gamma_byte_off: arena.offset(node.inputs[1]) as u32,
                        beta_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        num_groups: *num_groups as u32,
                        eps: *eps,
                    });
                }
                Op::LayerNorm2d { eps } => {
                    // NCHW: x [n,c,h,w], gamma/beta [c]. Host step (schedule-only).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::LayerNorm2dHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        gamma_byte_off: arena.offset(node.inputs[1]) as u32,
                        beta_byte_off: arena.offset(node.inputs[2]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                        eps: *eps,
                    });
                }
                Op::ResizeNearest2x => {
                    // NCHW nearest 2× upsample. Host step (schedule-only).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2xHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
                }
                Op::Reverse { axes } => {
                    // Batch-general flip. Host step (schedule-only).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let dims: Vec<u32> = (0..rank)
                        .map(|i| in_shape.dim(i).unwrap_static() as u32)
                        .collect();
                    let mut rev_mask = vec![false; rank];
                    for &a in axes {
                        if a < rank {
                            rev_mask[a] = true;
                        }
                    }
                    schedule.push(Step::ReverseHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        dims,
                        rev_mask,
                        elem_bytes: in_shape.dtype().size_bytes() as u32,
                    });
                }
                Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                    // ArgMax/ArgMin. Host step (schedule-only).
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let rank = in_shape.rank();
                    let outer: usize = (0..*axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let reduced = in_shape.dim(*axis).unwrap_static();
                    let inner: usize = (*axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    schedule.push(Step::ArgReduceHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        outer: outer as u32,
                        reduced: reduced as u32,
                        inner: inner as u32,
                        is_max: matches!(node.op, Op::ArgMax { .. }),
                    });
                }
                Op::Scan {
                    body,
                    length,
                    save_trajectory,
                    num_bcast,
                    num_xs,
                    ..
                } => {
                    // Host fallback (no unified memory): compile the body once,
                    // loop it on the CPU against a readback of the arena span.
                    let nb = *num_bcast as usize;
                    let nx = *num_xs as usize;
                    let plan = rlx_cpu::thunk::compile_scan_body(body, nb, nx);
                    let bcast_outer: Vec<(usize, usize)> = (0..nb)
                        .map(|i| {
                            let id = node.inputs[1 + i];
                            (arena.offset(id), graph.node(id).shape.size_bytes().unwrap())
                        })
                        .collect();
                    let xs_outer: Vec<(usize, usize)> = (0..nx)
                        .map(|i| {
                            let id = node.inputs[1 + nb + i];
                            let total = graph.node(id).shape.size_bytes().unwrap();
                            (arena.offset(id), total / *length as usize)
                        })
                        .collect();
                    schedule.push(Step::ScanHost {
                        plan: std::sync::Arc::new(plan),
                        outer_init_off: arena.offset(node.inputs[0]),
                        outer_final_off: arena.offset(node.id),
                        length: *length,
                        save_trajectory: *save_trajectory,
                        xs_outer,
                        bcast_outer,
                    });
                }
                Op::Custom { name, attrs, .. } => match name.as_str() {
                    "llada2.group_limited_gate" => {
                        let sig_id = node.inputs[0];
                        let route_id = node.inputs[1];
                        let n_elems = graph.node(sig_id).shape.num_elements().unwrap() as u32;
                        let mut attr_buf = [0u8; 20];
                        let n = attrs.len().min(20);
                        attr_buf[..n].copy_from_slice(&attrs[..n]);
                        schedule.push(Step::Llada2GroupLimitedGate {
                            sig_byte_off: arena.offset(sig_id) as u32,
                            route_byte_off: arena.offset(route_id) as u32,
                            out_byte_off: arena.offset(node.id) as u32,
                            n_elems,
                            attrs: attr_buf,
                        });
                    }
                    "umap.knn" => {
                        let pw_id = node.inputs[0];
                        let pw_shape = graph.node(pw_id).shape.dims();
                        let n = pw_shape[0].unwrap_static() as u32;
                        let k = if attrs.len() >= 4 {
                            u32::from_le_bytes(attrs[..4].try_into().unwrap())
                        } else {
                            panic!("rlx-wgpu: umap.knn attrs missing k");
                        };
                        let pw_off = arena.offset(pw_id) as u32;
                        let out_off = arena.offset(node.id) as u32;
                        if n as usize >= crate::umap_knn_host::UMAP_KNN_GPU_MIN_N {
                            let p = UmapKnnParams {
                                n,
                                k,
                                pw_off: pw_off / 4,
                                out_off: out_off / 4,
                                _p0: 0,
                                _p1: 0,
                                _p2: 0,
                            };
                            schedule.push(Step::UmapKnn { params: p });
                            let uk = umap_knn_kernel(&dev.device);
                            let u = emit_uniform(std::mem::size_of::<UmapKnnParams>());
                            let bg = bind_op_output_window(&dev.device, uk, &arena, node.id, &u);
                            uniforms.push(u);
                            bind_groups.push(bg);
                        } else {
                            schedule.push(Step::UmapKnnHost {
                                pairwise_byte_off: pw_off,
                                out_byte_off: out_off,
                                n,
                                k,
                            });
                        }
                    }
                    "gdino.ms_deform_attn" => {
                        let in_offs: Vec<(u32, u32)> = node
                            .inputs
                            .iter()
                            .map(|&id| {
                                let bytes = graph.node(id).shape.num_elements().unwrap() * 4;
                                (arena.offset(id) as u32, bytes as u32)
                            })
                            .collect();
                        let out_bytes = (node.shape.num_elements().unwrap() * 4) as u32;
                        schedule.push(Step::MsDeformAttnHost {
                            in_offs,
                            out_byte_off: arena.offset(node.id) as u32,
                            out_bytes,
                            attrs: attrs.clone(),
                        });
                    }
                    other => panic!("rlx-wgpu: unsupported Op::Custom('{other}')"),
                },
                Op::GroupedMatMul => {
                    // Inputs: input [M, K], weight [E, K, N], expert_idx [M]
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let w_dims = graph.node(w_id).shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = w_dims[2].unwrap_static() as u32;
                    let ne = w_dims[0].unwrap_static() as u32;
                    let p = GroupedMatmulParams {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        w_off: (arena.offset(w_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                    };
                    schedule.push(Step::GroupedMatmul { params: p });
                    let gk = grouped_matmul_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GroupedMatmulParams>());
                    let bg = bind_op_output_window(&dev.device, gk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::DequantGroupedMatMul { scheme } => {
                    let in_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    let idx_id = node.inputs[2];
                    let in_dims = graph.node(in_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let m = in_dims[0].unwrap_static() as u32;
                    let k = in_dims[1].unwrap_static() as u32;
                    let n = out_dims[out_dims.len() - 1].unwrap_static() as u32;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let slab_bytes = (k as usize * n as usize) / block_elems * block_bytes;
                    let total_bytes = graph.node(w_id).shape.num_elements().unwrap();
                    let ne = (total_bytes / slab_bytes.max(1)) as u32;
                    schedule.push(Step::DequantGroupedMatmulGguf {
                        m,
                        k,
                        n,
                        num_experts: ne,
                        scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                        x_byte_off: arena.offset(in_id) as u64,
                        w_byte_off: arena.offset(w_id) as u64,
                        idx_byte_off: arena.offset(idx_id) as u64,
                        out_byte_off: arena.offset(node.id) as u64,
                    });
                    // Host step (builds its own bind groups): schedule-only, like
                    // `GroupNorm`. The uniform/bind_group lanes are indexed by
                    // GPU-step count (host steps skipped), so pushing here would
                    // shift every later GPU op's lane. Do NOT push.
                }
                Op::TopK { k } => {
                    let in_id = node.inputs[0];
                    let in_dims = graph.node(in_id).shape.dims();
                    let inner = in_dims.last().unwrap().unwrap_static() as u32;
                    let outer: u32 = in_dims[..in_dims.len() - 1]
                        .iter()
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let p = TopKParams {
                        outer,
                        inner,
                        k: *k as u32,
                        in_off: (arena.offset(in_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    schedule.push(Step::TopK { params: p });
                    let tk = topk_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<TopKParams>());
                    let bg = bind_op_output_window(&dev.device, tk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::ScatterAdd => {
                    // Inputs: updates [num_updates, trailing], indices [num_updates].
                    // Output: [out_dim, trailing]. Implemented as two phases:
                    //   1. Zero `out_dim * trailing` slots.
                    //   2. CAS-loop atomic-accumulate `num_updates * trailing` updates.
                    let upd_id = node.inputs[0];
                    let idx_id = node.inputs[1];
                    let upd_dims = graph.node(upd_id).shape.dims();
                    let out_dims = node.shape.dims();
                    let num_updates = upd_dims[0].unwrap_static() as u32;
                    let trailing: u32 = upd_dims
                        .iter()
                        .skip(1)
                        .map(|d| d.unwrap_static() as u32)
                        .product::<u32>()
                        .max(1);
                    let out_dim = out_dims[0].unwrap_static() as u32;
                    let out_total = out_dim * trailing;

                    let common = ScatterAddParams {
                        op: 0,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        upd_off: (arena.offset(upd_id) / 4) as u32,
                        idx_off: (arena.offset(idx_id) / 4) as u32,
                        out_total,
                        num_updates,
                        trailing,
                        out_dim,
                    };
                    let sk = scatter_add_kernel(&dev.device);

                    // Phase 0: zero.
                    schedule.push(Step::ScatterAdd { params: common });
                    let u0 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                    let bg0 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u0);
                    uniforms.push(u0);
                    bind_groups.push(bg0);

                    // Phase 1: accumulate.
                    let mut acc = common;
                    acc.op = 1;
                    schedule.push(Step::ScatterAdd { params: acc });
                    let u1 = emit_uniform(std::mem::size_of::<ScatterAddParams>());
                    let bg1 = bind_op_output_window(&dev.device, sk, &arena, node.id, &u1);
                    uniforms.push(u1);
                    bind_groups.push(bg1);
                }
                Op::FusedResidualLN { has_bias, eps } => {
                    // Inputs: [x, residual, [bias], gamma, beta].
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3]) // bias unused
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = FusedResidualLnParams {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::FusedResidualLn { params: p });
                    let frk = fused_residual_ln_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FusedResidualLnParams>());
                    let bg = bind_op_output_window(&dev.device, frk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::FusedResidualRmsNorm { has_bias, eps } => {
                    let x_id = node.inputs[0];
                    let r_id = node.inputs[1];
                    let (bias_id, g_id, b_id) = if *has_bias {
                        (node.inputs[2], node.inputs[3], node.inputs[4])
                    } else {
                        (x_id, node.inputs[2], node.inputs[3])
                    };
                    let in_dims = node.shape.dims();
                    let inner = in_dims[in_dims.len() - 1].unwrap_static() as u32;
                    let total: u32 = in_dims.iter().map(|d| d.unwrap_static() as u32).product();
                    let outer = total / inner.max(1);
                    let p = FusedResidualRmsNormParams {
                        outer,
                        inner,
                        in_off: (arena.offset(x_id) / 4) as u32,
                        residual_off: (arena.offset(r_id) / 4) as u32,
                        bias_off: (arena.offset(bias_id) / 4) as u32,
                        gamma_off: (arena.offset(g_id) / 4) as u32,
                        beta_off: (arena.offset(b_id) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        has_bias: if *has_bias { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                    };
                    schedule.push(Step::FusedResidualRmsNorm { params: p });
                    let frk = fused_residual_rms_norm_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<FusedResidualRmsNormParams>());
                    let bg = bind_op_output_window(&dev.device, frk, &arena, node.id, &u);
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::DequantMatMul { scheme } => {
                    use rlx_ir::QuantScheme;
                    let x_id = node.inputs[0];
                    let w_id = node.inputs[1];
                    // Rank-agnostic GEMM dims: `n` = last output dim, `m` = the
                    // product of the leading output dims (e.g. batch·seq), `k` =
                    // last input dim. A 2D-only `out_dims[1]` read collapses a 3D
                    // decode output `[1, 1, hidden]` to `n = 1` (the seq axis),
                    // which then mis-sizes the weight window. Mirrors
                    // `gguf_gpu::dequant_gguf_scratch_bytes`.
                    let out_total = node.shape.num_elements().unwrap_or(0) as u32;
                    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static() as u32;
                    let m = out_total / n.max(1);
                    let x_total = graph.node(x_id).shape.num_elements().unwrap_or(0) as u32;
                    let k = x_total / m.max(1);
                    if scheme.is_gguf() {
                        schedule.push(Step::DequantMatmulGguf {
                            m,
                            k,
                            n,
                            scheme_id: crate::gguf_host::gguf_scheme_id(*scheme),
                            x_byte_off: arena.offset(x_id) as u64,
                            w_byte_off: arena.offset(w_id) as u64,
                            out_byte_off: arena.offset(node.id) as u64,
                        });
                        // Host step (`run_dequant_matmul_gguf_gpu` builds its own
                        // bind groups): schedule-only, like `GroupNorm`. The
                        // uniform/bind_group lanes are indexed by GPU-step count
                        // (`gpu_ui`/`gpu_bi`, host steps skipped), so pushing here
                        // would shift every later GPU op's lane — fatal in the
                        // packed decode graph where GGUF ops interleave with
                        // RmsNorm/RoPE/Attention/SwiGLU. Do NOT push.
                    } else {
                        let (block_size, scheme_id) = match scheme {
                            QuantScheme::Int8Block { block_size } => (*block_size, 0u32),
                            QuantScheme::Int8BlockAsym { block_size } => (*block_size, 1u32),
                            QuantScheme::Int4Block { block_size } => (*block_size, 2u32),
                            QuantScheme::Fp8E4m3 => (1, 3u32),
                            QuantScheme::Fp8E5m2 => (1, 4u32),
                            QuantScheme::Nvfp4Block => (rlx_ir::NVFP4_GROUP_SIZE as u32, 5u32),
                            other => panic!("rlx-wgpu DequantMatMul: unsupported scheme {other:?}"),
                        };
                        let scale_id = node.inputs[2];
                        let zp_id = node.inputs[3];
                        let p = DequantMatmulParams {
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off: (arena.offset(x_id) / 4) as u32,
                            w_off: (arena.offset(w_id) / 4) as u32,
                            scale_off: (arena.offset(scale_id) / 4) as u32,
                            zp_off: (arena.offset(zp_id) / 4) as u32,
                            out_off: (arena.offset(node.id) / 4) as u32,
                            _p0: 0,
                            _p1: 0,
                        };
                        schedule.push(Step::DequantMatmul { params: p });
                        let dk = dequant_matmul_kernel(&dev.device);
                        let u = emit_uniform(std::mem::size_of::<DequantMatmulParams>());
                        let bg = bind_op_output_window(&dev.device, dk, &arena, node.id, &u);
                        uniforms.push(u);
                        bind_groups.push(bg);
                    }
                }
                Op::RmsNormBackwardInput { eps, .. }
                | Op::RmsNormBackwardGamma { eps, .. }
                | Op::RmsNormBackwardBeta { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let foff = |i: usize| (arena.offset(node.inputs[i]) / 4) as u32;
                    let wrt = match &node.op {
                        Op::RmsNormBackwardInput { .. } => 0u32,
                        Op::RmsNormBackwardGamma { .. } => 1u32,
                        Op::RmsNormBackwardBeta { .. } => 2u32,
                        _ => unreachable!(),
                    };
                    let p = RmsNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: foff(0),
                        gamma_off: foff(1),
                        beta_off: foff(2),
                        dy_off: foff(3),
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        wrt,
                    };
                    let rk = if wrt == 0 {
                        rms_norm_backward_kernel(&dev.device)
                    } else {
                        rms_norm_backward_param_kernel(&dev.device)
                    };
                    let u = emit_uniform(std::mem::size_of::<RmsNormBwdParams>());
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                    match &node.op {
                        Op::RmsNormBackwardInput { .. } => {
                            schedule.push(Step::RmsNormBackwardInput { params: p });
                        }
                        Op::RmsNormBackwardGamma { .. } => {
                            schedule.push(Step::RmsNormBackwardGamma { params: p });
                        }
                        Op::RmsNormBackwardBeta { .. } => {
                            schedule.push(Step::RmsNormBackwardBeta { params: p });
                        }
                        _ => unreachable!(),
                    }
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::LayerNormBackwardInput { eps, .. } => {
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    let p = LayerNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        gamma_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dy_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        scratch_off: 0,
                    };
                    let rk = layer_norm_backward_input_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                    schedule.push(Step::LayerNormBackwardInput { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::LayerNormBackwardGamma { eps, .. } => {
                    // Inputs: [x, dy] — gamma_off is unused for this op.
                    // Emit two steps: a multi-workgroup partial that
                    // writes per-chunk dgamma to the tail scratch zone,
                    // and a single-workgroup reduce that sums chunks
                    // into the final dgamma slot.
                    let x_shape = &graph.node(node.inputs[0]).shape;
                    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (x_shape.num_elements().unwrap() / h.max(1) as usize) as u32;
                    const ROWS_PER_WG: u32 = 16;
                    let num_workgroups = rows.div_ceil(ROWS_PER_WG.max(1));
                    let scratch_off_words = (arena.scratch_off / 4) as u32;
                    let partial_params = LayerNormBwdParams {
                        outer: rows,
                        inner: h,
                        x_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        gamma_off: 0,
                        dy_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        out_off: 0, // unused by the partial kernel
                        eps_bits: eps.to_bits(),
                        scratch_off: scratch_off_words,
                    };
                    let reduce_params = LayerNormBwdParams {
                        // `outer` for the reduce kernel carries the
                        // number of partial chunks we just emitted.
                        outer: num_workgroups,
                        inner: h,
                        x_off: 0,
                        gamma_off: 0,
                        dy_off: 0,
                        out_off: (arena.offset(node.id) / 4) as u32,
                        eps_bits: eps.to_bits(),
                        scratch_off: scratch_off_words,
                    };
                    let p_k = layer_norm_backward_gamma_partial_kernel(&dev.device);
                    let r_k = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                    let p_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let r_u = emit_uniform(std::mem::size_of::<LayerNormBwdParams>());
                    let p_bg = bind_op_output_window(&dev.device, p_k, &arena, node.id, &p_u);
                    let r_bg = bind_op_output_window(&dev.device, r_k, &arena, node.id, &r_u);
                    schedule.push(Step::LayerNormBackwardGammaPartial {
                        params: partial_params,
                        num_workgroups,
                    });
                    schedule.push(Step::LayerNormBackwardGammaReduce {
                        params: reduce_params,
                    });
                    uniforms.push(p_u);
                    uniforms.push(r_u);
                    bind_groups.push(p_bg);
                    bind_groups.push(r_bg);
                }
                Op::RopeBackward { head_dim, n_rot } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let (batch, seq, hidden) = if dy_shape.rank() >= 3 {
                        (
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                            dy_shape.dim(2).unwrap_static() as u32,
                        )
                    } else {
                        (
                            1,
                            dy_shape.dim(0).unwrap_static() as u32,
                            dy_shape.dim(1).unwrap_static() as u32,
                        )
                    };
                    let cos_len = graph.node(node.inputs[1]).shape.num_elements().unwrap() as u32;
                    let p = RopeBwdParams {
                        batch,
                        seq,
                        hidden,
                        head_dim: *head_dim as u32,
                        n_rot: *n_rot as u32,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        cos_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        sin_off: (arena.offset(node.inputs[2]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        cos_len,
                    };
                    let rk = rope_backward_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<RopeBwdParams>());
                    let bg = bind_op_output_window(&dev.device, rk, &arena, node.id, &u);
                    schedule.push(Step::RopeBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::CumsumBackward { exclusive, .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let cols = dy_shape.dim(dy_shape.rank() - 1).unwrap_static() as u32;
                    let rows = (dy_shape.num_elements().unwrap() / cols.max(1) as usize) as u32;
                    let p = CumsumBwdParams {
                        outer: rows,
                        inner: cols,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        dx_off: (arena.offset(node.id) / 4) as u32,
                        exclusive: if *exclusive { 1 } else { 0 },
                        _p0: 0,
                        _p1: 0,
                        _p2: 0,
                    };
                    let ck = cumsum_backward_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<CumsumBwdParams>());
                    let bg = bind_op_output_window(&dev.device, ck, &arena, node.id, &u);
                    schedule.push(Step::CumsumBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                Op::GatherBackward { .. } => {
                    let dy_shape = &graph.node(node.inputs[0]).shape;
                    let idx_shape = &graph.node(node.inputs[1]).shape;
                    let out_shape = &node.shape;
                    let rank = out_shape.rank();
                    let axis = match &node.op {
                        Op::GatherBackward { axis } => *axis,
                        _ => 0,
                    };
                    let axis_u = if axis < 0 {
                        (rank as i32 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let outer: usize = (0..axis_u)
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let num_idx = idx_shape.dim(axis_u).unwrap_static();
                    let trailing: usize = (axis_u + 1..dy_shape.rank())
                        .map(|i| dy_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_dim = out_shape.dim(axis_u).unwrap_static();
                    let p = GatherBwdParams {
                        outer: outer as u32,
                        axis_dim: axis_dim as u32,
                        num_idx: num_idx as u32,
                        trailing: trailing as u32,
                        dy_off: (arena.offset(node.inputs[0]) / 4) as u32,
                        idx_off: (arena.offset(node.inputs[1]) / 4) as u32,
                        dst_off: (arena.offset(node.id) / 4) as u32,
                        _p0: 0,
                    };
                    let zk = gather_backward_zero_kernel(&dev.device);
                    let u = emit_uniform(std::mem::size_of::<GatherBwdParams>());
                    let bg = bind_op_output_window(&dev.device, zk, &arena, node.id, &u);
                    schedule.push(Step::GatherBackward { params: p });
                    uniforms.push(u);
                    bind_groups.push(bg);
                }
                #[cfg(feature = "splat")]
                Op::GaussianSplatRender {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRender {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatRenderBackward {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatRenderBackward {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        d_loss_byte_off: arena.offset(node.inputs[7]) as u32,
                        d_loss_len: elem_len(node.inputs[7]),
                        packed_byte_off: arena.offset(node.id) as u32,
                        packed_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                        loss_grad_clip: *loss_grad_clip,
                        sh_band: *sh_band,
                        max_anisotropy: *max_anisotropy,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatPrepare {
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    schedule.push(Step::GaussianSplatPrepare {
                        positions_byte_off: arena.offset(node.inputs[0]) as u32,
                        positions_len: elem_len(node.inputs[0]),
                        scales_byte_off: arena.offset(node.inputs[1]) as u32,
                        scales_len: elem_len(node.inputs[1]),
                        rotations_byte_off: arena.offset(node.inputs[2]) as u32,
                        rotations_len: elem_len(node.inputs[2]),
                        opacities_byte_off: arena.offset(node.inputs[3]) as u32,
                        opacities_len: elem_len(node.inputs[3]),
                        colors_byte_off: arena.offset(node.inputs[4]) as u32,
                        colors_len: elem_len(node.inputs[4]),
                        sh_coeffs_byte_off: arena.offset(node.inputs[5]) as u32,
                        sh_coeffs_len: elem_len(node.inputs[5]),
                        meta_byte_off: arena.offset(node.inputs[6]) as u32,
                        meta_len: elem_len(node.inputs[6]),
                        prep_byte_off: arena.offset(node.id) as u32,
                        prep_len: node.shape.num_elements().unwrap_or(0) as u32,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        radius_scale: *radius_scale,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                #[cfg(feature = "splat")]
                Op::GaussianSplatRasterize {
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    let elem_len = |id: NodeId| -> u32 {
                        graph.node(id).shape.num_elements().unwrap_or(0) as u32
                    };
                    let prep_id = node.inputs[0];
                    let count = match &graph.node(prep_id).op {
                        rlx_ir::Op::GaussianSplatPrepare { .. } => {
                            elem_len(graph.node(prep_id).inputs[0]) / 3
                        }
                        _ => 1,
                    };
                    schedule.push(Step::GaussianSplatRasterize {
                        prep_byte_off: arena.offset(prep_id) as u32,
                        prep_len: elem_len(prep_id),
                        meta_byte_off: arena.offset(node.inputs[1]) as u32,
                        meta_len: elem_len(node.inputs[1]),
                        dst_byte_off: arena.offset(node.id) as u32,
                        dst_len: node.shape.num_elements().unwrap_or(0) as u32,
                        count,
                        width: *width,
                        height: *height,
                        tile_size: *tile_size,
                        alpha_cutoff: *alpha_cutoff,
                        max_splat_steps: *max_splat_steps,
                        transmittance_threshold: *transmittance_threshold,
                        max_list_entries: *max_list_entries,
                    });
                }

                Op::If { .. } | Op::While { .. } => {
                    // Should be unreachable: unfuse.rs inlines both branches
                    // (If) or unrolls max_iterations (While) into the parent
                    // graph using primitive ops + Where for the gating. If
                    // we hit this arm, the unfusion pass has a gap.
                    panic!(
                        "rlx-wgpu: Op::If/While leaked past unfusion pass — \
                            check unfuse.rs::expand_if / expand_while"
                    );
                }
                Op::RngNormal {
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::RngNormalHost {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len,
                        mean: *mean,
                        scale: *scale,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                Op::RngUniform {
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let len = node.shape.num_elements().unwrap_or(0) as u32;
                    schedule.push(Step::RngUniformHost {
                        dst_byte_off: arena.offset(node.id) as u32,
                        len,
                        low: *low,
                        high: *high,
                        key: *key,
                        op_seed: *op_seed,
                    });
                }
                // Standalone nearest 2× upsample: the region-marking pass wraps
                // a bare `Op::ResizeNearest2x` into a single-step TransformRegion.
                // Route that to the host resize helper (mirrors the bare arm).
                Op::TransformRegion { steps, .. }
                    if steps.len() == 1
                        && matches!(
                            steps[0],
                            rlx_ir::op::TransformStep::ResizeNearest2x(
                                rlx_ir::op::ChainOperand::Input(0)
                            )
                        ) =>
                {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    schedule.push(Step::ResizeNearest2xHost {
                        src_byte_off: arena.offset(node.inputs[0]) as u32,
                        dst_byte_off: arena.offset(node.id) as u32,
                        n: in_shape.dim(0).unwrap_static() as u32,
                        c: in_shape.dim(1).unwrap_static() as u32,
                        h: in_shape.dim(2).unwrap_static() as u32,
                        w: in_shape.dim(3).unwrap_static() as u32,
                    });
                }
                other => panic!(
                    "rlx-wgpu: op {other:?} not yet lowered (v2 covers Matmul, \
                     Binary, Compare, Activation, Where — fall back to CPU/Metal/MLX)"
                ),
            }
        }

        if rlx_ir::env::flag("RLX_WGPU_SCHEDULE") || rlx_ir::env::flag("RLX_DISPATCH_REPORT") {
            let mut counts: std::collections::BTreeMap<&'static str, usize> =
                std::collections::BTreeMap::new();
            let mut fft_gpu = 0usize;
            let mut fft_host = 0usize;
            for s in &schedule {
                *counts.entry(step_name(s)).or_insert(0) += 1;
                match s {
                    Step::FftGpu { .. } => fft_gpu += 1,
                    Step::FftHost { .. } => fft_host += 1,
                    _ => {}
                }
            }
            let arena_mb = arena.size as f64 / (1u64 << 20) as f64;
            eprintln!(
                "[rlx-wgpu] schedule: {} steps, arena={arena_mb:.1} MiB, fft_gpu={fft_gpu}, fft_host={fft_host}",
                schedule.len()
            );
            for (n, c) in &counts {
                eprintln!("    {c:>4} × {n}");
            }
        }

        let coop_f16_vk = schedule_uses_coop_f16_vk(&schedule);

        Self {
            graph,
            arena,
            dequant_scratch_off,
            schedule,
            input_offsets,
            param_offsets,
            uniforms,
            bind_groups,
            meta_buffers,
            unresolved: None,
            last_binding: None,
            pending_params: HashMap::new(),
            pending_param_bytes: HashMap::new(),
            active_extent: None,
            uniforms_active_extent: None,
            input_staging_hashes: HashMap::new(),
            coop_f16_vk,
            coop_f16_b_param,
            coop_f16_vk_wide_b: HashSet::new(),
            coop_f16_vk_wide_bind_groups,
            coop_f16_host_activations,
            stashed_params: HashMap::new(),
            readback_staging: None,
            tiny_readback: None,
            dispatch_only: false,
            fft_gpu_steps,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            pending_read_indices: None,
            rng,
        }
    }
}
