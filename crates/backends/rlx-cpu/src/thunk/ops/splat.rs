#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_gaussian_splat_render(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GaussianSplatRender {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let elem_len = |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
        Thunk::GaussianSplatRender {
            positions_off: node_offset(arena, node.inputs[0]),
            positions_len: elem_len(node.inputs[0]),
            scales_off: node_offset(arena, node.inputs[1]),
            scales_len: elem_len(node.inputs[1]),
            rotations_off: node_offset(arena, node.inputs[2]),
            rotations_len: elem_len(node.inputs[2]),
            opacities_off: node_offset(arena, node.inputs[3]),
            opacities_len: elem_len(node.inputs[3]),
            colors_off: node_offset(arena, node.inputs[4]),
            colors_len: elem_len(node.inputs[4]),
            sh_coeffs_off: node_offset(arena, node.inputs[5]),
            sh_coeffs_len: elem_len(node.inputs[5]),
            meta_off: node_offset(arena, node.inputs[6]),
            dst_off: node_offset(arena, node.id),
            dst_len: node.shape.num_elements().unwrap_or(0),
            width: *width,
            height: *height,
            tile_size: *tile_size,
            radius_scale: *radius_scale,
            alpha_cutoff: *alpha_cutoff,
            max_splat_steps: *max_splat_steps,
            transmittance_threshold: *transmittance_threshold,
            max_list_entries: *max_list_entries,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gaussian_splat_render_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GaussianSplatRenderBackward {
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
    } = &node.op
    else {
        unreachable!()
    };
    {
        let elem_len = |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
        Thunk::GaussianSplatRenderBackward {
            positions_off: node_offset(arena, node.inputs[0]),
            positions_len: elem_len(node.inputs[0]),
            scales_off: node_offset(arena, node.inputs[1]),
            scales_len: elem_len(node.inputs[1]),
            rotations_off: node_offset(arena, node.inputs[2]),
            rotations_len: elem_len(node.inputs[2]),
            opacities_off: node_offset(arena, node.inputs[3]),
            opacities_len: elem_len(node.inputs[3]),
            colors_off: node_offset(arena, node.inputs[4]),
            colors_len: elem_len(node.inputs[4]),
            sh_coeffs_off: node_offset(arena, node.inputs[5]),
            sh_coeffs_len: elem_len(node.inputs[5]),
            meta_off: node_offset(arena, node.inputs[6]),
            d_loss_off: node_offset(arena, node.inputs[7]),
            d_loss_len: elem_len(node.inputs[7]),
            packed_off: node_offset(arena, node.id),
            packed_len: node.shape.num_elements().unwrap_or(0),
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
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gaussian_splat_prepare(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GaussianSplatPrepare {
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let elem_len = |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
        Thunk::GaussianSplatPrepare {
            positions_off: node_offset(arena, node.inputs[0]),
            positions_len: elem_len(node.inputs[0]),
            scales_off: node_offset(arena, node.inputs[1]),
            scales_len: elem_len(node.inputs[1]),
            rotations_off: node_offset(arena, node.inputs[2]),
            rotations_len: elem_len(node.inputs[2]),
            opacities_off: node_offset(arena, node.inputs[3]),
            opacities_len: elem_len(node.inputs[3]),
            colors_off: node_offset(arena, node.inputs[4]),
            colors_len: elem_len(node.inputs[4]),
            sh_coeffs_off: node_offset(arena, node.inputs[5]),
            sh_coeffs_len: elem_len(node.inputs[5]),
            meta_off: node_offset(arena, node.inputs[6]),
            meta_len: elem_len(node.inputs[6]),
            prep_off: node_offset(arena, node.id),
            prep_len: node.shape.num_elements().unwrap_or(0),
            width: *width,
            height: *height,
            tile_size: *tile_size,
            radius_scale: *radius_scale,
            alpha_cutoff: *alpha_cutoff,
            max_splat_steps: *max_splat_steps,
            transmittance_threshold: *transmittance_threshold,
            max_list_entries: *max_list_entries,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gaussian_splat_rasterize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GaussianSplatRasterize {
        width,
        height,
        tile_size,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let elem_len = |id: NodeId| -> usize { graph.node(id).shape.num_elements().unwrap_or(0) };
        let prep_id = node.inputs[0];
        let count = match &graph.node(prep_id).op {
            rlx_ir::Op::GaussianSplatPrepare { .. } => elem_len(graph.node(prep_id).inputs[0]) / 3,
            _ => 1,
        };
        Thunk::GaussianSplatRasterize {
            prep_off: node_offset(arena, prep_id),
            prep_len: elem_len(prep_id),
            meta_off: node_offset(arena, node.inputs[1]),
            meta_len: elem_len(node.inputs[1]),
            dst_off: node_offset(arena, node.id),
            dst_len: node.shape.num_elements().unwrap_or(0),
            count,
            width: *width,
            height: *height,
            tile_size: *tile_size,
            alpha_cutoff: *alpha_cutoff,
            max_splat_steps: *max_splat_steps,
            transmittance_threshold: *transmittance_threshold,
            max_list_entries: *max_list_entries,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gaussian_splat_render(t: &Thunk, base: *mut u8) {
    let Thunk::GaussianSplatRender {
        positions_off,
        positions_len,
        scales_off,
        scales_len,
        rotations_off,
        rotations_len,
        opacities_off,
        opacities_len,
        colors_off,
        colors_len,
        sh_coeffs_off,
        sh_coeffs_len,
        meta_off,
        dst_off,
        dst_len,
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        crate::splat::execute_gaussian_splat_render(
            *positions_off,
            *positions_len,
            *scales_off,
            *scales_len,
            *rotations_off,
            *rotations_len,
            *opacities_off,
            *opacities_len,
            *colors_off,
            *colors_len,
            *sh_coeffs_off,
            *sh_coeffs_len,
            *meta_off,
            *dst_off,
            *dst_len,
            *width,
            *height,
            *tile_size,
            *radius_scale,
            *alpha_cutoff,
            *max_splat_steps,
            *transmittance_threshold,
            *max_list_entries,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_gaussian_splat_render_backward(t: &Thunk, base: *mut u8) {
    let Thunk::GaussianSplatRenderBackward {
        positions_off,
        positions_len,
        scales_off,
        scales_len,
        rotations_off,
        rotations_len,
        opacities_off,
        opacities_len,
        colors_off,
        colors_len,
        sh_coeffs_off,
        sh_coeffs_len,
        meta_off,
        d_loss_off,
        d_loss_len,
        packed_off,
        packed_len,
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
    } = t
    else {
        unreachable!()
    };
    unsafe {
        crate::splat::execute_gaussian_splat_render_backward(
            *positions_off,
            *positions_len,
            *scales_off,
            *scales_len,
            *rotations_off,
            *rotations_len,
            *opacities_off,
            *opacities_len,
            *colors_off,
            *colors_len,
            *sh_coeffs_off,
            *sh_coeffs_len,
            *meta_off,
            *d_loss_off,
            *d_loss_len,
            *packed_off,
            *packed_len,
            *width,
            *height,
            *tile_size,
            *radius_scale,
            *alpha_cutoff,
            *max_splat_steps,
            *transmittance_threshold,
            *max_list_entries,
            *loss_grad_clip,
            *sh_band,
            *max_anisotropy,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_gaussian_splat_prepare(t: &Thunk, base: *mut u8) {
    let Thunk::GaussianSplatPrepare {
        positions_off,
        positions_len,
        scales_off,
        scales_len,
        rotations_off,
        rotations_len,
        opacities_off,
        opacities_len,
        colors_off,
        colors_len,
        sh_coeffs_off,
        sh_coeffs_len,
        meta_off,
        meta_len,
        prep_off,
        prep_len,
        width,
        height,
        tile_size,
        radius_scale,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        crate::splat::execute_gaussian_splat_prepare(
            *positions_off,
            *positions_len,
            *scales_off,
            *scales_len,
            *rotations_off,
            *rotations_len,
            *opacities_off,
            *opacities_len,
            *colors_off,
            *colors_len,
            *sh_coeffs_off,
            *sh_coeffs_len,
            *meta_off,
            *meta_len,
            *prep_off,
            *prep_len,
            *width,
            *height,
            *tile_size,
            *radius_scale,
            *alpha_cutoff,
            *max_splat_steps,
            *transmittance_threshold,
            *max_list_entries,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_gaussian_splat_rasterize(t: &Thunk, base: *mut u8) {
    let Thunk::GaussianSplatRasterize {
        prep_off,
        prep_len,
        meta_off,
        meta_len,
        dst_off,
        dst_len,
        count,
        width,
        height,
        tile_size,
        alpha_cutoff,
        max_splat_steps,
        transmittance_threshold,
        max_list_entries,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        crate::splat::execute_gaussian_splat_rasterize(
            *prep_off,
            *prep_len,
            *meta_off,
            *meta_len,
            *dst_off,
            *dst_len,
            *count,
            *width,
            *height,
            *tile_size,
            *alpha_cutoff,
            *max_splat_steps,
            *transmittance_threshold,
            *max_list_entries,
            base,
        );
    }
}
