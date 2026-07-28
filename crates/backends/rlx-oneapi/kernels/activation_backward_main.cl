// Plumbing half of the `activation_backward` kernel. The derivative math is
// @generated once in `rlxsl::opencl_activation_backward_module` (relu-first ids,
// matching `activation_bwd_op_id` in src/backend.rs — the CUDA/wgpu scheme, NOT
// the forward unary switch) and prepended to this file by build.rs. The gradient
// is auto-differentiated from the forward manifest, so it matches the forward we
// ship and stays in lockstep with every other backend.
//
// dx = d(activation)/dx · dy.
__kernel void activation_backward(__global float* arena,
                                  uint n,
                                  uint x_off, uint dy_off, uint dx_off,
                                  uint op) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float x = arena[x_off + i];
    float dy = arena[dy_off + i];
    arena[dx_off + i] = rlx_activation_backward(op, x, dy);
}
