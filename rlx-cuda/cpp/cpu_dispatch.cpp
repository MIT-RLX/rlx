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

// HIP-CPU dispatch entry points for the rlx-cuda validation path.
// Compiled only when `cargo build --features hip-cpu-validate`.
//
// HIP-CPU executes "GPU" kernels on CPU threads via std::thread. The
// kernel sources in `src/kernels/*.cu` use plain CUDA syntax —
// `__global__`, `blockIdx`, `threadIdx`, `__shared__` — which HIP-CPU
// recognizes when `__HIP_CPU_RT__` is defined.
//
// Each `launch_<kernel>` wraps `hipLaunchKernelGGL` so the Rust side
// can call into a stable C ABI. Coverage: all 32 kernel entry points
// (= 30 .cu sources, with matmul + scatter_add each contributing one
// extra entry).

#include <hip/hip_runtime.h>

#include "binary.cu"
#include "fused_binary_unary.cu"
#include "unary.cu"
#include "copy.cu"
#include "matmul.cu"
#include "compare.cu"
#include "where_select.cu"
#include "reduce.cu"
#include "softmax.cu"
#include "layernorm.cu"
#include "fused_residual_ln.cu"
#include "gather.cu"
#include "narrow.cu"
#include "concat.cu"
#include "transpose.cu"
#include "expand.cu"
#include "attention.cu"
#include "argmax.cu"
#include "rope.cu"
#include "cumsum.cu"
#include "topk.cu"
#include "grouped_matmul.cu"
#include "scatter_add.cu"
#include "dequant_matmul.cu"
#include "sample.cu"
#include "selective_scan.cu"
#include "pool1d.cu"
#include "pool2d.cu"
#include "pool3d.cu"
#include "conv1d.cu"
#include "conv2d.cu"
#include "conv3d.cu"
#include "elementwise_region.cu"

#define LAUNCH(kfunc, gx, gy, gz, bx, by, bz, ...)                          \
    do {                                                                    \
        hipLaunchKernelGGL(kfunc, dim3((gx), (gy), (gz)),                   \
                                  dim3((bx), (by), (bz)), 0, 0,             \
                           __VA_ARGS__);                                    \
        hipDeviceSynchronize();                                             \
    } while (0)

extern "C" {

// ── Element-wise (1-D dispatch, block_x = 256) ─────────────────────

void launch_binary(float* a, unsigned int n, unsigned int ao, unsigned int bo,
                   unsigned int co, unsigned int op,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(binary, gx,1,1, bx,1,1, a, n, ao, bo, co, op);
}

void launch_unary(float* a, unsigned int n, unsigned int io, unsigned int oo,
                  unsigned int op, unsigned int gx, unsigned int bx) {
    LAUNCH(unary, gx,1,1, bx,1,1, a, n, io, oo, op);
}

void launch_copy(float* a, unsigned int n, unsigned int io, unsigned int oo,
                 unsigned int gx, unsigned int bx) {
    LAUNCH(copy, gx,1,1, bx,1,1, a, n, io, oo);
}

void launch_compare(float* a, unsigned int n, unsigned int ao, unsigned int bo,
                    unsigned int co, unsigned int op,
                    unsigned int gx, unsigned int bx) {
    LAUNCH(compare, gx,1,1, bx,1,1, a, n, ao, bo, co, op);
}

void launch_where_select(float* a, unsigned int n, unsigned int cond_o,
                         unsigned int xo, unsigned int yo, unsigned int oo,
                         unsigned int gx, unsigned int bx) {
    LAUNCH(where_select, gx,1,1, bx,1,1, a, n, cond_o, xo, yo, oo);
}

// ── MatMul + DequantMatMul + GroupedMatmul (2-D dispatch) ──────────

void launch_matmul(float* a,
                   unsigned int m, unsigned int k, unsigned int n,
                   unsigned int ao, unsigned int bo, unsigned int co,
                   unsigned int batch,
                   unsigned int abs_, unsigned int bbs, unsigned int cbs,
                   unsigned int has_bias, unsigned int bias_off,
                   unsigned int act_id,
                   unsigned int gx, unsigned int gy, unsigned int gz,
                   unsigned int bx, unsigned int by) {
    LAUNCH(matmul, gx,gy,gz, bx,by,1,
        a, m,k,n, ao,bo,co, batch, abs_,bbs,cbs, has_bias,bias_off,act_id);
}

void launch_grouped_matmul(float* a,
                           unsigned int m, unsigned int k, unsigned int n,
                           unsigned int num_experts,
                           unsigned int io, unsigned int wo,
                           unsigned int idx_o, unsigned int oo,
                           unsigned int gx, unsigned int gy,
                           unsigned int bx, unsigned int by) {
    LAUNCH(grouped_matmul, gx,gy,1, bx,by,1,
        a, m,k,n, num_experts, io,wo,idx_o,oo);
}

void launch_dequant_matmul(float* a,
                           unsigned int m, unsigned int k, unsigned int n,
                           unsigned int block_size, unsigned int scheme_id,
                           unsigned int xo, unsigned int wo,
                           unsigned int sco, unsigned int zo, unsigned int oo,
                           unsigned int gx, unsigned int gy,
                           unsigned int bx, unsigned int by) {
    LAUNCH(dequant_matmul, gx,gy,1, bx,by,1,
        a, m,k,n, block_size, scheme_id, xo,wo,sco,zo,oo);
}

// ── Reductions (1-D over outer rows) ───────────────────────────────

void launch_reduce(float* a, unsigned int outer, unsigned int inner,
                   unsigned int io, unsigned int oo, unsigned int op,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(reduce, gx,1,1, bx,1,1, a, outer,inner, io,oo, op);
}

void launch_softmax(float* a, unsigned int outer, unsigned int inner,
                    unsigned int io, unsigned int oo,
                    unsigned int gx, unsigned int bx) {
    LAUNCH(softmax, gx,1,1, bx,1,1, a, outer,inner, io,oo);
}

void launch_layernorm(float* a, unsigned int outer, unsigned int inner,
                      unsigned int io, unsigned int oo,
                      unsigned int go, unsigned int beta_o,
                      unsigned int eps_bits, unsigned int op,
                      unsigned int gx, unsigned int bx) {
    LAUNCH(rlx_norm, gx,1,1, bx,1,1, a, outer,inner, io,oo, go,beta_o, eps_bits, op);
}

void launch_fused_residual_ln(float* a, unsigned int outer, unsigned int inner,
                              unsigned int io, unsigned int ro,
                              unsigned int bias_o, unsigned int go, unsigned int beta_o,
                              unsigned int oo, unsigned int eps_bits,
                              unsigned int has_bias,
                              unsigned int gx, unsigned int bx) {
    LAUNCH(fused_residual_ln, gx,1,1, bx,1,1,
        a, outer,inner, io,ro, bias_o,go,beta_o,oo, eps_bits, has_bias);
}

void launch_cumsum(float* a, unsigned int outer, unsigned int inner,
                   unsigned int io, unsigned int oo, unsigned int exclusive,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(cumsum, gx,1,1, bx,1,1, a, outer,inner, io,oo, exclusive);
}

void launch_argmax(float* a, unsigned int outer, unsigned int inner,
                   unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(argmax, gx,1,1, bx,1,1, a, outer,inner, io,oo);
}

void launch_topk(float* a, unsigned int outer, unsigned int inner,
                 unsigned int k, unsigned int io, unsigned int oo,
                 unsigned int gx, unsigned int bx) {
    LAUNCH(topk, gx,1,1, bx,1,1, a, outer,inner, k, io,oo);
}

// ── Shape ops ───────────────────────────────────────────────────────

void launch_gather(float* a, unsigned int n_out, unsigned int n_idx,
                   unsigned int dim, unsigned int vocab,
                   unsigned int io, unsigned int idx_o, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(gather, gx,1,1, bx,1,1, a, n_out,n_idx, dim,vocab, io,idx_o,oo);
}

void launch_narrow(float* a, unsigned int total, unsigned int outer,
                   unsigned int inner, unsigned int axis_in,
                   unsigned int axis_out, unsigned int start,
                   unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(narrow, gx,1,1, bx,1,1, a, total,outer,inner, axis_in,axis_out, start, io,oo);
}

void launch_concat(float* a, unsigned int total, unsigned int outer,
                   unsigned int inner, unsigned int axis_in,
                   unsigned int axis_out, unsigned int start,
                   unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(concat, gx,1,1, bx,1,1, a, total,outer,inner, axis_in,axis_out, start, io,oo);
}

void launch_transpose(float* a, unsigned int rank, unsigned int out_total,
                      unsigned int io, unsigned int oo, const unsigned int* meta,
                      unsigned int gx, unsigned int bx) {
    LAUNCH(transpose, gx,1,1, bx,1,1, a, rank, out_total, io, oo, meta);
}

void launch_expand(float* a, unsigned int rank, unsigned int out_total,
                   unsigned int io, unsigned int oo, const unsigned int* meta,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(expand, gx,1,1, bx,1,1, a, rank, out_total, io, oo, meta);
}

// ── Attention + Rope ───────────────────────────────────────────────

void launch_attention(float* a,
                      unsigned int batch, unsigned int heads,
                      unsigned int seq_q, unsigned int seq_k,
                      unsigned int head_dim,
                      unsigned int qo, unsigned int ko,
                      unsigned int vo, unsigned int oo,
                      unsigned int mask_o, unsigned int mask_kind,
                      unsigned int scale_bits, unsigned int window,
                      unsigned int gx, unsigned int bx) {
    LAUNCH(attention, gx,1,1, bx,1,1,
        a, batch,heads,seq_q,seq_k,head_dim,
        qo,ko,vo,oo, mask_o,mask_kind,scale_bits,window);
}

void launch_rope(float* a, unsigned int n_total, unsigned int seq,
                 unsigned int head_dim, unsigned int half,
                 unsigned int io, unsigned int co, unsigned int so, unsigned int oo,
                 unsigned int last_dim,
                 unsigned int gx, unsigned int bx) {
    LAUNCH(rope, gx,1,1, bx,1,1,
        a, n_total,seq,head_dim,half, io,co,so,oo, last_dim);
}

// ── ScatterAdd (two phases) ────────────────────────────────────────

void launch_scatter_add_zero(float* a, unsigned int oo, unsigned int total,
                             unsigned int gx, unsigned int bx) {
    LAUNCH(scatter_add_zero, gx,1,1, bx,1,1, a, oo, total);
}

void launch_scatter_add_acc(float* a, unsigned int oo, unsigned int upd_o,
                            unsigned int idx_o, unsigned int n_upd,
                            unsigned int trailing, unsigned int out_dim,
                            unsigned int gx, unsigned int bx) {
    LAUNCH(scatter_add_acc, gx,1,1, bx,1,1,
        a, oo, upd_o, idx_o, n_upd, trailing, out_dim);
}

// ── Sample + SelectiveScan ─────────────────────────────────────────

void launch_sample(float* a, unsigned int outer, unsigned int inner,
                   unsigned int io, unsigned int oo,
                   unsigned int top_k, unsigned int top_p_bits,
                   unsigned int temp_bits,
                   unsigned int seed_lo, unsigned int seed_hi,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(sample, gx,1,1, bx,1,1,
        a, outer,inner, io,oo, top_k, top_p_bits, temp_bits, seed_lo, seed_hi);
}

void launch_selective_scan(float* a, unsigned int batch, unsigned int seq,
                           unsigned int hidden, unsigned int state_size,
                           unsigned int xo, unsigned int dt_o,
                           unsigned int ao, unsigned int bo,
                           unsigned int co, unsigned int oo,
                           unsigned int gx, unsigned int bx) {
    LAUNCH(selective_scan, gx,1,1, bx,1,1,
        a, batch,seq,hidden,state_size, xo,dt_o,ao,bo,co,oo);
}

// ── Pool / Conv (1D, 2D, 3D) ───────────────────────────────────────

void launch_pool1d(float* a, unsigned int n, unsigned int c, unsigned int l,
                   unsigned int l_out, unsigned int kl, unsigned int sl,
                   unsigned int pl, unsigned int op,
                   unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(pool1d, gx,1,1, bx,1,1, a, n,c,l, l_out, kl,sl,pl, op, io,oo);
}

void launch_pool2d(float* a, unsigned int n, unsigned int c, unsigned int h,
                   unsigned int w, unsigned int h_out, unsigned int w_out,
                   unsigned int kh, unsigned int kw,
                   unsigned int sh, unsigned int sw,
                   unsigned int ph, unsigned int pw, unsigned int op,
                   unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(pool2d, gx,1,1, bx,1,1,
        a, n,c,h,w, h_out,w_out, kh,kw, sh,sw, ph,pw, op, io,oo);
}

void launch_pool3d(float* a, unsigned int n, unsigned int c,
                   unsigned int d, unsigned int h, unsigned int w,
                   unsigned int d_out, unsigned int h_out, unsigned int w_out,
                   unsigned int kd, unsigned int kh, unsigned int kw,
                   unsigned int sd, unsigned int sh, unsigned int sw,
                   unsigned int pd, unsigned int ph, unsigned int pw,
                   unsigned int op, unsigned int io, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(pool3d, gx,1,1, bx,1,1,
        a, n,c,d,h,w, d_out,h_out,w_out,
        kd,kh,kw, sd,sh,sw, pd,ph,pw, op, io,oo);
}

void launch_conv1d(float* a, unsigned int n, unsigned int c_in,
                   unsigned int c_out, unsigned int l, unsigned int l_out,
                   unsigned int kl, unsigned int sl, unsigned int pl,
                   unsigned int dl, unsigned int groups,
                   unsigned int io, unsigned int wo, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(conv1d, gx,1,1, bx,1,1,
        a, n,c_in,c_out, l, l_out, kl,sl,pl,dl, groups, io,wo,oo);
}

void launch_conv2d(float* a, unsigned int n, unsigned int c_in,
                   unsigned int c_out, unsigned int h, unsigned int w,
                   unsigned int h_out, unsigned int w_out,
                   unsigned int kh, unsigned int kw,
                   unsigned int sh, unsigned int sw,
                   unsigned int ph, unsigned int pw,
                   unsigned int dh, unsigned int dw, unsigned int groups,
                   unsigned int io, unsigned int wo, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(conv2d, gx,1,1, bx,1,1,
        a, n,c_in,c_out, h,w, h_out,w_out, kh,kw, sh,sw, ph,pw, dh,dw,
        groups, io,wo,oo);
}

void launch_conv3d(float* a, unsigned int n, unsigned int c_in,
                   unsigned int c_out, unsigned int d, unsigned int h,
                   unsigned int w, unsigned int d_out, unsigned int h_out,
                   unsigned int w_out,
                   unsigned int kd, unsigned int kh, unsigned int kw,
                   unsigned int sd, unsigned int sh, unsigned int sw,
                   unsigned int pd, unsigned int ph, unsigned int pw,
                   unsigned int dd, unsigned int dh, unsigned int dw,
                   unsigned int groups,
                   unsigned int io, unsigned int wo, unsigned int oo,
                   unsigned int gx, unsigned int bx) {
    LAUNCH(conv3d, gx,1,1, bx,1,1,
        a, n,c_in,c_out, d,h,w, d_out,h_out,w_out,
        kd,kh,kw, sd,sh,sw, pd,ph,pw, dd,dh,dw,
        groups, io,wo,oo);
}

void launch_fused_binary_unary(float* a, unsigned int n,
                               unsigned int ao, unsigned int bo, unsigned int oo,
                               unsigned int bin_op, unsigned int un_op,
                               unsigned int gx, unsigned int bx) {
    LAUNCH(fused_binary_unary, gx,1,1, bx,1,1,
        a, n, ao, bo, oo, bin_op, un_op);
}

void launch_elementwise_region(float* a, unsigned int len,
                               unsigned int num_inputs, unsigned int num_steps,
                               unsigned int dst_off, const unsigned int* meta,
                               unsigned int scalar_input_mask,
                               const unsigned int* input_modulus,
                               unsigned int gx, unsigned int bx) {
    // Pack the modulus into the same struct the .cu kernel expects.
    InputModulus mod_struct;
    for (int i = 0; i < 16; ++i) mod_struct.v[i] = input_modulus[i];
    LAUNCH(elementwise_region, gx,1,1, bx,1,1,
        a, len, num_inputs, num_steps, dst_off, meta,
        scalar_input_mask, mod_struct);
}

}  // extern "C"
