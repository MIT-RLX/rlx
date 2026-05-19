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

//! Rust FFI bindings to the HIP-CPU validation path.
//!
//! Only compiled when `cargo build --features hip-cpu-validate`. The
//! corresponding C++ launch wrappers live in `cpp/cpu_dispatch.cpp`,
//! built by `build.rs` against HIP-CPU's header-only runtime.
//!
//! This is a **dev-only** validation surface. Production CUDA dispatch
//! goes through `crate::backend::CudaExecutable` + cudarc + libcuda on
//! a real NVIDIA host. The CPU path lets us run the same `.cu` kernel
//! sources on CPU threads from Mac (or any host without an NVIDIA
//! driver) so we can catch IR-lowering and kernel-logic bugs before
//! paying for cloud-GPU time.
//!
//! Coverage: all 32 kernel entry points exposed by
//! `cpp/cpu_dispatch.cpp` (= 30 `.cu` files, with matmul + scatter_add
//! contributing extras). Each kernel exposes:
//!   • `extern "C" launch_<name>` — raw FFI entry
//!   • a safe Rust wrapper that picks a default dispatch grid

#![cfg(feature = "hip-cpu-validate")]

use std::os::raw::c_uint;

const BLOCK_X: u32 = 256;

// ── Raw FFI declarations (mirror cpp/cpu_dispatch.cpp) ───────────────

unsafe extern "C" {
    pub fn launch_binary(
        a: *mut f32,
        n: c_uint,
        ao: c_uint,
        bo: c_uint,
        co: c_uint,
        op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_fused_binary_unary(
        a: *mut f32,
        n: c_uint,
        ao: c_uint,
        bo: c_uint,
        oo: c_uint,
        bin_op: c_uint,
        un_op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_unary(
        a: *mut f32,
        n: c_uint,
        io: c_uint,
        oo: c_uint,
        op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_copy(a: *mut f32, n: c_uint, io: c_uint, oo: c_uint, gx: c_uint, bx: c_uint);
    pub fn launch_compare(
        a: *mut f32,
        n: c_uint,
        ao: c_uint,
        bo: c_uint,
        co: c_uint,
        op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_where_select(
        a: *mut f32,
        n: c_uint,
        cond_o: c_uint,
        xo: c_uint,
        yo: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_matmul(
        a: *mut f32,
        m: c_uint,
        k: c_uint,
        n: c_uint,
        ao: c_uint,
        bo: c_uint,
        co: c_uint,
        batch: c_uint,
        abs_: c_uint,
        bbs: c_uint,
        cbs: c_uint,
        has_bias: c_uint,
        bias_off: c_uint,
        act_id: c_uint,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
    );
    pub fn launch_grouped_matmul(
        a: *mut f32,
        m: c_uint,
        k: c_uint,
        n: c_uint,
        num_experts: c_uint,
        io: c_uint,
        wo: c_uint,
        idx_o: c_uint,
        oo: c_uint,
        gx: c_uint,
        gy: c_uint,
        bx: c_uint,
        by: c_uint,
    );
    pub fn launch_dequant_matmul(
        a: *mut f32,
        m: c_uint,
        k: c_uint,
        n: c_uint,
        block_size: c_uint,
        scheme_id: c_uint,
        xo: c_uint,
        wo: c_uint,
        sco: c_uint,
        zo: c_uint,
        oo: c_uint,
        gx: c_uint,
        gy: c_uint,
        bx: c_uint,
        by: c_uint,
    );

    pub fn launch_reduce(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_softmax(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_layernorm(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        go: c_uint,
        beta_o: c_uint,
        eps_bits: c_uint,
        op: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_fused_residual_ln(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        ro: c_uint,
        bias_o: c_uint,
        go: c_uint,
        beta_o: c_uint,
        oo: c_uint,
        eps_bits: c_uint,
        has_bias: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_cumsum(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        exclusive: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_argmax(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_topk(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        k: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_gather(
        a: *mut f32,
        n_out: c_uint,
        n_idx: c_uint,
        dim: c_uint,
        vocab: c_uint,
        io: c_uint,
        idx_o: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_narrow(
        a: *mut f32,
        total: c_uint,
        outer: c_uint,
        inner: c_uint,
        axis_in: c_uint,
        axis_out: c_uint,
        start: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_concat(
        a: *mut f32,
        total: c_uint,
        outer: c_uint,
        inner: c_uint,
        axis_in: c_uint,
        axis_out: c_uint,
        start: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_transpose(
        a: *mut f32,
        rank: c_uint,
        out_total: c_uint,
        io: c_uint,
        oo: c_uint,
        meta: *const c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_expand(
        a: *mut f32,
        rank: c_uint,
        out_total: c_uint,
        io: c_uint,
        oo: c_uint,
        meta: *const c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_attention(
        a: *mut f32,
        batch: c_uint,
        heads: c_uint,
        seq_q: c_uint,
        seq_k: c_uint,
        head_dim: c_uint,
        qo: c_uint,
        ko: c_uint,
        vo: c_uint,
        oo: c_uint,
        mask_o: c_uint,
        mask_kind: c_uint,
        scale_bits: c_uint,
        window: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_rope(
        a: *mut f32,
        n_total: c_uint,
        seq: c_uint,
        head_dim: c_uint,
        half: c_uint,
        io: c_uint,
        co: c_uint,
        so: c_uint,
        oo: c_uint,
        last_dim: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_scatter_add_zero(a: *mut f32, oo: c_uint, total: c_uint, gx: c_uint, bx: c_uint);
    pub fn launch_scatter_add_acc(
        a: *mut f32,
        oo: c_uint,
        upd_o: c_uint,
        idx_o: c_uint,
        n_upd: c_uint,
        trailing: c_uint,
        out_dim: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_sample(
        a: *mut f32,
        outer: c_uint,
        inner: c_uint,
        io: c_uint,
        oo: c_uint,
        top_k: c_uint,
        top_p_bits: c_uint,
        temp_bits: c_uint,
        seed_lo: c_uint,
        seed_hi: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_selective_scan(
        a: *mut f32,
        batch: c_uint,
        seq: c_uint,
        hidden: c_uint,
        state_size: c_uint,
        xo: c_uint,
        dt_o: c_uint,
        ao: c_uint,
        bo: c_uint,
        co: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_pool1d(
        a: *mut f32,
        n: c_uint,
        c: c_uint,
        l: c_uint,
        l_out: c_uint,
        kl: c_uint,
        sl: c_uint,
        pl: c_uint,
        op: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_pool2d(
        a: *mut f32,
        n: c_uint,
        c: c_uint,
        h: c_uint,
        w: c_uint,
        h_out: c_uint,
        w_out: c_uint,
        kh: c_uint,
        kw: c_uint,
        sh: c_uint,
        sw: c_uint,
        ph: c_uint,
        pw: c_uint,
        op: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_pool3d(
        a: *mut f32,
        n: c_uint,
        c: c_uint,
        d: c_uint,
        h: c_uint,
        w: c_uint,
        d_out: c_uint,
        h_out: c_uint,
        w_out: c_uint,
        kd: c_uint,
        kh: c_uint,
        kw: c_uint,
        sd: c_uint,
        sh: c_uint,
        sw: c_uint,
        pd: c_uint,
        ph: c_uint,
        pw: c_uint,
        op: c_uint,
        io: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_conv1d(
        a: *mut f32,
        n: c_uint,
        c_in: c_uint,
        c_out: c_uint,
        l: c_uint,
        l_out: c_uint,
        kl: c_uint,
        sl: c_uint,
        pl: c_uint,
        dl: c_uint,
        groups: c_uint,
        io: c_uint,
        wo: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_conv2d(
        a: *mut f32,
        n: c_uint,
        c_in: c_uint,
        c_out: c_uint,
        h: c_uint,
        w: c_uint,
        h_out: c_uint,
        w_out: c_uint,
        kh: c_uint,
        kw: c_uint,
        sh: c_uint,
        sw: c_uint,
        ph: c_uint,
        pw: c_uint,
        dh: c_uint,
        dw: c_uint,
        groups: c_uint,
        io: c_uint,
        wo: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );
    pub fn launch_conv3d(
        a: *mut f32,
        n: c_uint,
        c_in: c_uint,
        c_out: c_uint,
        d: c_uint,
        h: c_uint,
        w: c_uint,
        d_out: c_uint,
        h_out: c_uint,
        w_out: c_uint,
        kd: c_uint,
        kh: c_uint,
        kw: c_uint,
        sd: c_uint,
        sh: c_uint,
        sw: c_uint,
        pd: c_uint,
        ph: c_uint,
        pw: c_uint,
        dd: c_uint,
        dh: c_uint,
        dw: c_uint,
        groups: c_uint,
        io: c_uint,
        wo: c_uint,
        oo: c_uint,
        gx: c_uint,
        bx: c_uint,
    );

    pub fn launch_elementwise_region(
        a: *mut f32,
        len: c_uint,
        num_inputs: c_uint,
        num_steps: c_uint,
        dst_off: c_uint,
        meta: *const c_uint,
        scalar_input_mask: c_uint,
        input_modulus: *const c_uint,
        gx: c_uint,
        bx: c_uint,
    );
}

#[inline]
fn grid_1d(n: u32) -> u32 {
    (n + BLOCK_X - 1) / BLOCK_X
}

// ── Safe wrappers (sized 1-D dispatch unless noted) ─────────────────

pub fn run_binary(a: &mut [f32], n: u32, ao: u32, bo: u32, co: u32, op: u32) {
    unsafe {
        launch_binary(a.as_mut_ptr(), n, ao, bo, co, op, grid_1d(n), BLOCK_X);
    }
}
pub fn run_fused_binary_unary(
    a: &mut [f32],
    n: u32,
    ao: u32,
    bo: u32,
    oo: u32,
    bin_op: u32,
    un_op: u32,
) {
    unsafe {
        launch_fused_binary_unary(
            a.as_mut_ptr(),
            n,
            ao,
            bo,
            oo,
            bin_op,
            un_op,
            grid_1d(n),
            BLOCK_X,
        );
    }
}
pub fn run_unary(a: &mut [f32], n: u32, io: u32, oo: u32, op: u32) {
    unsafe {
        launch_unary(a.as_mut_ptr(), n, io, oo, op, grid_1d(n), BLOCK_X);
    }
}
pub fn run_copy(a: &mut [f32], n: u32, io: u32, oo: u32) {
    unsafe {
        launch_copy(a.as_mut_ptr(), n, io, oo, grid_1d(n), BLOCK_X);
    }
}
pub fn run_compare(a: &mut [f32], n: u32, ao: u32, bo: u32, co: u32, op: u32) {
    unsafe {
        launch_compare(a.as_mut_ptr(), n, ao, bo, co, op, grid_1d(n), BLOCK_X);
    }
}
pub fn run_where_select(a: &mut [f32], n: u32, cond_o: u32, xo: u32, yo: u32, oo: u32) {
    unsafe {
        launch_where_select(a.as_mut_ptr(), n, cond_o, xo, yo, oo, grid_1d(n), BLOCK_X);
    }
}

pub fn run_matmul(a: &mut [f32], m: u32, k: u32, n: u32, ao: u32, bo: u32, co: u32, batch: u32) {
    let abs_ = if batch > 1 { m * k } else { 0 };
    let bbs = if batch > 1 { k * n } else { 0 };
    let cbs = if batch > 1 { m * n } else { 0 };
    unsafe {
        launch_matmul(
            a.as_mut_ptr(),
            m,
            k,
            n,
            ao,
            bo,
            co,
            batch,
            abs_,
            bbs,
            cbs,
            /*has_bias*/ 0,
            0,
            /*act_id*/ 0xFFFF,
            (n + 63) / 64,
            (m + 63) / 64,
            batch,
            16,
            16,
        );
    }
}
pub fn run_grouped_matmul(
    a: &mut [f32],
    m: u32,
    k: u32,
    n: u32,
    num_experts: u32,
    io: u32,
    wo: u32,
    idx_o: u32,
    oo: u32,
) {
    unsafe {
        launch_grouped_matmul(
            a.as_mut_ptr(),
            m,
            k,
            n,
            num_experts,
            io,
            wo,
            idx_o,
            oo,
            (n + 7) / 8,
            (m + 7) / 8,
            8,
            8,
        );
    }
}
pub fn run_dequant_matmul(
    a: &mut [f32],
    m: u32,
    k: u32,
    n: u32,
    block_size: u32,
    scheme_id: u32,
    xo: u32,
    wo: u32,
    sco: u32,
    zo: u32,
    oo: u32,
) {
    unsafe {
        launch_dequant_matmul(
            a.as_mut_ptr(),
            m,
            k,
            n,
            block_size,
            scheme_id,
            xo,
            wo,
            sco,
            zo,
            oo,
            (n + 7) / 8,
            (m + 7) / 8,
            8,
            8,
        );
    }
}

// Block-per-row reductions: grid.x = outer, block.x = 256. Each block
// handles a full row via warp-shuffle + cross-warp combine.
const REDUCE_BLOCK: u32 = 256;

pub fn run_reduce(a: &mut [f32], outer: u32, inner: u32, io: u32, oo: u32, op: u32) {
    unsafe {
        launch_reduce(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            oo,
            op,
            outer,
            REDUCE_BLOCK,
        );
    }
}
pub fn run_softmax(a: &mut [f32], outer: u32, inner: u32, io: u32, oo: u32) {
    unsafe {
        launch_softmax(a.as_mut_ptr(), outer, inner, io, oo, outer, REDUCE_BLOCK);
    }
}
pub fn run_layernorm(
    a: &mut [f32],
    outer: u32,
    inner: u32,
    io: u32,
    oo: u32,
    go: u32,
    beta_o: u32,
    eps: f32,
    op: u32,
) {
    unsafe {
        launch_layernorm(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            oo,
            go,
            beta_o,
            eps.to_bits(),
            op,
            outer,
            REDUCE_BLOCK,
        );
    }
}
pub fn run_fused_residual_ln(
    a: &mut [f32],
    outer: u32,
    inner: u32,
    io: u32,
    ro: u32,
    bias_o: u32,
    go: u32,
    beta_o: u32,
    oo: u32,
    eps: f32,
    has_bias: u32,
) {
    unsafe {
        launch_fused_residual_ln(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            ro,
            bias_o,
            go,
            beta_o,
            oo,
            eps.to_bits(),
            has_bias,
            outer,
            REDUCE_BLOCK,
        );
    }
}
pub fn run_cumsum(a: &mut [f32], outer: u32, inner: u32, io: u32, oo: u32, exclusive: u32) {
    unsafe {
        launch_cumsum(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            oo,
            exclusive,
            grid_1d(outer),
            BLOCK_X,
        );
    }
}
pub fn run_argmax(a: &mut [f32], outer: u32, inner: u32, io: u32, oo: u32) {
    unsafe {
        launch_argmax(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            oo,
            grid_1d(outer),
            BLOCK_X,
        );
    }
}
pub fn run_topk(a: &mut [f32], outer: u32, inner: u32, k: u32, io: u32, oo: u32) {
    unsafe {
        launch_topk(
            a.as_mut_ptr(),
            outer,
            inner,
            k,
            io,
            oo,
            grid_1d(outer),
            BLOCK_X,
        );
    }
}

pub fn run_gather(
    a: &mut [f32],
    n_out: u32,
    n_idx: u32,
    dim: u32,
    vocab: u32,
    io: u32,
    idx_o: u32,
    oo: u32,
) {
    unsafe {
        launch_gather(
            a.as_mut_ptr(),
            n_out,
            n_idx,
            dim,
            vocab,
            io,
            idx_o,
            oo,
            grid_1d(n_out),
            BLOCK_X,
        );
    }
}
pub fn run_narrow(
    a: &mut [f32],
    total: u32,
    outer: u32,
    inner: u32,
    axis_in: u32,
    axis_out: u32,
    start: u32,
    io: u32,
    oo: u32,
) {
    unsafe {
        launch_narrow(
            a.as_mut_ptr(),
            total,
            outer,
            inner,
            axis_in,
            axis_out,
            start,
            io,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_concat(
    a: &mut [f32],
    total: u32,
    outer: u32,
    inner: u32,
    axis_in: u32,
    axis_out: u32,
    start: u32,
    io: u32,
    oo: u32,
) {
    unsafe {
        launch_concat(
            a.as_mut_ptr(),
            total,
            outer,
            inner,
            axis_in,
            axis_out,
            start,
            io,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_transpose(a: &mut [f32], rank: u32, out_total: u32, io: u32, oo: u32, meta: &[u32]) {
    unsafe {
        launch_transpose(
            a.as_mut_ptr(),
            rank,
            out_total,
            io,
            oo,
            meta.as_ptr(),
            grid_1d(out_total),
            BLOCK_X,
        );
    }
}
pub fn run_expand(a: &mut [f32], rank: u32, out_total: u32, io: u32, oo: u32, meta: &[u32]) {
    unsafe {
        launch_expand(
            a.as_mut_ptr(),
            rank,
            out_total,
            io,
            oo,
            meta.as_ptr(),
            grid_1d(out_total),
            BLOCK_X,
        );
    }
}

pub fn run_attention(
    a: &mut [f32],
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    qo: u32,
    ko: u32,
    vo: u32,
    oo: u32,
    mask_o: u32,
    mask_kind: u32,
    scale: f32,
    window: u32,
) {
    // FlashAttention-1 geometry: BR=16 q-rows per block, 128 threads/block.
    // The cpp launch wrapper takes a flat (gx, bx) pair, so under HIP-CPU
    // we issue a 1-D launch with grid.x = q_blocks * batch * heads. The
    // kernel checks `gridDim.y == 1` and decodes (q_block, bh) from
    // blockIdx.x. Production CUDA uses the natural 2-D grid.
    let q_blocks = (seq_q + 15) / 16;
    unsafe {
        launch_attention(
            a.as_mut_ptr(),
            batch,
            heads,
            seq_q,
            seq_k,
            head_dim,
            qo,
            ko,
            vo,
            oo,
            mask_o,
            mask_kind,
            scale.to_bits(),
            window,
            q_blocks * batch * heads,
            128,
        );
    }
}
pub fn run_rope(
    a: &mut [f32],
    n_total: u32,
    seq: u32,
    head_dim: u32,
    half: u32,
    io: u32,
    co: u32,
    so: u32,
    oo: u32,
    last_dim: u32,
) {
    unsafe {
        launch_rope(
            a.as_mut_ptr(),
            n_total,
            seq,
            head_dim,
            half,
            io,
            co,
            so,
            oo,
            last_dim,
            grid_1d(n_total),
            BLOCK_X,
        );
    }
}

pub fn run_scatter_add(
    a: &mut [f32],
    oo: u32,
    total: u32,
    upd_o: u32,
    idx_o: u32,
    n_upd: u32,
    trailing: u32,
    out_dim: u32,
) {
    unsafe {
        launch_scatter_add_zero(a.as_mut_ptr(), oo, total, grid_1d(total), BLOCK_X);
        let acc_total = n_upd * trailing;
        launch_scatter_add_acc(
            a.as_mut_ptr(),
            oo,
            upd_o,
            idx_o,
            n_upd,
            trailing,
            out_dim,
            grid_1d(acc_total),
            BLOCK_X,
        );
    }
}

pub fn run_sample(
    a: &mut [f32],
    outer: u32,
    inner: u32,
    io: u32,
    oo: u32,
    top_k: u32,
    top_p: f32,
    temperature: f32,
    seed: u64,
) {
    unsafe {
        launch_sample(
            a.as_mut_ptr(),
            outer,
            inner,
            io,
            oo,
            top_k,
            top_p.to_bits(),
            temperature.to_bits(),
            seed as u32,
            (seed >> 32) as u32,
            grid_1d(outer),
            BLOCK_X,
        );
    }
}
pub fn run_selective_scan(
    a: &mut [f32],
    batch: u32,
    seq: u32,
    hidden: u32,
    state_size: u32,
    xo: u32,
    dt_o: u32,
    ao: u32,
    bo: u32,
    co: u32,
    oo: u32,
) {
    let total = batch * hidden;
    unsafe {
        launch_selective_scan(
            a.as_mut_ptr(),
            batch,
            seq,
            hidden,
            state_size,
            xo,
            dt_o,
            ao,
            bo,
            co,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}

pub fn run_pool1d(
    a: &mut [f32],
    n: u32,
    c: u32,
    l: u32,
    l_out: u32,
    kl: u32,
    sl: u32,
    pl: u32,
    op: u32,
    io: u32,
    oo: u32,
) {
    let total = n * c * l_out;
    unsafe {
        launch_pool1d(
            a.as_mut_ptr(),
            n,
            c,
            l,
            l_out,
            kl,
            sl,
            pl,
            op,
            io,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_pool2d(
    a: &mut [f32],
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    op: u32,
    io: u32,
    oo: u32,
) {
    let total = n * c * h_out * w_out;
    unsafe {
        launch_pool2d(
            a.as_mut_ptr(),
            n,
            c,
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
            op,
            io,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_pool3d(
    a: &mut [f32],
    n: u32,
    c: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    op: u32,
    io: u32,
    oo: u32,
) {
    let total = n * c * d_out * h_out * w_out;
    unsafe {
        launch_pool3d(
            a.as_mut_ptr(),
            n,
            c,
            d,
            h,
            w,
            d_out,
            h_out,
            w_out,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            op,
            io,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_conv1d(
    a: &mut [f32],
    n: u32,
    c_in: u32,
    c_out: u32,
    l: u32,
    l_out: u32,
    kl: u32,
    sl: u32,
    pl: u32,
    dl: u32,
    groups: u32,
    io: u32,
    wo: u32,
    oo: u32,
) {
    let total = n * c_out * l_out;
    unsafe {
        launch_conv1d(
            a.as_mut_ptr(),
            n,
            c_in,
            c_out,
            l,
            l_out,
            kl,
            sl,
            pl,
            dl,
            groups,
            io,
            wo,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_conv2d(
    a: &mut [f32],
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    io: u32,
    wo: u32,
    oo: u32,
) {
    let total = n * c_out * h_out * w_out;
    unsafe {
        launch_conv2d(
            a.as_mut_ptr(),
            n,
            c_in,
            c_out,
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
            groups,
            io,
            wo,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}
pub fn run_conv3d(
    a: &mut [f32],
    n: u32,
    c_in: u32,
    c_out: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    io: u32,
    wo: u32,
    oo: u32,
) {
    let total = n * c_out * d_out * h_out * w_out;
    unsafe {
        launch_conv3d(
            a.as_mut_ptr(),
            n,
            c_in,
            c_out,
            d,
            h,
            w,
            d_out,
            h_out,
            w_out,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            dd,
            dh,
            dw,
            groups,
            io,
            wo,
            oo,
            grid_1d(total),
            BLOCK_X,
        );
    }
}

/// PLAN L2 — interpreted N-ary element-wise chain. `meta` must hold
/// 144 u32 words: input_offs[0..16] then chain[0..128] (32 steps × 4).
/// `input_modulus` is 16 u32s (per-input element count for trailing-
/// shape broadcast; 0 means no broadcast).
pub fn run_elementwise_region(
    a: &mut [f32],
    len: u32,
    num_inputs: u32,
    num_steps: u32,
    dst_off: u32,
    meta: &[u32],
    scalar_input_mask: u32,
    input_modulus: &[u32; 16],
) {
    assert_eq!(
        meta.len(),
        144,
        "run_elementwise_region: meta must be 144 u32 words \
         (16 input_offs + 128 chain), got {}",
        meta.len()
    );
    unsafe {
        launch_elementwise_region(
            a.as_mut_ptr(),
            len,
            num_inputs,
            num_steps,
            dst_off,
            meta.as_ptr(),
            scalar_input_mask,
            input_modulus.as_ptr(),
            grid_1d(len),
            BLOCK_X,
        );
    }
}
