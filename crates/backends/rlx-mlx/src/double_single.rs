// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::DoubleSingle` on MLX via `fast::metal_kernel`: the same `rlxsl::dw`
//! arithmetic and [`DfPlan`] indexing as the Metal backend. MLX compiles
//! custom kernels in its default `MathMode::Safe` (precise), which the
//! error-free transforms need. The plan is baked into the kernel header, so
//! MLX compiles each shape once.

#![cfg(rlx_mlx_host)]

use crate::array::{Array, MlxError};
use rlx_ir::double_single::{DF_MAX_RANK, DfOp, DfPlan};

const ELEMENTWISE: &str = r#"
const uint gid = thread_position_in_grid.x;
if (gid >= DF_LEN) { return; }
uint ia = df_off(gid, DF_RANK, D0, S0);
uint ib = df_off(gid, DF_RANK, D0, S1);
DwF32 x = DwF32{a[2u * ia], a[2u * ia + 1u]};
DwF32 y = DwF32{b[2u * ib], b[2u * ib + 1u]};
DwF32 r;
switch (DF_OP) {
    case 0: r = dw_add(x, y); break;
    case 1: r = dw_sub(x, y); break;
    case 2: r = dw_mul(x, y); break;
    case 4: r = dw_sqrt(x); break;
    case 5: r = dw_exp(x); break;
    case 6: r = (y.hi > x.hi || (y.hi == x.hi && y.lo > x.lo)) ? y : x; break;
    default: r = dw_div(x, y); break;
}
out[2u * gid] = r.hi;
out[2u * gid + 1u] = r.lo;
"#;

const MATMUL: &str = r#"
const uint gid = thread_position_in_grid.x;
if (gid >= DF_LEN) { return; }
uint j = gid % DF_N;
uint row = gid / DF_N;
uint i = row % DF_M;
uint bi = row / DF_M;
uint a0 = bi * DF_A_BATCH + i * DF_K;
uint b0 = bi * DF_B_BATCH + j;
DwF32 acc = DwF32{0.0f, 0.0f};
for (uint kk = 0; kk < DF_K; ++kk) {
    uint ia = a0 + kk, ib = b0 + kk * DF_N;
    acc = dw_add(acc, dw_mul(DwF32{a[2u * ia], a[2u * ia + 1u]}, DwF32{b[2u * ib], b[2u * ib + 1u]}));
}
out[2u * gid] = acc.hi;
out[2u * gid + 1u] = acc.lo;
"#;

const REDUCE: &str = r#"
const uint gid = thread_position_in_grid.x;
if (gid >= DF_LEN) { return; }
uint base = df_off(gid, DF_KEPT_RANK, D0, S0);
DwF32 acc = DF_OP == 1u ? DwF32{a[2u * base], a[2u * base + 1u]} : DwF32{0.0f, 0.0f};
for (uint r = DF_OP == 1u ? 1u : 0u; r < DF_RED_LEN; ++r) {
    uint ix = base + df_off(r, DF_RED_RANK, D1, S2);
    DwF32 x = DwF32{a[2u * ix], a[2u * ix + 1u]};
    if (DF_OP == 1u) {
        if (x.hi > acc.hi || (x.hi == acc.hi && x.lo > acc.lo)) { acc = x; }
    } else {
        acc = dw_add(acc, x);
    }
}
out[2u * gid] = acc.hi;
out[2u * gid + 1u] = acc.lo;
"#;

fn arr(name: &str, v: &[u32; DF_MAX_RANK]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x}u")).collect();
    format!(
        "constant uint {name}[{DF_MAX_RANK}] = {{{}}};\n",
        items.join(", ")
    )
}

/// Run `op` on pair arrays `a` (and `b`) of `a_len` / `b_len` floats per
/// `plan`; returns a flat f32 array of `2 * len` values (reshape to the node's
/// pair shape).
pub fn double_single_ondevice(
    op: &DfOp,
    plan: &DfPlan,
    a: (&Array, usize),
    b: Option<(&Array, usize)>,
) -> Result<Array, MlxError> {
    let mut header = String::new();
    let (body, len, kind) = match plan {
        DfPlan::Elementwise {
            len,
            rank,
            out_dims,
            a_strides,
            b_strides,
        } => {
            header += &format!("constant uint DF_RANK = {rank}u;\n");
            header += &arr("D0", out_dims);
            header += &arr("S0", a_strides);
            header += &arr("S1", b_strides);
            (ELEMENTWISE, *len, "ew")
        }
        DfPlan::MatMul {
            batch,
            m,
            k,
            n,
            a_batch,
            b_batch,
        } => {
            header += &format!(
                "constant uint DF_M = {m}u;\nconstant uint DF_K = {k}u;\nconstant uint DF_N = {n}u;\n\
                 constant uint DF_A_BATCH = {a_batch}u;\nconstant uint DF_B_BATCH = {b_batch}u;\n"
            );
            (MATMUL, batch * m * n, "mm")
        }
        DfPlan::Sum {
            out_len,
            red_len,
            kept_rank,
            kept_dims,
            kept_strides,
            red_rank,
            red_dims,
            red_strides,
        } => {
            header += &format!(
                "constant uint DF_RED_LEN = {red_len}u;\nconstant uint DF_KEPT_RANK = {kept_rank}u;\n\
                 constant uint DF_RED_RANK = {red_rank}u;\n"
            );
            header += &arr("D0", kept_dims);
            header += &arr("S0", kept_strides);
            header += &arr("D1", red_dims);
            header += &arr("S2", red_strides);
            (REDUCE, *out_len, "red")
        }
    };
    header = format!(
        "{}\nconstant uint DF_LEN = {len}u;\nconstant uint DF_OP = {}u;\n{header}\
         inline uint df_off(uint i, uint rank, constant uint* dims, constant uint* strides) {{\n\
         \x20   uint off = 0;\n\
         \x20   for (int d = int(rank) - 1; d >= 0; --d) {{ uint n = dims[d]; off += (i % n) * strides[d]; i /= n; }}\n\
         \x20   return off;\n}}\n",
        rlxsl::dw::double_single_prelude(rlxsl::Lang::Msl),
        op.code(),
    );
    // The name keys MLX's compile cache: one kernel per plan.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&header, &mut h);
    let name = format!("df_{kind}_{:x}", std::hash::Hasher::finish(&h));
    let fa = crate::ops::reshape(a.0, &[a.1 as i32])?;
    let fb = match b {
        Some((b, n)) => crate::ops::reshape(b, &[n as i32])?,
        None => crate::ops::reshape(a.0, &[a.1 as i32])?,
    };
    crate::round_to_format::dispatch(
        &name,
        body,
        &header,
        &[("a", &fa), ("b", &fb)],
        2 * len,
        (len.max(1), len.clamp(1, 256)),
    )
}
