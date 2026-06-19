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

enable f16;

// Full f16-compute matmul: f16 operands, f16 multiply, f32 accumulator.
// EXPERIMENTAL — currently slower than the f32 baseline on Apple
// Silicon via wgpu 29 / naga 29 (see EMPIRICAL FINDING below).
//
// A (activations) is f32 in the arena and gets DOWNCAST to f16 at the
// shared-memory load. B (weights) is already f16 in the weight buffer
// (vec4<f16> reads). The inner-loop FMA uses f16 operands; Apple
// Silicon GPUs *can* run f16 ALU at 2× f32 throughput. Accumulation
// stays in f32 to avoid loss-of-precision over BERT-class K (≤ 4096):
// each f16 multiply has ~3 ulps of quantization noise, and over a
// 768-element dot product an f16 acc would drift well past 1 % rel
// error.
//
// EMPIRICAL FINDING (Apple M4 Pro, wgpu 29.0.3 / naga 29):
//
//     Even with full f16 compute (this kernel) and vec4<f16> reads,
//     matmul is consistently 4–5× SLOWER than the f32 baseline on
//     BERT shapes. Same direction as the storage-only `matmul_f16w`
//     variant. The naga WGSL→MSL emission for `enable f16; let v: f16
//     = ...; acc + f32(a*b)` does not appear to unlock Apple's f16 ALU
//     fast path (or whatever it emits is slower than its f32 path due
//     to widening overhead). To actually beat f32 on Apple via wgpu we
//     would need either:
//       (a) wgpu/naga to expose `simdgroup_matrix` intrinsics (not
//           yet in the WGSL spec), or
//       (b) a different code-gen path that triggers Metal's hardware
//           f16 GEMM units explicitly (out of reach from portable WGSL).
//
// The kernel is correct (max|Δ|=2.78e-3 on BERT MiniLM-L6 vs CPU,
// cosine ≥ 0.9999); it just isn't a perf win today. Kept in tree as
// the dispatch foundation so a future wgpu/naga release that does
// unlock the win lands here without architectural changes.
//
// Same 32×32 output tile / 4×4 register block as `matmul.wgsl` so
// dispatch math is unchanged. Shared memory drops 2× vs the f32
// kernel (4 KB instead of 8 KB).
//
// Output is f32 (cast from acc at the epilogue) and written back to
// the f32 arena. Bias add + activation also run in f32.
//
// Alignment requirement: B's per-matmul base offset must be a multiple
// of 4 (in f16 elements) AND `n` must be divisible by 4.
//
// SHADER_F16 feature must be enabled at device creation.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,        // offset in f16 elements
    c_off: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;
const RM: u32 = 4u;
const RN: u32 = 4u;

// Bind group ordering matches `build_kernel_3` in kernels/mod.rs.
@group(0) @binding(0) var<storage, read_write> arena:    array<f32>;
@group(0) @binding(1) var<uniform>             params:   Params;
@group(0) @binding(2) var<storage, read>       weights:  array<vec4<f16>>;

// f16 shared memory: half the bytes vs the f32 kernel.
var<workgroup> tile_a: array<array<f16, 16>, 32>;
var<workgroup> tile_b: array<array<f16, 32>, 16>;

// Exact GELU — A&S 7.1.26 erf, matches rlx-cpu's scalar_gelu.
fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    if (params.act_id == 0xFFFFu) { return v; }
    switch (params.act_id) {
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(8, 8)
fn matmul_f16_compute(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let bz = gid.z;
    if (bz >= params.batch) { return; }

    let lr = lid.y;
    let lc = lid.x;
    let row_base = wid.y * TILE_M + lr * RM;
    let col_base = wid.x * TILE_N + lc * RN;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;  // f16 elements
    let c_base = params.c_off + bz * params.c_batch_stride;

    // f32 accumulator for K-stability; promotes only at the FMA boundary.
    var acc: array<array<f32, 4>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // tile_a: read f32 arena, cast to f16 at load. Each thread
        // writes 4 rows × 2 cols = 8 elements.
        for (var i: u32 = 0u; i < RM; i = i + 1u) {
            let m_local = lr * RM + i;
            let global_row = wid.y * TILE_M + m_local;
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let k_local = lc * 2u + j;
                let global_k = t * TILE_K + k_local;
                if (global_row < params.m && global_k < params.k) {
                    tile_a[m_local][k_local] = f16(arena[a_base + global_row * params.k + global_k]);
                } else {
                    tile_a[m_local][k_local] = f16(0.0);
                }
            }
        }
        // tile_b: vec4<f16> reads — one 8-byte load per K-row. Each
        // thread covers 2 K-rows × 4 N-cols (the 4 N-cols are 4-aligned
        // and so map to ONE vec4<f16> load).
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            let n_local_base = lc * RN;
            let global_col_base = wid.x * TILE_N + n_local_base;
            if (global_k < params.k && global_col_base + 3u < params.n) {
                let f16_off = b_base + global_k * params.n + global_col_base;
                let v: vec4<f16> = weights[f16_off / 4u];
                tile_b[k_local][n_local_base + 0u] = v.x;
                tile_b[k_local][n_local_base + 1u] = v.y;
                tile_b[k_local][n_local_base + 2u] = v.z;
                tile_b[k_local][n_local_base + 3u] = v.w;
            } else {
                // Boundary fallback (BERT shapes never hit this).
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    let n_local = n_local_base + j;
                    let global_col = wid.x * TILE_N + n_local;
                    if (global_k < params.k && global_col < params.n) {
                        let f16_off = b_base + global_k * params.n + global_col;
                        let vec_idx = f16_off / 4u;
                        let comp = f16_off % 4u;
                        let v = weights[vec_idx];
                        var s: f16 = f16(0.0);
                        switch (comp) {
                            case 0u: { s = v.x; }
                            case 1u: { s = v.y; }
                            case 2u: { s = v.z; }
                            default: { s = v.w; }
                        }
                        tile_b[k_local][n_local] = s;
                    } else {
                        tile_b[k_local][n_local] = f16(0.0);
                    }
                }
            }
        }

        workgroupBarrier();

        // Inner loop: f16 operands, f16 multiply, accumulate widened
        // result in f32. The multiply is the performance-critical op
        // and runs at 2× f32 throughput on Apple GPUs. The widen
        // f16→f32 to update `acc` is one extra cycle per FMA but
        // cheaper than the f32-multiply we'd be doing otherwise.
        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f16, 4>;
            var b_reg: array<f16, 4>;
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                a_reg[i] = tile_a[lr * RM + i][k];
            }
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                b_reg[j] = tile_b[k][lc * RN + j];
            }
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    // f16 * f16 = f16 (fast path), then widen to f32
                    // for the f32 acc update.
                    acc[i][j] = acc[i][j] + f32(a_reg[i] * b_reg[j]);
                }
            }
        }

        workgroupBarrier();
    }

    // Epilogue: bias + activation in f32, write f32 to arena.
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        let global_row = row_base + i;
        if (global_row >= params.m) { continue; }
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            let global_col = col_base + j;
            if (global_col >= params.n) { continue; }
            var v = acc[i][j];
            if (params.has_bias != 0u) {
                v = v + arena[params.bias_off + global_col];
            }
            v = apply_act(v);
            arena[c_base + global_row * params.n + global_col] = v;
        }
    }
}
