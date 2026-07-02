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

// Matmul with f16 weight storage. EXPERIMENTAL — currently slower
// than the f32 baseline on Apple Silicon; see WHY below.
//
// Reads the B (weight) input as `array<vec4<f16>>` (8-byte vector
// loads, 4 weights per access) and casts to f32 for compute. Same
// 32×32 output tile / 4×4 register block as `matmul.wgsl`.
//
// WHY THIS IS SLOWER (currently): BERT-class matmul is COMPUTE-BOUND
// at the shapes we care about — arithmetic intensity ≈ 112 FLOPs/byte,
// well above the Apple-GPU break-even of ~7.5 FLOPs/byte. Halving B's
// read bandwidth by storing f16 wins us nothing because we're not
// waiting on memory. Worse, the per-element f16→f32 widen on the load
// path adds work to the already-saturated compute pipe. Measured 4-5×
// slowdown vs f32 matmul on M4 Pro across MiniLM-L6/L12, BGE-small/base.
//
// PATH TO ACTUAL F16 SPEEDUP (not yet implemented): full f16 COMPUTE,
// not just storage. Apple GPUs run f16 ALU at 2× f32 throughput, but
// only when the multiplies and FMAs stay in f16. A future kernel
// would: (a) cast tile_a to f16 at load, (b) keep tile_b as f16 in
// shared mem, (c) accumulate in f32 (or f16 for short-K cases), (d)
// cast acc → f32 only at output. That's a real rewrite, not a
// drop-in change.
//
// Alignment requirement: B's per-matmul base offset must be a multiple
// of 4 (in f16 elements) AND `n` must be divisible by 4. BERT shapes
// always satisfy this (every weight matrix's leading dim is a power
// of 2 ≥ 64). Other callers must check before dispatching.
//
// SHADER_F16 feature must be enabled at device creation.
//
// Current dispatch: opt-in via `RLX_WGPU_F16_WEIGHTS=1`. Default OFF.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,        // offset in f16 elements; must be multiple of 4
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

// Bind group ordering matches `build_kernel_3` in kernels/mod.rs:
// 0=storage(rw), 1=uniform, 2=storage(ro).
@group(0) @binding(0) var<storage, read_write> arena:    array<f32>;
@group(0) @binding(1) var<uniform>             params:   Params;
// `weights` is an array of `vec4<f16>`; each element is 8 bytes
// holding 4 contiguous f16 weights. The buffer's underlying bytes
// are identical to a flat `array<f16>` — only the binding type
// differs, which lets us issue one global-memory load per 4 weights.
@group(0) @binding(2) var<storage, read>       weights:  array<vec4<f16>>;

var<workgroup> tile_a: array<array<f32, 16>, 32>;
var<workgroup> tile_b: array<array<f32, 32>, 16>;

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
        case 9u, 11u: {
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
fn matmul_f16w(
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

    var acc: array<array<f32, 4>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // tile_a (32×16) read from f32 arena. Unchanged.
        for (var i: u32 = 0u; i < RM; i = i + 1u) {
            let m_local = lr * RM + i;
            let global_row = wid.y * TILE_M + m_local;
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let k_local = lc * 2u + j;
                let global_k = t * TILE_K + k_local;
                if (global_row < params.m && global_k < params.k) {
                    tile_a[m_local][k_local] = arena[a_base + global_row * params.k + global_k];
                } else {
                    tile_a[m_local][k_local] = 0.0;
                }
            }
        }
        // tile_b (16×32): vec4<f16> reads. Each thread covers 2 K-rows ×
        // RN=4 N-cols; the 4 N-cols are 4-aligned (lc * 4 + j) and so
        // map to exactly ONE vec4<f16> per K-row. One load per K-row
        // instead of 4 — cuts B's global-memory traffic 4×.
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            let n_local_base = lc * RN;
            let global_col_base = wid.x * TILE_N + n_local_base;
            if (global_k < params.k && global_col_base + 3u < params.n) {
                let f16_off = b_base + global_k * params.n + global_col_base;
                let v: vec4<f16> = weights[f16_off / 4u];
                let vf: vec4<f32> = vec4<f32>(v);
                tile_b[k_local][n_local_base + 0u] = vf.x;
                tile_b[k_local][n_local_base + 1u] = vf.y;
                tile_b[k_local][n_local_base + 2u] = vf.z;
                tile_b[k_local][n_local_base + 3u] = vf.w;
            } else {
                // Boundary fallback: scalar reads. For BERT shapes this
                // arm is unreachable (n always multiple of 4) but kept
                // for correctness on arbitrary inputs.
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    let n_local = n_local_base + j;
                    let global_col = wid.x * TILE_N + n_local;
                    if (global_k < params.k && global_col < params.n) {
                        let f16_off = b_base + global_k * params.n + global_col;
                        let vec_idx = f16_off / 4u;
                        let comp = f16_off % 4u;
                        let v = weights[vec_idx];
                        var s: f32 = 0.0;
                        switch (comp) {
                            case 0u: { s = f32(v.x); }
                            case 1u: { s = f32(v.y); }
                            case 2u: { s = f32(v.z); }
                            default: { s = f32(v.w); }
                        }
                        tile_b[k_local][n_local] = s;
                    } else {
                        tile_b[k_local][n_local] = 0.0;
                    }
                }
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 4>;
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                a_reg[i] = tile_a[lr * RM + i][k];
            }
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                b_reg[j] = tile_b[k][lc * RN + j];
            }
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j];
                }
            }
        }

        workgroupBarrier();
    }

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
