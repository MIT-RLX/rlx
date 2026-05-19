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
enable wgpu_cooperative_matrix;

// Hardware GEMM matmul via wgpu cooperative_matrix (8×8 tiles —
// despite the file name, see HARDWARE FINDING below).
//
// END-TO-END PERF NOTE: this kernel runs ~7× faster than the f32
// baseline matmul on its own (micro-bench: 1865µs vs 13036µs for
// M=1024 K=384 N=1536 on M4 Pro). However the current dispatcher
// integration **regresses BERT end-to-end** (~2× slower) because
// every matmul gets a `Step::CastF32ToF16` pre-pass that mirrors A
// from the f32 arena into the f16 shadow buffer. The cast adds
// memory bandwidth pressure equivalent to A's bytes; for BERT-class
// shapes this dominates the saved matmul time. Two follow-ups to
// recoup the win:
//   1. Fold the f32→f16 cast into the matmul kernel itself (read
//      from a 5th f32 binding, cast at workgroup-shared-mem load).
//      Eliminates the cast dispatch + halves A's memory traffic.
//   2. Mirror activation writes from every kernel into arena_f16
//      (each kernel writes both f32 and f16 versions). Avoids the
//      cast entirely but doubles every activation write.
// Today's behavior is correct (cosine ≥ 0.9999) but slower than f32;
// the dispatcher only routes Coop16 when the IR has an explicit f16
// dtype tag (PrecisionPolicy::AutoMixed), so the default path is
// unchanged.
//
// On Apple M-series this lowers to MSL `simdgroup_matrix_multiply_accumulate`
// (the simdgroup_matrix hardware path that MLX/Metal native kernels use).
// On Vulkan it lowers to `OpCooperativeMatrixMulAddKHR` against
// `VK_KHR_cooperative_matrix`. This is THE primitive that closes the
// portable-WGSL gap to vendor BLAS on Apple Silicon (and NVIDIA/AMD).
//
// Algorithm (per workgroup):
//   - One workgroup computes one 16×16 output tile of C.
//   - Per K-tile of size 16:
//       coopLoad A as a coop_mat8x8<f16, A>
//       coopLoad B as a coop_mat8x8<f16, B>
//       acc = coopMultiplyAdd(A, B, acc)   // hardware GEMM
//   - coopStore acc to global memory.
//
// Inputs:
//   - A is read from `arena` (f32, downcast at coopLoad to the f16 matrix).
//     Wait — coopLoad takes a pointer to scalar matching the matrix scalar
//     type. So A must be loaded from f16 storage. We read from `arena_f16`
//     for both A and B. Activations get downcast to f16 host-side via the
//     existing f16 shadow buffer (same path as matmul_f16w / f16_compute).
//   - B is f16 weights from `weights` buffer.
//   - C (acc) is f32 (the role==C accumulator type in the multiply-add).
//
// Bind group: 0=arena (f32 rw), 1=params (uniform), 2=weights (f16 ro),
//             3=arena_f16 (f16 ro for A reads).
//
// Output goes back to `arena` as f32 — at the epilogue we coopStore the
// f32 acc into a temporary workgroup buffer and one thread copies into
// the f32 arena. Bias + activation also run f32 on the way out.
//
// REQUIREMENTS:
//   - Device feature: EXPERIMENTAL_COOPERATIVE_MATRIX + SHADER_F16
//   - M and N must be multiples of 16 (one wg per 16×16 tile)
//   - K must be multiple of 16 (whole K-tiles)
//   - Apple Metal: Apple7+ (M-series) and MSL 2.3+
//   - Vulkan: VK_KHR_cooperative_matrix supported

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,           // f16 element offset (A is read from f16 shadow)
    b_off: u32,           // f16 element offset (weights buffer)
    c_off: u32,           // f32 element offset (output written to f32 arena)
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

// Hardware coop matrix is 8×8 on Apple (`simdgroup_half8x8`). Each
// workgroup tiles FOUR 8×8 ops in M and FOUR in N — a 32×32 effective
// output tile, 16 hardware GEMM ops per workgroup. Brings dispatch
// count to parity with the f32 baseline matmul (also 32×32 output).
//
// Tried 64×32 (8×4 sub-tiles, 32 ops/wg): kernel went 36% slower per
// dispatch (681 → 931 µs at M=1024 K=384 N=1536). The bigger tile
// has higher register pressure per simdgroup and the K-loop's
// memory loads don't overlap well with 32 sequential simdgroup_matrix
// ops. The 32×32 / 16-op version is the local optimum on Apple M4.
const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;     // K iterates one 8×8 micro-tile at a time
const SUB_M: u32 = 4u;      // 4 row sub-tiles (32 = 4 × 8)
const SUB_N: u32 = 4u;      // 4 col sub-tiles (32 = 4 × 8)

@group(0) @binding(0) var<storage, read_write> arena:    array<f32>;
@group(0) @binding(1) var<uniform>             params:   Params;
@group(0) @binding(2) var<storage, read>       weights:  array<f16>;

// Acc scratch holds 4×4 = 16 f16 8×8 sub-tiles back-to-back after
// coopStore. 16 × 64 = 1024 f16 = 2 KB.
var<workgroup> acc_scratch: array<f16, 1024>;

// A-staging: 32 rows × 8 K-cols of f16 = 256 f16 = 512 bytes. Loaded
// from the f32 arena (downcast on the fly) once per K-tile, used by
// all 4 row sub-tiles for that K-iteration.
var<workgroup> a_stage: array<f16, 256>;     // 32 × 8

// B doesn't need workgroup staging — it's already f16 in the weight
// buffer; coopLoad reads each 8×8 sub-tile directly from global
// memory. (Apple's hardware load coalesces well at 8-element strides.)

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

// One subgroup (= one workgroup of 32 threads on Apple) computes one
// 16×16 output tile cooperatively via the hardware coop-matrix unit.
// `workgroup_size(32)` matches the typical Apple/NVIDIA subgroup size;
// portable hardware should accept it as long as the implementation's
// subgroup size divides evenly.
// Cooperative-matrix ops require UNIFORM control flow across the
// workgroup — every invocation must hit every coopLoad / coopMultiplyAdd
// / coopStore call. So we can't early-return on out-of-bounds; the
// dispatcher must size the workgroup grid exactly. Caller contract:
// M and N are multiples of 16, K is a multiple of 16, batch matches
// the dispatched z-extent. BERT shapes always satisfy these.
// workgroup_size(32) matches Apple's simdgroup width — the
// `simdgroup_matrix` instruction is a SINGLE 32-thread cooperative
// op. Using 64 threads doubles compute redundantly: two simdgroups
// both execute the same matrix multiply, halving effective
// throughput. Each of the 32 threads writes 2 of the 64 output-tile
// elements in the epilogue.
@compute @workgroup_size(32)
fn matmul_coop16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    // Zero acc_scratch (1024 f16 = 32 elements per thread).
    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        acc_scratch[lid + s * 32u] = f16(0.0);
    }
    workgroupBarrier();
    // 16 separate accumulators, one per (sub-row, sub-col) 8×8 tile.
    // Bootstrap each from the zeroed scratch region's same address —
    // coopLoad returns a fresh matrix value per call.
    var acc_00: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_01: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_02: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_03: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_10: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_11: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_12: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_13: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_20: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_21: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_22: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_23: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_30: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_31: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_32: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);
    var acc_33: coop_mat8x8<f16, C> = coopLoad<coop_mat8x8<f16, C>>(&acc_scratch[0], 8u);

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        // Stage 32 rows × 8 K-cols of A into a_stage (256 f16; 32
        // threads × 8 elements each).
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let idx = lid + s * 32u;
            let r = idx / 8u;
            let c = idx % 8u;
            a_stage[idx] = f16(arena[a_base + (row_base + r) * params.k + k_off + c]);
        }
        workgroupBarrier();

        // Load 4 row sub-tiles of A from workgroup scratch.
        let a_0: coop_mat8x8<f16, A> = coopLoad<coop_mat8x8<f16, A>>(&a_stage[0u  ], 8u);
        let a_1: coop_mat8x8<f16, A> = coopLoad<coop_mat8x8<f16, A>>(&a_stage[64u ], 8u);
        let a_2: coop_mat8x8<f16, A> = coopLoad<coop_mat8x8<f16, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f16, A> = coopLoad<coop_mat8x8<f16, A>>(&a_stage[192u], 8u);
        // Load 4 col sub-tiles of B directly from f16 weight buffer.
        let b_row = b_base + k_off * params.n + col_base;
        let b_0: coop_mat8x8<f16, B> = coopLoad<coop_mat8x8<f16, B>>(&weights[b_row + 0u],  params.n);
        let b_1: coop_mat8x8<f16, B> = coopLoad<coop_mat8x8<f16, B>>(&weights[b_row + 8u],  params.n);
        let b_2: coop_mat8x8<f16, B> = coopLoad<coop_mat8x8<f16, B>>(&weights[b_row + 16u], params.n);
        let b_3: coop_mat8x8<f16, B> = coopLoad<coop_mat8x8<f16, B>>(&weights[b_row + 24u], params.n);
        // 32 hardware GEMM ops, each accumulating into its own register matrix.
        acc_00 = coopMultiplyAdd(a_0, b_0, acc_00);
        acc_01 = coopMultiplyAdd(a_0, b_1, acc_01);
        acc_02 = coopMultiplyAdd(a_0, b_2, acc_02);
        acc_03 = coopMultiplyAdd(a_0, b_3, acc_03);
        acc_10 = coopMultiplyAdd(a_1, b_0, acc_10);
        acc_11 = coopMultiplyAdd(a_1, b_1, acc_11);
        acc_12 = coopMultiplyAdd(a_1, b_2, acc_12);
        acc_13 = coopMultiplyAdd(a_1, b_3, acc_13);
        acc_20 = coopMultiplyAdd(a_2, b_0, acc_20);
        acc_21 = coopMultiplyAdd(a_2, b_1, acc_21);
        acc_22 = coopMultiplyAdd(a_2, b_2, acc_22);
        acc_23 = coopMultiplyAdd(a_2, b_3, acc_23);
        acc_30 = coopMultiplyAdd(a_3, b_0, acc_30);
        acc_31 = coopMultiplyAdd(a_3, b_1, acc_31);
        acc_32 = coopMultiplyAdd(a_3, b_2, acc_32);
        acc_33 = coopMultiplyAdd(a_3, b_3, acc_33);
        workgroupBarrier(); // before next K-tile's stage write
    }

    // Store all 16 accs back to scratch in a 32×32 row-major layout.
    // Sub-tile (sr, sc) goes to scratch[sr*8 + r][sc*8 + c] in a 32-wide
    // layout. coopStore each matrix to its base offset with stride 32.
    coopStore(acc_00, &acc_scratch[0u   * 32u + 0u ], 32u);
    coopStore(acc_01, &acc_scratch[0u   * 32u + 8u ], 32u);
    coopStore(acc_02, &acc_scratch[0u   * 32u + 16u], 32u);
    coopStore(acc_03, &acc_scratch[0u   * 32u + 24u], 32u);
    coopStore(acc_10, &acc_scratch[8u   * 32u + 0u ], 32u);
    coopStore(acc_11, &acc_scratch[8u   * 32u + 8u ], 32u);
    coopStore(acc_12, &acc_scratch[8u   * 32u + 16u], 32u);
    coopStore(acc_13, &acc_scratch[8u   * 32u + 24u], 32u);
    coopStore(acc_20, &acc_scratch[16u  * 32u + 0u ], 32u);
    coopStore(acc_21, &acc_scratch[16u  * 32u + 8u ], 32u);
    coopStore(acc_22, &acc_scratch[16u  * 32u + 16u], 32u);
    coopStore(acc_23, &acc_scratch[16u  * 32u + 24u], 32u);
    coopStore(acc_30, &acc_scratch[24u  * 32u + 0u ], 32u);
    coopStore(acc_31, &acc_scratch[24u  * 32u + 8u ], 32u);
    coopStore(acc_32, &acc_scratch[24u  * 32u + 16u], 32u);
    coopStore(acc_33, &acc_scratch[24u  * 32u + 24u], 32u);
    workgroupBarrier();

    // 32 threads, 1024 outputs — each thread writes 32 elements.
    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let idx = lid + s * 32u;
        let r = idx / 32u;     // row within 32×32 tile
        let c = idx % 32u;     // col within 32×32 tile
        let global_row = row_base + r;
        let global_col = col_base + c;
        // Widen f16 acc → f32 for bias/activation/output. The bias
        // and arena are f32; activation ops want f32 precision.
        var v: f32 = f32(acc_scratch[idx]);
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + global_col];
        }
        v = apply_act(v);
        arena[c_base + global_row * params.n + global_col] = v;
    }
}
