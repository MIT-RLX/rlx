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

// Multinomial sampling kernel with optional top-k / top-p / temperature.
// One thread per batch row; serial inside each row (vocab N typically
// 32k–50k — fast enough on a single SIMD lane for batch=1 inference).
//
// Algorithm:
//   1. Scale logits by 1/temperature in place; track row max.
//   2. Convert to probabilities (exp(x - max) / sum_exp). Probs land in
//      the input slot — we clobber it. Sample's input is consumed.
//   3. If top_k > 0 OR top_p < 1: do a serial selection-sort of the
//      probabilities; for each pick mark as "selected" by storing the
//      probability as a negative value. Stop when we've collected k
//      tokens *and* cumulative ≥ top_p.
//   4. Renormalize the selected set; zero the rest.
//   5. Sample by cumsum + uniform-threshold using a per-row LCG seeded
//      by `seed XOR row`.

struct Params {
    outer: u32,
    inner: u32,        // vocab
    in_off: u32,
    out_off: u32,
    top_k: u32,
    top_p_bits: u32,   // bitcast<f32>
    temp_bits: u32,    // bitcast<f32>
    seed_lo: u32,
    seed_hi: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const NEG_INF: f32 = -3.4e38;
const SENTINEL_EPS: f32 = 1e-30;

// ── Threefry-2×32-20 ─────────────────────────────────────────────
//
// Counter-based PRNG, same family MLX (and JAX) use. Stateless: feed
// in (counter, key), get out two 32-bit words. 20 rounds with key
// injections every 4 rounds.
//
// Reference: Salmon et al. "Parallel random numbers: as easy as 1, 2, 3"
// (Random123 paper, 2011). Rotation constants R_2x32 verbatim from
// the spec; KS_PARITY is the 32-bit Skein parity word.

fn rotl32(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

fn threefry2x32_20(c_in: vec2<u32>, k_in: vec2<u32>) -> vec2<u32> {
    let KS_PARITY: u32 = 0x1BD11BDAu;
    let ks0 = k_in.x;
    let ks1 = k_in.y;
    let ks2 = ks0 ^ ks1 ^ KS_PARITY;

    var x0 = c_in.x + ks0;
    var x1 = c_in.y + ks1;

    // Rotation schedule: 8 constants, cycled across the 20 rounds.
    // The macro below interleaves (ks-extend round + injection) so
    // we don't need an `if` branch each round.

    // Rounds 1-4
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 1
    x0 = x0 + ks1; x1 = x1 + ks2; x1 = x1 + 1u;

    // Rounds 5-8
    x0 = x0 + x1; x1 = rotl32(x1, 17u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 29u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 16u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 24u); x1 = x1 ^ x0;
    // Inject 2
    x0 = x0 + ks2; x1 = x1 + ks0; x1 = x1 + 2u;

    // Rounds 9-12
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 3
    x0 = x0 + ks0; x1 = x1 + ks1; x1 = x1 + 3u;

    // Rounds 13-16
    x0 = x0 + x1; x1 = rotl32(x1, 17u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 29u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 16u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 24u); x1 = x1 ^ x0;
    // Inject 4
    x0 = x0 + ks1; x1 = x1 + ks2; x1 = x1 + 4u;

    // Rounds 17-20
    x0 = x0 + x1; x1 = rotl32(x1, 13u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 15u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1, 26u); x1 = x1 ^ x0;
    x0 = x0 + x1; x1 = rotl32(x1,  6u); x1 = x1 ^ x0;
    // Inject 5
    x0 = x0 + ks2; x1 = x1 + ks0; x1 = x1 + 5u;

    return vec2<u32>(x0, x1);
}

@compute @workgroup_size(64)
fn sample(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ngs: vec3<u32>) {
    let row = gid.x + gid.y * ngs.x * 64u;
    if (row >= params.outer) { return; }
    let base = params.in_off + row * params.inner;
    let temp = bitcast<f32>(params.temp_bits);
    let top_p = bitcast<f32>(params.top_p_bits);
    let inv_temp = 1.0 / max(temp, 1e-6);

    // 1. Apply temperature; track row max.
    var m: f32 = NEG_INF;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let v = arena[base + i] * inv_temp;
        arena[base + i] = v;
        m = max(m, v);
    }
    // 2. exp / sum → probabilities.
    var sum_e: f32 = 0.0;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let e = exp(arena[base + i] - m);
        arena[base + i] = e;
        sum_e = sum_e + e;
    }
    let inv_sum = 1.0 / sum_e;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        arena[base + i] = arena[base + i] * inv_sum;
    }

    // 3. Top-K / Top-P filter via selection.
    let need_filter = (params.top_k > 0u) || (top_p < 1.0 && top_p > 0.0);
    if (need_filter) {
        var cum: f32 = 0.0;
        var picked_count: u32 = 0u;
        let k_limit = select(params.inner, params.top_k, params.top_k > 0u);
        loop {
            if (picked_count >= k_limit) { break; }
            // Find the largest still-positive probability.
            var best_v: f32 = -1.0;
            var best_i: u32 = 0u;
            var found: bool = false;
            for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
                let v = arena[base + i];
                if (v >= 0.0 && v > best_v) {
                    best_v = v;
                    best_i = i;
                    found = true;
                }
            }
            if (!found) { break; }
            // Mark as selected by negating + sentinel offset.
            arena[base + best_i] = -best_v - SENTINEL_EPS;
            cum = cum + best_v;
            picked_count = picked_count + 1u;
            // Stop early if top-p satisfied AND we picked at least 1.
            if (top_p < 1.0 && cum >= top_p) { break; }
        }
        // Restore selected; zero unselected; sum for renormalization.
        var new_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            let v = arena[base + i];
            if (v < 0.0) {
                let restored = -v - SENTINEL_EPS;
                arena[base + i] = restored;
                new_sum = new_sum + restored;
            } else {
                arena[base + i] = 0.0;
            }
        }
        let inv_new = 1.0 / max(new_sum, 1e-12);
        for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
            arena[base + i] = arena[base + i] * inv_new;
        }
    }

    // 4. Multinomial sample via Gumbel-max: argmax(log(p) + g) where
    //    g_i = -log(-log(u_i)) is Gumbel(0, 1) noise. Equivalent to
    //    softmax-then-categorical but a single argmax pass — and
    //    matches MLX's `categorical` algorithm so picks are bit-
    //    equivalent given the same uniform stream + seed convention.
    //
    //    JAX-style seed → key derivation: rather than feed the raw
    //    u64 seed into per-cell Threefry, we first split the seed
    //    once (Threefry on counter=(0,0) with the raw seed as key) and
    //    use *that* output as the working key. Matches the protocol
    //    JAX/MLX use to turn a `PRNGKey(seed)` into the key the
    //    categorical sampler actually consumes — a no-op statistically
    //    but the bit pattern of u_i shifts to match.
    let raw_key = vec2<u32>(params.seed_lo, params.seed_hi);
    let key = threefry2x32_20(vec2<u32>(0u, 0u), raw_key);
    var best_score: f32 = NEG_INF;
    var picked: u32 = 0u;
    for (var i: u32 = 0u; i < params.inner; i = i + 1u) {
        let p = arena[base + i];
        if (p <= 0.0) { continue; }   // filtered by top_k / top_p
        let ctr = vec2<u32>(row, i);
        let rng2 = threefry2x32_20(ctr, key);
        // Two-word concat → 53-bit-ish double-equivalent uniform on [0, 1).
        // Then -log(-log(u)) = Gumbel(0, 1).
        let u: f32 = f32(rng2.x >> 8u) / 16777216.0;
        // Avoid log(0) = -inf if u rolled exactly 0.
        let u_safe: f32 = max(u, 1e-30);
        let g: f32 = -log(-log(u_safe));
        let score: f32 = log(p) + g;
        if (score > best_score) {
            best_score = score;
            picked = i;
        }
    }
    arena[params.out_off + row] = f32(picked);
}
