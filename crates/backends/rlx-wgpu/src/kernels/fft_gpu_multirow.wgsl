// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

// native-gpu-fft: multi-row on-chip FFT for SMALL n (<=1024). One workgroup
// handles `rows` independent FFT rows packed into a 16 KB workgroup buffer
// (rows = floor(2048 / n)), so a single small-n transform no longer leaves the
// workgroup underutilized — the MLX "threadgroup memory batching improves
// throughput for small n" idea. Each row r occupies sh[r*n .. r*n+n]; radix-2
// in-place (bit-reversal load, all stages, store), all rows in lockstep so the
// barriers stay uniform. `params.tile` carries `rows`.

struct Params {
    off: u32,
    dst_off: u32,
    n: u32,
    log2n: u32,
    inverse: u32,
    norm_scale: f32,
    outer: u32,
    tile: u32,           // rows per workgroup
    inner_stages: u32,
    q_or_hs: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> shm: array<vec2<f32>, 2048>; // 16 KB

fn cmulm(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn fft_multirow(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let rows = params.tile;
    let base_row = wgid.x * rows;
    let tid = lid.x;
    let tg = 256u;
    let half = n / 2u;

    // Bit-reversal load: rows*n elements, row r → sh[r*n + rev(k)].
    var idx = tid;
    loop {
        if (idx >= rows * n) { break; }
        let r = idx / n;
        let k = idx % n;
        let gr = base_row + r;
        if (gr < params.outer) {
            let src = params.off + gr * 2u * n;
            let rev = reverseBits(k) >> (32u - log2n);
            shm[r * n + rev] = vec2<f32>(arena[src + k], arena[src + n + k]);
        }
        idx = idx + tg;
    }
    workgroupBarrier();

    let sgn = select(-1.0, 1.0, params.inverse != 0u);
    let two_pi = 6.28318530717958647692;
    var len = 2u;
    loop {
        if (len > n) { break; }
        let h2 = len >> 1u;
        let theta_base = sgn * two_pi / f32(len);
        // rows * (n/2) butterflies, threads strided across all rows.
        var b = tid;
        loop {
            if (b >= rows * half) { break; }
            let r = b / half;
            let bb = b % half;
            let group = bb / h2;
            let kin = bb % h2;
            let i = r * n + group * len + kin;
            let j = i + h2;
            let t = cmulm(vec2<f32>(cos(theta_base * f32(kin)), sin(theta_base * f32(kin))), shm[j]);
            let u = shm[i];
            shm[i] = u + t;
            shm[j] = u - t;
            b = b + tg;
        }
        workgroupBarrier();
        len = len << 1u;
    }

    // Store.
    idx = tid;
    loop {
        if (idx >= rows * n) { break; }
        let r = idx / n;
        let k = idx % n;
        let gr = base_row + r;
        if (gr < params.outer) {
            let dst = params.dst_off + gr * 2u * n;
            arena[dst + k] = shm[r * n + k].x * params.norm_scale;
            arena[dst + n + k] = shm[r * n + k].y * params.norm_scale;
        }
        idx = idx + tg;
    }
}
