// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

// native-gpu-fft: single-kernel on-chip FFT for n in (1024, 4096], in a separate
// module from fft_gpu.wgsl because its 4096-element workgroup buffer needs 32 KB
// (sh = 4096 * vec2<f32>). Only instantiated when the device reports
// max_compute_workgroup_storage_size >= 32 KB (Apple Silicon / desktop); WebGPU's
// portable 16 KB floor keeps the radix-2 (2048) / multi-kernel path. Ports the
// verified Metal MSL recipe: in-place DIT, digit-reversed load, highest sensible
// radix per size (radix-8 for pow-8, radix-4 incl. the 2·4^m mixed case via a
// trailing radix-2 stage, radix-2 floor).

struct Params {
    off: u32,
    dst_off: u32,
    n: u32,
    log2n: u32,
    inverse: u32,
    norm_scale: f32,
    outer: u32,
    tile: u32,
    inner_stages: u32,
    q_or_hs: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// 4096 * 8 bytes = 32 KB. Shared by all three entry points (one runs per dispatch).
var<workgroup> sh: array<vec2<f32>, 4096>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-register 4-point DFT (radix-4 butterfly core). rs=+1 forward, -1 inverse.
fn dft4(x0: vec2<f32>, x1: vec2<f32>, x2: vec2<f32>, x3: vec2<f32>, rs: f32) -> array<vec2<f32>, 4> {
    let t0 = x0 + x2;
    let t1 = x0 - x2;
    let t2 = x1 + x3;
    let t3 = x1 - x3;
    var o: array<vec2<f32>, 4>;
    o[0] = t0 + t2;
    o[2] = t0 - t2;
    o[1] = vec2<f32>(t1.x + rs * t3.y, t1.y - rs * t3.x);
    o[3] = vec2<f32>(t1.x - rs * t3.y, t1.y + rs * t3.x);
    return o;
}

const TWO_PI: f32 = 6.28318530717958647692;

// Radix-2 on-chip (any pow-2 n<=4096) — bit-reversal load + all radix-2 stages.
// Kept mainly as the A/B baseline (RLX_FFT_RADIX=2) on >=32 KB devices.
@compute @workgroup_size(256)
fn fft_radix2_big(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let src = params.off + row * 2u * n;
    let dst = params.dst_off + row * 2u * n;
    let tid = lid.x;
    let tg = 256u;

    var k = tid;
    loop {
        if (k >= n) { break; }
        let rev = reverseBits(k) >> (32u - log2n);
        sh[rev] = vec2<f32>(arena[src + k], arena[src + n + k]);
        k = k + tg;
    }
    workgroupBarrier();

    let sgn = select(-1.0, 1.0, params.inverse != 0u);
    var len = 2u;
    loop {
        if (len > n) { break; }
        let h2 = len >> 1u;
        let theta_base = sgn * TWO_PI / f32(len);
        var b = tid;
        loop {
            if (b >= n / 2u) { break; }
            let group = b / h2;
            let kin = b % h2;
            let i_lo = group * len + kin;
            let i_hi = i_lo + h2;
            let theta = theta_base * f32(kin);
            let t = cmul(vec2<f32>(cos(theta), sin(theta)), sh[i_hi]);
            let ulo = sh[i_lo];
            sh[i_lo] = ulo + t;
            sh[i_hi] = ulo - t;
            b = b + tg;
        }
        workgroupBarrier();
        len = len << 1u;
    }

    k = tid;
    loop {
        if (k >= n) { break; }
        arena[dst + k] = sh[k].x * params.norm_scale;
        arena[dst + n + k] = sh[k].y * params.norm_scale;
        k = k + tg;
    }
}

// Radix-4 in-place (pow-4 and the 2·4^m mixed case, e.g. n=2048 via a trailing
// radix-2 stage). Mixed-radix digit-reversal load → radix-4 DIT stages → store.
@compute @workgroup_size(256)
fn fft_radix4_big(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let src = params.off + row * 2u * n;
    let dst = params.dst_off + row * 2u * n;
    let tid = lid.x;
    let tg = 256u;
    let d = log2n >> 1u;       // radix-4 stages
    let has_r2 = log2n & 1u;   // leading factor 2 → trailing radix-2 stage

    var k = tid;
    loop {
        if (k >= n) { break; }
        var r = 0u;
        var kk = select(k, k >> 1u, has_r2 != 0u);
        for (var i = 0u; i < d; i = i + 1u) { r = (r << 2u) | (kk & 3u); kk = kk >> 2u; }
        if (has_r2 != 0u) { r = r | ((k & 1u) << (log2n - 1u)); }
        sh[r] = vec2<f32>(arena[src + k], arena[src + n + k]);
        k = k + tg;
    }
    workgroupBarrier();

    let sgn = select(-1.0, 1.0, params.inverse != 0u);
    let rs = select(1.0, -1.0, params.inverse != 0u);
    var dist = 1u;
    for (var s = 0u; s < d; s = s + 1u) {
        let l = dist << 2u;
        var j = tid;
        loop {
            if (j >= n / 4u) { break; }
            let kk = j & (dist - 1u);
            let base = (j / dist) * l + kk;
            let u0 = sh[base];
            var u1 = sh[base + dist];
            var u2 = sh[base + 2u * dist];
            var u3 = sh[base + 3u * dist];
            let a = sgn * TWO_PI * f32(kk) / f32(l);
            u1 = cmul(u1, vec2<f32>(cos(a), sin(a)));
            u2 = cmul(u2, vec2<f32>(cos(2.0 * a), sin(2.0 * a)));
            u3 = cmul(u3, vec2<f32>(cos(3.0 * a), sin(3.0 * a)));
            let y = dft4(u0, u1, u2, u3, rs);
            sh[base] = y[0];
            sh[base + dist] = y[1];
            sh[base + 2u * dist] = y[2];
            sh[base + 3u * dist] = y[3];
            j = j + tg;
        }
        workgroupBarrier();
        dist = dist << 2u;
    }

    if (has_r2 != 0u) {
        let l = dist << 1u; // 2*dist = n
        var j = tid;
        loop {
            if (j >= n / 2u) { break; }
            let kk = j & (dist - 1u);
            let base = (j / dist) * l + kk;
            let u0 = sh[base];
            var u1 = sh[base + dist];
            let a = sgn * TWO_PI * f32(kk) / f32(l);
            u1 = cmul(u1, vec2<f32>(cos(a), sin(a)));
            sh[base] = u0 + u1;
            sh[base + dist] = u0 - u1;
            j = j + tg;
        }
        workgroupBarrier();
    }

    k = tid;
    loop {
        if (k >= n) { break; }
        arena[dst + k] = sh[k].x * params.norm_scale;
        arena[dst + n + k] = sh[k].y * params.norm_scale;
        k = k + tg;
    }
}

// Radix-8 in-place (pure pow-8, e.g. n=4096). Base-8 digit-reversal load →
// radix-8 DIT stages → store. The 8-point DFT is two radix-4 cores + a W8
// radix-2 combine.
@compute @workgroup_size(256)
fn fft_radix8_big(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let src = params.off + row * 2u * n;
    let dst = params.dst_off + row * 2u * n;
    let tid = lid.x;
    let tg = 256u;
    let d = log2n / 3u; // base-8 digits

    var k = tid;
    loop {
        if (k >= n) { break; }
        var r = 0u;
        var kk = k;
        for (var i = 0u; i < d; i = i + 1u) { r = (r << 3u) | (kk & 7u); kk = kk >> 3u; }
        sh[r] = vec2<f32>(arena[src + k], arena[src + n + k]);
        k = k + tg;
    }
    workgroupBarrier();

    let sgn = select(-1.0, 1.0, params.inverse != 0u);
    let rs = select(1.0, -1.0, params.inverse != 0u);
    let r2 = 0.70710678118654752;
    var dist = 1u;
    for (var s = 0u; s < d; s = s + 1u) {
        let l = dist << 3u;
        var j = tid;
        loop {
            if (j >= n / 8u) { break; }
            let kk = j & (dist - 1u);
            let base = (j / dist) * l + kk;
            var u: array<vec2<f32>, 8>;
            for (var i = 0u; i < 8u; i = i + 1u) { u[i] = sh[base + i * dist]; }
            let a = sgn * TWO_PI * f32(kk) / f32(l);
            for (var i = 1u; i < 8u; i = i + 1u) {
                let ang = f32(i) * a;
                u[i] = cmul(u[i], vec2<f32>(cos(ang), sin(ang)));
            }
            let e = dft4(u[0], u[2], u[4], u[6], rs);
            let o = dft4(u[1], u[3], u[5], u[7], rs);
            let c0 = o[0];
            let c1 = cmul(vec2<f32>(r2, sgn * r2), o[1]);
            let c2 = cmul(vec2<f32>(0.0, sgn), o[2]);
            let c3 = cmul(vec2<f32>(-r2, sgn * r2), o[3]);
            sh[base] = e[0] + c0;
            sh[base + 4u * dist] = e[0] - c0;
            sh[base + 1u * dist] = e[1] + c1;
            sh[base + 5u * dist] = e[1] - c1;
            sh[base + 2u * dist] = e[2] + c2;
            sh[base + 6u * dist] = e[2] - c2;
            sh[base + 3u * dist] = e[3] + c3;
            sh[base + 7u * dist] = e[3] - c3;
            j = j + tg;
        }
        workgroupBarrier();
        dist = dist << 3u;
    }

    k = tid;
    loop {
        if (k >= n) { break; }
        arena[dst + k] = sh[k].x * params.norm_scale;
        arena[dst + n + k] = sh[k].y * params.norm_scale;
        k = k + tg;
    }
}
