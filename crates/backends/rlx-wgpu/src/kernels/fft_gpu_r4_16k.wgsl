// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// native-gpu-fft: portable 16 KB on-chip radix-4 FFT for n in (1024, 2048].
// sh = 2048 * vec2<f32> = 16 KB — within WebGPU's maxComputeWorkgroupStorageSize
// floor, so it needs no device-limit gate (unlike the 32 KB radix-8 path, which
// halves wgpu occupancy and regresses). Right-sizing the workgroup buffer to the
// transform keeps two workgroups resident per core where the 32 KB kernel allowed
// only one. Generalized radix-4 DIT: mixed-radix digit-reversal load, radix-4
// stages, and a trailing radix-2 stage for the 2·4^m case (n=2048 = 2·4^5).

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

var<workgroup> sh16: array<vec2<f32>, 2048>; // 16 KB

fn cmul16(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn dft4_16(x0: vec2<f32>, x1: vec2<f32>, x2: vec2<f32>, x3: vec2<f32>, rs: f32) -> array<vec2<f32>, 4> {
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

const TWO_PI_16: f32 = 6.28318530717958647692;

@compute @workgroup_size(256)
fn fft_radix4_16k(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) ngs: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.y + wgid.z * ngs.y;
    if (row >= params.outer) { return; }
    let src = params.off + row * 2u * n;
    let dst = params.dst_off + row * 2u * n;
    let tid = lid.x;
    let tg = 256u;
    let d = log2n >> 1u;
    let has_r2 = log2n & 1u;

    var k = tid;
    loop {
        if (k >= n) { break; }
        var r = 0u;
        var kk = select(k, k >> 1u, has_r2 != 0u);
        for (var i = 0u; i < d; i = i + 1u) { r = (r << 2u) | (kk & 3u); kk = kk >> 2u; }
        if (has_r2 != 0u) { r = r | ((k & 1u) << (log2n - 1u)); }
        sh16[r] = vec2<f32>(arena[src + k], arena[src + n + k]);
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
            let u0 = sh16[base];
            var u1 = sh16[base + dist];
            var u2 = sh16[base + 2u * dist];
            var u3 = sh16[base + 3u * dist];
            let a = sgn * TWO_PI_16 * f32(kk) / f32(l);
            u1 = cmul16(u1, vec2<f32>(cos(a), sin(a)));
            u2 = cmul16(u2, vec2<f32>(cos(2.0 * a), sin(2.0 * a)));
            u3 = cmul16(u3, vec2<f32>(cos(3.0 * a), sin(3.0 * a)));
            let y = dft4_16(u0, u1, u2, u3, rs);
            sh16[base] = y[0];
            sh16[base + dist] = y[1];
            sh16[base + 2u * dist] = y[2];
            sh16[base + 3u * dist] = y[3];
            j = j + tg;
        }
        workgroupBarrier();
        dist = dist << 2u;
    }

    if (has_r2 != 0u) {
        let l = dist << 1u;
        var j = tid;
        loop {
            if (j >= n / 2u) { break; }
            let kk = j & (dist - 1u);
            let base = (j / dist) * l + kk;
            let u0 = sh16[base];
            var u1 = sh16[base + dist];
            let a = sgn * TWO_PI_16 * f32(kk) / f32(l);
            u1 = cmul16(u1, vec2<f32>(cos(a), sin(a)));
            sh16[base] = u0 + u1;
            sh16[base + dist] = u0 - u1;
            j = j + tg;
        }
        workgroupBarrier();
    }

    k = tid;
    loop {
        if (k >= n) { break; }
        arena[dst + k] = sh16[k].x * params.norm_scale;
        arena[dst + n + k] = sh16[k].y * params.norm_scale;
        k = k + tg;
    }
}
