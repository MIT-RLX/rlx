// Multi-kernel f32 FFT (gpu-fft strategy), RLX 2N real-block layout.

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

fn re_at(base: u32, k: u32, n: u32) -> f32 {
    return arena[base + k];
}
fn im_at(base: u32, k: u32, n: u32) -> f32 {
    return arena[base + n + k];
}
fn set_re(base: u32, k: u32, n: u32, v: f32) {
    arena[base + k] = v;
}
fn set_im(base: u32, k: u32, n: u32, v: f32) {
    arena[base + n + k] = v;
}

// Single-kernel path (N <= 1024): bit-reverse load + all stages in TG mem.
var<workgroup> sre_full: array<f32, 1024>;
var<workgroup> sim_full: array<f32, 1024>;

@compute @workgroup_size(256)
fn fft_radix2_full(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let src_base = params.off + row * 2u * n;
    let dst_base = params.dst_off + row * 2u * n;
    let tid = lid.x;
    let tg_size = 256u;

    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        let rev = reverseBits(k) >> (32u - log2n);
        sre_full[rev] = re_at(src_base, k, n);
        sim_full[rev] = im_at(src_base, k, n);
        k = k + tg_size;
    }
    workgroupBarrier();

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    let two_pi = 6.28318530717958647692;
    var len: u32 = 2u;
    loop {
        if (len > n) { break; }
        let h2 = len >> 1u;
        let theta_base = sign * two_pi / f32(len);
        var b: u32 = tid;
        loop {
            if (b >= n / 2u) { break; }
            let group = b / h2;
            let k_in = b % h2;
            let i_lo = group * len + k_in;
            let i_hi = i_lo + h2;
            let theta = theta_base * f32(k_in);
            let wre = cos(theta);
            let wim = sin(theta);
            let t_re = wre * sre_full[i_hi] - wim * sim_full[i_hi];
            let t_im = wre * sim_full[i_hi] + wim * sre_full[i_hi];
            let u_re = sre_full[i_lo];
            let u_im = sim_full[i_lo];
            sre_full[i_lo] = u_re + t_re;
            sim_full[i_lo] = u_im + t_im;
            sre_full[i_hi] = u_re - t_re;
            sim_full[i_hi] = u_im - t_im;
            b = b + tg_size;
        }
        workgroupBarrier();
        len = len << 1u;
    }

    k = tid;
    loop {
        if (k >= n) { break; }
        set_re(dst_base, k, n, sre_full[k] * params.norm_scale);
        set_im(dst_base, k, n, sim_full[k] * params.norm_scale);
        k = k + tg_size;
    }
}

// Bit-reverse one row before multi-kernel outer stages.
@compute @workgroup_size(256)
fn fft_bit_reverse(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let n = params.n;
    let k = gid.x;
    if (k >= n) { return; }
    let base = params.off + row * 2u * n;
    let rev = reverseBits(k) >> (32u - params.log2n);
    if (k >= rev) { return; }
    let tr = re_at(base, k, n);
    let ti = im_at(base, k, n);
    set_re(base, k, n, re_at(base, rev, n));
    set_im(base, k, n, im_at(base, rev, n));
    set_re(base, rev, n, tr);
    set_im(base, rev, n, ti);
}

// Inner shared-memory tile (tile <= 1024).
var<workgroup> sre_in: array<f32, 1024>;
var<workgroup> sim_in: array<f32, 1024>;

@compute @workgroup_size(512)
fn fft_inner(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let n = params.n;
    let tile = params.tile;
    let half_tile = tile / 2u;
    let tile_id = wgid.x;
    let num_tiles = (n + tile - 1u) / tile;
    if (tile_id >= num_tiles) { return; }
    let local = lid.x;
    if (local >= half_tile) { return; }

    let row_base = params.off + row * 2u * n;
    let tile_base = tile_id * tile;

    if (tile_base + local < n) {
        sre_in[local] = re_at(row_base, tile_base + local, n);
        sim_in[local] = im_at(row_base, tile_base + local, n);
    }
    if (tile_base + local + half_tile < n) {
        sre_in[local + half_tile] = re_at(row_base, tile_base + local + half_tile, n);
        sim_in[local + half_tile] = im_at(row_base, tile_base + local + half_tile, n);
    }
    workgroupBarrier();

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    let pi = 3.14159265358979323846;
    for (var s: u32 = 0u; s < params.inner_stages; s = s + 1u) {
        let hs = 1u << s;
        let k = local % hs;
        let i = (local / hs) * (hs * 2u) + k;
        let j = i + hs;
        let angle = sign * pi * f32(k) / f32(hs);
        let cos_a = cos(angle);
        let sin_a = sin(angle);
        let ur = sre_in[i];
        let ui = sim_in[i];
        let vr = cos_a * sre_in[j] - sin_a * sim_in[j];
        let vi = sin_a * sre_in[j] + cos_a * sim_in[j];
        sre_in[i] = ur + vr;
        sim_in[i] = ui + vi;
        sre_in[j] = ur - vr;
        sim_in[j] = ui - vi;
        workgroupBarrier();
    }

    let scale = params.norm_scale;
    if (tile_base + local < n) {
        set_re(row_base, tile_base + local, n, sre_in[local] * scale);
        set_im(row_base, tile_base + local, n, sim_in[local] * scale);
    }
    if (tile_base + local + half_tile < n) {
        set_re(row_base, tile_base + local + half_tile, n, sre_in[local + half_tile] * scale);
        set_im(row_base, tile_base + local + half_tile, n, sim_in[local + half_tile] * scale);
    }
}

@compute @workgroup_size(256)
fn fft_outer_r4(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let n = params.n;
    let q = params.q_or_hs;
    let tid = gid.x;
    if (tid >= n / 4u) { return; }

    let base = params.off + row * 2u * n;
    let k = tid % q;
    let group = tid / q;
    let p = group * (q * 4u) + k;

    let ar = re_at(base, p, n);
    let ai = im_at(base, p, n);
    let br = re_at(base, p + q, n);
    let bi = im_at(base, p + q, n);
    let cr = re_at(base, p + q * 2u, n);
    let ci = im_at(base, p + q * 2u, n);
    let dr = re_at(base, p + q * 3u, n);
    let di = im_at(base, p + q * 3u, n);

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    let neg_sign = select(1.0, -1.0, params.inverse != 0u);
    let angle1 = sign * 3.14159265358979323846 * f32(k) / f32(q);
    let cos1 = cos(angle1);
    let sin1 = sin(angle1);
    let w1b_r = cos1 * br - sin1 * bi;
    let w1b_i = sin1 * br + cos1 * bi;
    let w1d_r = cos1 * dr - sin1 * di;
    let w1d_i = sin1 * dr + cos1 * di;

    let u0r = ar + w1b_r;
    let u0i = ai + w1b_i;
    let u1r = ar - w1b_r;
    let u1i = ai - w1b_i;
    let u2r = cr + w1d_r;
    let u2i = ci + w1d_i;
    let u3r = cr - w1d_r;
    let u3i = ci - w1d_i;

    let angle2a = sign * 3.14159265358979323846 * f32(k) / f32(q * 2u);
    let cos2a = cos(angle2a);
    let sin2a = sin(angle2a);
    let cos2b = neg_sign * sin2a;
    let sin2b = sign * cos2a;

    let w2a_u2r = cos2a * u2r - sin2a * u2i;
    let w2a_u2i = sin2a * u2r + cos2a * u2i;
    let w2b_u3r = cos2b * u3r - sin2b * u3i;
    let w2b_u3i = sin2b * u3r + cos2b * u3i;

    let scale = params.norm_scale;
    set_re(base, p, n, (u0r + w2a_u2r) * scale);
    set_im(base, p, n, (u0i + w2a_u2i) * scale);
    set_re(base, p + q * 2u, n, (u0r - w2a_u2r) * scale);
    set_im(base, p + q * 2u, n, (u0i - w2a_u2i) * scale);
    set_re(base, p + q, n, (u1r + w2b_u3r) * scale);
    set_im(base, p + q, n, (u1i + w2b_u3i) * scale);
    set_re(base, p + q * 3u, n, (u1r - w2b_u3r) * scale);
    set_im(base, p + q * 3u, n, (u1i - w2b_u3i) * scale);
}

@compute @workgroup_size(256)
fn fft_outer_r2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let row = wgid.y;
    if (row >= params.outer) { return; }
    let n = params.n;
    let half_stride = params.q_or_hs;
    let tid = gid.x;
    if (tid >= n / 2u) { return; }

    let base = params.off + row * 2u * n;
    let k = tid % half_stride;
    let i = (tid / half_stride) * (half_stride * 2u) + k;
    let j = i + half_stride;

    let sign = select(-1.0, 1.0, params.inverse != 0u);
    let angle = sign * 3.14159265358979323846 * f32(k) / f32(half_stride);
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let ur = re_at(base, i, n);
    let ui = im_at(base, i, n);
    let vr = cos_a * re_at(base, j, n) - sin_a * im_at(base, j, n);
    let vi = sin_a * re_at(base, j, n) + cos_a * im_at(base, j, n);
    let scale = params.norm_scale;
    set_re(base, i, n, (ur + vr) * scale);
    set_im(base, i, n, (ui + vi) * scale);
    set_re(base, j, n, (ur - vr) * scale);
    set_im(base, j, n, (ui - vi) * scale);
}
