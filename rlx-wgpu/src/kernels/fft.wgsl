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

// 1D radix-2 Cooley-Tukey FFT, f32, in-place per row. One workgroup
// per row; layout matches rlx-cpu / rlx-metal exactly — each row is
// 2N f32 with the first N elements real and the next N imaginary.
// Capped at N=1024 by workgroup memory (8KB per of `sre` + `sim`).

struct Params {
    src_off: u32,
    dst_off: u32,
    n: u32,
    log2n: u32,
    inverse: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const FFT_N_MAX: u32 = 1024u;
var<workgroup> sre: array<f32, 1024>;
var<workgroup> sim: array<f32, 1024>;

@compute @workgroup_size(256)
fn fft_radix2(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = params.n;
    let log2n = params.log2n;
    let row = wgid.x;
    let tid = lid.x;
    let tg_size = 256u;
    let row_base = row * 2u * n;

    // Bit-reverse load. `reverse_bits` is a 32-bit hardware op; shift
    // right by (32 - log2n) to keep only the relevant bits.
    var k: u32 = tid;
    loop {
        if (k >= n) { break; }
        let rev = reverseBits(k) >> (32u - log2n);
        sre[rev] = arena[params.src_off + row_base + k];
        sim[rev] = arena[params.src_off + row_base + n + k];
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
            let k_in  = b % h2;
            let i_lo  = group * len + k_in;
            let i_hi  = i_lo + h2;
            let theta = theta_base * f32(k_in);
            let wre = cos(theta);
            let wim = sin(theta);
            let t_re = wre * sre[i_hi] - wim * sim[i_hi];
            let t_im = wre * sim[i_hi] + wim * sre[i_hi];
            let u_re = sre[i_lo];
            let u_im = sim[i_lo];
            sre[i_lo] = u_re + t_re;
            sim[i_lo] = u_im + t_im;
            sre[i_hi] = u_re - t_re;
            sim[i_hi] = u_im - t_im;
            b = b + tg_size;
        }
        workgroupBarrier();
        len = len << 1u;
    }

    // Store back to dst (may equal src — loads already pulled to WG mem).
    k = tid;
    loop {
        if (k >= n) { break; }
        arena[params.dst_off + row_base + k]     = sre[k];
        arena[params.dst_off + row_base + n + k] = sim[k];
        k = k + tg_size;
    }
}
