// Packed AdaLayerNorm backward: out = [dx ∥ dscale ∥ dshift] (1-D floats).
// Launch: grid=(mod_rows,1,1), block=(256,1,1).

struct Params {
    mod_rows: u32,
    seq_per_mod: u32,
    inner: u32,
    x_off: u32,
    scale_off: u32,
    dy_off: u32,
    out_off: u32,
    eps_bits: u32,
    layer_norm: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> partial_sum: array<f32, 256>;

fn wg_sum(tid: u32, val: f32, bsz: u32) -> f32 {
    partial_sum[tid] = val;
    workgroupBarrier();
    var stride = bsz / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial_sum[tid] = partial_sum[tid] + partial_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    return partial_sum[0];
}

@compute @workgroup_size(256)
fn ada_layer_norm_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = wid.x;
    if (m >= params.mod_rows || params.inner == 0u) { return; }
    let tid = lid.x;
    let bsz = 256u;
    let inner = params.inner;
    let nx = params.mod_rows * params.seq_per_mod * inner;
    let mod_len = params.mod_rows * inner;
    let mod_base = m * inner;
    let n_inv = 1.0 / f32(inner);
    let eps = bitcast<f32>(params.eps_bits);
    let do_ln = params.layer_norm != 0u;

    var i = tid;
    loop {
        if (i >= inner) { break; }
        arena[params.out_off + nx + mod_base + i] = 0.0;
        arena[params.out_off + nx + mod_len + mod_base + i] = 0.0;
        i += bsz;
    }
    workgroupBarrier();

    for (var seq: u32 = 0u; seq < params.seq_per_mod; seq = seq + 1u) {
        let row = m * params.seq_per_mod + seq;
        let x_base = params.x_off + row * inner;
        let dy_base = params.dy_off + row * inner;
        let dx_base = params.out_off + row * inner;

        var local_sum: f32 = 0.0;
        var local_sumsq: f32 = 0.0;
        i = tid;
        loop {
            if (i >= inner) { break; }
            let v = arena[x_base + i];
            local_sum = local_sum + v;
            local_sumsq = local_sumsq + v * v;
            i += bsz;
        }
        let sum_x = wg_sum(tid, local_sum, bsz);
        let sum_x2 = wg_sum(tid, local_sumsq, bsz);

        var mean: f32 = 0.0;
        var inv: f32;
        if (do_ln) {
            // STABLE TWO-PASS variance via a second workgroup reduction over
            // (x − mean)². The one-pass E[x²] − (E[x])² identity cancels in
            // f32 under a large DC offset (pre-norm activations), corrupting
            // the norm/gradient on wgpu only. `do_ln` is a uniform param, so
            // the extra `wg_sum` barrier is reached by every thread.
            mean = sum_x * n_inv;
            var local_sq: f32 = 0.0;
            i = tid;
            loop {
                if (i >= inner) { break; }
                let dd = arena[x_base + i] - mean;
                local_sq = local_sq + dd * dd;
                i += bsz;
            }
            let sum_sq = wg_sum(tid, local_sq, bsz);
            inv = inverseSqrt(sum_sq * n_inv + eps);
        } else {
            inv = inverseSqrt(sum_x2 * n_inv + eps);
        }

        var local_sy: f32 = 0.0;
        var local_sxh: f32 = 0.0;
        i = tid;
        loop {
            if (i >= inner) { break; }
            let n = (arena[x_base + i] - mean) * inv;
            let d = arena[dy_base + i];
            let sc = arena[params.scale_off + mod_base + i];
            let sy = d * (1.0 + sc);
            arena[params.out_off + nx + mod_base + i] =
                arena[params.out_off + nx + mod_base + i] + d * n;
            arena[params.out_off + nx + mod_len + mod_base + i] =
                arena[params.out_off + nx + mod_len + mod_base + i] + d;
            local_sy = local_sy + sy;
            local_sxh = local_sxh + sy * n;
            i += bsz;
        }
        let sum_sy = wg_sum(tid, local_sy, bsz);
        let sum_sxh = wg_sum(tid, local_sxh, bsz);
        let m_sy = sum_sy * n_inv;
        let m_sxh = sum_sxh * n_inv;

        i = tid;
        loop {
            if (i >= inner) { break; }
            let n = (arena[x_base + i] - mean) * inv;
            let d = arena[dy_base + i];
            let sc = arena[params.scale_off + mod_base + i];
            let sy = d * (1.0 + sc);
            if (do_ln) {
                arena[dx_base + i] = inv * (sy - m_sy - n * m_sxh);
            } else {
                let n_rms = arena[x_base + i] * inv;
                arena[dx_base + i] = inv * (sy - n_rms * m_sxh);
            }
            i += bsz;
        }
        workgroupBarrier();
    }
}
