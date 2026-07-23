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

// FakeQuantize forward: clamp(round(x / s), -q_max, q_max) * s.
// Matches `rlx_cpu::thunk::ops::quant::exec_fake_quantize` for Fixed,
// PerBatch, and EMA (LSQ / Backward stay on HostOp or dedicated kernels).
// Channel layout:
//   c = (i / inner) % chan_dim
// Rounding: Rust `f32::round` (half away from zero), not ties-to-even.

__device__ __forceinline__ float apply_fq(float x, float s, float q_max) {
    float scaled = x / s;
    // Match Rust `f32::round` (half away from zero): sign(x)*floor(|x|+0.5).
    // Do not use copysign alone — sign(0)=0, so 0 rounds to 0.
    float sgn = (scaled > 0.0f) - (scaled < 0.0f);
    float rounded = sgn * floorf(fabsf(scaled) + 0.5f);
    float qv = fminf(fmaxf(rounded, -q_max), q_max);
    return qv * s;
}

__device__ __forceinline__ unsigned int fq_channel_of(
        unsigned int i, unsigned int chan_dim, unsigned int inner) {
    if (chan_dim <= 1u) return 0u;
    return (i / inner) % chan_dim;
}

// One thread per element. Scale from `scale_off[c]` (Fixed).
extern "C" __global__ void fake_quantize_fixed(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int in_off,
    unsigned int scale_off,
    unsigned int out_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q_max = __int_as_float((int)q_max_bits);
    unsigned int c = fq_channel_of(i, chan_dim, inner);
    float s = fmaxf(arena[scale_off + c], 1e-12f);
    arena[out_off + i] = apply_fq(arena[in_off + i], s, q_max);
}

// One thread per channel. s = max(|x|) / q_max, then quantize that channel
// (PerBatch).
extern "C" __global__ void fake_quantize_perbatch(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chan_dim) return;
    float q_max = __int_as_float((int)q_max_bits);

    float max_abs = 0.0f;
    unsigned int stride = chan_dim * inner;
    unsigned int outer = n / (stride == 0u ? 1u : stride);
    for (unsigned int o = 0u; o < outer; ++o) {
        unsigned int base = o * stride + c * inner;
        for (unsigned int j = 0u; j < inner; ++j) {
            max_abs = fmaxf(max_abs, fabsf(arena[in_off + base + j]));
        }
    }
    // axis=None: chan_dim=1, inner=n → outer=1, single scan of all elements.
    // When n is not a multiple of stride (shouldn't happen), fall back to a
    // full-tensor scan for this channel.
    if (outer * stride != n) {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                max_abs = fmaxf(max_abs, fabsf(arena[in_off + i]));
            }
        }
    }

    float s = fmaxf(max_abs / q_max, 1e-12f);

    for (unsigned int o = 0u; o < outer; ++o) {
        unsigned int base = o * stride + c * inner;
        for (unsigned int j = 0u; j < inner; ++j) {
            unsigned int idx = base + j;
            arena[out_off + idx] = apply_fq(arena[in_off + idx], s, q_max);
        }
    }
    if (outer * stride != n) {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                arena[out_off + i] = apply_fq(arena[in_off + i], s, q_max);
            }
        }
    }
}

// One thread per channel. Per-batch max-abs → blend into running scale
// state (in-place), then quantize. Matches CPU EMA:
//   cur = max(|x|) / q_max
//   state' = (state <= 0) ? cur : decay·state + (1−decay)·cur
extern "C" __global__ void fake_quantize_ema(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int decay_bits,
    unsigned int in_off,
    unsigned int scale_off,
    unsigned int out_off
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chan_dim) return;
    float q_max = __int_as_float((int)q_max_bits);
    float decay = __int_as_float((int)decay_bits);

    float max_abs = 0.0f;
    unsigned int stride = chan_dim * inner;
    unsigned int outer = n / (stride == 0u ? 1u : stride);
    for (unsigned int o = 0u; o < outer; ++o) {
        unsigned int base = o * stride + c * inner;
        for (unsigned int j = 0u; j < inner; ++j) {
            max_abs = fmaxf(max_abs, fabsf(arena[in_off + base + j]));
        }
    }
    if (outer * stride != n) {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                max_abs = fmaxf(max_abs, fabsf(arena[in_off + i]));
            }
        }
    }

    float cur = fmaxf(max_abs / q_max, 1e-12f);
    float prev = arena[scale_off + c];
    float blended = (prev <= 0.0f) ? cur : (decay * prev + (1.0f - decay) * cur);
    arena[scale_off + c] = blended;
    float s = blended;

    for (unsigned int o = 0u; o < outer; ++o) {
        unsigned int base = o * stride + c * inner;
        for (unsigned int j = 0u; j < inner; ++j) {
            unsigned int idx = base + j;
            arena[out_off + idx] = apply_fq(arena[in_off + idx], s, q_max);
        }
    }
    if (outer * stride != n) {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                arena[out_off + i] = apply_fq(arena[in_off + i], s, q_max);
            }
        }
    }
}

// FakeQuantizeLSQ forward is identical to Fixed — reuse `fake_quantize_fixed`.

__device__ __forceinline__ float round_half_away_fq(float x) {
    float sgn = (x > 0.0f) - (x < 0.0f);
    return sgn * floorf(fabsf(x) + 0.5f);
}

__device__ __forceinline__ float lsq_psi(float z, float q_max) {
    if (fabsf(z) <= q_max) {
        return -z + round_half_away_fq(z);
    }
    return (z > 0.0f) ? q_max : -q_max;
}

// LSQ ∂L/∂x: STE-clipped — dx = dy when |x/s| ≤ q_max, else 0.
extern "C" __global__ void fake_quantize_lsq_bwd_x(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int x_off,
    unsigned int scale_off,
    unsigned int dy_off,
    unsigned int dx_off
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float q_max = __int_as_float((int)q_max_bits);
    unsigned int c = fq_channel_of(i, chan_dim, inner);
    float s = fmaxf(arena[scale_off + c], 1e-12f);
    float z = arena[x_off + i] / s;
    arena[dx_off + i] = (fabsf(z) <= q_max) ? arena[dy_off + i] : 0.0f;
}

// LSQ ∂L/∂s[c]: sum_i ψ(x_i/s) * dy_i
// ψ(z) = -z + round(z) inside range, sign(z)·q_max outside.
// One thread per channel.
extern "C" __global__ void fake_quantize_lsq_bwd_scale(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int x_off,
    unsigned int scale_off,
    unsigned int dy_off,
    unsigned int dscale_off
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chan_dim) return;
    float q_max = __int_as_float((int)q_max_bits);
    float s = fmaxf(arena[scale_off + c], 1e-12f);

    float acc = 0.0f;
    unsigned int stride = chan_dim * inner;
    unsigned int outer = n / (stride == 0u ? 1u : stride);

    if (outer * stride == n) {
        for (unsigned int o = 0u; o < outer; ++o) {
            unsigned int base = o * stride + c * inner;
            for (unsigned int j = 0u; j < inner; ++j) {
                unsigned int idx = base + j;
                float z = arena[x_off + idx] / s;
                acc += lsq_psi(z, q_max) * arena[dy_off + idx];
            }
        }
    } else {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                float z = arena[x_off + i] / s;
                acc += lsq_psi(z, q_max) * arena[dy_off + i];
            }
        }
    }
    arena[dscale_off + c] = acc;
}

__device__ __forceinline__ float fq_ste_dx(
        float x, float dy, float s, float bound, unsigned int ste_kind) {
    if (ste_kind == 0u) {
        return dy;
    }
    if (ste_kind == 1u) {
        return (fabsf(x) <= bound) ? dy : 0.0f;
    }
    if (ste_kind == 2u) {
        float t = tanhf(x / s);
        return dy * (1.0f - t * t);
    }
    float attenuation = fmaxf(1.0f - fabsf(x / bound), 0.0f);
    return dy * attenuation;
}

// FakeQuantizeBackward (STE). Recomputes PerBatch scale from x, then:
//   0 Identity:         dx = dy
//   1 ClippedIdentity:  dx = dy if |x| ≤ q_max·s else 0
//   2 Tanh:             dx = dy · (1 − tanh²(x/s))
//   3 HardTanh:         dx = dy · max(0, 1 − |x/(q_max·s)|)
// One thread per channel (derives s, then writes all elems of that channel).
extern "C" __global__ void fake_quantize_backward(
    float* arena,
    unsigned int n,
    unsigned int chan_dim,
    unsigned int inner,
    unsigned int q_max_bits,
    unsigned int ste_kind,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int dx_off
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chan_dim) return;
    float q_max = __int_as_float((int)q_max_bits);

    float max_abs = 0.0f;
    unsigned int stride = chan_dim * inner;
    unsigned int outer = n / (stride == 0u ? 1u : stride);
    if (outer * stride == n) {
        for (unsigned int o = 0u; o < outer; ++o) {
            unsigned int base = o * stride + c * inner;
            for (unsigned int j = 0u; j < inner; ++j) {
                max_abs = fmaxf(max_abs, fabsf(arena[x_off + base + j]));
            }
        }
    } else {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                max_abs = fmaxf(max_abs, fabsf(arena[x_off + i]));
            }
        }
    }
    float s = fmaxf(max_abs / q_max, 1e-12f);
    float bound = q_max * s;

    if (outer * stride == n) {
        for (unsigned int o = 0u; o < outer; ++o) {
            unsigned int base = o * stride + c * inner;
            for (unsigned int j = 0u; j < inner; ++j) {
                unsigned int idx = base + j;
                arena[dx_off + idx] = fq_ste_dx(
                    arena[x_off + idx], arena[dy_off + idx], s, bound, ste_kind);
            }
        }
    } else {
        for (unsigned int i = 0u; i < n; ++i) {
            if (fq_channel_of(i, chan_dim, inner) == c) {
                arena[dx_off + i] = fq_ste_dx(
                    arena[x_off + i], arena[dy_off + i], s, bound, ste_kind);
            }
        }
    }
}
