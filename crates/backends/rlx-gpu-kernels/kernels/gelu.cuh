// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Shared GELU helpers — match rlx-cpu `scalar_gelu` (Abramowitz & Stegun 7.1.26).

__device__ __forceinline__ float gelu_erf(float x) {
    float arg = x * 0.70710678118654752f;
    float s = (arg >= 0.0f) ? 1.0f : -1.0f;
    float xa = fabsf(arg);
    float t = 1.0f / (1.0f + 0.3275911f * xa);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                + t * (-1.453152027f + t * 1.061405429f))));
    float e = s * (1.0f - poly * expf(-xa * xa));
    return 0.5f * x * (1.0f + e);
}

__device__ __forceinline__ float gelu_approx(float x) {
    const float c = 0.7978845608028654f;
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    inner = fminf(fmaxf(inner, -15.0f), 15.0f);
    return 0.5f * x * (1.0f + tanhf(inner));
}
