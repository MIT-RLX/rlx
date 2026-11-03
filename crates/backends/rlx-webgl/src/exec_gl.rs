// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WebGL2 render-to-texture executor for a [`Plan`].
//!
//! Every tensor is an `R32F` texture (one float per texel); every op
//! is a fragment shader that writes one output element per fragment (an
//! ungrouped conv3d computes up to 16 channels per fragment into RGBA scratch
//! targets, then copies them out). The GLSL
//! mirrors [`crate::exec_cpu`] one-to-one (same op/activation codes), so the
//! numerics are the ones the native parity test verifies — only the GL
//! plumbing here needs in-browser validation. WebGL2 `readPixels` is
//! synchronous, so the whole path is synchronous.

use crate::plan::{Act, Bin, Cmp, LeafSource, Plan, Red, ScatterRed, Step};
use crate::{Result, WebglError};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{
    WebGl2RenderingContext as GL, WebGlProgram, WebGlShader, WebGlTexture, WebGlUniformLocation,
};

const HEAD: &str = "#version 300 es\nprecision highp float; precision highp int;\n";

// Activation forward/derivative — must match `exec_cpu::act_f` / `act_df`.
// Codes: 0 Relu 1 Neg 2 Exp 3 Log 4 Sqrt 5 Rsqrt 6 Sigmoid 7 Tanh 8 Abs 9 Sin 10 Cos 11 Silu 12 Recip
// 13 Gelu 14 GeluApprox 15 Softplus 16 Tan 17 Atan 18 Round 19 Floor 20 Ceil 21 Sign 22 Elu
// 23 Erf 24 HardSwish 25 HardSigmoid 26 Mish 27 Softsign 28 LogSigmoid.
const ACT_FNS: &str = r#"
float sigmoidf(float x) { return 1.0 / (1.0 + exp(-x)); }
float erfAS(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float p = t * (0.2548296 + t * (-0.28449674 + t * (1.4214138 + t * (-1.4531521 + t * 1.0614054))));
    float r = 1.0 - p * exp(-x * x);
    return x < 0.0 ? -r : r;
}
float softplusf(float x) { return max(x, 0.0) + log(1.0 + exp(-abs(x))); }
float inside(float v, float lo, float hi) { return (v > lo && v < hi) ? 1.0 : 0.0; }
float geluArg(float x) { return clamp(0.7978846 * (x + 0.044715 * x * x * x), -15.0, 15.0); }
float actF(int k, float x) {
    if (k == 0) return max(x, 0.0);
    else if (k == 1) return -x;
    else if (k == 2) return exp(x);
    else if (k == 3) return log(x);
    else if (k == 4) return sqrt(x);
    else if (k == 5) return inversesqrt(x);
    else if (k == 6) return sigmoidf(x);
    else if (k == 7) return tanh(x);
    else if (k == 8) return abs(x);
    else if (k == 9) return sin(x);
    else if (k == 10) return cos(x);
    else if (k == 11) return x * sigmoidf(x);
    else if (k == 13) return 0.5 * x * (1.0 + erfAS(x * 0.70710678));
    else if (k == 14) return 0.5 * x * (1.0 + tanh(geluArg(x)));
    else if (k == 15) return softplusf(x);
    else if (k == 16) return tan(x);
    else if (k == 17) return atan(x);
    else if (k == 18) return roundEven(x);
    else if (k == 19) return floor(x);
    else if (k == 20) return ceil(x);
    else if (k == 21) return sign(x);
    else if (k == 22) return x > 0.0 ? x : exp(x) - 1.0;
    else if (k == 23) return erfAS(x);
    else if (k == 24) return x * clamp(x + 3.0, 0.0, 6.0) / 6.0;
    else if (k == 25) return clamp(x / 6.0 + 0.5, 0.0, 1.0);
    else if (k == 26) return x * tanh(softplusf(x));
    else if (k == 27) return x / (1.0 + abs(x));
    else if (k == 28) return min(x, 0.0) - log(1.0 + exp(-abs(x)));
    else return 1.0 / x;
}
float actDF(int k, float x) {
    if (k == 0) return x > 0.0 ? 1.0 : 0.0;
    else if (k == 1) return -1.0;
    else if (k == 2) return exp(x);
    else if (k == 3) return 1.0 / x;
    else if (k == 4) return 0.5 / sqrt(x);
    else if (k == 5) return -0.5 / (x * sqrt(x));
    else if (k == 6) { float s = sigmoidf(x); return s * (1.0 - s); }
    else if (k == 7) { float t = tanh(x); return 1.0 - t * t; }
    else if (k == 8) return sign(x);
    else if (k == 9) return cos(x);
    else if (k == 10) return -sin(x);
    else if (k == 11) { float s = sigmoidf(x); return s + x * s * (1.0 - s); }
    else if (k == 13) return 0.5 * (1.0 + erfAS(x * 0.70710678)) + x * exp(-0.5 * x * x) * 0.3989423;
    else if (k == 14) {
        float t = tanh(geluArg(x));
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978846 * (1.0 + 0.134145 * x * x);
    }
    else if (k == 15) return sigmoidf(x);
    else if (k == 16) { float c = cos(x); return 1.0 / (c * c); }
    else if (k == 17) return 1.0 / (1.0 + x * x);
    else if (k == 18) return 1.0;
    else if (k == 19 || k == 20 || k == 21) return 0.0;
    else if (k == 22) return x > 0.0 ? 1.0 : exp(x);
    else if (k == 23) return 1.1283792 * exp(-x * x);
    else if (k == 24) return (clamp(x + 3.0, 0.0, 6.0) + x * inside(x, -3.0, 3.0)) / 6.0;
    else if (k == 25) return inside(x, -3.0, 3.0) / 6.0;
    else if (k == 26) { float t = tanh(softplusf(x)); return t + x * (1.0 - t * t) * sigmoidf(x); }
    else if (k == 27) { float d = 1.0 + abs(x); return 1.0 / (d * d); }
    else if (k == 28) return 1.0 / (1.0 + exp(x));
    else return -1.0 / (x * x);
}
"#;

const VERT: &str = r#"#version 300 es
void main() {
    vec2 v = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
    gl_Position = vec4(v, 0.0, 1.0);
}"#;

const FS_UNARY: &str = r#"uniform highp sampler2D A; uniform int uAct;
out vec4 o;
void main() { o = vec4(actF(uAct, texelFetch(A, ivec2(gl_FragCoord.xy), 0).r), 0.0, 0.0, 1.0); }"#;

const FS_ACTBACK: &str = r#"uniform highp sampler2D X; uniform highp sampler2D DY; uniform int uAct;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    o = vec4(texelFetch(DY, p, 0).r * actDF(uAct, texelFetch(X, p, 0).r), 0.0, 0.0, 1.0);
}"#;

const FS_BINARY: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B; uniform int uOp;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float a = texelFetch(A, p, 0).r;
    float b = texelFetch(B, p, 0).r;
    float r = uOp == 0 ? a + b : uOp == 1 ? a - b : uOp == 2 ? a * b : uOp == 3 ? a / b
            : uOp == 4 ? max(a, b) : uOp == 5 ? min(a, b) : uOp == 6 ? pow(a, b)
            : uOp == 7 ? a - b * trunc(a / b)
            : uOp == 8 ? float(int(a) & int(b))
            : uOp == 9 ? float(int(a) | int(b))
            : uOp == 10 ? float(int(a) ^ int(b))
            : uOp == 11 ? float(int(a) << int(b))
            : uOp == 12 ? float(int(a) >> int(b))
            : atan(a, b);
    o = vec4(r, 0.0, 0.0, 1.0);
}"#;

const FS_COMPARE: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B; uniform int uOp;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float a = texelFetch(A, p, 0).r;
    float b = texelFetch(B, p, 0).r;
    bool t = uOp == 0 ? a == b : uOp == 1 ? a != b : uOp == 2 ? a < b : uOp == 3 ? a <= b
           : uOp == 4 ? a > b : a >= b;
    o = vec4(t ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}"#;

const FS_WHERE: &str = r#"uniform highp sampler2D C; uniform highp sampler2D A; uniform highp sampler2D B;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float c = texelFetch(C, p, 0).r;
    o = vec4(c != 0.0 ? texelFetch(A, p, 0).r : texelFetch(B, p, 0).r, 0.0, 0.0, 1.0);
}"#;

// Rows of `A` and of the output run over all batch blocks (`batch·m`); block
// `p.y / uM` reads its own `k` rows of `B` when `uBBatched`, else the shared ones.
// `uNatural`: every slot has its logical `rows × last-dim` layout, so texels
// are elements; otherwise (a slot widened past the texture limit) the output
// texel is a linear index and the operands are read by linear index.
const FS_MATMUL: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B; uniform int uK;
uniform int uM; uniform int uN; uniform int uBBatched;
uniform int uNatural; uniform int uACols; uniform int uBCols; uniform int uOCols; uniform int uTotal;
out vec4 o;
// Element `i` of a slot laid out `cols` wide (any layout, see `widen_tall_slots`).
float lin(highp sampler2D T, int i, int cols) { return texelFetch(T, ivec2(i % cols, i / cols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy); // x=col, y=row
    float s = 0.0;
    if (uNatural != 0) {
        int bRow0 = uBBatched != 0 ? (p.y / uM) * uK : 0;
        for (int l = 0; l < uK; l++) {
            s += texelFetch(A, ivec2(l, p.y), 0).r * texelFetch(B, ivec2(p.x, bRow0 + l), 0).r;
        }
    } else {
        int e = p.y * uOCols + p.x;
        if (e >= uTotal) { o = vec4(0.0); return; }
        int row = e / uN, col = e % uN;
        int bRow0 = uBBatched != 0 ? (row / uM) * uK : 0;
        for (int l = 0; l < uK; l++) {
            s += lin(A, row * uK + l, uACols) * lin(B, (bRow0 + l) * uN + col, uBCols);
        }
    }
    o = vec4(s, 0.0, 0.0, 1.0);
}"#;

const FS_GATHER: &str = r#"uniform highp sampler2D SRC; uniform highp isampler2D IDX;
uniform int uSrcCols;
out vec4 o;
void main() {
    int srcLin = texelFetch(IDX, ivec2(gl_FragCoord.xy), 0).r;
    // Negative index (PAD sentinel u32::MAX → -1) reads as 0 (conv/im2col pad).
    float v = srcLin < 0 ? 0.0 : texelFetch(SRC, ivec2(srcLin % uSrcCols, srcLin / uSrcCols), 0).r;
    o = vec4(v, 0.0, 0.0, 1.0);
}"#;

const FS_REDUCE: &str = r#"uniform highp sampler2D SRC; uniform highp isampler2D GROUPS;
uniform int uFanin; uniform int uSrcCols; uniform int uOutCols; uniform int uGCols; uniform int uOp;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int oi = p.y * uOutCols + p.x;
    float acc = uOp == 2 ? -3.0e38 : uOp == 3 ? 3.0e38 : uOp == 4 ? 1.0 : 0.0;
    float count = 0.0;
    for (int j = 0; j < uFanin; j++) {
        int gl = oi * uFanin + j;
        int g = texelFetch(GROUPS, ivec2(gl % uGCols, gl / uGCols), 0).r;
        if (g >= 0) {
            float v = texelFetch(SRC, ivec2(g % uSrcCols, g / uSrcCols), 0).r;
            count += 1.0;
            if (uOp == 0 || uOp == 1) acc += v;
            else if (uOp == 2) acc = max(acc, v);
            else if (uOp == 3) acc = min(acc, v);
            else acc *= v;
        }
    }
    float res = (uOp == 1 && count > 0.0) ? acc / count : acc;
    o = vec4(res, 0.0, 0.0, 1.0);
}"#;

// `uSlotCols`: the texture width of `A` and the output (they share a layout);
// equal to `uCols` unless the slots were widened, when texels are read by
// linear index.
const FS_SOFTMAX: &str = r#"uniform highp sampler2D A; uniform int uCols; uniform int uSlotCols; uniform int uTotal;
out vec4 o;
// Element `i` of a slot laid out `cols` wide (any layout, see `widen_tall_slots`).
float lin(highp sampler2D T, int i, int cols) { return texelFetch(T, ivec2(i % cols, i / cols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int e = p.y * uSlotCols + p.x;
    if (e >= uTotal) { o = vec4(0.0); return; }
    int r0 = (e / uCols) * uCols;
    float m = -3.0e38;
    for (int c = 0; c < uCols; c++) m = max(m, lin(A, r0 + c, uSlotCols));
    float sum = 0.0;
    for (int c = 0; c < uCols; c++) sum += exp(lin(A, r0 + c, uSlotCols) - m);
    o = vec4(exp(texelFetch(A, p, 0).r - m) / sum, 0.0, 0.0, 1.0);
}"#;

// `uSlotCols` / `uGCols`: texture widths of `X` (and the output) and of
// `G` / `B`; with natural layouts `uCols` and `uCols`.
const FS_LAYERNORM: &str = r#"uniform highp sampler2D X; uniform highp sampler2D G; uniform highp sampler2D B;
uniform int uCols; uniform float uEps; uniform int uSlotCols; uniform int uGCols; uniform int uTotal;
out vec4 o;
// Element `i` of a slot laid out `cols` wide (any layout, see `widen_tall_slots`).
float lin(highp sampler2D T, int i, int cols) { return texelFetch(T, ivec2(i % cols, i / cols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int e = p.y * uSlotCols + p.x;
    if (e >= uTotal) { o = vec4(0.0); return; }
    int r0 = (e / uCols) * uCols, c = e % uCols;
    float mean = 0.0;
    for (int i = 0; i < uCols; i++) mean += lin(X, r0 + i, uSlotCols);
    mean /= float(uCols);
    float var = 0.0;
    for (int i = 0; i < uCols; i++) {
        float d = lin(X, r0 + i, uSlotCols) - mean;
        var += d * d;
    }
    var /= float(uCols);
    float norm = (texelFetch(X, p, 0).r - mean) * inversesqrt(var + uEps);
    float g = lin(G, c, uGCols);
    float b = lin(B, c, uGCols);
    o = vec4(norm * g + b, 0.0, 0.0, 1.0);
}"#;

// Layout uniforms as in `FS_LAYERNORM`.
const FS_RMSNORM: &str = r#"uniform highp sampler2D X; uniform highp sampler2D G; uniform highp sampler2D B;
uniform int uCols; uniform float uEps; uniform int uSlotCols; uniform int uGCols; uniform int uTotal;
out vec4 o;
// Element `i` of a slot laid out `cols` wide (any layout, see `widen_tall_slots`).
float lin(highp sampler2D T, int i, int cols) { return texelFetch(T, ivec2(i % cols, i / cols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int e = p.y * uSlotCols + p.x;
    if (e >= uTotal) { o = vec4(0.0); return; }
    int r0 = (e / uCols) * uCols, c = e % uCols;
    float n_inv = 1.0 / float(uCols);
    // Two-pass: mean(x²) = mean((x-mean)^2) + mean^2. Matches the CPU oracle.
    float sum = 0.0;
    for (int i = 0; i < uCols; i++) { sum += lin(X, r0 + i, uSlotCols); }
    float mean = sum * n_inv;
    float dev = 0.0;
    for (int i = 0; i < uCols; i++) {
        float d = lin(X, r0 + i, uSlotCols) - mean;
        dev += d * d;
    }
    float norm = texelFetch(X, p, 0).r * inversesqrt(dev * n_inv + mean * mean + uEps);
    float g = lin(G, c, uGCols);
    float b = lin(B, c, uGCols);
    o = vec4(norm * g + b, 0.0, 0.0, 1.0);
}"#;

const FS_ARGREDUCE: &str = r#"uniform highp sampler2D SRC; uniform highp isampler2D GROUPS;
uniform int uFanin; uniform int uSrcCols; uniform int uOutCols; uniform int uGCols; uniform int uIsMax;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int oi = p.y * uOutCols + p.x;
    float best = uIsMax == 1 ? -3.0e38 : 3.0e38;
    int bj = 0;
    for (int j = 0; j < uFanin; j++) {
        int gl = oi * uFanin + j;
        int g = texelFetch(GROUPS, ivec2(gl % uGCols, gl / uGCols), 0).r;
        float v = texelFetch(SRC, ivec2(g % uSrcCols, g / uSrcCols), 0).r;
        if ((uIsMax == 1 && v > best) || (uIsMax == 0 && v < best)) { best = v; bj = j; }
    }
    o = vec4(float(bj), 0.0, 0.0, 1.0);
}"#;

// Standalone complex `Op::Cast` lane-move — one OUTPUT lane per fragment.
// `lin` is the flat output-lane index; `uSrcCols`/`uOutCols` unflatten to the
// (lane-aware) 2D texture footprints. Mirrors `exec_cpu` ComplexCast + the
// wgpu/cuda/vulkan `complex_cast` 6-mode table. C64 = 2 lanes, C128 = 4 lanes.
const FS_COMPLEX_CAST: &str = r#"uniform highp sampler2D SRC;
uniform int uMode; uniform int uSrcCols; uniform int uOutCols;
out vec4 o;
float src(int lin) { return texelFetch(SRC, ivec2(lin % uSrcCols, lin / uSrcCols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    float v = 0.0;
    if (uMode == 0) {            // real → C64
        int k = lin / 2; int j = lin - 2 * k;
        v = (j == 0) ? src(k) : 0.0;
    } else if (uMode == 1) {     // C64 → real
        v = src(2 * lin);
    } else if (uMode == 2) {     // real → C128
        int k = lin / 4; int j = lin - 4 * k;
        v = (j == 0) ? src(k) : 0.0;
    } else if (uMode == 3) {     // C128 → real
        v = src(4 * lin);
    } else if (uMode == 4) {     // C64 → C128
        int k = lin / 4; int j = lin - 4 * k;
        if (j == 0) v = src(2 * k);
        else if (j == 2) v = src(2 * k + 1);
        else v = 0.0;
    } else {                     // uMode == 5: C128 → C64
        int k = lin / 2; int j = lin - 2 * k;
        v = (j == 0) ? src(4 * k) : src(4 * k + 2);
    }
    o = vec4(v, 0.0, 0.0, 1.0);
}"#;

// C64 element-wise binary — one OUTPUT lane per fragment. Each fragment reads
// BOTH `[re, im]` lanes of its operands (the reason this can't ride FS_BINARY's
// scalar-per-lane model). Broadcast is per-operand complex-element modulo
// (`k % n_a`, `k % n_b`), matching rlx-cpu. uOp reuses the Bin codes; only
// Add(0)/Sub(1)/Mul(2)/Div(3) reach here (max/min/pow rejected at lowering).
const FS_BINARY_C64: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B;
uniform int uOp; uniform int uNa; uniform int uNb;
uniform int uACols; uniform int uBCols; uniform int uOutCols;
out vec4 o;
float af(int lin) { return texelFetch(A, ivec2(lin % uACols, lin / uACols), 0).r; }
float bf(int lin) { return texelFetch(B, ivec2(lin % uBCols, lin / uBCols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    int k = lin / 2; int j = lin - 2 * k;
    int ka = k % uNa;
    int kb = k % uNb;
    float ar = af(2 * ka); float ai = af(2 * ka + 1);
    float br = bf(2 * kb); float bi = bf(2 * kb + 1);
    float cr = 0.0; float ci = 0.0;
    if (uOp == 0) { cr = ar + br; ci = ai + bi; }
    else if (uOp == 1) { cr = ar - br; ci = ai - bi; }
    else if (uOp == 2) { cr = ar * br - ai * bi; ci = ar * bi + ai * br; }
    else { float d = br * br + bi * bi; cr = (ar * br + ai * bi) / d; ci = (ai * br - ar * bi) / d; }
    o = vec4(j == 0 ? cr : ci, 0.0, 0.0, 1.0);
}"#;

const FS_GATHER_RT: &str = r#"uniform highp sampler2D TABLE; uniform highp sampler2D IDXT;
uniform highp isampler2D WHICH; uniform highp isampler2D BASE;
uniform int uTableCols; uniform int uIdxCols; uniform int uAxisStride; uniform int uWrap;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int wi = texelFetch(WHICH, p, 0).r;
    float idxf = texelFetch(IDXT, ivec2(wi % uIdxCols, wi / uIdxCols), 0).r;
    int ix = int(floor(idxf + 0.5));
    if (uWrap > 0) {
        if (ix < 0) ix += uWrap;
        ix = clamp(ix, 0, uWrap - 1);
    }
    int tlin = texelFetch(BASE, p, 0).r + ix * uAxisStride;
    o = vec4(texelFetch(TABLE, ivec2(tlin % uTableCols, tlin / uTableCols), 0).r, 0.0, 0.0, 1.0);
}"#;

// Mirrors `exec_cpu` ScatterRuntime: each output folds in, in `j` order, the
// updates whose (rounded, negative `+ uWrap`) index equals its key.
const FS_SCATTER_RT: &str = r#"uniform highp sampler2D DATA; uniform highp sampler2D UPD;
uniform highp sampler2D IDXT; uniform highp isampler2D IBASE; uniform highp isampler2D UBASE;
uniform highp isampler2D KEY;
uniform int uHasData; uniform int uCount; uniform int uIStride; uniform int uUStride;
uniform int uWrap; uniform int uRed; uniform int uUpdCols; uniform int uIdxCols;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float acc = uHasData != 0 ? texelFetch(DATA, p, 0).r : 0.0;
    int ib = texelFetch(IBASE, p, 0).r;
    int ub = texelFetch(UBASE, p, 0).r;
    int key = texelFetch(KEY, p, 0).r;
    if (ib >= 0) {
        for (int j = 0; j < uCount; j++) {
            int il = ib + j * uIStride;
            int ix = int(floor(texelFetch(IDXT, ivec2(il % uIdxCols, il / uIdxCols), 0).r + 0.5));
            if (ix < 0) ix += uWrap;
            if (ix == key) {
                int ul = ub + j * uUStride;
                float u = texelFetch(UPD, ivec2(ul % uUpdCols, ul / uUpdCols), 0).r;
                acc = uRed == 0 ? u : uRed == 1 ? acc + u : uRed == 2 ? acc * u
                    : uRed == 3 ? max(acc, u) : min(acc, u);
            }
        }
    }
    o = vec4(acc, 0.0, 0.0, 1.0);
}"#;

// Mirrors `exec_cpu` SortAxis: the output at rank `pos` is the element that
// exactly `pos` others sort before (ties: smaller index first).
const FS_SORT: &str = r#"uniform highp sampler2D SRC;
uniform int uLen; uniform int uInner; uniform int uK; uniform int uDesc; uniform int uIdx;
uniform int uSrcCols; uniform int uOutCols;
out vec4 o;
float at(int row, int j, int ii) {
    int l = (row * uLen + j) * uInner + ii;
    return texelFetch(SRC, ivec2(l % uSrcCols, l / uSrcCols), 0).r;
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int i = p.y * uOutCols + p.x;
    int row = i / (uK * uInner); int pos = (i / uInner) % uK; int ii = i % uInner;
    float res = 0.0;
    for (int j = 0; j < uLen; j++) {
        float v = at(row, j, ii);
        int rank = 0;
        for (int q = 0; q < uLen; q++) {
            float a = at(row, q, ii);
            if ((uDesc != 0 ? a > v : a < v) || (a == v && q < j)) rank++;
        }
        if (rank == pos) { res = uIdx != 0 ? float(j) : v; break; }
    }
    o = vec4(res, 0.0, 0.0, 1.0);
}"#;

// Mirrors `exec_cpu::fft_stage_lane`: one radix-2 Stockham butterfly per
// output lane, twiddles from the host-computed table.
const FS_FFT_STAGE: &str = r#"uniform highp sampler2D SRC; uniform highp sampler2D TW;
uniform int uN; uniform int uNs; uniform int uInBlock; uniform int uOutBlock;
uniform int uSrcCols; uniform int uTwCols; uniform int uOutCols; uniform float uScale;
out vec4 o;
float src(int row, int e, int c) {
    int l = row * 2 * uN + (uInBlock != 0 ? c * uN + e : 2 * e + c);
    return texelFetch(SRC, ivec2(l % uSrcCols, l / uSrcCols), 0).r;
}
float tw(int l) { return texelFetch(TW, ivec2(l % uTwCols, l / uTwCols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int i = p.y * uOutCols + p.x;
    int row = i / (2 * uN); int l = i % (2 * uN);
    int k = uOutBlock != 0 ? l % uN : l / 2;
    int comp = uOutBlock != 0 ? l / uN : l % 2;
    int jm = k % uNs;
    int j = (k / (2 * uNs)) * uNs + jm;
    float ar = src(row, j, 0); float ai = src(row, j, 1);
    float br = src(row, j + uN / 2, 0); float bi = src(row, j + uN / 2, 1);
    float wr = tw(2 * (uNs - 1 + jm)); float wi = tw(2 * (uNs - 1 + jm) + 1);
    float tr = br * wr - bi * wi; float ti = br * wi + bi * wr;
    float s = (k % (2 * uNs)) < uNs ? 1.0 : -1.0;
    o = vec4((comp == 0 ? ar + s * tr : ai + s * ti) * uScale, 0.0, 0.0, 1.0);
}"#;

fn scatter_red_code(r: ScatterRed) -> i32 {
    match r {
        ScatterRed::Replace => 0,
        ScatterRed::Add => 1,
        ScatterRed::Mul => 2,
        ScatterRed::Max => 3,
        ScatterRed::Min => 4,
    }
}

fn act_code(a: Act) -> i32 {
    match a {
        Act::Relu => 0,
        Act::Neg => 1,
        Act::Exp => 2,
        Act::Log => 3,
        Act::Sqrt => 4,
        Act::Rsqrt => 5,
        Act::Sigmoid => 6,
        Act::Tanh => 7,
        Act::Abs => 8,
        Act::Sin => 9,
        Act::Cos => 10,
        Act::Silu => 11,
        Act::Recip => 12,
        Act::Gelu => 13,
        Act::GeluApprox => 14,
        Act::Softplus => 15,
        Act::Tan => 16,
        Act::Atan => 17,
        Act::Round => 18,
        Act::Floor => 19,
        Act::Ceil => 20,
        Act::Sign => 21,
        Act::Elu => 22,
        Act::Erf => 23,
        Act::HardSwish => 24,
        Act::HardSigmoid => 25,
        Act::Mish => 26,
        Act::Softsign => 27,
        Act::LogSigmoid => 28,
    }
}

fn bin_code(b: Bin) -> i32 {
    match b {
        Bin::Add => 0,
        Bin::Sub => 1,
        Bin::Mul => 2,
        Bin::Div => 3,
        Bin::Max => 4,
        Bin::Min => 5,
        Bin::Pow => 6,
        Bin::Mod => 7,
        Bin::BitAnd => 8,
        Bin::BitOr => 9,
        Bin::BitXor => 10,
        Bin::Shl => 11,
        Bin::Shr => 12,
        Bin::Atan2 => 13,
    }
}

// Linear addressing: a slot's element `i` is texel `(i % cols, i / cols)`,
// whatever its texture shape (widened or not). `uTotal` bounds the output:
// a widened slot's last row is padding.

// Direct NCDHW conv3d, one output element per fragment; must match
// `exec_cpu`'s `Step::Conv3d` (same summation order).
const FS_CONV3D: &str = r#"uniform highp sampler2D X; uniform highp sampler2D W; uniform highp sampler2D BIAS; uniform int uAct;
uniform int uXCols; uniform int uWCols; uniform int uBCols; uniform int uOutCols; uniform int uTotal;
uniform int uHasBias; uniform int uCin; uniform int uCout; uniform int uGroups;
uniform ivec3 uIn; uniform ivec3 uOut; uniform ivec3 uK; uniform ivec3 uS; uniform ivec3 uP; uniform ivec3 uD;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int ow = lin % uOut.z; int t = lin / uOut.z;
    int oh = t % uOut.y; t /= uOut.y;
    int od = t % uOut.x; t /= uOut.x;
    int co = t % uCout; int n = t / uCout;
    int cinG = uCin / uGroups; int g = co / (uCout / uGroups);
    int kVol = uK.x * uK.y * uK.z; int inHW = uIn.y * uIn.z; int inVol = uIn.x * inHW;
    float s = uHasBias != 0 ? texelFetch(BIAS, ivec2(co % uBCols, co / uBCols), 0).r : 0.0;
    for (int ci = 0; ci < cinG; ci++) {
        int xb = (n * uCin + g * cinG + ci) * inVol;
        int wb = (co * cinG + ci) * kVol;
        for (int kd = 0; kd < uK.x; kd++) {
            int id = od * uS.x + kd * uD.x - uP.x;
            if (id < 0 || id >= uIn.x) continue;
            for (int kh = 0; kh < uK.y; kh++) {
                int ih = oh * uS.y + kh * uD.y - uP.y;
                if (ih < 0 || ih >= uIn.y) continue;
                // A kernel row is a run along W in both tensors: place its
                // first tap with one divide each, then step texel by texel
                // (a divide per tap cost more than the multiply-add).
                // (from its first tap inside the volume: `%` of a negative
                // index is undefined in GLSL)
                int iw0 = ow * uS.z - uP.z;
                int kw0 = iw0 < 0 ? (uD.z - 1 - iw0) / uD.z : 0;
                int xi = xb + id * inHW + ih * uIn.z + iw0 + kw0 * uD.z;
                ivec2 xt = ivec2(xi % uXCols, xi / uXCols);
                int wi = wb + (kd * uK.y + kh) * uK.z + kw0;
                ivec2 wt = ivec2(wi % uWCols, wi / uWCols);
                for (int kw = kw0; kw < uK.z; kw++) {
                    int iw = iw0 + kw * uD.z;
                    if (iw >= 0 && iw < uIn.z) {
                        s += texelFetch(X, xt, 0).r * texelFetch(W, wt, 0).r;
                    }
                    xt.x += uD.z;
                    while (xt.x >= uXCols) { xt.x -= uXCols; xt.y++; }
                    wt.x++;
                    if (wt.x == uWCols) { wt.x = 0; wt.y++; }
                }
            }
        }
    }
    o = vec4(uAct < 0 ? s : actF(uAct, s), 0.0, 0.0, 1.0);
}"#;

// Conv weights [cout, tap] regrouped for [`fs_conv3d_quads`] with `uT`
// targets: texel `(pass·tap + j)·uT + q` holds tap `j` of quad
// `g = pass·uT + q`, output channels 4g .. 4g+3 (zero past `uCout`) — a
// pass's quads side by side at each tap.
const FS_PACK_W4: &str = r#"uniform highp sampler2D W;
uniform int uWCols; uniform int uOutCols; uniform int uTotal; uniform int uTap; uniform int uCout;
uniform int uT;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int rest = lin / uT;
    int g = (rest / uTap) * uT + lin % uT; int j = rest % uTap;
    vec4 v = vec4(0.0);
    for (int c = 0; c < 4; c++) {
        int co = 4 * g + c;
        if (co < uCout) {
            int i = co * uTap + j;
            v[c] = texelFetch(W, ivec2(i % uWCols, i / uWCols), 0).r;
        }
    }
    o = v;
}"#;

/// Ungrouped conv3d, `4·t` output channels per fragment (`t` ≤ 4 RGBA
/// targets): one batch item's channels `uCo0 .. uCo0 + 4t` over its output
/// volume, each input tap fetched once for all of them and their weights `t`
/// neighbouring texels of W4 (`FS_PACK_W4`). [`fs_unpack_quads`] copies the result into the R32F output. Each
/// channel sums in `FS_CONV3D`'s order.
fn fs_conv3d_quads(t: usize) -> String {
    let outs: String = (0..t)
        .map(|j| format!("layout(location = {j}) out vec4 o{j};\n"))
        .collect();
    let stores: String = (0..t).map(|j| format!("    o{j} = s[{j}];\n")).collect();
    let zero: String = (0..t).map(|j| format!("o{j} = vec4(0.0); ")).collect();
    format!(
        r#"uniform highp sampler2D X; uniform highp sampler2D W4; uniform highp sampler2D BIAS;
uniform int uXCols; uniform int uWCols; uniform int uBCols; uniform int uOutCols; uniform int uTotal;
uniform int uHasBias; uniform int uCin; uniform int uCout; uniform int uN; uniform int uCo0;
uniform ivec3 uIn; uniform ivec3 uOut; uniform ivec3 uK; uniform ivec3 uS; uniform ivec3 uP; uniform ivec3 uD;
{outs}#define T {t}
void main() {{
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) {{ {zero}return; }}
    int ow = lin % uOut.z; int r = lin / uOut.z;
    int oh = r % uOut.y; int od = r / uOut.y;
    int kVol = uK.x * uK.y * uK.z; int inHW = uIn.y * uIn.z; int inVol = uIn.x * inHW;
    vec4 s[T];
    for (int j = 0; j < T; j++) {{
        s[j] = vec4(0.0);
        if (uHasBias != 0) {{
            for (int c = 0; c < 4; c++) {{
                int co = min(uCo0 + 4 * j + c, uCout - 1);
                s[j][c] = texelFetch(BIAS, ivec2(co % uBCols, co / uBCols), 0).r;
            }}
        }}
    }}
    for (int ci = 0; ci < uCin; ci++) {{
        int xb = (uN * uCin + ci) * inVol;
        int wb = ((uCo0 / (4 * T)) * uCin + ci) * kVol * T;
        for (int kd = 0; kd < uK.x; kd++) {{
            int id = od * uS.x + kd * uD.x - uP.x;
            if (id < 0 || id >= uIn.x) continue;
            for (int kh = 0; kh < uK.y; kh++) {{
                int ih = oh * uS.y + kh * uD.y - uP.y;
                if (ih < 0 || ih >= uIn.y) continue;
                // A kernel row is a run along W in both tensors: place its
                // first tap with one divide each, then step texel by texel
                // (from its first tap inside the volume: `%` of a negative
                // index is undefined in GLSL)
                int iw0 = ow * uS.z - uP.z;
                int kw0 = iw0 < 0 ? (uD.z - 1 - iw0) / uD.z : 0;
                int xi = xb + id * inHW + ih * uIn.z + iw0 + kw0 * uD.z;
                ivec2 xt = ivec2(xi % uXCols, xi / uXCols);
                int wi = wb + ((kd * uK.y + kh) * uK.z + kw0) * T;
                ivec2 wt = ivec2(wi % uWCols, wi / uWCols);
                for (int kw = kw0; kw < uK.z; kw++) {{
                    int iw = iw0 + kw * uD.z;
                    if (iw >= 0 && iw < uIn.z) {{
                        float xv = texelFetch(X, xt, 0).r;
                        for (int j = 0; j < T; j++) s[j] += xv * texelFetch(W4, ivec2(wt.x + j, wt.y), 0);
                    }}
                    xt.x += uD.z;
                    while (xt.x >= uXCols) {{ xt.x -= uXCols; xt.y++; }}
                    // (W4's width is a multiple of T: a tap's quads share a row)
                    wt.x += T;
                    if (wt.x == uWCols) {{ wt.x = 0; wt.y++; }}
                }}
            }}
        }}
    }}
{stores}}}"#
    )
}

/// One [`fs_conv3d_quads`] result into its R32F output: elements `uLo ..
/// uHi` are channel-major runs of `uSp`; everything else is left as drawn
/// before.
fn fs_unpack_quads(t: usize) -> String {
    let samplers: String = (0..t)
        .map(|j| format!("uniform highp sampler2D T{j};\n"))
        .collect();
    let pick: String = (0..t)
        .map(|j| format!("    if (c / 4 == {j}) v = texelFetch(T{j}, at, 0);\n"))
        .collect();
    format!(
        r#"{samplers}uniform int uTCols; uniform int uOutCols; uniform int uLo; uniform int uHi; uniform int uSp; uniform int uAct;
out vec4 o;
void main() {{
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin < uLo || lin >= uHi) discard;
    int r = lin - uLo; int q = r % uSp; int c = r / uSp;
    ivec2 at = ivec2(q % uTCols, q / uTCols);
    vec4 v = vec4(0.0);
{pick}    float y = v[c & 3];
    o = vec4(uAct < 0 ? y : actF(uAct, y), 0.0, 0.0, 1.0);
}}"#
    )
}

// ArgMax / ArgMin along the middle of [outer, len, inner]: first extreme wins.
const FS_ARGAXIS: &str = r#"uniform highp sampler2D SRC;
uniform int uSrcCols; uniform int uOutCols; uniform int uTotal; uniform int uLen; uniform int uInner; uniform int uIsMax;
out vec4 o;
float src(int i) { return texelFetch(SRC, ivec2(i % uSrcCols, i / uSrcCols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int base = (lin / uInner) * uLen * uInner + lin % uInner;
    float best = src(base);
    int at = 0;
    for (int j = 1; j < uLen; j++) {
        float v = src(base + j * uInner);
        if (uIsMax != 0 ? v > best : v < best) { best = v; at = j; }
    }
    o = vec4(float(at), 0.0, 0.0, 1.0);
}"#;

// 1×1×1 conv + argmax over its channels, the scores never stored.
const FS_CONV1_ARGMAX: &str = r#"uniform highp sampler2D X; uniform highp sampler2D W; uniform highp sampler2D BIAS;
uniform int uXCols; uniform int uWCols; uniform int uBCols; uniform int uOutCols; uniform int uTotal;
uniform int uHasBias; uniform int uCin; uniform int uCout; uniform int uSpatial;
out vec4 o;
float fx(int i) { return texelFetch(X, ivec2(i % uXCols, i / uXCols), 0).r; }
float fw(int i) { return texelFetch(W, ivec2(i % uWCols, i / uWCols), 0).r; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int n = lin / uSpatial; int q = lin % uSpatial;
    float best = 0.0; int at = 0;
    for (int co = 0; co < uCout; co++) {
        float s = uHasBias != 0 ? texelFetch(BIAS, ivec2(co % uBCols, co / uBCols), 0).r : 0.0;
        for (int ci = 0; ci < uCin; ci++) s += fx((n * uCin + ci) * uSpatial + q) * fw(co * uCin + ci);
        if (co == 0 || s > best) { best = s; at = co; }
    }
    o = vec4(float(at), 0.0, 0.0, 1.0);
}"#;

// GroupNorm, a level of the sum tree: one fragment per (group, chunk) sums
// its chunk of the level below — of x (uMode 1 off) or of (x − mean)² (uMode
// 1, the first level of the variance).
const FS_GN_PARTIAL: &str = r#"uniform highp sampler2D X; uniform highp sampler2D MEAN;
uniform int uXCols; uniform int uOutCols; uniform int uTotal; uniform int uRun; uniform int uChunk;
uniform int uNChunks; uniform int uMode;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int ng = lin / uNChunks;
    int start = ng * uRun + (lin % uNChunks) * uChunk;
    int end = min(start + uChunk, (ng + 1) * uRun);
    float m = uMode == 1 ? texelFetch(MEAN, ivec2(ng, 0), 0).r : 0.0;
    float s = 0.0;
    ivec2 xt = ivec2(start % uXCols, start / uXCols);
    for (int i = start; i < end; i++) {
        float v = texelFetch(X, xt, 0).r - m;
        s += uMode == 1 ? v * v : v;
        xt.x++;
        if (xt.x == uXCols) { xt.x = 0; xt.y++; }
    }
    o = vec4(s, 0.0, 0.0, 1.0);
}"#;

// GroupNorm, pass 2 and 4: a group's chunk sums over its element count.
const FS_GN_STAT: &str = r#"uniform highp sampler2D P;
uniform int uNChunks; uniform int uRun;
out vec4 o;
void main() {
    int ng = int(gl_FragCoord.x);
    float s = 0.0;
    for (int c = 0; c < uNChunks; c++) s += texelFetch(P, ivec2(ng * uNChunks + c, 0), 0).r;
    o = vec4(s / float(uRun), 0.0, 0.0, 1.0);
}"#;

// GroupNorm, pass 5: (x − mean)/√(var + eps) · gamma[c] + beta[c].
const FS_GN_APPLY: &str = r#"uniform highp sampler2D X; uniform highp sampler2D MEAN; uniform highp sampler2D VAR;
uniform highp sampler2D G; uniform highp sampler2D B;
uniform int uXCols; uniform int uOutCols; uniform int uTotal; uniform int uRun; uniform int uGroups;
uniform int uCpg; uniform int uSpatial; uniform float uEps; uniform int uAct;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int ng = lin / uRun;
    int ch = (ng % uGroups) * uCpg + (lin % uRun) / uSpatial;
    float x = texelFetch(X, ivec2(lin % uXCols, lin / uXCols), 0).r;
    float m = texelFetch(MEAN, ivec2(ng, 0), 0).r;
    float v = texelFetch(VAR, ivec2(ng, 0), 0).r;
    float g = texelFetch(G, ivec2(ch, 0), 0).r;
    float b = texelFetch(B, ivec2(ch, 0), 0).r;
    float y = (x - m) * inversesqrt(v + uEps) * g + b;
    o = vec4(uAct < 0 ? y : actF(uAct, y), 0.0, 0.0, 1.0);
}"#;

// Broadcast by arithmetic: unravel the output index, dot the input strides.
const FS_BROADCAST: &str = r#"uniform highp sampler2D SRC;
uniform int uSrcCols; uniform int uOutCols; uniform int uTotal; uniform int uRank;
uniform int uDims[8]; uniform int uStr[8];
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int rest = lin; int at = 0;
    for (int k = 7; k >= 0; k--) {
        if (k >= uRank) continue;
        at += (rest % uDims[k]) * uStr[k];
        rest /= uDims[k];
    }
    o = vec4(texelFetch(SRC, ivec2(at % uSrcCols, at / uSrcCols), 0).r, 0.0, 0.0, 1.0);
}"#;

// One value everywhere: a broadcast scalar.
const FS_SPLAT: &str = r#"uniform highp sampler2D SRC;
out vec4 o;
void main() { o = vec4(texelFetch(SRC, ivec2(0, 0), 0).r, 0.0, 0.0, 1.0); }"#;

// Element for element, between two texture shapes.
const FS_COPY: &str = r#"uniform highp sampler2D SRC;
uniform int uSrcCols; uniform int uOutCols; uniform int uTotal;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    o = vec4(lin < uTotal ? texelFetch(SRC, ivec2(lin % uSrcCols, lin / uSrcCols), 0).r : 0.0, 0.0, 0.0, 1.0);
}"#;

// Two tensors as [outer, len] rows, joined row by row.
const FS_CONCAT: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B;
uniform int uACols; uniform int uBCols; uniform int uOutCols; uniform int uTotal; uniform int uALen; uniform int uBLen;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int row = lin / (uALen + uBLen); int j = lin % (uALen + uBLen);
    float v;
    if (j < uALen) { int i = row * uALen + j; v = texelFetch(A, ivec2(i % uACols, i / uACols), 0).r; }
    else { int i = row * uBLen + j - uALen; v = texelFetch(B, ivec2(i % uBCols, i / uBCols), 0).r; }
    o = vec4(v, 0.0, 0.0, 1.0);
}"#;

// NCDHW max / average pool; padding skipped by a max, zero in an average.
const FS_POOL3D: &str = r#"uniform highp sampler2D SRC;
uniform int uSrcCols; uniform int uOutCols; uniform int uTotal; uniform int uIsMax;
uniform ivec3 uIn; uniform ivec3 uOut; uniform ivec3 uK; uniform ivec3 uS; uniform ivec3 uP;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int ov = uOut.x * uOut.y * uOut.z; int iv = uIn.x * uIn.y * uIn.z;
    int c = lin / ov; int q = lin % ov;
    int od = q / (uOut.y * uOut.z); int oh = q / uOut.z % uOut.y; int ow = q % uOut.z;
    float acc = uIsMax != 0 ? -3.0e38 : 0.0;
    for (int kd = 0; kd < uK.x; kd++) {
        int id = od * uS.x + kd - uP.x;
        if (id < 0 || id >= uIn.x) continue;
        for (int kh = 0; kh < uK.y; kh++) {
            int ih = oh * uS.y + kh - uP.y;
            if (ih < 0 || ih >= uIn.y) continue;
            for (int kw = 0; kw < uK.z; kw++) {
                int iw = ow * uS.z + kw - uP.z;
                if (iw < 0 || iw >= uIn.z) continue;
                int i = c * iv + (id * uIn.y + ih) * uIn.z + iw;
                float v = texelFetch(SRC, ivec2(i % uSrcCols, i / uSrcCols), 0).r;
                acc = uIsMax != 0 ? max(acc, v) : acc + v;
            }
        }
    }
    o = vec4(uIsMax != 0 ? acc : acc / float(uK.x * uK.y * uK.z), 0.0, 0.0, 1.0);
}"#;

// NCDHW nearest resize: src = min(dst · in / out, in − 1) per axis.
const FS_RESIZE3D: &str = r#"uniform highp sampler2D SRC;
uniform int uSrcCols; uniform int uOutCols; uniform int uTotal; uniform ivec3 uIn; uniform ivec3 uOut;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int lin = p.y * uOutCols + p.x;
    if (lin >= uTotal) { o = vec4(0.0); return; }
    int ov = uOut.x * uOut.y * uOut.z; int iv = uIn.x * uIn.y * uIn.z;
    int c = lin / ov; int q = lin % ov;
    ivec3 at = ivec3(q / (uOut.y * uOut.z), q / uOut.z % uOut.y, q % uOut.z);
    ivec3 s = min(at * uIn / uOut, uIn - 1);
    int i = c * iv + (s.x * uIn.y + s.y) * uIn.z + s.z;
    o = vec4(texelFetch(SRC, ivec2(i % uSrcCols, i / uSrcCols), 0).r, 0.0, 0.0, 1.0);
}"#;

fn cmp_code(c: Cmp) -> i32 {
    match c {
        Cmp::Eq => 0,
        Cmp::Ne => 1,
        Cmp::Lt => 2,
        Cmp::Le => 3,
        Cmp::Gt => 4,
        Cmp::Ge => 5,
    }
}

fn red_code(r: Red) -> i32 {
    match r {
        Red::Sum => 0,
        Red::Mean => 1,
        Red::Max => 2,
        Red::Min => 3,
        Red::Prod => 4,
    }
}

/// A compiled WebGL2 backend (programs + framebuffer). Reusable across runs.
pub struct GlBackend {
    gl: GL,
    /// `MAX_TEXTURE_SIZE`: every tensor is one 2-D texture, so no tensor may
    /// have more rows (or columns) than this.
    max_texture: usize,
    /// Param / constant leaf textures by slot, kept across runs: weights are
    /// uploaded once, not on every call (an LM decode step used to re-upload
    /// the whole model). [`GlBackend::forget_param`] drops one.
    leaf_cache: std::cell::RefCell<HashMap<usize, WebGlTexture>>,
    /// Slot of each cached param, by name.
    param_slots: std::cell::RefCell<HashMap<String, usize>>,
    /// Index textures (gather / reduce groups) by `(step, which)` — fixed by
    /// the plan, so built once.
    idx_cache: std::cell::RefCell<HashMap<(usize, u8), WebGlTexture>>,
    /// Column count of each [`GlBackend::cached_linear_int`] texture.
    idx_cols: std::cell::RefCell<HashMap<(usize, u8), usize>>,
    /// Which plan the caches above hold ([`GlBackend::use_plan`]): one
    /// backend (one context, its programs compiled once) serves many.
    plan: std::cell::Cell<u64>,
    /// `getUniformLocation` answers by uniform name, then program: in
    /// Safari it is a synchronous trip to the GPU process, and it was asked
    /// for every uniform of every draw.
    uniforms:
        std::cell::RefCell<HashMap<String, Vec<(WebGlProgram, Option<WebGlUniformLocation>)>>>,
    /// Target sizes whose framebuffer was found complete (the check is a
    /// synchronous trip too, and was made every draw).
    complete: std::cell::RefCell<std::collections::HashSet<(usize, usize)>>,
    /// Whether a float target reads back as RED (one float a texel) rather
    /// than RGBA, asked once.
    read_red: std::cell::Cell<Option<bool>>,
    /// Runs submitted ([`GlBackend::submit`]) and not yet collected, by plan.
    pending: std::cell::RefCell<HashMap<u64, std::collections::VecDeque<Pending>>>,
    /// Float textures this backend made, with their `(w, h, format)`, and
    /// those free to hand out again: a model run in slabs asks for the same
    /// few sizes run after run, and made and dropped each time the browser
    /// allocated GPU memory every run.
    sized: std::cell::RefCell<Vec<(WebGlTexture, (usize, usize, u32))>>,
    free: std::cell::RefCell<Vec<(WebGlTexture, (usize, usize, u32))>>,
    /// Pixel-pack buffers free to hand out again, with their bytes.
    free_pbos: std::cell::RefCell<Vec<(web_sys::WebGlBuffer, usize)>>,
    /// The half-precision inputs of the run in progress ([`Self::submit_half`]).
    half_inputs: std::cell::RefCell<Vec<(String, Vec<u16>)>>,
    /// Inputs of the run in progress already on the GPU ([`resident`]).
    gpu_inputs: std::cell::RefCell<Vec<(String, WebGlTexture)>>,
    /// [`resident`]'s programs, compiled on first use.
    resident: std::cell::RefCell<resident::Programs>,
    fbo: web_sys::WebGlFramebuffer,
    unary: WebGlProgram,
    actback: WebGlProgram,
    binary: WebGlProgram,
    compare: WebGlProgram,
    where_: WebGlProgram,
    matmul: WebGlProgram,
    gather: WebGlProgram,
    reduce: WebGlProgram,
    softmax: WebGlProgram,
    layernorm: WebGlProgram,
    rmsnorm: WebGlProgram,
    argreduce: WebGlProgram,
    gather_rt: WebGlProgram,
    scatter_rt: WebGlProgram,
    sort: WebGlProgram,
    fft_stage: WebGlProgram,
    complex_cast: WebGlProgram,
    binary_c64: WebGlProgram,
    conv3d: WebGlProgram,
    /// [`fs_conv3d_quads`] / [`fs_unpack_quads`] for 1 ..= 4 targets
    conv3d_quads: Vec<WebGlProgram>,
    unpack_quads: Vec<WebGlProgram>,
    pack_w4: WebGlProgram,
    argaxis: WebGlProgram,
    conv1_argmax: WebGlProgram,
    gn_partial: WebGlProgram,
    gn_stat: WebGlProgram,
    gn_apply: WebGlProgram,
    copy: WebGlProgram,
    splat: WebGlProgram,
    broadcast: WebGlProgram,
    concat: WebGlProgram,
    pool3d: WebGlProgram,
    resize3d: WebGlProgram,
}

fn err<T: std::fmt::Debug>(ctx: &str, e: T) -> WebglError {
    WebglError(format!("{ctx}: {e:?}"))
}

#[path = "resident.rs"]
mod resident;
pub use resident::{Gather, PendingStats, ResidentStore};

/// A submitted run's outputs, in pixel-pack buffers.
struct Pending {
    reads: Vec<PendingRead>,
}

struct PendingRead {
    buf: web_sys::WebGlBuffer,
    bytes: usize,
    texels: usize,
    len: usize,
    red: bool,
}

/// Give the context back at once: a browser keeps only so many (Safari 16)
/// and drops the oldest past that, and a model run a layer at a time makes
/// one backend a layer.
impl Drop for GlBackend {
    fn drop(&mut self) {
        if let Ok(Some(ext)) = self.gl.get_extension("WEBGL_lose_context")
            && let Ok(f) = js_sys::Reflect::get(&ext, &"loseContext".into())
            && let Ok(f) = f.dyn_into::<js_sys::Function>()
        {
            let _ = f.call0(&ext);
        }
    }
}

impl GlBackend {
    /// Create an offscreen WebGL2 context and compile the op programs.
    pub fn new() -> Result<Self> {
        let canvas = web_sys::OffscreenCanvas::new(1, 1).map_err(|e| err("OffscreenCanvas", e))?;
        let gl = canvas
            .get_context("webgl2")
            .map_err(|e| err("get_context", e))?
            .ok_or_else(|| WebglError("no webgl2 context".into()))?
            .dyn_into::<GL>()
            .map_err(|_| WebglError("context is not WebGl2RenderingContext".into()))?;

        gl.get_extension("EXT_color_buffer_float")
            .map_err(|e| err("get_extension", e))?
            .ok_or_else(|| WebglError("EXT_color_buffer_float unavailable".into()))?;

        let max_texture = gl
            .get_parameter(GL::MAX_TEXTURE_SIZE)
            .ok()
            .and_then(|v| v.as_f64())
            .map_or(4096, |v| v as usize);

        // render targets a conv pass writes: the 4 WebGL2 guarantees. Eight
        // (Apple GPUs allow it; one pass for 30 channels) was no faster.
        let max_targets = gl
            .get_parameter(GL::MAX_DRAW_BUFFERS)
            .ok()
            .and_then(|v| v.as_f64())
            .map_or(4, |v| v as usize)
            .clamp(1, 4);

        let fbo = gl
            .create_framebuffer()
            .ok_or_else(|| WebglError("create_framebuffer".into()))?;

        // Activation shaders need the act helper functions; the rest don't.
        let plain = |body: &str| Self::program(&gl, &format!("{HEAD}{body}"));
        let with_act = |body: &str| Self::program(&gl, &format!("{HEAD}{ACT_FNS}{body}"));

        Ok(Self {
            max_texture,
            leaf_cache: Default::default(),
            param_slots: Default::default(),
            idx_cache: Default::default(),
            idx_cols: Default::default(),
            plan: Default::default(),
            uniforms: Default::default(),
            complete: Default::default(),
            read_red: Default::default(),
            pending: Default::default(),
            sized: Default::default(),
            free: Default::default(),
            free_pbos: Default::default(),
            half_inputs: Default::default(),
            gpu_inputs: Default::default(),
            resident: Default::default(),
            unary: with_act(FS_UNARY)?,
            actback: with_act(FS_ACTBACK)?,
            binary: plain(FS_BINARY)?,
            compare: plain(FS_COMPARE)?,
            where_: plain(FS_WHERE)?,
            matmul: plain(FS_MATMUL)?,
            gather: plain(FS_GATHER)?,
            reduce: plain(FS_REDUCE)?,
            softmax: plain(FS_SOFTMAX)?,
            layernorm: plain(FS_LAYERNORM)?,
            rmsnorm: plain(FS_RMSNORM)?,
            argreduce: plain(FS_ARGREDUCE)?,
            gather_rt: plain(FS_GATHER_RT)?,
            scatter_rt: plain(FS_SCATTER_RT)?,
            sort: plain(FS_SORT)?,
            fft_stage: plain(FS_FFT_STAGE)?,
            complex_cast: plain(FS_COMPLEX_CAST)?,
            binary_c64: plain(FS_BINARY_C64)?,
            conv3d: with_act(FS_CONV3D)?,
            conv3d_quads: (1..=max_targets)
                .map(|t| plain(&fs_conv3d_quads(t)))
                .collect::<Result<_>>()?,
            unpack_quads: (1..=max_targets)
                .map(|t| with_act(&fs_unpack_quads(t)))
                .collect::<Result<_>>()?,
            pack_w4: plain(FS_PACK_W4)?,
            argaxis: plain(FS_ARGAXIS)?,
            conv1_argmax: plain(FS_CONV1_ARGMAX)?,
            gn_partial: plain(FS_GN_PARTIAL)?,
            gn_stat: plain(FS_GN_STAT)?,
            gn_apply: with_act(FS_GN_APPLY)?,
            copy: plain(FS_COPY)?,
            splat: plain(FS_SPLAT)?,
            broadcast: plain(FS_BROADCAST)?,
            concat: plain(FS_CONCAT)?,
            pool3d: plain(FS_POOL3D)?,
            resize3d: plain(FS_RESIZE3D)?,
            fbo,
            gl,
        })
    }

    fn shader(gl: &GL, kind: u32, src: &str) -> Result<WebGlShader> {
        let sh = gl
            .create_shader(kind)
            .ok_or_else(|| WebglError("create_shader".into()))?;
        gl.shader_source(&sh, src);
        gl.compile_shader(&sh);
        if !gl
            .get_shader_parameter(&sh, GL::COMPILE_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl.get_shader_info_log(&sh).unwrap_or_default();
            return Err(WebglError(format!("shader compile: {log}")));
        }
        Ok(sh)
    }

    fn program(gl: &GL, fs: &str) -> Result<WebGlProgram> {
        let vs = Self::shader(gl, GL::VERTEX_SHADER, VERT)?;
        let fs = Self::shader(gl, GL::FRAGMENT_SHADER, fs)?;
        let p = gl
            .create_program()
            .ok_or_else(|| WebglError("create_program".into()))?;
        gl.attach_shader(&p, &vs);
        gl.attach_shader(&p, &fs);
        gl.link_program(&p);
        if !gl
            .get_program_parameter(&p, GL::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            let log = gl.get_program_info_log(&p).unwrap_or_default();
            return Err(WebglError(format!("program link: {log}")));
        }
        Ok(p)
    }

    fn uniform(&self, p: &WebGlProgram, name: &str) -> Option<WebGlUniformLocation> {
        let mut cache = self.uniforms.borrow_mut();
        let progs = cache.entry(name.to_string()).or_default();
        if let Some((_, loc)) = progs.iter().find(|(q, _)| q == p) {
            return loc.clone();
        }
        let loc = self.gl.get_uniform_location(p, name);
        progs.push((p.clone(), loc.clone()));
        loc
    }

    /// A tensor laid out as `h` rows of `w` must fit one texture.
    fn check_size(&self, w: usize, h: usize) -> Result<()> {
        if w > self.max_texture || h > self.max_texture {
            return Err(WebglError(format!(
                "tensor of {h} rows x {w} columns exceeds this browser's MAX_TEXTURE_SIZE \
                 {} — keep it off the WebGL graph (e.g. gather embeddings on the host)",
                self.max_texture
            )));
        }
        Ok(())
    }

    fn data_texture(&self, w: usize, h: usize, data: &[f32]) -> Result<WebGlTexture> {
        self.check_size(w, h)?;
        let gl = &self.gl;
        // Single-channel R32F: one float per texel. (RGBA32F with the value in
        // R spent 16 bytes per value, plus a 4x staging copy on upload — a
        // 230M-parameter model needed ~3.7 GB of textures.) Shaders read `.r`
        // either way. A short `data` (a slot smaller than its texture) is
        // zero-padded.
        let mut padded;
        let data = if data.len() < w * h {
            padded = data.to_vec();
            padded.resize(w * h, 0.0);
            &padded[..]
        } else {
            &data[..w * h]
        };
        // one of this size free: new contents, the same storage
        if let Some(tex) = self.acquire((w, h, GL::R32F)) {
            gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
            // SAFETY: the view is consumed by tex_sub_image_2d before any JS alloc.
            unsafe {
                let view = js_sys::Float32Array::view(data);
                gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                    GL::TEXTURE_2D,
                    0,
                    0,
                    0,
                    w as i32,
                    h as i32,
                    GL::RED,
                    GL::FLOAT,
                    Some(&view),
                )
                .map_err(|e| err("tex_sub_image_2d(r32f)", e))?;
            }
            return Ok(tex);
        }
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        // SAFETY: the view is consumed by tex_image_2d before any JS alloc.
        unsafe {
            let view = js_sys::Float32Array::view(data);
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                GL::TEXTURE_2D,
                0,
                GL::R32F as i32,
                w as i32,
                h as i32,
                0,
                GL::RED,
                GL::FLOAT,
                Some(&view),
            )
            .map_err(|e| err("tex_image_2d(r32f)", e))?;
        }
        self.set_nearest();
        self.sized
            .borrow_mut()
            .push((tex.clone(), (w, h, GL::R32F)));
        Ok(tex)
    }

    /// [`Self::data_texture`] from IEEE half bits: an R16F texture.
    fn half_texture(&self, w: usize, h: usize, data: &[u16]) -> Result<WebGlTexture> {
        self.check_size(w, h)?;
        let gl = &self.gl;
        let mut padded;
        let data = if data.len() < w * h {
            padded = data.to_vec();
            padded.resize(w * h, 0);
            &padded[..]
        } else {
            &data[..w * h]
        };
        let (tex, fresh) = match self.acquire((w, h, GL::R16F)) {
            Some(t) => (t, false),
            None => {
                let t = gl
                    .create_texture()
                    .ok_or_else(|| WebglError("create_texture".into()))?;
                self.sized.borrow_mut().push((t.clone(), (w, h, GL::R16F)));
                (t, true)
            }
        };
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        // SAFETY: the view is consumed by the upload before any JS alloc.
        unsafe {
            let view = js_sys::Uint16Array::view(data);
            if fresh {
                gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                    GL::TEXTURE_2D,
                    0,
                    GL::R16F as i32,
                    w as i32,
                    h as i32,
                    0,
                    GL::RED,
                    GL::HALF_FLOAT,
                    Some(&view),
                )
                .map_err(|e| err("tex_image_2d(r16f)", e))?;
                self.set_nearest();
            } else {
                gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                    GL::TEXTURE_2D,
                    0,
                    0,
                    0,
                    w as i32,
                    h as i32,
                    GL::RED,
                    GL::HALF_FLOAT,
                    Some(&view),
                )
                .map_err(|e| err("tex_sub_image_2d(r16f)", e))?;
            }
        }
        Ok(tex)
    }

    /// A free texture of `key` = `(w, h, internal format)`, if there is one.
    fn acquire(&self, key: (usize, usize, u32)) -> Option<WebGlTexture> {
        let mut free = self.free.borrow_mut();
        let i = free.iter().position(|(_, k)| *k == key)?;
        Some(free.swap_remove(i).0)
    }

    /// Done with `t`: free to hand out again if this backend sized it, else
    /// deleted. The free ones are capped in bytes, the oldest going first —
    /// GPU memory counts against a phone's tab, and textures of sizes no
    /// longer asked for would sit there for nothing.
    fn release(&self, t: WebGlTexture) {
        const KEEP_BYTES: usize = 96 << 20;
        let key = self
            .sized
            .borrow()
            .iter()
            .find(|(s, _)| *s == t)
            .map(|(_, k)| *k);
        let Some(k) = key else {
            self.gl.delete_texture(Some(&t));
            return;
        };
        let bytes = |k: &(usize, usize, u32)| {
            k.0 * k.1
                * match k.2 {
                    GL::RGBA32F => 16,
                    GL::R16F => 2,
                    _ => 4,
                }
        };
        let mut free = self.free.borrow_mut();
        free.push((t, k));
        let mut held: usize = free.iter().map(|(_, k)| bytes(k)).sum();
        while held > KEEP_BYTES && !free.is_empty() {
            let (old, k) = free.remove(0);
            held -= bytes(&k);
            self.sized.borrow_mut().retain(|(s, _)| *s != old);
            self.gl.delete_texture(Some(&old));
        }
    }

    /// Delete every free texture (another plan's sizes).
    fn drain_free(&self) {
        let old: Vec<_> = self.free.borrow_mut().drain(..).collect();
        let mut sized = self.sized.borrow_mut();
        for (t, _) in old {
            sized.retain(|(s, _)| *s != t);
            self.gl.delete_texture(Some(&t));
        }
    }

    fn int_texture(&self, w: usize, h: usize, data: &[i32]) -> Result<WebGlTexture> {
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        unsafe {
            let view = js_sys::Int32Array::view(data);
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                GL::TEXTURE_2D,
                0,
                GL::R32I as i32,
                w as i32,
                h as i32,
                0,
                GL::RED_INTEGER,
                GL::INT,
                Some(&view),
            )
            .map_err(|e| err("tex_image_2d(r32i)", e))?;
        }
        self.set_nearest();
        Ok(tex)
    }

    fn target_texture(&self, w: usize, h: usize) -> Result<WebGlTexture> {
        self.target_texture_as(w, h, GL::R32F, GL::RED)
    }

    /// A render target of `internal`/`format` texels (R32F or RGBA32F).
    fn target_texture_as(
        &self,
        w: usize,
        h: usize,
        internal: u32,
        format: u32,
    ) -> Result<WebGlTexture> {
        self.check_size(w, h)?;
        if let Some(tex) = self.acquire((w, h, internal)) {
            return Ok(tex);
        }
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        self.sized
            .borrow_mut()
            .push((tex.clone(), (w, h, internal)));
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            GL::TEXTURE_2D,
            0,
            // Renderable with EXT_color_buffer_float, which this backend
            // already requires for float targets.
            internal as i32,
            w as i32,
            h as i32,
            0,
            format,
            GL::FLOAT,
            None,
        )
        .map_err(|e| err("tex_image_2d(target)", e))?;
        self.set_nearest();
        Ok(tex)
    }

    fn set_nearest(&self) {
        let gl = &self.gl;
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MIN_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_MAG_FILTER, GL::NEAREST as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_S, GL::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(GL::TEXTURE_2D, GL::TEXTURE_WRAP_T, GL::CLAMP_TO_EDGE as i32);
    }

    fn bind_target(&self, tex: &WebGlTexture, w: usize, h: usize) -> Result<()> {
        let gl = &self.gl;
        gl.bind_framebuffer(GL::FRAMEBUFFER, Some(&self.fbo));
        gl.framebuffer_texture_2d(
            GL::FRAMEBUFFER,
            GL::COLOR_ATTACHMENT0,
            GL::TEXTURE_2D,
            Some(tex),
            0,
        );
        if !self.complete.borrow().contains(&(w, h)) {
            if gl.check_framebuffer_status(GL::FRAMEBUFFER) != GL::FRAMEBUFFER_COMPLETE {
                return Err(WebglError("framebuffer incomplete".into()));
            }
            self.complete.borrow_mut().insert((w, h));
        }
        gl.viewport(0, 0, w as i32, h as i32);
        Ok(())
    }

    /// Whether the bound float target can be read as RED floats.
    fn reads_red(&self) -> bool {
        if let Some(r) = self.read_red.get() {
            return r;
        }
        let q = |p| {
            self.gl
                .get_parameter(p)
                .ok()
                .and_then(|v| v.as_f64())
                .map(|v| v as u32)
        };
        let r = q(GL::IMPLEMENTATION_COLOR_READ_FORMAT) == Some(GL::RED)
            && q(GL::IMPLEMENTATION_COLOR_READ_TYPE) == Some(GL::FLOAT);
        self.read_red.set(Some(r));
        r
    }

    fn bind_tex_unit(&self, prog: &WebGlProgram, name: &str, unit: u32, tex: &WebGlTexture) {
        let gl = &self.gl;
        gl.active_texture(GL::TEXTURE0 + unit);
        gl.bind_texture(GL::TEXTURE_2D, Some(tex));
        if let Some(loc) = self.uniform(prog, name) {
            gl.uniform1i(Some(&loc), unit as i32);
        }
    }

    fn draw(&self) {
        self.gl.draw_arrays(GL::TRIANGLES, 0, 3);
    }

    /// [`Self::draw`] over a `w × h` target in bands of rows, each its own
    /// draw call, flushed: a heavy kernel (a 3-D conv over a volume) as one
    /// call can run past the browser's GPU watchdog, which loses the context.
    /// `work` is the cost of one fragment (taps read).
    fn draw_banded(&self, w: usize, h: usize, work: usize) {
        /// taps read per draw call: a few ms on an M-series GPU. Twice the
        /// 2²⁸ this was: smaller bands left a 30-channel conv of a brain ~5
        /// rows (82k fragments) a draw, and subcortical ran 6.2–7.5 s
        /// against 5.3–6.0 s; 2³⁰ was no faster, one draw a pass 2.5× slower.
        const BAND_WORK: usize = 1 << 29;
        let rows = (BAND_WORK / (w * work.max(1))).clamp(1, h.max(1));
        if rows >= h {
            return self.draw();
        }
        let gl = &self.gl;
        gl.enable(GL::SCISSOR_TEST);
        for y in (0..h).step_by(rows) {
            gl.scissor(0, y as i32, w as i32, rows.min(h - y) as i32);
            self.draw();
            gl.flush();
        }
        gl.disable(GL::SCISSOR_TEST);
    }

    fn set_ivec3(&self, prog: &WebGlProgram, name: &str, v: [usize; 3]) {
        if let Some(loc) = self.uniform(prog, name) {
            self.gl
                .uniform3i(Some(&loc), v[0] as i32, v[1] as i32, v[2] as i32);
        }
    }

    fn read_back(&self, tex: &WebGlTexture, w: usize, h: usize) -> Result<Vec<f32>> {
        let gl = &self.gl;
        self.bind_target(tex, w, h)?;
        if self.reads_red() {
            let mut out = vec![0f32; w * h];
            unsafe {
                let view = js_sys::Float32Array::view_mut_raw(out.as_mut_ptr(), out.len());
                gl.read_pixels_with_opt_array_buffer_view(
                    0,
                    0,
                    w as i32,
                    h as i32,
                    GL::RED,
                    GL::FLOAT,
                    Some(&view),
                )
                .map_err(|e| err("read_pixels", e))?;
            }
            return Ok(out);
        }
        // RGBA floats (what every WebGL2 can read a float target as), a band
        // of rows at a time: read whole, the staging was four times the
        // tensor — a phone's tab ran out on a 3-D net's slab.
        const BAND_TEXELS: usize = 1 << 20;
        let rows = (BAND_TEXELS / w.max(1)).clamp(1, h.max(1));
        let mut rgba = vec![0f32; w * rows * 4];
        let mut out = Vec::with_capacity(w * h);
        let mut y = 0;
        while y < h {
            let n = rows.min(h - y);
            let band = &mut rgba[..w * n * 4];
            unsafe {
                let view = js_sys::Float32Array::view_mut_raw(band.as_mut_ptr(), band.len());
                gl.read_pixels_with_opt_array_buffer_view(
                    0,
                    y as i32,
                    w as i32,
                    n as i32,
                    GL::RGBA,
                    GL::FLOAT,
                    Some(&view),
                )
                .map_err(|e| err("read_pixels", e))?;
            }
            out.extend(band.chunks_exact(4).map(|p| p[0]));
            y += n;
        }
        Ok(out)
    }

    /// Start a render pass into a fresh target of size `w×h` with `prog` bound.
    fn begin(&self, prog: &WebGlProgram, w: usize, h: usize) -> Result<WebGlTexture> {
        let t = self.target_texture(w, h)?;
        self.bind_target(&t, w, h)?;
        self.gl.use_program(Some(prog));
        Ok(t)
    }

    /// Drop the cached texture of param `name` (its value changed); it is
    /// uploaded again on the next run.
    /// Hold `plan`'s caches from now on: another plan's slots mean other
    /// tensors, so its textures go. A backend is a context and some fifty
    /// compiled programs — share one rather than make one a plan (a model
    /// run a layer at a time is a plan a layer).
    pub fn use_plan(&self, plan: u64) {
        if self.plan.replace(plan) == plan {
            return;
        }
        // another plan: its weights and the sizes it asked for go
        let leaves: Vec<_> = self
            .leaf_cache
            .borrow_mut()
            .drain()
            .map(|(_, t)| t)
            .collect();
        for t in leaves {
            self.sized.borrow_mut().retain(|(s, _)| *s != t);
            self.gl.delete_texture(Some(&t));
        }
        self.drain_free();
        for (b, _) in self.free_pbos.borrow_mut().drain(..) {
            self.gl.delete_buffer(Some(&b));
        }
        for (_, t) in self.idx_cache.borrow_mut().drain() {
            self.gl.delete_texture(Some(&t));
        }
        self.param_slots.borrow_mut().clear();
        self.idx_cols.borrow_mut().clear();
    }

    /// [`Self::forget_param`] when `plan` is the one cached.
    pub fn forget_param_of(&self, plan: u64, name: &str) {
        if self.plan.get() == plan {
            self.forget_param(name);
        }
    }

    pub fn forget_param(&self, name: &str) {
        let t = self
            .param_slots
            .borrow_mut()
            .remove(name)
            .and_then(|slot| self.leaf_cache.borrow_mut().remove(&slot));
        if let Some(t) = t {
            self.release(t);
        }
    }

    /// A plan-fixed index list as a texture read by LINEAR index (`lin %
    /// cols`, `lin / cols`), as wide as the browser allows: a reduce's group
    /// table has one row per output otherwise, which overflows
    /// `MAX_TEXTURE_SIZE` for any output with more than ~16k elements.
    /// Returns the texture and its column count.
    fn cached_linear_int(
        &self,
        key: (usize, u8),
        data: impl FnOnce() -> Vec<i32>,
    ) -> Result<(WebGlTexture, usize)> {
        let cols_key = (key.0, key.1 | 0x80);
        if let (Some(t), Some(c)) = (
            self.idx_cache.borrow().get(&key),
            self.idx_cols.borrow().get(&cols_key),
        ) {
            return Ok((t.clone(), *c));
        }
        let mut d = data();
        let cols = d.len().clamp(1, self.max_texture);
        let rows = d.len().div_ceil(cols).max(1);
        d.resize(cols * rows, -1);
        let t = self.int_texture(cols, rows, &d)?;
        self.idx_cache.borrow_mut().insert(key, t.clone());
        self.idx_cols.borrow_mut().insert(cols_key, cols);
        Ok((t, cols))
    }

    /// An index texture fixed by the plan: built on first use, then reused.
    fn cached_int(
        &self,
        key: (usize, u8),
        w: usize,
        h: usize,
        data: impl FnOnce() -> Vec<i32>,
    ) -> Result<WebGlTexture> {
        if let Some(t) = self.idx_cache.borrow().get(&key) {
            return Ok(t.clone());
        }
        // A widened output slot has a padded last row; its padding texels
        // read index 0 (in range) and are never read back.
        let mut d = data();
        d.resize(w * h, 0);
        let t = self.int_texture(w, h, &d)?;
        self.idx_cache.borrow_mut().insert(key, t.clone());
        Ok(t)
    }

    /// Execute `plan` with the given named inputs/params on the GPU.
    pub fn run(&self, plan: &Plan, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let tex = self.exec(plan, inputs)?;
        let outs = self.read_outputs(plan, tex);
        // A lost context (iOS drops it when the tab's GPU memory runs out)
        // reads back zeros without an error, and an allocation that failed
        // leaves only OUT_OF_MEMORY behind: neither is a result.
        self.check_alive()?;
        outs
    }

    fn check_alive(&self) -> Result<()> {
        if self.context_lost() {
            return Err(WebglError("the WebGL2 context was lost".into()));
        }
        if self.gl.get_error() == GL::OUT_OF_MEMORY {
            return Err(WebglError("WebGL2 ran out of memory".into()));
        }
        Ok(())
    }

    /// [`Self::run`] without waiting for the GPU: the outputs are copied
    /// into pixel-pack buffers and [`Self::collect`] (the oldest run of
    /// `plan_id` first) maps them. Meanwhile the CPU can prepare the next
    /// run — a run that waited for its read-back left the GPU idle while
    /// the caller gathered the next input and stored the last output.
    pub fn submit(&self, plan_id: u64, plan: &Plan, inputs: &[(&str, &[f32])]) -> Result<()> {
        let tex = self.exec(plan, inputs)?;
        let gl = &self.gl;
        let mut reads = Vec::new();
        for &slot in &plan.outputs {
            let (h, w) = plan.slot_dims[slot];
            self.bind_target(self.tex(&tex, slot)?, w, h)?;
            let red = self.reads_red();
            let floats = w * h * if red { 1 } else { 4 };
            let bytes = floats * 4;
            let reuse = {
                let mut free = self.free_pbos.borrow_mut();
                free.iter()
                    .position(|(_, b)| *b == bytes)
                    .map(|i| free.swap_remove(i).0)
            };
            let buf = match reuse {
                Some(b) => {
                    gl.bind_buffer(GL::PIXEL_PACK_BUFFER, Some(&b));
                    b
                }
                None => {
                    let b = gl
                        .create_buffer()
                        .ok_or_else(|| WebglError("create_buffer".into()))?;
                    gl.bind_buffer(GL::PIXEL_PACK_BUFFER, Some(&b));
                    gl.buffer_data_with_i32(GL::PIXEL_PACK_BUFFER, bytes as i32, GL::STREAM_READ);
                    b
                }
            };
            gl.read_pixels_with_i32(
                0,
                0,
                w as i32,
                h as i32,
                if red { GL::RED } else { GL::RGBA },
                GL::FLOAT,
                0,
            )
            .map_err(|e| err("read_pixels(pack)", e))?;
            reads.push(PendingRead {
                buf,
                bytes,
                texels: w * h,
                len: plan.slot_len[slot],
                red,
            });
        }
        gl.bind_buffer(GL::PIXEL_PACK_BUFFER, None);
        self.free_run_textures(tex);
        // send it to the GPU now: queued, it would wait for the next call
        // that waits (the collect), and nothing would overlap
        gl.flush();
        self.pending
            .borrow_mut()
            .entry(plan_id)
            .or_default()
            .push_back(Pending { reads });
        Ok(())
    }

    /// [`Self::submit`] with some inputs as IEEE half bits (`half`), sent
    /// as R16F textures: half the bytes of f32.
    pub fn submit_half(
        &self,
        plan_id: u64,
        plan: &Plan,
        inputs: &[(&str, &[f32])],
        half: Vec<(String, Vec<u16>)>,
    ) -> Result<Vec<(String, Vec<u16>)>> {
        *self.half_inputs.borrow_mut() = half;
        let r = self.submit(plan_id, plan, inputs);
        // the buffers back to the caller, for its next window
        let half = std::mem::take(&mut *self.half_inputs.borrow_mut());
        r.map(|()| half)
    }

    /// The outputs of `plan_id`'s oldest [`Self::submit`]ted run (waiting
    /// for the GPU if it is not done).
    pub fn collect(&self, plan_id: u64) -> Result<Vec<Vec<f32>>> {
        let run = self
            .pending
            .borrow_mut()
            .get_mut(&plan_id)
            .and_then(|q| q.pop_front())
            .ok_or_else(|| WebglError("collect: nothing submitted".into()))?;
        let gl = &self.gl;
        let mut outs = Vec::with_capacity(run.reads.len());
        for r in run.reads {
            gl.bind_buffer(GL::PIXEL_PACK_BUFFER, Some(&r.buf));
            let mut out = vec![0f32; r.texels];
            if r.red {
                unsafe {
                    let view = js_sys::Float32Array::view_mut_raw(out.as_mut_ptr(), out.len());
                    gl.get_buffer_sub_data_with_i32_and_array_buffer_view(
                        GL::PIXEL_PACK_BUFFER,
                        0,
                        &view,
                    );
                }
            } else {
                // RGBA a band at a time (whole, four times the tensor)
                const BAND: usize = 1 << 20;
                let mut rgba = vec![0f32; BAND.min(r.texels) * 4];
                let mut at = 0;
                while at < r.texels {
                    let n = BAND.min(r.texels - at);
                    let band = &mut rgba[..n * 4];
                    unsafe {
                        let view =
                            js_sys::Float32Array::view_mut_raw(band.as_mut_ptr(), band.len());
                        gl.get_buffer_sub_data_with_i32_and_array_buffer_view(
                            GL::PIXEL_PACK_BUFFER,
                            (at * 16) as i32,
                            &view,
                        );
                    }
                    for (o, p) in out[at..at + n].iter_mut().zip(band.chunks_exact(4)) {
                        *o = p[0];
                    }
                    at += n;
                }
            }
            gl.bind_buffer(GL::PIXEL_PACK_BUFFER, None);
            // kept for the next run (two in flight at most)
            let mut free = self.free_pbos.borrow_mut();
            if free.len() < 4 {
                free.push((r.buf, r.bytes));
            } else {
                gl.delete_buffer(Some(&r.buf));
            }
            out.truncate(r.len);
            outs.push(out);
        }
        self.check_alive()?;
        Ok(outs)
    }

    /// Run `plan`'s steps; the textures by slot (outputs among them).
    fn exec(&self, plan: &Plan, inputs: &[(&str, &[f32])]) -> Result<Vec<Option<WebGlTexture>>> {
        let input_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        let mut tex: Vec<Option<WebGlTexture>> = (0..plan.slot_len.len()).map(|_| None).collect();
        // The step that reads each slot last: its texture goes after it. A
        // volume model holds two or three activations at a time, not all of
        // them (ten 30-channel layers over a brain crop: ~14 GB at once).
        let mut last_read = vec![usize::MAX; plan.slot_len.len()];
        for (si, step) in plan.steps.iter().enumerate() {
            for s in step.reads() {
                last_read[s] = si;
            }
        }

        // `globalThis.__rlxWebglProfile = true`: wait for the GPU after each
        // step and log the time by step kind (rank, don't quote: the waits
        // themselves cost)
        let profile = js_sys::Reflect::get(&js_sys::global(), &"__rlxWebglProfile".into())
            .is_ok_and(|v| v.is_truthy());
        let mut spent: Vec<(String, f64, usize)> = Vec::new();
        for (si, step) in plan.steps.iter().enumerate() {
            let t0 = if profile { js_sys::Date::now() } else { 0.0 };
            self.run_step(plan, si, step, &input_map, &mut tex)?;
            if profile {
                self.gl.finish();
                // (a leaf's Debug prints its data)
                let kind = match step {
                    Step::Leaf { .. } => "Leaf".to_string(),
                    _ => format!("{step:?}"),
                };
                let kind = kind
                    .split([' ', '{', '('])
                    .next()
                    .unwrap_or("?")
                    .to_string();
                let ms = js_sys::Date::now() - t0;
                match spent.iter_mut().find(|e| e.0 == kind) {
                    Some(e) => (e.1, e.2) = (e.1 + ms, e.2 + 1),
                    None => spent.push((kind, ms, 1)),
                }
            }
            for s in step.reads() {
                if last_read[s] == si
                    && !plan.outputs.contains(&s)
                    && !self.leaf_cache.borrow().contains_key(&s)
                    && let Some(t) = tex[s].take()
                {
                    self.release(t);
                }
            }
        }
        if profile {
            spent.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total: f64 = spent.iter().map(|e| e.1).sum();
            let mut msg = format!("[rlx-webgl] step profile: {total:.0} ms\n");
            for (k, ms, n) in &spent {
                msg += &format!(
                    "  {k:<16} {n:>5}× {ms:>9.1} ms {:>5.1}%\n",
                    100.0 * ms / total
                );
            }
            web_sys::console::log_1(&msg.into());
        }
        Ok(tex)
    }

    /// Whether the browser lost this context: nothing more runs on it.
    pub fn context_lost(&self) -> bool {
        self.gl.is_context_lost()
    }

    fn run_step(
        &self,
        plan: &Plan,
        si: usize,
        step: &Step,
        input_map: &HashMap<&str, &[f32]>,
        tex: &mut [Option<WebGlTexture>],
    ) -> Result<()> {
        let dims = |s: usize| plan.slot_dims[s];
        {
            match step {
                Step::Leaf { out, src } => {
                    let persistent = !matches!(src, LeafSource::Input(_));
                    if persistent && let Some(t) = self.leaf_cache.borrow().get(out) {
                        tex[*out] = Some(t.clone());
                        return Ok(());
                    }
                    let (r, c) = dims(*out);
                    // an input already on the GPU, laid out as this slot
                    if let LeafSource::Input(name) = src
                        && let Some(t) = self
                            .gpu_inputs
                            .borrow()
                            .iter()
                            .find(|(n, _)| n == name)
                            .map(|(_, t)| t.clone())
                    {
                        tex[*out] = Some(t);
                        return Ok(());
                    }
                    // an input given in half precision: an R16F texture, half
                    // the bytes to send (the shaders read `.r` either way)
                    let half = match src {
                        LeafSource::Input(name) => self
                            .half_inputs
                            .borrow()
                            .iter()
                            .position(|(n, _)| n == name),
                        _ => None,
                    };
                    let t = if let Some(i) = half {
                        let halves = self.half_inputs.borrow();
                        self.half_texture(c, r, &halves[i].1)?
                    } else {
                        // (the slice itself: a copy of each input every run was
                        // a slab window duplicated)
                        let data: &[f32] = match src {
                            LeafSource::Input(name) | LeafSource::Param(name) => {
                                input_map.get(name.as_str()).copied().ok_or_else(|| {
                                    WebglError(format!("missing input/param '{name}'"))
                                })?
                            }
                            LeafSource::Const(d) => d,
                        };
                        self.data_texture(c, r, data)?
                    };
                    if persistent {
                        self.leaf_cache.borrow_mut().insert(*out, t.clone());
                        if let LeafSource::Param(name) = src {
                            self.param_slots.borrow_mut().insert(name.clone(), *out);
                        }
                    }
                    tex[*out] = Some(t);
                }
                Step::Unary { out, a, act } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.unary, c, r)?;
                    self.bind_tex_unit(&self.unary, "A", 0, self.tex(tex, *a)?);
                    self.set_i32(&self.unary, "uAct", act_code(*act));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::ActBack { out, x, dy, act } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.actback, c, r)?;
                    self.bind_tex_unit(&self.actback, "X", 0, self.tex(tex, *x)?);
                    self.bind_tex_unit(&self.actback, "DY", 1, self.tex(tex, *dy)?);
                    self.set_i32(&self.actback, "uAct", act_code(*act));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Binary { out, a, b, op } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.binary, c, r)?;
                    self.bind_tex_unit(&self.binary, "A", 0, self.tex(tex, *a)?);
                    self.bind_tex_unit(&self.binary, "B", 1, self.tex(tex, *b)?);
                    self.set_i32(&self.binary, "uOp", bin_code(*op));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Compare { out, a, b, cmp } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.compare, c, r)?;
                    self.bind_tex_unit(&self.compare, "A", 0, self.tex(tex, *a)?);
                    self.bind_tex_unit(&self.compare, "B", 1, self.tex(tex, *b)?);
                    self.set_i32(&self.compare, "uOp", cmp_code(*cmp));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Where { out, cond, a, b } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.where_, c, r)?;
                    self.bind_tex_unit(&self.where_, "C", 0, self.tex(tex, *cond)?);
                    self.bind_tex_unit(&self.where_, "A", 1, self.tex(tex, *a)?);
                    self.bind_tex_unit(&self.where_, "B", 2, self.tex(tex, *b)?);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::MatMul {
                    out,
                    a,
                    b,
                    m,
                    k,
                    n,
                    batch,
                    b_batched,
                } => {
                    let ((or, oc), (_, ac), (_, bc)) = (dims(*out), dims(*a), dims(*b));
                    let natural = oc == *n && or == batch * m && ac == *k && bc == *n;
                    let t = self.begin(&self.matmul, oc, or)?;
                    self.bind_tex_unit(&self.matmul, "A", 0, self.tex(tex, *a)?);
                    self.bind_tex_unit(&self.matmul, "B", 1, self.tex(tex, *b)?);
                    self.set_i32(&self.matmul, "uK", *k as i32);
                    self.set_i32(&self.matmul, "uM", *m as i32);
                    self.set_i32(&self.matmul, "uN", *n as i32);
                    self.set_i32(&self.matmul, "uBBatched", i32::from(*b_batched));
                    self.set_i32(&self.matmul, "uNatural", i32::from(natural));
                    self.set_i32(&self.matmul, "uACols", ac as i32);
                    self.set_i32(&self.matmul, "uBCols", bc as i32);
                    self.set_i32(&self.matmul, "uOCols", oc as i32);
                    self.set_i32(&self.matmul, "uTotal", (batch * m * n) as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Gather { out, src, idx } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let idx_tex =
                        self.cached_int((si, 0), c, r, || idx.iter().map(|&v| v as i32).collect())?;
                    let t = self.begin(&self.gather, c, r)?;
                    self.bind_tex_unit(&self.gather, "SRC", 0, self.tex(tex, *src)?);
                    self.bind_tex_unit(&self.gather, "IDX", 1, &idx_tex);
                    self.set_i32(&self.gather, "uSrcCols", sc as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Reduce {
                    out,
                    src,
                    groups,
                    fanin,
                    op,
                } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let (g_tex, g_cols) = self.cached_linear_int((si, 0), || {
                        groups.iter().map(|&v| v as i32).collect()
                    })?;
                    let t = self.begin(&self.reduce, c, r)?;
                    self.bind_tex_unit(&self.reduce, "SRC", 0, self.tex(tex, *src)?);
                    self.bind_tex_unit(&self.reduce, "GROUPS", 1, &g_tex);
                    self.set_i32(&self.reduce, "uFanin", *fanin as i32);
                    self.set_i32(&self.reduce, "uSrcCols", sc as i32);
                    self.set_i32(&self.reduce, "uOutCols", c as i32);
                    self.set_i32(&self.reduce, "uGCols", g_cols as i32);
                    self.set_i32(&self.reduce, "uOp", red_code(*op));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Softmax { out, a, rows, cols } => {
                    let (or, oc) = dims(*out);
                    let t = self.begin(&self.softmax, oc, or)?;
                    self.bind_tex_unit(&self.softmax, "A", 0, self.tex(tex, *a)?);
                    self.set_i32(&self.softmax, "uCols", *cols as i32);
                    self.set_i32(&self.softmax, "uSlotCols", oc as i32);
                    self.set_i32(&self.softmax, "uTotal", (rows * cols) as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::LayerNorm {
                    out,
                    x,
                    gamma,
                    beta,
                    rows,
                    cols,
                    eps,
                } => {
                    let ((or, oc), (_, gc)) = (dims(*out), dims(*gamma));
                    let t = self.begin(&self.layernorm, oc, or)?;
                    self.bind_tex_unit(&self.layernorm, "X", 0, self.tex(tex, *x)?);
                    self.bind_tex_unit(&self.layernorm, "G", 1, self.tex(tex, *gamma)?);
                    self.bind_tex_unit(&self.layernorm, "B", 2, self.tex(tex, *beta)?);
                    self.set_i32(&self.layernorm, "uCols", *cols as i32);
                    self.set_i32(&self.layernorm, "uSlotCols", oc as i32);
                    self.set_i32(&self.layernorm, "uGCols", gc as i32);
                    self.set_i32(&self.layernorm, "uTotal", (rows * cols) as i32);
                    self.set_f32(&self.layernorm, "uEps", *eps);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::RmsNorm {
                    out,
                    x,
                    gamma,
                    beta,
                    rows,
                    cols,
                    eps,
                } => {
                    let ((or, oc), (_, gc)) = (dims(*out), dims(*gamma));
                    let t = self.begin(&self.rmsnorm, oc, or)?;
                    self.bind_tex_unit(&self.rmsnorm, "X", 0, self.tex(tex, *x)?);
                    self.bind_tex_unit(&self.rmsnorm, "G", 1, self.tex(tex, *gamma)?);
                    self.bind_tex_unit(&self.rmsnorm, "B", 2, self.tex(tex, *beta)?);
                    self.set_i32(&self.rmsnorm, "uCols", *cols as i32);
                    self.set_i32(&self.rmsnorm, "uSlotCols", oc as i32);
                    self.set_i32(&self.rmsnorm, "uGCols", gc as i32);
                    self.set_i32(&self.rmsnorm, "uTotal", (rows * cols) as i32);
                    self.set_f32(&self.rmsnorm, "uEps", *eps);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::ArgReduce {
                    out,
                    src,
                    groups,
                    fanin,
                    is_max,
                } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let (g_tex, g_cols) = self.cached_linear_int((si, 0), || {
                        groups.iter().map(|&v| v as i32).collect()
                    })?;
                    let t = self.begin(&self.argreduce, c, r)?;
                    self.bind_tex_unit(&self.argreduce, "SRC", 0, self.tex(tex, *src)?);
                    self.bind_tex_unit(&self.argreduce, "GROUPS", 1, &g_tex);
                    self.set_i32(&self.argreduce, "uFanin", *fanin as i32);
                    self.set_i32(&self.argreduce, "uSrcCols", sc as i32);
                    self.set_i32(&self.argreduce, "uOutCols", c as i32);
                    self.set_i32(&self.argreduce, "uGCols", g_cols as i32);
                    self.set_i32(&self.argreduce, "uIsMax", if *is_max { 1 } else { 0 });
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::GatherRuntime {
                    out,
                    table,
                    indices,
                    which,
                    base,
                    axis_stride,
                    wrap,
                } => {
                    let (r, c) = dims(*out);
                    let (_tr, tc) = dims(*table);
                    let (_ir, ic) = dims(*indices);
                    let which_tex = self
                        .cached_int((si, 0), c, r, || which.iter().map(|&v| v as i32).collect())?;
                    let base_tex = self
                        .cached_int((si, 1), c, r, || base.iter().map(|&v| v as i32).collect())?;
                    let t = self.begin(&self.gather_rt, c, r)?;
                    self.bind_tex_unit(&self.gather_rt, "TABLE", 0, self.tex(tex, *table)?);
                    self.bind_tex_unit(&self.gather_rt, "IDXT", 1, self.tex(tex, *indices)?);
                    self.bind_tex_unit(&self.gather_rt, "WHICH", 2, &which_tex);
                    self.bind_tex_unit(&self.gather_rt, "BASE", 3, &base_tex);
                    self.set_i32(&self.gather_rt, "uTableCols", tc as i32);
                    self.set_i32(&self.gather_rt, "uIdxCols", ic as i32);
                    self.set_i32(&self.gather_rt, "uAxisStride", *axis_stride as i32);
                    self.set_i32(&self.gather_rt, "uWrap", *wrap as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::ScatterRuntime {
                    out,
                    data,
                    updates,
                    indices,
                    ibase,
                    ubase,
                    key,
                    count,
                    istride,
                    ustride,
                    wrap,
                    red,
                } => {
                    let (r, c) = dims(*out);
                    let (_ur, uc) = dims(*updates);
                    let (_ir, ic) = dims(*indices);
                    let int_tab = |which: u8, v: &[u32]| {
                        self.cached_int((si, which), c, r, || v.iter().map(|&x| x as i32).collect())
                    };
                    let (ib_tex, ub_tex, key_tex) =
                        (int_tab(0, ibase)?, int_tab(1, ubase)?, int_tab(2, key)?);
                    let p = &self.scatter_rt;
                    let t = self.begin(p, c, r)?;
                    // Without `data`, bind any texture: the shader never reads it.
                    let data_tex = self.tex(tex, data.unwrap_or(*updates))?;
                    self.bind_tex_unit(p, "DATA", 0, data_tex);
                    self.bind_tex_unit(p, "UPD", 1, self.tex(tex, *updates)?);
                    self.bind_tex_unit(p, "IDXT", 2, self.tex(tex, *indices)?);
                    self.bind_tex_unit(p, "IBASE", 3, &ib_tex);
                    self.bind_tex_unit(p, "UBASE", 4, &ub_tex);
                    self.bind_tex_unit(p, "KEY", 5, &key_tex);
                    self.set_i32(p, "uHasData", i32::from(data.is_some()));
                    self.set_i32(p, "uCount", *count as i32);
                    self.set_i32(p, "uIStride", *istride as i32);
                    self.set_i32(p, "uUStride", *ustride as i32);
                    self.set_i32(p, "uWrap", *wrap as i32);
                    self.set_i32(p, "uRed", scatter_red_code(*red));
                    self.set_i32(p, "uUpdCols", uc as i32);
                    self.set_i32(p, "uIdxCols", ic as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::FftStage {
                    out,
                    src,
                    tw,
                    n,
                    ns,
                    in_block,
                    out_block,
                    scale,
                } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let (_tr, tc) = dims(*tw);
                    let p = &self.fft_stage;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.bind_tex_unit(p, "TW", 1, self.tex(tex, *tw)?);
                    self.set_i32(p, "uN", *n as i32);
                    self.set_i32(p, "uNs", *ns as i32);
                    self.set_i32(p, "uInBlock", i32::from(*in_block));
                    self.set_i32(p, "uOutBlock", i32::from(*out_block));
                    self.set_i32(p, "uSrcCols", sc as i32);
                    self.set_i32(p, "uTwCols", tc as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_f32(p, "uScale", *scale);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Host { out, inputs, op } => {
                    // rlx-cpu runs it (see `crate::host`): read back, run, upload.
                    let mut ins = Vec::with_capacity(inputs.len());
                    for &s in inputs {
                        let (sr, sc) = dims(s);
                        let mut v = self.read_back(self.tex(tex, s)?, sc, sr)?;
                        v.truncate(plan.slot_len[s]);
                        ins.push(v);
                    }
                    let refs: Vec<&[f32]> = ins.iter().map(Vec::as_slice).collect();
                    let r = op.run(&refs)?;
                    let (rr, rc) = dims(*out);
                    tex[*out] = Some(self.data_texture(rc, rr, &r)?);
                }
                Step::Linalg {
                    out,
                    a,
                    b,
                    n,
                    nrhs,
                    batch,
                    op,
                } => {
                    // Host-staged (see `crate::linalg`): read back, solve, upload.
                    let read = |s: usize| -> Result<Vec<f32>> {
                        let (sr, sc) = dims(s);
                        let mut v = self.read_back(self.tex(tex, s)?, sc, sr)?;
                        v.truncate(plan.slot_len[s]);
                        Ok(v)
                    };
                    let av = read(*a)?;
                    let bv = b.map(read).transpose()?;
                    let r = crate::linalg::run(*op, &av, bv.as_deref(), *n, *nrhs, *batch);
                    let (rr, rc) = dims(*out);
                    tex[*out] = Some(self.data_texture(rc, rr, &r)?);
                }
                Step::SortAxis {
                    out,
                    src,
                    outer: _,
                    len,
                    inner,
                    k,
                    descending,
                    indices,
                } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let p = &self.sort;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uLen", *len as i32);
                    self.set_i32(p, "uInner", *inner as i32);
                    self.set_i32(p, "uK", *k as i32);
                    self.set_i32(p, "uDesc", i32::from(*descending));
                    self.set_i32(p, "uIdx", i32::from(*indices));
                    self.set_i32(p, "uSrcCols", sc as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::ComplexCast {
                    out,
                    src,
                    mode,
                    n: _,
                } => {
                    // One output lane per fragment; the (lane-aware) out/src slot
                    // cols unflatten gl_FragCoord and the source lane index.
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let t = self.begin(&self.complex_cast, c, r)?;
                    self.bind_tex_unit(&self.complex_cast, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(&self.complex_cast, "uMode", *mode as i32);
                    self.set_i32(&self.complex_cast, "uSrcCols", sc as i32);
                    self.set_i32(&self.complex_cast, "uOutCols", c as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::BinaryC64 {
                    out,
                    a,
                    b,
                    op,
                    n: _,
                    n_a,
                    n_b,
                } => {
                    let (r, c) = dims(*out);
                    let (_ar, ac) = dims(*a);
                    let (_br, bc) = dims(*b);
                    let t = self.begin(&self.binary_c64, c, r)?;
                    self.bind_tex_unit(&self.binary_c64, "A", 0, self.tex(tex, *a)?);
                    self.bind_tex_unit(&self.binary_c64, "B", 1, self.tex(tex, *b)?);
                    self.set_i32(&self.binary_c64, "uOp", bin_code(*op));
                    self.set_i32(&self.binary_c64, "uNa", *n_a as i32);
                    self.set_i32(&self.binary_c64, "uNb", *n_b as i32);
                    self.set_i32(&self.binary_c64, "uACols", ac as i32);
                    self.set_i32(&self.binary_c64, "uBCols", bc as i32);
                    self.set_i32(&self.binary_c64, "uOutCols", c as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Custom { name, .. } => {
                    // Host/transport `collective.*` ops cannot run in-browser:
                    // a fragment shader can't drive a process group, and the
                    // rlx-driver transport uses std::net TCP sockets, which
                    // don't exist on wasm32. Report that plainly.
                    return Err(WebglError(format!(
                        "collective '{name}' unavailable in browser: no TCP transport on wasm32. \
                         Run the collective graph on the native CPU executor (exec_cpu)."
                    )));
                }
                Step::Conv3d {
                    out,
                    x,
                    w,
                    bias,
                    n,
                    cin,
                    din,
                    cout,
                    dout,
                    k,
                    stride,
                    pad,
                    dil,
                    groups,
                    act,
                } => {
                    if *groups == 1 && *cout >= 4 {
                        let t = self.conv3d_quads(plan, tex, step)?;
                        tex[*out] = Some(t);
                        return Ok(());
                    }
                    let (r, c) = dims(*out);
                    let p = &self.conv3d;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "X", 0, self.tex(tex, *x)?);
                    self.bind_tex_unit(p, "W", 1, self.tex(tex, *w)?);
                    // an unbiased conv binds the weights as a stand-in (unread)
                    let b = bias.unwrap_or(*w);
                    self.bind_tex_unit(p, "BIAS", 2, self.tex(tex, b)?);
                    self.set_i32(p, "uXCols", dims(*x).1 as i32);
                    self.set_i32(p, "uWCols", dims(*w).1 as i32);
                    self.set_i32(p, "uBCols", dims(b).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(
                        p,
                        "uTotal",
                        (n * cout * dout.iter().product::<usize>()) as i32,
                    );
                    self.set_i32(p, "uHasBias", i32::from(bias.is_some()));
                    self.set_i32(p, "uAct", act.map_or(-1, act_code));
                    self.set_i32(p, "uCin", *cin as i32);
                    self.set_i32(p, "uCout", *cout as i32);
                    self.set_i32(p, "uGroups", *groups as i32);
                    self.set_ivec3(p, "uIn", *din);
                    self.set_ivec3(p, "uOut", *dout);
                    self.set_ivec3(p, "uK", *k);
                    self.set_ivec3(p, "uS", *stride);
                    self.set_ivec3(p, "uP", *pad);
                    self.set_ivec3(p, "uD", *dil);
                    self.draw_banded(c, r, cin / groups * k.iter().product::<usize>());
                    tex[*out] = Some(t);
                }
                Step::ArgAxis {
                    out,
                    src,
                    outer,
                    len,
                    inner,
                    is_max,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.argaxis;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uSrcCols", dims(*src).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", (outer * inner) as i32);
                    self.set_i32(p, "uLen", *len as i32);
                    self.set_i32(p, "uInner", *inner as i32);
                    self.set_i32(p, "uIsMax", i32::from(*is_max));
                    self.draw_banded(c, r, *len);
                    tex[*out] = Some(t);
                }
                Step::Conv1ArgMax {
                    out,
                    x,
                    w,
                    bias,
                    n,
                    cin,
                    cout,
                    spatial,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.conv1_argmax;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "X", 0, self.tex(tex, *x)?);
                    self.bind_tex_unit(p, "W", 1, self.tex(tex, *w)?);
                    let b = bias.unwrap_or(*w);
                    self.bind_tex_unit(p, "BIAS", 2, self.tex(tex, b)?);
                    self.set_i32(p, "uXCols", dims(*x).1 as i32);
                    self.set_i32(p, "uWCols", dims(*w).1 as i32);
                    self.set_i32(p, "uBCols", dims(b).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", (n * spatial) as i32);
                    self.set_i32(p, "uHasBias", i32::from(bias.is_some()));
                    self.set_i32(p, "uCin", *cin as i32);
                    self.set_i32(p, "uCout", *cout as i32);
                    self.set_i32(p, "uSpatial", *spatial as i32);
                    self.draw_banded(c, r, cin * cout);
                    tex[*out] = Some(t);
                }
                Step::GroupNorm {
                    out,
                    x,
                    gamma,
                    beta,
                    n,
                    c: ch,
                    groups,
                    spatial,
                    eps,
                    act,
                } => {
                    // mean, then the variance about it (two-pass, as the CPU
                    // mirror), each a tree of chunk sums: 256 elements a
                    // fragment, then 256 of those, … to one per group. (One
                    // level of ~4096-element chunks capped at a texture row
                    // gave a brain-sized channel 16k fragments of 15k serial
                    // reads each.)
                    const CHUNK: usize = 256;
                    let run = ch / groups * spatial;
                    let ng = n * groups;
                    let xt = self.tex(tex, *x)?.clone();
                    let x_cols = dims(*x).1;
                    let stat = |mean: Option<&WebGlTexture>| -> Result<WebGlTexture> {
                        let pp = &self.gn_partial;
                        // (texture, its columns, values per group)
                        let mut cur: Option<WebGlTexture> = None;
                        let (mut cols, mut len) = (x_cols, run);
                        let mut first = true;
                        while first || len > 1 {
                            let chunks = len.div_ceil(CHUNK);
                            let total = ng * chunks;
                            let pc = total.min(self.max_texture);
                            let pr = total.div_ceil(pc);
                            let partial = self.begin(pp, pc, pr)?;
                            self.bind_tex_unit(pp, "X", 0, cur.as_ref().unwrap_or(&xt));
                            self.bind_tex_unit(pp, "MEAN", 1, mean.unwrap_or(&xt));
                            self.set_i32(pp, "uXCols", cols as i32);
                            self.set_i32(pp, "uOutCols", pc as i32);
                            self.set_i32(pp, "uTotal", total as i32);
                            self.set_i32(pp, "uRun", len as i32);
                            self.set_i32(pp, "uChunk", CHUNK as i32);
                            self.set_i32(pp, "uNChunks", chunks as i32);
                            self.set_i32(pp, "uMode", i32::from(first && mean.is_some()));
                            self.draw_banded(pc, pr, CHUNK);
                            if let Some(t) = cur.replace(partial) {
                                self.release(t);
                            }
                            (cols, len, first) = (pc, chunks, false);
                        }
                        // one sum per group, in a row: over the element count
                        let sums = cur.expect("a level");
                        let ps = &self.gn_stat;
                        let st = self.begin(ps, ng, 1)?;
                        self.bind_tex_unit(ps, "P", 0, &sums);
                        self.set_i32(ps, "uNChunks", 1);
                        self.set_i32(ps, "uRun", run as i32);
                        self.draw();
                        self.release(sums);
                        Ok(st)
                    };
                    let mean = stat(None)?;
                    let var = stat(Some(&mean))?;
                    let (r, c) = dims(*out);
                    let pa = &self.gn_apply;
                    let t = self.begin(pa, c, r)?;
                    self.bind_tex_unit(pa, "X", 0, &xt);
                    self.bind_tex_unit(pa, "MEAN", 1, &mean);
                    self.bind_tex_unit(pa, "VAR", 2, &var);
                    self.bind_tex_unit(pa, "G", 3, self.tex(tex, *gamma)?);
                    self.bind_tex_unit(pa, "B", 4, self.tex(tex, *beta)?);
                    self.set_i32(pa, "uXCols", x_cols as i32);
                    self.set_i32(pa, "uOutCols", c as i32);
                    self.set_i32(pa, "uTotal", (ng * run) as i32);
                    self.set_i32(pa, "uRun", run as i32);
                    self.set_i32(pa, "uGroups", *groups as i32);
                    self.set_i32(pa, "uCpg", (ch / groups) as i32);
                    self.set_i32(pa, "uSpatial", *spatial as i32);
                    self.set_f32(pa, "uEps", *eps);
                    self.set_i32(pa, "uAct", act.map_or(-1, act_code));
                    self.draw();
                    self.release(mean);
                    self.release(var);
                    tex[*out] = Some(t);
                }
                Step::Concat {
                    out,
                    a,
                    b,
                    outer,
                    a_len,
                    b_len,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.concat;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "A", 0, self.tex(tex, *a)?);
                    self.bind_tex_unit(p, "B", 1, self.tex(tex, *b)?);
                    self.set_i32(p, "uACols", dims(*a).1 as i32);
                    self.set_i32(p, "uBCols", dims(*b).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", (outer * (a_len + b_len)) as i32);
                    self.set_i32(p, "uALen", *a_len as i32);
                    self.set_i32(p, "uBLen", *b_len as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Pool3d {
                    out,
                    src,
                    nc,
                    din,
                    dout,
                    k,
                    stride,
                    pad,
                    is_max,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.pool3d;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uSrcCols", dims(*src).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", (nc * dout.iter().product::<usize>()) as i32);
                    self.set_i32(p, "uIsMax", i32::from(*is_max));
                    self.set_ivec3(p, "uIn", *din);
                    self.set_ivec3(p, "uOut", *dout);
                    self.set_ivec3(p, "uK", *k);
                    self.set_ivec3(p, "uS", *stride);
                    self.set_ivec3(p, "uP", *pad);
                    self.draw_banded(c, r, k.iter().product());
                    tex[*out] = Some(t);
                }
                Step::Resize3d {
                    out,
                    src,
                    nc,
                    din,
                    dout,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.resize3d;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uSrcCols", dims(*src).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", (nc * dout.iter().product::<usize>()) as i32);
                    self.set_ivec3(p, "uIn", *din);
                    self.set_ivec3(p, "uOut", *dout);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Broadcast {
                    out,
                    src,
                    out_dims,
                    in_strides,
                } => {
                    let (r, c) = dims(*out);
                    let p = &self.broadcast;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uSrcCols", dims(*src).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(p, "uTotal", out_dims.iter().product::<usize>() as i32);
                    self.set_i32(p, "uRank", out_dims.len() as i32);
                    let pad8 = |v: &[usize]| {
                        let mut a = [1i32; 8];
                        for (i, &x) in v.iter().enumerate() {
                            a[i] = x as i32;
                        }
                        a
                    };
                    if let Some(loc) = self.uniform(p, "uDims") {
                        self.gl
                            .uniform1iv_with_i32_array(Some(&loc), &pad8(out_dims));
                    }
                    if let Some(loc) = self.uniform(p, "uStr") {
                        self.gl
                            .uniform1iv_with_i32_array(Some(&loc), &pad8(in_strides));
                    }
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Splat { out, src } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.splat, c, r)?;
                    self.bind_tex_unit(&self.splat, "SRC", 0, self.tex(tex, *src)?);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Copy { out, src } => {
                    let (r, c) = dims(*out);
                    let p = &self.copy;
                    let t = self.begin(p, c, r)?;
                    self.bind_tex_unit(p, "SRC", 0, self.tex(tex, *src)?);
                    self.set_i32(p, "uSrcCols", dims(*src).1 as i32);
                    self.set_i32(p, "uOutCols", c as i32);
                    self.set_i32(
                        p,
                        "uTotal",
                        plan.slot_len[*src].min(plan.slot_len[*out]) as i32,
                    );
                    self.draw();
                    tex[*out] = Some(t);
                }
            }
        }
        Ok(())
    }

    /// [`Step::Conv3d`] (ungrouped) `4·t` output channels per fragment: the
    /// weights regrouped once into RGBA texels (`FS_PACK_W4`), then per batch
    /// item and group of `t` channel quads a pass over the output volume
    /// into `t` RGBA targets ([`fs_conv3d_quads`]) copied into the output's
    /// rows ([`fs_unpack_quads`]). The targets hold `4·t` channels, a small
    /// part of a wide layer's output. Reads each input tap once for `4·t`
    /// channels instead of once per channel, and four weights per fetch.
    fn conv3d_quads(
        &self,
        plan: &Plan,
        tex: &[Option<WebGlTexture>],
        step: &Step,
    ) -> Result<WebGlTexture> {
        let Step::Conv3d {
            out,
            x,
            w,
            bias,
            n,
            cin,
            din,
            cout,
            dout,
            k,
            stride,
            pad,
            dil,
            act,
            ..
        } = step
        else {
            unreachable!("conv3d_quads runs Step::Conv3d")
        };
        let dims = |s: usize| plan.slot_dims[s];
        let gl = &self.gl;
        let tap = cin * k.iter().product::<usize>();
        // as few passes of up to 16 channels (four targets: the most for
        // which WebGL2 guarantees draw buffers) as the layer needs, then as
        // few targets as cover them: 30 channels are two passes of 16, 20
        // two of 12. Against four channels a fragment, 16 is 1.5× on
        // MindGrab and subcortical; 8 and 12 fall between.
        let most = self.conv3d_quads.len();
        let passes = cout.div_ceil(4 * most);
        let t_n = cout.div_ceil(4 * passes);
        let per = 4 * t_n;
        // the weights pad to a whole number of passes
        let quads = cout.div_ceil(per) * t_n;
        let lay = |len: usize| {
            let c = len.clamp(1, self.max_texture);
            (c, len.div_ceil(c).max(1))
        };

        // weights, regrouped
        let (wc, wr) = {
            let len = quads * tap;
            let wc = (len.min(self.max_texture) / t_n).max(1) * t_n;
            (wc, len.div_ceil(wc).max(1))
        };
        let w4 = self.target_texture_as(wc, wr, GL::RGBA32F, GL::RGBA)?;
        self.bind_target(&w4, wc, wr)?;
        let p = &self.pack_w4;
        gl.use_program(Some(p));
        self.bind_tex_unit(p, "W", 0, self.tex(tex, *w)?);
        self.set_i32(p, "uWCols", dims(*w).1 as i32);
        self.set_i32(p, "uOutCols", wc as i32);
        self.set_i32(p, "uTotal", (quads * tap) as i32);
        self.set_i32(p, "uTap", tap as i32);
        self.set_i32(p, "uCout", *cout as i32);
        self.set_i32(p, "uT", t_n as i32);
        self.draw();

        let (r, c) = dims(*out);
        let t = self.target_texture(c, r)?;
        let sp: usize = dout.iter().product();
        let (qc, qr) = lay(sp);
        let targets = (0..t_n)
            .map(|_| self.target_texture_as(qc, qr, GL::RGBA32F, GL::RGBA))
            .collect::<Result<Vec<_>>>()?;
        let attach = |j: usize, tex: Option<&WebGlTexture>| {
            gl.framebuffer_texture_2d(
                GL::FRAMEBUFFER,
                GL::COLOR_ATTACHMENT0 + j as u32,
                GL::TEXTURE_2D,
                tex,
                0,
            )
        };
        let buffers = |m: usize| {
            let a = js_sys::Array::new();
            for j in 0..m {
                a.push(&(GL::COLOR_ATTACHMENT0 + j as u32).into());
            }
            gl.draw_buffers(&a);
        };
        let b = bias.unwrap_or(*w);
        for ni in 0..*n {
            for g in (0..quads).step_by(t_n) {
                self.bind_target(&targets[0], qc, qr)?;
                for (j, q) in targets.iter().enumerate().skip(1) {
                    attach(j, Some(q));
                }
                if t_n > 1 {
                    buffers(t_n);
                }
                let p = &self.conv3d_quads[t_n - 1];
                gl.use_program(Some(p));
                self.bind_tex_unit(p, "X", 0, self.tex(tex, *x)?);
                self.bind_tex_unit(p, "W4", 1, &w4);
                self.bind_tex_unit(p, "BIAS", 2, self.tex(tex, b)?);
                self.set_i32(p, "uXCols", dims(*x).1 as i32);
                self.set_i32(p, "uWCols", wc as i32);
                self.set_i32(p, "uBCols", dims(b).1 as i32);
                self.set_i32(p, "uOutCols", qc as i32);
                self.set_i32(p, "uTotal", sp as i32);
                self.set_i32(p, "uHasBias", i32::from(bias.is_some()));
                self.set_i32(p, "uCin", *cin as i32);
                self.set_i32(p, "uCout", *cout as i32);
                self.set_i32(p, "uN", ni as i32);
                self.set_i32(p, "uCo0", (4 * g) as i32);
                self.set_ivec3(p, "uIn", *din);
                self.set_ivec3(p, "uOut", *dout);
                self.set_ivec3(p, "uK", *k);
                self.set_ivec3(p, "uS", *stride);
                self.set_ivec3(p, "uP", *pad);
                self.set_ivec3(p, "uD", *dil);
                self.draw_banded(qc, qr, tap * t_n);
                if t_n > 1 {
                    // back to the one target every other pass draws into
                    buffers(1);
                    for j in 1..t_n {
                        attach(j, None);
                    }
                }

                // into the output rows holding elements lo .. hi
                let lo = (ni * cout + 4 * g) * sp;
                let hi = (ni * cout + (4 * g + per).min(*cout)) * sp;
                self.bind_target(&t, c, r)?;
                let p = &self.unpack_quads[t_n - 1];
                gl.use_program(Some(p));
                for (j, q) in targets.iter().enumerate() {
                    self.bind_tex_unit(p, &format!("T{j}"), j as u32, q);
                }
                self.set_i32(p, "uTCols", qc as i32);
                self.set_i32(p, "uOutCols", c as i32);
                self.set_i32(p, "uLo", lo as i32);
                self.set_i32(p, "uHi", hi as i32);
                self.set_i32(p, "uSp", sp as i32);
                self.set_i32(p, "uAct", act.map_or(-1, act_code));
                let (y0, y1) = (lo / c, (hi - 1) / c + 1);
                gl.enable(GL::SCISSOR_TEST);
                gl.scissor(0, y0 as i32, c as i32, (y1 - y0) as i32);
                self.draw();
                gl.disable(GL::SCISSOR_TEST);
            }
        }
        for q in targets {
            self.release(q);
        }
        self.release(w4);
        Ok(t)
    }

    /// Read the outputs back, then free this run's own textures (inputs and
    /// what is left of the intermediates); cached weights and index textures
    /// stay.
    fn read_outputs(&self, plan: &Plan, tex: Vec<Option<WebGlTexture>>) -> Result<Vec<Vec<f32>>> {
        let dims = |s: usize| plan.slot_dims[s];
        let outs = plan
            .outputs
            .iter()
            .map(|&s| {
                let (r, c) = dims(s);
                let mut v = self.read_back(self.tex(&tex, s)?, c, r)?;
                v.truncate(plan.slot_len[s]);
                Ok(v)
            })
            .collect();
        self.free_run_textures(tex);
        outs
    }

    /// Free a run's own textures (inputs, intermediates, outputs); the
    /// cached weights and index textures stay.
    fn free_run_textures(&self, tex: Vec<Option<WebGlTexture>>) {
        let cache = self.leaf_cache.borrow();
        for (slot, t) in tex.into_iter().enumerate() {
            if let Some(t) = t
                && !cache.contains_key(&slot)
            {
                self.release(t);
            }
        }
    }

    fn tex<'a>(&self, tex: &'a [Option<WebGlTexture>], slot: usize) -> Result<&'a WebGlTexture> {
        tex[slot]
            .as_ref()
            .ok_or_else(|| WebglError(format!("slot {slot} not produced")))
    }

    fn set_i32(&self, prog: &WebGlProgram, name: &str, v: i32) {
        if let Some(loc) = self.uniform(prog, name) {
            self.gl.uniform1i(Some(&loc), v);
        }
    }

    fn set_f32(&self, prog: &WebGlProgram, name: &str, v: f32) {
        if let Some(loc) = self.uniform(prog, name) {
            self.gl.uniform1f(Some(&loc), v);
        }
    }
}
