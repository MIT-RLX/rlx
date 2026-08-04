// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! WebGL2 render-to-texture executor for a [`Plan`].
//!
//! Every tensor is an `RGBA32F` texture (value in the `R` channel); every op
//! is a fragment shader that writes one output element per fragment. The GLSL
//! mirrors [`crate::exec_cpu`] one-to-one (same op/activation codes), so the
//! numerics are the ones the native parity test verifies — only the GL
//! plumbing here needs in-browser validation. WebGL2 `readPixels` is
//! synchronous, so the whole path is synchronous.

use crate::plan::{Act, Bin, Cmp, LeafSource, Plan, Red, Step};
use crate::{Result, WebglError};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{
    WebGl2RenderingContext as GL, WebGlProgram, WebGlShader, WebGlTexture, WebGlUniformLocation,
};

const HEAD: &str = "#version 300 es\nprecision highp float; precision highp int;\n";

// Activation forward/derivative — must match `exec_cpu::act_f` / `act_df`.
// Codes: 0 Relu 1 Neg 2 Exp 3 Log 4 Sqrt 5 Rsqrt 6 Sigmoid 7 Tanh 8 Abs 9 Sin 10 Cos 11 Silu 12 Recip.
const ACT_FNS: &str = r#"
float sigmoidf(float x) { return 1.0 / (1.0 + exp(-x)); }
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

const FS_MATMUL: &str = r#"uniform highp sampler2D A; uniform highp sampler2D B; uniform int uK;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy); // x=col, y=row
    float s = 0.0;
    for (int l = 0; l < uK; l++) {
        s += texelFetch(A, ivec2(l, p.y), 0).r * texelFetch(B, ivec2(p.x, l), 0).r;
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
uniform int uFanin; uniform int uSrcCols; uniform int uOutCols; uniform int uOp;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int oi = p.y * uOutCols + p.x;
    float acc = uOp == 2 ? -3.0e38 : uOp == 3 ? 3.0e38 : uOp == 4 ? 1.0 : 0.0;
    float count = 0.0;
    for (int j = 0; j < uFanin; j++) {
        int g = texelFetch(GROUPS, ivec2(j, oi), 0).r;
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

const FS_SOFTMAX: &str = r#"uniform highp sampler2D A; uniform int uCols;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float m = -3.0e38;
    for (int c = 0; c < uCols; c++) m = max(m, texelFetch(A, ivec2(c, p.y), 0).r);
    float sum = 0.0;
    for (int c = 0; c < uCols; c++) sum += exp(texelFetch(A, ivec2(c, p.y), 0).r - m);
    o = vec4(exp(texelFetch(A, p, 0).r - m) / sum, 0.0, 0.0, 1.0);
}"#;

const FS_LAYERNORM: &str = r#"uniform highp sampler2D X; uniform highp sampler2D G; uniform highp sampler2D B;
uniform int uCols; uniform float uEps;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float mean = 0.0;
    for (int i = 0; i < uCols; i++) mean += texelFetch(X, ivec2(i, p.y), 0).r;
    mean /= float(uCols);
    float var = 0.0;
    for (int i = 0; i < uCols; i++) {
        float d = texelFetch(X, ivec2(i, p.y), 0).r - mean;
        var += d * d;
    }
    var /= float(uCols);
    float norm = (texelFetch(X, p, 0).r - mean) * inversesqrt(var + uEps);
    float g = texelFetch(G, ivec2(p.x, 0), 0).r;
    float b = texelFetch(B, ivec2(p.x, 0), 0).r;
    o = vec4(norm * g + b, 0.0, 0.0, 1.0);
}"#;

const FS_RMSNORM: &str = r#"uniform highp sampler2D X; uniform highp sampler2D G; uniform highp sampler2D B;
uniform int uCols; uniform float uEps;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float n_inv = 1.0 / float(uCols);
    // Two-pass: mean(x²) = mean((x-mean)^2) + mean^2. Matches the CPU oracle.
    float sum = 0.0;
    for (int i = 0; i < uCols; i++) { sum += texelFetch(X, ivec2(i, p.y), 0).r; }
    float mean = sum * n_inv;
    float dev = 0.0;
    for (int i = 0; i < uCols; i++) {
        float d = texelFetch(X, ivec2(i, p.y), 0).r - mean;
        dev += d * d;
    }
    float norm = texelFetch(X, p, 0).r * inversesqrt(dev * n_inv + mean * mean + uEps);
    float g = texelFetch(G, ivec2(p.x, 0), 0).r;
    float b = texelFetch(B, ivec2(p.x, 0), 0).r;
    o = vec4(norm * g + b, 0.0, 0.0, 1.0);
}"#;

const FS_ARGREDUCE: &str = r#"uniform highp sampler2D SRC; uniform highp isampler2D GROUPS;
uniform int uFanin; uniform int uSrcCols; uniform int uOutCols; uniform int uIsMax;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int oi = p.y * uOutCols + p.x;
    float best = uIsMax == 1 ? -3.0e38 : 3.0e38;
    int bj = 0;
    for (int j = 0; j < uFanin; j++) {
        int g = texelFetch(GROUPS, ivec2(j, oi), 0).r;
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
uniform int uTableCols; uniform int uIdxCols; uniform int uAxisStride;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int wi = texelFetch(WHICH, p, 0).r;
    float idxf = texelFetch(IDXT, ivec2(wi % uIdxCols, wi / uIdxCols), 0).r;
    int ix = int(idxf + 0.5);
    int tlin = texelFetch(BASE, p, 0).r + ix * uAxisStride;
    o = vec4(texelFetch(TABLE, ivec2(tlin % uTableCols, tlin / uTableCols), 0).r, 0.0, 0.0, 1.0);
}"#;

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
    complex_cast: WebGlProgram,
    binary_c64: WebGlProgram,
}

fn err<T: std::fmt::Debug>(ctx: &str, e: T) -> WebglError {
    WebglError(format!("{ctx}: {e:?}"))
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

        let fbo = gl
            .create_framebuffer()
            .ok_or_else(|| WebglError("create_framebuffer".into()))?;

        // Activation shaders need the act helper functions; the rest don't.
        let plain = |body: &str| Self::program(&gl, &format!("{HEAD}{body}"));
        let with_act = |body: &str| Self::program(&gl, &format!("{HEAD}{ACT_FNS}{body}"));

        Ok(Self {
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
            complex_cast: plain(FS_COMPLEX_CAST)?,
            binary_c64: plain(FS_BINARY_C64)?,
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
        self.gl.get_uniform_location(p, name)
    }

    fn data_texture(&self, w: usize, h: usize, data: &[f32]) -> Result<WebGlTexture> {
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        let mut rgba = vec![0f32; w * h * 4];
        for (i, &v) in data.iter().enumerate() {
            rgba[i * 4] = v;
        }
        // SAFETY: the view is consumed by tex_image_2d before any JS alloc.
        unsafe {
            let view = js_sys::Float32Array::view(&rgba);
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
                GL::TEXTURE_2D,
                0,
                GL::RGBA32F as i32,
                w as i32,
                h as i32,
                0,
                GL::RGBA,
                GL::FLOAT,
                Some(&view),
            )
            .map_err(|e| err("tex_image_2d(rgba32f)", e))?;
        }
        self.set_nearest();
        Ok(tex)
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
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            GL::TEXTURE_2D,
            0,
            GL::RGBA32F as i32,
            w as i32,
            h as i32,
            0,
            GL::RGBA,
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
        if gl.check_framebuffer_status(GL::FRAMEBUFFER) != GL::FRAMEBUFFER_COMPLETE {
            return Err(WebglError("framebuffer incomplete".into()));
        }
        gl.viewport(0, 0, w as i32, h as i32);
        Ok(())
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

    fn read_back(&self, tex: &WebGlTexture, w: usize, h: usize) -> Result<Vec<f32>> {
        let gl = &self.gl;
        self.bind_target(tex, w, h)?;
        let mut rgba = vec![0f32; w * h * 4];
        unsafe {
            let view = js_sys::Float32Array::view_mut_raw(rgba.as_mut_ptr(), rgba.len());
            gl.read_pixels_with_opt_array_buffer_view(
                0,
                0,
                w as i32,
                h as i32,
                GL::RGBA,
                GL::FLOAT,
                Some(&view),
            )
            .map_err(|e| err("read_pixels", e))?;
        }
        Ok((0..w * h).map(|i| rgba[i * 4]).collect())
    }

    /// Start a render pass into a fresh target of size `w×h` with `prog` bound.
    fn begin(&self, prog: &WebGlProgram, w: usize, h: usize) -> Result<WebGlTexture> {
        let t = self.target_texture(w, h)?;
        self.bind_target(&t, w, h)?;
        self.gl.use_program(Some(prog));
        Ok(t)
    }

    /// Execute `plan` with the given named inputs/params on the GPU.
    pub fn run(&self, plan: &Plan, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        let input_map: HashMap<&str, &[f32]> = inputs.iter().copied().collect();
        let mut tex: Vec<Option<WebGlTexture>> = (0..plan.slot_len.len()).map(|_| None).collect();
        let dims = |s: usize| plan.slot_dims[s];

        for step in &plan.steps {
            match step {
                Step::Leaf { out, src } => {
                    let data: Vec<f32> = match src {
                        LeafSource::Input(name) | LeafSource::Param(name) => input_map
                            .get(name.as_str())
                            .map(|d| d.to_vec())
                            .ok_or_else(|| WebglError(format!("missing input/param '{name}'")))?,
                        LeafSource::Const(d) => d.clone(),
                    };
                    let (r, c) = dims(*out);
                    tex[*out] = Some(self.data_texture(c, r, &data)?);
                }
                Step::Unary { out, a, act } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.unary, c, r)?;
                    self.bind_tex_unit(&self.unary, "A", 0, self.tex(&tex, *a)?);
                    self.set_i32(&self.unary, "uAct", act_code(*act));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::ActBack { out, x, dy, act } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.actback, c, r)?;
                    self.bind_tex_unit(&self.actback, "X", 0, self.tex(&tex, *x)?);
                    self.bind_tex_unit(&self.actback, "DY", 1, self.tex(&tex, *dy)?);
                    self.set_i32(&self.actback, "uAct", act_code(*act));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Binary { out, a, b, op } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.binary, c, r)?;
                    self.bind_tex_unit(&self.binary, "A", 0, self.tex(&tex, *a)?);
                    self.bind_tex_unit(&self.binary, "B", 1, self.tex(&tex, *b)?);
                    self.set_i32(&self.binary, "uOp", bin_code(*op));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Compare { out, a, b, cmp } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.compare, c, r)?;
                    self.bind_tex_unit(&self.compare, "A", 0, self.tex(&tex, *a)?);
                    self.bind_tex_unit(&self.compare, "B", 1, self.tex(&tex, *b)?);
                    self.set_i32(&self.compare, "uOp", cmp_code(*cmp));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Where { out, cond, a, b } => {
                    let (r, c) = dims(*out);
                    let t = self.begin(&self.where_, c, r)?;
                    self.bind_tex_unit(&self.where_, "C", 0, self.tex(&tex, *cond)?);
                    self.bind_tex_unit(&self.where_, "A", 1, self.tex(&tex, *a)?);
                    self.bind_tex_unit(&self.where_, "B", 2, self.tex(&tex, *b)?);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::MatMul { out, a, b, m, k, n } => {
                    let t = self.begin(&self.matmul, *n, *m)?;
                    self.bind_tex_unit(&self.matmul, "A", 0, self.tex(&tex, *a)?);
                    self.bind_tex_unit(&self.matmul, "B", 1, self.tex(&tex, *b)?);
                    self.set_i32(&self.matmul, "uK", *k as i32);
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Gather { out, src, idx } => {
                    let (r, c) = dims(*out);
                    let (_sr, sc) = dims(*src);
                    let idx_i32: Vec<i32> = idx.iter().map(|&v| v as i32).collect();
                    let idx_tex = self.int_texture(c, r, &idx_i32)?;
                    let t = self.begin(&self.gather, c, r)?;
                    self.bind_tex_unit(&self.gather, "SRC", 0, self.tex(&tex, *src)?);
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
                    let n_out = r * c;
                    let groups_i32: Vec<i32> = groups.iter().map(|&v| v as i32).collect();
                    let g_tex = self.int_texture(*fanin, n_out, &groups_i32)?;
                    let t = self.begin(&self.reduce, c, r)?;
                    self.bind_tex_unit(&self.reduce, "SRC", 0, self.tex(&tex, *src)?);
                    self.bind_tex_unit(&self.reduce, "GROUPS", 1, &g_tex);
                    self.set_i32(&self.reduce, "uFanin", *fanin as i32);
                    self.set_i32(&self.reduce, "uSrcCols", sc as i32);
                    self.set_i32(&self.reduce, "uOutCols", c as i32);
                    self.set_i32(&self.reduce, "uOp", red_code(*op));
                    self.draw();
                    tex[*out] = Some(t);
                }
                Step::Softmax { out, a, rows, cols } => {
                    let t = self.begin(&self.softmax, *cols, *rows)?;
                    self.bind_tex_unit(&self.softmax, "A", 0, self.tex(&tex, *a)?);
                    self.set_i32(&self.softmax, "uCols", *cols as i32);
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
                    let t = self.begin(&self.layernorm, *cols, *rows)?;
                    self.bind_tex_unit(&self.layernorm, "X", 0, self.tex(&tex, *x)?);
                    self.bind_tex_unit(&self.layernorm, "G", 1, self.tex(&tex, *gamma)?);
                    self.bind_tex_unit(&self.layernorm, "B", 2, self.tex(&tex, *beta)?);
                    self.set_i32(&self.layernorm, "uCols", *cols as i32);
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
                    let t = self.begin(&self.rmsnorm, *cols, *rows)?;
                    self.bind_tex_unit(&self.rmsnorm, "X", 0, self.tex(&tex, *x)?);
                    self.bind_tex_unit(&self.rmsnorm, "G", 1, self.tex(&tex, *gamma)?);
                    self.bind_tex_unit(&self.rmsnorm, "B", 2, self.tex(&tex, *beta)?);
                    self.set_i32(&self.rmsnorm, "uCols", *cols as i32);
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
                    let g_i32: Vec<i32> = groups.iter().map(|&v| v as i32).collect();
                    let g_tex = self.int_texture(*fanin, r * c, &g_i32)?;
                    let t = self.begin(&self.argreduce, c, r)?;
                    self.bind_tex_unit(&self.argreduce, "SRC", 0, self.tex(&tex, *src)?);
                    self.bind_tex_unit(&self.argreduce, "GROUPS", 1, &g_tex);
                    self.set_i32(&self.argreduce, "uFanin", *fanin as i32);
                    self.set_i32(&self.argreduce, "uSrcCols", sc as i32);
                    self.set_i32(&self.argreduce, "uOutCols", c as i32);
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
                } => {
                    let (r, c) = dims(*out);
                    let (_tr, tc) = dims(*table);
                    let (_ir, ic) = dims(*indices);
                    let which_i: Vec<i32> = which.iter().map(|&v| v as i32).collect();
                    let base_i: Vec<i32> = base.iter().map(|&v| v as i32).collect();
                    let which_tex = self.int_texture(c, r, &which_i)?;
                    let base_tex = self.int_texture(c, r, &base_i)?;
                    let t = self.begin(&self.gather_rt, c, r)?;
                    self.bind_tex_unit(&self.gather_rt, "TABLE", 0, self.tex(&tex, *table)?);
                    self.bind_tex_unit(&self.gather_rt, "IDXT", 1, self.tex(&tex, *indices)?);
                    self.bind_tex_unit(&self.gather_rt, "WHICH", 2, &which_tex);
                    self.bind_tex_unit(&self.gather_rt, "BASE", 3, &base_tex);
                    self.set_i32(&self.gather_rt, "uTableCols", tc as i32);
                    self.set_i32(&self.gather_rt, "uIdxCols", ic as i32);
                    self.set_i32(&self.gather_rt, "uAxisStride", *axis_stride as i32);
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
                    self.bind_tex_unit(&self.complex_cast, "SRC", 0, self.tex(&tex, *src)?);
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
                    self.bind_tex_unit(&self.binary_c64, "A", 0, self.tex(&tex, *a)?);
                    self.bind_tex_unit(&self.binary_c64, "B", 1, self.tex(&tex, *b)?);
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
            }
        }

        plan.outputs
            .iter()
            .map(|&s| {
                let (r, c) = dims(s);
                self.read_back(self.tex(&tex, s)?, c, r)
            })
            .collect()
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
