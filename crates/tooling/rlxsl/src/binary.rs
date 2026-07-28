// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Binary elementwise op manifest** — one definition per [`BinaryOp`], many
//! backends. Sibling of the unary-activation manifest in the crate root: the
//! standalone `binary` kernel historically re-expressed the same per-op scalar
//! math by hand in WGSL, CUDA C, MSL, GLSL and OpenCL-C, and they had *drifted*:
//!
//! * **Pow.** The CPU oracle is Rust `f32::powf`, which returns a signed result
//!   for a negative base with an integer exponent (`(-2)³ = -8`) and `NaN` only
//!   for a negative base with a genuinely fractional exponent. CUDA `powf` and
//!   OpenCL `pow` already match that. WGSL/GLSL/MSL `pow` is `exp(b·log(a))`,
//!   which is `NaN` for *any* negative base — so bare `pow(a,b)` there was wrong
//!   (e.g. `x³` on signed x in VITS/GELU). Only rlx-vulkan had the fix; wgpu and
//!   Metal shipped the bug. This manifest emits the native primitive where it is
//!   correct and the sign-corrected form where it is not.
//! * **Bitwise / shift width.** The CPU oracle casts through `i64`; CUDA matched
//!   (`long long`) but oneAPI/wgpu/Vulkan used 32-bit. This manifest uses 64-bit
//!   where the language has it (CUDA/OpenCL/MSL) and 32-bit on WGSL/GLSL, which
//!   have no 64-bit integer — the one unavoidable narrowing, and it only bites
//!   values that don't fit in `i32`.
//!
//! [`eval_binary`] is the interpreter oracle (checked against the CPU backend in
//! `rlx-runtime`), and the per-language [`Lang`] emitters render the same op set.

use crate::Lang;
use rlx_ir::op::BinaryOp;

/// Evaluate `op(a, b)` with the semantics the CPU backend defines (Rust
/// `powf`/`%`/`atan2`, `i64` bitwise) — the oracle the emitted kernels target.
pub fn eval_binary(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => a.powf(b),
        BinaryOp::Mod => a % b, // Rust `%` == C `fmod` (sign of dividend)
        BinaryOp::BitAnd => ((a as i64) & (b as i64)) as f32,
        BinaryOp::BitOr => ((a as i64) | (b as i64)) as f32,
        BinaryOp::BitXor => ((a as i64) ^ (b as i64)) as f32,
        BinaryOp::Shl => ((a as i64) << (b as i64)) as f32,
        BinaryOp::Shr => ((a as i64) >> (b as i64)) as f32,
        BinaryOp::Atan2 => a.atan2(b),
    }
}

// ── Per-language spellings ───────────────────────────────────────────────────

/// The widest signed integer the language has, for bitwise/shift reinterpret.
/// CUDA/OpenCL/MSL get 64-bit (matches the CPU oracle); WGSL/GLSL have only 32.
fn int_cast(e: &str, lang: Lang) -> String {
    match lang {
        Lang::Cuda => format!("(long long)({e})"),
        Lang::OpenCl => format!("(long)({e})"),
        Lang::Msl => format!("long({e})"),
        Lang::Wgsl => format!("i32({e})"),
        Lang::Glsl => format!("int({e})"),
    }
}

/// Cast an integer expression back to `float`.
fn float_cast(e: &str, lang: Lang) -> String {
    match lang {
        Lang::Cuda | Lang::OpenCl => format!("(float)({e})"),
        Lang::Msl => format!("float({e})"),
        Lang::Wgsl => format!("f32({e})"),
        Lang::Glsl => format!("float({e})"),
    }
}

/// The shift-amount operand. WGSL requires an unsigned RHS for `<<`/`>>`.
fn shift_amt(e: &str, lang: Lang) -> String {
    match lang {
        Lang::Wgsl => format!("u32({e})"),
        _ => int_cast(e, lang),
    }
}

fn fmax_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Cuda => "fmaxf",
        Lang::Msl | Lang::OpenCl => "fmax",
        Lang::Wgsl | Lang::Glsl => "max",
    }
}
fn fmin_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Cuda => "fminf",
        Lang::Msl | Lang::OpenCl => "fmin",
        Lang::Wgsl | Lang::Glsl => "min",
    }
}
fn atan2_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Cuda => "atan2f",
        Lang::Glsl => "atan", // GLSL's 2-arg `atan` is atan2
        Lang::Wgsl | Lang::Msl | Lang::OpenCl => "atan2",
    }
}

/// `fmod(a, b)` — C `fmod` semantics (sign of dividend, == Rust `%`).
fn mod_expr(lang: Lang) -> String {
    match lang {
        Lang::Cuda => "fmodf(a, b)".to_string(),
        Lang::Msl | Lang::OpenCl => "fmod(a, b)".to_string(),
        // No `fmod` builtin: the truncated remainder reproduces it exactly.
        Lang::Wgsl | Lang::Glsl => "a - b * trunc(a / b)".to_string(),
    }
}

/// `pow(a, b)` matching Rust `f32::powf`. CUDA/OpenCL have a native primitive
/// with the right negative-base semantics; WGSL/MSL/GLSL `pow` is `NaN` for a
/// negative base, so emit the sign-corrected form (`(stmts, "_pw")`).
fn pow_expr(lang: Lang) -> (Vec<String>, String) {
    match lang {
        Lang::Cuda => (Vec::new(), "powf(a, b)".to_string()),
        Lang::OpenCl => (Vec::new(), "pow(a, b)".to_string()),
        Lang::Wgsl => (
            vec![
                "var _pw: f32;".to_string(),
                "let _rb = round(b);".to_string(),
                "if (a >= 0.0 || abs(b - _rb) >= 1e-4) { _pw = pow(a, b); }".to_string(),
                "else { let _pm = pow(abs(a), _rb); _pw = select(_pm, -_pm, (i32(_rb) & 1) != 0); }"
                    .to_string(),
            ],
            "_pw".to_string(),
        ),
        // MSL / GLSL: same algorithm with C-family syntax + ternary.
        Lang::Msl | Lang::Glsl => (
            vec![
                "float _pw;".to_string(),
                "float _rb = round(b);".to_string(),
                "if (a >= 0.0 || abs(b - _rb) >= 1e-4) { _pw = pow(a, b); }".to_string(),
                "else { float _pm = pow(abs(a), _rb); _pw = ((int(_rb) & 1) != 0) ? -_pm : _pm; }"
                    .to_string(),
            ],
            "_pw".to_string(),
        ),
    }
}

/// The scalar math for `op` in `lang` as `(statements, expr)` — the building
/// block for the `rlx_binary_apply` switch. `statements` are `let`/`float`
/// temporaries (only Pow needs them, on WGSL/MSL/GLSL); `expr` is the value.
pub fn emit_binary(op: BinaryOp, lang: Lang) -> (Vec<String>, String) {
    let simple = |e: String| (Vec::new(), e);
    let bit = |sym: &str| {
        float_cast(
            &format!("{} {sym} {}", int_cast("a", lang), int_cast("b", lang)),
            lang,
        )
    };
    match op {
        BinaryOp::Add => simple("a + b".to_string()),
        BinaryOp::Sub => simple("a - b".to_string()),
        BinaryOp::Mul => simple("a * b".to_string()),
        BinaryOp::Div => simple("a / b".to_string()),
        BinaryOp::Max => simple(format!("{}(a, b)", fmax_name(lang))),
        BinaryOp::Min => simple(format!("{}(a, b)", fmin_name(lang))),
        BinaryOp::Pow => pow_expr(lang),
        BinaryOp::Mod => simple(mod_expr(lang)),
        BinaryOp::BitAnd => simple(bit("&")),
        BinaryOp::BitOr => simple(bit("|")),
        BinaryOp::BitXor => simple(bit("^")),
        BinaryOp::Shl => simple(float_cast(
            &format!("{} << {}", int_cast("a", lang), shift_amt("b", lang)),
            lang,
        )),
        BinaryOp::Shr => simple(float_cast(
            &format!("{} >> {}", int_cast("a", lang), shift_amt("b", lang)),
            lang,
        )),
        BinaryOp::Atan2 => simple(format!("{}(a, b)", atan2_name(lang))),
    }
}

// ── Emitters ─────────────────────────────────────────────────────────────────

/// Shared body of the `*_binary_module` emitters: a `switch (op)` over every
/// [`BinaryOp`] in opcode order, each `case` returning the op's value.
fn binary_switch(lang: Lang, signature: &str, default_close: &str) -> String {
    let mut s = String::from(
        "// @generated by rlxsl::binary — do not edit by hand.\n\
         // Binary-op math is defined once in rlxsl::binary::emit_binary.\n",
    );
    s.push_str(signature);
    for op in BinaryOp::ALL {
        let id = op.opcode();
        let (stmts, expr) = emit_binary(op, lang);
        let pre = if stmts.is_empty() {
            String::new()
        } else {
            format!("{} ", stmts.join(" "))
        };
        s.push_str(&format!("        case {id}u: {{ {pre}return {expr}; }} // {op:?}\n"));
    }
    s.push_str(default_close);
    s
}

/// WGSL `rlx_binary_apply(op, a, b)` — single source for wgpu's `binary` kernel.
pub fn wgsl_binary_module() -> String {
    binary_switch(
        Lang::Wgsl,
        "fn rlx_binary_apply(op: u32, a: f32, b: f32) -> f32 {\n    switch op {\n",
        "        default: { return 0.0; }\n    }\n}\n",
    )
}

/// CUDA `__device__` binary dispatch generated from the manifest.
pub fn cuda_binary_module() -> String {
    binary_switch(
        Lang::Cuda,
        "__device__ __forceinline__ float rlx_binary_apply(unsigned int op, float a, float b) {\n    switch (op) {\n",
        "        default: return 0.0f;\n    }\n}\n",
    )
}

/// MSL binary dispatch generated from the manifest.
pub fn msl_binary_module() -> String {
    binary_switch(
        Lang::Msl,
        "inline float rlx_binary_apply(uint op, float a, float b) {\n    switch (op) {\n",
        "        default: return 0.0f;\n    }\n}\n",
    )
}

/// GLSL (native Vulkan) binary dispatch generated from the manifest.
pub fn glsl_binary_module() -> String {
    binary_switch(
        Lang::Glsl,
        "float rlx_binary_apply(uint op, float a, float b) {\n    switch (op) {\n",
        "        default: return 0.0;\n    }\n}\n",
    )
}

/// OpenCL-C (Intel oneAPI) binary dispatch generated from the manifest.
pub fn opencl_binary_module() -> String {
    binary_switch(
        Lang::OpenCl,
        "inline float rlx_binary_apply(uint op, float a, float b) {\n    switch (op) {\n",
        "        default: return 0.0f;\n    }\n}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent Rust reference (the CPU-backend semantics), restated so the
    /// interpreter is checked against something other than itself.
    fn reference(op: BinaryOp, a: f32, b: f32) -> f32 {
        match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Max => a.max(b),
            BinaryOp::Min => a.min(b),
            BinaryOp::Pow => a.powf(b),
            BinaryOp::Mod => a % b,
            BinaryOp::BitAnd => ((a as i64) & (b as i64)) as f32,
            BinaryOp::BitOr => ((a as i64) | (b as i64)) as f32,
            BinaryOp::BitXor => ((a as i64) ^ (b as i64)) as f32,
            BinaryOp::Shl => ((a as i64) << (b as i64)) as f32,
            BinaryOp::Shr => ((a as i64) >> (b as i64)) as f32,
            BinaryOp::Atan2 => a.atan2(b),
        }
    }

    #[test]
    fn eval_matches_reference_including_negative_base_pow() {
        // Operand ranges chosen per op so the *oracle itself* is well-defined:
        // shifts need a non-negative, < 63 amount (Rust panics otherwise, as
        // would the CPU backend); the arithmetic ops span sign and fraction.
        let arith = [-3.0f32, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0];
        let ints = [-8.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0, 255.0];
        let shifts = [0.0f32, 1.0, 2.0, 3.0, 5.0, 8.0];
        for &op in &BinaryOp::ALL {
            let (avs, bvs): (&[f32], &[f32]) = match op {
                BinaryOp::Shl | BinaryOp::Shr => (&ints, &shifts),
                BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => (&ints, &ints),
                _ => (&arith, &arith),
            };
            for &a in avs {
                for &b in bvs {
                    let got = eval_binary(op, a, b);
                    let want = reference(op, a, b);
                    if got.is_nan() {
                        assert!(want.is_nan(), "{op:?}({a},{b}): eval NaN but ref {want}");
                    } else {
                        let err = (got - want).abs() / want.abs().max(1.0);
                        assert!(err < 1e-6, "{op:?}({a},{b}): eval={got} ref={want}");
                    }
                }
            }
        }
        // The headline drift: negative base, integer exponent stays signed/real.
        assert_eq!(eval_binary(BinaryOp::Pow, -2.0, 2.0), 4.0);
        assert_eq!(eval_binary(BinaryOp::Pow, -2.0, 3.0), -8.0);
        assert!(eval_binary(BinaryOp::Pow, -2.0, 2.5).is_nan());
    }

    #[test]
    fn every_language_emits_all_ops() {
        for lang in [
            Lang::Wgsl,
            Lang::Cuda,
            Lang::Msl,
            Lang::Glsl,
            Lang::OpenCl,
        ] {
            let m = match lang {
                Lang::Wgsl => wgsl_binary_module(),
                Lang::Cuda => cuda_binary_module(),
                Lang::Msl => msl_binary_module(),
                Lang::Glsl => glsl_binary_module(),
                Lang::OpenCl => opencl_binary_module(),
            };
            assert_eq!(
                m.matches("case ").count(),
                BinaryOp::ALL.len(),
                "{lang:?} case count"
            );
            assert!(m.contains("rlx_binary_apply") && m.contains("default"));
        }
        // Native pow where correct; sign-corrected form where `pow` NaNs.
        assert!(cuda_binary_module().contains("powf(a, b)"));
        assert!(opencl_binary_module().contains("pow(a, b)"));
        assert!(wgsl_binary_module().contains("select(_pm, -_pm"));
        assert!(msl_binary_module().contains("? -_pm : _pm"));
        // 64-bit bitwise where available; 32-bit only on WGSL/GLSL.
        assert!(cuda_binary_module().contains("(long long)"));
        assert!(opencl_binary_module().contains("(long)"));
        assert!(wgsl_binary_module().contains("i32(a)"));
    }
}
