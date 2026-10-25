// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! String → enum for every string-selected option on the binding surface.
//!
//! Each parser is named after the op-table argument kind that calls it
//! (`style: rope_style` → [`rope_style`]), so declaring a parameter of an
//! existing kind is the whole wiring. Matching ignores case, `_`, `-` and
//! spaces, so `"gptj"`, `"GPT-J"` and `"gpt_j"` read the same. `activation`
//! and `binop` read the IR's own `ALL` tables, so a variant the IR gains is
//! reachable by name without an edit here.
//!
//! [`ALL_KINDS`] lists the accepted spellings per kind, for the generators
//! that emit typed enums (Swift, `.pyi` literals).

use rlx_ir::fft::FftNorm;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, PadMode, QrPart, ReduceOp, SvdPart};
use rlx_ir::ops::spectral::WindowKind;
use rlx_ir::ops::upsample::InterpMode;
use rlx_ir::ops::vq::VqMetric;
use rlx_ir::{
    AdaNormKind, DType, FirMode, QuantScheme, RopeStyle, ScaleLayout, ScaledFormat,
    ScatterNdReduction, SpdMatFn,
};

use crate::{ApiError, ApiResult};

/// Lowercase with `_` / `-` / spaces dropped — the comparison key.
pub fn key(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '_' | '-' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

fn unknown<T>(what: &str, got: &str, expected: &str) -> ApiResult<T> {
    Err(ApiError::type_(format!(
        "unknown {what} '{got}' ({expected})"
    )))
}

// ── dtypes ──────────────────────────────────────────────────

pub fn dtype(s: &str) -> ApiResult<DType> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => DType::F32,
        "f16" | "float16" | "half" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f64" | "float64" | "double" => DType::F64,
        "i8" | "int8" => DType::I8,
        "u8" | "uint8" => DType::U8,
        "i16" | "int16" => DType::I16,
        "i32" | "int32" => DType::I32,
        "u32" | "uint32" => DType::U32,
        "i64" | "int64" => DType::I64,
        "bool" => DType::Bool,
        "c64" | "complex64" => DType::C64,
        "c128" | "complex128" => DType::C128,
        other => {
            return Err(ApiError::type_(format!(
                "unknown dtype '{other}' (f32, f16, bf16, i32, …)"
            )));
        }
    })
}

/// The canonical spelling [`dtype`] reads back.
pub fn dtype_label(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F64 => "f64",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C64 => "c64",
        DType::C128 => "c128",
    }
}

// ── elementwise ─────────────────────────────────────────────

/// Any [`Activation`] by its variant name — `"gelu_approx"`, `"hardSwish"`,
/// `"log-sigmoid"` all match.
pub fn activation(s: &str) -> ApiResult<Activation> {
    let want = key(s);
    Activation::ALL
        .into_iter()
        .find(|a| key(&format!("{a:?}")) == want)
        .ok_or_else(|| {
            let names: Vec<String> = Activation::ALL.iter().map(|a| format!("{a:?}")).collect();
            ApiError::type_(format!("unknown activation '{s}' ({})", names.join(", ")))
        })
}

/// An optional activation: `None`, `""`, `"none"`, `"identity"` → no activation.
pub fn opt_activation(s: Option<&str>) -> ApiResult<Option<Activation>> {
    match s {
        None => Ok(None),
        Some(a) if matches!(key(a).as_str(), "" | "none" | "identity" | "linear") => Ok(None),
        Some(a) => activation(a).map(Some),
    }
}

/// Any [`BinaryOp`] by variant name (`"atan2"`, `"bit_and"`, `"shl"`, …) or
/// operator symbol (`% ** & | ^ << >>`, plus `+ - * /`).
pub fn binop(s: &str) -> ApiResult<BinaryOp> {
    let symbol = match s.trim() {
        "+" => Some(BinaryOp::Add),
        "-" => Some(BinaryOp::Sub),
        "*" => Some(BinaryOp::Mul),
        "/" => Some(BinaryOp::Div),
        "%" => Some(BinaryOp::Mod),
        "**" => Some(BinaryOp::Pow),
        "&" => Some(BinaryOp::BitAnd),
        "|" => Some(BinaryOp::BitOr),
        "^" => Some(BinaryOp::BitXor),
        "<<" => Some(BinaryOp::Shl),
        ">>" => Some(BinaryOp::Shr),
        _ => None,
    };
    let want = key(s);
    symbol
        .or_else(|| {
            BinaryOp::ALL
                .into_iter()
                .find(|op| key(&format!("{op:?}")) == want)
        })
        .ok_or_else(|| {
            let names: Vec<String> = BinaryOp::ALL.iter().map(|o| format!("{o:?}")).collect();
            ApiError::type_(format!("unknown binary op '{s}' ({})", names.join(", ")))
        })
}

pub fn cmp_op(s: &str) -> ApiResult<CmpOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "eq" | "==" => CmpOp::Eq,
        "ne" | "!=" => CmpOp::Ne,
        "lt" | "<" => CmpOp::Lt,
        "le" | "<=" => CmpOp::Le,
        "gt" | ">" => CmpOp::Gt,
        "ge" | ">=" => CmpOp::Ge,
        other => return Err(ApiError::type_(format!("unknown comparison '{other}'"))),
    })
}

pub fn reduce_op(s: &str) -> ApiResult<ReduceOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "sum" => ReduceOp::Sum,
        "mean" => ReduceOp::Mean,
        "max" => ReduceOp::Max,
        "min" => ReduceOp::Min,
        "prod" => ReduceOp::Prod,
        other => return Err(ApiError::type_(format!("unknown reduction '{other}'"))),
    })
}

// ── attention / position ────────────────────────────────────

/// `"none"`, `"causal"`, `"sliding:<N>"` — a mask the kernel synthesizes.
pub fn mask(s: &str) -> ApiResult<MaskKind> {
    let lower = s.trim().to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("sliding:") {
        return rest
            .parse::<usize>()
            .map(MaskKind::SlidingWindow)
            .map_err(|_| {
                ApiError::type_(format!(
                    "sliding mask needs a window: 'sliding:4096', got '{s}'"
                ))
            });
    }
    Ok(match lower.as_str() {
        "none" => MaskKind::None,
        "causal" => MaskKind::Causal,
        other => {
            return Err(ApiError::type_(format!(
                "unknown mask kind '{other}' (none, causal, sliding:<N>)"
            )));
        }
    })
}

/// `"neox"` (HF, the default) or `"gptj"` (adjacent pairs; GGUF Llama).
pub fn rope_style(s: &str) -> ApiResult<RopeStyle> {
    match key(s).as_str() {
        "neox" | "half" | "hf" => Ok(RopeStyle::NeoX),
        "gptj" | "interleaved" | "adjacent" | "gguf" => Ok(RopeStyle::GptJ),
        _ => unknown("RoPE style", s, "neox, gptj"),
    }
}

// ── layers ──────────────────────────────────────────────────

/// `"nearest"` or `"linear"`.
pub fn interp(s: &str) -> ApiResult<InterpMode> {
    match key(s).as_str() {
        "nearest" => Ok(InterpMode::Nearest),
        "linear" | "bilinear" | "trilinear" => Ok(InterpMode::Linear),
        _ => unknown("interpolation mode", s, "nearest, linear"),
    }
}

/// adaLN pre-normalization: `"layer_norm"` (DiT) or `"rms_norm"`.
pub fn ada_norm(s: &str) -> ApiResult<AdaNormKind> {
    match key(s).as_str() {
        "layernorm" | "ln" | "layer" => Ok(AdaNormKind::LayerNorm),
        "rmsnorm" | "rms" => Ok(AdaNormKind::RmsNorm),
        _ => unknown("adaLN norm", s, "layer_norm, rms_norm"),
    }
}

/// ONNX ScatterND `reduction`: `"none"`, `"add"`, `"mul"`, `"max"`, `"min"`.
pub fn scatter_red(s: &str) -> ApiResult<ScatterNdReduction> {
    match key(s).as_str() {
        "none" | "" => Ok(ScatterNdReduction::None),
        "add" | "sum" => Ok(ScatterNdReduction::Add),
        "mul" | "prod" => Ok(ScatterNdReduction::Mul),
        "max" => Ok(ScatterNdReduction::Max),
        "min" => Ok(ScatterNdReduction::Min),
        _ => unknown("scatter reduction", s, "none, add, mul, max, min"),
    }
}

/// `"constant"` (filled with `value`), `"reflect"`, `"replicate"`, `"circular"`.
pub fn pad_mode(s: &str, value: f32) -> ApiResult<PadMode> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "constant" | "zero" | "zeros" => PadMode::Constant(value),
        "reflect" => PadMode::Reflect,
        "replicate" | "edge" => PadMode::Replicate,
        "circular" | "wrap" => PadMode::Circular,
        other => {
            return Err(ApiError::type_(format!(
                "unknown pad mode '{other}' (constant, reflect, replicate, circular)"
            )));
        }
    })
}

/// `"l2"` or `"cosine"`.
pub fn vq_metric(s: &str) -> ApiResult<VqMetric> {
    match key(s).as_str() {
        "l2" | "euclidean" => Ok(VqMetric::L2),
        "cosine" | "cos" => Ok(VqMetric::Cosine),
        _ => unknown("VQ metric", s, "l2, cosine"),
    }
}

// ── linear algebra ──────────────────────────────────────────

pub fn qr_part(s: &str) -> ApiResult<QrPart> {
    match s.trim().to_ascii_lowercase().as_str() {
        "q" => Ok(QrPart::Q),
        "r" => Ok(QrPart::R),
        other => Err(ApiError::type_(format!(
            "qr part must be 'q' or 'r', got '{other}'"
        ))),
    }
}

pub fn svd_part(s: &str) -> ApiResult<SvdPart> {
    match s.trim().to_ascii_lowercase().as_str() {
        "u" => Ok(SvdPart::U),
        "s" => Ok(SvdPart::S),
        "vt" | "v" => Ok(SvdPart::Vt),
        other => Err(ApiError::type_(format!(
            "svd part must be 'u', 's' or 'vt', got '{other}'"
        ))),
    }
}

/// Batched SPD matrix function: `"logm"`, `"expm"`, `"sqrtm"`, `"invsqrtm"`.
pub fn spd_fn(s: &str) -> ApiResult<SpdMatFn> {
    match key(s).as_str() {
        "logm" | "log" => Ok(SpdMatFn::Logm),
        "expm" | "exp" => Ok(SpdMatFn::Expm),
        "sqrtm" | "sqrt" => Ok(SpdMatFn::Sqrtm),
        "invsqrtm" | "invsqrt" => Ok(SpdMatFn::Invsqrtm),
        _ => unknown("SPD matrix function", s, "logm, expm, sqrtm, invsqrtm"),
    }
}

// ── signal ──────────────────────────────────────────────────

/// `"backward"` (the default, unscaled forward), `"ortho"`, `"forward"`.
pub fn fft_norm(s: &str) -> ApiResult<FftNorm> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "backward" | "none" => FftNorm::Backward,
        "ortho" | "orthonormal" => FftNorm::Ortho,
        "forward" => FftNorm::Forward,
        other => {
            return Err(ApiError::type_(format!(
                "unknown FFT norm '{other}' (backward, ortho, forward)"
            )));
        }
    })
}

/// Analysis window: `"rectangular"`, `"hann"`, `"hamming"`, `"blackman"`.
pub fn window(s: &str) -> ApiResult<WindowKind> {
    match key(s).as_str() {
        "rectangular" | "boxcar" | "none" => Ok(WindowKind::Rectangular),
        "hann" | "hanning" => Ok(WindowKind::Hann),
        "hamming" => Ok(WindowKind::Hamming),
        "blackman" => Ok(WindowKind::Blackman),
        _ => unknown("window", s, "rectangular, hann, hamming, blackman"),
    }
}

/// NumPy convolution modes plus `"causal"` (`lfilter` with an FIR kernel).
pub fn fir_mode(s: &str) -> ApiResult<FirMode> {
    match key(s).as_str() {
        "full" => Ok(FirMode::Full),
        "same" => Ok(FirMode::Same),
        "valid" => Ok(FirMode::Valid),
        "causal" | "lfilter" => Ok(FirMode::Causal),
        _ => unknown("FIR mode", s, "full, same, valid, causal"),
    }
}

// ── quantization ────────────────────────────────────────────

/// Minifloat element format — `ScaledFormat`'s own parser (`"f8e4m3"`, …).
pub fn scaled_fmt(s: &str) -> ApiResult<ScaledFormat> {
    s.trim()
        .to_ascii_lowercase()
        .parse()
        .map_err(|e: String| ApiError::type_(format!("scaled format: {e}")))
}

/// Block-scale layout — `ScaleLayout`'s own parser (`"per_tensor"`, `"mx"`, …).
pub fn scale_layout(s: &str) -> ApiResult<ScaleLayout> {
    s.trim()
        .to_ascii_lowercase()
        .parse()
        .map_err(|e: String| ApiError::type_(format!("scale layout: {e}")))
}

/// A `DequantMatMul` weight scheme.
///
/// GGUF names (`"Q4_K"`, `"IQ2_XXS"`, …) go through `rlx-gguf`'s own table
/// (feature `gguf`). The non-GGUF schemes take their block parameters after a
/// colon: `"int8:32"`, `"int8_asym:32"`, `"int4:32"`, `"fp8e4m3"`, `"fp8e5m2"`,
/// `"nvfp4"`, `"mlx_affine:4:64"`, `"mlx_mxfp4:32"`, `"mlx_mxfp8:32"`,
/// `"mxfp4x2:32"`.
pub fn quant_scheme(s: &str) -> ApiResult<QuantScheme> {
    let mut parts = s.trim().split(':');
    let head = key(parts.next().unwrap_or(""));
    let nums: Vec<u32> = parts
        .map(|p| p.trim().parse::<u32>())
        .collect::<Result<_, _>>()
        .map_err(|_| ApiError::type_(format!("quant scheme '{s}': parameters must be integers")))?;
    let at = |i: usize, default: u32| nums.get(i).copied().unwrap_or(default);
    let scheme = match head.as_str() {
        "int8" | "int8block" => Some(QuantScheme::Int8Block {
            block_size: at(0, 32),
        }),
        "int8asym" | "int8blockasym" => Some(QuantScheme::Int8BlockAsym {
            block_size: at(0, 32),
        }),
        "int4" | "int4block" => Some(QuantScheme::Int4Block {
            block_size: at(0, 32),
        }),
        "fp8e4m3" => Some(QuantScheme::Fp8E4m3),
        "fp8e5m2" => Some(QuantScheme::Fp8E5m2),
        "nvfp4" | "nvfp4block" => Some(QuantScheme::Nvfp4Block),
        "mlxaffine" => Some(QuantScheme::MlxAffine {
            bits: at(0, 4) as u8,
            group_size: at(1, 64),
        }),
        "mlxmxfp4" => Some(QuantScheme::MlxMxfp4 {
            group_size: at(0, 32),
        }),
        "mlxmxfp8" => Some(QuantScheme::MlxMxfp8 {
            group_size: at(0, 32),
        }),
        "mxfp4x2" | "mxfp4x2block" => Some(QuantScheme::MxFp4x2Block {
            group_size: at(0, 32),
        }),
        _ => None,
    };
    if let Some(scheme) = scheme {
        return Ok(scheme);
    }
    #[cfg(feature = "gguf")]
    if let Some(scheme) = rlx_gguf::GgmlType::from_name(s.trim()).and_then(rlx_gguf::quant_scheme) {
        return Ok(scheme);
    }
    Err(ApiError::type_(format!(
        "unknown quant scheme '{s}' (a GGUF type such as Q4_K — needs the `gguf` feature — or \
         int8:<block>, int8_asym:<block>, int4:<block>, fp8e4m3, fp8e5m2, nvfp4, \
         mlx_affine:<bits>:<group>, mlx_mxfp4:<group>, mlx_mxfp8:<group>, mxfp4x2:<group>)"
    )))
}

// ── the spelling table ──────────────────────────────────────

/// Canonical spellings per string-selected kind, for generators that emit a
/// typed enum. Kinds whose values are open-ended (`mask`'s `sliding:<N>`,
/// `quant_scheme`'s block parameters, GGUF names) list their common forms;
/// the target language must still accept any string.
pub const ALL_KINDS: &[(&str, &[&str])] = &[
    (
        "dtype",
        &[
            "f32", "f16", "bf16", "f64", "i8", "u8", "i16", "i32", "u32", "i64", "bool", "c64",
            "c128",
        ],
    ),
    ("cmp_op", &["eq", "ne", "lt", "le", "gt", "ge"]),
    ("reduce_op", &["sum", "mean", "max", "min", "prod"]),
    ("mask", &["none", "causal", "sliding:<N>"]),
    ("rope_style", &["neox", "gptj"]),
    ("interp", &["nearest", "linear"]),
    ("ada_norm", &["layer_norm", "rms_norm"]),
    ("scatter_red", &["none", "add", "mul", "max", "min"]),
    (
        "pad_mode",
        &["constant", "reflect", "replicate", "circular"],
    ),
    ("vq_metric", &["l2", "cosine"]),
    ("qr_part", &["q", "r"]),
    ("svd_part", &["u", "s", "vt"]),
    ("spd_fn", &["logm", "expm", "sqrtm", "invsqrtm"]),
    ("fft_norm", &["backward", "ortho", "forward"]),
    ("window", &["rectangular", "hann", "hamming", "blackman"]),
    ("fir_mode", &["full", "same", "valid", "causal"]),
    (
        "scaled_fmt",
        &["f8e4m3", "f8e5m2", "f6e3m2", "f6e2m3", "f4e2m1"],
    ),
    ("scale_layout", &["per_tensor", "mx", "nvfp4"]),
    (
        "quant_scheme",
        &[
            "int8:<block>",
            "int8_asym:<block>",
            "int4:<block>",
            "fp8e4m3",
            "fp8e5m2",
            "nvfp4",
            "mlx_affine:<bits>:<group>",
            "mlx_mxfp4:<group>",
            "mlx_mxfp8:<group>",
            "mxfp4x2:<group>",
            "Q4_K",
        ],
    ),
];

/// Variant names of an IR enum with an `ALL` table, for the generators.
pub fn activation_names() -> Vec<String> {
    Activation::ALL
        .iter()
        .map(|a| snake(&format!("{a:?}")))
        .collect()
}

/// [`BinaryOp`] variant names, snake_case.
pub fn binop_names() -> Vec<String> {
    BinaryOp::ALL
        .iter()
        .map(|o| snake(&format!("{o:?}")))
        .collect()
}

/// `GeluApprox` → `gelu_approx`.
pub fn snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ir_activation_and_binop_round_trips_by_name() {
        for a in Activation::ALL {
            assert_eq!(activation(&format!("{a:?}")).unwrap(), a);
            assert_eq!(activation(&snake(&format!("{a:?}"))).unwrap(), a);
        }
        for op in BinaryOp::ALL {
            assert_eq!(binop(&snake(&format!("{op:?}"))).unwrap(), op);
        }
        assert_eq!(binop("**").unwrap(), BinaryOp::Pow);
    }

    #[test]
    fn listed_spellings_parse() {
        for (kind, names) in ALL_KINDS {
            for n in names.iter().filter(|n| !n.contains('<') && **n != "Q4_K") {
                let ok = match *kind {
                    "dtype" => dtype(n).is_ok(),
                    "cmp_op" => cmp_op(n).is_ok(),
                    "reduce_op" => reduce_op(n).is_ok(),
                    "mask" => mask(n).is_ok(),
                    "rope_style" => rope_style(n).is_ok(),
                    "interp" => interp(n).is_ok(),
                    "ada_norm" => ada_norm(n).is_ok(),
                    "scatter_red" => scatter_red(n).is_ok(),
                    "pad_mode" => pad_mode(n, 0.0).is_ok(),
                    "vq_metric" => vq_metric(n).is_ok(),
                    "qr_part" => qr_part(n).is_ok(),
                    "svd_part" => svd_part(n).is_ok(),
                    "spd_fn" => spd_fn(n).is_ok(),
                    "fft_norm" => fft_norm(n).is_ok(),
                    "window" => window(n).is_ok(),
                    "fir_mode" => fir_mode(n).is_ok(),
                    "scaled_fmt" => scaled_fmt(n).is_ok(),
                    "scale_layout" => scale_layout(n).is_ok(),
                    "quant_scheme" => quant_scheme(n).is_ok(),
                    other => panic!("no parser for kind '{other}'"),
                };
                assert!(ok, "{kind}: '{n}' does not parse");
            }
        }
        assert!(mask("sliding:128").is_ok());
        assert!(quant_scheme("mlx_affine:4:64").is_ok());
    }

    #[test]
    fn unknown_names_say_what_is_accepted() {
        let e = activation("nope").unwrap_err();
        assert!(e.message.contains("LogSigmoid"), "{e}");
        let e = rope_style("rotary").unwrap_err();
        assert!(e.message.contains("neox, gptj"), "{e}");
    }
}
