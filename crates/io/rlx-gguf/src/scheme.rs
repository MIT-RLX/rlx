// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`GgmlType`] → RLX [`QuantScheme`] (feature `ir`): the one table every
//! consumer of packed GGUF weights maps file dtypes through — rlx-cpu's
//! `quant_scheme_for_ggml` and rlx-flow's `GgufSource` both read it.

use crate::GgmlType;
use rlx_ir::quant::QuantScheme;

/// The RLX [`QuantScheme`] for a GGUF tensor dtype when the type is a
/// supported packed-quant layout for `Op::DequantMatMul`; `None` for float
/// and unsupported types.
pub fn quant_scheme(dtype: GgmlType) -> Option<QuantScheme> {
    match dtype {
        GgmlType::Q4_0 => Some(QuantScheme::GgufQ4_0),
        GgmlType::Q4_1 => Some(QuantScheme::GgufQ4_1),
        GgmlType::Q5_0 => Some(QuantScheme::GgufQ5_0),
        GgmlType::Q5_1 => Some(QuantScheme::GgufQ5_1),
        GgmlType::Q8_0 => Some(QuantScheme::GgufQ8_0),
        GgmlType::Q2K => Some(QuantScheme::GgufQ2K),
        GgmlType::Q3K => Some(QuantScheme::GgufQ3K),
        GgmlType::Q4K => Some(QuantScheme::GgufQ4K),
        GgmlType::Q5K => Some(QuantScheme::GgufQ5K),
        GgmlType::Q6K => Some(QuantScheme::GgufQ6K),
        GgmlType::Q8K => Some(QuantScheme::GgufQ8K),
        GgmlType::IQ4NL => Some(QuantScheme::GgufIQ4NL),
        GgmlType::IQ4XS => Some(QuantScheme::GgufIQ4XS),
        GgmlType::IQ2XXS => Some(QuantScheme::GgufIQ2XXS),
        GgmlType::IQ2XS => Some(QuantScheme::GgufIQ2XS),
        GgmlType::IQ2S => Some(QuantScheme::GgufIQ2S),
        GgmlType::IQ3XXS => Some(QuantScheme::GgufIQ3XXS),
        GgmlType::IQ3S => Some(QuantScheme::GgufIQ3S),
        GgmlType::IQ1S => Some(QuantScheme::GgufIQ1S),
        GgmlType::IQ1M => Some(QuantScheme::GgufIQ1M),
        GgmlType::TQ1_0 => Some(QuantScheme::GgufTQ1_0),
        GgmlType::TQ2_0 => Some(QuantScheme::GgufTQ2_0),
        GgmlType::MXFP4 => Some(QuantScheme::GgufMXFP4),
        GgmlType::NVFP4 => Some(QuantScheme::GgufNVFP4),
        GgmlType::Q1_0 => Some(QuantScheme::GgufQ1_0),
        GgmlType::Q2_0 => Some(QuantScheme::GgufQ2_0G64),
        GgmlType::FV5 => Some(QuantScheme::GgufFV5),
        GgmlType::FV5B => Some(QuantScheme::GgufFV5B),
        GgmlType::G8_0 => Some(QuantScheme::GgufG8_0),
        // PrismML Ternary Bonsai 2: PQ2_0 is byte-identical to Q2_0 at a
        // distinct type id, PTQ1_0 is base-3 ternary at group 128.
        GgmlType::PQ2_0 => Some(QuantScheme::GgufQ2_0),
        GgmlType::PTQ1_0 => Some(QuantScheme::GgufPtq1_0),
        _ => None,
    }
}
