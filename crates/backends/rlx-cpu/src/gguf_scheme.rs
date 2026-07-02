//! Map [`rlx_gguf::GgmlType`] to RLX [`QuantScheme`] for graph builders.

use rlx_gguf::GgmlType;
use rlx_ir::quant::QuantScheme;

/// Returns the RLX [`QuantScheme`] for a GGUF tensor dtype when the type
/// is a supported packed-quant layout for `Op::DequantMatMul`.
pub fn quant_scheme_for_ggml(dtype: GgmlType) -> Option<QuantScheme> {
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
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_scheme_for_ggml_legacy_blocks() {
        assert_eq!(
            quant_scheme_for_ggml(GgmlType::Q4_0),
            Some(QuantScheme::GgufQ4_0)
        );
        assert_eq!(
            quant_scheme_for_ggml(GgmlType::Q5_1),
            Some(QuantScheme::GgufQ5_1)
        );
        assert_eq!(quant_scheme_for_ggml(GgmlType::F32), None);
    }
}
