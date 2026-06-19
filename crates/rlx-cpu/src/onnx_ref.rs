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

//! Reference CPU kernels for generic `Op::Custom("onnx.*")` ops.

use std::sync::Arc;

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

const MOD: &str = "onnx.Mod";
const IS_NAN: &str = "onnx.IsNaN";
const CONCAT_FROM_SEQUENCE: &str = "onnx.ConcatFromSequence";
const KITTEN_CONCAT_FROM_SEQUENCE: &str = "onnx.KittenConcatFromSequence";
const EXPAND_I64_ALIGN: &str = "onnx.ExpandI64Align";
const ALIGNMENT_RANGE: &str = "onnx.AlignmentRange";
const VOCODER_WAVEFORM_SLICE: &str = "onnx.VocoderWaveformSlice";

struct ModKernel;
struct IsNaNKernel;
struct ConcatFromSequenceKernel;

impl ConcatFromSequenceKernel {
    fn run(inputs: &[CpuTensorRef<'_>], output: CpuTensorMut<'_>) -> Result<(), String> {
        if inputs.len() < 4 {
            return Err(format!("expected 4 inputs, got {}", inputs.len()));
        }
        let duration_mask = inputs[0].expect_i64("duration_mask")?;
        let range_ids = inputs[1].expect_i64("range_ids")?;
        let split_lens = inputs[2].expect_i64("split_lens")?;
        let trip = inputs[3].expect_i64("trip_count")?;
        let out = output.expect_i64_mut("output")?;
        let trip_count = crate::onnx_control_flow::resolve_concat_trip_count(
            trip,
            duration_mask.len(),
            split_lens.len(),
        );
        out.fill(0);
        crate::onnx_control_flow::concat_alignment_durations(
            duration_mask,
            range_ids,
            split_lens,
            trip_count,
            out,
        );
        Ok(())
    }
}

impl CpuKernel for ConcatFromSequenceKernel {
    fn name(&self) -> &str {
        CONCAT_FROM_SEQUENCE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        Self::run(inputs, output)
    }
}

struct KittenConcatFromSequenceAlias;

struct ExpandI64AlignKernel;

impl ExpandI64AlignKernel {
    fn run(inputs: &[CpuTensorRef<'_>], output: CpuTensorMut<'_>) -> Result<(), String> {
        if inputs.len() < 2 {
            return Err(format!("expected 2 inputs, got {}", inputs.len()));
        }
        let data = inputs[0].expect_i64("data")?;
        let shape = inputs[1].expect_i64("shape")?;
        let out = output.expect_i64_mut("output")?;
        crate::onnx_control_flow::expand_i64_align(data, shape, out);
        Ok(())
    }
}

impl CpuKernel for ExpandI64AlignKernel {
    fn name(&self) -> &str {
        EXPAND_I64_ALIGN
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        Self::run(inputs, output)
    }
}

struct AlignmentRangeKernel;

struct VocoderWaveformSliceKernel;

impl CpuKernel for VocoderWaveformSliceKernel {
    fn name(&self) -> &str {
        VOCODER_WAVEFORM_SLICE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 2 {
            return Err(format!("expected 2 inputs, got {}", inputs.len()));
        }
        let wave = inputs[0].expect_f32("wave")?;
        let align = inputs[1].expect_i64("align_frames")?;
        let frames = align.first().copied().unwrap_or(0);
        let in_shape: Vec<usize> = inputs[0]
            .shape()
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let out_shape: Vec<usize> = output
            .shape()
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let time_axis = if in_shape.len() == 3 {
            2
        } else {
            in_shape.len().saturating_sub(1)
        };
        let out = output.expect_f32_mut("out")?;
        crate::onnx_control_flow::vocoder_waveform_slice(
            wave, &in_shape, time_axis, frames, out, &out_shape,
        );
        Ok(())
    }
}

impl CpuKernel for AlignmentRangeKernel {
    fn name(&self) -> &str {
        ALIGNMENT_RANGE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("onnx.AlignmentRange expected frame-count input".into());
        }
        let limit = inputs[0].expect_i64("frame_count")?;
        let out = output.expect_i64_mut("range")?;
        crate::onnx_control_flow::alignment_range_ids(limit, out);
        Ok(())
    }
}

impl CpuKernel for KittenConcatFromSequenceAlias {
    fn name(&self) -> &str {
        KITTEN_CONCAT_FROM_SEQUENCE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        ConcatFromSequenceKernel::execute(&ConcatFromSequenceKernel, inputs, output, attrs)
    }
}

impl CpuKernel for ModKernel {
    fn name(&self) -> &str {
        MOD
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs
            .first()
            .ok_or("onnx.Mod: missing a")?
            .expect_f32("a")?;
        let b = inputs
            .get(1)
            .ok_or("onnx.Mod: missing b")?
            .expect_f32("b")?;
        let out = output.expect_f32_mut("out")?;
        let fmod = attrs.first().copied().unwrap_or(0) != 0;
        let n = a.len().min(b.len()).min(out.len());
        for i in 0..n {
            out[i] = if fmod {
                a[i] % b[i]
            } else {
                let q = (a[i] / b[i]).trunc();
                a[i] - q * b[i]
            };
        }
        Ok(())
    }
}

impl CpuKernel for IsNaNKernel {
    fn name(&self) -> &str {
        IS_NAN
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs
            .first()
            .ok_or("onnx.IsNaN: missing input")?
            .expect_f32("x")?;
        let out = output.expect_bool_mut("out")?;
        let n = x.len().min(out.len());
        for i in 0..n {
            out[i] = u8::from(x[i].is_nan());
        }
        Ok(())
    }
}

/// Register generic ONNX reference kernels (Mod, IsNaN, ConcatFromSequence, …).
pub fn register_onnx_reference_kernels() {
    register_cpu_kernel(Arc::new(ModKernel));
    register_cpu_kernel(Arc::new(IsNaNKernel));
    register_cpu_kernel(Arc::new(ConcatFromSequenceKernel));
    register_cpu_kernel(Arc::new(KittenConcatFromSequenceAlias));
    register_cpu_kernel(Arc::new(ExpandI64AlignKernel));
    register_cpu_kernel(Arc::new(AlignmentRangeKernel));
    register_cpu_kernel(Arc::new(VocoderWaveformSliceKernel));
    crate::onnx_indexing::register_onnx_indexing_kernels();
}

pub fn onnx_concat_from_sequence_name() -> &'static str {
    CONCAT_FROM_SEQUENCE
}
