// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Host-side execution for ops with no stable MIL lowering (FFT, log-mel
// filterbank, token sampling, RNG fills). Used by the hybrid ANE runner when
// a graph mixes these with CoreML-lowerable compute.

use std::collections::HashMap;

use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::op_registry::lookup_coreml_kernel;
use crate::{CoremlError, Result};

/// Custom ops that have a native MIL lowering (not host-segmented).
fn mil_lowerable_custom(name: &str) -> bool {
    matches!(name, "onnx.ScatterND")
}

/// Ops executed on the host between CoreML segments (not lowered to MIL).
///
/// Prefer [`is_host_node`] when a graph is available — `Conv3d` /
/// `ConvTranspose3d` are MIL-native only with baked Param/Constant weights
/// (CoreML rejects dynamic 3D weights).
pub fn is_host_op(op: &Op) -> bool {
    match op {
        Op::Fft { .. }
        | Op::LogMel
        | Op::Sample { .. }
        | Op::RngNormal { .. }
        | Op::RngUniform { .. }
        | Op::WelchPeaks { .. }
        // `carry = false` LSTM/GRU/RNN lower natively to MIL (`mil::rnn`);
        // only the stateful decode-carry form stays on the host.
        | Op::Lstm { carry: true, .. }
        | Op::Gru { carry: true, .. }
        | Op::Rnn { carry: true, .. }
        | Op::Mamba2 { .. }
        | Op::ScanBackward { .. }
        | Op::ScanBackwardXs { .. }
        | Op::ScatterElements { .. }
        | Op::GatherNd { .. }
        | Op::GatherElements { .. } => true,
        // Native MIL `while_loop` lowering (default) keeps the scan ON-DEVICE —
        // no host split, so the whole graph stays one CoreML model instead of
        // `length`+1 separately-compiled segments (the SPD-eigensolver cost:
        // tensorcspnet/coreml 88s→68s, spdnet 64s→0.2s). Native ONLY for the
        // final-carry form whose body is itself fully MIL-lowerable (no nested
        // host ops); the trajectory form, a host-op body, or
        // `RLX_COREML_NATIVE_SCAN=0` all host-fall-back.
        Op::Scan {
            save_trajectory,
            body,
            ..
        } => {
            *save_trajectory
                || std::env::var("RLX_COREML_NATIVE_SCAN").as_deref() == Ok("0")
                || body.nodes().iter().any(|n| is_host_node(body, n.id))
        }
        // Most `Op::Custom` stay host (ONNX reference kernels). ScatterND has a
        // MIL `scatter_nd` path so F5-TTS Transformer does not hybridize into
        // ~88 host segments (which breaks CoreML input declaration).
        Op::Custom { name, .. } => !mil_lowerable_custom(name),
        // Full-coverage host fallbacks (no MIL arm, or CPU-only kernels).
        // Fma / fused epilogues / FakeQuantize / Conv3d(with Param weights)
        // lower to MIL — see `mil::{activation,conv_pool,matmul,norm,quant}`.
        Op::FakeQuantizeLSQ { .. }
        | Op::FakeQuantizeLSQBackwardX { .. }
        | Op::FakeQuantizeLSQBackwardScale { .. }
        | Op::ElementwiseRegion { .. }
        | Op::TransformRegion { .. }
        | Op::BatchElementwiseRegion { .. }
        | Op::DotGeneral { .. }
        | Op::DenseSolve
        | Op::BatchedDenseSolve
        // Cholesky / TriangularSolve / Det / LogDet host-stage to CPU LAPACK (no MIL arm).
        | Op::Cholesky
        | Op::TriangularSolve { .. }
        | Op::Det
        | Op::LogDet
        // Sort / ArgSort: stable strided sort on CPU, no MIL arm.
        | Op::Sort { .. } | Op::Svd { .. } | Op::Qr { .. }
        | Op::ArgSort { .. }
        | Op::Im2Col { .. }
        | Op::ReluBackward
        | Op::ActivationBackward { .. }
        | Op::FakeQuantizeBackward { .. }
        // ComplexNormSq / ComplexNormSqBackward / Conjugate / FftButterflyStage
        // lower natively to MIL (interleaved F32); see `mil::complex`.
        | Op::SoftmaxCrossEntropy
        | Op::PartitionedConv { .. }
        | Op::QMatMul { .. }
        | Op::QConv2d { .. }
        | Op::ScaledMatMul { .. }
        | Op::ScaledQuantize { .. }
        | Op::ScaledQuantScale { .. }
        | Op::ScaledDequantize { .. }
        | Op::FusedConvBiasAct { .. }
        | Op::FusedTransformerLayer { .. }
        | Op::If { .. }
        | Op::While { .. }
        | Op::GaussianSplatRender { .. }
        | Op::GaussianSplatRenderBackward { .. }
        | Op::GaussianSplatPrepare { .. }
        | Op::GaussianSplatRasterize { .. }
        | Op::CustomFn { .. }
        | Op::LogMelBackward
        | Op::BiMap
        | Op::ReEig { .. }
        | Op::LogEig { .. }
        | Op::SpdBatchNorm { .. }
        | Op::SpdKarcherMean { .. }
        | Op::ReEigBackward { .. }
        | Op::LogEigBackward { .. }
        | Op::SpdBatchNormBackwardX { .. }
        | Op::SpdBatchNormBackwardG { .. }
        | Op::SpdKarcherMeanWeighted { .. }
        | Op::SpdLogMap
        | Op::SpdExpMap
        | Op::SpdParallelTransport
        | Op::SpdMatrixFnBatch { .. }
        | Op::SpdLogMapBackward
        | Op::SpdExpMapBackward
        | Op::SpdParallelTransportBackward
        | Op::SpdMatrixFnBatchBackward { .. }
        | Op::Eigh
        | Op::EighBackward
        | Op::EighBatch
        | Op::EighBatchBackward
        | Op::RopeBackward { .. }
        | Op::CumsumBackward { .. }
        | Op::GatherBackward { .. }
        | Op::BatchNormInferenceBackwardInput { .. }
        | Op::BatchNormInferenceBackwardGamma { .. }
        | Op::BatchNormInferenceBackwardBeta
        | Op::MaxPool3dBackward { .. }
        | Op::Conv3dBackwardInput { .. }
        | Op::Conv3dBackwardWeight { .. }
        | Op::Interpolate3d { .. } => true,
        // Native MIL under `training`; host-fallback otherwise so claiming the
        // OpKind for coverage does not hit the MIL Unsupported arm.
        #[cfg(not(feature = "training"))]
        Op::MaxPool2dBackward { .. }
        | Op::Conv2dBackwardInput { .. }
        | Op::Conv2dBackwardWeight { .. }
        | Op::SoftmaxCrossEntropyWithLogits
        | Op::SoftmaxCrossEntropyBackward
        | Op::AttentionBackward { .. }
        | Op::LayerNormBackwardInput { .. }
        | Op::LayerNormBackwardGamma { .. }
        | Op::RmsNormBackwardInput { .. }
        | Op::RmsNormBackwardGamma { .. }
        | Op::RmsNormBackwardBeta { .. }
        | Op::GroupNormBackwardInput { .. }
        | Op::GroupNormBackwardGamma { .. }
        | Op::GroupNormBackwardBeta { .. }
        | Op::AdaLayerNormBackward { .. }
        | Op::GatedResidualBackward { .. } => true,
        _ => false,
    }
}

/// Graph-aware host decision (see [`is_host_op`]).
pub fn is_host_node(graph: &Graph, id: NodeId) -> bool {
    let node = graph.node(id);
    match &node.op {
        // CoreML's 3D `conv` / `conv_transpose` reject dynamic weights —
        // only Param/Constant weights stay on the MIL path.
        Op::Conv3d { .. } | Op::ConvTranspose3d { .. } => {
            let w = &graph.node(node.inputs[1]).op;
            !matches!(w, Op::Param { .. } | Op::Constant { .. })
        }
        other => is_host_op(other),
    }
}

/// Run one host op; `env` maps producer `NodeId` → f32 tensor (row-major).
pub fn run_host_node(
    graph: &Graph,
    id: NodeId,
    env: &HashMap<u32, Vec<f32>>,
    _params: &HashMap<String, Vec<f32>>,
    typed_params: &crate::mil::TypedParams,
) -> Result<Vec<f32>> {
    let node = graph.node(id);
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("host_exec: missing value for v{}", nid.0)))
    };
    match &node.op {
        Op::Fft { inverse, norm } => {
            let x = load(node.inputs[0])?;
            // The FFT row is the innermost dim (real-block layout: first N real,
            // second N imag → row = 2N). Deriving it from x.len() collapses the
            // whole batch into one row (outer=1), which is wrong for batch>1.
            let in_shape = graph.shape(node.inputs[0]);
            let row = in_shape
                .dim(in_shape.rank().saturating_sub(1))
                .unwrap_static();
            fft1d_f32(&x, row, *inverse, *norm)
        }
        Op::LogMel => {
            let spec = load(node.inputs[0])?;
            let filters = load(node.inputs[1])?;
            let spec_shape = graph.shape(node.inputs[0]).clone();
            let filt_shape = graph.shape(node.inputs[1]).clone();
            let meta = rlx_ir::audio::log_mel_meta(&spec_shape, &filt_shape)
                .map_err(CoremlError::Runtime)?;
            let mut out = vec![0f32; meta.outer * meta.n_mels];
            rlx_ir::audio::log_mel_block_f32(
                &spec,
                &filters,
                meta.outer,
                meta.n_fft,
                meta.n_bins,
                meta.n_mels,
                &mut out,
            );
            Ok(out)
        }
        Op::Sample {
            top_k,
            top_p,
            temperature,
            seed,
        } => {
            let logits = load(node.inputs[0])?;
            let shape = graph.shape(node.inputs[0]).clone();
            let rank = shape.rank();
            let batch = if rank >= 2 {
                shape.dim(rank - 2).unwrap_static()
            } else {
                1
            };
            let vocab = shape.dim(rank - 1).unwrap_static();
            sample_f32(&logits, batch, vocab, *top_k, *top_p, *temperature, *seed)
        }
        Op::RngNormal {
            mean,
            scale,
            key,
            op_seed,
        } => {
            let out_len = node.shape.num_elements().unwrap_or(0);
            let mut out = vec![0f32; out_len];
            rlx_ir::fill_normal_like(
                &mut out,
                *mean,
                *scale,
                rlx_ir::RngOptions::default(),
                *key,
                *op_seed,
            );
            Ok(out)
        }
        Op::RngUniform {
            low,
            high,
            key,
            op_seed,
        } => {
            let out_len = node.shape.num_elements().unwrap_or(0);
            let mut out = vec![0f32; out_len];
            rlx_ir::fill_uniform_like(
                &mut out,
                *low,
                *high,
                rlx_ir::RngOptions::default(),
                *key,
                *op_seed,
            );
            Ok(out)
        }
        Op::WelchPeaks { k, n_segments } => {
            let spec = load(node.inputs[0])?;
            let spec_shape = graph.shape(node.inputs[0]).clone();
            let rank = spec_shape.rank();
            let welch_batch = if rank >= 2 {
                spec_shape.dim(rank - 2).unwrap_static()
            } else {
                1
            };
            let n_fft2 = spec_shape.dim(rank - 1).unwrap_static();
            let n_fft = n_fft2 / 2;
            let out_len = node.shape.num_elements().unwrap_or(0);
            let mut out = vec![0f32; out_len];
            rlx_ir::audio::welch_peaks_block_f32(
                &spec,
                welch_batch,
                n_fft,
                *n_segments,
                *k,
                &mut out,
            );
            Ok(out)
        }
        Op::Lstm {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
        } => run_lstm_f32(
            graph,
            node,
            env,
            *hidden_size,
            *num_layers,
            *bidirectional,
            *carry,
        ),
        Op::Gru {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
        } => run_gru_f32(
            graph,
            node,
            env,
            *hidden_size,
            *num_layers,
            *bidirectional,
            *carry,
        ),
        Op::Rnn {
            hidden_size,
            num_layers,
            bidirectional,
            carry,
            relu,
        } => run_rnn_f32(
            graph,
            node,
            env,
            *hidden_size,
            *num_layers,
            *bidirectional,
            *carry,
            *relu,
        ),
        Op::Mamba2 {
            head_dim,
            state_size,
        } => run_mamba2_f32(graph, node, env, *head_dim, *state_size),
        Op::Custom { name, attrs, .. } => {
            run_custom_f32(graph, node, env, name, attrs, typed_params)
        }
        Op::Scan { .. } => {
            for nid in &node.inputs {
                if !env.contains_key(&nid.0) {
                    return Err(CoremlError::Runtime(format!(
                        "scan: missing value for v{}",
                        nid.0
                    )));
                }
            }
            Ok(rlx_cpu::thunk::run_scan_node_f32(node, |nid| {
                env.get(&nid.0).cloned().unwrap_or_default()
            }))
        }
        Op::ScanBackward { .. } | Op::ScanBackwardXs { .. } => {
            for nid in &node.inputs {
                if !env.contains_key(&nid.0) {
                    return Err(CoremlError::Runtime(format!(
                        "ScanBackward: missing value for v{}",
                        nid.0
                    )));
                }
            }
            Ok(rlx_cpu::thunk::run_host_op_node_f32(graph, node, |nid| {
                env.get(&nid.0).cloned().unwrap_or_default()
            }))
        }
        Op::ScatterElements { .. } | Op::GatherNd { .. } | Op::GatherElements { .. } => {
            for nid in &node.inputs {
                if !env.contains_key(&nid.0) {
                    return Err(CoremlError::Runtime(format!(
                        "indexing host op: missing value for v{}",
                        nid.0
                    )));
                }
            }
            Ok(rlx_cpu::thunk::run_host_op_node_f32(graph, node, |nid| {
                env.get(&nid.0).cloned().unwrap_or_default()
            }))
        }
        // Generic CPU one-op eval for host-segmented nodes. Segment planning
        // (including graph-aware Conv3d dynamic-weight host) decides the split;
        // do not re-gate on `is_host_op` here.
        _other => {
            for nid in &node.inputs {
                if !env.contains_key(&nid.0) {
                    return Err(CoremlError::Runtime(format!(
                        "host_exec: missing value for v{}",
                        nid.0
                    )));
                }
            }
            Ok(rlx_cpu::thunk::run_host_op_node_f32(graph, node, |nid| {
                env.get(&nid.0).cloned().unwrap_or_default()
            }))
        }
    }
}

/// `row` is the per-FFT row length (real-block layout: first `row/2` real, then
/// `row/2` imag), i.e. the innermost dim of the Op::Fft input. `outer = len/row`
/// independent FFTs are run.
fn fft1d_f32(x: &[f32], row: usize, inverse: bool, norm: FftNorm) -> Result<Vec<f32>> {
    if row == 0 || !row.is_multiple_of(2) {
        return Err(CoremlError::Runtime(format!(
            "fft: empty or odd-length row {row}"
        )));
    }
    let n_complex = row / 2;
    if x.is_empty() || !x.len().is_multiple_of(row) {
        return Err(CoremlError::Runtime(format!(
            "fft: length {} not divisible by row size {row}",
            x.len()
        )));
    }
    let outer = x.len() / row;
    let mut out = x.to_vec();
    #[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
    {
        let base = out.as_mut_ptr() as *mut u8;
        unsafe {
            rlx_cpu::thunk::execute_fft1d_f32(0, 0, outer, n_complex, inverse, norm.tag(), base);
        }
        Ok(out)
    }
    #[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
    {
        let _ = (inverse, norm, outer, n_complex);
        Err(CoremlError::Unsupported(
            "fft host execution requires macOS/iOS".into(),
        ))
    }
}

#[cfg(all(test, all(target_vendor = "apple", not(target_os = "watchos"))))]
mod fft_tests {
    use super::*;

    #[test]
    fn batched_fft_runs_each_row_independently() {
        // Real-block layout, n_complex=4 → row=8. Run one row, then the same row
        // duplicated: every output row must equal the single-row FFT. The old
        // code derived the row size from x.len() (outer=1), collapsing the batch
        // into one big FFT and failing this.
        let row = 8usize;
        let one: Vec<f32> = vec![1.0, 2.0, -3.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        let single = fft1d_f32(&one, row, false, FftNorm::Backward).unwrap();
        let mut two = one.clone();
        two.extend_from_slice(&one);
        let batched = fft1d_f32(&two, row, false, FftNorm::Backward).unwrap();
        assert_eq!(batched.len(), 16);
        for i in 0..row {
            assert!((batched[i] - single[i]).abs() < 1e-4, "row0[{i}]");
            assert!((batched[row + i] - single[i]).abs() < 1e-4, "row1[{i}]");
        }
        // Sanity: a constant real signal → DC bin = sum, others ≈ 0.
        assert!((single[0] - 0.5).abs() < 1e-4); // re DC = 1+2-3+0.5
    }
}

fn sample_f32(
    logits: &[f32],
    batch: usize,
    vocab: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: u64,
) -> Result<Vec<f32>> {
    if logits.len() != batch * vocab {
        return Err(CoremlError::Runtime(format!(
            "sample: logits len {} != batch*vocab {}",
            logits.len(),
            batch * vocab
        )));
    }
    let mut out = vec![0f32; batch];
    let mut rng = rlx_ir::Philox4x32::new(if seed == 0 { 0xDEADBEEF } else { seed });
    for bi in 0..batch {
        let row = &logits[bi * vocab..(bi + 1) * vocab];
        out[bi] = sample_row(row, top_k.min(vocab), top_p, temperature, &mut rng) as f32;
    }
    Ok(out)
}

fn sample_row(
    row: &[f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rng: &mut rlx_ir::Philox4x32,
) -> usize {
    let v = row.len();
    if v == 0 {
        return 0;
    }
    if temperature <= 0.0 || top_k == 1 {
        return argmax_row(row);
    }
    let mut logits: Vec<f32> = row.to_vec();
    if temperature != 1.0 {
        let inv = 1.0 / temperature;
        logits.iter_mut().for_each(|x| *x *= inv);
    }
    if top_k > 0 && top_k < v {
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let kth = logits[idx[top_k - 1]];
        for l in logits.iter_mut() {
            if *l < kth {
                *l = f32::NEG_INFINITY;
            }
        }
    }
    if top_p < 1.0 {
        nucleus_filter(&mut logits, top_p);
    }
    softmax_inplace(&mut logits);
    multinomial(&logits, rng)
}

fn argmax_row(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = row[0];
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn nucleus_filter(logits: &mut [f32], top_p: f32) {
    let v = logits.len();
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut probs = pairs.iter().map(|(_, l)| (*l).exp()).collect::<Vec<_>>();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        probs.iter_mut().for_each(|p| *p /= sum);
    }
    let mut cum = 0f32;
    let mut cut = v;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> = pairs.iter().take(cut).map(|(i, _)| *i).collect();
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn softmax_inplace(logits: &mut [f32]) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for l in logits.iter_mut() {
        *l = (*l - max).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum;
        }
    }
}

fn multinomial(probs: &[f32], rng: &mut rlx_ir::Philox4x32) -> usize {
    let u: f32 = rng.next_f32();
    let mut cum = 0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Re-encode `n` f32 host values into the little-endian bytes of `dtype`.
fn f32_to_dtype_bytes(f: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => f.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F64 => f.iter().flat_map(|&x| (x as f64).to_le_bytes()).collect(),
        DType::I64 => f.iter().flat_map(|&x| (x as i64).to_le_bytes()).collect(),
        DType::I32 => f.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
        DType::I16 => f.iter().flat_map(|&x| (x as i16).to_le_bytes()).collect(),
        DType::I8 => f.iter().map(|&x| x as i8 as u8).collect(),
        DType::U32 => f.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect(),
        DType::U8 => f.iter().map(|&x| x as u8).collect(),
        DType::Bool => f.iter().map(|&x| u8::from(x != 0.0)).collect(),
        _ => f.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Decode `dtype` little-endian bytes back to f32 host values.
fn dtype_bytes_to_f32(b: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::F64 => b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I16 => b
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I8 => b.iter().map(|&x| x as i8 as f32).collect(),
        DType::U32 => b
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::U8 | DType::Bool => b.iter().map(|&x| x as f32).collect(),
        _ => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}

fn run_custom_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    name: &str,
    attrs: &[u8],
    typed_params: &crate::mil::TypedParams,
) -> Result<Vec<f32>> {
    // Gather each operand from the f32 `env`.
    let mut in_f32: Vec<(Vec<f32>, Shape)> = Vec::with_capacity(node.inputs.len());
    for &inp in &node.inputs {
        let shape = graph.shape(inp).clone();
        let f32s = env
            .get(&inp.0)
            .ok_or_else(|| CoremlError::Runtime(format!("host_exec: missing input v{}", inp.0)))?;
        in_f32.push((f32s.clone(), shape));
    }
    let out_shape = node.shape.clone();

    // Prefer a registered CoreML-side kernel (f32 bytes in/out).
    if let Some(kernel) = lookup_coreml_kernel(name) {
        let in_bufs: Vec<(Vec<u8>, Shape)> = in_f32
            .iter()
            .map(|(f, s)| (f.iter().flat_map(|x| x.to_le_bytes()).collect(), s.clone()))
            .collect();
        let in_refs: Vec<(&[u8], &Shape)> =
            in_bufs.iter().map(|(b, s)| (b.as_slice(), s)).collect();
        let out_len = out_shape.num_elements().unwrap_or(0) * DType::F32.size_bytes();
        let mut out_bytes = vec![0u8; out_len];
        kernel
            .execute(&in_refs, (&mut out_bytes, &out_shape), attrs)
            .map_err(|e| CoremlError::Runtime(format!("Op::Custom('{name}'): {e}")))?;
        return Ok(out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }

    // Fall back to the rlx-cpu ONNX reference kernel (dtype-aware) — the same
    // generic host-delegate the Metal/MLX/wgpu backends use. `env` is f32, so
    // re-encode each operand to its declared dtype, run the kernel, cast back.
    rlx_cpu::onnx_ref::register_onnx_reference_kernels();
    if rlx_cpu::op_registry::lookup_cpu_kernel(name).is_none() {
        return Err(CoremlError::Runtime(format!(
            "host_exec: no CoremlKernel or rlx-cpu reference kernel for Op::Custom('{name}')"
        )));
    }
    // CoreML's `promote_int_to_f32` demotes integer tensors to F32 in the graph.
    // Host reference kernels still expect the original integer dtypes — restore
    // them here (same idea as the Gather/Scatter index override below).
    let (in_dtype_override, out_dtype_override) =
        custom_int_dtype_overrides(name, attrs, in_f32.len(), graph, node, typed_params);
    let idx_pos: Option<usize> = match name {
        "onnx.GatherND" | "onnx.ScatterND" | "onnx.ScatterElements" | "onnx.GatherElements" => {
            Some(1)
        }
        "onnx.OneHot" => Some(0),
        _ => None,
    };
    let in_shapes: Vec<Shape> = in_f32
        .iter()
        .enumerate()
        .map(|(i, (_, s))| {
            if Some(i) == idx_pos {
                s.clone().with_dtype(DType::I64)
            } else if let Some(dt) = in_dtype_override.get(i).copied().flatten() {
                s.clone().with_dtype(dt)
            } else {
                s.clone()
            }
        })
        .collect();
    let run_out_shape = match out_dtype_override {
        Some(dt) => out_shape.clone().with_dtype(dt),
        None => out_shape.clone(),
    };
    let in_bytes: Vec<Vec<u8>> = in_f32
        .iter()
        .zip(in_shapes.iter())
        .map(|((f, _), s)| f32_to_dtype_bytes(f, s.dtype()))
        .collect();
    let in_pairs: Vec<(&[u8], &Shape)> = in_bytes
        .iter()
        .zip(in_shapes.iter())
        .map(|(b, s)| (b.as_slice(), s))
        .collect();
    let out_n = run_out_shape.num_elements().unwrap_or(0);
    let mut out = vec![0u8; out_n * run_out_shape.dtype().size_bytes()];
    rlx_cpu::op_registry::run_custom_op_host(name, &in_pairs, (&mut out, &run_out_shape), attrs)
        .map_err(|e| CoremlError::Runtime(format!("Op::Custom('{name}'): {e}")))?;
    Ok(dtype_bytes_to_f32(&out, run_out_shape.dtype()))
}

/// Integer dtypes that `promote_int_to_f32` stripped, but the CPU reference
/// kernel still requires. Returns per-input overrides + optional output override.
fn custom_int_dtype_overrides(
    name: &str,
    attrs: &[u8],
    n_inputs: usize,
    graph: &Graph,
    node: &rlx_ir::Node,
    typed_params: &crate::mil::TypedParams,
) -> (Vec<Option<DType>>, Option<DType>) {
    let param_dtype = |inp_idx: usize| -> Option<DType> {
        let pid = *node.inputs.get(inp_idx)?;
        match &graph.node(pid).op {
            Op::Param { name } => typed_params.get(name).map(|(_, d)| *d),
            _ => None,
        }
    };
    let mut ins = vec![None; n_inputs];
    let out = match name {
        // attrs[0]: 0=quantized u8, 1=scale f32, 2=zero_point u8
        "onnx.DynamicQuantizeLinearExport" => match attrs.first().copied().unwrap_or(0) {
            0 | 2 => Some(DType::U8),
            _ => None,
        },
        // act_q is u8; act_zp u8; weight i8 when still quantized.
        "onnx.QMatMul" => {
            if !ins.is_empty() {
                ins[0] = Some(DType::U8);
            }
            if ins.len() > 2 {
                ins[2] = Some(DType::U8);
            }
            if let Some(DType::I8) = param_dtype(3) {
                ins[3] = Some(DType::I8);
            }
            if ins.len() > 5 {
                ins[5] = Some(DType::I8);
            }
            None
        }
        "onnx.QMatMulBaked" => {
            if !ins.is_empty() {
                ins[0] = Some(DType::U8);
            }
            if ins.len() > 2 {
                ins[2] = Some(DType::U8);
            }
            None
        }
        // ONNX DynamicQuantizeLSTM: W/R are int8 (inputs 1,2); trailing zp may be u8.
        "onnx.DynamicQuantizeLSTM" => {
            if ins.len() > 1 {
                ins[1] = Some(DType::I8);
            }
            if ins.len() > 2 {
                ins[2] = Some(DType::I8);
            }
            // When quantized: trailing W_zp / R_zp accept I8 (not U8).
            if n_inputs >= 8 {
                ins[n_inputs - 3] = Some(DType::I8);
                ins[n_inputs - 1] = Some(DType::I8);
            }
            None
        }
        // I64 alignment / control-flow ops (all inputs + output are i64).
        "onnx.ConcatFromSequence"
        | "onnx.KittenConcatFromSequence"
        | "onnx.ExpandI64Align"
        | "onnx.AlignmentRange"
        | "onnx.AlignmentScatterIndices" => {
            for slot in ins.iter_mut() {
                *slot = Some(DType::I64);
            }
            Some(DType::I64)
        }
        // F0IfSelect: align is i64; f0/n are f32; output f32.
        "onnx.F0IfSelect" => {
            if ins.len() > 1 {
                // typical: [f0, align] or [f0, n, align] — force any non-f32-looking
                // trailing control input. Safer: mark all but first as I64 when
                // they came from integer params; otherwise force input 1.
                ins[1] = Some(DType::I64);
            }
            None
        }
        _ => None,
    };
    (ins, out)
}

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn run_lstm_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) -> Result<Vec<f32>> {
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("lstm: missing value for v{}", nid.0)))
    };
    let x = load(node.inputs[0])?;
    let w_ih = load(node.inputs[1])?;
    let w_hh = load(node.inputs[2])?;
    let bias = load(node.inputs[3])?;
    let (h0, c0) = if carry {
        (load(node.inputs[4])?, load(node.inputs[5])?)
    } else {
        (Vec::new(), Vec::new())
    };

    let in_shape = graph.shape(node.inputs[0]).clone();
    let rank = in_shape.rank();
    let batch = in_shape.dim(rank - 3).unwrap_static();
    let seq = in_shape.dim(rank - 2).unwrap_static();
    let input_size = in_shape.dim(rank - 1).unwrap_static();
    let out_len = node.shape.num_elements().unwrap_or(0);
    let ex = rlx_cpu::thunk::rnn_expected_lens(
        4,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
    );
    rlx_cpu::thunk::check_rnn_lens(
        "lstm",
        &[
            ("x", x.len(), ex.x),
            ("w_ih", w_ih.len(), ex.w_ih),
            ("w_hh", w_hh.len(), ex.w_hh),
            ("bias", bias.len(), ex.bias),
        ],
    )
    .map_err(CoremlError::Runtime)?;
    let mut arena: Vec<u8> = Vec::new();
    let mut push_f32 = |v: &[f32]| -> usize {
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        off
    };
    let x_off = push_f32(&x);
    let wih_off = push_f32(&w_ih);
    let whh_off = push_f32(&w_hh);
    let bias_off = push_f32(&bias);
    let h0_off = if carry { push_f32(&h0) } else { 0 };
    let c0_off = if carry { push_f32(&c0) } else { 0 };
    let dst_off = arena.len();
    arena.resize(arena.len() + out_len * 4, 0);
    unsafe {
        rlx_cpu::thunk::execute_lstm_f32(
            x_off,
            wih_off,
            whh_off,
            bias_off,
            h0_off,
            c0_off,
            dst_off,
            batch,
            seq,
            input_size,
            hidden,
            num_layers,
            bidirectional,
            carry,
            arena.as_mut_ptr(),
        );
    }
    let dst_bytes = &arena[dst_off..dst_off + out_len * 4];
    Ok(dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
fn run_lstm_f32(
    _graph: &Graph,
    _node: &rlx_ir::Node,
    _env: &HashMap<u32, Vec<f32>>,
    _hidden: usize,
    _num_layers: usize,
    _bidirectional: bool,
    _carry: bool,
) -> Result<Vec<f32>> {
    Err(CoremlError::Unsupported(
        "lstm host execution requires macOS/iOS".into(),
    ))
}

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn run_gru_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
) -> Result<Vec<f32>> {
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("gru: missing value for v{}", nid.0)))
    };
    let x = load(node.inputs[0])?;
    let w_ih = load(node.inputs[1])?;
    let w_hh = load(node.inputs[2])?;
    let b_ih = load(node.inputs[3])?;
    let b_hh = load(node.inputs[4])?;
    let h0 = if carry {
        load(node.inputs[5])?
    } else {
        Vec::new()
    };

    let in_shape = graph.shape(node.inputs[0]).clone();
    let rank = in_shape.rank();
    let batch = in_shape.dim(rank - 3).unwrap_static();
    let seq = in_shape.dim(rank - 2).unwrap_static();
    let input_size = in_shape.dim(rank - 1).unwrap_static();
    let out_len = node.shape.num_elements().unwrap_or(0);
    let ex = rlx_cpu::thunk::rnn_expected_lens(
        3,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
    );
    rlx_cpu::thunk::check_rnn_lens(
        "gru",
        &[
            ("x", x.len(), ex.x),
            ("w_ih", w_ih.len(), ex.w_ih),
            ("w_hh", w_hh.len(), ex.w_hh),
            ("b_ih", b_ih.len(), ex.bias),
            ("b_hh", b_hh.len(), ex.bias),
        ],
    )
    .map_err(CoremlError::Runtime)?;
    let mut arena: Vec<u8> = Vec::new();
    let mut push_f32 = |v: &[f32]| -> usize {
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        off
    };
    let x_off = push_f32(&x);
    let wih_off = push_f32(&w_ih);
    let whh_off = push_f32(&w_hh);
    let bih_off = push_f32(&b_ih);
    let bhh_off = push_f32(&b_hh);
    let h0_off = if carry { push_f32(&h0) } else { 0 };
    let dst_off = arena.len();
    arena.resize(arena.len() + out_len * 4, 0);
    unsafe {
        rlx_cpu::thunk::execute_gru_f32(
            x_off,
            wih_off,
            whh_off,
            bih_off,
            bhh_off,
            h0_off,
            dst_off,
            batch,
            seq,
            input_size,
            hidden,
            num_layers,
            bidirectional,
            carry,
            arena.as_mut_ptr(),
        );
    }
    let dst_bytes = &arena[dst_off..dst_off + out_len * 4];
    Ok(dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
fn run_gru_f32(
    _graph: &Graph,
    _node: &rlx_ir::Node,
    _env: &HashMap<u32, Vec<f32>>,
    _hidden: usize,
    _num_layers: usize,
    _bidirectional: bool,
    _carry: bool,
) -> Result<Vec<f32>> {
    Err(CoremlError::Unsupported(
        "gru host execution requires macOS/iOS".into(),
    ))
}

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn run_rnn_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    relu: bool,
) -> Result<Vec<f32>> {
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("rnn: missing value for v{}", nid.0)))
    };
    let x = load(node.inputs[0])?;
    let w_ih = load(node.inputs[1])?;
    let w_hh = load(node.inputs[2])?;
    let bias = load(node.inputs[3])?;
    let h0 = if carry {
        load(node.inputs[4])?
    } else {
        Vec::new()
    };

    let in_shape = graph.shape(node.inputs[0]).clone();
    let rank = in_shape.rank();
    let batch = in_shape.dim(rank - 3).unwrap_static();
    let seq = in_shape.dim(rank - 2).unwrap_static();
    let input_size = in_shape.dim(rank - 1).unwrap_static();
    let out_len = node.shape.num_elements().unwrap_or(0);
    let ex = rlx_cpu::thunk::rnn_expected_lens(
        1,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
    );
    rlx_cpu::thunk::check_rnn_lens(
        "rnn",
        &[
            ("x", x.len(), ex.x),
            ("w_ih", w_ih.len(), ex.w_ih),
            ("w_hh", w_hh.len(), ex.w_hh),
            ("bias", bias.len(), ex.bias),
        ],
    )
    .map_err(CoremlError::Runtime)?;
    let mut arena: Vec<u8> = Vec::new();
    let mut push_f32 = |v: &[f32]| -> usize {
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        off
    };
    let x_off = push_f32(&x);
    let wih_off = push_f32(&w_ih);
    let whh_off = push_f32(&w_hh);
    let bias_off = push_f32(&bias);
    let h0_off = if carry { push_f32(&h0) } else { 0 };
    let dst_off = arena.len();
    arena.resize(arena.len() + out_len * 4, 0);
    unsafe {
        rlx_cpu::thunk::execute_rnn_f32(
            x_off,
            wih_off,
            whh_off,
            bias_off,
            h0_off,
            dst_off,
            batch,
            seq,
            input_size,
            hidden,
            num_layers,
            bidirectional,
            carry,
            relu,
            arena.as_mut_ptr(),
        );
    }
    let dst_bytes = &arena[dst_off..dst_off + out_len * 4];
    Ok(dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
fn run_rnn_f32(
    _graph: &Graph,
    _node: &rlx_ir::Node,
    _env: &HashMap<u32, Vec<f32>>,
    _hidden: usize,
    _num_layers: usize,
    _bidirectional: bool,
    _carry: bool,
    _relu: bool,
) -> Result<Vec<f32>> {
    Err(CoremlError::Unsupported(
        "rnn host execution requires macOS/iOS".into(),
    ))
}

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn run_mamba2_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    head_dim: usize,
    state_size: usize,
) -> Result<Vec<f32>> {
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("mamba2: missing value for v{}", nid.0)))
    };
    let x = load(node.inputs[0])?;
    let dt = load(node.inputs[1])?;
    let a = load(node.inputs[2])?;
    let b = load(node.inputs[3])?;
    let c = load(node.inputs[4])?;
    let x_shape = graph.shape(node.inputs[0]).clone();
    let batch = x_shape.dim(0).unwrap_static();
    let seq = x_shape.dim(1).unwrap_static();
    let heads = x_shape.dim(2).unwrap_static();
    let out_len = node.shape.num_elements().unwrap_or(0);
    let mut arena: Vec<u8> = Vec::new();
    let mut push_f32 = |v: &[f32]| -> usize {
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        off
    };
    let x_off = push_f32(&x);
    let dt_off = push_f32(&dt);
    let a_off = push_f32(&a);
    let b_off = push_f32(&b);
    let c_off = push_f32(&c);
    let dst_off = arena.len();
    arena.resize(arena.len() + out_len * 4, 0);
    unsafe {
        rlx_cpu::thunk::execute_mamba2_f32(
            x_off,
            dt_off,
            a_off,
            b_off,
            c_off,
            dst_off,
            batch,
            seq,
            heads,
            head_dim,
            state_size,
            arena.as_mut_ptr(),
        );
    }
    let dst_bytes = &arena[dst_off..dst_off + out_len * 4];
    Ok(dst_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
fn run_mamba2_f32(
    _graph: &Graph,
    _node: &rlx_ir::Node,
    _env: &HashMap<u32, Vec<f32>>,
    _head_dim: usize,
    _state_size: usize,
) -> Result<Vec<f32>> {
    Err(CoremlError::Unsupported(
        "mamba2 host execution requires macOS/iOS".into(),
    ))
}
