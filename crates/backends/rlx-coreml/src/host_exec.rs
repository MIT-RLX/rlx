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

/// Ops executed on the host between CoreML segments (not lowered to MIL).
pub fn is_host_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Fft { .. }
            | Op::LogMel
            | Op::Sample { .. }
            | Op::RngNormal { .. }
            | Op::RngUniform { .. }
            | Op::WelchPeaks { .. }
            | Op::Lstm { .. }
            | Op::Gru { .. }
            | Op::Rnn { .. }
            | Op::Mamba2 { .. }
            | Op::Scan { .. }
            | Op::Custom { .. }
    )
}

/// Run one host op; `env` maps producer `NodeId` → f32 tensor (row-major).
pub fn run_host_node(
    graph: &Graph,
    id: NodeId,
    env: &HashMap<u32, Vec<f32>>,
    _params: &HashMap<String, Vec<f32>>,
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
        Op::Custom { name, attrs, .. } => run_custom_f32(graph, node, env, name, attrs),
        Op::Scan { .. } => run_scan_f32(graph, node, env),
        other => Err(CoremlError::Unsupported(format!(
            "host_exec: not a host op {:?}",
            other
        ))),
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

/// General `Op::Scan` on the host: compile the body once, lay the outer inputs
/// out in a local byte arena, and run rlx-cpu's `execute_scan_host` (same
/// reference kernels as the CPU backend). Enables IIR (`biquad`) between MIL
/// segments.
fn run_scan_f32(
    _graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
) -> Result<Vec<f32>> {
    let (body, length, save_trajectory, num_bcast, num_xs) = match &node.op {
        Op::Scan {
            body,
            length,
            save_trajectory,
            num_bcast,
            num_xs,
            ..
        } => (body, *length, *save_trajectory, *num_bcast as usize, *num_xs as usize),
        _ => unreachable!("run_scan_f32 called on non-Scan op"),
    };
    let load = |nid: NodeId| -> Result<Vec<f32>> {
        env.get(&nid.0)
            .cloned()
            .ok_or_else(|| CoremlError::Runtime(format!("scan: missing value for v{}", nid.0)))
    };
    let plan = rlx_cpu::thunk::compile_scan_body(body, num_bcast, num_xs);

    let mut arena: Vec<u8> = Vec::new();
    let init = load(node.inputs[0])?;
    let init_off = arena.len();
    arena.extend(init.iter().flat_map(|f| f.to_le_bytes()));

    let mut bcast_outer: Vec<(usize, usize)> = Vec::with_capacity(num_bcast);
    for i in 0..num_bcast {
        let v = load(node.inputs[1 + i])?;
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        bcast_outer.push((off, v.len() * 4));
    }
    let mut xs_outer: Vec<(usize, usize)> = Vec::with_capacity(num_xs);
    for i in 0..num_xs {
        let v = load(node.inputs[1 + num_bcast + i])?;
        let total = v.len() * 4;
        let off = arena.len();
        arena.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        xs_outer.push((off, total / length as usize));
    }

    let out_len = node.shape.num_elements().unwrap_or(0);
    let final_off = arena.len();
    arena.resize(final_off + out_len * 4, 0);
    unsafe {
        rlx_cpu::thunk::execute_scan_host(
            arena.as_mut_ptr(),
            &plan,
            init_off,
            final_off,
            length,
            save_trajectory,
            &xs_outer,
            &bcast_outer,
        );
    }
    Ok(arena[final_off..final_off + out_len * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn run_custom_f32(
    graph: &Graph,
    node: &rlx_ir::Node,
    env: &HashMap<u32, Vec<f32>>,
    name: &str,
    attrs: &[u8],
) -> Result<Vec<f32>> {
    let kernel = lookup_coreml_kernel(name).ok_or_else(|| {
        CoremlError::Runtime(format!(
            "host_exec: no CoremlKernel registered for Op::Custom('{name}')"
        ))
    })?;

    let mut in_bufs: Vec<(Vec<u8>, Shape)> = Vec::with_capacity(node.inputs.len());
    for &inp in &node.inputs {
        let shape = graph.shape(inp).clone();
        let f32s = env
            .get(&inp.0)
            .ok_or_else(|| CoremlError::Runtime(format!("host_exec: missing input v{}", inp.0)))?;
        let bytes: Vec<u8> = f32s.iter().flat_map(|f| f.to_le_bytes()).collect();
        in_bufs.push((bytes, shape));
    }
    let in_refs: Vec<(&[u8], &Shape)> = in_bufs.iter().map(|(b, s)| (b.as_slice(), s)).collect();
    let out_shape = node.shape.clone();
    let out_len = out_shape.num_elements().unwrap_or(0) * DType::F32.size_bytes();
    let mut out_bytes = vec![0u8; out_len];
    kernel
        .execute(&in_refs, (&mut out_bytes, &out_shape), attrs)
        .map_err(|e| CoremlError::Runtime(format!("Op::Custom('{name}'): {e}")))?;
    if !out_bytes.len().is_multiple_of(4) {
        return Err(CoremlError::Runtime(format!(
            "Op::Custom('{name}'): output not f32-aligned"
        )));
    }
    Ok(out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
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
    let _dirs = if bidirectional { 2 } else { 1 };
    let out_len = node.shape.num_elements().unwrap_or(0);
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
