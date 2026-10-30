// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run `Op::Fft` inside the CoreML program instead of on the host.
//!
//! MIL has no FFT op, so `Op::Fft` used to be a host op ([`crate::host_exec`]):
//! the graph was split around it and the transform ran on the CPU between
//! CoreML segments — slower than a plain CPU FFT, and never on the Neural
//! Engine or GPU. This pass rewrites each eligible `Op::Fft` into a constant
//! DFT matrix and one `MatMul`, which MIL lowers natively.
//!
//! In `Op::Fft`'s 2N real-block layout (`[re₀…re_{N-1}, im₀…im_{N-1}]` on the
//! last axis) the complex DFT is a single real matmul `y = x · W`,
//!
//! ```text
//!     W = scale · [[ C,   s·S ],
//!                  [ −s·S, C  ]]      C[t,k] = cos 2πtk/N,  S[t,k] = sin 2πtk/N
//! ```
//!
//! with `s = −1` forward / `+1` inverse and the normalisation folded into
//! `scale` ([`FftNorm::output_scale`]). Exact for every `N`, not just powers of
//! two.
//!
//! **Size cap.** `W` is `(2N)²` floats — 16 MiB at N = 1024, 64 MiB at 2048 —
//! and the matmul is O(N²) per row, so only rows up to [`max_n`] are rewritten;
//! longer transforms stay on the host. `RLX_COREML_FFT=host` disables the pass,
//! `RLX_COREML_FFT_MAX_N` moves the cap.
//!
//! **Precision.** The matrix is built in f64 and baked as f32. On the Neural
//! Engine CoreML computes in f16, which costs accuracy over long rows — the
//! tests measure it per compute unit.

use std::collections::HashMap;

use rlx_ir::fft::FftNorm;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Default largest row length (complex points) rewritten to a DFT matmul.
pub const DEFAULT_MAX_N: usize = 2048;

/// Largest row length rewritten (env `RLX_COREML_FFT_MAX_N`, default
/// [`DEFAULT_MAX_N`]).
pub fn max_n() -> usize {
    rlx_ir::env::var("RLX_COREML_FFT_MAX_N")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_MAX_N)
}

fn enabled() -> bool {
    !matches!(rlx_ir::env::var("RLX_COREML_FFT"), Some(v) if v.eq_ignore_ascii_case("host"))
}

/// How an eligible `Op::Fft` is lowered for CoreML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftForm {
    /// Leave it on the host (fastest on CPU / GPU: O(N log N)).
    Host,
    /// f32 DFT-matrix matmul inside the ML Program (CoreML places it on CPU).
    MatMul,
    /// f16 1×1-convolution island (`cast → conv → cast`) — the form CoreML
    /// puts on the Neural Engine. f16 costs accuracy: ~3e-4 relative at
    /// N = 1024 on the ANE.
    ConvF16,
}

/// Form for a model compiled for `units`: the conv island when the Neural
/// Engine is allowed, else the host FFT. `RLX_COREML_FFT=host|matmul|conv`
/// overrides.
pub fn form_for(units: crate::ComputeUnits) -> FftForm {
    match rlx_ir::env::var("RLX_COREML_FFT")
        .map(|v| v.to_ascii_lowercase())
        .as_deref()
    {
        Some("host") => FftForm::Host,
        Some("matmul") => FftForm::MatMul,
        Some("conv") => FftForm::ConvF16,
        _ => match units {
            crate::ComputeUnits::All | crate::ComputeUnits::CpuAndNeuralEngine => FftForm::ConvF16,
            _ => FftForm::Host,
        },
    }
}

/// True when the graph is essentially just FFTs: every node is an eligible
/// `Op::Fft` or cheap data movement around it (inputs, constants, reshape,
/// narrow, concat, …), with at least one FFT. Such a model is lowered whole as
/// f16 when the Neural Engine is allowed — CoreML keeps an f16 island inside
/// an f32-I/O model on the CPU, but places an all-f16 one on the ANE.
pub fn fft_only(g: &Graph) -> bool {
    let mut any_fft = false;
    for n in g.nodes() {
        match &n.op {
            Op::Fft { .. } => {
                if eligible(g, n.id).is_none() {
                    return false;
                }
                any_fft = true;
            }
            Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Reshape { .. }
            | Op::Narrow { .. }
            | Op::Concat { .. }
            | Op::Transpose { .. }
            | Op::Cast { .. } => {}
            _ => return false,
        }
    }
    any_fft
}

/// True when the graph's only host ops are FFTs this pass can move into the
/// model — so a caller choosing compute units can prefer the Neural Engine
/// instead of falling back to CPU-only for a host split.
pub fn only_offloadable_host_ops(g: &Graph) -> bool {
    let mut any_fft = false;
    for n in g.nodes() {
        if crate::host_exec::is_host_node(g, n.id) {
            if eligible(g, n.id).is_none() {
                return false;
            }
            any_fft = true;
        }
    }
    any_fft && enabled()
}

/// `(2n)²` row-major DFT matrix for the 2N real-block layout.
pub fn dft_block_matrix(n: usize, inverse: bool, norm: FftNorm) -> Vec<f32> {
    let w = 2 * n;
    let s = if inverse { 1.0 } else { -1.0 };
    let scale = norm.output_scale(n, inverse);
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut m = vec![0f32; w * w];
    for t in 0..n {
        for k in 0..n {
            // Reduce t·k mod n first: keeps the angle small, so cos/sin stay
            // accurate for large n.
            let ang = two_pi * ((t * k) % n) as f64 / n as f64;
            let (c, si) = (ang.cos() * scale, ang.sin() * scale);
            m[t * w + k] = c as f32; //            re_in → re_out
            m[t * w + n + k] = (s * si) as f32; //  re_in → im_out
            m[(n + t) * w + k] = (-s * si) as f32; // im_in → re_out
            m[(n + t) * w + n + k] = c as f32; //    im_in → im_out
        }
    }
    m
}

/// The row length `n` if `node` is an `Op::Fft` this pass rewrites.
fn eligible(g: &Graph, id: NodeId) -> Option<(usize, bool, FftNorm)> {
    let node = g.node(id);
    let Op::Fft { inverse, norm, .. } = node.op else {
        return None;
    };
    let xs = &g.node(node.inputs[0]).shape;
    if xs.dtype() != DType::F32 || node.shape.dtype() != DType::F32 || *xs != node.shape {
        return None;
    }
    if xs
        .dims()
        .iter()
        .any(|d| matches!(d, rlx_ir::Dim::Dynamic(_)))
    {
        return None;
    }
    let last = xs.dim(xs.rank() - 1).unwrap_static();
    if last == 0 || !last.is_multiple_of(2) || last / 2 > max_n() {
        return None;
    }
    Some((last / 2, inverse, norm))
}

/// Rewrite eligible `Op::Fft` nodes per `form` (see [`FftForm`]). Returns the
/// graph unchanged for [`FftForm::Host`] or when there is nothing to do.
pub fn lower(g: Graph, form: FftForm) -> Graph {
    if form == FftForm::Host
        || !enabled()
        || !g.nodes().iter().any(|n| eligible(&g, n.id).is_some())
    {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut mats: HashMap<(usize, bool, FftNorm), NodeId> = HashMap::new();
    for node in g.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if let Some((n, inverse, norm)) = eligible(&g, node.id) {
            let xs = node.shape.clone();
            let rank = xs.rank();
            let batch: usize = (0..rank - 1).map(|i| xs.dim(i).unwrap_static()).product();
            let conv = form == FftForm::ConvF16;
            let w = *mats.entry((n, inverse, norm)).or_insert_with(|| {
                let m = dft_block_matrix(n, inverse, norm);
                let w2 = 2 * n;
                // Conv weight is [C_out, C_in, 1, 1] = Wᵀ.
                let data: Vec<f32> = if conv {
                    (0..w2 * w2).map(|i| m[(i % w2) * w2 + i / w2]).collect()
                } else {
                    m
                };
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dims: Vec<usize> = if conv {
                    vec![w2, w2, 1, 1]
                } else {
                    vec![w2, w2]
                };
                out.add_node(
                    Op::Constant { data: bytes },
                    vec![],
                    Shape::new(&dims, DType::F32),
                )
            });
            let flat = Shape::new(&[batch, 2 * n], DType::F32);
            let y = if conv {
                let nchw = Shape::new(&[batch, 2 * n, 1, 1], DType::F32);
                let x4 = out.reshape(
                    new_inputs[0],
                    vec![batch as i64, (2 * n) as i64, 1, 1],
                    nchw,
                );
                let y4 = out.conv2d(x4, w, [1, 1], [1, 1], [0, 0], [1, 1], 1);
                out.reshape(y4, vec![batch as i64, (2 * n) as i64], flat)
            } else {
                let x2 = out.reshape(
                    new_inputs[0],
                    vec![batch as i64, (2 * n) as i64],
                    flat.clone(),
                );
                out.matmul(x2, w, flat)
            };
            let dims: Vec<i64> = (0..rank)
                .map(|i| xs.dim(i).unwrap_static() as i64)
                .collect();
            out.reshape(y, dims, xs)
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive complex DFT on the 2N block layout, f64 reference.
    fn reference(row: &[f32], inverse: bool, norm: FftNorm) -> Vec<f64> {
        let n = row.len() / 2;
        let s = if inverse { 1.0 } else { -1.0 };
        let scale = norm.output_scale(n, inverse);
        let mut out = vec![0f64; 2 * n];
        for k in 0..n {
            let (mut re, mut im) = (0f64, 0f64);
            for t in 0..n {
                let ang = 2.0 * core::f64::consts::PI * ((t * k) % n) as f64 / n as f64;
                let (a, b) = (row[t] as f64, row[n + t] as f64);
                re += a * ang.cos() - b * s * ang.sin();
                im += a * s * ang.sin() + b * ang.cos();
            }
            out[k] = re * scale;
            out[n + k] = im * scale;
        }
        out
    }

    #[test]
    fn block_matrix_matches_the_dft() {
        for (n, inverse, norm) in [
            (8, false, FftNorm::Backward),
            (16, true, FftNorm::Forward),
            (12, false, FftNorm::Ortho), // not a power of two
            (5, true, FftNorm::Backward),
        ] {
            let row: Vec<f32> = (0..2 * n)
                .map(|i| ((i * 7 + 3) as f32 * 0.37).sin())
                .collect();
            let w = dft_block_matrix(n, inverse, norm);
            let want = reference(&row, inverse, norm);
            for k in 0..2 * n {
                let got: f64 = (0..2 * n)
                    .map(|t| row[t] as f64 * w[t * 2 * n + k] as f64)
                    .sum();
                assert!(
                    (got - want[k]).abs() < 1e-4,
                    "n={n} inv={inverse} k={k}: {got} vs {}",
                    want[k]
                );
            }
        }
    }

    #[test]
    fn rewrites_eligible_fft_and_leaves_no_host_op() {
        let mut g = Graph::new("fft");
        let x = g.input("x", rlx_ir::Shape::new(&[3, 2, 64], DType::F32));
        let y = g.add_node(
            Op::Fft {
                inverse: false,
                norm: FftNorm::Backward,
                algo: Default::default(),
            },
            vec![x],
            rlx_ir::Shape::new(&[3, 2, 64], DType::F32),
        );
        g.set_outputs(vec![y]);
        let out = lower(g, FftForm::MatMul);
        assert!(!out.nodes().iter().any(|n| matches!(n.op, Op::Fft { .. })));
        assert!(
            !out.nodes()
                .iter()
                .any(|n| crate::host_exec::is_host_node(&out, n.id))
        );
        assert_eq!(
            out.node(out.outputs[0]).shape,
            rlx_ir::Shape::new(&[3, 2, 64], DType::F32)
        );
    }

    #[test]
    fn rows_over_the_cap_stay_on_the_host() {
        let mut g = Graph::new("big");
        let len = 2 * (DEFAULT_MAX_N * 2);
        let x = g.input("x", rlx_ir::Shape::new(&[1, len], DType::F32));
        let y = g.add_node(
            Op::Fft {
                inverse: false,
                norm: FftNorm::Backward,
                algo: Default::default(),
            },
            vec![x],
            rlx_ir::Shape::new(&[1, len], DType::F32),
        );
        g.set_outputs(vec![y]);
        let out = lower(g, FftForm::MatMul);
        assert!(out.nodes().iter().any(|n| matches!(n.op, Op::Fft { .. })));
    }
}
