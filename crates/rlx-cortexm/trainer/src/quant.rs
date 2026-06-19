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

//! Activation calibration + INT8 quantization.
//!
//! Mirrors the per-tensor symmetric scheme used by `tools/train_mnist.py`:
//!
//! 1. Run a *forward-only* graph (no SCE, no loss, no gradients) that
//!    exposes the input + each post-activation feature map as outputs.
//!    Walk a calibration batch through it, take the max-abs at each tap.
//!    Divide by 127 to get the per-tensor scale.
//! 2. Quantize each weight tensor per-tensor symmetric (`scale = max_abs / 127`).
//!    Bias is quantized to i32 in the accumulator scale
//!    (`x_scale * w_scale`) — same as TFLite Micro / CMSIS-NN.
//! 3. Permute conv weights from PyTorch-style `[O, I, kH, kW]` (which is
//!    what `Op::Conv` consumes) into the NHWC-friendly `[O, kH, kW, I]`
//!    layout the firmware kernel reads.
//! 4. The FC layer trains in NCHW (so the flatten reads `[c, h, w]`
//!    C-major) but the firmware flattens the NHWC pool output
//!    `[h, w, c]` H-major. Permute the FC input axis to match.

use rlx_ir::op::*;
use rlx_ir::*;

use crate::Args;
use crate::mnist::{Dataset, PIXELS};
use crate::train::{TrainedModel, write_arena};

/// Per-layer bit-width assignment, used by `--mixed-precision` mode.
/// `None` for any layer means "use args.weight_bits as the global
/// fallback".
#[derive(Debug, Clone, Copy, Default)]
pub struct PerLayerBits {
    pub conv1: Option<u8>,
    pub conv2: Option<u8>,
    pub fc: Option<u8>,
}

impl PerLayerBits {
    /// All three layers at the same bit width.
    pub fn uniform(bits: u8) -> Self {
        Self {
            conv1: Some(bits),
            conv2: Some(bits),
            fc: Some(bits),
        }
    }
    pub fn for_conv1(&self, fallback: u8) -> u8 {
        self.conv1.unwrap_or(fallback)
    }
    pub fn for_conv2(&self, fallback: u8) -> u8 {
        self.conv2.unwrap_or(fallback)
    }
    pub fn for_fc(&self, fallback: u8) -> u8 {
        self.fc.unwrap_or(fallback)
    }

    /// Total weight storage in bytes, given the model shapes.
    pub fn weight_bytes(&self, fallback: u8) -> usize {
        let c1 = self.for_conv1(fallback) as usize;
        let c2 = self.for_conv2(fallback) as usize;
        let fc = self.for_fc(fallback) as usize;
        let n_c1w = 8 * 3 * 3;
        let n_c2w = 16 * 8 * 3 * 3;
        let n_fcw = 10 * 400;
        (n_c1w * c1 + n_c2w * c2 + n_fcw * fc).div_ceil(8)
    }
}

/// Search the smallest per-layer bit-width assignment that meets
/// `target_acc` on the given calibration set. Tries assignments
/// from smallest (i2/i2/i2) to largest (i8/i8/i8) and returns the
/// first that satisfies the floor.
///
/// This is a trainer-side helper; it doesn't touch the IR. The
/// caller re-runs the full quantize+host-validate path for each
/// candidate. Cheap — TinyConv quantizes in milliseconds.
pub fn search_per_layer_bits(
    target_acc: f64,
    candidates: &[PerLayerBits],
    mut measure: impl FnMut(PerLayerBits) -> f64,
) -> Option<(PerLayerBits, f64)> {
    let mut sorted: Vec<_> = candidates.to_vec();
    sorted.sort_by_key(|c| c.weight_bytes(8));
    for cand in sorted {
        let acc = measure(cand);
        if acc >= target_acc {
            return Some((cand, acc));
        }
    }
    None
}

/// Final quantized model — what `emit` writes out as Rust source.
///
/// Conv and FC weights are quantized **per output channel**: each
/// `oc` row gets its own `w_scale[oc]`, biases live in the matching
/// per-channel acc-scale, and the kernel epilogue applies a
/// per-channel `mult[oc] = (in_scale * w_scale[oc]) / out_scale`.
/// Per-channel typically buys 0.5–2 pp on small conv stacks because
/// the loud filters stop saturating the rest into noise.
pub struct QuantizedModel {
    pub conv1_w: Vec<i8>,  // [c_out=8, kH=3, kW=3, c_in=1]   — NHWC layout
    pub conv1_b: Vec<i32>, // [8]
    pub conv2_w: Vec<i8>,  // [16, 3, 3, 8]
    pub conv2_b: Vec<i32>, // [16]
    pub fc_w: Vec<i8>,     // [10, 400]   — input axis is (h, w, c) HWC
    pub fc_b: Vec<i32>,    // [10]

    pub x_scale: f32,
    pub c1_scale: f32,
    pub p1_scale: f32,
    pub c2_scale: f32,
    pub p2_scale: f32,
    /// Per-output-channel weight scales.
    pub w1_scale: Vec<f32>, // [8]
    pub w2_scale: Vec<f32>,  // [16]
    pub wfc_scale: Vec<f32>, // [10]
    pub fc_out_scale: f32,

    /// Embedded one-image e2e test (`TEST_IMAGE`/`TEST_LABEL` in the
    /// emitted file). The `test_image` is i8 NHWC `[28, 28, 1]`.
    pub test_image: Vec<i8>,
    pub test_label: u8,

    pub fp32_test_accuracy: f64,
    /// Bits per weight: 8, 4, or 2. Determines whether `conv*_w` /
    /// `fc_w` are raw i8 (one weight per byte) or packed (2 per byte
    /// for i4, 4 per byte for i2). Activations are always i8.
    pub weight_bits: u8,
}

impl QuantizedModel {
    /// Per-channel requantization multiplier for conv1.
    pub fn conv1_mult(&self) -> Vec<f32> {
        self.w1_scale
            .iter()
            .map(|&w| (self.x_scale * w) / self.c1_scale)
            .collect()
    }
    pub fn conv2_mult(&self) -> Vec<f32> {
        self.w2_scale
            .iter()
            .map(|&w| (self.p1_scale * w) / self.c2_scale)
            .collect()
    }
    pub fn fc_mult(&self) -> Vec<f32> {
        self.wfc_scale
            .iter()
            .map(|&w| (self.p2_scale * w) / self.fc_out_scale)
            .collect()
    }
}

pub fn calibrate_and_quantize(
    model: &TrainedModel,
    dataset: &Dataset,
    args: &Args,
) -> Result<QuantizedModel, String> {
    let scales = run_calibration(model, dataset, args)?;
    eprintln!(
        "calibration: x={:.4e} c1={:.4e} p1={:.4e} c2={:.4e} p2={:.4e} (weight_bits={})",
        scales.x, scales.c1, scales.p1, scales.c2, scales.p2, args.weight_bits
    );

    // ── Quantize weights (per output channel, packed at args.weight_bits) ──
    let bits = args.weight_bits;
    let (conv1_w_q, w1_scale) = quantize_conv_weight(&model.conv1_w, &[8, 1, 3, 3], bits);
    let (conv2_w_q, w2_scale) = quantize_conv_weight(&model.conv2_w, &[16, 8, 3, 3], bits);
    let (fc_w_q, wfc_scale) = quantize_fc_weight(
        &model.fc_w,
        /*c=*/ 16,
        /*h=*/ 5,
        /*w=*/ 5,
        bits,
    );

    // ── Quantize biases (i32 in per-channel acc-scale) ──────
    let conv1_b_q = quantize_bias_per_channel(&model.conv1_b, scales.x, &w1_scale);
    let conv2_b_q = quantize_bias_per_channel(&model.conv2_b, scales.p1, &w2_scale);
    let fc_b_q = quantize_bias_per_channel(&model.fc_b, scales.p2, &wfc_scale);

    // ── FC output scale ─────────────────────────────────────
    // Pick to span the observed logit range; matches the Python script.
    let fc_out_scale = compute_fc_out_scale(model, dataset, args)?;

    // ── Embedded test image ─────────────────────────────────
    let (test_image, test_label) = build_embedded_test(dataset, scales.x);

    Ok(QuantizedModel {
        conv1_w: conv1_w_q,
        conv1_b: conv1_b_q,
        conv2_w: conv2_w_q,
        conv2_b: conv2_b_q,
        fc_w: fc_w_q,
        fc_b: fc_b_q,
        x_scale: scales.x,
        c1_scale: scales.c1,
        p1_scale: scales.p1,
        c2_scale: scales.c2,
        p2_scale: scales.p2,
        w1_scale,
        w2_scale,
        wfc_scale,
        fc_out_scale,
        test_image,
        test_label,
        fp32_test_accuracy: model.fp32_test_accuracy,
        weight_bits: args.weight_bits,
    })
}

// ─────────────────────── Calibration ─────────────────────────

struct ActivationScales {
    x: f32,
    c1: f32,
    p1: f32,
    c2: f32,
    p2: f32,
}

fn run_calibration(
    model: &TrainedModel,
    dataset: &Dataset,
    args: &Args,
) -> Result<ActivationScales, String> {
    // Build a forward-only graph (no SCE, no grads). Add each
    // calibration tap to the output list so the memory planner keeps
    // its arena slot alive to end-of-execution.
    let f = DType::F32;
    let b = args.batch;
    let mut g = Graph::new("calibration");
    let xn = g.input("x", Shape::new(&[b, 1, 28, 28], f));
    let w1 = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let b1 = g.param("conv1_b", Shape::new(&[8], f));
    let w2 = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let b2 = g.param("conv2_b", Shape::new(&[16], f));

    // Wrap weights in FakeQuantize when QAT is on so the calibration
    // graph sees the SAME quantized weights the firmware will use at
    // deployment time. The fake-quant op output is what the conv
    // reads; the *original* param NodeIds (`w1`, `w2`) are what the
    // caller writes trained weights into, so don't shadow them.
    let (w1_use, w2_use) = if args.qat_enabled() {
        let bits = args.weight_bits;
        (
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![w1],
                Shape::new(&[8, 1, 3, 3], f),
            ),
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![w2],
                Shape::new(&[16, 8, 3, 3], f),
            ),
        )
    } else {
        (w1, w2)
    };

    let c1 = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![xn, w1_use],
        Shape::new(&[b, 8, 26, 26], f),
    );
    let c1 = bias_add_4d(&mut g, c1, b1, b, 8, 26, 26);
    let c1_relu = g.activation(Activation::Relu, c1, Shape::new(&[b, 8, 26, 26], f));
    let p1 = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![c1_relu],
        Shape::new(&[b, 8, 13, 13], f),
    );

    let c2 = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![p1, w2_use],
        Shape::new(&[b, 16, 11, 11], f),
    );
    let c2 = bias_add_4d(&mut g, c2, b2, b, 16, 11, 11);
    let c2_relu = g.activation(Activation::Relu, c2, Shape::new(&[b, 16, 11, 11], f));
    let p2 = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![c2_relu],
        Shape::new(&[b, 16, 5, 5], f),
    );
    g.set_outputs(vec![xn, c1_relu, p1, c2_relu, p2]);

    let (g, remap) = rlx_opt::legalize_broadcast::run_with_remap(g);
    let xn = remap[&xn];
    let c1_relu = remap[&c1_relu];
    let p1 = remap[&p1];
    let c2_relu = remap[&c2_relu];
    let p2 = remap[&p2];
    let w1 = remap[&w1];
    let b1 = remap[&b1];
    let w2 = remap[&w2];
    let b2 = remap[&b2];

    // ── Build the calibrator ──────────────────────────────────
    // Order matters: [x, c1_relu, p1, c2_relu, p2] — `Calibrator::scales()`
    // returns parallel-indexed Vec<f32> we destructure below.
    let mut cal = rlx_cpu::calibrate::Calibrator::new(&g, vec![xn, c1_relu, p1, c2_relu, p2]);
    crate::train::fill_constants_into_arena(&g, cal.arena_mut());
    write_arena(cal.arena_mut(), w1, &model.conv1_w);
    write_arena(cal.arena_mut(), b1, &model.conv1_b);
    write_arena(cal.arena_mut(), w2, &model.conv2_w);
    write_arena(cal.arena_mut(), b2, &model.conv2_b);

    let n_batches = 10.min(dataset.train.len() / b);
    for bi in 0..n_batches {
        // Pack one batch of images into the input slot.
        let img_off = cal.arena().byte_offset(xn);
        let buf = cal.arena_mut().raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(img_off) as *mut f32;
            for i in 0..b {
                let src = dataset.train.image(bi * b + i);
                for j in 0..PIXELS {
                    *p.add(i * PIXELS + j) = src[j];
                }
            }
        }
        cal.step();
    }

    let s = cal.scales();
    Ok(ActivationScales {
        x: s[0],
        c1: s[1],
        p1: s[2],
        c2: s[3],
        p2: s[4],
    })
}

fn bias_add_4d(
    g: &mut Graph,
    x: NodeId,
    bias: NodeId,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> NodeId {
    // Same workaround pattern as `crate::graph::bias_add_4d`: emit a
    // `[1, C, 1, 1]` reshape + plain Op::Binary; the
    // `LegalizeBroadcast` pass that runs ahead of `compile_thunks`
    // will materialize the broadcast via Op::Expand.
    let f = DType::F32;
    let _ = b;
    let _ = h;
    let _ = w;
    let bias_4d = g.add_node(
        Op::Reshape {
            new_shape: vec![1, c as i64, 1, 1],
        },
        vec![bias],
        Shape::new(&[1, c, 1, 1], f),
    );
    g.binary(BinaryOp::Add, x, bias_4d, Shape::new(&[b, c, h, w], f))
}

// ───────────────────── Weight quantization ────────────────────

fn quantize_conv_weight(w: &[f32], shape_oihw: &[usize], bits: u8) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(w.len(), shape_oihw.iter().product::<usize>());
    let (o, i, kh, kw) = (shape_oihw[0], shape_oihw[1], shape_oihw[2], shape_oihw[3]);
    // Permute [O, I, kH, kW] → [O, kH, kW, I] for NHWC firmware kernel.
    let mut nhwc = vec![0f32; w.len()];
    for oc in 0..o {
        for ic in 0..i {
            for h in 0..kh {
                for ww in 0..kw {
                    let src = ((oc * i + ic) * kh + h) * kw + ww;
                    let dst = ((oc * kh + h) * kw + ww) * i + ic;
                    nhwc[dst] = w[src];
                }
            }
        }
    }
    // Per-output-channel scale and quantize-then-pack.
    let row_len = kh * kw * i;
    let q_max = max_pos_code(bits);
    let scales: Vec<f32> = (0..o)
        .map(|oc| symmetric_scale_for_bits(&nhwc[oc * row_len..(oc + 1) * row_len], q_max))
        .collect();
    // Logical i8 buffer first; then bit-pack.
    let mut logical = vec![0i8; nhwc.len()];
    for oc in 0..o {
        let s = scales[oc];
        for k in 0..row_len {
            logical[oc * row_len + k] =
                sat_to_bits((nhwc[oc * row_len + k] / s).round() as i32, bits);
        }
    }
    (pack_bits(&logical, bits), scales)
}

/// Permute the FC weight's input axis from C-major (the layout used
/// by `Op::Reshape` after NCHW pooling: `idx = c*H*W + h*W + w`) to
/// HWC-major (the layout the firmware kernel sees from NHWC pool
/// output: `idx = h*W*C + w*C + c`), then transpose [I, O] → [O, I]
/// so the emitted weight matches `dense_i8`'s expected `[O, I]`
/// row-major layout.
fn quantize_fc_weight(
    w_io: &[f32],
    c: usize,
    h: usize,
    ww: usize,
    bits: u8,
) -> (Vec<i8>, Vec<f32>) {
    // w_io shape [I=c*h*ww, O=10] row-major.
    let i = c * h * ww;
    let o = w_io.len() / i;
    debug_assert_eq!(o * i, w_io.len());
    let mut w_oi = vec![0f32; w_io.len()];
    for k in 0..i {
        // Decompose k = c_idx * H * W + h_idx * W + w_idx (NCHW C-major).
        let c_idx = k / (h * ww);
        let r = k % (h * ww);
        let h_idx = r / ww;
        let w_idx = r % ww;
        // Re-encode in NHWC HWC-major.
        let k_hwc = h_idx * ww * c + w_idx * c + c_idx;
        for oc in 0..o {
            // Transpose [I, O] → [O, I] simultaneously.
            w_oi[oc * i + k_hwc] = w_io[k * o + oc];
        }
    }
    let q_max = max_pos_code(bits);
    let scales: Vec<f32> = (0..o)
        .map(|oc| symmetric_scale_for_bits(&w_oi[oc * i..(oc + 1) * i], q_max))
        .collect();
    let mut logical = vec![0i8; w_oi.len()];
    for oc in 0..o {
        let s = scales[oc];
        for k in 0..i {
            logical[oc * i + k] = sat_to_bits((w_oi[oc * i + k] / s).round() as i32, bits);
        }
    }
    (pack_bits(&logical, bits), scales)
}

fn quantize_bias_per_channel(b: &[f32], in_scale: f32, w_scale: &[f32]) -> Vec<i32> {
    debug_assert_eq!(b.len(), w_scale.len());
    b.iter()
        .zip(w_scale.iter())
        .map(|(&x, &ws)| {
            let acc = in_scale * ws;
            let v = (x / acc).round();
            v.clamp(i32::MIN as f32, i32::MAX as f32) as i32
        })
        .collect()
}

/// Largest positive code at a given weight bit width (symmetric range
/// `[-q_max, q_max]`; the asymmetric 2's-complement codepoint at the
/// negative end is left unused so quantization stays balanced).
///   8 → 127, 4 → 7, 2 → 1 (ternary)
fn max_pos_code(bits: u8) -> i32 {
    match bits {
        8 => 127,
        4 => 7,
        2 => 1,
        _ => panic!("unsupported weight_bits {bits}"),
    }
}

fn symmetric_scale_for_bits(t: &[f32], q_max: i32) -> f32 {
    let m = t.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    (m / q_max as f32).max(1e-12)
}

fn sat_to_bits(v: i32, bits: u8) -> i8 {
    let q = max_pos_code(bits);
    v.clamp(-q, q) as i8
}

/// Pack an `[i8]` slice of values within `[-q_max, q_max]` into bytes
/// at `bits` bits per element. For `bits=8` this is a no-op
/// (`logical → logical`). For `bits=4` two values share a byte: the
/// lower nibble is `logical[2k]`, the upper is `logical[2k+1]`. For
/// `bits=2` four values share a byte, ordered LSB → MSB by lane.
/// Sign-magnitude doesn't apply — values are stored 2's-complement
/// in the low `bits` bits and zero-extended; the firmware kernel
/// sign-extends on read.
fn pack_bits(logical: &[i8], bits: u8) -> Vec<i8> {
    if bits == 8 {
        return logical.to_vec();
    }
    let mask: u8 = match bits {
        4 => 0x0F,
        2 => 0x03,
        _ => unreachable!(),
    };
    let per_byte = (8 / bits) as usize;
    let n_bytes = logical.len().div_ceil(per_byte);
    let mut out = vec![0u8; n_bytes];
    for (i, &v) in logical.iter().enumerate() {
        let byte_idx = i / per_byte;
        let lane = (i % per_byte) * (bits as usize);
        out[byte_idx] |= ((v as u8) & mask) << lane;
    }
    out.into_iter().map(|b| b as i8).collect()
}

fn sat_i8(v: i32) -> i8 {
    v.clamp(-127, 127) as i8
}

// ───────────────── FC output scale (from logits) ─────────────

fn compute_fc_out_scale(
    model: &TrainedModel,
    dataset: &Dataset,
    args: &Args,
) -> Result<f32, String> {
    // Run a full forward over one batch, take the logits' max-abs.
    let f = DType::F32;
    let b = args.batch;
    let mut g = Graph::new("logits_calibration");
    let xn = g.input("x", Shape::new(&[b, 1, 28, 28], f));
    let w1 = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let b1 = g.param("conv1_b", Shape::new(&[8], f));
    let w2 = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let b2 = g.param("conv2_b", Shape::new(&[16], f));
    let wfc = g.param("fc_w", Shape::new(&[400, 10], f));
    let bfc = g.param("fc_b", Shape::new(&[10], f));

    // Same QAT wrap as in run_calibration — keeps the FC-output scale
    // calibrated against the deployment-quantized weights. Don't shadow
    // the original param NodeIds; the caller writes weights into those.
    let (w1_use, w2_use, wfc_use) = if args.qat_enabled() {
        let bits = args.weight_bits;
        (
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![w1],
                Shape::new(&[8, 1, 3, 3], f),
            ),
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![w2],
                Shape::new(&[16, 8, 3, 3], f),
            ),
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![wfc],
                Shape::new(&[400, 10], f),
            ),
        )
    } else {
        (w1, w2, wfc)
    };

    let c1 = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![xn, w1_use],
        Shape::new(&[b, 8, 26, 26], f),
    );
    let c1 = bias_add_4d(&mut g, c1, b1, b, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[b, 8, 26, 26], f));
    let p1 = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![c1],
        Shape::new(&[b, 8, 13, 13], f),
    );
    let c2 = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![p1, w2_use],
        Shape::new(&[b, 16, 11, 11], f),
    );
    let c2 = bias_add_4d(&mut g, c2, b2, b, 16, 11, 11);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[b, 16, 11, 11], f));
    let p2 = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![c2],
        Shape::new(&[b, 16, 5, 5], f),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, 400],
        },
        vec![p2],
        Shape::new(&[b, 400], f),
    );
    let mm = g.matmul(flat, wfc_use, Shape::new(&[b, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, bfc, Shape::new(&[b, 10], f));
    g.set_outputs(vec![logits]);

    let (g, remap) = rlx_opt::legalize_broadcast::run_with_remap(g);
    let xn = remap[&xn];
    let logits = remap[&logits];
    let w1 = remap[&w1];
    let b1 = remap[&b1];
    let w2 = remap[&w2];
    let b2 = remap[&b2];
    let wfc = remap[&wfc];
    let bfc = remap[&bfc];

    let mut cal = rlx_cpu::calibrate::Calibrator::new(&g, vec![logits]);
    crate::train::fill_constants_into_arena(&g, cal.arena_mut());
    write_arena(cal.arena_mut(), w1, &model.conv1_w);
    write_arena(cal.arena_mut(), b1, &model.conv1_b);
    write_arena(cal.arena_mut(), w2, &model.conv2_w);
    write_arena(cal.arena_mut(), b2, &model.conv2_b);
    write_arena(cal.arena_mut(), wfc, &model.fc_w);
    write_arena(cal.arena_mut(), bfc, &model.fc_b);

    // Single batch is enough — we just want the typical logit range.
    let img_off = cal.arena().byte_offset(xn);
    let buf = cal.arena_mut().raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(img_off) as *mut f32;
        for i in 0..b {
            let src = dataset.train.image(i);
            for j in 0..PIXELS {
                *p.add(i * PIXELS + j) = src[j];
            }
        }
    }
    cal.step();
    Ok(cal.scales()[0])
}

// ──────────────────── Embedded e2e test image ─────────────────

fn build_embedded_test(dataset: &Dataset, x_scale: f32) -> (Vec<i8>, u8) {
    // First test-set image, quantized with the trained x_scale.
    let img = dataset.test.image(0);
    let q: Vec<i8> = img
        .iter()
        .map(|&v| sat_i8((v / x_scale).round() as i32))
        .collect();
    let label = dataset.test.labels[0] as u8;
    (q, label)
}
