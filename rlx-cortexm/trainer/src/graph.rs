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

//! TinyConv-MNIST graph builder.
//!
//! Architecture (NCHW, matching the PyTorch original):
//! ```text
//!   x         [B,  1, 28, 28]      input
//!   conv1     [B,  8, 26, 26]      3×3 valid conv, bias added per-channel
//!   relu1
//!   pool1     [B,  8, 13, 13]      2×2 stride-2 max-pool
//!   conv2     [B, 16, 11, 11]      3×3 valid conv, bias added per-channel
//!   relu2
//!   pool2     [B, 16,  5,  5]
//!   flatten   [B, 400]
//!   logits    [B, 10]              FC + bias
//!   loss      [B]                  softmax+cross-entropy per row
//!   mean      []                   batch-mean → scalar loss
//! ```
//!
//! `build_train_graph` returns the gradient graph plus all the NodeIds
//! the trainer needs to read/write at runtime: inputs, parameters,
//! gradients, the loss, and the post-SCE logits (added to the graph's
//! output list so its arena slot survives to end-of-execution).
//!
//! ## Channel-broadcast workaround
//!
//! Conv adds a per-channel bias `[C]` to a feature map `[B, C, H, W]`.
//! `rlx-cpu`'s `Op::Binary` lowering currently treats any
//! `out_len % rhs_len == 0` case as a last-axis bias, which would
//! produce `bc[0], bc[1], bc[0], bc[1], …` alternating across
//! positions — wrong for channel-broadcast. Workaround: reshape the
//! bias to `[1, C, 1, 1]`, then explicitly `Op::Expand` it to the
//! full `[B, C, H, W]` shape, then plain element-wise Add. This makes
//! the broadcast happen in the dedicated Expand thunk (correct
//! per-axis strides) and the Add becomes a pure same-shape op.

use rlx_ir::op::*;
use rlx_ir::*;

pub struct Spec {
    pub batch: usize,
    /// When `Some(bits)`, each weight tensor is wrapped in
    /// `Op::FakeQuantize { bits }` (per-output-channel) so the SGD
    /// optimizer sees the deployment-time round during training.
    /// `None` → no fake-quant; weights train at FP32 and only get
    /// quantized at the end (post-training quantization).
    pub qat_bits: Option<u8>,
}

/// Output of `build_train_graph` — every NodeId the trainer needs.
pub struct TrainGraph {
    pub graph: Graph,

    // Inputs (filled per iteration).
    pub input: NodeId,    // x          [B, 1, 28, 28]
    pub labels: NodeId,   // labels     [B]
    pub d_output: NodeId, // d/d_loss   scalar (always [1.0])

    // Parameters (initialised once, updated by SGD).
    pub params: Vec<ParamSlot>,

    // Outputs of the graph: [loss, grads..., logits].
    pub loss: NodeId,
    pub logits: NodeId,
}

/// Bookkeeping for a single trainable parameter.
pub struct ParamSlot {
    /// Human-readable name. Currently only used for logging at training
    /// startup, but the slot order matters — the calibrator and emitter
    /// rely on a specific layout (conv1_w, conv1_b, conv2_w, conv2_b,
    /// fc_w, fc_b).
    #[allow(dead_code)]
    pub name: &'static str,
    pub shape: Vec<usize>,
    /// NodeId for the parameter (read/write at runtime).
    pub param: NodeId,
    /// NodeId for the gradient w.r.t. this parameter.
    pub grad: NodeId,
}

impl ParamSlot {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

impl TrainGraph {
    /// Run `LegalizeBroadcast` on the inner graph and remap every
    /// `NodeId` we hold (input, labels, d_output, params, grads, loss,
    /// logits) into the new graph's coordinate space.
    pub fn legalize_broadcast(self) -> Self {
        let (new_graph, remap) = rlx_opt::legalize_broadcast::run_with_remap(self.graph);
        let r = |id: rlx_ir::NodeId| remap[&id];
        TrainGraph {
            graph: new_graph,
            input: r(self.input),
            labels: r(self.labels),
            d_output: r(self.d_output),
            params: self
                .params
                .into_iter()
                .map(|p| ParamSlot {
                    name: p.name,
                    shape: p.shape,
                    param: r(p.param),
                    grad: r(p.grad),
                })
                .collect(),
            loss: r(self.loss),
            logits: r(self.logits),
        }
    }
}

pub fn build_train_graph(spec: &Spec) -> TrainGraph {
    let f = DType::F32;
    let b = spec.batch;
    let mut g = Graph::new("tinyconv_train");

    // ── Inputs ────────────────────────────────────────────────
    let x = g.input("x", Shape::new(&[b, 1, 28, 28], f));
    let labels = g.input("labels", Shape::new(&[b], f));

    // ── Parameters ────────────────────────────────────────────
    let conv1_w = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[8], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[16], f));
    let fc_w = g.param("fc_w", Shape::new(&[400, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));

    // ── QAT: fake-quantize weights so SGD sees the rounding ────
    // Per-channel axis = 0 (output channels) for all three weight
    // tensors. The trainer's deployment-time quantizer also chooses
    // per-channel scales, so the training-time and inference-time
    // schemes match.
    let (conv1_w_q, conv2_w_q, fc_w_q) = if let Some(bits) = spec.qat_bits {
        let s_c1 = Shape::new(&[8, 1, 3, 3], f);
        let s_c2 = Shape::new(&[16, 8, 3, 3], f);
        let s_fc = Shape::new(&[400, 10], f);
        (
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![conv1_w],
                s_c1,
            ),
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![conv2_w],
                s_c2,
            ),
            g.add_node(
                Op::FakeQuantize {
                    bits,
                    axis: Some(0),
                    ste: rlx_ir::op::SteKind::default(),
                    scale_mode: rlx_ir::op::ScaleMode::default(),
                },
                vec![fc_w],
                s_fc,
            ),
        )
    } else {
        (conv1_w, conv2_w, fc_w)
    };

    // ── Conv block 1 ──────────────────────────────────────────
    let c1 = conv2d(&mut g, x, conv1_w_q, b, 1, 8, 28, 28, 26, 26);
    let c1 = bias_add_4d(&mut g, c1, conv1_b, b, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[b, 8, 26, 26], f));
    let p1 = maxpool(&mut g, c1, b, 8, 26, 26, 13, 13);

    // ── Conv block 2 ──────────────────────────────────────────
    let c2 = conv2d(&mut g, p1, conv2_w_q, b, 8, 16, 13, 13, 11, 11);
    let c2 = bias_add_4d(&mut g, c2, conv2_b, b, 16, 11, 11);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[b, 16, 11, 11], f));
    let p2 = maxpool(&mut g, c2, b, 16, 11, 11, 5, 5);

    // ── FC head ───────────────────────────────────────────────
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![b as i64, 400],
        },
        vec![p2],
        Shape::new(&[b, 400], f),
    );
    let mm = g.matmul(flat, fc_w_q, Shape::new(&[b, 10], f));
    // Last-axis bias add: rlx-cpu handles this case correctly via
    // `Thunk::BiasAdd` (rhs is the trailing dim, no channel-shape
    // confusion).
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[b, 10], f));

    // ── Loss ──────────────────────────────────────────────────
    let loss_per = g.softmax_cross_entropy_with_logits(logits, labels);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);

    // ── Build gradient graph ──────────────────────────────────
    let param_ids = vec![conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b];
    // `grad_with_loss` runs `rlx_autodiff::prepare_graph_for_ad` internally
    // (unfuse fused ops, scans, control-flow) before the VJP walk.
    let mut bwd = rlx_autodiff::grad_with_loss(&g, &param_ids);

    // Look up each NodeId's mirror in the bwd graph. The mirroring
    // preserves NodeId values (autodiff iterates `forward.nodes()` in
    // order and calls `add_node`, so id 0 maps to id 0, etc.), so the
    // forward NodeIds are valid in the bwd graph for all inputs and
    // parameters. The bwd graph's output list is currently
    // `[loss, g_conv1_w, ..., g_fc_b]`. Append `logits` so its arena
    // slot stays alive to end-of-execution and we can read predictions
    // for accuracy reporting.
    let mut outputs = bwd.outputs.clone();
    outputs.push(logits);
    bwd.set_outputs(outputs.clone());

    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("autodiff inserts an Input named `d_output`");

    let bwd_outputs = &bwd.outputs;
    let loss_id = bwd_outputs[0];
    let grad_ids: Vec<NodeId> = bwd_outputs[1..1 + param_ids.len()].to_vec();
    let logits_id = bwd_outputs[1 + param_ids.len()];

    let param_specs: Vec<(&'static str, Vec<usize>, NodeId)> = vec![
        ("conv1_w", vec![8, 1, 3, 3], conv1_w),
        ("conv1_b", vec![8], conv1_b),
        ("conv2_w", vec![16, 8, 3, 3], conv2_w),
        ("conv2_b", vec![16], conv2_b),
        ("fc_w", vec![400, 10], fc_w),
        ("fc_b", vec![10], fc_b),
    ];

    let params: Vec<ParamSlot> = param_specs
        .into_iter()
        .zip(grad_ids)
        .map(|((name, shape, param), grad)| ParamSlot {
            name,
            shape,
            param,
            grad,
        })
        .collect();

    TrainGraph {
        graph: bwd,
        input: x,
        labels,
        d_output,
        params,
        loss: loss_id,
        logits: logits_id,
    }
}

// ─────────────────────────── helpers ────────────────────────────

fn conv2d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    b: usize,
    c_in: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> NodeId {
    g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[b, c_out, h_out, w_out], DType::F32),
    )
    // c_in unused after shape calc; keep param for clarity at callsite.
    .tap(|_| {
        let _ = (c_in, h_in, w_in);
    })
}

// `tap` extension — passes a NodeId through unchanged after running a
// closure for side effects (used to silence unused-variable lints
// without restructuring the helper).
trait NodeIdExt {
    fn tap<F: FnOnce(&Self)>(self, f: F) -> Self;
}
impl NodeIdExt for NodeId {
    fn tap<F: FnOnce(&Self)>(self, f: F) -> Self {
        f(&self);
        self
    }
}

fn maxpool(
    g: &mut Graph,
    x: NodeId,
    b: usize,
    c: usize,
    _h_in: usize,
    _w_in: usize,
    h_out: usize,
    w_out: usize,
) -> NodeId {
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[b, c, h_out, w_out], DType::F32),
    )
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
    // Channel-broadcast: reshape `[C]` → `[1, C, 1, 1]` and let the
    // `LegalizeBroadcast` pass insert an Expand at compile time. We
    // emit a clean Op::Binary here; the pass guarantees the rlx-cpu
    // thunk sees same-shape operands. (Earlier versions of the
    // trainer materialized the Expand by hand because the pass didn't
    // exist yet.)
    let f = DType::F32;
    let bias_4d = g.add_node(
        Op::Reshape {
            new_shape: vec![1, c as i64, 1, 1],
        },
        vec![bias],
        Shape::new(&[1, c, 1, 1], f),
    );
    g.binary(BinaryOp::Add, x, bias_4d, Shape::new(&[b, c, h, w], f))
}
