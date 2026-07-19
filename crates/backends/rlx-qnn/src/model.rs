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

//! Lightweight description of the subgraph we know how to lower to a QNN model.
//!
//! Supported shapes:
//!
//! * rank-2 `MatMul`: `out[M,N] = in0[M,K] · in1[K,N]`
//! * `Linear`: `out = in0 · in1 + in2` with the same static `M`/`K`/`N`
//! * `LinearRelu`: `out = relu(in0 · in1 + in2)` with the same static `M`/`K`/`N`
//! * `MatMulSoftmax`: `out = softmax(in0 · in1, axis=1)` with the same static dims
//! * `Mlp2`: two layers — `LinearRelu(M,K,H)` then `Linear(M,H,N)`
//! * `LinearStatic`: `out = in0 · W + b` with `W`/`b` baked as QNN `STATIC`
//!
//! [`Model::from_graph`] recognizes any of these patterns in an `rlx-ir`
//! [`Graph`]; the `Model::*` constructors build them directly for tests / CLI.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{Graph, Op};

/// A single operation we can emit as a QNN node.
#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    /// `out[m,n] = in0[m,k] · in1[k,n]` (row-major), f32.
    MatMul {
        name: String,
        m: usize,
        k: usize,
        n: usize,
    },
    /// `out[m,n] = in0[m,k] · in1[k,n] + in2[m,n]`, f32.
    Linear {
        name: String,
        m: usize,
        k: usize,
        n: usize,
    },
    /// `out[m,n] = relu(in0[m,k] · in1[k,n] + in2[m,n])`, f32.
    LinearRelu {
        name: String,
        m: usize,
        k: usize,
        n: usize,
    },
    /// `out[m,n] = softmax(in0[m,k] · in1[k,n], axis=1)`, f32.
    MatMulSoftmax {
        name: String,
        m: usize,
        k: usize,
        n: usize,
    },
    /// `out[m,n] = in0[m,k] · W[k,n] + b[m,n]` with `W`/`b` as QNN `STATIC`.
    /// `w` is `[K·N]` row-major; `b` is `[M·N]` row-major.
    LinearStatic {
        name: String,
        m: usize,
        k: usize,
        n: usize,
        w: Vec<f32>,
        b: Vec<f32>,
    },
}

/// A model to emit: a name and the ordered layers.
///
/// Single-layer patterns use one [`Layer`]; [`Model::mlp2`] stacks
/// `LinearRelu` then `Linear` (dims `M×K→H` then `M×H→N`).
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub name: String,
    pub layers: Vec<Layer>,
}

impl Model {
    /// Build a one-matmul model directly.
    pub fn single_matmul(name: impl Into<String>, m: usize, k: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            layers: vec![Layer::MatMul {
                name: "matmul_0".to_string(),
                m,
                k,
                n,
            }],
        }
    }

    /// Build a one-layer linear model directly.
    pub fn linear(name: impl Into<String>, m: usize, k: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            layers: vec![Layer::Linear {
                name: "linear_0".to_string(),
                m,
                k,
                n,
            }],
        }
    }

    /// Build a one-layer linear+relu model directly.
    pub fn linear_relu(name: impl Into<String>, m: usize, k: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            layers: vec![Layer::LinearRelu {
                name: "linear_relu_0".to_string(),
                m,
                k,
                n,
            }],
        }
    }

    /// Build a matmul+softmax model directly (`softmax` on the last axis).
    pub fn matmul_softmax(name: impl Into<String>, m: usize, k: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            layers: vec![Layer::MatMulSoftmax {
                name: "matmul_softmax_0".to_string(),
                m,
                k,
                n,
            }],
        }
    }

    /// Two-layer MLP: `out = relu(in0·w1+b1)·w2+b2` with hidden width `h`.
    pub fn mlp2(name: impl Into<String>, m: usize, k: usize, h: usize, n: usize) -> Self {
        Self {
            name: name.into(),
            layers: vec![
                Layer::LinearRelu {
                    name: "linear_relu_0".to_string(),
                    m,
                    k,
                    n: h,
                },
                Layer::Linear {
                    name: "linear_1".to_string(),
                    m,
                    k: h,
                    n,
                },
            ],
        }
    }

    /// Linear with baked STATIC weight/bias (activation-only runtime input).
    /// Uses deterministic seed-0 weights (same LCG as the CLI emit path).
    pub fn linear_static(name: impl Into<String>, m: usize, k: usize, n: usize) -> Self {
        let (w, b) = seed0_weight_bias(m, k, n);
        Self::linear_static_with_weights(name, m, k, n, w, b)
    }

    /// Linear with caller-supplied STATIC weight/bias.
    pub fn linear_static_with_weights(
        name: impl Into<String>,
        m: usize,
        k: usize,
        n: usize,
        w: Vec<f32>,
        b: Vec<f32>,
    ) -> Self {
        assert_eq!(w.len(), k * n, "w must be K*N");
        assert_eq!(b.len(), m * n, "b must be M*N");
        Self {
            name: name.into(),
            layers: vec![Layer::LinearStatic {
                name: "linear_static_0".to_string(),
                m,
                k,
                n,
                w,
                b,
            }],
        }
    }

    /// Recognize the supported subgraph in an `rlx-ir` graph.
    ///
    /// Single-output graphs: rank-2 [`Op::MatMul`], `Add(MatMul, …)` (or
    /// `LinearStatic` when weight+bias are f32 [`Op::Constant`]),
    /// `Relu(Add(MatMul, …))`, `Softmax(MatMul, axis=1)`, or two-layer
    /// `Add(MatMul(Relu(Add(MatMul,…)),…),…)` (`Mlp2`). Anything else is a
    /// clear `Err` rather than a silent partial lowering.
    pub fn from_graph(g: &Graph) -> Result<Self, String> {
        // `Op::Scan` (recurrence, e.g. IIR biquad) has no QNN representation —
        // unroll it into inlined body replicas (primitives) first.
        let scan_unrolled;
        let g = if g.nodes().iter().any(|n| matches!(n.op, Op::Scan { .. })) {
            use rlx_opt::pass::Pass as _;
            scan_unrolled = rlx_opt::LowerScan.run(g.clone());
            &scan_unrolled
        } else {
            g
        };
        if g.outputs.len() != 1 {
            return Err(format!(
                "rlx-qnn codegen lowers a single-output graph; got {} outputs",
                g.outputs.len()
            ));
        }
        let out = g.node(g.outputs[0]);
        match &out.op {
            Op::MatMul => Self::matmul_layer_from_node(g, out),
            Op::Binary(BinaryOp::Add) => {
                // Prefer Mlp2 when the Add's MatMul lhs is a LinearRelu tower.
                if let Ok(m) = Self::mlp2_from_graph(g, out) {
                    return Ok(m);
                }
                // Prefer LinearStatic when W and b are f32 Constants.
                if let Ok(m) = Self::linear_static_from_graph(g, out) {
                    return Ok(m);
                }
                Self::linear_layer_from_graph(g, out)
            }
            Op::Activation(Activation::Relu) => Self::linear_relu_layer_from_graph(g, out),
            Op::Softmax { axis } => Self::matmul_softmax_layer_from_graph(g, out, *axis),
            other => Err(format!(
                "rlx-qnn codegen lowers MatMul, Linear, LinearRelu, MatMulSoftmax, \
                 LinearStatic, or Mlp2; output op is {other:?}"
            )),
        }
    }

    fn matmul_layer_from_node(g: &Graph, out: &rlx_ir::Node) -> Result<Self, String> {
        if out.inputs.len() != 2 {
            return Err(format!(
                "MatMul expects 2 inputs, found {}",
                out.inputs.len()
            ));
        }
        let (m, k, n) = matmul_dims(g, out.inputs[0], out.inputs[1])?;
        Ok(Self {
            name: g.name.clone(),
            layers: vec![Layer::MatMul {
                name: "matmul_0".to_string(),
                m,
                k,
                n,
            }],
        })
    }

    fn linear_layer_from_graph(g: &Graph, add: &rlx_ir::Node) -> Result<Self, String> {
        let (m, k, n) = matmul_add_dims(g, add, "Linear")?;
        Ok(Self {
            name: g.name.clone(),
            layers: vec![Layer::Linear {
                name: "linear_0".to_string(),
                m,
                k,
                n,
            }],
        })
    }

    fn linear_relu_layer_from_graph(g: &Graph, relu: &rlx_ir::Node) -> Result<Self, String> {
        if relu.inputs.len() != 1 {
            return Err(format!("Relu expects 1 input, found {}", relu.inputs.len()));
        }
        let add = g.node(relu.inputs[0]);
        if !matches!(add.op, Op::Binary(BinaryOp::Add)) {
            return Err(format!(
                "LinearRelu expects Relu(Add(...)); add op is {:?}",
                add.op
            ));
        }
        let (m, k, n) = matmul_add_dims(g, add, "LinearRelu")?;
        Ok(Self {
            name: g.name.clone(),
            layers: vec![Layer::LinearRelu {
                name: "linear_relu_0".to_string(),
                m,
                k,
                n,
            }],
        })
    }

    fn matmul_softmax_layer_from_graph(
        g: &Graph,
        sm: &rlx_ir::Node,
        axis: i32,
    ) -> Result<Self, String> {
        if sm.inputs.len() != 1 {
            return Err(format!(
                "Softmax expects 1 input, found {}",
                sm.inputs.len()
            ));
        }
        // axis=1 or -1 both mean last dim on rank-2.
        if axis != 1 && axis != -1 {
            return Err(format!(
                "MatMulSoftmax expects Softmax axis 1 or -1; got {axis}"
            ));
        }
        let mm = g.node(sm.inputs[0]);
        if !matches!(mm.op, Op::MatMul) {
            return Err(format!(
                "MatMulSoftmax expects Softmax(MatMul(...)); matmul op is {:?}",
                mm.op
            ));
        }
        if mm.inputs.len() != 2 {
            return Err(format!(
                "MatMulSoftmax MatMul expects 2 inputs, found {}",
                mm.inputs.len()
            ));
        }
        let (m, k, n) = matmul_dims(g, mm.inputs[0], mm.inputs[1])?;
        Ok(Self {
            name: g.name.clone(),
            layers: vec![Layer::MatMulSoftmax {
                name: "matmul_softmax_0".to_string(),
                m,
                k,
                n,
            }],
        })
    }

    /// The single matmul layer (milestone-1 convenience).
    pub fn matmul(&self) -> Result<(usize, usize, usize), String> {
        match self.layers.as_slice() {
            [Layer::MatMul { m, k, n, .. }] => Ok((*m, *k, *n)),
            _ => Err("model is not a single MatMul".to_string()),
        }
    }

    /// The single linear layer (milestone-1 convenience).
    pub fn linear_dims(&self) -> Result<(usize, usize, usize), String> {
        match self.layers.as_slice() {
            [Layer::Linear { m, k, n, .. }] => Ok((*m, *k, *n)),
            _ => Err("model is not a single Linear".to_string()),
        }
    }

    /// The single linear+relu layer (milestone-1 convenience).
    pub fn linear_relu_dims(&self) -> Result<(usize, usize, usize), String> {
        match self.layers.as_slice() {
            [Layer::LinearRelu { m, k, n, .. }] => Ok((*m, *k, *n)),
            _ => Err("model is not a single LinearRelu".to_string()),
        }
    }

    /// The single matmul+softmax layer (milestone-1 convenience).
    pub fn matmul_softmax_dims(&self) -> Result<(usize, usize, usize), String> {
        match self.layers.as_slice() {
            [Layer::MatMulSoftmax { m, k, n, .. }] => Ok((*m, *k, *n)),
            _ => Err("model is not a single MatMulSoftmax".to_string()),
        }
    }

    /// Two-layer MLP dims `(M, K, H, N)`.
    pub fn mlp2_dims(&self) -> Result<(usize, usize, usize, usize), String> {
        match self.layers.as_slice() {
            [
                Layer::LinearRelu { m, k, n: h, .. },
                Layer::Linear {
                    m: m2, k: h2, n, ..
                },
            ] if m == m2 && h == h2 => Ok((*m, *k, *h, *n)),
            _ => Err("model is not an Mlp2 (LinearRelu → Linear)".to_string()),
        }
    }

    /// Static-weight linear dims `(M, K, N)`.
    pub fn linear_static_dims(&self) -> Result<(usize, usize, usize), String> {
        match self.layers.as_slice() {
            [Layer::LinearStatic { m, k, n, .. }] => Ok((*m, *k, *n)),
            _ => Err("model is not a LinearStatic".to_string()),
        }
    }

    /// `out = Add(MatMul(x, W_const), b_const)` → LinearStatic with baked f32.
    fn linear_static_from_graph(g: &Graph, add: &rlx_ir::Node) -> Result<Self, String> {
        let (m, k, n) = matmul_add_dims(g, add, "LinearStatic")?;
        let mm = g.node(add.inputs[0]);
        if mm.inputs.len() != 2 {
            return Err("LinearStatic MatMul expects 2 inputs".into());
        }
        // Activation must be an Input (runtime APP_WRITE); W and b Constants.
        let act = g.node(mm.inputs[0]);
        if !matches!(act.op, Op::Input { .. }) {
            return Err(format!(
                "LinearStatic expects Input activation; got {:?}",
                act.op
            ));
        }
        let w = f32_constant_payload(g, mm.inputs[1], k * n, "weight")?;
        let b = f32_constant_payload(g, add.inputs[1], m * n, "bias")?;
        Ok(Self::linear_static_with_weights(
            g.name.clone(),
            m,
            k,
            n,
            w,
            b,
        ))
    }

    /// `out = Add(MatMul(Relu(Add(MatMul(x,w1),b1)), w2), b2)`.
    fn mlp2_from_graph(g: &Graph, add2: &rlx_ir::Node) -> Result<Self, String> {
        if add2.inputs.len() != 2 {
            return Err(format!(
                "Mlp2 outer Add expects 2 inputs, found {}",
                add2.inputs.len()
            ));
        }
        let mm2 = g.node(add2.inputs[0]);
        if !matches!(mm2.op, Op::MatMul) {
            return Err(format!(
                "Mlp2 expects Add(MatMul(...), ...); got {:?}",
                mm2.op
            ));
        }
        if mm2.inputs.len() != 2 {
            return Err(format!(
                "Mlp2 second MatMul expects 2 inputs, found {}",
                mm2.inputs.len()
            ));
        }
        let relu = g.node(mm2.inputs[0]);
        if !matches!(relu.op, Op::Activation(Activation::Relu)) {
            return Err(format!(
                "Mlp2 expects MatMul(Relu(...), ...); got {:?}",
                relu.op
            ));
        }
        if relu.inputs.len() != 1 {
            return Err(format!(
                "Mlp2 Relu expects 1 input, found {}",
                relu.inputs.len()
            ));
        }
        let add1 = g.node(relu.inputs[0]);
        if !matches!(add1.op, Op::Binary(BinaryOp::Add)) {
            return Err(format!("Mlp2 expects Relu(Add(...)); got {:?}", add1.op));
        }
        let (m, k, h) = matmul_add_dims(g, add1, "Mlp2 hidden")?;
        let (m2, h2, n) = matmul_dims(g, mm2.inputs[0], mm2.inputs[1])?;
        if m2 != m || h2 != h {
            return Err(format!(
                "Mlp2 layer dims disagree: hidden out [{m},{h}] vs second MatMul [{m2},{h2}]→N"
            ));
        }
        let bias2 = g.shape(add2.inputs[1]);
        if bias2.rank() != 2 {
            return Err(format!(
                "Mlp2 outer bias must be rank-2; got rank {}",
                bias2.rank()
            ));
        }
        let bm = bias2.dim(0).unwrap_static();
        let bn = bias2.dim(1).unwrap_static();
        if bm != m || bn != n {
            return Err(format!(
                "Mlp2 outer bias shape [{bm},{bn}] does not match MatMul out [{m},{n}]"
            ));
        }
        Ok(Self::mlp2(g.name.clone(), m, k, h, n))
    }
}

/// Deterministic f32 weights for STATIC packing (CLI / tests).
/// LCG seed 0xC0FFEE — bit-stable across hosts.
pub(crate) fn seed0_weight_bias(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0xC0FFEE_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((state >> 33) as u32) as f32 / (1u32 << 31) as f32;
        u * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..k * n).map(|_| next()).collect();
    let b: Vec<f32> = (0..m * n).map(|_| next()).collect();
    (w, b)
}

fn f32_constant_payload(
    g: &Graph,
    id: rlx_ir::NodeId,
    expect_elems: usize,
    label: &str,
) -> Result<Vec<f32>, String> {
    let node = g.node(id);
    if node.shape.dtype() != rlx_ir::DType::F32 {
        return Err(format!(
            "LinearStatic {label}: expected F32 Constant, got {:?}",
            node.shape.dtype()
        ));
    }
    let Op::Constant { data } = &node.op else {
        return Err(format!(
            "LinearStatic {label}: expected Constant, got {:?}",
            node.op
        ));
    };
    if data.len() != expect_elems * 4 {
        return Err(format!(
            "LinearStatic {label}: byte len {} != {} ({} f32 elems)",
            data.len(),
            expect_elems * 4,
            expect_elems
        ));
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn matmul_add_dims(
    g: &Graph,
    add: &rlx_ir::Node,
    label: &str,
) -> Result<(usize, usize, usize), String> {
    if add.inputs.len() != 2 {
        return Err(format!(
            "{label} Add expects 2 inputs, found {}",
            add.inputs.len()
        ));
    }
    let mm = g.node(add.inputs[0]);
    if !matches!(mm.op, Op::MatMul) {
        return Err(format!(
            "{label} expects Add(MatMul(...), ...); matmul op is {:?}",
            mm.op
        ));
    }
    if mm.inputs.len() != 2 {
        return Err(format!(
            "{label} MatMul expects 2 inputs, found {}",
            mm.inputs.len()
        ));
    }
    let (m, k, n) = matmul_dims(g, mm.inputs[0], mm.inputs[1])?;
    let bias = g.shape(add.inputs[1]);
    if bias.rank() != 2 {
        return Err(format!(
            "{label} bias must be rank-2; got rank {}",
            bias.rank()
        ));
    }
    let bm = bias.dim(0).unwrap_static();
    let bn = bias.dim(1).unwrap_static();
    if bm != m || bn != n {
        return Err(format!(
            "{label} bias shape [{bm},{bn}] does not match MatMul out [{m},{n}]"
        ));
    }
    Ok((m, k, n))
}

fn matmul_dims(
    g: &Graph,
    lhs_id: rlx_ir::NodeId,
    rhs_id: rlx_ir::NodeId,
) -> Result<(usize, usize, usize), String> {
    let lhs = g.shape(lhs_id);
    let rhs = g.shape(rhs_id);
    if lhs.rank() != 2 || rhs.rank() != 2 {
        return Err(format!(
            "milestone 1 supports rank-2 MatMul only; got lhs rank {}, rhs rank {} \
             (batched MatMul lands with the multi-op milestone)",
            lhs.rank(),
            rhs.rank()
        ));
    }
    let m = lhs.dim(0).unwrap_static();
    let k = lhs.dim(1).unwrap_static();
    let k2 = rhs.dim(0).unwrap_static();
    let n = rhs.dim(1).unwrap_static();
    if k != k2 {
        return Err(format!(
            "MatMul contracting dims disagree: lhs K={k}, rhs K={k2}"
        ));
    }
    Ok((m, k, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    #[test]
    fn from_graph_reads_matmul_dims() {
        let mut g = Graph::new("mm");
        let x = g.input("x", Shape::new(&[3, 5], DType::F32));
        let w = g.param("w", Shape::new(&[5, 7], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[3, 7], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.matmul().unwrap(), (3, 5, 7));
    }

    #[test]
    fn from_graph_reads_linear_dims() {
        use rlx_ir::op::BinaryOp;

        let mut g = Graph::new("linear");
        let x = g.input("x", Shape::new(&[3, 5], DType::F32));
        let w = g.input("w", Shape::new(&[5, 7], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[3, 7], DType::F32));
        let b = g.input("b", Shape::new(&[3, 7], DType::F32));
        let y = g.binary(BinaryOp::Add, mm, b, Shape::new(&[3, 7], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.linear_dims().unwrap(), (3, 5, 7));
    }

    #[test]
    fn from_graph_reads_linear_static_constants() {
        use rlx_ir::op::BinaryOp;

        let (m, k, n) = (2usize, 3, 2);
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let b: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0]; // 2x2
        let mut g = Graph::new("linstatic");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let wi = {
            let mut bytes = Vec::with_capacity(w.len() * 4);
            for v in &w {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            g.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&[k, n], DType::F32),
            )
        };
        let mm = g.matmul(x, wi, Shape::new(&[m, n], DType::F32));
        let bi = {
            let mut bytes = Vec::with_capacity(b.len() * 4);
            for v in &b {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            g.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&[m, n], DType::F32),
            )
        };
        let y = g.binary(BinaryOp::Add, mm, bi, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.linear_static_dims().unwrap(), (m, k, n));
        match &model.layers[0] {
            Layer::LinearStatic { w: ww, b: bb, .. } => {
                assert_eq!(ww, &w);
                assert_eq!(bb, &b);
            }
            other => panic!("expected LinearStatic, got {other:?}"),
        }
    }

    #[test]
    fn from_graph_reads_linear_relu_dims() {
        use rlx_ir::op::{Activation, BinaryOp};

        let mut g = Graph::new("linrelu");
        let x = g.input("x", Shape::new(&[3, 5], DType::F32));
        let w = g.input("w", Shape::new(&[5, 7], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[3, 7], DType::F32));
        let b = g.input("b", Shape::new(&[3, 7], DType::F32));
        let xw_b = g.binary(BinaryOp::Add, mm, b, Shape::new(&[3, 7], DType::F32));
        let y = g.activation(Activation::Relu, xw_b, Shape::new(&[3, 7], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.linear_relu_dims().unwrap(), (3, 5, 7));
    }

    #[test]
    fn from_graph_reads_matmul_softmax_dims() {
        let mut g = Graph::new("mmsm");
        let x = g.input("x", Shape::new(&[3, 5], DType::F32));
        let w = g.input("w", Shape::new(&[5, 7], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[3, 7], DType::F32));
        let y = g.softmax(mm, 1, Shape::new(&[3, 7], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.matmul_softmax_dims().unwrap(), (3, 5, 7));
    }

    #[test]
    fn from_graph_reads_mlp2_dims() {
        use rlx_ir::op::{Activation, BinaryOp};

        let (m, k, h, n) = (3usize, 5, 8, 4);
        let mut g = Graph::new("mlp2");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w1 = g.input("w1", Shape::new(&[k, h], DType::F32));
        let mm1 = g.matmul(x, w1, Shape::new(&[m, h], DType::F32));
        let b1 = g.input("b1", Shape::new(&[m, h], DType::F32));
        let a1 = g.binary(BinaryOp::Add, mm1, b1, Shape::new(&[m, h], DType::F32));
        let r = g.activation(Activation::Relu, a1, Shape::new(&[m, h], DType::F32));
        let w2 = g.input("w2", Shape::new(&[h, n], DType::F32));
        let mm2 = g.matmul(r, w2, Shape::new(&[m, n], DType::F32));
        let b2 = g.input("b2", Shape::new(&[m, n], DType::F32));
        let y = g.binary(BinaryOp::Add, mm2, b2, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);

        let model = Model::from_graph(&g).expect("recognized");
        assert_eq!(model.mlp2_dims().unwrap(), (m, k, h, n));
    }

    #[test]
    fn from_graph_rejects_non_matmul_output() {
        let mut g = Graph::new("add");
        let a = g.input("a", Shape::new(&[4], DType::F32));
        let b = g.input("b", Shape::new(&[4], DType::F32));
        let c = g.binary(
            rlx_ir::op::BinaryOp::Add,
            a,
            b,
            Shape::new(&[4], DType::F32),
        );
        g.set_outputs(vec![c]);
        assert!(Model::from_graph(&g).is_err());
    }
}
