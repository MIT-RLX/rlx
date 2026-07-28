// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lightweight description of the subgraph we know how to lower to CSL.
//!
//! Milestone 1 supports exactly one shape: a single rank-2 `MatMul`
//! `Y[M,N] = X[M,K] · W[K,N]`. [`Model::from_graph`] recognizes that pattern
//! in an `rlx-ir` [`Graph`] and reads `M`/`K`/`N` from the operand shapes;
//! [`Model::single_matmul`] builds one directly for tests / the CLI.

use rlx_ir::{Graph, Op};

/// A single operation we can emit as a CSL kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layer {
    /// `Y[m,n] = X[m,k] · W[k,n]` (row-major), f32.
    MatMul {
        name: String,
        m: usize,
        k: usize,
        n: usize,
    },
}

/// A program to emit: a name and the ordered layers.
///
/// Milestone 1 holds exactly one [`Layer`]; the field is a `Vec` so the
/// multi-op / fused milestones drop in without an API break.
#[derive(Debug, Clone, PartialEq, Eq)]
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
                name: "matmul".to_string(),
                m,
                k,
                n,
            }],
        }
    }

    /// Recognize the supported subgraph in an `rlx-ir` graph.
    ///
    /// Milestone-1 scope: the graph must have a single output that is a
    /// rank-2 [`Op::MatMul`] with statically-known dims. Anything else is a
    /// clear `Err` rather than a silent partial lowering.
    pub fn from_graph(g: &Graph) -> Result<Self, String> {
        // `Op::Scan` (recurrence, e.g. IIR biquad) has no CSL form — unroll it
        // into inlined body replicas (primitives the codegen handles) first.
        let scan_unrolled;
        let g = if g.nodes().iter().any(|n| matches!(n.op, Op::Scan { .. })) {
            use rlx_opt::pass::Pass as _;
            scan_unrolled = rlx_opt::LowerScan.run(g.clone());
            &scan_unrolled
        } else {
            g
        };
        // Distributed collectives are host/transport ops (they need an OS network
        // stack + threads to talk to other ranks). They cannot be expressed inside
        // a single-device CSL fabric program — reject with a specific, actionable
        // error rather than the generic "lowers only Op::MatMul" path below.
        if let Some(name) = g.nodes().iter().find_map(|n| match &n.op {
            Op::Custom { name, .. } if name.starts_with("collective.") => Some(name.clone()),
            _ => None,
        }) {
            return Err(format!(
                "rlx-cerebras: '{name}' is a distributed host/transport collective \
                 — it cannot be expressed in a single-device CSL fabric program. \
                 Run the distributed graph on a host-capable backend (CPU/CUDA/Metal/…)."
            ));
        }
        if g.outputs.len() != 1 {
            return Err(format!(
                "rlx-cerebras milestone 1 lowers a single-output graph; got {} outputs",
                g.outputs.len()
            ));
        }
        let out = g.node(g.outputs[0]);
        if !matches!(out.op, Op::MatMul) {
            return Err(format!(
                "rlx-cerebras milestone 1 lowers only Op::MatMul; output op is {:?}",
                out.op
            ));
        }
        if out.inputs.len() != 2 {
            return Err(format!(
                "MatMul expects 2 inputs, found {}",
                out.inputs.len()
            ));
        }
        let lhs = g.shape(out.inputs[0]);
        let rhs = g.shape(out.inputs[1]);
        if lhs.rank() != 2 || rhs.rank() != 2 {
            return Err(format!(
                "milestone 1 supports rank-2 MatMul only; got lhs rank {}, rhs rank {} \
                 (batched MatMul lands with multi-PE tiling)",
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
        Ok(Self {
            name: g.name.clone(),
            layers: vec![Layer::MatMul {
                name: "matmul".to_string(),
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

    #[test]
    fn from_graph_rejects_collective_with_specific_message() {
        // A distributed collective can't be lowered to a single-device CSL
        // fabric program; the error must name the op and say so specifically,
        // not fall through to the generic "lowers only Op::MatMul" path.
        let mut g = Graph::new("ar");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let ar = g.add_node(
            rlx_ir::Op::Custom {
                name: "collective.all_reduce".to_string(),
                num_inputs: 1,
                attrs: vec![],
            },
            vec![x],
            Shape::new(&[4], DType::F32),
        );
        g.set_outputs(vec![ar]);

        let err = Model::from_graph(&g).expect_err("collective must be rejected");
        assert!(
            err.contains("collective.all_reduce") && err.contains("host/transport collective"),
            "unexpected error: {err}"
        );
    }
}
