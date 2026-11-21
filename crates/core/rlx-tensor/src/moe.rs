// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Routing and packed-weight ops for mixture-of-experts language models:
//! top-k selection, `take_along_axis`, per-row expert matmuls, and matmuls
//! over GGUF-packed and MLX-format (quantized) weight blobs.
//!
//! Index tensors are f32-encoded, as everywhere in `rlx-ir`. A packed weight
//! is a `U8` tensor (declare it with [`crate::GraphScope::param`] and a `U8`
//! shape) holding the quant blob exactly as stored in the file; its layout is
//! the scheme's (see [`QuantScheme`]).

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, NodeId, Op, Shape};

use crate::tensor::Tensor;

impl Tensor {
    fn adopt_all(&self, others: &[&Tensor]) -> Vec<NodeId> {
        std::iter::once(self.id)
            .chain(others.iter().map(|t| self.adopt(t)))
            .collect()
    }

    fn push(&self, op: Op, inputs: Vec<NodeId>, shape: Shape) -> Self {
        let id = self.handle.with_graph(|g| g.add_node(op, inputs, shape));
        Self::new(self.handle.clone(), id)
    }

    fn static_dims(&self) -> Vec<usize> {
        self.shape()
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }

    /// Indices (f32-encoded) of the `k` largest values along the last axis,
    /// largest first; ties go to the smaller index.
    pub fn top_k(&self, k: usize) -> Self {
        let mut dims = self.static_dims();
        *dims.last_mut().expect("top_k of a scalar") = k;
        self.push(Op::TopK { k }, vec![self.id], Shape::new(&dims, DType::F32))
    }

    /// `take_along_axis`: the element of `self` at `indices` along `axis`,
    /// shaped like `indices` (ONNX `GatherElements`).
    pub fn gather_elements(&self, indices: &Tensor, axis: usize) -> Self {
        let inputs = self.adopt_all(&[indices]);
        self.push(
            Op::GatherElements { axis: axis as i32 },
            inputs,
            Shape::new(&indices.static_dims(), self.dtype()),
        )
    }

    /// `x · W[e]` per row: `self` `[M, K]`, `bank` `[E, K, N]`, `expert_idx`
    /// `[M]` → `[M, N]`.
    pub fn grouped_matmul(&self, bank: &Tensor, expert_idx: &Tensor) -> Self {
        let m = self.static_dims()[0];
        let n = *bank.static_dims().last().expect("grouped_matmul bank rank");
        let inputs = self.adopt_all(&[bank, expert_idx]);
        self.push(Op::GroupedMatMul, inputs, Shape::new(&[m, n], DType::F32))
    }

    /// `x · Wᵀ` over a packed `[out_dim, in_dim]` GGUF blob (`U8`), contracting
    /// the last axis of `self`.
    pub fn dequant_matmul(&self, packed: &Tensor, scheme: QuantScheme, out_dim: usize) -> Self {
        let mut dims = self.static_dims();
        *dims.last_mut().expect("dequant_matmul of a scalar") = out_dim;
        let inputs = self.adopt_all(&[packed]);
        self.push(
            Op::DequantMatMul { scheme },
            inputs,
            Shape::new(&dims, DType::F32),
        )
    }

    /// Per-row expert matmul over a packed bank: `num_experts` contiguous
    /// `[out_dim, in_dim]` GGUF slabs (GGUF's native `ffn_*_exps` order).
    /// `self` `[M, in_dim]`, `expert_idx` `[M]` → `[M, out_dim]`.
    pub fn dequant_grouped_matmul(
        &self,
        bank: &Tensor,
        expert_idx: &Tensor,
        scheme: QuantScheme,
        out_dim: usize,
    ) -> Self {
        let m = self.static_dims()[0];
        let inputs = self.adopt_all(&[bank, expert_idx]);
        self.push(
            Op::DequantGroupedMatMul { scheme },
            inputs,
            Shape::new(&[m, out_dim], DType::F32),
        )
    }
}

impl Tensor {
    /// Per-row expert matmul over MLX-format experts: `bank` packed codes
    /// (`U8`), `scales` / `biases` `[E, out_dim, in_dim / group]` holding the
    /// decoded values (f32 or BF16; biases zero for FP4 / FP8 codes).
    /// `self` `[M, in_dim]`, `expert_idx` `[M]` → `[M, out_dim]`.
    pub fn dequant_grouped_matmul_mlx(
        &self,
        bank: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        expert_idx: &Tensor,
        scheme: QuantScheme,
        out_dim: usize,
    ) -> Self {
        let m = self.static_dims()[0];
        let inputs = self.adopt_all(&[bank, scales, biases, expert_idx]);
        self.push(
            Op::DequantGroupedMatMulMlx { scheme },
            inputs,
            Shape::new(&[m, out_dim], DType::F32),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::GraphScope;
    use rlx_ir::{DType, Shape};

    #[test]
    fn shapes() {
        let mut s = GraphScope::new("moe");
        let x = s.input("x", Shape::new(&[3, 8], DType::F32));
        let idx = x.top_k(2);
        assert_eq!(idx.shape(), Shape::new(&[3, 2], DType::F32));
        assert_eq!(
            x.gather_elements(&idx, 1).shape(),
            Shape::new(&[3, 2], DType::F32)
        );
        let bank = s.param("bank", Shape::new(&[4, 8, 5], DType::F32));
        let e = s.input("e", Shape::new(&[3], DType::F32));
        assert_eq!(
            x.grouped_matmul(&bank, &e).shape(),
            Shape::new(&[3, 5], DType::F32)
        );
        let q = s.param("q", Shape::new(&[40], DType::U8));
        let y = x.dequant_matmul(&q, rlx_ir::quant::QuantScheme::GgufQ8_0, 6);
        assert_eq!(y.shape(), Shape::new(&[3, 6], DType::F32));
    }
}
