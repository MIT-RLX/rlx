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

//! Graph I/O builders: inputs, parameters (plan #53).

use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Graph input (runtime-provided tensor).
    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> NodeId {
        let name: String = name.into();
        self.push(Op::Input { name: name.clone() }, vec![], shape, Some(name))
    }

    /// Model parameter (weight loaded at init).
    pub fn param(&mut self, name: impl Into<String>, shape: Shape) -> NodeId {
        let name: String = name.into();
        self.push(Op::Param { name: name.clone() }, vec![], shape, Some(name))
    }

    /// Generic node constructor for custom ops.
    pub fn add_node(&mut self, op: Op, inputs: Vec<NodeId>, shape: Shape) -> NodeId {
        self.push(op, inputs, shape, None)
    }

    /// Build an `Op::Custom` node, dispatching shape inference through
    /// the global op registry. The named op must already be registered
    /// via [`crate::register_op`]; `attrs` is forwarded verbatim to
    /// the impl's `infer_shape` (and later, at execution time, to its
    /// per-backend kernel).
    ///
    /// Panics if `name` is not registered or if `inputs.len()` does
    /// not match the registered `num_inputs()` — both are programmer
    /// errors that should fail loudly at graph-build time, not silently
    /// at execution.
    pub fn custom_op(
        &mut self,
        name: impl Into<String>,
        attrs: Vec<u8>,
        inputs: Vec<NodeId>,
    ) -> NodeId {
        let name: String = name.into();
        let ext = crate::lookup_op(&name)
            .unwrap_or_else(|| panic!("custom_op: '{name}' is not registered in the op registry"));
        assert_eq!(
            ext.num_inputs(),
            inputs.len(),
            "custom_op '{name}': registered op expects {} inputs, got {}",
            ext.num_inputs(),
            inputs.len(),
        );
        let in_shapes: Vec<&Shape> = inputs.iter().map(|id| self.shape(*id)).collect();
        let out_shape = ext.infer_shape(&in_shapes, &attrs);
        let num_inputs = ext.num_inputs() as u32;
        self.push(
            Op::Custom {
                name,
                num_inputs,
                attrs,
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Build an `Op::Custom` node with a caller-supplied output shape,
    /// **bypassing** the registry's `infer_shape`. Use this for ops
    /// whose output shape can't be determined by static input shapes
    /// alone — most importantly, ops with multiple logical outputs
    /// packed into one buffer.
    ///
    /// The canonical multi-output pattern:
    ///
    /// ```ignore
    /// // Sparse-LU returns L_values + U_values packed end-to-end.
    /// // Caller knows nnz_L and nnz_U from the symbolic factor.
    /// let lu = g.custom_op_packed(
    ///     "sparse_lu",
    ///     attrs,
    ///     vec![A, b],
    ///     Shape::new(&[nnz_L + nnz_U], DType::F64),
    /// );
    /// let l_vals = g.narrow_(lu, 0, 0, nnz_L);
    /// let u_vals = g.narrow_(lu, 0, nnz_L, nnz_U);
    /// ```
    ///
    /// The op must still be registered (so `num_inputs` validation
    /// and autodiff routing still work); only the shape is overridden.
    pub fn custom_op_packed(
        &mut self,
        name: impl Into<String>,
        attrs: Vec<u8>,
        inputs: Vec<NodeId>,
        out_shape: Shape,
    ) -> NodeId {
        let name: String = name.into();
        let ext = crate::lookup_op(&name).unwrap_or_else(|| {
            panic!("custom_op_packed: '{name}' is not registered in the op registry")
        });
        assert_eq!(
            ext.num_inputs(),
            inputs.len(),
            "custom_op_packed '{name}': registered op expects {} inputs, got {}",
            ext.num_inputs(),
            inputs.len(),
        );
        let num_inputs = ext.num_inputs() as u32;
        self.push(
            Op::Custom {
                name,
                num_inputs,
                attrs,
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// 1D FFT along the last axis.
    ///
    /// * **F32 / F64** — 2N real-block layout: last axis is `[re…, im…]`.
    /// * **C64** — interleaved `[re, im]` pairs per complex element.
    ///
    /// Output shape matches input. Radix-2 when `N` is a power of two,
    /// Bluestein otherwise. Default normalization is unnormalized
    /// (`FftNorm::Backward`; `ifft(fft(x)) = N·x`).
    pub fn fft(&mut self, x: NodeId, inverse: bool) -> NodeId {
        self.fft_norm(x, inverse, crate::fft::FftNorm::Backward)
    }

    /// 1D FFT with explicit normalization mode.
    pub fn fft_norm(&mut self, x: NodeId, inverse: bool, norm: crate::fft::FftNorm) -> NodeId {
        let s = self.shape(x).clone();
        crate::fft::fft_meta(&s);
        self.push(Op::Fft { inverse, norm }, vec![x], s, None)
    }

    /// 1D FFT along an arbitrary axis. Lowers to
    /// `Transpose(axis ↔ last) → Fft(last) → Transpose(last ↔ axis)`.
    ///
    /// AD is free: both `Op::Transpose` and `Op::Fft` have VJP/JVP rules.
    pub fn fft_axis(&mut self, x: NodeId, axis: usize, inverse: bool) -> NodeId {
        use crate::infer::GraphExt as _;
        let rank = self.shape(x).rank();
        assert!(
            axis < rank,
            "fft_axis: axis {axis} out of range for rank-{rank} tensor"
        );
        let last = rank - 1;
        if axis == last {
            return self.fft(x, inverse);
        }
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis, last);

        let x_t = self.transpose_(x, perm.clone());
        let y_t = self.fft(x_t, inverse);
        self.transpose_(y_t, perm)
    }

    /// N-dimensional FFT along `axes` (NumPy `fftn` semantics).
    ///
    /// Applies a 1D FFT along each listed axis in ascending order.
    /// Empty `axes` is a no-op. For multi-axis transforms on tensors
    /// with more than one spatial dimension, use `DType::C64`; the
    /// F32/F64 2N-block layout only describes a single complex axis.
    pub fn fftn(&mut self, x: NodeId, axes: &[usize], inverse: bool) -> NodeId {
        let rank = self.shape(x).rank();
        let axes = crate::fft::normalize_fftn_axes(rank, axes);
        if axes.is_empty() {
            return x;
        }
        if axes.len() > 1 && !self.shape(x).dtype().is_complex() {
            panic!(
                "fftn: multi-axis FFT on {:?} requires DType::C64; \
                 the F32/F64 2N real-block layout supports only one complex axis — \
                 call fft_axis for a single transform",
                self.shape(x).dtype()
            );
        }
        let mut y = x;
        for axis in axes {
            y = self.fft_axis(y, axis, inverse);
        }
        y
    }

    /// Inverse N-dimensional FFT — alias for `fftn(..., inverse: true)`.
    pub fn ifftn(&mut self, x: NodeId, axes: &[usize]) -> NodeId {
        self.fftn(x, axes, true)
    }
}
