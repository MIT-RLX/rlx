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

    /// 1D FFT along the last axis of the 2N real-block complex layout.
    /// Last axis size must be even (so the 2N real-block layout
    /// resolves to an integer number of complex points). Output shape
    /// == input shape.
    ///
    /// The CPU kernel uses radix-2 Cooley-Tukey when the complex
    /// length `N = last/2` is a power of two, and Bluestein's
    /// algorithm (chirp z-transform) otherwise. There is no
    /// size restriction beyond `last` being even.
    ///
    /// See `Op::Fft` for the normalization convention
    /// (unnormalized; ifft(fft(x)) = N·x).
    pub fn fft(&mut self, x: NodeId, inverse: bool) -> NodeId {
        let s = self.shape(x).clone();
        assert!(s.rank() >= 1, "fft: tensor must have at least 1 axis");
        let last = s.rank() - 1;
        match s.dim(last) {
            crate::shape::Dim::Static(n) => {
                assert!(
                    n % 2 == 0,
                    "fft: last axis size {n} must be even (2N real-block layout)"
                );
            }
            _ => panic!("fft: dynamic last-axis size not supported"),
        }
        self.push(Op::Fft { inverse }, vec![x], s, None)
    }

    /// 1D FFT along an arbitrary axis (not just the last). Lowers to
    /// `Transpose(axis ↔ last) → Fft(last) → Transpose(last ↔ axis)`
    /// — the 2N-real-block convention is intrinsic to whichever axis
    /// the FFT runs along, and `Op::Transpose` is a pure permutation,
    /// so semantics transport correctly.
    ///
    /// Limitation: this still describes a tensor with a *single*
    /// complex axis. True ND `fftn` (e.g. 2D FFT of a 2D-complex
    /// array, where two axes are independently complex) cannot be
    /// expressed in the 2N-real-block layout — it needs native
    /// `DType::C64` to keep the real/imag split off the axis grid.
    /// See PLAN.md for the deferred C64 workstream.
    ///
    /// AD is free: Transpose and Fft both have VJP and JVP rules,
    /// so the composition differentiates automatically.
    pub fn fft_axis(&mut self, x: NodeId, axis: usize, inverse: bool) -> NodeId {
        use crate::infer::GraphExt as _;
        let rank = self.shape(x).rank();
        assert!(
            axis < rank,
            "fft_axis: axis {axis} out of range for rank-{rank} tensor"
        );
        let last = rank - 1;
        if axis == last {
            // Fast path — no transpose needed.
            return self.fft(x, inverse);
        }
        // perm = identity with `axis` ↔ `last` swapped. Same perm in
        // both directions because it's a transposition (its own inverse).
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis, last);

        let x_t = self.transpose_(x, perm.clone());
        let y_t = self.fft(x_t, inverse);
        self.transpose_(y_t, perm)
    }
}
