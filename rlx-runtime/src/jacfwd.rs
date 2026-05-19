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

//! Forward-mode Jacobian materialization.
//!
//! Pre-vmap convenience: given a compiled JVP graph (output of
//! `rlx_opt::autodiff_fwd::jvp`), assemble the Jacobian by running
//! the graph once per standard-basis unit vector and stacking the
//! tangent outputs.
//!
//! Use this when the *input* dimension is small (Circulax component
//! groups have handfuls of params) — once a `vmap` transformation
//! lands the same shape gets vectorised into a single call.
//!
//! ## Layout convention
//!
//! For each primal output of shape `output_shape`, the Jacobian is
//! returned as a flat byte buffer encoding a row-major
//! `[output_size, wrt_size]` matrix where:
//!
//! * `output_size = product(output_shape)`
//! * `wrt_size    = product(wrt_shape)`
//!
//! Element `[i, j]` is `∂output[i] / ∂wrt[j]` after both shapes are
//! flattened. Callers reshape to the natural
//! `[output_shape..., wrt_shape...]` if convenient.
//!
//! ## Compose with [`crate::Session`]
//!
//! ```ignore
//! use rlx_opt::autodiff_fwd::jvp;
//! use rlx_runtime::{Session, Device, jacfwd};
//!
//! let jvp_graph = jvp(&forward, &[wrt_node]);
//! let mut compiled = Session::new(Device::Cpu).compile(jvp_graph);
//! let jacs = jacfwd(&mut compiled, &primals, "x", &[3], DType::F64);
//! ```

use crate::compiled::CompiledGraph;
use rlx_ir::DType;

/// One Jacobian per primal output of the original forward graph.
#[derive(Debug, Clone)]
pub struct JacobianBytes {
    /// Flat row-major `[output_size, wrt_size]` matrix.
    pub bytes: Vec<u8>,
    /// Number of elements in the primal output (= rows).
    pub output_size: usize,
    /// Number of elements in the wrt input (= columns).
    pub wrt_size: usize,
    /// Element dtype of `bytes`.
    pub dtype: DType,
}

impl JacobianBytes {
    /// Reinterpret the byte buffer as `&[f64]` (row-major).
    /// Panics if `dtype != F64` or the byte length isn't a multiple of 8.
    pub fn as_f64(&self) -> &[f64] {
        assert_eq!(
            self.dtype,
            DType::F64,
            "as_f64: dtype is {:?}, not F64",
            self.dtype
        );
        assert_eq!(
            self.bytes.len(),
            self.output_size * self.wrt_size * 8,
            "as_f64: byte length doesn't match shape"
        );
        // SAFETY: bytes are 8-aligned (rlx-runtime allocates with at
        // least 8-byte alignment) and the byte length is a multiple of 8.
        unsafe {
            std::slice::from_raw_parts(self.bytes.as_ptr() as *const f64, self.bytes.len() / 8)
        }
    }

    /// Reinterpret as `&[f32]` (row-major). Mirror of `as_f64`.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(
            self.dtype,
            DType::F32,
            "as_f32: dtype is {:?}, not F32",
            self.dtype
        );
        assert_eq!(self.bytes.len(), self.output_size * self.wrt_size * 4);
        unsafe {
            std::slice::from_raw_parts(self.bytes.as_ptr() as *const f32, self.bytes.len() / 4)
        }
    }
}

/// Materialize the Jacobian of every primal output w.r.t. the input
/// named `wrt_name`. The compiled graph must be the result of
/// `rlx_opt::autodiff_fwd::jvp(forward, &[wrt_node])` — it has a
/// `tangent_<wrt_name>` Input that we drive with unit vectors, and
/// outputs `[primals..., tangents...]`.
///
/// `primals` carries values for every non-tangent input (including
/// any `Param` that was bound externally; if the model uses
/// `set_param_typed` for params, call it before invoking `jacfwd` —
/// params persist across the multiple runs).
///
/// One JVP run per element of `wrt_shape`; cost scales linearly with
/// the wrt dimension. Use reverse-mode (`grad_with_loss`) when the
/// output dimension is what's small instead.
pub fn jacfwd(
    compiled: &mut CompiledGraph,
    primals: &[(&str, &[u8], DType)],
    wrt_name: &str,
    wrt_shape: &[usize],
    dtype: DType,
) -> Vec<JacobianBytes> {
    let elem_size = dtype.size_bytes();
    let wrt_size: usize = wrt_shape.iter().product();
    if wrt_size == 0 {
        return Vec::new();
    }

    let tangent_name = format!("tangent_{wrt_name}");
    let mut tangent_buf = vec![0u8; wrt_size * elem_size];

    // First run sets the tangent to e_0 and gives us the output
    // shapes / sizes. After that we know how to size the Jacobian
    // buffers and can fill them column by column.
    set_unit(&mut tangent_buf, 0, dtype);
    let first = run_one(compiled, primals, &tangent_name, &tangent_buf, dtype);
    // Outputs are [primals_0..k-1, tangents_0..k-1].
    assert!(
        first.len().is_multiple_of(2),
        "jacfwd: JVP graph must have even output count [primals..., tangents...], got {}",
        first.len()
    );
    let n_outs = first.len() / 2;

    // Allocate Jacobian buffers — `output_size` discovered per-output.
    let mut jacs: Vec<JacobianBytes> = (0..n_outs)
        .map(|i| {
            let (bytes, dt) = &first[n_outs + i];
            debug_assert_eq!(
                *dt, dtype,
                "jacfwd: tangent output {} has dtype {:?}, expected {:?}",
                i, dt, dtype
            );
            let output_size = bytes.len() / elem_size;
            JacobianBytes {
                bytes: vec![0u8; output_size * wrt_size * elem_size],
                output_size,
                wrt_size,
                dtype,
            }
        })
        .collect();

    // Write column 0 from the first run, then loop for columns 1..wrt_size.
    write_column(&first[n_outs..], &mut jacs, 0, elem_size);

    for j in 1..wrt_size {
        // Reset previous slot, set new unit. (set_unit writes a 1 at
        // index j; we still need to clear j-1.)
        clear_index(&mut tangent_buf, j - 1, dtype);
        set_unit(&mut tangent_buf, j, dtype);

        let outs = run_one(compiled, primals, &tangent_name, &tangent_buf, dtype);
        write_column(&outs[n_outs..], &mut jacs, j, elem_size);
    }

    jacs
}

/// Single-shot run with a freshly-set tangent slot.
fn run_one(
    compiled: &mut CompiledGraph,
    primals: &[(&str, &[u8], DType)],
    tangent_name: &str,
    tangent_bytes: &[u8],
    dtype: DType,
) -> Vec<(Vec<u8>, DType)> {
    let mut all = primals.to_vec();
    all.push((tangent_name, tangent_bytes, dtype));
    compiled.run_typed(&all)
}

/// Copy each tangent output into column `j` of its Jacobian.
fn write_column(
    tangent_outputs: &[(Vec<u8>, DType)],
    jacs: &mut [JacobianBytes],
    j: usize,
    elem_size: usize,
) {
    debug_assert_eq!(tangent_outputs.len(), jacs.len());
    for (out_idx, (bytes, _)) in tangent_outputs.iter().enumerate() {
        let jac = &mut jacs[out_idx];
        debug_assert_eq!(
            bytes.len(),
            jac.output_size * elem_size,
            "tangent output size changed mid-jacfwd run"
        );
        // Row-major [output_size, wrt_size] → column j is element
        // i*wrt_size + j for each i. Single byte-stripe write per row.
        for i in 0..jac.output_size {
            let dst_off = (i * jac.wrt_size + j) * elem_size;
            let src_off = i * elem_size;
            jac.bytes[dst_off..dst_off + elem_size]
                .copy_from_slice(&bytes[src_off..src_off + elem_size]);
        }
    }
}

fn set_unit(buf: &mut [u8], idx: usize, dtype: DType) {
    match dtype {
        DType::F64 => {
            let off = idx * 8;
            buf[off..off + 8].copy_from_slice(&1.0_f64.to_le_bytes());
        }
        DType::F32 => {
            let off = idx * 4;
            buf[off..off + 4].copy_from_slice(&1.0_f32.to_le_bytes());
        }
        other => panic!("jacfwd: dtype {other:?} not supported (f64 / f32 only today)"),
    }
}

fn clear_index(buf: &mut [u8], idx: usize, dtype: DType) {
    let n = dtype.size_bytes();
    let off = idx * n;
    for b in &mut buf[off..off + n] {
        *b = 0;
    }
}

#[cfg(test)]
#[cfg(feature = "cpu")]
mod tests {

    use rlx_ir::{Graph, Shape};
    use rlx_opt::autodiff_fwd::jvp;

    fn f64_bytes(xs: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(xs.len() * 8);
        for x in xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// `f(b) = 3·b` ⇒ `df/db = diag(3)` — smallest possible jacfwd
    /// shape check that doesn't depend on any other AD machinery.
    /// Builds a graph that scales `b` by a constant via `Mul`, runs
    /// `jvp`, then `jacfwd`, and asserts the result is a diagonal of 3s.
    #[test]
    fn jacfwd_scalar_mul_gives_diagonal() {
        use rlx_ir::DType;
        use rlx_ir::op::BinaryOp;
        let n = 4usize;

        let mut g = Graph::new("scale");
        let b = g.input("b", Shape::new(&[n], DType::F64));
        // Scale constant: a 1-D tensor of 3s.
        let three_bytes = f64_bytes(&vec![3.0; n]);
        let three = g.add_node(
            rlx_ir::Op::Constant { data: three_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let y = g.binary(BinaryOp::Mul, b, three, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![y]);

        let jg = jvp(&g, &[b]);
        let mut compiled = crate::Session::new(crate::Device::Cpu).compile(jg);

        let b_data = vec![10.0_f64; n];
        let jacs = super::jacfwd(
            &mut compiled,
            &[("b", &f64_bytes(&b_data), DType::F64)],
            "b",
            &[n],
            DType::F64,
        );
        assert_eq!(jacs.len(), 1);
        let jac = &jacs[0];
        assert_eq!(jac.output_size, n);
        assert_eq!(jac.wrt_size, n);
        let m = jac.as_f64();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 3.0 } else { 0.0 };
                assert!(
                    (m[i * n + j] - want).abs() < 1e-12,
                    "jac[{i},{j}] = {} (expected {want})",
                    m[i * n + j]
                );
            }
        }
    }
}
