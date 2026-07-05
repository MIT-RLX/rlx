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

//! `pyrlx.Graph` — Python builder over `rlx_ir::Graph`.
//!
//! # Surface parity
//!
//! Every public builder on `rlx_ir::Graph` and its [`GraphExt`] companions is
//! reachable from Python with the same name. Where `GraphExt` infers output
//! shape, the Python method omits `out_shape` (e.g. `matmul`, `add`, `conv2d`).
//! Where the IR requires an explicit shape, the Python signature includes it
//! (e.g. `matmul_with_shape`, `lora_matmul`, `sample`).
//!
//! # Naming
//!
//! Reserved Python keywords get a trailing underscore: `where_`, `eq_`, `lt_`,
//! `gt_`, `ge_`, `ne_`.
//!
//! # Literals
//!
//! `constant(value, dtype="f32")` inserts a rank-0 [`Op::Constant`] node that
//! broadcasts in binary ops. `f16` / `bf16` literals are lowered as f32 +
//! `cast`.
//!
//! # Lifecycle
//!
//! `Session.compile` moves the inner `Graph` out of `PyGraph`; subsequent calls
//! raise "graph already consumed".

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op, Shape, fft::FftNorm};

use crate::dtype::parse_dtype;

// ── tiny string parsers ────────────────────────────────────────

/// Parse a scale-layout name for `scaled_matmul` — delegates to
/// `ScaleLayout: FromStr` ("per_tensor" | "mx" | "nvfp4" | "mx/<block>").
fn scale_layout_from_py(layout: &str) -> PyResult<rlx_ir::ScaleLayout> {
    layout
        .parse::<rlx_ir::ScaleLayout>()
        .map_err(PyValueError::new_err)
}

fn shape_from_py(dims: Vec<usize>, dtype: &str) -> PyResult<Shape> {
    Ok(Shape::new(&dims, parse_dtype(dtype)?))
}

fn parse_usize_pair(v: Vec<usize>, label: &str) -> PyResult<[usize; 2]> {
    if v.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{label} must have exactly 2 elements, got {}",
            v.len()
        )));
    }
    Ok([v[0], v[1]])
}

fn act_from_str(s: &str) -> PyResult<Activation> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "gelu" => Activation::Gelu,
        "gelu_approx" => Activation::GeluApprox,
        "silu" => Activation::Silu,
        "relu" => Activation::Relu,
        "sigmoid" => Activation::Sigmoid,
        "tanh" => Activation::Tanh,
        "exp" => Activation::Exp,
        "log" => Activation::Log,
        "sqrt" => Activation::Sqrt,
        "rsqrt" => Activation::Rsqrt,
        "neg" => Activation::Neg,
        "abs" => Activation::Abs,
        "round" => Activation::Round,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown activation '{other}'"
            )));
        }
    })
}

fn binop_from_str(s: &str) -> PyResult<BinaryOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "add" | "+" => BinaryOp::Add,
        "sub" | "-" => BinaryOp::Sub,
        "mul" | "*" => BinaryOp::Mul,
        "div" | "/" => BinaryOp::Div,
        "max" => BinaryOp::Max,
        "min" => BinaryOp::Min,
        "pow" => BinaryOp::Pow,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown binary op '{other}'"
            )));
        }
    })
}

fn cmp_from_str(s: &str) -> PyResult<CmpOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "eq" | "==" => CmpOp::Eq,
        "ne" | "!=" => CmpOp::Ne,
        "lt" | "<" => CmpOp::Lt,
        "le" | "<=" => CmpOp::Le,
        "gt" | ">" => CmpOp::Gt,
        "ge" | ">=" => CmpOp::Ge,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown comparison '{other}'"
            )));
        }
    })
}

fn compare_nodes(g: &mut Graph, op: CmpOp, a: u32, b: u32) -> PyResult<u32> {
    let s = rlx_ir::shape::compare_shape(g.shape(NodeId(a)), g.shape(NodeId(b)))
        .map_err(|e| PyValueError::new_err(format!("compare shape: {e}")))?;
    Ok(g.add_node(Op::Compare(op), vec![NodeId(a), NodeId(b)], s).0)
}

fn reduce_from_str(s: &str) -> PyResult<ReduceOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "sum" => ReduceOp::Sum,
        "mean" => ReduceOp::Mean,
        "max" => ReduceOp::Max,
        "min" => ReduceOp::Min,
        "prod" => ReduceOp::Prod,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown reduction '{other}'"
            )));
        }
    })
}

fn fft_norm_from_str(s: &str) -> PyResult<FftNorm> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "backward" | "none" => FftNorm::Backward,
        "forward" => FftNorm::Forward,
        "ortho" | "orthonormal" => FftNorm::Ortho,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown fft norm '{other}' (expected backward, forward, or ortho)"
            )));
        }
    })
}

/// Parse mask-kind strings: `"none"`, `"causal"`, or `"sliding:N"`.
fn mask_kind_from_str(s: &str) -> PyResult<MaskKind> {
    let lower = s.trim().to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("sliding:") {
        let w: usize = rest.parse().map_err(|_| {
            PyValueError::new_err(format!(
                "sliding window size must be a non-negative integer, got '{rest}'"
            ))
        })?;
        return Ok(MaskKind::SlidingWindow(w));
    }
    Ok(match lower.as_str() {
        "none" => MaskKind::None,
        "causal" => MaskKind::Causal,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown mask kind '{other}' (none, causal, sliding:<N>)"
            )));
        }
    })
}

// ── PyGraph ────────────────────────────────────────────────────

/// Symbolic computation graph consumed by [`crate::session::PySession::compile`].
///
/// Builder methods return integer node ids. Prefer shape-inferred helpers
/// (`matmul`, `gelu`, `conv2d`, …) over explicit-shape variants unless you
/// need a non-default output layout.
#[pyclass(name = "Graph", module = "pyrlx._pyrlx")]
pub(crate) struct PyGraph {
    /// `Option` so we can move it out at compile time (rlx-runtime
    /// consumes the Graph by value). After compile, the graph is gone.
    pub(crate) inner: Option<Graph>,
}

pub(crate) fn graph_ref(g: &PyGraph) -> PyResult<&Graph> {
    g.inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("graph already consumed"))
}

fn consumed() -> PyErr {
    PyRuntimeError::new_err("this Graph has been consumed by Session.compile() — build a new one")
}

impl PyGraph {
    fn g(&mut self) -> PyResult<&mut Graph> {
        self.inner.as_mut().ok_or_else(consumed)
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new(name: &str) -> Self {
        Self {
            inner: Some(Graph::new(name.to_string())),
        }
    }

    // ── I/O ─────────────────────────────────────────────────

    fn input(&mut self, name: &str, shape: Vec<usize>, dtype: &str) -> PyResult<u32> {
        let s = shape_from_py(shape, dtype)?;
        Ok(self.g()?.input(name.to_string(), s).0)
    }

    fn param(&mut self, name: &str, shape: Vec<usize>, dtype: &str) -> PyResult<u32> {
        let s = shape_from_py(shape, dtype)?;
        Ok(self.g()?.param(name.to_string(), s).0)
    }

    /// Broadcastable rank-0 literal.
    ///
    /// Inserts [`Op::Constant`] with [`Shape::scalar`]. Use in binary ops or
    /// pass to the DSL via ``x * 2.0``. Out-of-range values raise
    /// `ValueError`; `f16` / `bf16` are built as f32 constant + `cast`.
    #[pyo3(signature = (value, dtype = "f32"))]
    fn constant(&mut self, value: f64, dtype: &str) -> PyResult<u32> {
        let dt = parse_dtype(dtype)?;
        Ok(self
            .g()?
            .try_constant(value, dt)
            .map_err(PyValueError::new_err)?
            .0)
    }

    fn set_outputs(&mut self, outputs: Vec<u32>) -> PyResult<()> {
        self.g()?
            .set_outputs(outputs.into_iter().map(NodeId).collect());
        Ok(())
    }

    /// `(dims, dtype_str)` for a node — useful for debugging shape
    /// inference and writing parity tests against the Rust side.
    fn shape_of(&self, id: u32) -> PyResult<(Vec<i64>, &'static str)> {
        let g = self.inner.as_ref().ok_or_else(consumed)?;
        let s = g.shape(NodeId(id));
        let dims: Vec<i64> = s
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => *n as i64,
                rlx_ir::Dim::Dynamic(_) => -1,
            })
            .collect();
        Ok((dims, crate::dtype::dtype_label(s.dtype())))
    }

    // ── Linear algebra (inferred via GraphExt) ──────────────

    fn matmul(&mut self, lhs: u32, rhs: u32) -> PyResult<u32> {
        Ok(self.g()?.mm(NodeId(lhs), NodeId(rhs)).0)
    }
    fn matmul_with_shape(
        &mut self,
        lhs: u32,
        rhs: u32,
        out_shape: Vec<usize>,
        dtype: &str,
    ) -> PyResult<u32> {
        let s = shape_from_py(out_shape, dtype)?;
        Ok(self.g()?.matmul(NodeId(lhs), NodeId(rhs), s).0)
    }

    /// Native low-precision GEMM (TN: `lhs [m,k] · rhs [n,k]ᵀ → [m,n]`). Both
    /// operands are dynamically quantized to the minifloat `format`. `format` is
    /// an `fNeXmY` name — a named tensor-core format ("f8e4m3", "f8e5m2",
    /// "f6e2m3", "f6e3m2", "f4e2m1", the `…fnuz` variants) or a parameterized
    /// split like "f4e3m0". `layout` is "per_tensor", "mx", or "nvfp4". `rhs`
    /// must already be K-last (`[n, k]`).
    #[pyo3(signature = (lhs, rhs, format = "f8e4m3", layout = "per_tensor"))]
    fn scaled_matmul(&mut self, lhs: u32, rhs: u32, format: &str, layout: &str) -> PyResult<u32> {
        let fmt: rlx_ir::ScaledFormat = format
            .parse()
            .map_err(|e: String| PyValueError::new_err(e))?;
        let layout = scale_layout_from_py(layout)?;
        Ok(self
            .g()?
            .scaled_matmul(NodeId(lhs), NodeId(rhs), fmt, layout)
            .0)
    }

    fn lora_matmul(
        &mut self,
        x: u32,
        w: u32,
        a: u32,
        b: u32,
        scale: f32,
        out_shape: Vec<usize>,
        dtype: &str,
    ) -> PyResult<u32> {
        let s = shape_from_py(out_shape, dtype)?;
        Ok(self
            .g()?
            .lora_matmul(NodeId(x), NodeId(w), NodeId(a), NodeId(b), scale, s)
            .0)
    }

    /// Dense linear solve `x = A⁻¹·b`. Inputs: `a [N, N]`, `b [N]` or
    /// `b [N, K]`. Output shape matches `b`. Today f64 is the only
    /// dtype with a CPU lowering — the IR + autodiff are dtype-agnostic
    /// but the kernel uses LAPACK `dgesv`.
    fn dense_solve(&mut self, a: u32, b: u32, out_shape: Vec<usize>, dtype: &str) -> PyResult<u32> {
        let s = shape_from_py(out_shape, dtype)?;
        Ok(self.g()?.dense_solve(NodeId(a), NodeId(b), s).0)
    }

    /// Batched dense linear solve. Inputs: `a [B, N, N]`, `b [B, N]`
    /// or `b [B, N, K]`. Output shape matches `b`. Per-batch
    /// independent — typically constructed by `vmap` of `dense_solve`,
    /// but exposed here too for hand-written batched workloads.
    fn batched_dense_solve(
        &mut self,
        a: u32,
        b: u32,
        out_shape: Vec<usize>,
        dtype: &str,
    ) -> PyResult<u32> {
        let s = shape_from_py(out_shape, dtype)?;
        Ok(self.g()?.batched_dense_solve(NodeId(a), NodeId(b), s).0)
    }

    /// User-defined sub-graph with optional override AD rules
    /// (JAX-shaped `custom_vjp` / `custom_jvp`). Takes ownership of
    /// `fwd_body`, `vjp_body`, and `jvp_body` — pass `None` to fall
    /// through to natural autodiff for that direction. See
    /// `rlx_ir::Op::CustomFn` for the body-graph naming conventions
    /// (`primal_output`, `d_output`, `tangent_<i>`).
    #[pyo3(signature = (inputs, fwd_body, vjp_body=None, jvp_body=None))]
    fn custom_fn(
        &mut self,
        inputs: Vec<u32>,
        fwd_body: &Bound<'_, PyGraph>,
        vjp_body: Option<&Bound<'_, PyGraph>>,
        jvp_body: Option<&Bound<'_, PyGraph>>,
    ) -> PyResult<u32> {
        let fwd_g = fwd_body.borrow_mut().inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("custom_fn: fwd_body has already been consumed")
        })?;
        let vjp_g = match vjp_body {
            Some(b) => Some(b.borrow_mut().inner.take().ok_or_else(|| {
                PyRuntimeError::new_err("custom_fn: vjp_body has already been consumed")
            })?),
            None => None,
        };
        let jvp_g = match jvp_body {
            Some(b) => Some(b.borrow_mut().inner.take().ok_or_else(|| {
                PyRuntimeError::new_err("custom_fn: jvp_body has already been consumed")
            })?),
            None => None,
        };
        let ids: Vec<NodeId> = inputs.into_iter().map(NodeId).collect();
        Ok(self.g()?.custom_fn(ids, fwd_g, vjp_g, jvp_g).0)
    }

    #[pyo3(signature = (input, weight, bias, out_shape, dtype, activation=None))]
    fn fused_matmul_bias_act(
        &mut self,
        input: u32,
        weight: u32,
        bias: u32,
        out_shape: Vec<usize>,
        dtype: &str,
        activation: Option<&str>,
    ) -> PyResult<u32> {
        let s = shape_from_py(out_shape, dtype)?;
        let act = match activation {
            Some(a) => Some(act_from_str(a)?),
            None => None,
        };
        Ok(self
            .g()?
            .fused_matmul_bias_act(NodeId(input), NodeId(weight), NodeId(bias), act, s)
            .0)
    }

    // ── Element-wise ────────────────────────────────────────

    fn binary(&mut self, op: &str, lhs: u32, rhs: u32) -> PyResult<u32> {
        let bo = binop_from_str(op)?;
        // GraphExt has add/sub/mul/div but not max/min/pow — fall back
        // to explicit shape inference via `binary_shape` for those.
        let g = self.g()?;
        let id = match bo {
            BinaryOp::Add => g.add(NodeId(lhs), NodeId(rhs)),
            BinaryOp::Sub => g.sub(NodeId(lhs), NodeId(rhs)),
            BinaryOp::Mul => g.mul(NodeId(lhs), NodeId(rhs)),
            BinaryOp::Div => g.div(NodeId(lhs), NodeId(rhs)),
            other => {
                let s = rlx_ir::shape::binary_shape(g.shape(NodeId(lhs)), g.shape(NodeId(rhs)))
                    .map_err(|e| PyValueError::new_err(format!("binary({other:?}) shape: {e}")))?;
                g.binary(other, NodeId(lhs), NodeId(rhs), s)
            }
        };
        Ok(id.0)
    }

    fn add(&mut self, a: u32, b: u32) -> PyResult<u32> {
        Ok(self.g()?.add(NodeId(a), NodeId(b)).0)
    }
    fn sub(&mut self, a: u32, b: u32) -> PyResult<u32> {
        Ok(self.g()?.sub(NodeId(a), NodeId(b)).0)
    }
    fn mul(&mut self, a: u32, b: u32) -> PyResult<u32> {
        Ok(self.g()?.mul(NodeId(a), NodeId(b)).0)
    }
    fn div(&mut self, a: u32, b: u32) -> PyResult<u32> {
        Ok(self.g()?.div(NodeId(a), NodeId(b)).0)
    }

    fn activation(&mut self, kind: &str, input: u32) -> PyResult<u32> {
        // unary_shape == identity, so use the explicit builder
        // for consistency with non-GraphExt activations.
        let g = self.g()?;
        let s = rlx_ir::shape::unary_shape(g.shape(NodeId(input)));
        Ok(g.activation(act_from_str(kind)?, NodeId(input), s).0)
    }

    // Shorthand activations (PyTorch / GraphExt parity)
    fn gelu(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.gelu(NodeId(x)).0)
    }
    /// Tanh-approximation GELU (PyTorch / many ViT checkpoints).
    fn gelu_approx(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.gelu_approx(NodeId(x)).0)
    }
    fn silu(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.silu(NodeId(x)).0)
    }
    fn relu(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.relu(NodeId(x)).0)
    }

    /// Nearest-neighbor 2x upsample (NCHW: H and W double).
    fn resize_nearest_2x(&mut self, x: u32) -> PyResult<u32> {
        let g = self.g()?;
        let in_s = g.shape(NodeId(x));
        if in_s.rank() != 4 {
            return Err(PyValueError::new_err(
                "resize_nearest_2x requires 4D NCHW input",
            ));
        }
        let (n, c, h, w) = (
            in_s.dim(0).unwrap_static(),
            in_s.dim(1).unwrap_static(),
            in_s.dim(2).unwrap_static(),
            in_s.dim(3).unwrap_static(),
        );
        let out_s = Shape::new(&[n, c, h * 2, w * 2], in_s.dtype());
        Ok(g.add_node(Op::ResizeNearest2x, vec![NodeId(x)], out_s).0)
    }
    fn exp(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.exp(NodeId(x)).0)
    }
    fn sqrt(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.sqrt(NodeId(x)).0)
    }
    fn neg(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.neg(NodeId(x)).0)
    }
    fn tanh(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.tanh(NodeId(x)).0)
    }

    /// Identity forward, zero backward (`jax.lax.stop_gradient` / `detach`).
    fn stop_gradient(&mut self, x: u32) -> PyResult<u32> {
        Ok(self.g()?.stop_gradient(NodeId(x)).0)
    }

    // ── Convolution (NCHW, shape-inferred) ────────────────

    /// NCHW convolution; output shape inferred from padding / stride / dilation.
    #[pyo3(signature = (input, weight, kernel_size, stride, padding, dilation, groups = 1))]
    fn conv2d(
        &mut self,
        input: u32,
        weight: u32,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    ) -> PyResult<u32> {
        Ok(self
            .g()?
            .conv2d(
                NodeId(input),
                NodeId(weight),
                parse_usize_pair(kernel_size, "kernel_size")?,
                parse_usize_pair(stride, "stride")?,
                parse_usize_pair(padding, "padding")?,
                parse_usize_pair(dilation, "dilation")?,
                groups,
            )
            .0)
    }

    /// Transposed NCHW convolution (upsampling decoder path).
    #[pyo3(signature = (
        input, weight, kernel_size, stride, padding, dilation,
        output_padding, groups = 1
    ))]
    fn conv_transpose2d(
        &mut self,
        input: u32,
        weight: u32,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        output_padding: Vec<usize>,
        groups: usize,
    ) -> PyResult<u32> {
        Ok(self
            .g()?
            .conv_transpose2d(
                NodeId(input),
                NodeId(weight),
                parse_usize_pair(kernel_size, "kernel_size")?,
                parse_usize_pair(stride, "stride")?,
                parse_usize_pair(padding, "padding")?,
                parse_usize_pair(dilation, "dilation")?,
                parse_usize_pair(output_padding, "output_padding")?,
                groups,
            )
            .0)
    }

    // ── Comparison ──────────────────────────────────────────
    // Trailing underscore avoids shadowing Python's reserved keywords
    // and the `eq` / `lt` magic-method names users might rely on.

    fn compare(&mut self, op: &str, lhs: u32, rhs: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, cmp_from_str(op)?, lhs, rhs)
    }

    fn eq_(&mut self, a: u32, b: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, CmpOp::Eq, a, b)
    }
    fn lt_(&mut self, a: u32, b: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, CmpOp::Lt, a, b)
    }
    fn gt_(&mut self, a: u32, b: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, CmpOp::Gt, a, b)
    }
    fn ge_(&mut self, a: u32, b: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, CmpOp::Ge, a, b)
    }
    fn ne_(&mut self, a: u32, b: u32) -> PyResult<u32> {
        compare_nodes(self.g()?, CmpOp::Ne, a, b)
    }

    /// `where(cond, on_true, on_false)` — Python keyword clash fixed
    /// with a trailing underscore.
    fn where_(&mut self, cond: u32, a: u32, b: u32) -> PyResult<u32> {
        let g = self.g()?;
        let s = rlx_ir::shape::unary_shape(g.shape(NodeId(a)));
        Ok(
            g.add_node(Op::Where, vec![NodeId(cond), NodeId(a), NodeId(b)], s)
                .0,
        )
    }

    // ── Reduction ───────────────────────────────────────────

    #[pyo3(signature = (input, op, axes, keep_dim = false))]
    fn reduce(&mut self, input: u32, op: &str, axes: Vec<usize>, keep_dim: bool) -> PyResult<u32> {
        let g = self.g()?;
        let s = rlx_ir::shape::reduce_shape(g.shape(NodeId(input)), &axes, keep_dim)
            .map_err(|e| PyValueError::new_err(format!("reduce shape: {e}")))?;
        Ok(
            g.reduce(NodeId(input), reduce_from_str(op)?, axes, keep_dim, s)
                .0,
        )
    }
    fn sum(&mut self, x: u32, axes: Vec<usize>, keep_dim: bool) -> PyResult<u32> {
        Ok(self.g()?.sum(NodeId(x), axes, keep_dim).0)
    }
    fn mean(&mut self, x: u32, axes: Vec<usize>, keep_dim: bool) -> PyResult<u32> {
        Ok(self.g()?.mean(NodeId(x), axes, keep_dim).0)
    }

    #[pyo3(signature = (input, axis = -1))]
    fn softmax(&mut self, input: u32, axis: i32) -> PyResult<u32> {
        Ok(self.g()?.sm(NodeId(input), axis).0)
    }

    #[pyo3(signature = (input, axis, exclusive = false))]
    fn cumsum(&mut self, input: u32, axis: i32, exclusive: bool) -> PyResult<u32> {
        let g = self.g()?;
        let s = rlx_ir::shape::unary_shape(g.shape(NodeId(input)));
        Ok(g.cumsum(NodeId(input), axis, exclusive, s).0)
    }

    #[pyo3(signature = (logits, top_k, top_p, temperature, seed, output_shape, dtype = "i32"))]
    fn sample(
        &mut self,
        logits: u32,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
        output_shape: Vec<usize>,
        dtype: &str,
    ) -> PyResult<u32> {
        let s = shape_from_py(output_shape, dtype)?;
        Ok(self
            .g()?
            .sample(NodeId(logits), top_k, top_p, temperature, seed, s)
            .0)
    }

    // ── Shape ops ───────────────────────────────────────────

    fn reshape(&mut self, x: u32, new_shape: Vec<i64>) -> PyResult<u32> {
        Ok(self.g()?.reshape_(NodeId(x), new_shape).0)
    }
    fn transpose(&mut self, x: u32, perm: Vec<usize>) -> PyResult<u32> {
        Ok(self.g()?.transpose_(NodeId(x), perm).0)
    }
    fn narrow(&mut self, x: u32, axis: usize, start: usize, length: usize) -> PyResult<u32> {
        Ok(self.g()?.narrow_(NodeId(x), axis, start, length).0)
    }
    fn concat(&mut self, inputs: Vec<u32>, axis: usize) -> PyResult<u32> {
        let ids = inputs.into_iter().map(NodeId).collect();
        Ok(self.g()?.concat_(ids, axis).0)
    }
    fn gather(&mut self, table: u32, indices: u32, axis: usize) -> PyResult<u32> {
        Ok(self.g()?.gather_(NodeId(table), NodeId(indices), axis).0)
    }

    fn cast(&mut self, x: u32, to: &str) -> PyResult<u32> {
        Ok(self.g()?.cast(NodeId(x), parse_dtype(to)?).0)
    }

    // ── FFT ─────────────────────────────────────────────────

    #[pyo3(signature = (x, inverse = false))]
    fn fft(&mut self, x: u32, inverse: bool) -> PyResult<u32> {
        Ok(self.g()?.fft(NodeId(x), inverse).0)
    }

    #[pyo3(signature = (x, inverse = false, norm = "backward"))]
    fn fft_norm(&mut self, x: u32, inverse: bool, norm: &str) -> PyResult<u32> {
        Ok(self
            .g()?
            .fft_norm(NodeId(x), inverse, fft_norm_from_str(norm)?)
            .0)
    }

    #[pyo3(signature = (x, norm = "backward"))]
    fn fft_real(&mut self, x: u32, norm: &str) -> PyResult<(u32, u32)> {
        let (re, im) = self.g()?.fft_real(NodeId(x), fft_norm_from_str(norm)?);
        Ok((re.0, im.0))
    }

    #[pyo3(signature = (x, norm = "backward"))]
    fn rfft(&mut self, x: u32, norm: &str) -> PyResult<(u32, u32)> {
        let (re, im) = self.g()?.rfft(NodeId(x), fft_norm_from_str(norm)?);
        Ok((re.0, im.0))
    }

    #[pyo3(signature = (re, im, n, norm = "backward"))]
    fn irfft(&mut self, re: u32, im: u32, n: usize, norm: &str) -> PyResult<u32> {
        Ok(self
            .g()?
            .irfft(NodeId(re), NodeId(im), n, fft_norm_from_str(norm)?)
            .0)
    }

    fn fftfreq(&mut self, n: usize) -> PyResult<u32> {
        Ok(self.g()?.fftfreq_tensor(n).0)
    }

    fn rfftfreq(&mut self, n: usize) -> PyResult<u32> {
        Ok(self.g()?.rfftfreq_tensor(n).0)
    }

    fn psd(&mut self, re: u32, im: u32) -> PyResult<u32> {
        Ok(self.g()?.psd(NodeId(re), NodeId(im)).0)
    }

    #[pyo3(signature = (x, norm = "backward"))]
    fn psd_real(&mut self, x: u32, norm: &str) -> PyResult<u32> {
        Ok(self.g()?.psd_real(NodeId(x), fft_norm_from_str(norm)?).0)
    }

    // ── Normalization ───────────────────────────────────────

    #[pyo3(signature = (input, gamma, beta, axis = -1, eps = 1e-5))]
    fn layer_norm(
        &mut self,
        input: u32,
        gamma: u32,
        beta: u32,
        axis: i32,
        eps: f32,
    ) -> PyResult<u32> {
        let g = self.g()?;
        let s = rlx_ir::shape::unary_shape(g.shape(NodeId(input)));
        Ok(
            g.layer_norm(NodeId(input), NodeId(gamma), NodeId(beta), axis, eps, s)
                .0,
        )
    }

    #[pyo3(signature = (input, gamma, beta, eps = 1e-5))]
    fn rms_norm(&mut self, input: u32, gamma: u32, beta: u32, eps: f32) -> PyResult<u32> {
        Ok(self
            .g()?
            .rms_norm(NodeId(input), NodeId(gamma), NodeId(beta), eps)
            .0)
    }

    /// LayerNorm over channel dim for NCHW feature maps.
    #[pyo3(signature = (input, gamma, beta, eps = 1e-5))]
    fn layer_norm2d(&mut self, input: u32, gamma: u32, beta: u32, eps: f32) -> PyResult<u32> {
        Ok(self
            .g()?
            .layer_norm2d(NodeId(input), NodeId(gamma), NodeId(beta), eps)
            .0)
    }

    /// GroupNorm for NCHW tensors.
    #[pyo3(signature = (input, gamma, beta, num_groups, eps = 1e-5))]
    fn group_norm(
        &mut self,
        input: u32,
        gamma: u32,
        beta: u32,
        num_groups: usize,
        eps: f32,
    ) -> PyResult<u32> {
        Ok(self
            .g()?
            .group_norm(NodeId(input), NodeId(gamma), NodeId(beta), num_groups, eps)
            .0)
    }

    // ── Attention ───────────────────────────────────────────

    /// Attention with an explicit mask tensor (`MaskKind::Custom`).
    fn attention(
        &mut self,
        q: u32,
        k: u32,
        v: u32,
        mask: u32,
        num_heads: usize,
        head_dim: usize,
    ) -> PyResult<u32> {
        Ok(self
            .g()?
            .attention_(
                NodeId(q),
                NodeId(k),
                NodeId(v),
                NodeId(mask),
                num_heads,
                head_dim,
            )
            .0)
    }

    /// Attention with a kernel-synthesized mask:
    ///   `mask_kind="none" | "causal" | "sliding:<N>"`.
    /// No mask tensor is allocated; out shape == Q shape.
    fn attention_kind(
        &mut self,
        q: u32,
        k: u32,
        v: u32,
        num_heads: usize,
        head_dim: usize,
        mask_kind: &str,
    ) -> PyResult<u32> {
        let kind = mask_kind_from_str(mask_kind)?;
        let g = self.g()?;
        let s = rlx_ir::shape::attention_shape(g.shape(NodeId(q)));
        Ok(g.attention_kind(
            NodeId(q),
            NodeId(k),
            NodeId(v),
            num_heads,
            head_dim,
            kind,
            s,
        )
        .0)
    }

    // ── RoPE ────────────────────────────────────────────────

    fn rope(&mut self, x: u32, cos: u32, sin: u32, head_dim: usize) -> PyResult<u32> {
        Ok(self
            .g()?
            .rope(NodeId(x), NodeId(cos), NodeId(sin), head_dim)
            .0)
    }

    /// Partial RoPE: rotate the first `n_rot` dims (must be even, ≤ `head_dim`).
    #[pyo3(signature = (x, cos, sin, head_dim, n_rot))]
    fn rope_n(
        &mut self,
        x: u32,
        cos: u32,
        sin: u32,
        head_dim: usize,
        n_rot: usize,
    ) -> PyResult<u32> {
        Ok(self
            .g()?
            .rope_n(NodeId(x), NodeId(cos), NodeId(sin), head_dim, n_rot)
            .0)
    }

    // ── Misc ────────────────────────────────────────────────

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.as_ref().ok_or_else(consumed)?.len())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(g) => format!("<pyrlx.Graph nodes={}>", g.len()),
            None => "<pyrlx.Graph (consumed by compile)>".into(),
        }
    }
}
