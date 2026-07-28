// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The internal, rlx-shaped op vocabulary that the aten registry lowers to.
//!
//! Each aten node expands into one or more [`Instr`] (SSA: a result name + a
//! [`Call`]). Two consumers walk the same `Vec<Instr>`: [`crate::hir_build`]
//! (builds a live `HirModule` to run/verify) and [`crate::emit`] (prints the
//! generated crate's `graph.rs`). Keeping the aten→rlx mapping in one place
//! (the registry) means run-parity and generated-code parity can never drift.

use anyhow::{Result, bail};
use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Shape};

/// SSA value name (an FX node id, input id, weight id, or a synthesized
/// intermediate like `addmm__mm`).
pub type Value = String;

#[derive(Debug, Clone)]
pub struct InputDef {
    pub name: String,
    /// Concrete example extent per axis (a dynamic axis carries its example size,
    /// used for static shape-inference during lowering; the compile pass re-infers
    /// the symbolic shape and `DimBinding` specializes it per run).
    pub shape: Vec<usize>,
    /// Per-axis dynamic symbol: `Some(sym)` marks a dynamic dim, `None` static.
    pub dyn_dims: Vec<Option<u32>>,
    pub dtype: DType,
}

impl InputDef {
    /// The HIR input shape at `dtype`, dynamic axes as `Dim::Dynamic(sym)`.
    pub fn hir_shape(&self, dtype: DType) -> Shape {
        if self.dyn_dims.iter().all(Option::is_none) {
            return shape_of(&self.shape, dtype);
        }
        let dims: Vec<rlx_ir::shape::Dim> = self
            .shape
            .iter()
            .enumerate()
            .map(|(ax, &s)| match self.dyn_dims.get(ax).copied().flatten() {
                Some(sym) => rlx_ir::shape::Dim::Dynamic(sym),
                None => rlx_ir::shape::Dim::Static(s),
            })
            .collect();
        Shape::from_dims(&dims, dtype)
    }
    /// Whether any axis is dynamic.
    pub fn is_dynamic(&self) -> bool {
        self.dyn_dims.iter().any(Option::is_some)
    }
}

#[derive(Debug, Clone)]
pub struct ParamDef {
    /// SSA name args reference (FX placeholder id).
    pub value_id: String,
    /// HIR param name + safetensors key (state_dict FQN).
    pub key: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// An rlx-shaped op with resolved operands. Mirrors `HirGraphExt` methods.
#[derive(Debug, Clone)]
pub enum Call {
    /// Matmul (possibly batched via broadcasting leading dims).
    Mm(Value, Value),
    /// Elementwise add/sub/mul/div (with numpy-style broadcasting).
    Binary(BinaryOp, Value, Value),
    /// Unary activation (gelu/silu/relu/sigmoid/tanh/exp/sqrt/neg/...).
    Act(Activation, Value),
    /// LayerNorm over the last axis.
    Ln {
        x: Value,
        gamma: Value,
        beta: Value,
        eps: f32,
    },
    /// RMSNorm over the last axis (`beta` is a synthesized zero param).
    RmsNorm {
        x: Value,
        gamma: Value,
        beta: Value,
        eps: f32,
    },
    /// Reshape/view to a concrete shape.
    Reshape { x: Value, shape: Vec<i64> },
    /// Permute axes.
    Transpose { x: Value, perm: Vec<usize> },
    /// Slice a single axis (step 1).
    Narrow {
        x: Value,
        axis: usize,
        start: usize,
        len: usize,
    },
    /// Concatenate along an axis.
    Concat { xs: Vec<Value>, axis: usize },
    /// Gather rows / embedding lookup.
    Gather {
        table: Value,
        indices: Value,
        axis: usize,
    },
    /// Softmax over an axis.
    Softmax { x: Value, axis: i32 },
    /// Reduce (sum/mean/...) over axes.
    Reduce {
        op: ReduceOp,
        x: Value,
        axes: Vec<usize>,
        keep_dim: bool,
    },
    /// Dtype cast.
    Cast { x: Value, to: DType },
    /// 2-D convolution (no bias — bias is a separate broadcast add).
    Conv2d {
        x: Value,
        weight: Value,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        groups: usize,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// Scaled dot-product attention (q/k/v already `[B, H, S, D]`).
    Attention {
        q: Value,
        k: Value,
        v: Value,
        num_heads: usize,
        head_dim: usize,
        mask: MaskKind,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// Rotary position embedding (NeoX pairing).
    Rope {
        x: Value,
        cos: Value,
        sin: Value,
        head_dim: usize,
    },
    /// A constant tensor filled with a single scalar value (aten full/full_like).
    Full {
        value: f64,
        shape: Vec<usize>,
        dtype: DType,
    },
    /// 2-D transposed convolution (no bias — bias is a separate broadcast add).
    ConvTranspose2d {
        x: Value,
        weight: Value,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_padding: [usize; 2],
        groups: usize,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// Attention with an explicit additive bias/mask tensor (4th operand).
    AttentionBias {
        q: Value,
        k: Value,
        v: Value,
        bias: Value,
        num_heads: usize,
        head_dim: usize,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// Column vector `[[0], [step], [2*step], …]` of shape `[rows, 1]` — the
    /// per-row base offset used to turn per-row indices into flat gather
    /// indices (`x_flat[row*E + idx]`).
    Iota {
        rows: usize,
        step: i64,
        dtype: DType,
    },
    /// 1-D constant ramp `[start, start+step, …]` of length `len` (aten arange).
    Arange {
        start: f64,
        step: f64,
        len: usize,
        dtype: DType,
    },
    /// Bilinear/bicubic 2-D resize (NCHW) to an explicit `[out_h, out_w]`,
    /// decomposed by the HIR builder into separable interpolation matmuls
    /// (universal ops). `cubic` selects the bicubic kernel; `antialias` the
    /// widened downsampling filter (`_aa` overloads).
    Resize {
        x: Value,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        cubic: bool,
        antialias: bool,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// `grid_sample(input[N,C,H,W], grid[N,Ho,Wo,2])` → `[N,C,Ho,Wo]`, decomposed
    /// by the HIR builder (gather + arithmetic; all modes/paddings). Universal ops.
    GridSample {
        input: Value,
        grid: Value,
        mode: rlx_ir::hir::GridMode,
        pad: rlx_ir::hir::GridPad,
        align_corners: bool,
        out: Vec<usize>,
        out_dtype: DType,
    },
    /// A "direct" single-`Op` node defined by the [`crate::nodeop`] macro table
    /// (Compare / Where / Expand / BinaryShaped / Pool / TopK / …). Build + emit
    /// are generated from one table entry per op.
    Node(crate::nodeop::NodeOp),
    /// Result is an alias of an existing value (clone/contiguous/detach/no-op).
    Alias(Value),
}

#[derive(Debug, Clone)]
pub struct Instr {
    pub result: Value,
    pub call: Call,
    /// Provenance comment (original aten op + shapes/dtypes) for the generated
    /// crate, so a reader can trace each rlx op back to its PyTorch source.
    pub note: Option<String>,
}

/// The fully-lowered model: everything needed to build a `HirModule` or emit
/// generated source, with no residual aten vocabulary.
#[derive(Debug, Clone, Default)]
pub struct Lowered {
    pub name: String,
    pub inputs: Vec<InputDef>,
    pub params: Vec<ParamDef>,
    /// Synthesized zero-filled params (e.g. RMSNorm beta) — not in safetensors.
    pub zero_params: Vec<ParamDef>,
    pub instrs: Vec<Instr>,
    pub outputs: Vec<Value>,
    /// Original aten op histogram (op string, count), for a provenance header.
    pub source_histogram: Vec<(String, usize)>,
}

// ── shape / dtype helpers ────────────────────────────────────────────────────
pub fn dtype_from_str(s: &str) -> Result<DType> {
    Ok(match s {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f64" => DType::F64,
        "i8" => DType::I8,
        "i16" => DType::I16,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "u8" => DType::U8,
        "u32" => DType::U32,
        "bool" => DType::Bool,
        "c64" => DType::C64,
        "c128" => DType::C128,
        other => bail!("unsupported dtype token: {other:?}"),
    })
}

pub fn dtype_token(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F64 => "F64",
        DType::I8 => "I8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::Bool => "Bool",
        DType::C64 => "C64",
        DType::C128 => "C128",
    }
}

pub fn shape_of(dims: &[usize], dt: DType) -> Shape {
    Shape::new(dims, dt)
}

pub fn is_float_dtype(dt: DType) -> bool {
    matches!(dt, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
}

pub fn dims_usize(dims: &[i64]) -> Vec<usize> {
    dims.iter().map(|&d| d.max(0) as usize).collect()
}

/// Row-major little-endian bytes for a `numel`-length tensor filled with `value`.
pub fn fill_bytes(value: f64, dtype: DType, numel: usize) -> Result<Vec<u8>> {
    let one: Vec<u8> = match dtype {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F64 => value.to_le_bytes().to_vec(),
        DType::I64 => (value as i64).to_le_bytes().to_vec(),
        DType::I32 => (value as i32).to_le_bytes().to_vec(),
        DType::Bool => vec![(value != 0.0) as u8],
        other => bail!("full/full_like with dtype {other:?} not supported"),
    };
    let mut out = Vec::with_capacity(one.len() * numel);
    for _ in 0..numel {
        out.extend_from_slice(&one);
    }
    Ok(out)
}
