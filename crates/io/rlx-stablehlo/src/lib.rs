// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Export an rlx graph as StableHLO MLIR.**
//!
//! The text this produces is the input format of IREE
//! (`iree-compile --iree-input-type=stablehlo`) and of other MLIR toolchains
//! built on it, including vendor NPU plugins.
//!
//! ```text
//!   rlx Graph
//!     │  decompose backward ops, legalize to the HLO op set  (as rlx-tpu does)
//!     │  rlx_tpu::ir_passes::prepare_graph_for_hlo
//!     │  rlx_tpu::lower_graph_with_rng_and_params            → HloModuleProto
//!     ▼
//!   print.rs                                                 → StableHLO text
//! ```
//!
//! The lowering is the TPU backend's, unchanged: one rlx → HLO mapping, two
//! sinks (PJRT bytes for TPU, StableHLO text here), so op coverage and
//! semantics cannot drift apart.
//!
//! **Layout.** rlx convolutions are NCHW / OIHW. StableHLO spells the layout in
//! each convolution's `dimension_numbers`, so the graph exports as-is with no
//! transposes; IREE picks its own internal layout.
//!
//! **Fused ops** (attention, norms, RoPE, …) are exported as their primitive
//! decompositions. IREE does not accept `stablehlo.composite`.
//!
//! **Not exported:** graphs that need host-executed segments on the TPU path
//! (`rlx_tpu::segment::needs_orchestration`) and ops lowered to a
//! `custom-call`; both return [`ExportError::Unsupported`].

mod print;

use rlx_ir::{DType, Graph};

/// Why a graph could not be exported.
#[derive(Debug, Clone)]
pub enum ExportError {
    /// The graph uses op kinds the HLO op set cannot express.
    Legalize(String),
    /// A construct with no StableHLO form here.
    Unsupported(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Legalize(s) => write!(f, "rlx-stablehlo: {s}"),
            ExportError::Unsupported(s) => write!(f, "rlx-stablehlo: unsupported: {s}"),
        }
    }
}

impl std::error::Error for ExportError {}

/// Where the graph's `Param`s (weights) go in the exported module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Weights {
    /// Function arguments after the inputs: the caller passes every weight on
    /// every call.
    Arguments,
    /// IREE parameters: each `Param` becomes a `util.global` bound to
    /// `#flow.parameter.named<scope::name>`, read from a parameter file at
    /// load time (`iree-run-module --parameters=<scope>=weights.safetensors`;
    /// `.irpa` and `.gguf` work too). The function takes only the inputs.
    /// [`write_safetensors`] writes such a file.
    Parameters { scope: String },
}

/// Export options.
#[derive(Debug, Clone)]
pub struct StableHloOptions {
    /// Name of the exported function (`iree-run-module --function=…`).
    pub function: String,
    /// Seed and stream for random ops, as for compiled graphs.
    pub rng: rlx_ir::RngOptions,
    pub weights: Weights,
}

impl Default for StableHloOptions {
    fn default() -> Self {
        Self {
            function: "main".into(),
            rng: rlx_ir::RngOptions::default(),
            weights: Weights::Arguments,
        }
    }
}

impl StableHloOptions {
    /// Weights as IREE parameters under `scope` (see [`Weights::Parameters`]).
    pub fn with_parameters(mut self, scope: impl Into<String>) -> Self {
        self.weights = Weights::Parameters {
            scope: scope.into(),
        };
        self
    }
}

/// One function argument or result.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSig {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
}

/// One exported function's calling convention.
#[derive(Debug, Clone)]
pub struct FunctionSig {
    pub name: String,
    pub args: Vec<TensorSig>,
    pub parameters: Vec<TensorSig>,
    pub results: Vec<TensorSig>,
}

/// An exported module and its calling convention.
///
/// `args` / `parameters` / `results` describe the first (for [`export`], the
/// only) function; `functions` lists every function in the module.
#[derive(Debug, Clone)]
pub struct StableHloModule {
    /// The MLIR text.
    pub text: String,
    /// Function arguments in order: the graph's `Input`s, then — with
    /// [`Weights::Arguments`] — its `Param`s.
    pub args: Vec<TensorSig>,
    /// With [`Weights::Parameters`], the `Param`s the module reads by name.
    pub parameters: Vec<TensorSig>,
    /// Function results, in `graph.outputs` order (names are `output{i}`).
    pub results: Vec<TensorSig>,
    pub functions: Vec<FunctionSig>,
}

/// Export `graph` as StableHLO.
///
/// Shapes must be static: bind symbolic dims first
/// (`rlx_ir::DimBinding`). Weights are function arguments, not constants, so
/// the module stays small and one export serves every checkpoint.
pub fn export(graph: Graph, opts: &StableHloOptions) -> Result<StableHloModule, ExportError> {
    let (printed, sig) = export_function(graph, opts, &opts.function)?;
    Ok(module(vec![printed], vec![sig]))
}

/// Export one static specialization of a symbolic `graph` per binding, as
/// functions `@{function}_{suffix}` in one module — the usual way to serve
/// several sequence lengths (prefill buckets) from one artifact. With
/// [`Weights::Parameters`] every function reads the same parameters, so one
/// weight file serves them all.
///
/// ```ignore
/// use rlx_ir::{DimBinding, dynamic::sym};
/// let specs: Vec<(String, DimBinding)> = [16, 128, 512]
///     .iter()
///     .map(|&s| (format!("seq{s}"), DimBinding::from_pairs(&[(sym::SEQ, s)])))
///     .collect();
/// let m = rlx_stablehlo::export_specializations(graph, &specs, &opts)?;
/// ```
pub fn export_specializations<S: AsRef<str>>(
    graph: Graph,
    specs: &[(S, rlx_ir::DimBinding)],
    opts: &StableHloOptions,
) -> Result<StableHloModule, ExportError> {
    if specs.is_empty() {
        return Err(ExportError::Unsupported("no specializations given".into()));
    }
    let mut printed = Vec::new();
    let mut sigs = Vec::new();
    for (suffix, binding) in specs {
        let g = rlx_ir::dynamic::specialize(&graph, binding);
        let name = format!("{}_{}", opts.function, suffix.as_ref());
        let (p, sig) = export_function(g, opts, &name)?;
        printed.push(p);
        sigs.push(sig);
    }
    Ok(module(printed, sigs))
}

fn module(printed: Vec<print::Printed>, functions: Vec<FunctionSig>) -> StableHloModule {
    let first = functions[0].clone();
    StableHloModule {
        text: print::assemble(&printed),
        args: first.args,
        parameters: first.parameters,
        results: first.results,
        functions,
    }
}

fn export_function(
    graph: Graph,
    opts: &StableHloOptions,
    func_name: &str,
) -> Result<(print::Printed, FunctionSig), ExportError> {
    if let Some(n) = graph.nodes().iter().find(|n| !n.shape.is_static()) {
        return Err(ExportError::Unsupported(format!(
            "symbolic shape on node {:?} — bind dims before exporting",
            n.id
        )));
    }
    // Same preparation as the TPU backend, minus mixed precision: an export
    // keeps the graph's dtypes.
    let graph = rlx_autodiff::decompose_backward::decompose_backward_ops_except(
        graph,
        rlx_tpu::SUPPORTED_OPS,
    );
    let graph = rlx_compile::rewrite::legalize_or_rewrite_for_backend_with_config(
        graph,
        rlx_tpu::SUPPORTED_OPS,
        Default::default(),
    )
    .map_err(|errors| {
        ExportError::Legalize(rlx_compile::format_legalize_error("stablehlo", &errors))
    })?;
    let graph = rlx_tpu::ir_passes::prepare_graph_for_hlo(graph);
    if rlx_tpu::segment::needs_orchestration(&graph) {
        return Err(ExportError::Unsupported(
            "the graph needs host-executed segments (ops with no HLO lowering)".into(),
        ));
    }

    let lowered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rlx_tpu::lower_graph_with_rng_and_params(&graph, opts.rng, None)
    }))
    .map_err(|p| {
        let msg = p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "lowering panicked".into());
        ExportError::Unsupported(msg)
    })?;
    if !lowered.gguf_deferred.is_empty() {
        return Err(ExportError::Unsupported(
            "GGUF DequantMatMul weights supplied at run time; dequantize before exporting".into(),
        ));
    }

    let proto = <rlx_tpu::xla::HloModuleProto as prost::Message>::decode(lowered.bytes.as_slice())
        .map_err(|e| ExportError::Unsupported(format!("decoding the lowered module: {e}")))?;

    let sigs = |names: &[String], dtypes: &[DType], shapes: &[Vec<i64>]| -> Vec<TensorSig> {
        names
            .iter()
            .zip(dtypes)
            .zip(shapes)
            .map(|((name, dtype), shape)| TensorSig {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
            })
            .collect()
    };
    let inputs = sigs(
        &lowered.input_names,
        &lowered.input_dtypes,
        &lowered.input_shapes,
    );
    let params = sigs(
        &lowered.param_names,
        &lowered.param_dtypes,
        &lowered.param_shapes,
    );
    let results = lowered
        .output_dtypes
        .iter()
        .zip(&lowered.output_shapes)
        .enumerate()
        .map(|(i, (dtype, shape))| TensorSig {
            name: format!("output{i}"),
            dtype: *dtype,
            shape: shape.clone(),
        })
        .collect();

    let names: Vec<String> = inputs
        .iter()
        .chain(&params)
        .map(|a| a.name.clone())
        .collect();
    let src = match &opts.weights {
        Weights::Arguments => print::ParamSource::Arguments,
        Weights::Parameters { scope } => print::ParamSource::Named { scope },
    };
    let printed = print::print_module(&proto, func_name, &names, params.len(), src)?;
    let (args, parameters) = match opts.weights {
        Weights::Arguments => (inputs.into_iter().chain(params).collect(), Vec::new()),
        Weights::Parameters { .. } => (inputs, params),
    };
    Ok((
        printed,
        FunctionSig {
            name: func_name.to_string(),
            args,
            parameters,
            results,
        },
    ))
}

/// One tensor for [`write_safetensors`]: name, dtype, shape, and raw
/// little-endian element bytes.
pub struct SafetensorsEntry<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: &'a [i64],
    pub data: &'a [u8],
}

/// Write `entries` as a `.safetensors` file — the parameter file a
/// [`Weights::Parameters`] export reads (IREE loads safetensors directly).
pub fn write_safetensors(
    path: &std::path::Path,
    entries: &[SafetensorsEntry<'_>],
) -> std::io::Result<()> {
    use std::io::Write as _;
    let dtype = |d: DType| -> std::io::Result<&'static str> {
        Ok(match d {
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U32 => "U32",
            DType::Bool => "BOOL",
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("safetensors has no dtype for {other:?}"),
                ));
            }
        })
    };
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (i, e) in entries.iter().enumerate() {
        let want = e.shape.iter().product::<i64>().max(0) as usize * e.dtype.size_bytes();
        if e.data.len() != want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "{}: {} bytes for shape {:?} {:?}",
                    e.name,
                    e.data.len(),
                    e.shape,
                    e.dtype
                ),
            ));
        }
        if i > 0 {
            header.push(',');
        }
        let name = e.name.replace('\\', "\\\\").replace('"', "\\\"");
        let shape: Vec<String> = e.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            dtype(e.dtype)?,
            shape.join(","),
            offset + e.data.len()
        ));
        offset += e.data.len();
    }
    header.push('}');
    // Pad so the data starts 8-byte aligned.
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for e in entries {
        f.write_all(e.data)?;
    }
    f.flush()
}
