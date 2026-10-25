// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diagnose a graph that fails on a device: what failed, where, and which
//! compile option gets past it.
//!
//! [`check`](crate::check) is static; this one runs. It compiles and runs the
//! graph on the device and on a reference (CPU), with the same deterministic
//! inputs and params, and reports:
//!
//! * a **diagnosis class** — `unsupported-op`, `compile-panic`, `run-panic`,
//!   `non-finite`, `mismatch`, … — with a hint for what to try;
//! * **probes** — the same comparison with one compile option changed at a
//!   time (fusion off, F32, common-IR kernels, …). A probe that passes names a
//!   workaround and, more usefully, the subsystem the defect is in;
//! * **bisect** — the graph re-run with every op exposed as an output, compared
//!   op by op, so the first diverging op is named. When nothing diverges once
//!   the ops are exposed, exposing them blocked the fusion that is wrong, and
//!   that is reported instead.
//!
//! `cargo rlx debug graph.json --device metal --probe --bisect` is the CLI.

use std::panic::{AssertUnwindSafe, catch_unwind};

use rlx_ir::{Graph, NodeId, Op};

use crate::{CompileOptions, Device, KernelDispatchConfig, KernelDispatchPolicy, Session};

/// What to run and how strictly to compare.
#[derive(Debug, Clone)]
pub struct DebugOptions {
    pub device: Device,
    pub reference: Device,
    /// Element passes when `|a-b| <= atol` or relative error `<= rtol`.
    pub atol: f32,
    pub rtol: f32,
    pub probe: bool,
    pub bisect: bool,
    pub seed: u64,
}

impl DebugOptions {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            reference: Device::Cpu,
            atol: 1e-4,
            rtol: 1e-3,
            probe: false,
            bisect: false,
            seed: 0x5eed,
        }
    }
}

/// How one run compared with the reference.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Pass {
        max_abs: f32,
    },
    /// The device rejected or could not compile the graph.
    CompileFailed {
        class: &'static str,
        message: String,
    },
    RunPanicked {
        message: String,
    },
    OutputCount {
        reference: usize,
        device: usize,
    },
    OutputLen {
        output: usize,
        reference: usize,
        device: usize,
    },
    /// Device produced NaN/Inf where the reference is finite.
    NonFinite {
        output: usize,
        index: usize,
        value: f32,
        reference: f32,
    },
    Mismatch {
        output: usize,
        index: usize,
        value: f32,
        reference: f32,
        max_abs: f32,
    },
}

impl Outcome {
    pub fn passed(&self) -> bool {
        matches!(self, Outcome::Pass { .. })
    }

    /// Short machine-readable class.
    pub fn class(&self) -> &'static str {
        match self {
            Outcome::Pass { .. } => "pass",
            Outcome::CompileFailed { class, .. } => class,
            Outcome::RunPanicked { .. } => "run-panic",
            Outcome::OutputCount { .. } | Outcome::OutputLen { .. } => "shape-mismatch",
            Outcome::NonFinite { .. } => "non-finite",
            Outcome::Mismatch { .. } => "mismatch",
        }
    }

    fn summary(&self) -> String {
        match self {
            Outcome::Pass { max_abs } => format!("pass (max |Δ| {max_abs:.3e})"),
            Outcome::CompileFailed { message, .. } | Outcome::RunPanicked { message } => {
                message.lines().next().unwrap_or("").to_string()
            }
            Outcome::OutputCount { reference, device } => {
                format!("{device} outputs, reference has {reference}")
            }
            Outcome::OutputLen {
                output,
                reference,
                device,
            } => format!("output {output}: {device} elements, reference has {reference}"),
            Outcome::NonFinite {
                output,
                index,
                value,
                reference,
            } => format!("output {output}[{index}] = {value} (reference {reference})"),
            Outcome::Mismatch {
                output,
                index,
                value,
                reference,
                max_abs,
            } => format!(
                "output {output}[{index}] = {value}, reference {reference} (max |Δ| {max_abs:.3e})"
            ),
        }
    }
}

/// One option change and how the graph fared under it.
#[derive(Debug, Clone)]
pub struct Probe {
    pub name: &'static str,
    pub outcome: Outcome,
}

/// The first op whose output disagrees with the reference.
#[derive(Debug, Clone)]
pub struct Divergence {
    pub node: NodeId,
    pub op: String,
    pub name: Option<String>,
    pub max_abs: f32,
    /// Every operand of the op agreed — the op itself computes it wrong.
    pub inputs_agree: bool,
    /// Every operand is a graph `Input`/`Param`/`Constant`, so the data reached
    /// the op through upload, not through another kernel.
    pub reads_boundary: bool,
}

#[derive(Debug, Clone)]
pub enum Bisect {
    NotRun,
    /// Exposed run could not complete either; the outcome says why.
    Failed(Outcome),
    FirstDivergence(Divergence),
    /// Every op agrees once exposed: the defect is in a fusion the exposure
    /// prevented.
    OnlyWhenFused,
}

#[derive(Debug, Clone)]
pub struct DebugReport {
    pub device: Device,
    pub outcome: Outcome,
    pub probes: Vec<Probe>,
    pub bisect: Bisect,
}

impl DebugReport {
    pub fn hint(&self) -> &'static str {
        match self.outcome.class() {
            "pass" => "the device matches the reference",
            "reference-failed" => {
                "the reference itself fails — the graph is the problem, not the device"
            }
            "dynamic-shape" => "bind symbolic dims (CompileOptions::dim_binding) before debugging",
            "device-unavailable" => {
                "build with the backend's feature (e.g. `--features metal`) on a host that has the device"
            }
            "unsupported-op" => {
                "the backend claims none of these kinds; add them to its SUPPORTED_OPS or rewrite upstream (`cargo rlx check` shows the dispatch report)"
            }
            "compile-panic" => {
                "a pass or the backend's lowering panicked; --probe shows whether a pipeline stage is involved"
            }
            "run-panic" => "the executable panicked at run time; --bisect names the op",
            "non-finite" => "NaN/Inf only on the device; RLX_DEBUG_NANS=1 localizes the producer",
            "shape-mismatch" => "the device returned a different output layout",
            "mismatch" => {
                "numbers differ; --bisect names the first diverging op, --probe the subsystem"
            }
            _ => "",
        }
    }
}

impl std::fmt::Display for DebugReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "diagnosis: {}", self.outcome.class())?;
        writeln!(f, "  device: {:?}", self.device)?;
        writeln!(f, "  detail: {}", self.outcome.summary())?;
        writeln!(f, "  hint:   {}", self.hint())?;
        if !self.probes.is_empty() {
            writeln!(f, "probes (one option changed at a time):")?;
            for p in &self.probes {
                let mark = if p.outcome.passed() { "PASS" } else { "fail" };
                writeln!(f, "  {mark}  {:<26} {}", p.name, p.outcome.summary())?;
            }
        }
        match &self.bisect {
            Bisect::NotRun => {}
            Bisect::Failed(o) => writeln!(f, "bisect: exposed run failed: {}", o.summary())?,
            Bisect::OnlyWhenFused => writeln!(
                f,
                "bisect: every op agrees once exposed — a fusion the exposure blocked is wrong"
            )?,
            Bisect::FirstDivergence(d) => writeln!(
                f,
                "bisect: first divergence at node {:?} {}{} (max |Δ| {:.3e}){}",
                d.node,
                d.op,
                d.name
                    .as_deref()
                    .map(|n| format!(" `{n}`"))
                    .unwrap_or_default(),
                d.max_abs,
                match (d.inputs_agree, d.reads_boundary) {
                    (_, true) => {
                        " — it reads only graph inputs/params: suspect the op or the \
                         upload of its operands (dtype, layout)"
                    }
                    (true, false) => " — its inputs agree, so this op is wrong",
                    (false, false) => "",
                }
            )?,
        }
        Ok(())
    }
}

/// Run the diagnosis.
pub fn debug_graph(graph: &Graph, opts: &DebugOptions) -> DebugReport {
    let feeds = match Feeds::for_graph(graph, opts.seed) {
        Ok(f) => f,
        Err(dims) => {
            return DebugReport {
                device: opts.device,
                outcome: Outcome::CompileFailed {
                    class: "dynamic-shape",
                    message: format!("symbolic dims on {dims}"),
                },
                probes: Vec::new(),
                bisect: Bisect::NotRun,
            };
        }
    };
    let base = CompileOptions::default();
    let reference = match execute(graph, opts.reference, &base, &feeds) {
        Ok(r) => r,
        Err(o) => {
            return DebugReport {
                device: opts.device,
                outcome: Outcome::CompileFailed {
                    class: "reference-failed",
                    message: o.summary(),
                },
                probes: Vec::new(),
                bisect: Bisect::NotRun,
            };
        }
    };
    let outcome = compare_run(graph, opts, &base, &feeds, &reference);

    let probes = if opts.probe && !outcome.passed() {
        probe_options(opts.device)
            .into_iter()
            .map(|(name, o)| Probe {
                name,
                outcome: compare_run(graph, opts, &o, &feeds, &reference),
            })
            .collect()
    } else {
        Vec::new()
    };

    let bisect = if opts.bisect && !outcome.passed() {
        bisect(graph, opts, &feeds)
    } else {
        Bisect::NotRun
    };

    DebugReport {
        device: opts.device,
        outcome,
        probes,
        bisect,
    }
}

/// Every option the probe tries, each alone.
fn probe_options(device: Device) -> Vec<(&'static str, CompileOptions)> {
    let base = CompileOptions::default;
    let fusion = |f: fn(&mut crate::FusionOptions)| {
        let mut fusion_opts = base().fusion_opts;
        f(&mut fusion_opts);
        CompileOptions {
            fusion_opts,
            ..base()
        }
    };
    let mut out = vec![
        ("fusion off", fusion(|f| f.skip_fusion = true)),
        (
            "unfuse elementwise regions",
            fusion(|f| {
                f.unfuse_elementwise_regions = true;
                f.keep_elementwise_regions = false;
            }),
        ),
        (
            "no weight-concat fusion",
            fusion(|f| f.no_weight_concat = true),
        ),
        (
            "no conv-bias-act fusion",
            CompileOptions {
                disable_conv_bias_act_fusion: true,
                ..base()
            },
        ),
        (
            "precision f32",
            CompileOptions {
                policy: Some(crate::PrecisionPolicy::AlwaysF32),
                ..base()
            },
        ),
        (
            "common-IR kernels",
            CompileOptions {
                kernel_dispatch: KernelDispatchConfig::new(KernelDispatchPolicy::ForceCommon),
                ..base()
            },
        ),
        (
            "no constant folding",
            CompileOptions {
                constant_folding: false,
                ..base()
            },
        ),
    ];
    if device == Device::Metal {
        out.push((
            "no MPSGraph",
            CompileOptions {
                disable_mpsgraph: true,
                ..base()
            },
        ));
    }
    out
}

fn compare_run(
    graph: &Graph,
    opts: &DebugOptions,
    options: &CompileOptions,
    feeds: &Feeds,
    reference: &[Vec<f32>],
) -> Outcome {
    match execute(graph, opts.device, options, feeds) {
        Ok(got) => compare(reference, &got, opts.atol, opts.rtol),
        Err(o) => o,
    }
}

fn compare(reference: &[Vec<f32>], got: &[Vec<f32>], atol: f32, rtol: f32) -> Outcome {
    if reference.len() != got.len() {
        return Outcome::OutputCount {
            reference: reference.len(),
            device: got.len(),
        };
    }
    let mut max_abs = 0f32;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    for (oi, (r, g)) in reference.iter().zip(got).enumerate() {
        if r.len() != g.len() {
            return Outcome::OutputLen {
                output: oi,
                reference: r.len(),
                device: g.len(),
            };
        }
        for (i, (&a, &b)) in r.iter().zip(g).enumerate() {
            if a.is_finite() && !b.is_finite() {
                return Outcome::NonFinite {
                    output: oi,
                    index: i,
                    value: b,
                    reference: a,
                };
            }
            if !a.is_finite() {
                continue;
            }
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            if d > atol && d > rtol * a.abs().max(b.abs()) && first_bad.is_none() {
                first_bad = Some((oi, i, b, a));
            }
        }
    }
    match first_bad {
        Some((output, index, value, reference)) => Outcome::Mismatch {
            output,
            index,
            value,
            reference,
            max_abs,
        },
        None => Outcome::Pass { max_abs },
    }
}

/// The first op, in node order, whose exposed output disagrees.
fn bisect(graph: &Graph, opts: &DebugOptions, feeds: &Feeds) -> Bisect {
    let (exposed, ids) = expose_compute_nodes(graph);
    let base = CompileOptions::default();
    let reference = match execute(&exposed, opts.reference, &base, feeds) {
        Ok(r) => r,
        Err(o) => return Bisect::Failed(o),
    };
    let got = match execute(&exposed, opts.device, &base, feeds) {
        Ok(g) => g,
        Err(o) => return Bisect::Failed(o),
    };
    if got.len() != reference.len() {
        return Bisect::Failed(Outcome::OutputCount {
            reference: reference.len(),
            device: got.len(),
        });
    }
    let n_out = graph.outputs.len();
    let mut bad = std::collections::HashMap::new();
    for (k, &id) in ids.iter().enumerate() {
        let (r, g) = (&reference[n_out + k], &got[n_out + k]);
        if let Outcome::Mismatch { max_abs, .. } | Outcome::NonFinite { value: max_abs, .. } =
            compare(
                std::slice::from_ref(r),
                std::slice::from_ref(g),
                opts.atol,
                opts.rtol,
            )
        {
            bad.insert(id, max_abs);
        }
    }
    for node in graph.nodes() {
        if let Some(&max_abs) = bad.get(&node.id) {
            let leaf = |id: &NodeId| {
                matches!(
                    graph.node(*id).op,
                    Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
                )
            };
            return Bisect::FirstDivergence(Divergence {
                node: node.id,
                op: format!("{:?}", node.op.kind()),
                name: node.name.clone(),
                max_abs,
                inputs_agree: node.inputs.iter().all(|i| !bad.contains_key(i)),
                reads_boundary: node.inputs.iter().all(leaf),
            });
        }
    }
    Bisect::OnlyWhenFused
}

/// The graph with its outputs followed by every compute node.
fn expose_compute_nodes(graph: &Graph) -> (Graph, Vec<NodeId>) {
    let mut g = graph.clone();
    let ids: Vec<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            )
        })
        .filter(|n| n.shape.dtype().is_float())
        .map(|n| n.id)
        .collect();
    let mut outs = graph.outputs.clone();
    outs.extend(&ids);
    g.set_outputs(outs);
    (g, ids)
}

/// Deterministic inputs and params for every `Input` / `Param`.
struct Feeds {
    inputs: Vec<(String, Vec<f32>)>,
    params: Vec<(String, Vec<f32>)>,
}

impl Feeds {
    fn for_graph(graph: &Graph, seed: u64) -> Result<Self, String> {
        let mut state = seed | 1;
        let mut next = move || {
            // xorshift64*: deterministic across platforms.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let v = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            ((v >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        for n in graph.nodes() {
            let (name, is_param) = match &n.op {
                Op::Input { name } => (name, false),
                Op::Param { name } => (name, true),
                _ => continue,
            };
            let len = n
                .shape
                .num_elements()
                .ok_or_else(|| format!("{} `{name}`", if is_param { "param" } else { "input" }))?;
            let float = n.shape.dtype().is_float();
            // Integer feeds are indices more often than not; zero is in range
            // for every gather/embedding.
            let data: Vec<f32> = (0..len)
                .map(|_| {
                    if float {
                        next() * if is_param { 0.1 } else { 1.0 }
                    } else {
                        0.0
                    }
                })
                .collect();
            if is_param {
                params.push((name.clone(), data));
            } else {
                inputs.push((name.clone(), data));
            }
        }
        Ok(Self { inputs, params })
    }
}

/// Compile and run with panics caught and classified.
fn execute(
    graph: &Graph,
    device: Device,
    options: &CompileOptions,
    feeds: &Feeds,
) -> Result<Vec<Vec<f32>>, Outcome> {
    if !crate::is_available(device) {
        return Err(Outcome::CompileFailed {
            class: "device-unavailable",
            message: format!("{device:?} is not available in this build or on this host"),
        });
    }
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let compiled = catch_unwind(AssertUnwindSafe(|| {
        Session::new(device).compile_with(graph.clone(), options)
    }));
    let result = match compiled {
        Err(p) => {
            let message = panic_message(p.as_ref());
            let class = if message.contains("doesn't claim support") {
                "unsupported-op"
            } else {
                "compile-panic"
            };
            Err(Outcome::CompileFailed { class, message })
        }
        Ok(mut c) => catch_unwind(AssertUnwindSafe(|| {
            for (n, d) in &feeds.params {
                c.set_param(n, d);
            }
            let ins: Vec<(&str, &[f32])> = feeds
                .inputs
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            c.run(&ins)
        }))
        .map_err(|p| Outcome::RunPanicked {
            message: panic_message(p.as_ref()),
        }),
    };
    std::panic::set_hook(prev);
    result
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "<non-string panic payload>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Shape};

    fn small() -> Graph {
        let mut g = Graph::new("small");
        let s = Shape::new(&[2, 8], DType::F32);
        let x = g.input("x", s.clone());
        let w = g.param("w", Shape::new(&[8, 8], DType::F32));
        let m = g.matmul(x, w, s.clone());
        let r = g.add_node(Op::Activation(Activation::Relu), vec![m], s.clone());
        let y = g.binary(BinaryOp::Add, r, x, s);
        g.set_outputs(vec![y]);
        g
    }

    #[test]
    fn cpu_against_cpu_passes() {
        let r = debug_graph(&small(), &DebugOptions::new(Device::Cpu));
        assert!(r.outcome.passed(), "{r}");
    }

    #[test]
    fn compare_classifies() {
        let r = vec![vec![1.0, 2.0]];
        assert!(compare(&r, &[vec![1.0, 2.0]], 1e-4, 1e-3).passed());
        assert_eq!(
            compare(&r, &[vec![1.0, 2.5]], 1e-4, 1e-3).class(),
            "mismatch"
        );
        assert_eq!(
            compare(&r, &[vec![1.0, f32::NAN]], 1e-4, 1e-3).class(),
            "non-finite"
        );
        assert_eq!(
            compare(&r, &[vec![1.0]], 1e-4, 1e-3).class(),
            "shape-mismatch"
        );
        assert_eq!(compare(&r, &[], 1e-4, 1e-3).class(), "shape-mismatch");
    }

    #[test]
    fn exposure_keeps_outputs_first() {
        let g = small();
        let (e, ids) = expose_compute_nodes(&g);
        assert_eq!(e.outputs[0], g.outputs[0]);
        assert_eq!(e.outputs.len(), 1 + ids.len());
        assert_eq!(ids.len(), 3, "matmul, relu, add");
    }

    #[test]
    fn feeds_are_deterministic() {
        let a = Feeds::for_graph(&small(), 7).unwrap();
        let b = Feeds::for_graph(&small(), 7).unwrap();
        assert_eq!(a.inputs, b.inputs);
        assert_eq!(a.params, b.params);
        assert!(a.inputs[0].1.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
