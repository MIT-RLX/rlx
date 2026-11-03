// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident training: the optimizer update fused into the graph.
//!
//! A host-side optimizer downloads the gradients, updates the parameters on the
//! CPU, and re-uploads them — every step. Measured on an MNIST MLP
//! (`784→128→10`, batch 64, 102 k parameters), that host half is **40–55% of the
//! step and does not vary with the device**:
//!
//! | device | forward+backward+upload | host optimizer | total |
//! |---|---|---|---|
//! | cpu | 207 µs | 255 µs | 462 µs |
//! | metal | 402 µs | 275 µs | 677 µs |
//! | wgpu | 4825 µs | 302 µs | 5127 µs |
//!
//! Buying a faster device does nothing for that column. So this module appends
//! the update to the backward graph as ordinary ops — `m' = β₁m + (1-β₁)g`,
//! `v' = β₂v + (1-β₂)g²`, `p' = p - lr·m̂/(√v̂+ε)` — and keeps the parameters and
//! moments in device buffers across steps via
//! [`bind_gpu_handle`](crate::CompiledGraph::bind_gpu_handle) +
//! [`set_gpu_handle_feed`](crate::CompiledGraph::set_gpu_handle_feed). Forward,
//! backward and update become one on-device computation, and only the scalar
//! loss is read back.
//!
//! Backends without handle support — CPU included — take a host-chained path
//! that feeds the parameters and moments as ordinary inputs and reads the
//! updated values back. The arithmetic is the same graph either way, so results
//! match; only the transfers differ. [`ResidentTrainer::is_resident`] reports
//! which path is live, because "resident" that silently fell back is a
//! performance claim nobody can check.
//!
//! Lifted from `rlx-models`' `rlx-tune`, which had it but could not share it:
//! nothing here is model-specific, and a binding should not have to reach into
//! a downstream repo for the fast path.

use std::collections::HashMap;

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::{CompileOptions, CompiledGraph, Device, Session};

/// Which trainable parameter, by name and node.
///
/// The node id is what autodiff differentiates against; the name is what binds
/// the value. Both are needed because the backward graph renumbers nothing but
/// re-tags the parameter as an input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainableParam {
    pub name: String,
    pub node: NodeId,
}

/// Adam / AdamW hyperparameters.
///
/// `weight_decay > 0` gives AdamW's decoupled decay (`p -= lr·wd·p`), not
/// L2-in-the-gradient; the two differ once the moments are involved, and
/// conflating them is a silent change of algorithm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamSpec {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamSpec {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamSpec {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            ..Self::default()
        }
    }
}

/// Why a fused step could not be built.
#[derive(Debug)]
pub enum FuseError {
    /// The forward graph has no outputs, so there is no loss to differentiate.
    NoLoss,
    /// `wrt` was empty — nothing to train.
    NoTrainableParams,
    /// A named parameter is not a `Param` in the forward graph.
    UnknownParam(String),
    /// No initial value was supplied for a trainable parameter.
    MissingInitialValue(String),
}

impl std::fmt::Display for FuseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoLoss => write!(
                f,
                "the forward graph has no outputs; the first must be the loss"
            ),
            Self::NoTrainableParams => write!(f, "no trainable parameters were given"),
            Self::UnknownParam(name) => write!(f, "'{name}' is not a Param in the forward graph"),
            Self::MissingInitialValue(name) => {
                write!(f, "no initial value for trainable parameter '{name}'")
            }
        }
    }
}

impl std::error::Error for FuseError {}

/// Where one fused parameter's updated `p' / m' / v'` land in the outputs.
#[derive(Clone, Debug)]
struct FusedParam {
    name: String,
    m_name: String,
    v_name: String,
    scale_name: String,
    p_out: usize,
    m_out: usize,
    v_out: usize,
}

/// The fused forward+backward+update graph and its output layout.
struct FusedStep {
    graph: Graph,
    params: Vec<FusedParam>,
}

/// Muon hyperparameters for the fused step: rlx-optim's `Muon` on the device,
/// and with `adamw` set its `MuonAdamW` (AdamW for every param that is not a
/// matrix with both sides above 1, Muon's rate `muon_scale` × the AdamW one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonSpec {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub weight_decay: f32,
    pub ns_steps: u32,
    /// `(a, b, c)` of `X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X`.
    pub ns_coeffs: (f32, f32, f32),
    pub adamw: Option<AdamSpec>,
    pub muon_scale: f32,
}

impl MuonSpec {
    /// rlx-optim's `Muon::new` defaults.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.95,
            nesterov: true,
            weight_decay: 0.0,
            ns_steps: 5,
            ns_coeffs: (3.4445, -4.7750, 2.0315),
            adamw: None,
            muon_scale: 1.0,
        }
    }
}

/// The update the fused step runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FusedRule {
    Adam(AdamSpec),
    Muon(MuonSpec),
}

impl FusedRule {
    /// The rate fed as `__lr`.
    fn lr(&self) -> f32 {
        match self {
            Self::Adam(a) => a.lr,
            Self::Muon(m) => m.adamw.map_or(m.lr, |a| a.lr),
        }
    }

    /// The Adam part, whose bias corrections the step feeds.
    fn adam(&self) -> Option<AdamSpec> {
        match self {
            Self::Adam(a) => Some(*a),
            Self::Muon(m) => m.adamw,
        }
    }
}

/// `p'`, `m'`, `v'` of one param's AdamW update (gated: `gate` 0 holds all three).
#[allow(clippy::too_many_arguments)]
fn adam_update(
    g: &mut Graph,
    spec: &AdamSpec,
    p: NodeId,
    grad: NodeId,
    m: NodeId,
    v: NodeId,
    gate: NodeId,
    lr: NodeId,
    bc1: NodeId,
    bc2: NodeId,
) -> (NodeId, NodeId, NodeId) {
    let f = DType::F32;
    let c_b1 = g.constant(spec.beta1 as f64, f);
    let c_1mb1 = g.constant((1.0 - spec.beta1) as f64, f);
    let c_b2 = g.constant(spec.beta2 as f64, f);
    let c_1mb2 = g.constant((1.0 - spec.beta2) as f64, f);
    let c_eps = g.constant(spec.eps as f64, f);
    // Each builder call takes `&mut g`, so every intermediate needs its own
    // binding rather than nesting.
    // m' = β₁·m + (1-β₁)·g
    let m_decayed = g.mul(m, c_b1);
    let g_scaled = g.mul(grad, c_1mb1);
    let m_new = g.add(m_decayed, g_scaled);
    // v' = β₂·v + (1-β₂)·g²
    let sq = g.mul(grad, grad);
    let v_decayed = g.mul(v, c_b2);
    let sq_scaled = g.mul(sq, c_1mb2);
    let v_new = g.add(v_decayed, sq_scaled);
    // p' = p - gate·lr·(m'/bc1)/(√(v'/bc2)+ε)
    let mhat = g.div(m_new, bc1);
    let vhat = g.div(v_new, bc2);
    let root = g.sqrt(vhat);
    let denom = g.add(root, c_eps);
    let ratio = g.div(mhat, denom);
    let update = g.mul(ratio, lr);
    let gated = g.mul(update, gate);
    let stepped = g.sub(p, gated);
    let p_new = if spec.weight_decay != 0.0 {
        let c_wd = g.constant(spec.weight_decay as f64, f);
        let lrwd = g.mul(lr, c_wd);
        let decay = g.mul(p, lrwd);
        let gated_decay = g.mul(decay, gate);
        g.sub(stepped, gated_decay)
    } else {
        stepped
    };
    let m_kept = hold(g, gate, m_new, m);
    let v_kept = hold(g, gate, v_new, v);
    (p_new, m_kept, v_kept)
}

/// `gate·new + (1−gate)·old`: freezing holds the state too, so unfreezing
/// resumes rather than lurching on state built from updates never applied.
fn hold(g: &mut Graph, gate: NodeId, new: NodeId, old: NodeId) -> NodeId {
    let one = g.constant(1.0, DType::F32);
    let keep = g.sub(one, gate);
    let taken = g.mul(new, gate);
    let held = g.mul(old, keep);
    g.add(taken, held)
}

/// `p'`, `m'` of one param's Muon update: momentum, then for a `[rows, cols]`
/// matrix the Newton–Schulz orthogonalization scaled by `√max(rows, cols)`;
/// for anything else momentum SGD. `lr_factor` scales `lr` (MuonAdamW's
/// `muon_scale`); `wd` is the decoupled decay on that rate.
#[allow(clippy::too_many_arguments)]
fn muon_update(
    g: &mut Graph,
    spec: &MuonSpec,
    matrix: Option<(usize, usize)>,
    p: NodeId,
    grad: NodeId,
    m: NodeId,
    gate: NodeId,
    lr: NodeId,
    lr_factor: f32,
    wd: f32,
) -> (NodeId, NodeId) {
    let f = DType::F32;
    let c_mu = g.constant(spec.momentum as f64, f);
    // m' = μ·m + g
    let m_decayed = g.mul(m, c_mu);
    let m_new = g.add(m_decayed, grad);
    // Nesterov: g + μ·m'; else m'.
    let dir = if spec.nesterov {
        let ahead = g.mul(m_new, c_mu);
        g.add(grad, ahead)
    } else {
        m_new
    };
    let step = match matrix {
        Some((rows, cols)) => {
            // X = G / ‖G‖_F, the smaller side as rows (the Gram is min × min).
            let sq = g.mul(dir, dir);
            let total = g.sum(sq, vec![0, 1], false);
            let norm = g.sqrt(total);
            let c_tiny = g.constant(1e-12, f);
            let fro = g.add(norm, c_tiny);
            let mut x = g.div(dir, fro);
            let transposed = rows > cols;
            if transposed {
                x = g.transpose_(x, vec![1, 0]);
            }
            let (a, b, c) = spec.ns_coeffs;
            let (ca, cb, cc) = (
                g.constant(a as f64, f),
                g.constant(b as f64, f),
                g.constant(c as f64, f),
            );
            for _ in 0..spec.ns_steps {
                let xt = g.transpose_(x, vec![1, 0]);
                let gram = g.mm(x, xt);
                let gram2 = g.mm(gram, gram);
                let bg = g.mul(gram, cb);
                let cg = g.mul(gram2, cc);
                let poly = g.add(bg, cg);
                let px = g.mm(poly, x);
                let ax = g.mul(x, ca);
                x = g.add(ax, px);
            }
            if transposed {
                x = g.transpose_(x, vec![1, 0]);
            }
            let c_s = g.constant((rows.max(cols) as f64).sqrt(), f);
            g.mul(x, c_s)
        }
        None => dir,
    };
    // p' = p − gate·lr·factor·(step + wd·p)
    let with_decay = if wd != 0.0 {
        let c_wd = g.constant(wd as f64, f);
        let decay = g.mul(p, c_wd);
        g.add(step, decay)
    } else {
        step
    };
    let rate = if lr_factor != 1.0 {
        let c_f = g.constant(lr_factor as f64, f);
        g.mul(lr, c_f)
    } else {
        lr
    };
    let update = g.mul(with_decay, rate);
    let gated = g.mul(update, gate);
    let p_new = g.sub(p, gated);
    (p_new, hold(g, gate, m_new, m))
}

/// Append the update of `rule` to the backward of `forward` w.r.t. `wrt`.
///
/// Outputs become `[loss, aux…, p'₀, m'₀, v'₀, p'₁, …]`. Per-parameter inputs
/// `{name}__m` / `{name}__v` carry the state (Muon's momentum in `m`; `v`
/// passes through for its matrices) and `{name}__scale` gates the update (1 to
/// train, 0 to freeze); shared scalars `__lr` and, when an Adam part runs,
/// `__bc1`, `__bc2` carry the learning rate and the bias corrections, so a
/// schedule needs no recompile.
fn fused_graph(
    forward: &Graph,
    wrt: &[TrainableParam],
    rule: &FusedRule,
) -> Result<FusedStep, FuseError> {
    if forward.outputs.is_empty() {
        return Err(FuseError::NoLoss);
    }
    if wrt.is_empty() {
        return Err(FuseError::NoTrainableParams);
    }
    let wrt_ids: Vec<NodeId> = wrt.iter().map(|s| s.node).collect();
    let mut g = rlx_opt::autodiff::grad_with_loss(forward, &wrt_ids);
    let f = DType::F32;

    // `grad_with_loss` emits [loss, aux…, grads…] — aux being the forward's
    // outputs[1..], mirrored through. The gradients are the last `wrt.len()`.
    let grad_start = g
        .outputs
        .len()
        .checked_sub(wrt.len())
        .ok_or(FuseError::NoTrainableParams)?;
    let mut outputs: Vec<NodeId> = g.outputs[..grad_start].to_vec();
    let grads: Vec<NodeId> = g.outputs[grad_start..].to_vec();

    let lr = g.input("__lr", Shape::scalar(f));
    let bc = rule.adam().map(|_| {
        (
            g.input("__bc1", Shape::scalar(f)),
            g.input("__bc2", Shape::scalar(f)),
        )
    });

    let mut params = Vec::with_capacity(wrt.len());
    let mut param_ids = Vec::with_capacity(wrt.len());
    for (i, slot) in wrt.iter().enumerate() {
        let p = g
            .param_id(&slot.name)
            .ok_or_else(|| FuseError::UnknownParam(slot.name.clone()))?;
        param_ids.push(p);
        let shape = g.shape(p).clone();
        let grad = grads[i];
        let m_name = format!("{}__m", slot.name);
        let v_name = format!("{}__v", slot.name);
        let scale_name = format!("{}__scale", slot.name);
        let m = g.input(&m_name, shape.clone());
        let v = g.input(&v_name, shape.clone());
        // Scalar gate: 1 trains, 0 freezes.
        let gate = g.input(&scale_name, Shape::scalar(f));
        let dims: Vec<usize> = shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => *n,
                rlx_ir::Dim::Dynamic(_) => 0,
            })
            .collect();
        let (p_new, m_kept, v_kept) = match rule {
            FusedRule::Adam(spec) => {
                let (bc1, bc2) = bc.expect("Adam feeds its corrections");
                adam_update(&mut g, spec, p, grad, m, v, gate, lr, bc1, bc2)
            }
            FusedRule::Muon(spec) => {
                let is_matrix = dims.len() == 2 && dims[0] > 1 && dims[1] > 1;
                match (spec.adamw, is_matrix) {
                    // MuonAdamW: AdamW for what is not a matrix.
                    (Some(adam), false) => {
                        let (bc1, bc2) = bc.expect("MuonAdamW feeds its corrections");
                        adam_update(&mut g, &adam, p, grad, m, v, gate, lr, bc1, bc2)
                    }
                    (adamw, _) => {
                        // Muon's matrices are rank 2 (rlx-optim's `Muon`
                        // orthogonalizes any; MuonAdamW only both sides > 1).
                        let matrix = (dims.len() == 2 && (adamw.is_none() || is_matrix))
                            .then(|| (dims[0], dims[1]));
                        let (factor, wd) = match adamw {
                            Some(a) => (spec.muon_scale, a.weight_decay / spec.muon_scale),
                            None => (1.0, spec.weight_decay),
                        };
                        let (p_new, m_kept) =
                            muon_update(&mut g, spec, matrix, p, grad, m, gate, lr, factor, wd);
                        (p_new, m_kept, v)
                    }
                }
            }
        };
        let p_out = outputs.len();
        outputs.push(p_new);
        let m_out = outputs.len();
        outputs.push(m_kept);
        let v_out = outputs.len();
        outputs.push(v_kept);
        params.push(FusedParam {
            name: slot.name.clone(),
            m_name,
            v_name,
            scale_name,
            p_out,
            m_out,
            v_out,
        });
    }
    g.set_outputs(outputs);

    // Autodiff is done, so re-tag the trainable params as **inputs**: only graph
    // inputs can be bound to device-resident handles. The node id is unchanged,
    // so every reference to it still resolves.
    for (slot, &p) in wrt.iter().zip(&param_ids) {
        g.node_mut(p).op = Op::Input {
            name: slot.name.clone(),
        };
    }
    Ok(FusedStep { graph: g, params })
}

/// A trainer whose optimizer runs **inside** the compiled graph.
pub struct ResidentTrainer {
    compiled: CompiledGraph,
    params: Vec<FusedParam>,
    /// The Adam part's betas for the bias corrections, and `lr` the rate fed
    /// as `__lr` (for Muon alone, only `lr` is read).
    spec: AdamSpec,
    rule: FusedRule,
    step_count: i32,
    /// How many outputs precede the first `p'` — 1 for the loss plus any aux
    /// outputs the forward graph declared.
    leading_outputs: usize,
    /// Host mirrors: the source of truth on the fallback path, a lazily
    /// refreshed cache on the resident one.
    param_vals: HashMap<String, Vec<f32>>,
    m_vals: HashMap<String, Vec<f32>>,
    v_vals: HashMap<String, Vec<f32>>,
    /// Per-parameter update gate, 1.0 or 0.0.
    gates: HashMap<String, f32>,
    resident: bool,
}

impl ResidentTrainer {
    /// Compile the fused step for `device`, seeding parameters from `initial`
    /// (moments start at zero).
    ///
    /// Parameters present in `initial` but absent from `wrt` are uploaded once
    /// and never updated — the cheap kind of frozen, and how a LoRA base weight
    /// is held.
    pub fn new(
        forward: &Graph,
        wrt: &[TrainableParam],
        initial: &HashMap<String, Vec<f32>>,
        spec: &AdamSpec,
        device: Device,
    ) -> Result<Self, FuseError> {
        Self::with_options(forward, wrt, initial, spec, device, &CompileOptions::new())
    }

    /// [`Self::new`] with explicit compile options (precision policy, fusion).
    pub fn with_options(
        forward: &Graph,
        wrt: &[TrainableParam],
        initial: &HashMap<String, Vec<f32>>,
        spec: &AdamSpec,
        device: Device,
        options: &CompileOptions,
    ) -> Result<Self, FuseError> {
        Self::with_rule(
            forward,
            wrt,
            initial,
            &FusedRule::Adam(*spec),
            device,
            options,
        )
    }

    /// [`Self::with_options`] for any fused rule: AdamW, or Muon / MuonAdamW
    /// with the Newton–Schulz orthogonalization in the graph.
    pub fn with_rule(
        forward: &Graph,
        wrt: &[TrainableParam],
        initial: &HashMap<String, Vec<f32>>,
        rule: &FusedRule,
        device: Device,
        options: &CompileOptions,
    ) -> Result<Self, FuseError> {
        let leading_outputs = forward.outputs.len();
        let step = fused_graph(forward, wrt, rule)?;
        let spec = AdamSpec {
            lr: rule.lr(),
            ..rule.adam().unwrap_or_default()
        };
        let mut compiled = Session::new(device).compile_with(step.graph, options);

        let trainable: Vec<&str> = step.params.iter().map(|p| p.name.as_str()).collect();
        for (name, data) in initial {
            if !trainable.contains(&name.as_str()) {
                compiled.set_param(name, data);
            }
        }

        let mut param_vals = HashMap::new();
        let mut m_vals = HashMap::new();
        let mut v_vals = HashMap::new();
        let mut gates = HashMap::new();
        for fp in &step.params {
            let start = initial
                .get(&fp.name)
                .ok_or_else(|| FuseError::MissingInitialValue(fp.name.clone()))?
                .clone();
            let zeros = vec![0.0f32; start.len()];
            param_vals.insert(fp.name.clone(), start);
            m_vals.insert(fp.name.clone(), zeros.clone());
            v_vals.insert(fp.name.clone(), zeros);
            gates.insert(fp.name.clone(), 1.0);
        }

        // All-or-nothing: a half-resident step would read some parameters from
        // the device and some from the host, which is a correctness hazard, not
        // a partial optimization.
        let resident =
            try_bind_resident(&mut compiled, &step.params, &param_vals, &m_vals, &v_vals);

        Ok(Self {
            compiled,
            params: step.params,
            spec,
            rule: *rule,
            step_count: 0,
            leading_outputs,
            param_vals,
            m_vals,
            v_vals,
            gates,
            resident,
        })
    }

    /// Whether parameters and moments live in device buffers.
    ///
    /// False means the host-chained fallback is running: same arithmetic, same
    /// results, but a round trip per step.
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    pub fn device(&self) -> Device {
        self.compiled.device()
    }

    pub fn steps(&self) -> u64 {
        self.step_count as u64
    }

    /// The learning rate is a scalar graph input, so a schedule costs no
    /// recompile and nothing on the resident path.
    pub fn set_lr(&mut self, lr: f32) {
        self.spec.lr = lr;
    }

    pub fn lr(&self) -> f32 {
        self.spec.lr
    }

    /// Hold or release a parameter. Returns false if the name is not trainable.
    pub fn set_frozen(&mut self, name: &str, frozen: bool) -> bool {
        match self.gates.get_mut(name) {
            Some(gate) => {
                *gate = if frozen { 0.0 } else { 1.0 };
                true
            }
            None => false,
        }
    }

    pub fn is_frozen(&self, name: &str) -> bool {
        self.gates.get(name).is_some_and(|g| *g == 0.0)
    }

    pub fn trainable(&self) -> impl Iterator<Item = &str> {
        self.params.iter().map(|p| p.name.as_str())
    }

    /// One step on `inputs`, returning `[loss, aux…]`.
    ///
    /// On the resident path only these are read back; `p' / m' / v'` never leave
    /// the device.
    pub fn step(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.step_count += 1;
        let seed = [1.0f32];
        let lr = [self.spec.lr];
        let bc1 = [1.0 - self.spec.beta1.powi(self.step_count)];
        let bc2 = [1.0 - self.spec.beta2.powi(self.step_count)];

        let gate_values: Vec<(String, [f32; 1])> = self
            .params
            .iter()
            .map(|fp| (fp.scale_name.clone(), [self.gates[&fp.name]]))
            .collect();

        let mut run_inputs: Vec<(&str, &[f32])> = inputs.to_vec();
        run_inputs.push(("d_output", &seed));
        run_inputs.push(("__lr", &lr));
        if self.rule.adam().is_some() {
            run_inputs.push(("__bc1", &bc1));
            run_inputs.push(("__bc2", &bc2));
        }
        for (name, value) in &gate_values {
            run_inputs.push((name.as_str(), value));
        }

        let wanted: Vec<usize> = (0..self.leading_outputs).collect();
        if self.resident {
            self.compiled.run_read_outputs(&run_inputs, Some(&wanted))
        } else {
            for fp in &self.params {
                run_inputs.push((fp.name.as_str(), &self.param_vals[&fp.name]));
                run_inputs.push((fp.m_name.as_str(), &self.m_vals[&fp.name]));
                run_inputs.push((fp.v_name.as_str(), &self.v_vals[&fp.name]));
            }
            let outs = self.compiled.run(&run_inputs);
            for fp in &self.params {
                copy_out(&mut self.param_vals, &fp.name, outs.get(fp.p_out));
                copy_out(&mut self.m_vals, &fp.name, outs.get(fp.m_out));
                copy_out(&mut self.v_vals, &fp.name, outs.get(fp.v_out));
            }
            outs.into_iter().take(self.leading_outputs).collect()
        }
    }

    /// Run the graph without taking a step: `[loss, aux…]` only.
    ///
    /// Every gate is held at zero and the step counter is left alone, so a
    /// validation pass neither moves the weights nor advances Adam's bias
    /// correction. The update ops still execute — their results are simply
    /// written back unchanged — which costs a little arithmetic and saves
    /// compiling a second graph.
    pub fn forward_only(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let seed = [1.0f32];
        let lr = [self.spec.lr];
        // The counter is not advanced, so reuse the *current* corrections.
        let t = self.step_count.max(1);
        let bc1 = [1.0 - self.spec.beta1.powi(t)];
        let bc2 = [1.0 - self.spec.beta2.powi(t)];
        let held: Vec<(String, [f32; 1])> = self
            .params
            .iter()
            .map(|fp| (fp.scale_name.clone(), [0.0f32]))
            .collect();

        let mut run_inputs: Vec<(&str, &[f32])> = inputs.to_vec();
        run_inputs.push(("d_output", &seed));
        run_inputs.push(("__lr", &lr));
        if self.rule.adam().is_some() {
            run_inputs.push(("__bc1", &bc1));
            run_inputs.push(("__bc2", &bc2));
        }
        for (name, value) in &held {
            run_inputs.push((name.as_str(), value));
        }
        let wanted: Vec<usize> = (0..self.leading_outputs).collect();
        if self.resident {
            self.compiled.run_read_outputs(&run_inputs, Some(&wanted))
        } else {
            for fp in &self.params {
                run_inputs.push((fp.name.as_str(), &self.param_vals[&fp.name]));
                run_inputs.push((fp.m_name.as_str(), &self.m_vals[&fp.name]));
                run_inputs.push((fp.v_name.as_str(), &self.v_vals[&fp.name]));
            }
            let outs = self.compiled.run(&run_inputs);
            outs.into_iter().take(self.leading_outputs).collect()
        }
    }

    /// Current trainable weights, reading back from the device when resident.
    pub fn params(&mut self) -> HashMap<String, Vec<f32>> {
        if self.resident {
            for fp in &self.params {
                if let Some(v) = self.compiled.read_gpu_handle(&fp.name) {
                    self.param_vals.insert(fp.name.clone(), v);
                }
            }
        }
        self.param_vals.clone()
    }

    /// Overwrite the trainable weights — resuming from a checkpoint.
    ///
    /// Rebinds the device buffers on the resident path, so the next step sees
    /// the restored values rather than the ones already on the device.
    pub fn set_params(&mut self, values: &HashMap<String, Vec<f32>>) {
        for (name, data) in values {
            if let Some(slot) = self.param_vals.get_mut(name) {
                let n = slot.len().min(data.len());
                slot[..n].copy_from_slice(&data[..n]);
            } else {
                self.compiled.set_param(name, data);
            }
        }
        if self.resident {
            for fp in &self.params {
                self.compiled
                    .bind_gpu_handle(&fp.name, &self.param_vals[&fp.name]);
                self.compiled.set_gpu_handle_feed(&fp.name, fp.p_out);
            }
        }
    }

    /// Optimizer moments, for checkpointing. `(name, m, v)` per parameter.
    pub fn moments(&mut self) -> Vec<(String, Vec<f32>, Vec<f32>)> {
        if self.resident {
            for fp in &self.params {
                if let Some(m) = self.compiled.read_gpu_handle(&fp.m_name) {
                    self.m_vals.insert(fp.name.clone(), m);
                }
                if let Some(v) = self.compiled.read_gpu_handle(&fp.v_name) {
                    self.v_vals.insert(fp.name.clone(), v);
                }
            }
        }
        self.params
            .iter()
            .map(|fp| {
                (
                    fp.name.clone(),
                    self.m_vals[&fp.name].clone(),
                    self.v_vals[&fp.name].clone(),
                )
            })
            .collect()
    }

    /// Restore moments and the step counter from a checkpoint.
    ///
    /// The counter matters: Adam's bias correction is a function of it, so
    /// resuming at step 0 with warm moments takes a step the uninterrupted run
    /// never would.
    pub fn restore_moments(
        &mut self,
        moments: &[(String, Vec<f32>, Vec<f32>)],
        step_count: u64,
    ) -> bool {
        let mut all = true;
        for (name, m, v) in moments {
            if let Some(slot) = self.m_vals.get_mut(name) {
                let n = slot.len().min(m.len());
                slot[..n].copy_from_slice(&m[..n]);
            } else {
                all = false;
            }
            if let Some(slot) = self.v_vals.get_mut(name) {
                let n = slot.len().min(v.len());
                slot[..n].copy_from_slice(&v[..n]);
            } else {
                all = false;
            }
        }
        self.step_count = step_count as i32;
        if self.resident {
            for fp in &self.params {
                self.compiled
                    .bind_gpu_handle(&fp.m_name, &self.m_vals[&fp.name]);
                self.compiled
                    .bind_gpu_handle(&fp.v_name, &self.v_vals[&fp.name]);
                self.compiled.set_gpu_handle_feed(&fp.m_name, fp.m_out);
                self.compiled.set_gpu_handle_feed(&fp.v_name, fp.v_out);
            }
        }
        all
    }
}

fn copy_out(map: &mut HashMap<String, Vec<f32>>, name: &str, out: Option<&Vec<f32>>) {
    if let (Some(dst), Some(src)) = (map.get_mut(name), out) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
    }
}

/// Bind parameters and moments as device buffers and feed the updated outputs
/// back into them. True only if every bind and feed succeeded.
fn try_bind_resident(
    compiled: &mut CompiledGraph,
    params: &[FusedParam],
    param_vals: &HashMap<String, Vec<f32>>,
    m_vals: &HashMap<String, Vec<f32>>,
    v_vals: &HashMap<String, Vec<f32>>,
) -> bool {
    let mut ok = true;
    for fp in params {
        ok &= compiled.bind_gpu_handle(&fp.name, &param_vals[&fp.name]);
        ok &= compiled.bind_gpu_handle(&fp.m_name, &m_vals[&fp.name]);
        ok &= compiled.bind_gpu_handle(&fp.v_name, &v_vals[&fp.name]);
        ok &= compiled.set_gpu_handle_feed(&fp.name, fp.p_out);
        ok &= compiled.set_gpu_handle_feed(&fp.m_name, fp.m_out);
        ok &= compiled.set_gpu_handle_feed(&fp.v_name, fp.v_out);
    }
    ok
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use rlx_ir::infer::GraphExt;

    /// `loss = mean((x·w + b - t)^2)` — small, but it exercises a matmul, a
    /// broadcast add and a reduction, so the backward has real structure.
    pub fn regression(batch: usize, dim: usize) -> (Graph, Vec<TrainableParam>) {
        let mut g = Graph::new("mse");
        let x = g.input("x", Shape::new(&[batch, dim], DType::F32));
        let t = g.input("t", Shape::new(&[batch, dim], DType::F32));
        let w = g.param("w", Shape::new(&[dim, dim], DType::F32));
        let b = g.param("b", Shape::new(&[1, dim], DType::F32));
        let pred = g.mm(x, w);
        let biased = g.add(pred, b);
        let diff = g.sub(biased, t);
        let sq = g.mul(diff, diff);
        let loss = g.mean(sq, vec![0, 1], false);
        g.set_outputs(vec![loss]);
        let wrt = vec![
            TrainableParam {
                name: "w".into(),
                node: w,
            },
            TrainableParam {
                name: "b".into(),
                node: b,
            },
        ];
        (g, wrt)
    }

    pub fn initial(dim: usize) -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();
        map.insert(
            "w".to_string(),
            (0..dim * dim).map(|i| 0.01 * (i % 7) as f32).collect(),
        );
        map.insert("b".to_string(), vec![0.0; dim]);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;
    use rlx_ir::infer::GraphExt;

    #[test]
    fn the_fused_update_matches_a_host_optimizer() {
        // The claim this module rests on: fusing the optimizer into the graph
        // changes *where* the arithmetic happens, not what it computes. Compared
        // against `rlx-optim`'s AdamW driven from the same gradients — which is
        // the optimizer the host path would have used.
        const DIM: usize = 4;
        const BATCH: usize = 2;
        let spec = AdamSpec {
            lr: 0.05,
            weight_decay: 0.01,
            ..AdamSpec::default()
        };
        let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
        let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];

        // Fused.
        let (forward, wrt) = regression(BATCH, DIM);
        let start = initial(DIM);
        let mut fused =
            ResidentTrainer::new(&forward, &wrt, &start, &spec, Device::Cpu).expect("fuses");
        let mut fused_losses = Vec::new();
        for _ in 0..12 {
            let out = fused.step(&[("x", &x), ("t", &t)]);
            fused_losses.push(out[0][0]);
        }
        let fused_params = fused.params();

        // Host: the same backward graph, gradients through `rlx-optim`.
        let (forward2, _) = regression(BATCH, DIM);
        let w_id = forward2.param_id("w").unwrap();
        let b_id = forward2.param_id("b").unwrap();
        let backward = rlx_opt::autodiff::grad_with_loss(&forward2, &[w_id, b_id]);
        let mut compiled = Session::new(Device::Cpu).compile(backward);
        let mut opt = rlx_optim::AdamW::new(spec.lr);
        opt.beta1 = spec.beta1;
        opt.beta2 = spec.beta2;
        opt.eps = spec.eps;
        opt.weight_decay = spec.weight_decay;
        let mut host: HashMap<String, Vec<f32>> = initial(DIM);
        let mut host_losses = Vec::new();
        for _ in 0..12 {
            compiled.set_param("w", &host["w"]);
            compiled.set_param("b", &host["b"]);
            let outs = compiled.run(&[("x", &x), ("t", &t), ("d_output", &[1.0])]);
            host_losses.push(outs[0][0]);
            let shapes: [(&str, Vec<usize>); 2] = [("w", vec![DIM, DIM]), ("b", vec![1, DIM])];
            for (i, (name, shape)) in shapes.iter().enumerate() {
                let param = host.get_mut(*name).unwrap();
                rlx_optim::Optimizer::step(&mut opt, name, shape, param, &outs[1 + i]);
            }
            rlx_optim::Optimizer::end_iteration(&mut opt);
        }

        // Step 1 runs both on identical parameters, so it isolates "same
        // formula" from "same rounding": it must agree exactly.
        assert_eq!(
            fused_losses[0].to_bits(),
            host_losses[0].to_bits(),
            "the first step should be bit-identical: {} vs {}",
            fused_losses[0],
            host_losses[0]
        );

        // Later steps drift by float reordering, not by algorithm: the fused
        // graph evaluates `p - lr·wd·p` where `rlx-optim` evaluates
        // `p·(1 - lr·wd)`, and twelve compounding steps of that is tens of f32
        // ulps. The bound is relative and stated rather than tuned until green.
        const REL: f32 = 1e-5;
        for (step, (a, b)) in fused_losses.iter().zip(host_losses.iter()).enumerate() {
            assert!(
                (a - b).abs() <= REL * b.abs().max(1.0),
                "step {step}: fused loss {a} vs host {b}"
            );
        }
        for name in ["w", "b"] {
            for (i, (a, b)) in fused_params[name].iter().zip(host[name].iter()).enumerate() {
                assert!(
                    (a - b).abs() <= REL * b.abs().max(1.0),
                    "{name}[{i}]: fused {a} vs host {b}"
                );
            }
        }
        // Loss must actually have moved, or the comparison proves nothing.
        assert!(
            fused_losses[11] < fused_losses[0] * 0.5,
            "the fused trainer did not learn: {fused_losses:?}"
        );
    }

    /// The fused `rule` and the host `opt` from the same start, 12 steps each →
    /// (fused losses, host losses, fused params, host params).
    #[allow(clippy::type_complexity)]
    fn fused_and_host(
        rule: FusedRule,
        mut opt: Box<dyn rlx_optim::Optimizer>,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        HashMap<String, Vec<f32>>,
        HashMap<String, Vec<f32>>,
    ) {
        const DIM: usize = 4;
        const BATCH: usize = 2;
        let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
        let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];
        let (forward, wrt) = regression(BATCH, DIM);
        let mut fused = ResidentTrainer::with_rule(
            &forward,
            &wrt,
            &initial(DIM),
            &rule,
            Device::Cpu,
            &CompileOptions::new(),
        )
        .expect("fuses");
        let fused_losses: Vec<f32> = (0..12)
            .map(|_| fused.step(&[("x", &x), ("t", &t)])[0][0])
            .collect();

        let (forward2, _) = regression(BATCH, DIM);
        let ids = [
            forward2.param_id("w").unwrap(),
            forward2.param_id("b").unwrap(),
        ];
        let mut compiled =
            Session::new(Device::Cpu).compile(rlx_opt::autodiff::grad_with_loss(&forward2, &ids));
        let mut host = initial(DIM);
        let mut host_losses = Vec::new();
        for _ in 0..12 {
            compiled.set_param("w", &host["w"]);
            compiled.set_param("b", &host["b"]);
            let outs = compiled.run(&[("x", &x), ("t", &t), ("d_output", &[1.0])]);
            host_losses.push(outs[0][0]);
            let mut w = host["w"].clone();
            let mut b = host["b"].clone();
            let mut items = [
                rlx_optim::OptItem {
                    name: "w",
                    shape: &[DIM, DIM],
                    param: &mut w,
                    grad: &outs[1],
                },
                rlx_optim::OptItem {
                    name: "b",
                    shape: &[1, DIM],
                    param: &mut b,
                    grad: &outs[2],
                },
            ];
            opt.step_batch(&mut items);
            opt.end_iteration();
            host.insert("w".into(), w);
            host.insert("b".into(), b);
        }
        (fused_losses, host_losses, fused.params(), host)
    }

    fn assert_close(
        fused: &[f32],
        host: &[f32],
        fp: &HashMap<String, Vec<f32>>,
        hp: &HashMap<String, Vec<f32>>,
    ) {
        // The Newton–Schulz iteration is fifteen f32 matmuls a step; the host
        // sums its Frobenius norm in f64. The bound is relative and stated.
        const REL: f32 = 1e-4;
        for (step, (a, b)) in fused.iter().zip(host).enumerate() {
            assert!(
                (a - b).abs() <= REL * b.abs().max(1.0),
                "step {step}: fused {a} vs host {b}"
            );
        }
        for name in ["w", "b"] {
            for (i, (a, b)) in fp[name].iter().zip(&hp[name]).enumerate() {
                assert!(
                    (a - b).abs() <= REL * b.abs().max(1.0),
                    "{name}[{i}]: fused {a} vs host {b}"
                );
            }
        }
        assert!(fused[11] < fused[0] * 0.9, "did not learn: {fused:?}");
    }

    #[test]
    fn the_fused_muon_matches_rlx_optims_muon() {
        let mut spec = MuonSpec::new(0.02);
        spec.weight_decay = 0.01;
        let mut host = rlx_optim::Muon::new(0.02);
        host.weight_decay = 0.01;
        let (f, h, fp, hp) = fused_and_host(FusedRule::Muon(spec), Box::new(host));
        assert_close(&f, &h, &fp, &hp);
        // Without Nesterov too.
        let mut spec = MuonSpec::new(0.02);
        spec.nesterov = false;
        let mut host = rlx_optim::Muon::new(0.02);
        host.nesterov = false;
        let (f, h, fp, hp) = fused_and_host(FusedRule::Muon(spec), Box::new(host));
        assert_close(&f, &h, &fp, &hp);
    }

    #[test]
    fn the_fused_muon_adamw_matches_rlx_optims() {
        let lr = 0.05;
        let host = rlx_optim::MuonAdamW::new(lr).with_weight_decay(0.01);
        let adamw = AdamSpec {
            lr,
            beta1: host.adamw.beta1,
            beta2: host.adamw.beta2,
            eps: host.adamw.eps,
            weight_decay: 0.01,
        };
        let spec = MuonSpec {
            lr: lr * host.muon_scale,
            adamw: Some(adamw),
            muon_scale: host.muon_scale,
            ..MuonSpec::new(lr)
        };
        let (f, h, fp, hp) = fused_and_host(FusedRule::Muon(spec), Box::new(host));
        assert_close(&f, &h, &fp, &hp);
    }

    #[test]
    fn freezing_holds_a_parameter_and_its_moments() {
        // The gate multiplies the whole delta *including* the moment refresh, so
        // unfreezing resumes rather than lurching on moments built from
        // gradients that were never applied.
        const DIM: usize = 4;
        let (forward, wrt) = regression(2, DIM);
        let start = initial(DIM);
        let mut trainer =
            ResidentTrainer::new(&forward, &wrt, &start, &AdamSpec::new(0.1), Device::Cpu)
                .expect("fuses");
        let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
        let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];

        assert!(trainer.set_frozen("b", true));
        assert!(!trainer.set_frozen("nope", true));
        assert!(trainer.is_frozen("b"));
        for _ in 0..5 {
            trainer.step(&[("x", &x), ("t", &t)]);
        }
        let held = trainer.params();
        assert_eq!(held["b"], start["b"], "a frozen parameter moved");
        assert_ne!(held["w"], start["w"], "the unfrozen parameter did not move");
        // Moments held too.
        let moments = trainer.moments();
        let (_, m_b, v_b) = moments.iter().find(|(n, _, _)| n == "b").unwrap();
        assert!(
            m_b.iter().all(|x| *x == 0.0) && v_b.iter().all(|x| *x == 0.0),
            "a frozen parameter accumulated moments"
        );

        trainer.set_frozen("b", false);
        for _ in 0..5 {
            trainer.step(&[("x", &x), ("t", &t)]);
        }
        assert_ne!(
            trainer.params()["b"],
            start["b"],
            "unfreezing did not resume updates"
        );
    }

    #[test]
    fn moments_and_step_count_round_trip() {
        // Resuming needs the counter as much as the buffers: Adam's bias
        // correction is a function of it, so restoring moments at step 0 takes a
        // step the uninterrupted run never would.
        const DIM: usize = 4;
        let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
        let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];
        let spec = AdamSpec::new(0.05);

        let (forward, wrt) = regression(2, DIM);
        let mut a =
            ResidentTrainer::new(&forward, &wrt, &initial(DIM), &spec, Device::Cpu).unwrap();
        for _ in 0..6 {
            a.step(&[("x", &x), ("t", &t)]);
        }
        let snapshot = a.params();
        let moments = a.moments();
        let steps = a.steps();
        let continued = {
            let mut out = Vec::new();
            for _ in 0..6 {
                out.push(a.step(&[("x", &x), ("t", &t)])[0][0]);
            }
            out
        };

        let (forward2, wrt2) = regression(2, DIM);
        let mut b =
            ResidentTrainer::new(&forward2, &wrt2, &initial(DIM), &spec, Device::Cpu).unwrap();
        b.set_params(&snapshot);
        assert!(b.restore_moments(&moments, steps));
        let resumed: Vec<f32> = (0..6)
            .map(|_| b.step(&[("x", &x), ("t", &t)])[0][0])
            .collect();

        for (step, (a, b)) in continued.iter().zip(resumed.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step}: resumed {b} != uninterrupted {a}"
            );
        }
    }

    #[test]
    fn an_unknown_trainable_parameter_is_an_error_not_a_panic() {
        let (forward, _) = regression(2, 4);
        let wrt = vec![TrainableParam {
            name: "nope".into(),
            node: NodeId(0),
        }];
        match ResidentTrainer::new(
            &forward,
            &wrt,
            &initial(4),
            &AdamSpec::default(),
            Device::Cpu,
        ) {
            Err(FuseError::UnknownParam(name)) => assert_eq!(name, "nope"),
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("'nope' is not a Param, so this must fail"),
        }
    }

    #[test]
    fn aux_outputs_survive_the_fusion() {
        // A forward graph ending in [loss, logits] should still hand both back:
        // reading accuracy during training should not cost a second forward.
        let mut g = Graph::new("aux");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        let w = g.param("w", Shape::new(&[3, 3], DType::F32));
        let logits = g.mm(x, w);
        let sq = g.mul(logits, logits);
        let loss = g.mean(sq, vec![0, 1], false);
        g.set_outputs(vec![loss, logits]);
        let wrt = vec![TrainableParam {
            name: "w".into(),
            node: w,
        }];
        let mut start = HashMap::new();
        start.insert("w".to_string(), vec![0.1; 9]);

        let mut trainer =
            ResidentTrainer::new(&g, &wrt, &start, &AdamSpec::new(0.01), Device::Cpu).unwrap();
        let out = trainer.step(&[("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
        assert_eq!(
            out.len(),
            2,
            "expected [loss, logits], got {} outputs",
            out.len()
        );
        assert_eq!(out[0].len(), 1, "loss should be scalar");
        assert_eq!(out[1].len(), 6, "logits should be [2, 3]");
    }
}

#[cfg(test)]
mod forward_only_tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn forward_only_moves_nothing() {
        // A validation pass must not move the weights and must not advance
        // Adam's bias correction, or the next real step differs from the one an
        // un-validated run would have taken.
        let (forward, wrt) = regression(2, 4);
        let start = initial(4);
        let mut trainer =
            ResidentTrainer::new(&forward, &wrt, &start, &AdamSpec::new(0.05), Device::Cpu)
                .unwrap();
        let x: Vec<f32> = vec![1.0, 0.5, -0.5, 2.0, 0.25, -1.0, 1.5, 0.0];
        let t: Vec<f32> = vec![1.0, -1.0, 0.5, 0.0, 0.0, 2.0, -1.0, 1.0];

        for _ in 0..4 {
            trainer.step(&[("x", &x), ("t", &t)]);
        }
        let before = trainer.params();
        let steps_before = trainer.steps();
        let moments_before = trainer.moments();

        let evaluated = trainer.forward_only(&[("x", &x), ("t", &t)]);
        assert_eq!(evaluated.len(), 1);
        assert_eq!(
            trainer.steps(),
            steps_before,
            "forward_only advanced the counter"
        );
        assert_eq!(trainer.params(), before, "forward_only moved the weights");
        assert_eq!(
            trainer.moments(),
            moments_before,
            "forward_only moved the moments"
        );

        // And the next real step matches one taken without the validation pass.
        let with_eval = trainer.step(&[("x", &x), ("t", &t)])[0][0];
        let (forward2, wrt2) = regression(2, 4);
        let mut clean = ResidentTrainer::new(
            &forward2,
            &wrt2,
            &initial(4),
            &AdamSpec::new(0.05),
            Device::Cpu,
        )
        .unwrap();
        for _ in 0..4 {
            clean.step(&[("x", &x), ("t", &t)]);
        }
        let without_eval = clean.step(&[("x", &x), ("t", &t)])[0][0];
        assert_eq!(
            with_eval.to_bits(),
            without_eval.to_bits(),
            "a validation pass changed the trajectory: {with_eval} vs {without_eval}"
        );
    }
}
