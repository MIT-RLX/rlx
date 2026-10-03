// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training from JavaScript.
//!
//! Two levels. [`Optimizer`] is `rlx-optim` — Adam, AdamW, Muon, SOAP, Lion
//! and the rest — driving arrays the script owns. [`Trainer`] is the whole
//! loop: it differentiates the forward graph once, compiles it once, keeps
//! the parameters, and gives back a scalar loss per step.
//!
//! ```js
//! const t = new rlx.Trainer(buildLoss(), {
//!   wrt: ["w1", "b1", "w2", "b2"],
//!   init: initialWeights,
//!   device: "metal",
//!   optimizer: { kind: "adamw", lr: 3e-4, weightDecay: 0.01 },
//! });
//! for (const batch of data) console.log(t.step(batch));
//! const trained = t.params();
//! ```
//!
//! The forward graph must have exactly one output, the scalar loss —
//! `rlx.grad` seeds `d_output` with `1`, which only means "differentiate the
//! loss" when the loss is what the graph ends on.
//!
//! # Where the parameters live
//!
//! The optimizer runs on the host and the updated weights are re-uploaded
//! each step. That is correct everywhere and fast enough for the model sizes
//! a script trains, but it is a host round trip per step: the device-resident
//! path, where the optimizer update is fused *into* the compiled graph, is a
//! core change and not something a binding should fake.

use std::collections::HashMap;

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, PropFlags};
use quickrs_core::value::{JsResult, Value};
use rlx_ir::{Dim, Graph, NodeId, Op};
use rlx_optim::{
    Adafactor, Adam, AdamW, Lamb, Lion, Mars, Muon, NAdamW, OptItem, Optimizer as OptimizerTrait,
    QHAdamW, RAdam, Sgd, Soap, Sophia,
};
use rlx_runtime::{CompiledGraph, Device, Session};

use crate::api::graph::take_graph;
use crate::convert::*;
use crate::handle::{self, CLASS_OPTIMIZER, CLASS_TRAINER};

// ── optimizer construction ──────────────────────────────────

fn num(ctx: &mut Context, options: &Value, name: &str, default: f32) -> JsResult<f32> {
    let v = field(ctx, options, name)?;
    if is_nullish(&v) {
        Ok(default)
    } else {
        to_f32(ctx, &v)
    }
}

fn flag(ctx: &mut Context, options: &Value, name: &str, default: bool) -> JsResult<bool> {
    let v = field(ctx, options, name)?;
    Ok(if is_nullish(&v) { default } else { to_bool(&v) })
}

/// Build one of `rlx-optim`'s algorithms from `{kind, lr, …}`.
///
/// Unknown hyperparameters are ignored rather than rejected, but an unknown
/// `kind` is an error: silently falling back to SGD would make a config typo
/// look like a bad learning rate.
pub fn build_optimizer(ctx: &mut Context, options: &Value) -> JsResult<Box<dyn OptimizerTrait>> {
    let kind_v = field(ctx, options, "kind")?;
    let kind = to_string_or(ctx, &kind_v, "adamw")?;
    let lr = num(ctx, options, "lr", 1e-3)?;
    let beta1 = num(ctx, options, "beta1", 0.9)?;
    let beta2 = num(ctx, options, "beta2", 0.999)?;
    let eps = num(ctx, options, "eps", 1e-8)?;
    let weight_decay = num(ctx, options, "weightDecay", 0.0)?;

    Ok(match kind.trim().to_ascii_lowercase().as_str() {
        "sgd" => {
            let mut o = Sgd::new(lr);
            o.momentum = num(ctx, options, "momentum", o.momentum)?;
            o.nesterov = flag(ctx, options, "nesterov", o.nesterov)?;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "adam" => {
            let mut o = Adam::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = weight_decay;
            // Same knob as AdamW: f32 moment arithmetic is 2.1x faster than the
            // f64-intermediate default and is what PyTorch does.
            o.f32_math = flag(ctx, options, "f32Math", o.f32_math)?;
            Box::new(o)
        }
        "adamw" => {
            let mut o = AdamW::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = num(ctx, options, "weightDecay", o.weight_decay)?;
            // `f32Math: true` keeps the moment arithmetic in f32, matching
            // PyTorch's default. rlx-optim defaults to f64 intermediates, which
            // are slightly more accurate and slightly slower — and are what the
            // fused (`resident`) path, being pure f32, does *not* do.
            o.f32_math = flag(ctx, options, "f32Math", o.f32_math)?;
            Box::new(o)
        }
        "lion" => {
            let mut o = Lion::new(lr);
            o.beta1 = num(ctx, options, "beta1", o.beta1)?;
            o.beta2 = num(ctx, options, "beta2", o.beta2)?;
            o.weight_decay = num(ctx, options, "weightDecay", o.weight_decay)?;
            Box::new(o)
        }
        "muon" => {
            let mut o = Muon::new(lr);
            o.momentum = num(ctx, options, "momentum", o.momentum)?;
            o.nesterov = flag(ctx, options, "nesterov", o.nesterov)?;
            o.weight_decay = num(ctx, options, "weightDecay", o.weight_decay)?;
            Box::new(o)
        }
        "radam" => {
            let mut o = RAdam::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "nadamw" => {
            let mut o = NAdamW::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = num(ctx, options, "weightDecay", o.weight_decay)?;
            Box::new(o)
        }
        "lamb" => {
            let mut o = Lamb::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "mars" => {
            let mut o = Mars::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.gamma = num(ctx, options, "gamma", o.gamma)?;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "soap" => {
            let mut o = Soap::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "sophia" => {
            let mut o = Sophia::new(lr);
            o.gamma = num(ctx, options, "gamma", o.gamma)?;
            o.rho = num(ctx, options, "rho", o.rho)?;
            o.eps = eps;
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        "qhadamw" => {
            let mut o = QHAdamW::new(lr);
            o.beta1 = beta1;
            o.beta2 = beta2;
            o.eps = eps;
            o.weight_decay = num(ctx, options, "weightDecay", o.weight_decay)?;
            Box::new(o)
        }
        "adafactor" => {
            let mut o = Adafactor::new();
            // Adafactor's relative step size means "no lr" is a real choice,
            // not a missing one — only override when the script says so.
            let lr_v = field(ctx, options, "lr")?;
            if !is_nullish(&lr_v) {
                o.lr = Some(to_f32(ctx, &lr_v)?);
            }
            o.weight_decay = weight_decay;
            Box::new(o)
        }
        other => {
            return ctx.throw_type(&format!(
                "unknown optimizer '{other}' (sgd, adam, adamw, lion, muon, radam, nadamw, \
                 lamb, mars, soap, sophia, qhadamw, adafactor)"
            ));
        }
    })
}

// ── standalone Optimizer ────────────────────────────────────

pub struct OptimizerSlot {
    inner: Box<dyn OptimizerTrait>,
    kind: String,
}

fn optimizer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut OptimizerSlot> {
    handle::borrow_mut(ctx, this, CLASS_OPTIMIZER, "rlx.Optimizer")
}

fn optimizer_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    // `new rlx.Optimizer("adamw", {lr})` and `new rlx.Optimizer({kind, lr})`
    // both work; the first is what a one-liner reaches for.
    let first = arg(args, 0).clone();
    let options = if first.as_string().is_some() {
        let mut merged = arg(args, 1).clone();
        if is_nullish(&merged) {
            merged = new_object(ctx, vec![]);
        }
        let kind = first.clone();
        if let Some(obj) = merged.as_object().cloned() {
            ctx.define_value(&obj, "kind", kind, PropFlags::C_W_E);
        }
        merged
    } else {
        first
    };
    let kind_v = field(ctx, &options, "kind")?;
    let kind = to_string_or(ctx, &kind_v, "adamw")?;
    let inner = build_optimizer(ctx, &options)?;
    Ok(handle::wrap(
        ctx,
        CLASS_OPTIMIZER,
        OptimizerSlot { inner, kind },
    ))
}

/// `step({name: {param, grad, shape}})` → the updated params, in place.
///
/// The arrays passed in are *copied*: JavaScript's `Float32Array` is not a
/// buffer Rust can mutate through safely while script may still reach it, so
/// the updated values come back as the return value.
fn optimizer_step(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let names = own_keys(ctx, &bag)?;
    let mut entries: Vec<(String, Vec<f32>, Vec<f32>, Vec<usize>)> =
        Vec::with_capacity(names.len());
    for name in names {
        let slot = field(ctx, &bag, &name)?;
        let param_v = field(ctx, &slot, "param")?;
        let grad_v = field(ctx, &slot, "grad")?;
        if is_nullish(&param_v) || is_nullish(&grad_v) {
            return ctx.throw_type(&format!(
                "optimizer.step('{name}'): expected {{param, grad, shape?}}"
            ));
        }
        let param = to_f32_vec(ctx, &param_v, &format!("param '{name}'"))?;
        let grad = to_f32_vec(ctx, &grad_v, &format!("grad '{name}'"))?;
        if param.len() != grad.len() {
            return ctx.throw_type(&format!(
                "optimizer.step('{name}'): param has {} values but grad has {}",
                param.len(),
                grad.len()
            ));
        }
        let shape_v = field(ctx, &slot, "shape")?;
        let shape = if is_nullish(&shape_v) {
            vec![param.len()]
        } else {
            to_usize_vec(ctx, &shape_v, "shape")?
        };
        entries.push((name, param, grad, shape));
    }

    {
        let slot = optimizer_of(ctx, this)?;
        let mut items: Vec<OptItem<'_>> = entries
            .iter_mut()
            .map(|(name, param, grad, shape)| OptItem {
                name: name.as_str(),
                shape: shape.as_slice(),
                param: param.as_mut_slice(),
                grad: grad.as_slice(),
            })
            .collect();
        slot.inner.step_batch(&mut items);
        slot.inner.end_iteration();
    }

    let obj = ctx.new_object();
    for (name, param, _, _) in &entries {
        let array = new_f32_array(ctx, param);
        ctx.define_value(&obj, name, array, PropFlags::C_W_E);
    }
    Ok(Value::Object(obj))
}

fn optimizer_set_lr(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let lr = to_f32(ctx, arg(args, 0))?;
    optimizer_of(ctx, this)?.inner.set_lr(lr);
    Ok(Value::Undefined)
}

fn optimizer_to_string(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let text = format!("[rlx.Optimizer {}]", optimizer_of(ctx, this)?.kind);
    Ok(new_string(ctx, &text))
}

// ── Trainer ─────────────────────────────────────────────────

pub struct TrainerSlot {
    /// The host path's compiled backward graph. `None` when the optimizer is
    /// fused into the graph instead — building both would double the compile.
    compiled: Option<CompiledGraph>,
    /// Set when the update runs on-device. See [`rlx_runtime::train`].
    resident: Option<rlx_runtime::train::ResidentTrainer>,
    optimizer: Box<dyn OptimizerTrait>,
    /// Which algorithm, for the checkpoint header — loading Adam moments into
    /// Lion would be nonsense, so the name is recorded and checked.
    #[cfg_attr(not(feature = "gguf"), allow(dead_code))]
    optimizer_kind: String,
    /// Trainable parameters, in the order `grad` emits their gradients.
    names: Vec<String>,
    shapes: Vec<Vec<usize>>,
    values: Vec<Vec<f32>>,
    /// Per trainable param: skip its optimizer update.
    ///
    /// The gradient is still computed — `wrt` is baked into the compiled graph —
    /// so this changes what moves, not what costs. Params with no gradient at
    /// all are the ones left out of `wrt`, which is how LoRA freezes a base
    /// weight.
    frozen: Vec<bool>,
    /// `init` entries that are not in `wrt`: uploaded once, never updated, but
    /// kept here so a checkpoint is complete and `params()` can report them.
    fixed: Vec<(String, Vec<f32>)>,
    device: &'static str,
    steps: u64,
    lr: f32,
    /// Global gradient-norm clip, applied across all trainable params together.
    clip_norm: Option<f32>,
    /// Global grad norm from the last step, before clipping.
    last_grad_norm: f32,
    /// Reused per-step input buffers — a 400-step loop should allocate once.
    scratch: InputScratch,
}

fn trainer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut TrainerSlot> {
    handle::borrow_mut(ctx, this, CLASS_TRAINER, "rlx.Trainer")
}

impl TrainerSlot {
    /// The host path's compiled graph.
    ///
    /// Only the host path has one: the resident path compiles a single fused
    /// graph that owns the update, so there is nothing separate to reach.
    fn host(&mut self) -> Option<&mut CompiledGraph> {
        self.compiled.as_mut()
    }

    fn is_resident(&self) -> bool {
        self.resident.is_some()
    }
}

/// Map param names to their node ids and static shapes.
fn param_nodes(graph: &Graph) -> HashMap<String, (NodeId, Vec<usize>)> {
    let mut out = HashMap::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        if let Op::Param { name } = &node.op {
            let dims: Vec<usize> = node
                .shape
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => *n,
                    Dim::Dynamic(_) => 0,
                })
                .collect();
            out.insert(name.clone(), (NodeId(index as u32), dims));
        }
    }
    out
}

/// `new rlx.Trainer(lossGraph, {wrt, init, device, precision, optimizer, frozen})`.
fn trainer_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let options = arg(args, 1).clone();

    let device_v = field(ctx, &options, "device")?;
    let device_label = to_string_or(ctx, &device_v, "cpu")?;
    let device = match rlx_runtime::parse_device(&device_label) {
        Ok(d) => d,
        Err(e) => return ctx.throw_type(&e.to_string()),
    };
    if !rlx_runtime::is_available(device) {
        return ctx.throw_type(&format!(
            "Trainer: device '{device_label}' is not in this build — rebuild with \
             `--features {device_label}`"
        ));
    }

    let wrt_v = field(ctx, &options, "wrt")?;
    let wrt_names = to_string_vec(ctx, &wrt_v)?;
    let policy_v = field(ctx, &options, "policy")?;
    let policy = crate::api::compile::parse_precision_policy(ctx, &policy_v)?;
    if let Some(policy) = policy.as_ref() {
        crate::api::compile::check_policy_supported(ctx, device, policy)?;
    }
    let clip_v = field(ctx, &options, "clipNorm")?;
    let clip_norm = if is_nullish(&clip_v) {
        None
    } else {
        Some(to_f32(ctx, &clip_v)?)
    };
    if wrt_names.is_empty() {
        return ctx.throw_type("Trainer: `wrt` must name at least one parameter to train");
    }
    let init_bag = field(ctx, &options, "init")?;
    let optimizer_opts = field(ctx, &options, "optimizer")?;
    let kind_v = field(ctx, &optimizer_opts, "kind")?;
    let optimizer_kind = to_string_or(ctx, &kind_v, "adamw")?;
    let lr = num(ctx, &optimizer_opts, "lr", 1e-3)?;
    let optimizer = build_optimizer(ctx, &optimizer_opts)?;

    // Every conversion that can run script happens before the graph moves.
    let mut init: Vec<(String, Vec<f32>)> = Vec::new();
    for name in own_keys(ctx, &init_bag)? {
        let value = field(ctx, &init_bag, &name)?;
        let data = to_f32_vec(ctx, &value, &format!("init['{name}']"))?;
        init.push((name, data));
    }

    let forward = take_graph(ctx, arg(args, 0))?;
    if forward.outputs.len() != 1 {
        return ctx.throw_type(&format!(
            "Trainer: the forward graph must have exactly one output (the scalar loss), got {}",
            forward.outputs.len()
        ));
    }
    let nodes = param_nodes(&forward);

    let mut wrt = Vec::with_capacity(wrt_names.len());
    let mut shapes = Vec::with_capacity(wrt_names.len());
    for name in &wrt_names {
        match nodes.get(name) {
            Some((node, dims)) => {
                wrt.push(*node);
                shapes.push(dims.clone());
            }
            None => {
                let mut known: Vec<&str> = nodes.keys().map(String::as_str).collect();
                known.sort();
                return ctx.throw_type(&format!(
                    "Trainer: '{name}' is not a Param in this graph — it has [{}]",
                    known.join(", ")
                ));
            }
        }
    }

    // ── the resident path: the optimizer update fused into the graph ──
    //
    // Measured on this shape (MNIST MLP, batch 64), the host optimizer is
    // 40-55% of a step and does not vary with the device. See
    // `rlx_runtime::train`.
    let resident_v = field(ctx, &options, "resident")?;
    if to_bool(&resident_v) {
        // Per-backend allow-list, because the fused path is only worth having
        // where it is both correct and faster, and neither is universal:
        //
        //   metal  verified — matches the host path exactly, and is faster
        //   cpu    refused  — no device buffers, so the fused graph feeds params
        //                     and moments as ordinary inputs and reads three
        //                     buffers back per parameter. Measured 3.3x *slower*
        //                     than the host optimizer.
        //   wgpu   verified — was refused: the trajectory diverged from step 2.
        //                     The cause was not the write-back but the arena-copy
        //                     bind window, sized `hi - lo` while starting at `lo`
        //                     floored to 256 — so an operand at the far end fell
        //                     outside the binding and its clamped access did
        //                     nothing. Fixed in `rlx-wgpu`'s
        //                     `dispatch_arena_copy_bytes`; now tracks CPU to
        //                     ~1e-7 for 8 steps
        //                     (`rlx-runtime/tests/resident_trainer_wgpu.rs`).
        //   others unverified — no hardware here to check on. Opt in with
        //                     `resident: "unverified"` rather than having this
        //                     quietly claim something untested.
        let forced = resident_v
            .as_string()
            .map(|s| s.to_string_lossy().eq_ignore_ascii_case("unverified"))
            .unwrap_or(false);
        let verified = matches!(device, Device::Metal | Device::Gpu);
        let refuse = match device {
            Device::Cpu => Some(
                "the CPU backend has no device buffers, so the fused update feeds the \
                 parameters and moments as ordinary inputs and reads three buffers back per \
                 parameter — measured 3.3x slower than the host optimizer",
            ),
            _ => None,
        };
        if let Some(reason) = refuse {
            return ctx.throw_type(&format!(
                "Trainer({{resident: true}}) on '{}': {reason}. Drop `resident` to use the \
                 host optimizer.",
                rlx_runtime::device_label(device)
            ));
        }
        if !verified && !forced {
            return ctx.throw_type(&format!(
                "Trainer({{resident: true}}) on '{}': the fused update is not verified on \
                 this backend. Pass `resident: \"unverified\"` to try it, and compare a few \
                 steps against the host path before trusting the numbers.",
                rlx_runtime::device_label(device)
            ));
        }
        let kind = optimizer_kind.trim().to_ascii_lowercase();
        if kind != "adam" && kind != "adamw" {
            return ctx.throw_type(&format!(
                "Trainer({{resident: true}}): only 'adam' and 'adamw' are fused into the \
                 graph; '{kind}' runs on the host. Drop `resident` to use it."
            ));
        }
        if clip_norm.is_some() {
            return ctx.throw_type(
                "Trainer({resident: true}): `clipNorm` needs the gradients on the host, and \
                 the fused path never brings them back. Drop one of the two.",
            );
        }
        // Defaults come from the host optimizer this replaces, not from a second
        // set written here. `AdamW::new` defaults `weight_decay` to 0.01 (as
        // PyTorch does); hardcoding 0.0 meant `{kind: "adamw", lr}` trained a
        // *different objective* on the two paths — 3.5e-4 per step on a 0.35
        // weight, with nothing to indicate it.
        let (d_beta1, d_beta2, d_eps, d_wd) = if kind == "adam" {
            let d = rlx_optim::Adam::new(lr);
            (d.beta1, d.beta2, d.eps, d.weight_decay)
        } else {
            let d = rlx_optim::AdamW::new(lr);
            (d.beta1, d.beta2, d.eps, d.weight_decay)
        };
        let weight_decay = num(ctx, &optimizer_opts, "weightDecay", d_wd)?;
        // Classic Adam folds L2 into the gradient; the fused graph implements
        // AdamW's decoupled decay. With a nonzero decay those are different
        // algorithms, so refuse rather than quietly substitute one.
        if kind == "adam" && weight_decay != 0.0 {
            return ctx.throw_type(
                "Trainer({resident: true}) with kind 'adam' and a nonzero weightDecay: \
                 classic Adam folds decay into the gradient, while the fused update applies \
                 AdamW's decoupled decay. Use kind 'adamw', or drop `resident`.",
            );
        }
        let spec = rlx_runtime::train::AdamSpec {
            lr,
            beta1: num(ctx, &optimizer_opts, "beta1", d_beta1)?,
            beta2: num(ctx, &optimizer_opts, "beta2", d_beta2)?,
            eps: num(ctx, &optimizer_opts, "eps", d_eps)?,
            weight_decay,
        };
        let slots: Vec<rlx_runtime::train::TrainableParam> = wrt_names
            .iter()
            .zip(wrt.iter())
            .map(|(name, node)| rlx_runtime::train::TrainableParam {
                name: name.clone(),
                node: *node,
            })
            .collect();
        let seed: HashMap<String, Vec<f32>> = init.iter().cloned().collect();
        let mut compile_opts = rlx_runtime::CompileOptions::new();
        if let Some(policy) = policy.clone() {
            compile_opts = compile_opts.policy(policy);
        }
        let built = crate::panics::catch(ctx, "Trainer(resident)", || {
            rlx_runtime::train::ResidentTrainer::with_options(
                &forward,
                &slots,
                &seed,
                &spec,
                device,
                &compile_opts,
            )
        })?;
        let trainer = match built {
            Ok(t) => t,
            Err(e) => return ctx.throw_type(&format!("Trainer(resident): {e}")),
        };
        let fixed: Vec<(String, Vec<f32>)> = init
            .into_iter()
            .filter(|(name, _)| !wrt_names.contains(name))
            .collect();
        let count = wrt_names.len();
        return Ok(handle::wrap(
            ctx,
            CLASS_TRAINER,
            TrainerSlot {
                compiled: None,
                resident: Some(trainer),
                optimizer,
                optimizer_kind,
                names: wrt_names,
                shapes,
                values: Vec::new(),
                frozen: vec![false; count],
                fixed,
                device: rlx_runtime::device_label(device),
                steps: 0,
                lr,
                clip_norm: None,
                last_grad_norm: f32::NAN,
                scratch: InputScratch::default(),
            },
        ));
    }

    // `grad_with_loss` panics on a `wrt` the loss does not reach (a typo, or a
    // parameter the graph never uses) and on an op with no VJP rule. Both are
    // reported rather than aborting the process.
    let backward = crate::panics::catch(ctx, "Trainer: differentiating the loss", || {
        rlx_opt::autodiff::grad_with_loss(&forward, &wrt)
    })?;
    let mut compiled = match policy {
        Some(policy) => {
            let opts = rlx_runtime::CompileOptions::new().policy(policy);
            crate::panics::catch(ctx, "Trainer: compiling", || {
                Session::new(device).compile_with(backward, &opts)
            })?
        }
        None => crate::panics::catch(ctx, "Trainer: compiling", || {
            Session::new(device).compile(backward)
        })?,
    };

    // Frozen params (present in `init` but not in `wrt`) are uploaded once;
    // trainable ones are re-uploaded per step from the trainer's own copy.
    let trainable_count = wrt_names.len();
    let mut values = vec![Vec::new(); trainable_count];
    let mut fixed: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, data) in init {
        match wrt_names.iter().position(|n| *n == name) {
            Some(i) => values[i] = data,
            None => {
                compiled.set_param(&name, &data);
                // Kept, not forgotten: a checkpoint without the frozen base
                // weights cannot reproduce the model it came from.
                fixed.push((name, data));
            }
        }
    }
    for (i, name) in wrt_names.iter().enumerate() {
        if values[i].is_empty() {
            return ctx.throw_type(&format!(
                "Trainer: no initial value for trainable parameter '{name}' — pass it in `init`"
            ));
        }
        let expected: usize = shapes[i].iter().product();
        if expected != 0 && values[i].len() != expected {
            return ctx.throw_type(&format!(
                "Trainer: '{name}' is declared {:?} ({expected} values) but `init` gave {}",
                shapes[i],
                values[i].len()
            ));
        }
    }

    Ok(handle::wrap(
        ctx,
        CLASS_TRAINER,
        TrainerSlot {
            compiled: Some(compiled),
            resident: None,
            optimizer,
            names: wrt_names,
            shapes,
            values,
            optimizer_kind,
            frozen: vec![false; trainable_count],
            fixed,
            device: rlx_runtime::device_label(device),
            steps: 0,
            lr,
            clip_norm,
            last_grad_norm: 0.0,
            scratch: InputScratch::default(),
        },
    ))
}

/// Load a batch into the trainer's own scratch, seeding `d_output`.
///
/// The scratch is moved out while script may run (argument conversion can hit
/// a getter) and put back before the graph executes, so the handle is never
/// borrowed twice.
fn load_batch(ctx: &mut Context, this: &Value, bag: &Value) -> JsResult<()> {
    let slot = trainer_of(ctx, this)?;
    let mut scratch = std::mem::take(&mut slot.scratch);
    let loaded = scratch.load(ctx, bag);
    // `grad` seeds the upstream gradient through this input; a caller that
    // supplied their own keeps it, so loss scaling stays available.
    if loaded.is_ok() {
        scratch.push("d_output", &[1.0]);
    }
    let slot = trainer_of(ctx, this)?;
    slot.scratch = scratch;
    loaded
}

/// Upload the trainable weights and run the backward graph.
fn forward_backward(ctx: &mut Context, this: &Value) -> JsResult<Vec<Vec<f32>>> {
    let slot = trainer_of(ctx, this)?;
    let Some(compiled) = slot.compiled.as_mut() else {
        return ctx.throw_internal("forward_backward: called on a resident trainer");
    };
    for (name, data) in slot.names.iter().zip(&slot.values) {
        compiled.set_param(name, data);
    }
    let pairs = slot.scratch.pairs();
    crate::panics::catch(ctx, "Trainer step", || compiled.run(&pairs))
}

/// One step on the resident path: the update is inside the graph, so there is
/// no optimizer to run and no gradient to bring back.
fn resident_step(ctx: &mut Context, this: &Value, evaluate_only: bool) -> JsResult<f32> {
    let slot = trainer_of(ctx, this)?;
    let pairs = slot.scratch.pairs();
    let Some(trainer) = slot.resident.as_mut() else {
        return ctx.throw_internal("resident_step: not a resident trainer");
    };
    let outputs = crate::panics::catch(ctx, "Trainer step (resident)", || {
        if evaluate_only {
            trainer.forward_only(&pairs)
        } else {
            trainer.step(&pairs)
        }
    })?;
    let loss = outputs
        .first()
        .and_then(|o| o.first().copied())
        .unwrap_or(f32::NAN);
    let slot = trainer_of(ctx, this)?;
    if !evaluate_only {
        slot.steps = slot
            .resident
            .as_ref()
            .map(|r| r.steps())
            .unwrap_or(slot.steps);
    }
    Ok(loss)
}

/// `step({x, y})` → the scalar loss, after applying one optimizer update.
fn trainer_step(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    load_batch(ctx, this, &bag)?;
    if trainer_of(ctx, this)?.is_resident() {
        return Ok(Value::number(resident_step(ctx, this, false)? as f64));
    }
    let outputs = forward_backward(ctx, this)?;

    let slot = trainer_of(ctx, this)?;
    // `grad_with_loss` emits [loss, ...grads] in `wrt` order.
    let Some(loss) = outputs.first().and_then(|o| o.first().copied()) else {
        return ctx.throw_internal("Trainer.step: the backward graph produced no loss output");
    };
    if outputs.len() != slot.names.len() + 1 {
        return ctx.throw_internal(&format!(
            "Trainer.step: expected {} gradients, got {}",
            slot.names.len(),
            outputs.len().saturating_sub(1)
        ));
    }

    apply_update(slot, &outputs);
    Ok(Value::number(loss as f64))
}

/// Clip, then hand the unfrozen parameters to the optimizer.
///
/// Clipping is computed over *every* trainable gradient, frozen or not: the
/// global norm is a property of the step, and excluding some tensors from the
/// norm while still taking the step would scale differently than an unfrozen
/// run of the same recipe.
fn apply_update(slot: &mut TrainerSlot, outputs: &[Vec<f32>]) {
    let grads: Vec<&[f32]> = outputs[1..].iter().map(|g| g.as_slice()).collect();
    // `l2_norm` takes one slice; the global norm is the root of the sum of
    // squares across all of them.
    let norm_sq: f64 = grads
        .iter()
        .map(|g| {
            let n = rlx_optim::l2_norm(g) as f64;
            n * n
        })
        .sum();
    slot.last_grad_norm = norm_sq.sqrt() as f32;
    let scale = match slot.clip_norm {
        Some(max) => rlx_optim::global_grad_clip_scale(grads.iter().copied(), max),
        None => 1.0,
    };

    // Scaling allocates only when clipping actually fires.
    let clipped: Option<Vec<Vec<f32>>> = if scale < 1.0 {
        Some(
            grads
                .iter()
                .map(|g| g.iter().map(|x| x * scale).collect())
                .collect(),
        )
    } else {
        None
    };

    let TrainerSlot {
        optimizer,
        names,
        shapes,
        values,
        frozen,
        ..
    } = slot;
    let mut items: Vec<OptItem<'_>> = Vec::with_capacity(names.len());
    for (i, ((name, shape), param)) in names
        .iter()
        .zip(shapes.iter())
        .zip(values.iter_mut())
        .enumerate()
    {
        if frozen[i] {
            continue;
        }
        items.push(OptItem {
            name: name.as_str(),
            shape: shape.as_slice(),
            param: param.as_mut_slice(),
            grad: match &clipped {
                Some(scaled) => scaled[i].as_slice(),
                None => outputs[i + 1].as_slice(),
            },
        });
    }
    optimizer.step_batch(&mut items);
    optimizer.end_iteration();
    slot.steps += 1;
}

/// Loss only, no update — for a validation split.
fn trainer_evaluate(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    load_batch(ctx, this, &bag)?;
    if trainer_of(ctx, this)?.is_resident() {
        return Ok(Value::number(resident_step(ctx, this, true)? as f64));
    }
    let outputs = forward_backward(ctx, this)?;
    let loss = outputs
        .first()
        .and_then(|o| o.first().copied())
        .unwrap_or(f32::NAN);
    Ok(Value::number(loss as f64))
}

// ── lifecycle: start / pause / resume / stop ────────────────

/// `run({steps, batch, lr, onStep, every})` → a summary of the stretch.
///
/// The trainer keeps every piece of state, so this *is* start, pause, resume and
/// stop: call it to run a stretch, return `false` from `onStep` to break out
/// early, call it again to carry on from exactly where it left off. There is no
/// separate `pause()` because there is nothing to pause — the loop is
/// synchronous and the state lives in the handle.
///
/// ```js
/// const r = t.run({ steps: 1000, batch: (i) => nextBatch(),
///                   lr: (i) => cosine(i), every: 100,
///                   onStep: (i, loss) => { log(i, loss); return !stopRequested; } });
/// // r = { steps, stopped, firstLoss, lastLoss, meanLoss, stepsPerSecond, gradNorm }
/// ```
fn trainer_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let options = arg(args, 0).clone();
    let steps_v = field(ctx, &options, "steps")?;
    let steps = if is_nullish(&steps_v) {
        1
    } else {
        to_usize(ctx, &steps_v, "run steps")?
    };
    let batch_fn = field(ctx, &options, "batch")?;
    if is_nullish(&batch_fn) {
        return ctx
            .throw_type("run({batch}): a `batch` function is required — it is what supplies data");
    }
    let lr_fn = field(ctx, &options, "lr")?;
    let on_step = field(ctx, &options, "onStep")?;
    let every_v = field(ctx, &options, "every")?;
    let every = if is_nullish(&every_v) {
        1
    } else {
        to_usize(ctx, &every_v, "run every")?.max(1)
    };

    let start_step = trainer_of(ctx, this)?.steps;
    let began = std::time::Instant::now();
    let mut first: Option<f32> = None;
    let mut last = f32::NAN;
    let mut sum = 0.0f64;
    let mut done = 0usize;
    let mut stopped = false;

    for i in 0..steps {
        let index = Value::number((start_step + i as u64) as f64);

        // Each callback runs with no handle borrowed, so a callback that reaches
        // back into the trainer sees a consistent object rather than an alias.
        if !is_nullish(&lr_fn) {
            let lr = ctx.call(
                lr_fn.clone(),
                Value::Undefined,
                std::slice::from_ref(&index),
            )?;
            if !is_nullish(&lr) {
                let lr = to_f32(ctx, &lr)?;
                let slot = trainer_of(ctx, this)?;
                slot.lr = lr;
                match slot.resident.as_mut() {
                    Some(trainer) => trainer.set_lr(lr),
                    None => slot.optimizer.set_lr(lr),
                }
            }
        }
        let bag = ctx.call(
            batch_fn.clone(),
            Value::Undefined,
            std::slice::from_ref(&index),
        )?;
        if is_nullish(&bag) {
            // A batch provider returning null means "data exhausted".
            stopped = true;
            break;
        }

        load_batch(ctx, this, &bag)?;
        let loss = if trainer_of(ctx, this)?.is_resident() {
            resident_step(ctx, this, false)?
        } else {
            let outputs = forward_backward(ctx, this)?;
            let slot = trainer_of(ctx, this)?;
            let Some(loss) = outputs.first().and_then(|o| o.first().copied()) else {
                return ctx.throw_internal("run: the backward graph produced no loss output");
            };
            if outputs.len() != slot.names.len() + 1 {
                return ctx.throw_internal(&format!(
                    "run: expected {} gradients, got {}",
                    slot.names.len(),
                    outputs.len().saturating_sub(1)
                ));
            }
            apply_update(slot, &outputs);
            loss
        };
        let slot = trainer_of(ctx, this)?;
        let _ = &slot;
        first.get_or_insert(loss);
        last = loss;
        sum += loss as f64;
        done += 1;

        if !is_nullish(&on_step) && (done.is_multiple_of(every) || i + 1 == steps) {
            let keep = ctx.call(
                on_step.clone(),
                Value::Undefined,
                &[index, Value::number(loss as f64)],
            )?;
            // Only an explicit `false` stops: a callback that logs and returns
            // nothing should not end the run.
            if matches!(keep, Value::Bool(false)) {
                stopped = true;
                break;
            }
        }
    }

    let seconds = began.elapsed().as_secs_f64().max(1e-9);
    let grad_norm = trainer_of(ctx, this)?.last_grad_norm;
    Ok(new_object(
        ctx,
        vec![
            ("steps", Value::number(done as f64)),
            ("stopped", Value::Bool(stopped)),
            ("firstLoss", Value::number(first.unwrap_or(f32::NAN) as f64)),
            ("lastLoss", Value::number(last as f64)),
            (
                "meanLoss",
                Value::number(if done > 0 {
                    sum / done as f64
                } else {
                    f64::NAN
                }),
            ),
            ("stepsPerSecond", Value::number(done as f64 / seconds)),
            ("gradNorm", Value::number(grad_norm as f64)),
        ],
    ))
}

// ── freezing ────────────────────────────────────────────────

/// `freeze(names)` / `freeze()` — stop updating these parameters.
///
/// Their gradients are still computed: `wrt` is baked into the compiled graph at
/// construction, so this changes what moves, not what it costs. To skip the
/// gradient too, leave the parameter out of `wrt` — which is how LoRA freezes a
/// base weight.
fn trainer_freeze(ctx: &mut Context, this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    let all = is_nullish(arg(args, 0));
    let names = if all {
        Vec::new()
    } else {
        to_string_vec(ctx, arg(args, 0))?
    };
    let freeze = magic == 0;
    let slot = trainer_of(ctx, this)?;
    // The resident path gates the update inside the graph with a scalar input
    // per parameter, so freezing costs one f32 rather than a recompile.
    if slot.resident.is_some() {
        let targets: Vec<String> = if all {
            slot.names.clone()
        } else {
            names.clone()
        };
        for name in &targets {
            let known = slot.names.iter().position(|n| n == name);
            match known {
                Some(i) => {
                    slot.frozen[i] = freeze;
                    if let Some(trainer) = slot.resident.as_mut() {
                        trainer.set_frozen(name, freeze);
                    }
                }
                None => {
                    let listed = slot.names.join(", ");
                    return ctx.throw_type(&format!(
                        "freeze('{name}'): not a trainable parameter — this trainer has [{listed}]"
                    ));
                }
            }
        }
        let count = slot.frozen.iter().filter(|f| **f).count();
        return Ok(Value::number(count as f64));
    }
    if all {
        slot.frozen.iter_mut().for_each(|f| *f = freeze);
    } else {
        for name in &names {
            match slot.names.iter().position(|n| n == name) {
                Some(i) => slot.frozen[i] = freeze,
                None => {
                    let known = slot.names.join(", ");
                    return ctx.throw_type(&format!(
                        "freeze('{name}'): not a trainable parameter — this trainer has [{known}]"
                    ));
                }
            }
        }
    }
    let count = slot.frozen.iter().filter(|f| **f).count();
    Ok(Value::number(count as f64))
}

/// Names currently frozen.
fn trainer_frozen(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names: Vec<String> = {
        let slot = trainer_of(ctx, this)?;
        slot.names
            .iter()
            .zip(slot.frozen.iter())
            .filter(|(_, f)| **f)
            .map(|(n, _)| n.clone())
            .collect()
    };
    Ok(new_string_array(ctx, &names))
}

/// `trainable()` — every parameter that receives a gradient, frozen or not.
fn trainer_trainable(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = trainer_of(ctx, this)?.names.clone();
    Ok(new_string_array(ctx, &names))
}

/// Global gradient norm from the most recent step, before clipping.
/// Global gradient norm from the most recent step, before clipping.
///
/// `null` on the resident path: the gradients are consumed inside the graph and
/// never come back, which is the point. Reporting a stale or invented number
/// would be worse than reporting none.
fn trainer_grad_norm(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let slot = trainer_of(ctx, this)?;
    if slot.resident.is_some() {
        return Ok(Value::Null);
    }
    Ok(Value::number(slot.last_grad_norm as f64))
}

/// Whether the optimizer update runs on-device.
///
/// `resident: true` asks for it; the backend may decline (CPU has no buffer
/// handles), and a performance claim nobody can check is not worth making.
fn trainer_is_resident(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let slot = trainer_of(ctx, this)?;
    Ok(Value::Bool(
        slot.resident.as_ref().is_some_and(|t| t.is_resident()),
    ))
}

/// Current weights as `{name: Float32Array}` — feed straight to
/// `compiled.setParams` on an inference graph, or to `rlx.writeGguf`.
fn trainer_params(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let snapshot: Vec<(String, Vec<f32>)> = {
        let slot = trainer_of(ctx, this)?;
        if let Some(trainer) = slot.resident.as_mut() {
            // Reads back from the device buffers when resident.
            let mut out: Vec<(String, Vec<f32>)> = trainer.params().into_iter().collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out.extend(slot.fixed.iter().cloned());
            out
        } else {
            // Every parameter, not just the ones that move. Returning only the
            // trainable set meant `setParams(trainer.params())` on an inference
            // graph left the frozen base weights at zero — a LoRA run whose training
            // loss fell to 0.11 while its evaluation read 14.58%.
            slot.names
                .iter()
                .cloned()
                .zip(slot.values.iter().cloned())
                .chain(slot.fixed.iter().cloned())
                .collect()
        }
    };
    let obj = ctx.new_object();
    for (name, data) in snapshot {
        let array = new_f32_array(ctx, &data);
        ctx.define_value(&obj, &name, array, PropFlags::C_W_E);
    }
    Ok(Value::Object(obj))
}

/// Overwrite the trainable weights — resuming from a checkpoint.
fn trainer_set_params(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let mut incoming: Vec<(String, Vec<f32>)> = Vec::new();
    for name in own_keys(ctx, &bag)? {
        let value = field(ctx, &bag, &name)?;
        let data = to_f32_vec(ctx, &value, &format!("setParams('{name}')"))?;
        incoming.push((name, data));
    }
    let slot = trainer_of(ctx, this)?;
    if slot.resident.is_some() {
        let trainable: Vec<String> = slot.names.clone();
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        let mut applied = 0usize;
        for (name, data) in incoming {
            if trainable.contains(&name) {
                applied += 1;
            }
            map.insert(name, data);
        }
        if let Some(trainer) = slot.resident.as_mut() {
            trainer.set_params(&map);
        }
        for (name, data) in map {
            if !trainable.contains(&name) {
                match slot.fixed.iter_mut().find(|(n, _)| *n == name) {
                    Some(existing) => existing.1 = data,
                    None => slot.fixed.push((name, data)),
                }
            }
        }
        return Ok(Value::number(applied as f64));
    }
    let mut applied = 0usize;
    for (name, data) in incoming {
        match slot.names.iter().position(|n| *n == name) {
            Some(i) => {
                if data.len() != slot.values[i].len() {
                    return ctx.throw_type(&format!(
                        "Trainer.setParams('{name}'): expected {} values, got {}",
                        slot.values[i].len(),
                        data.len()
                    ));
                }
                slot.values[i] = data;
                applied += 1;
            }
            // Not trainable: a fixed param, which goes to the device and into
            // the kept copy so a later checkpoint still has it.
            None => {
                if let Some(compiled) = slot.host() {
                    compiled.set_param(&name, &data);
                }
                match slot.fixed.iter_mut().find(|(n, _)| *n == name) {
                    Some(slot) => slot.1 = data,
                    None => slot.fixed.push((name, data)),
                }
            }
        }
    }
    Ok(Value::number(applied as f64))
}

fn trainer_set_lr(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let lr = to_f32(ctx, arg(args, 0))?;
    let slot = trainer_of(ctx, this)?;
    slot.lr = lr;
    match slot.resident.as_mut() {
        Some(trainer) => trainer.set_lr(lr),
        None => slot.optimizer.set_lr(lr),
    }
    Ok(Value::Undefined)
}

fn trainer_steps(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let slot = trainer_of(ctx, this)?;
    let steps = match slot.resident.as_ref() {
        Some(trainer) => trainer.steps(),
        None => slot.steps,
    };
    Ok(Value::number(steps as f64))
}

fn trainer_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = trainer_of(ctx, this)?.device;
    Ok(new_string(ctx, label))
}

fn trainer_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let slot = trainer_of(ctx, this)?;
        let where_ = match slot.resident.as_ref() {
            Some(t) if t.is_resident() => " update=on-device",
            Some(_) => " update=fused-host",
            None => "",
        };
        let steps = match slot.resident.as_ref() {
            Some(t) => t.steps(),
            None => slot.steps,
        };
        format!(
            "[rlx.Trainer device={} params={} steps={}{}]",
            slot.device,
            slot.names.len(),
            steps,
            where_
        )
    };
    Ok(new_string(ctx, &text))
}

// ── persistence ─────────────────────────────────────────────
//
// Checkpoints are GGUF: `rlx.openGguf` reads them, so does every other rlx
// tool, and a corrupt one is detectable without this crate. Tensors are named
// `param/<name>` (trainable), `fixed/<name>` (no gradient) and `opt/<slot>/<name>`
// (optimizer accumulators); the header carries the step, lr and algorithm.

#[cfg(feature = "gguf")]
const CKPT_VERSION: i64 = 1;

#[cfg(feature = "gguf")]
fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// `save(path)` → the number of tensors written.
///
/// Includes the optimizer's accumulators when the algorithm supports it. When it
/// does not, the header records that and `load` says so, because resuming Adam
/// with its moments reset is a different run, not the same one continued.
#[cfg(feature = "gguf")]
fn trainer_save(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use rlx_gguf::{GgmlType, GgufWriter, MetaValue};

    if let Some(refused) = handle::require_fs(ctx, "Trainer.save")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;

    // The on-disk layout is identical for both paths — `opt/m/<name>` and
    // `opt/v/<name>` — so a checkpoint written by a host run resumes on the
    // resident one and back.
    let (names, shapes, values, frozen, fixed, steps, lr, kind, buffers, opt_step) = {
        let slot = trainer_of(ctx, this)?;
        if let Some(trainer) = slot.resident.as_mut() {
            let params = trainer.params();
            let moments = trainer.moments();
            let opt_step = trainer.steps();
            let mut buffers: Vec<(String, Vec<f32>)> = Vec::with_capacity(moments.len() * 2);
            for (name, m, v) in moments {
                buffers.push((format!("m/{name}"), m));
                buffers.push((format!("v/{name}"), v));
            }
            let values: Vec<Vec<f32>> = slot
                .names
                .iter()
                .map(|n| params.get(n).cloned().unwrap_or_default())
                .collect();
            (
                slot.names.clone(),
                slot.shapes.clone(),
                values,
                slot.frozen.clone(),
                slot.fixed.clone(),
                opt_step,
                slot.lr,
                slot.optimizer_kind.clone(),
                Some(buffers),
                opt_step,
            )
        } else {
            let state = slot.optimizer.state_dict();
            let opt_step = state.as_ref().map(|s| s.step).unwrap_or(0);
            (
                slot.names.clone(),
                slot.shapes.clone(),
                slot.values.clone(),
                slot.frozen.clone(),
                slot.fixed.clone(),
                slot.steps,
                slot.lr,
                slot.optimizer_kind.clone(),
                state.map(|s| s.buffers),
                opt_step,
            )
        }
    };

    let mut writer = GgufWriter::new();
    writer.set_arch("rlx-checkpoint");
    writer.set_meta("rlx.checkpoint.version", MetaValue::I64(CKPT_VERSION));
    writer.set_meta("rlx.checkpoint.step", MetaValue::U64(steps));
    writer.set_meta("rlx.checkpoint.lr", MetaValue::F32(lr));
    writer.set_meta("rlx.checkpoint.optimizer", MetaValue::String(kind));
    writer.set_meta("rlx.checkpoint.optimizer_step", MetaValue::U64(opt_step));
    writer.set_meta(
        "rlx.checkpoint.has_optimizer_state",
        MetaValue::Bool(buffers.is_some()),
    );
    if !names.is_empty() {
        writer.set_meta(
            "rlx.checkpoint.trainable",
            MetaValue::Array(names.iter().cloned().map(MetaValue::String).collect()),
        );
    }
    // GGUF arrays carry an element type, so an empty one is not representable.
    // Absent means "nothing frozen", which is what `load` already defaults to.
    let frozen_names: Vec<MetaValue> = names
        .iter()
        .zip(frozen.iter())
        .filter(|(_, f)| **f)
        .map(|(n, _)| MetaValue::String(n.clone()))
        .collect();
    if !frozen_names.is_empty() {
        writer.set_meta("rlx.checkpoint.frozen", MetaValue::Array(frozen_names));
    }

    let mut written = 0usize;
    let add = |writer: &mut GgufWriter, name: String, shape: Vec<usize>, data: &[f32]| {
        writer.add_tensor_bytes(name, shape, GgmlType::F32, f32_to_bytes(data))
    };
    for ((name, shape), data) in names.iter().zip(shapes.iter()).zip(values.iter()) {
        // A dynamic axis records as 0, which GGUF cannot represent; fall back to
        // the flat length, which is all `load` needs.
        let dims = if shape.iter().all(|d| *d > 0) {
            shape.clone()
        } else {
            vec![data.len()]
        };
        if let Err(e) = add(&mut writer, format!("param/{name}"), dims, data) {
            return ctx.throw_internal(&format!("save: param '{name}': {e}"));
        }
        written += 1;
    }
    for (name, data) in &fixed {
        if let Err(e) = add(&mut writer, format!("fixed/{name}"), vec![data.len()], data) {
            return ctx.throw_internal(&format!("save: fixed '{name}': {e}"));
        }
        written += 1;
    }
    if let Some(state) = &buffers {
        for (key, data) in state {
            if let Err(e) = add(&mut writer, format!("opt/{key}"), vec![data.len()], data) {
                return ctx.throw_internal(&format!("save: optimizer '{key}': {e}"));
            }
            written += 1;
        }
    }

    match writer.write_to_path(&path) {
        Ok(()) => Ok(Value::number(written as f64)),
        Err(e) => ctx.throw_internal(&format!("save('{path}'): {e:#}")),
    }
}

/// `load(path)` → a report of what was restored.
///
/// Refuses a checkpoint whose optimizer differs from this trainer's: Adam
/// moments loaded into Lion would train, badly, with no sign anything was wrong.
/// A shape mismatch is refused for the same reason.
#[cfg(feature = "gguf")]
fn trainer_load(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use rlx_gguf::GgufFile;

    if let Some(refused) = handle::require_fs(ctx, "Trainer.load")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let file = match GgufFile::from_path_mmap(&path) {
        Ok(f) => f,
        Err(e) => return ctx.throw_internal(&format!("load('{path}'): {e:#}")),
    };

    let meta_str = |key: &str| {
        file.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    };
    let saved_kind = meta_str("rlx.checkpoint.optimizer");
    {
        let slot = trainer_of(ctx, this)?;
        if !saved_kind.is_empty() && saved_kind != slot.optimizer_kind {
            let mine = slot.optimizer_kind.clone();
            return ctx.throw_type(&format!(
                "load('{path}'): checkpoint was written by '{saved_kind}' but this trainer                  uses '{mine}'. The accumulators mean different things, so loading them                  would train quietly wrong."
            ));
        }
    }

    let step = file
        .metadata
        .get("rlx.checkpoint.step")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let had_state = file
        .metadata
        .get("rlx.checkpoint.has_optimizer_state")
        .map(|v| matches!(v, rlx_gguf::MetaValue::Bool(true)))
        .unwrap_or(false);

    // Read every tensor up front: `dequant_f32` borrows the file, and the
    // trainer handle must not be borrowed across it.
    let mut params: Vec<(String, Vec<f32>)> = Vec::new();
    let mut fixed: Vec<(String, Vec<f32>)> = Vec::new();
    let mut opt_buffers: Vec<(String, Vec<f32>)> = Vec::new();
    let mut tensor_names: Vec<String> = file.tensors.keys().cloned().collect();
    tensor_names.sort();
    for name in tensor_names {
        let data = match file.dequant_f32(&name) {
            Ok((data, _)) => data,
            Err(e) => return ctx.throw_internal(&format!("load: tensor '{name}': {e}")),
        };
        if let Some(rest) = name.strip_prefix("param/") {
            params.push((rest.to_string(), data));
        } else if let Some(rest) = name.strip_prefix("fixed/") {
            fixed.push((rest.to_string(), data));
        } else if let Some(rest) = name.strip_prefix("opt/") {
            opt_buffers.push((rest.to_string(), data));
        }
    }

    let slot = trainer_of(ctx, this)?;
    let mut restored = 0usize;
    let mut unknown: Vec<String> = Vec::new();
    for (name, data) in params {
        match slot.names.iter().position(|n| *n == name) {
            Some(i) => {
                if slot.values[i].len() != data.len() {
                    let want = slot.values[i].len();
                    return ctx.throw_type(&format!(
                        "load: '{name}' has {} values in the checkpoint but {want} in this                          model — the graphs do not match",
                        data.len()
                    ));
                }
                slot.values[i] = data;
                restored += 1;
            }
            None => unknown.push(name),
        }
    }
    for (name, data) in fixed {
        if let Some(compiled) = slot.host() {
            compiled.set_param(&name, &data);
        }
        match slot.fixed.iter_mut().find(|(n, _)| *n == name) {
            Some(existing) => existing.1 = data,
            None => slot.fixed.push((name, data)),
        }
        restored += 1;
    }

    // Frozen set comes back too, so a resumed run continues the same recipe.
    let frozen_names: Vec<String> = file
        .metadata
        .get("rlx.checkpoint.frozen")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    if slot.resident.is_some() {
        // Group `m/<name>` + `v/<name>` back into per-parameter triples.
        let mut moments: Vec<(String, Vec<f32>, Vec<f32>)> = Vec::new();
        for name in slot.names.clone() {
            let m = opt_buffers
                .iter()
                .find(|(k, _)| *k == format!("m/{name}"))
                .map(|(_, v)| v.clone());
            let v = opt_buffers
                .iter()
                .find(|(k, _)| *k == format!("v/{name}"))
                .map(|(_, v)| v.clone());
            if let (Some(m), Some(v)) = (m, v) {
                moments.push((name, m, v));
            }
        }
        let restored_params: HashMap<String, Vec<f32>> = slot
            .names
            .iter()
            .cloned()
            .zip(slot.values.iter().cloned())
            .collect();
        let optimizer_restored = match slot.resident.as_mut() {
            Some(trainer) => {
                trainer.set_params(&restored_params);
                !moments.is_empty() && trainer.restore_moments(&moments, step)
            }
            None => false,
        };
        for (i, name) in slot.names.clone().iter().enumerate() {
            let held = frozen_names.contains(name);
            slot.frozen[i] = held;
            if let Some(trainer) = slot.resident.as_mut() {
                trainer.set_frozen(name, held);
            }
        }
        slot.steps = step;
        let unknown_v = new_string_array(ctx, &unknown);
        return Ok(new_object(
            ctx,
            vec![
                ("step", Value::number(step as f64)),
                ("restored", Value::number(restored as f64)),
                ("optimizerRestored", Value::Bool(optimizer_restored)),
                ("hadOptimizerState", Value::Bool(had_state)),
                ("unknown", unknown_v),
            ],
        ));
    }

    let optimizer_restored = if opt_buffers.is_empty() {
        false
    } else {
        let opt_step = file
            .metadata
            .get("rlx.checkpoint.optimizer_step")
            .and_then(|v| v.as_u64())
            .unwrap_or(step);
        let state = rlx_optim::OptimizerState {
            step: opt_step,
            buffers: opt_buffers,
        };
        slot.optimizer.load_state_dict(&state)
    };
    slot.steps = step;
    if let Some(lr) = file
        .metadata
        .get("rlx.checkpoint.lr")
        .and_then(|v| match v {
            rlx_gguf::MetaValue::F32(x) => Some(*x),
            rlx_gguf::MetaValue::F64(x) => Some(*x as f32),
            _ => None,
        })
    {
        slot.lr = lr;
        slot.optimizer.set_lr(lr);
    }

    for (i, name) in slot.names.iter().enumerate() {
        slot.frozen[i] = frozen_names.contains(name);
    }

    let unknown_v = new_string_array(ctx, &unknown);
    Ok(new_object(
        ctx,
        vec![
            ("step", Value::number(step as f64)),
            ("restored", Value::number(restored as f64)),
            ("optimizerRestored", Value::Bool(optimizer_restored)),
            ("hadOptimizerState", Value::Bool(had_state)),
            ("unknown", unknown_v),
        ],
    ))
}

// ── installation ────────────────────────────────────────────

/// `freeze` and `unfreeze` differ only in the flag they write.
fn trainer_freeze_all(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    trainer_freeze(ctx, this, args, 0)
}

fn trainer_unfreeze_all(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    trainer_freeze(ctx, this, args, 1)
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_class! {
        ctx, namespace;
        name: "Optimizer",
        class: CLASS_OPTIMIZER,
        ctor: optimizer_new,
        methods: {
            "step" => optimizer_step, 1;
            "setLr" => optimizer_set_lr, 1;
            "toString" => optimizer_to_string, 0;
        }
    };
    js_class! {
        ctx, namespace;
        name: "Trainer",
        class: CLASS_TRAINER,
        ctor: trainer_new,
        methods: {
            "step" => trainer_step, 1;
            "run" => trainer_run, 1;
            "freeze" => trainer_freeze_all, 1;
            "unfreeze" => trainer_unfreeze_all, 1;
            "frozen" => trainer_frozen, 0;
            "trainable" => trainer_trainable, 0;
            "gradNorm" => trainer_grad_norm, 0;
            "isResident" => trainer_is_resident, 0;
            "evaluate" => trainer_evaluate, 1;
            "params" => trainer_params, 0;
            "setParams" => trainer_set_params, 1;
            "setLr" => trainer_set_lr, 1;
            "steps" => trainer_steps, 0;
            "device" => trainer_device, 0;
            "toString" => trainer_to_string, 0;
        }
    };
    #[cfg(feature = "gguf")]
    {
        let proto = handle::prototype_of(ctx, CLASS_TRAINER).expect("just registered");
        ctx.define_method(&proto, "save", trainer_save, 1, 0);
        ctx.define_method(&proto, "load", trainer_load, 1, 0);
    }
}
