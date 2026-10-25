// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training from JavaScript.
//!
//! Two levels. `rlx.Optimizer` is `rlx-optim` — Adam, AdamW, Muon, SOAP, Lion
//! and the rest — driving arrays the script owns. `rlx.Trainer` is the whole
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
//! The logic — host and resident update paths, clipping, freezing, GGUF
//! checkpoints — is `rlx_bind::runtime::train`, shared with pyrlx and the C
//! ABI. This file converts JS values and runs `run`'s callback loop, keeping no
//! handle borrowed while script runs.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::{JsObject, PropFlags};
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::train::{Optimizer, Resident, RunStats, StepItem, Trainer, TrainerConfig};

use crate::api::bind::{lift, report, to_json};
use crate::api::graph::take_graph;
use crate::convert::*;
use crate::handle::{self, CLASS_OPTIMIZER, CLASS_TRAINER};

fn f32_map(ctx: &mut Context, entries: Vec<(String, Vec<f32>)>) -> Value {
    let obj = ctx.new_object();
    for (name, data) in entries {
        let array = new_f32_array(ctx, &data);
        ctx.define_value(&obj, &name, array, PropFlags::C_W_E);
    }
    Value::Object(obj)
}

fn read_f32_map(ctx: &mut Context, bag: &Value, what: &str) -> JsResult<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::new();
    for name in own_keys(ctx, bag)? {
        let value = field(ctx, bag, &name)?;
        let data = to_f32_vec(ctx, &value, &format!("{what}['{name}']"))?;
        out.push((name, data));
    }
    Ok(out)
}

// ── standalone Optimizer ────────────────────────────────────

fn optimizer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Optimizer> {
    handle::borrow_mut(ctx, this, CLASS_OPTIMIZER, "rlx.Optimizer")
}

/// `new rlx.Optimizer("adamw", {lr})` or `new rlx.Optimizer({kind, lr})`.
fn optimizer_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let first = arg(args, 0).clone();
    let (kind, opts) = if first.as_string().is_some() {
        (
            Some(ctx.to_rust_string(&first)?),
            to_json(ctx, arg(args, 1))?,
        )
    } else {
        (None, to_json(ctx, &first)?)
    };
    let opt = lift(ctx, Optimizer::new(kind.as_deref(), &opts))?;
    Ok(handle::wrap(ctx, CLASS_OPTIMIZER, opt))
}

/// `step({name: {param, grad, shape?}})` → `{name: Float32Array}`, updated.
///
/// The arrays passed in are copied — a `Float32Array` is not a buffer Rust can
/// mutate safely while script may still reach it — so the updated values come
/// back as the return value.
fn optimizer_step(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let mut items = Vec::new();
    for name in own_keys(ctx, &bag)? {
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
        let shape_v = field(ctx, &slot, "shape")?;
        let shape = if is_nullish(&shape_v) {
            vec![param.len()]
        } else {
            to_usize_vec(ctx, &shape_v, "shape")?
        };
        items.push(StepItem {
            name,
            shape,
            param,
            grad,
        });
    }
    let r = optimizer_of(ctx, this)?.step(&mut items);
    lift(ctx, r)?;
    let out = items.into_iter().map(|it| (it.name, it.param)).collect();
    Ok(f32_map(ctx, out))
}

fn optimizer_set_lr(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let lr = to_f32(ctx, arg(args, 0))?;
    optimizer_of(ctx, this)?.set_lr(lr);
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
    inner: Trainer,
    /// Reused per-step input buffers — a 400-step loop should allocate once.
    scratch: InputScratch,
}

fn trainer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut TrainerSlot> {
    handle::borrow_mut(ctx, this, CLASS_TRAINER, "rlx.Trainer")
}

/// `new rlx.Trainer(lossGraph, {wrt, init, device, policy, optimizer,
/// clipNorm, resident})`.
fn trainer_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let options = arg(args, 1).clone();
    let device_v = field(ctx, &options, "device")?;
    let device = to_string_or(ctx, &device_v, "cpu")?;
    let wrt_v = field(ctx, &options, "wrt")?;
    let wrt = to_string_vec(ctx, &wrt_v)?;
    let policy_v = field(ctx, &options, "policy")?;
    let policy = crate::api::compile::parse_precision_policy(ctx, &policy_v)?;
    let clip_v = field(ctx, &options, "clipNorm")?;
    let clip_norm = if is_nullish(&clip_v) {
        None
    } else {
        Some(to_f32(ctx, &clip_v)?)
    };
    let optimizer_v = field(ctx, &options, "optimizer")?;
    let optimizer = to_json(ctx, &optimizer_v)?;
    let resident_v = field(ctx, &options, "resident")?;
    let resident_json = to_json(ctx, &resident_v)?;
    let resident = lift(ctx, Resident::from_json(&resident_json))?;
    // Every conversion that can run script happens before the graph moves.
    let init_bag = field(ctx, &options, "init")?;
    let init = read_f32_map(ctx, &init_bag, "init")?;
    let forward = take_graph(ctx, arg(args, 0))?;
    let cfg = TrainerConfig {
        wrt,
        init,
        device,
        optimizer,
        policy,
        clip_norm,
        resident,
    };
    let trainer = lift(ctx, Trainer::new(forward, cfg))?;
    Ok(handle::wrap(
        ctx,
        CLASS_TRAINER,
        TrainerSlot {
            inner: trainer,
            scratch: InputScratch::default(),
        },
    ))
}

/// Load a batch into the trainer's scratch. The scratch is moved out while
/// script may run (a getter in the bag) and put back before the graph runs,
/// so the handle is never borrowed twice.
fn load_batch(ctx: &mut Context, this: &Value, bag: &Value) -> JsResult<()> {
    let slot = trainer_of(ctx, this)?;
    let mut scratch = std::mem::take(&mut slot.scratch);
    let loaded = scratch.load(ctx, bag);
    let slot = trainer_of(ctx, this)?;
    slot.scratch = scratch;
    loaded
}

fn step_loaded(ctx: &mut Context, this: &Value, evaluate: bool) -> JsResult<f32> {
    let slot = trainer_of(ctx, this)?;
    let pairs = slot.scratch.pairs();
    let r = if evaluate {
        slot.inner.evaluate(&pairs)
    } else {
        slot.inner.step(&pairs)
    };
    lift(ctx, r)
}

/// `step({x, y})` → the scalar loss, after one optimizer update.
fn trainer_step(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    load_batch(ctx, this, &bag)?;
    Ok(Value::number(step_loaded(ctx, this, false)? as f64))
}

/// Loss only, no update — for a validation split.
fn trainer_evaluate(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    load_batch(ctx, this, &bag)?;
    Ok(Value::number(step_loaded(ctx, this, true)? as f64))
}

/// `run({steps, batch, lr, onStep, every})` → a summary of the stretch.
///
/// The trainer keeps every piece of state, so this *is* start, pause, resume
/// and stop: return `false` from `onStep` to break out, call `run` again to
/// carry on from exactly where it left off.
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

    let start = trainer_of(ctx, this)?.inner.steps();
    let mut stats = RunStats::new();
    for i in 0..steps {
        let index = Value::number((start + i as u64) as f64);
        // Each callback runs with no handle borrowed, so a callback that
        // reaches back into the trainer sees a consistent object.
        if !is_nullish(&lr_fn) {
            let lr = ctx.call(
                lr_fn.clone(),
                Value::Undefined,
                std::slice::from_ref(&index),
            )?;
            if !is_nullish(&lr) {
                let lr = to_f32(ctx, &lr)?;
                trainer_of(ctx, this)?.inner.set_lr(lr);
            }
        }
        let bag = ctx.call(
            batch_fn.clone(),
            Value::Undefined,
            std::slice::from_ref(&index),
        )?;
        if is_nullish(&bag) {
            // A batch provider returning null means "data exhausted".
            stats.stopped = true;
            break;
        }
        load_batch(ctx, this, &bag)?;
        let loss = step_loaded(ctx, this, false)?;
        stats.record(loss);
        if !is_nullish(&on_step) && stats.report_due(every, i, steps) {
            let keep = ctx.call(
                on_step.clone(),
                Value::Undefined,
                &[index, Value::number(loss as f64)],
            )?;
            // Only an explicit `false` stops: a logging callback that returns
            // nothing should not end the run.
            if matches!(keep, Value::Bool(false)) {
                stats.stopped = true;
                break;
            }
        }
    }
    let grad_norm = trainer_of(ctx, this)?.inner.grad_norm();
    let json = stats.summary(grad_norm);
    Ok(report(ctx, &json))
}

/// `freeze(names?)` / `unfreeze(names?)` → the frozen count. Gradients are
/// still computed; leave a param out of `wrt` to skip it (how LoRA freezes a
/// base weight).
fn trainer_freeze(ctx: &mut Context, this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    let names = if is_nullish(arg(args, 0)) {
        None
    } else {
        Some(to_string_vec(ctx, arg(args, 0))?)
    };
    let r = trainer_of(ctx, this)?
        .inner
        .set_frozen(names.as_deref(), magic == 0);
    let count = lift(ctx, r)?;
    Ok(Value::number(count as f64))
}

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

fn trainer_frozen(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = trainer_of(ctx, this)?.inner.frozen();
    Ok(new_string_array(ctx, &names))
}

/// Every parameter that receives a gradient, frozen or not.
fn trainer_trainable(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = trainer_of(ctx, this)?.inner.trainable().to_vec();
    Ok(new_string_array(ctx, &names))
}

/// Global gradient norm of the last step, before clipping; `null` on the
/// resident path, where the gradients never leave the device.
fn trainer_grad_norm(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(match trainer_of(ctx, this)?.inner.grad_norm() {
        Some(n) => Value::number(n as f64),
        None => Value::Null,
    })
}

/// Whether the optimizer update actually runs on-device.
fn trainer_is_resident(
    ctx: &mut Context,
    this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    Ok(Value::Bool(trainer_of(ctx, this)?.inner.is_resident()))
}

/// Every weight as `{name: Float32Array}` — feed straight to
/// `compiled.setParams` on an inference graph, or to `rlx.writeGguf`.
fn trainer_params(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let snapshot = trainer_of(ctx, this)?.inner.params();
    Ok(f32_map(ctx, snapshot))
}

/// Overwrite weights — resuming from a checkpoint → how many were trainable.
fn trainer_set_params(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let incoming = read_f32_map(ctx, &bag, "setParams")?;
    let r = trainer_of(ctx, this)?.inner.set_params(incoming);
    let applied = lift(ctx, r)?;
    Ok(Value::number(applied as f64))
}

fn trainer_set_lr(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let lr = to_f32(ctx, arg(args, 0))?;
    trainer_of(ctx, this)?.inner.set_lr(lr);
    Ok(Value::Undefined)
}

fn trainer_steps(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(Value::number(trainer_of(ctx, this)?.inner.steps() as f64))
}

fn trainer_device(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = trainer_of(ctx, this)?.inner.device_label();
    Ok(new_string(ctx, label))
}

fn trainer_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let t = &trainer_of(ctx, this)?.inner;
        format!(
            "[rlx.Trainer device={} params={} steps={}{}]",
            t.device_label(),
            t.trainable().len(),
            t.steps(),
            t.update_label()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── persistence ─────────────────────────────────────────────

/// `save(path)` → the number of tensors written (GGUF).
#[cfg(feature = "gguf")]
fn trainer_save(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "Trainer.save")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let r = trainer_of(ctx, this)?.inner.save(&path);
    let written = lift(ctx, r)?;
    Ok(Value::number(written as f64))
}

/// `load(path)` → `{step, restored, optimizerRestored, hadOptimizerState, unknown}`.
#[cfg(feature = "gguf")]
fn trainer_load(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "Trainer.load")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let r = trainer_of(ctx, this)?.inner.load(&path);
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

/// `rlx.Trainer.fromPackage(path, {function, dims, loss, train, optimizer,
/// device, policy, clipNorm, resident})`: a function of an `.rlxp` trained,
/// whatever the model is.
#[cfg(feature = "models")]
fn trainer_from_package(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "Trainer.fromPackage")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let mut opts = to_json(ctx, arg(args, 1))?;
    // The JS spelling of the trainer's own option.
    if let Some(v) = opts.get("clipNorm").cloned() {
        opts["clip_norm"] = v;
    }
    let trainer = lift(ctx, Trainer::from_package(&path, &opts))?;
    Ok(handle::wrap(
        ctx,
        CLASS_TRAINER,
        TrainerSlot {
            inner: trainer,
            scratch: InputScratch::default(),
        },
    ))
}

/// `trainer.savePackage(path, name?)`: what a package trainer trained,
/// written as a package (an adapter, an SAE, or the package itself).
#[cfg(feature = "models")]
fn trainer_save_package(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "Trainer.savePackage")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let name = if is_nullish(arg(args, 1)) {
        None
    } else {
        Some(ctx.to_rust_string(arg(args, 1))?)
    };
    let r = trainer_of(ctx, this)?
        .inner
        .save_package(&path, name.as_deref());
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

// ── installation ────────────────────────────────────────────

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
    #[cfg(feature = "models")]
    {
        js_statics! {
            ctx, namespace, "Trainer";
            "fromPackage" => trainer_from_package, 2;
        };
        let proto = handle::prototype_of(ctx, CLASS_TRAINER).expect("just registered");
        ctx.define_method(&proto, "savePackage", trainer_save_package, 2, 0);
    }
}
