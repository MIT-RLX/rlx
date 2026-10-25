// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Optimizer` and `Trainer` (feature `training`).

use std::ffi::{c_char, c_void};

use rlx_bind::ApiError;
use rlx_bind::runtime::train::{Optimizer, Resident, RunStats, StepItem, Trainer, TrainerConfig};

use crate::ffi::{RlxTensorMap, arg, borrow, guard, handle, out, release};
use crate::graph::{RlxGraph, take_graph};

// ── Optimizer ───────────────────────────────────────────────

/// One of rlx-optim's algorithms over arrays the caller owns.
pub struct RlxOptimizer(Optimizer);

/// An optimizer. `kind` (`NULL` = `options.kind`, else `"adamw"`): `sgd`,
/// `adam`, `adamw`, `lion`, `muon`, `radam`, `nadamw`, `lamb`, `mars`, `soap`,
/// `sophia`, `qhadamw`, `adafactor`. `options_json`: `{"lr": 1e-3,
/// "weight_decay": 0.01, …}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_optimizer_new(
    kind: *const c_char,
    options_json: *const c_char,
    out: *mut *mut RlxOptimizer,
) -> i32 {
    guard(|| unsafe {
        let kind = arg::opt_str(kind, "rlx_optimizer_new kind")?;
        let opts = arg::json(options_json, "rlx_optimizer_new options")?;
        let o = Optimizer::new(kind.as_deref(), &opts)?;
        out::put(out, handle(RlxOptimizer(o)), "rlx_optimizer_new out")
    })
}

/// Free an optimizer (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_optimizer_free(o: *mut RlxOptimizer) {
    unsafe { release(o) }
}

/// One update over `n` parameters, **in place**: `params[i]` (length
/// `lens[i]`) moves along `grads[i]`. `shapes`/`ndims` (`NULL` = flat) give
/// each parameter's dims, which matrix optimizers (Muon, SOAP) need. The
/// optimizer keys its state by name, so pass the same names every step.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_optimizer_step(
    o: *mut RlxOptimizer,
    names: *const *const c_char,
    params: *const *mut f32,
    grads: *const *const f32,
    lens: *const usize,
    shapes: *const *const usize,
    ndims: *const usize,
    n: usize,
) -> i32 {
    guard(|| unsafe {
        let what = "rlx_optimizer_step";
        let names = arg::strs(names, n, what)?;
        let params = arg::slice(params, n, what)?;
        let grads = arg::slice(grads, n, what)?;
        let lens = arg::slice(lens, n, what)?;
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            let shape = if shapes.is_null() || ndims.is_null() {
                vec![lens[i]]
            } else {
                let s = arg::slice(shapes, n, what)?;
                let d = arg::slice(ndims, n, what)?;
                arg::vec(s[i], d[i], what)?
            };
            items.push(StepItem {
                name: names[i].clone(),
                shape,
                param: arg::vec(params[i] as *const f32, lens[i], what)?,
                grad: arg::vec(grads[i], lens[i], what)?,
            });
        }
        borrow(o, "optimizer")?.0.step(&mut items)?;
        for (i, it) in items.iter().enumerate() {
            if lens[i] > 0 {
                std::ptr::copy_nonoverlapping(it.param.as_ptr(), params[i], lens[i]);
            }
        }
        Ok(())
    })
}

/// Set the learning rate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_optimizer_set_lr(o: *mut RlxOptimizer, lr: f32) -> i32 {
    guard(|| unsafe {
        borrow(o, "optimizer")?.0.set_lr(lr);
        Ok(())
    })
}

// ── Trainer ─────────────────────────────────────────────────

/// Differentiate-once / compile-once training over a scalar-loss graph.
pub struct RlxTrainer(Trainer);

/// One step's inputs, filled by an `RlxBatchFn` with `rlx_batch_set`.
pub struct RlxBatch {
    entries: Vec<(String, Vec<f32>)>,
    live: usize,
}

/// Supply step `step`'s inputs into `batch`: return `1` when filled, `0` when
/// the data is exhausted, a negative value to abort the run.
pub type RlxBatchFn =
    Option<unsafe extern "C" fn(user: *mut c_void, step: u64, batch: *mut RlxBatch) -> i32>;

/// Schedule the learning rate: write `*lr` and return `1`, or return `0` to
/// keep the current one.
pub type RlxLrFn = Option<unsafe extern "C" fn(user: *mut c_void, step: u64, lr: *mut f32) -> i32>;

/// Called after a step with its loss: return `0` to stop early, `1` to go on.
pub type RlxStepFn = Option<unsafe extern "C" fn(user: *mut c_void, step: u64, loss: f32) -> i32>;

/// Set one input of a batch (copied; buffers are reused across steps).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_batch_set(
    b: *mut RlxBatch,
    name: *const c_char,
    data: *const f32,
    len: usize,
) -> i32 {
    guard(|| unsafe {
        let b = borrow(b, "batch")?;
        let name = arg::str(name, "rlx_batch_set name")?;
        let data = arg::slice(data, len, "rlx_batch_set data")?;
        match b.entries[..b.live].iter_mut().find(|(n, _)| n == name) {
            Some((_, buf)) => {
                buf.clear();
                buf.extend_from_slice(data);
            }
            None => {
                if b.live < b.entries.len() {
                    let slot = &mut b.entries[b.live];
                    slot.0.clear();
                    slot.0.push_str(name);
                    slot.1.clear();
                    slot.1.extend_from_slice(data);
                } else {
                    b.entries.push((name.to_string(), data.to_vec()));
                }
                b.live += 1;
            }
        }
        Ok(())
    })
}

/// A trainer over `graph` (consumed). `wrt` names the `n_wrt` parameters to
/// train; `init_*` gives every parameter's initial values (those not in `wrt`
/// are uploaded once and never updated). `options_json` (`NULL` = defaults):
/// `{"device": "cpu", "optimizer": {"kind": "adamw", "lr": 1e-3}, "policy":
/// "mixed", "clip_norm": 1.0, "resident": true}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_new(
    graph: *mut RlxGraph,
    wrt: *const *const c_char,
    n_wrt: usize,
    init_names: *const *const c_char,
    init_data: *const *const f32,
    init_lens: *const usize,
    n_init: usize,
    options_json: *const c_char,
    out: *mut *mut RlxTrainer,
) -> i32 {
    guard(|| unsafe {
        let what = "rlx_trainer_new";
        let wrt = arg::strs(wrt, n_wrt, what)?;
        let init = arg::inputs(init_names, init_data, init_lens, n_init, what)?
            .into_iter()
            .map(|(n, d)| (n.to_string(), d.to_vec()))
            .collect();
        let opts = arg::json(options_json, what)?;
        let o = rlx_bind::runtime::opts::Opts::new("trainer", &opts)?;
        let device = o.str("device")?.unwrap_or("cpu").to_string();
        let optimizer = o.get("optimizer").cloned().unwrap_or_default();
        let policy = match o.get("policy") {
            Some(p) => rlx_bind::runtime::policy::parse(p)?,
            None => None,
        };
        let clip_norm = o.f64("clip_norm")?.map(|v| v as f32);
        let resident =
            Resident::from_json(o.get("resident").unwrap_or(&rlx_bind::runtime::Value::Null))?;
        o.finish()?;
        let g = take_graph(graph, "rlx_trainer_new graph")?;
        let t = Trainer::new(
            g,
            TrainerConfig {
                wrt,
                init,
                device,
                optimizer,
                policy,
                clip_norm,
                resident,
            },
        )?;
        out::put(out, handle(RlxTrainer(t)), "rlx_trainer_new out")
    })
}

/// Free a trainer (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_free(t: *mut RlxTrainer) {
    unsafe { release(t) }
}

/// One optimizer update on the `n` inputs; writes the loss.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_step(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    loss: *mut f32,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_trainer_step inputs")?;
        let l = borrow(t, "trainer")?.0.step(&inputs)?;
        out::put(loss, l, "rlx_trainer_step loss")
    })
}

/// The loss on the `n` inputs, no update.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_evaluate(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    loss: *mut f32,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_trainer_evaluate inputs")?;
        let l = borrow(t, "trainer")?.0.evaluate(&inputs)?;
        out::put(loss, l, "rlx_trainer_evaluate loss")
    })
}

/// Run up to `steps` steps: `batch` supplies each step's inputs (required),
/// `lr` schedules the learning rate and `on_step` sees each loss (both may be
/// `NULL`; `on_step` runs every `every` steps and on the last). `user` is
/// passed to every callback. Writes the summary JSON
/// (`{steps, stopped, first_loss, last_loss, mean_loss, steps_per_second, grad_norm}`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_run(
    t: *mut RlxTrainer,
    steps: usize,
    batch: RlxBatchFn,
    lr: RlxLrFn,
    on_step: RlxStepFn,
    every: usize,
    user: *mut c_void,
    summary: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let batch = batch.ok_or_else(|| ApiError::type_("rlx_trainer_run: `batch` is required"))?;
        let trainer = &mut borrow(t, "trainer")?.0;
        let start = trainer.steps();
        let mut stats = RunStats::new();
        let mut b = RlxBatch {
            entries: Vec::new(),
            live: 0,
        };
        for i in 0..steps {
            let index = start + i as u64;
            if let Some(f) = lr {
                let mut value = 0.0f32;
                if f(user, index, &mut value) == 1 {
                    trainer.set_lr(value);
                }
            }
            b.live = 0;
            match batch(user, index, &mut b) {
                1 => {}
                0 => {
                    stats.stopped = true;
                    break;
                }
                rc => {
                    return Err(ApiError::runtime(format!(
                        "rlx_trainer_run: batch callback returned {rc}"
                    )));
                }
            }
            let pairs: Vec<(&str, &[f32])> = b.entries[..b.live]
                .iter()
                .map(|(n, d)| (n.as_str(), d.as_slice()))
                .collect();
            let loss = trainer.step(&pairs)?;
            stats.record(loss);
            if let Some(f) = on_step
                && stats.report_due(every, i, steps)
                && f(user, index, loss) == 0
            {
                stats.stopped = true;
                break;
            }
        }
        out::json(
            summary,
            &stats.summary(trainer.grad_norm()),
            "rlx_trainer_run summary",
        )
    })
}

unsafe fn set_frozen(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    n: usize,
    freeze: bool,
    frozen_count: *mut usize,
    what: &str,
) -> rlx_bind::ApiResult<()> {
    let names = if names.is_null() {
        None
    } else {
        Some(unsafe { arg::strs(names, n, what) }?)
    };
    let count = unsafe { borrow(t, "trainer") }?
        .0
        .set_frozen(names.as_deref(), freeze)?;
    unsafe { out::put(frozen_count, count, what) }
}

/// Stop updating the `n` named params — every trainable param when `names` is
/// `NULL`. Writes how many are frozen now.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_freeze(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    n: usize,
    frozen_count: *mut usize,
) -> i32 {
    guard(|| unsafe { set_frozen(t, names, n, true, frozen_count, "rlx_trainer_freeze") })
}

/// Resume updating the `n` named params — every param when `names` is `NULL`.
/// Writes how many are still frozen.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_unfreeze(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    n: usize,
    frozen_count: *mut usize,
) -> i32 {
    guard(|| unsafe { set_frozen(t, names, n, false, frozen_count, "rlx_trainer_unfreeze") })
}

/// The frozen params, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_frozen(t: *mut RlxTrainer, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names = borrow(t, "trainer")?.0.frozen();
        out::strings(out, &names, "rlx_trainer_frozen out")
    })
}

/// Every param that receives a gradient, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_trainable(t: *mut RlxTrainer, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let names = borrow(t, "trainer")?.0.trainable().to_vec();
        out::strings(out, &names, "rlx_trainer_trainable out")
    })
}

/// The last step's global gradient norm; `*known` is `false` on the resident
/// path, where the gradients never leave the device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_grad_norm(
    t: *mut RlxTrainer,
    norm: *mut f32,
    known: *mut bool,
) -> i32 {
    guard(|| unsafe {
        let g = borrow(t, "trainer")?.0.grad_norm();
        out::put(norm, g.unwrap_or(f32::NAN), "rlx_trainer_grad_norm norm")?;
        out::put(known, g.is_some(), "rlx_trainer_grad_norm known")
    })
}

/// Whether the optimizer update actually runs on-device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_is_resident(t: *mut RlxTrainer) -> bool {
    unsafe { t.as_ref() }.is_some_and(|t| t.0.is_resident())
}

/// Every weight, trainable and fixed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_params(
    t: *mut RlxTrainer,
    out: *mut *mut RlxTensorMap,
) -> i32 {
    guard(|| unsafe {
        let params = borrow(t, "trainer")?.0.params();
        out::put(
            out,
            handle(RlxTensorMap::new(params)),
            "rlx_trainer_params out",
        )
    })
}

/// Overwrite weights; writes how many were trainable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_set_params(
    t: *mut RlxTrainer,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    applied: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let incoming = arg::inputs(names, data, lens, n, "rlx_trainer_set_params")?
            .into_iter()
            .map(|(n, d)| (n.to_string(), d.to_vec()))
            .collect();
        let count = borrow(t, "trainer")?.0.set_params(incoming)?;
        out::put(applied, count, "rlx_trainer_set_params out")
    })
}

/// Set the learning rate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_set_lr(t: *mut RlxTrainer, lr: f32) -> i32 {
    guard(|| unsafe {
        borrow(t, "trainer")?.0.set_lr(lr);
        Ok(())
    })
}

/// Steps taken so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_steps(t: *mut RlxTrainer, steps: *mut u64) -> i32 {
    guard(|| unsafe {
        let n = borrow(t, "trainer")?.0.steps();
        out::put(steps, n, "rlx_trainer_steps out")
    })
}

/// The trainer's device label.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_device(t: *mut RlxTrainer, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let label = borrow(t, "trainer")?.0.device_label();
        out::string(out, label, "rlx_trainer_device out")
    })
}

/// Train a function of the `.rlxp` package at `path`, whatever the model.
/// `options_json` (`NULL` = `{}`): `{function, dims, loss, train, optimizer,
/// device, policy, clip_norm, resident}`, over the package's own `train`.
#[cfg(feature = "models")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_from_package(
    path: *const c_char,
    options_json: *const c_char,
    out: *mut *mut RlxTrainer,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_trainer_from_package path")?;
        let opts = arg::json(options_json, "rlx_trainer_from_package options")?;
        let t = Trainer::from_package(path, &opts)?;
        out::put(out, handle(RlxTrainer(t)), "rlx_trainer_from_package out")
    })
}

/// Write what a package trainer trained at `path`: LoRA factors as an
/// adapter package (`name`, `NULL` = "trained", its name), an SAE as its own
/// package, else the trained package.
#[cfg(feature = "models")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_save_package(
    t: *mut RlxTrainer,
    path: *const c_char,
    name: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_trainer_save_package path")?;
        let name = arg::opt_str(name, "rlx_trainer_save_package name")?;
        borrow(t, "trainer")?.0.save_package(path, name.as_deref())
    })
}

/// Write a GGUF checkpoint (the layout rlx-js and pyrlx write); writes the
/// tensor count.
#[cfg(feature = "gguf")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_save(
    t: *mut RlxTrainer,
    path: *const c_char,
    written: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_trainer_save path")?;
        let n = borrow(t, "trainer")?.0.save(path)?;
        out::put(written, n, "rlx_trainer_save out")
    })
}

/// Restore a checkpoint; the report as JSON (`{step, restored,
/// optimizer_restored, had_optimizer_state, unknown}`).
#[cfg(feature = "gguf")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_trainer_load(
    t: *mut RlxTrainer,
    path: *const c_char,
    report: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_trainer_load path")?;
        let r = borrow(t, "trainer")?.0.load(path)?;
        out::json(report, &r, "rlx_trainer_load out")
    })
}
