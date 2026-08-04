// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! JAX-shaped composable function transforms.
//!
//! Where [`Tensor`] is the eager, NumPy-style surface, [`Func`] is the
//! *functional* one: a pure function of named inputs, traced once into a
//! graph. The transforms ([`Func::grad`], [`Func::vmap`], [`Func::jvp`],
//! [`Func::hvp`]) each take a `Func` and return a `Func`, so they compose —
//! exactly like `jax.jit(jax.vmap(jax.grad(f)))`:
//!
//! ```ignore
//! use rlx_tensor::{Func, shape};
//!
//! // f(x) = sum(x * x)
//! let f = Func::new("f", |s| {
//!     let x = s.input("x", shape![3]);
//!     (&x * &x).sum([0], false)
//! });
//!
//! // ∂f/∂x = 2x, then batch it over 4 rows — one fused graph.
//! let batched_grad = f.grad(&["x"]).vmap(&["x"], 4);
//! let g = batched_grad.run(&[("x", &xs)]); // with the `eval` feature
//! ```

use rlx_ir::Graph;

use crate::{GraphScope, Tensor};

/// A function of named inputs and parameters, traced into a graph. Unit of
/// composition for the [`grad`](Func::grad)/[`vmap`](Func::vmap)/
/// [`jvp`](Func::jvp)/[`hvp`](Func::hvp) transforms.
///
/// **Inputs vs parameters.** Declare per-call data with
/// [`GraphScope::input`] and trainable weights with [`GraphScope::param`].
/// Bind weight values once with [`with_param`](Func::with_param); they are
/// applied on every run and survive transforms — so `f.grad(&["w"])` gives
/// gradients w.r.t. the weights (training), runnable with just the inputs.
#[derive(Clone, Debug)]
pub struct Func {
    graph: Graph,
    /// Bound parameter values (`name -> data`), applied before each run.
    params: Vec<(String, Vec<f32>)>,
}

impl Func {
    /// Trace a single-output function. The closure builds the body from inputs
    /// declared via [`GraphScope::input`]; the returned tensor is the output.
    pub fn new(name: impl Into<String>, build: impl FnOnce(&mut GraphScope) -> Tensor) -> Self {
        Self::multi(name, |s| vec![build(s)])
    }

    /// Trace a multi-output function (the closure returns all outputs in order).
    pub fn multi(
        name: impl Into<String>,
        build: impl FnOnce(&mut GraphScope) -> Vec<Tensor>,
    ) -> Self {
        let mut scope = GraphScope::new(name);
        let outs = build(&mut scope);
        scope.set_outputs(outs);
        Self {
            graph: scope.finish(),
            params: Vec::new(),
        }
    }

    /// Wrap an already-built graph (its outputs are this function's outputs).
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph,
            params: Vec::new(),
        }
    }

    /// Bind a parameter (weight) value by name. Overwrites any prior binding
    /// for that name. Builder-style: chain several, then `run`/`grad`/`jit`.
    pub fn with_param(mut self, name: impl Into<String>, data: impl Into<Vec<f32>>) -> Self {
        let name = name.into();
        let data = data.into();
        if let Some(slot) = self.params.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = data;
        } else {
            self.params.push((name, data));
        }
        self
    }

    /// The underlying graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Number of outputs this function produces.
    pub fn output_count(&self) -> usize {
        self.graph.outputs.len()
    }

    /// The currently bound value of a parameter, if any. After
    /// [`train_step`](Func::train_step) this is the updated weight.
    pub fn param_binding(&self, name: &str) -> Option<&[f32]> {
        self.params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// The names of every trainable parameter (`Op::Param`) in the graph, in
    /// declaration order — exactly the `wrt` list a full-model training step
    /// needs, so you never hand-maintain it. Powers [`init_params`], the
    /// `*_all` training steps, and checkpoint round-trips.
    pub fn param_names(&self) -> Vec<String> {
        self.graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                rlx_ir::Op::Param { name } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// The static shape (dims) of a parameter by name, or `None` if there is no
    /// such param. A dynamic axis reads as `0`.
    pub fn param_shape_of(&self, name: &str) -> Option<Vec<usize>> {
        self.graph.nodes().iter().find_map(|n| match &n.op {
            rlx_ir::Op::Param { name: pn } if pn == name => Some(
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        rlx_ir::Dim::Static(s) => *s,
                        rlx_ir::Dim::Dynamic(_) => 0,
                    })
                    .collect(),
            ),
            _ => None,
        })
    }

    /// Bind every declared parameter by calling `init(name, &static_dims)` and
    /// using the returned `Vec<f32>` — one call seeds a whole model's weights,
    /// instead of a [`with_param`](Func::with_param) per tensor. Overwrites any
    /// existing bindings. Builder-style: chain before `train_step`.
    ///
    /// ```ignore
    /// let model = Func::from_graph(rlx! { … }).init_params(|name, dims| {
    ///     if name.ends_with(".bias") { vec![0.0; dims.iter().product()] }
    ///     else { he_init(dims) }
    /// });
    /// ```
    pub fn init_params(mut self, mut init: impl FnMut(&str, &[usize]) -> Vec<f32>) -> Self {
        let specs: Vec<(String, Vec<usize>)> = self
            .param_names()
            .into_iter()
            .map(|n| {
                let dims = self.param_shape_of(&n).unwrap_or_default();
                (n, dims)
            })
            .collect();
        for (name, dims) in specs {
            let data = init(&name, &dims);
            self = self.with_param(name, data);
        }
        self
    }

    /// Seed every parameter with i.i.d. Gaussian noise `N(0, stddev²)` from a
    /// deterministic SplitMix64 + Box–Muller stream keyed by `seed`
    /// (reproducible, no `rand` dependency). A one-liner for a from-scratch
    /// model; for per-tensor schemes (zeroed biases, fan-in scaling) pass your
    /// own closure to [`init_params`](Func::init_params).
    pub fn init_randn(self, seed: u64, stddev: f32) -> Self {
        // SplitMix64 → uniform in [0, 1).
        fn u01(state: &mut u64) -> f64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = *state;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            (x >> 11) as f64 / (1u64 << 53) as f64
        }
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        self.init_params(move |_name, dims| {
            let n: usize = dims.iter().map(|&d| d.max(1)).product();
            (0..n)
                .map(|_| {
                    let u1 = u01(&mut state).max(1e-12);
                    let u2 = u01(&mut state);
                    let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                    (g as f32) * stddev
                })
                .collect()
        })
    }

    /// Carry the bound params onto a transformed graph.
    #[cfg(feature = "autodiff")]
    fn derive(&self, graph: Graph) -> Func {
        Func {
            graph,
            params: self.params.clone(),
        }
    }

    /// Resolve `Op::Input` / `Op::Param` names to node ids, in request order.
    /// Differentiating w.r.t. a `Param` name is how training gradients work.
    #[cfg(feature = "autodiff")]
    fn node_ids_by_name(&self, names: &[&str]) -> Vec<rlx_ir::NodeId> {
        names
            .iter()
            .map(|want| {
                self.graph
                    .nodes()
                    .iter()
                    .find_map(|n| match &n.op {
                        rlx_ir::Op::Input { name } | rlx_ir::Op::Param { name } if name == want => {
                            Some(n.id)
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| panic!("Func: no input or param named {want:?}"))
            })
            .collect()
    }
}

/// Composable transforms (require `rlx_autodiff`; enabled by `grad` /
/// `transforms` / `autodiff`).
#[cfg(feature = "autodiff")]
impl Func {
    /// Reverse-mode gradient w.r.t. the named inputs. The first output is
    /// treated as the (scalar) loss; the unit seed is baked in, so the result
    /// is a function `inputs -> [∂loss/∂wrt …]` runnable with no extra feeds.
    pub fn grad(&self, wrt: &[&str]) -> Func {
        let ids = self.node_ids_by_name(wrt);
        let mut bwd = rlx_autodiff::grad(&self.graph, &ids);
        crate::tensor::bake_unit_seed(&mut bwd);
        self.derive(bwd)
    }

    /// Like [`grad`](Func::grad) but the result emits the loss **and** the
    /// gradients: outputs `[loss, ∂loss/∂wrt[0], …]`. The JAX `value_and_grad`.
    pub fn value_and_grad(&self, wrt: &[&str]) -> Func {
        let ids = self.node_ids_by_name(wrt);
        let mut bwd = rlx_autodiff::grad_with_loss(&self.graph, &ids);
        crate::tensor::bake_unit_seed(&mut bwd);
        self.derive(bwd)
    }

    /// [`value_and_grad`](Func::value_and_grad) w.r.t. **every** parameter in
    /// the graph — no hand-maintained name list. Outputs `[loss, ∂loss/∂p …]`
    /// ordered like [`param_names`](Func::param_names).
    pub fn value_and_grad_all(&self) -> Func {
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.value_and_grad(&refs)
    }

    /// [`grad`](Func::grad) w.r.t. every parameter in the graph.
    pub fn grad_all(&self) -> Func {
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.grad(&refs)
    }

    /// Vectorize over a leading batch axis on the named inputs (`out_axes = 0`).
    /// Inputs not listed are shared across the batch.
    pub fn vmap(&self, batched: &[&str], batch_size: usize) -> Func {
        self.derive(rlx_autodiff::vmap(&self.graph, batched, batch_size))
    }

    /// Forward-mode Jacobian-vector product w.r.t. the named inputs/params. The
    /// returned function gains a tangent input per `tangent_for` entry.
    pub fn jvp(&self, tangent_for: &[&str]) -> Func {
        let ids = self.node_ids_by_name(tangent_for);
        self.derive(rlx_autodiff::jvp(&self.graph, &ids))
    }

    /// Hessian-vector product w.r.t. the named inputs/params.
    pub fn hvp(&self, wrt: &[&str]) -> Func {
        let ids = self.node_ids_by_name(wrt);
        self.derive(rlx_autodiff::hvp(&self.graph, &ids))
    }
}

/// Execution (requires the `eval` feature).
#[cfg(feature = "eval")]
impl Func {
    /// Compile + run with named inputs, one `Vec<f32>` per output. The device
    /// is auto-selected (fastest compiled-in backend that can run the graph).
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_on(rlx_runtime::fastest_device_for(&self.graph), inputs)
    }

    /// Compile + run on an explicit device. The compiled graph is cached, so
    /// repeated `run` calls with different inputs recompile nothing — a real
    /// `jit`. Bound params are applied before each run.
    pub fn run_on(&self, device: rlx_runtime::Device, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let compiled = crate::cache::compiled(&self.graph, device);
        let mut c = compiled.borrow_mut();
        for (name, data) in &self.params {
            c.set_param(name, data);
        }
        c.run(inputs)
    }

    /// Compile now and return a callable that holds the compiled graph, so
    /// each [`Jitted::run`] skips even the cache lookup. The JAX-shaped
    /// `jit(f)`; composes after the other transforms:
    /// `f.grad(&["x"]).vmap(&["x"], 4).jit()`.
    pub fn jit(&self) -> Jitted {
        self.jit_on(rlx_runtime::fastest_device_for(&self.graph))
    }

    /// [`Func::jit`] targeting an explicit device.
    pub fn jit_on(&self, device: rlx_runtime::Device) -> Jitted {
        Jitted {
            compiled: crate::cache::compiled(&self.graph, device),
            params: self.params.clone(),
        }
    }
}

/// Training (requires the `optim` feature: autodiff + eval + `rlx_optim`).
#[cfg(feature = "optim")]
impl Func {
    /// One optimization step. Computes value + gradients over `wrt` (which must
    /// name bound params), applies `opt` to each param, and returns the updated
    /// function plus the loss. Drive a loop by reassigning the returned `Func`:
    ///
    /// ```ignore
    /// let mut opt = Sgd::new(0.1);
    /// for _ in 0..100 {
    ///     let (next, loss) = model.train_step(&mut opt, &["w"], &[("x", &xs)]);
    ///     model = next;
    /// }
    /// ```
    /// [`train_step`](Func::train_step) with a learning-rate schedule: sets
    /// `opt`'s LR to `schedule.lr_at(step)` before stepping. Drive a loop by
    /// passing the iteration index as `step`.
    pub fn train_step_at(
        &self,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        wrt: &[&str],
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        self.train_step(opt, wrt, inputs)
    }

    pub fn train_step(
        &self,
        opt: &mut dyn rlx_optim::Optimizer,
        wrt: &[&str],
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        let outputs = self.value_and_grad(wrt).run(inputs);
        self.apply_optimizer_step(opt, wrt, &outputs)
    }

    /// [`train_step`](Func::train_step) pinned to an explicit `device` instead
    /// of the auto-selected fastest backend — the one call you need to train a
    /// whole run on, say, Metal (no hand-rolled `value_and_grad().run_on()` +
    /// optimizer loop).
    pub fn train_step_on(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        wrt: &[&str],
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        let outputs = self.value_and_grad(wrt).run_on(device, inputs);
        self.apply_optimizer_step(opt, wrt, &outputs)
    }

    /// [`train_step_on`](Func::train_step_on) with a learning-rate schedule.
    pub fn train_step_at_on(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        wrt: &[&str],
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        self.train_step_on(device, opt, wrt, inputs)
    }

    /// [`train_step`](Func::train_step) w.r.t. **every** parameter — the `wrt`
    /// list is [`param_names`](Func::param_names), so the call is just
    /// `(opt, inputs)`.
    pub fn train_step_all(
        &self,
        opt: &mut dyn rlx_optim::Optimizer,
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.train_step(opt, &refs, inputs)
    }

    /// All-parameter [`train_step_at`](Func::train_step_at).
    pub fn train_step_all_at(
        &self,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        self.train_step_all(opt, inputs)
    }

    /// All-parameter [`train_step_on`](Func::train_step_on).
    pub fn train_step_all_on(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.train_step_on(device, opt, &refs, inputs)
    }

    /// The full-model, device-pinned, scheduled training step — the whole loop
    /// body for a from-scratch model is `model.train_step_all_at_on(dev, &mut
    /// opt, &sched, step, feed)`.
    pub fn train_step_all_at_on(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        self.train_step_all_on(device, opt, inputs)
    }

    /// [`train_step_all_at_on`](Func::train_step_all_at_on) with **global-L2-norm
    /// gradient clipping** to `max_grad_norm` — the standard cure for the loss
    /// spikes that can blow a run up to NaN late in training (~1.0 is typical).
    /// A non-positive `max_grad_norm` disables clipping (identical to the
    /// unclipped variant).
    pub fn train_step_all_at_on_clipped(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        max_grad_norm: f32,
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut outputs = self.value_and_grad(&refs).run_on(device, inputs);
        let scale = grad_clip_scale(&outputs, max_grad_norm);
        if scale < 1.0 {
            for g in outputs.iter_mut().skip(1) {
                for x in g.iter_mut() {
                    *x *= scale;
                }
            }
        }
        self.apply_optimizer_step(opt, &refs, &outputs)
    }

    /// **Quantization-aware** full-model step: each parameter is passed through
    /// `quant` (an in-place quantizer) before the forward, and the gradient is
    /// taken at that quantized point — but applied to the f32 **master** weights
    /// (a straight-through estimator). This trains the model at whatever emulated
    /// precision `quant` imposes, on any backend. Pair with [`crate::lowp`] for
    /// `fXmYeZ` / nvf4 / f8 / bf8 formats:
    ///
    /// ```ignore
    /// let (e, m, max) = rlx_tensor::lowp::parse_format("nvf4").unwrap();
    /// let (next, loss) = model.train_step_all_at_on_qat(
    ///     dev, &mut opt, &sched, step, 1.0,
    ///     |w| rlx_tensor::lowp::quantize_slice(w, e, m, max), feed);
    /// ```
    pub fn train_step_all_at_on_qat(
        &self,
        device: rlx_runtime::Device,
        opt: &mut dyn rlx_optim::Optimizer,
        schedule: &crate::LrSchedule,
        step: usize,
        max_grad_norm: f32,
        quant: impl Fn(&mut [f32]),
        inputs: &[(&str, &[f32])],
    ) -> (Func, Vec<f32>) {
        opt.set_lr(schedule.lr_at(step));
        let names = self.param_names();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        // Quantized forward model (STE): weights rounded to the target grid.
        let mut q = self.clone();
        for name in &names {
            let mut w = self
                .param_binding(name)
                .unwrap_or_else(|| panic!("qat: param {name:?} is not bound"))
                .to_vec();
            quant(&mut w);
            q = q.with_param(name.clone(), w);
        }
        let mut outputs = q.value_and_grad(&refs).run_on(device, inputs);
        let scale = grad_clip_scale(&outputs, max_grad_norm);
        if scale < 1.0 {
            for g in outputs.iter_mut().skip(1) {
                for x in g.iter_mut() {
                    *x *= scale;
                }
            }
        }
        // Optimizer updates SELF's f32 masters (not the quantized copy).
        self.apply_optimizer_step(opt, &refs, &outputs)
    }

    /// Apply `opt` to each `wrt` param given a `[loss, grad…]` output vector,
    /// returning the updated `Func` and the loss. Shared by every `train_step*`.
    fn apply_optimizer_step(
        &self,
        opt: &mut dyn rlx_optim::Optimizer,
        wrt: &[&str],
        outputs: &[Vec<f32>],
    ) -> (Func, Vec<f32>) {
        let loss = outputs[0].clone();
        // Collect every parameter's data + shape up front, then hand the whole
        // batch to `step_batch` (default = the same serial loop; overriding
        // optimizers can parallelize independent groups). Keeping the owned data
        // alive lets us build `&mut` OptItems without per-param `with_param`
        // clones inside the hot loop.
        let mut datas: Vec<Vec<f32>> = wrt
            .iter()
            .map(|name| {
                self.param_binding(name)
                    .unwrap_or_else(|| panic!("train_step: param {name:?} is not bound"))
                    .to_vec()
            })
            .collect();
        let shapes: Vec<Vec<usize>> = wrt.iter().map(|name| self.param_shape(name)).collect();
        let mut items: Vec<rlx_optim::OptItem> = datas
            .iter_mut()
            .enumerate()
            .map(|(i, data)| rlx_optim::OptItem {
                name: wrt[i],
                shape: &shapes[i],
                param: data.as_mut_slice(),
                grad: &outputs[i + 1],
            })
            .collect();
        opt.step_batch(&mut items);
        drop(items);
        opt.end_iteration();
        let mut updated = self.clone();
        for (name, data) in wrt.iter().zip(datas) {
            updated = updated.with_param(*name, data);
        }
        (updated, loss)
    }

    /// Static dims of a `Param` node by name.
    fn param_shape(&self, name: &str) -> Vec<usize> {
        self.graph
            .nodes()
            .iter()
            .find_map(|n| match &n.op {
                rlx_ir::Op::Param { name: pn } if pn == name => Some(
                    n.shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            rlx_ir::Dim::Static(s) => *s,
                            rlx_ir::Dim::Dynamic(_) => 0,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_else(|| panic!("train_step: no param named {name:?}"))
    }
}

/// Global-L2-norm gradient-clip factor for a `[loss, grad…]` output vector:
/// `min(1, max_norm / ‖grads‖₂)` over the concatenation of every gradient
/// tensor (indices `1..`). Returns `1.0` (no clip) when `max_norm <= 0` or the
/// grad norm is within budget. Accumulated in `f64` for numerical safety.
#[cfg(feature = "optim")]
fn grad_clip_scale(outputs: &[Vec<f32>], max_norm: f32) -> f32 {
    if max_norm <= 0.0 {
        return 1.0;
    }
    let mut sumsq = 0f64;
    for g in outputs.iter().skip(1) {
        for &x in g {
            sumsq += (x as f64) * (x as f64);
        }
    }
    let norm = sumsq.sqrt() as f32;
    if norm.is_finite() && norm > max_norm {
        max_norm / norm
    } else {
        1.0
    }
}

/// An already-compiled [`Func`], returned by [`Func::jit`]. Cloning is cheap
/// (shared compiled graph); calls run on the same backend artifact.
#[cfg(feature = "eval")]
#[derive(Clone)]
pub struct Jitted {
    compiled: std::rc::Rc<std::cell::RefCell<rlx_runtime::CompiledGraph>>,
    params: Vec<(String, Vec<f32>)>,
}

#[cfg(feature = "eval")]
impl Jitted {
    /// Run with named inputs, one `Vec<f32>` per output. No recompile, no
    /// cache lookup. Bound params are applied before each run.
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut c = self.compiled.borrow_mut();
        for (name, data) in &self.params {
            c.set_param(name, data);
        }
        c.run(inputs)
    }
}
