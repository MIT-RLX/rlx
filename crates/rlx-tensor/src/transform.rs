// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

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
        let loss = outputs[0].clone();
        let mut updated = self.clone();
        for (i, name) in wrt.iter().enumerate() {
            let grad = &outputs[i + 1];
            let shape = self.param_shape(name);
            let mut data = self
                .param_binding(name)
                .unwrap_or_else(|| panic!("train_step: param {name:?} is not bound"))
                .to_vec();
            opt.step(name, &shape, &mut data, grad);
            updated = updated.with_param(*name, data);
        }
        opt.end_iteration();
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
