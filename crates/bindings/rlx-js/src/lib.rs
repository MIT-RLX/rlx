// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-js
//!
//! An embedded JavaScript front end for RLX. The engine is
//! [`quickrs-core`](https://github.com/eugenehp/quickjs-rs) — a pure-Rust
//! QuickJS-ng — so there is no C toolchain, no `bindgen`, and no separate
//! runtime to install: `cargo build` produces one binary that speaks JS and
//! drives every backend this build was compiled with.
//!
//! ```no_run
//! let mut rt = rlx_js::Runtime::new();
//! let out = rt.eval(r#"
//!     const g = new rlx.Graph("dot");
//!     const x = g.input("x", [1, 4], "f32");
//!     const w = g.param("w", [4, 1], "f32");
//!     g.setOutputs([g.matmul(x, w)]);
//!
//!     const c = new rlx.Session("cpu").compile(g);
//!     c.setParam("w", new Float32Array([1, 2, 3, 4]));
//!     const [y] = c.run({ x: new Float32Array([1, 1, 1, 1]) });
//!     y[0];
//! "#).unwrap();
//! assert_eq!(out, "10");
//! ```
//!
//! ## What the script gets
//!
//! | area | surface |
//! |---|---|
//! | Discovery | `rlx.devices()`, `isAvailable(d)`, `backendsManifest()` |
//! | Placement | `fastestDeviceFor(g)`, `deviceReport(g)`, `check(g, opts)` |
//! | DSL | `rlx.dsl(name)` → chainable `Tensor`s, generated from `Graph.prototype` |
//! | Graph | ~101 methods: matmul, conv2d, attention, RoPE, norms, reductions, shape ops, linalg factorizations, quantization, FFT/STFT/PSD, `scaledMatmul`, `loraMatmul`, `customFn` |
//! | Compile | `new rlx.Session(opts)` → `.compile(g)` / `.compileWith(g, {fusion, kernelDispatch})` |
//! | Execute | `.setParam`, `.setParams`, `.run`, `.runTyped`, `.outputShapes` |
//! | Multi-backend | `rlx.Runner` (warm-all, benchmark, policy), `rlx.Router` (fallback chain) |
//! | Transforms | `rlx.grad`, `jvp`, `hvp`, `vmap`, `nthOrderGrad` |
//! | Train | `rlx.Trainer`, `rlx.Optimizer` *(feature `training`)* |
//! | Weights | `openGguf`, `quantize`, `writeGguf`, `convertToGguf`, `loadPt`/`Mlx`/`Dduf`/`Nemo`, `openRlxp`, `toRlxp` |
//! | Text | `loadTokenizer`, `renderChat`, `sampleNext` *(feature `text`)* |
//! | Files | `readFile`, `readText`, `writeFile`, `fileExists`, `env` — **opt-in**, see [`install_fs`] |
//! | Host | `console.log`, `print`, `scriptArgs` |
//!
//! ## Two spellings
//!
//! `new rlx.Graph()` threads integer node ids; `rlx.dsl()` returns the same
//! graph with sources that hand back a chainable [`Tensor`](api::dsl). The two
//! build **bit-identical** graphs — the DSL is a spelling, not a second
//! builder, and `examples/transformer_block_dsl.js` asserts it.
//!
//! ## Training
//!
//! ```no_run
//! # let mut rt = rlx_js::Runtime::new();
//! rt.eval(r#"
//!     const t = new rlx.Trainer(buildLoss(), {
//!       wrt: ["w"], init: { w: new Float32Array([0, 0, 0, 0]) },
//!       device: "cpu", optimizer: { kind: "adamw", lr: 0.1 },
//!     });
//!     for (const batch of data) t.step(batch);
//!     t.params();
//! "#).ok();
//! ```
//!
//! `Trainer` differentiates and compiles once and keeps the weights; the
//! optimizer is `rlx-optim`, running on the host. `resident: true` fuses the
//! update into the graph instead ([`rlx_runtime::train`]), which is worth
//! 1.4–2.1× above ~200 k parameters and a small loss below that. It is offered
//! only on backends where it is verified correct and faster;
//! `trainer.isResident()` reports what actually happened.
//!
//! ## How the surface is generated
//!
//! [`graph_ops!`](crate::graph_ops) and [`js_class!`](crate::js_class) generate
//! the method tables from declarative signatures. That is not only brevity: the
//! macro always reads arguments **before** borrowing the graph handle (argument
//! conversion can run a getter, and a getter can re-enter the same graph) and
//! derives `Function.length` from the signature, so the two cannot disagree.
//!
//! ## Why an embedded engine rather than a Node addon
//!
//! The two answer different questions. A napi addon puts RLX inside someone
//! else's process and inherits npm, threads and an event loop. This puts a
//! script inside *ours*: one static binary, deterministic, no I/O a script can
//! reach unless an embedder hands it one, and it cross-compiles to wherever
//! RLX already goes — including iOS and wasm, where a native addon cannot
//! follow. Config files, training recipes and eval harnesses are the shape
//! that fits.
//!
//! ## Backends
//!
//! Feature gates mirror `rlx-runtime`: `cpu` (default), `metal`, `mlx`,
//! `gpu`, `cuda`, `rocm`, `vulkan`. `rlx.devices()` reports what actually
//! made it in, and `new rlx.Session("cuda")` on a build without it throws a
//! `TypeError` naming the feature to rebuild with rather than falling back to
//! CPU silently.

#[macro_use]
pub mod macros;

pub mod api;
pub mod convert;
pub mod handle;
pub mod panics;

use quickrs_core::builtins::promise::PromiseState;
use quickrs_core::context::Context;
use quickrs_core::object::{ObjectData, PropFlags};
use quickrs_core::value::Value;

/// The settled state of `v`, or `None` when it is not a promise.
fn promise_state(v: &Value) -> Option<PromiseState> {
    let obj = v.as_object()?;
    let borrowed = obj.borrow();
    match &borrowed.data {
        ObjectData::Promise(data) => Some(data.state.clone()),
        _ => None,
    }
}

/// Install the `rlx` namespace (and `console` / `print`) into an existing
/// context.
///
/// Use this when the script also needs globals of your own — build the
/// `Context`, call this, then add yours. [`Runtime`] is this plus ownership.
pub fn install(ctx: &mut Context) {
    // Before anything can run: rlx asserts its invariants, and an assertion
    // reached from a script must be an exception rather than an abort.
    panics::install_hook();
    handle::init_registry(ctx);
    api::console::install(ctx);

    let namespace = ctx.new_object();
    api::devices::install(ctx, &namespace);
    api::graph::install(ctx, &namespace);
    api::session::install(ctx, &namespace);
    // After `graph`: it enumerates `Graph.prototype` to build `Tensor`.
    api::dsl::install(ctx, &namespace);
    api::autodiff::install(ctx, &namespace);
    api::routing::install(ctx, &namespace);
    api::check::install(ctx, &namespace);
    api::buffers::install(ctx, &namespace);
    api::timers::install(ctx, &namespace);
    #[cfg(feature = "training")]
    api::train::install(ctx, &namespace);
    #[cfg(feature = "text")]
    api::text::install(ctx, &namespace);
    #[cfg(feature = "gguf")]
    api::gguf::install(ctx, &namespace);
    #[cfg(feature = "weights")]
    api::weights::install(ctx, &namespace);

    let global = ctx.global();
    // Writable and configurable, so a script or a later embedder can wrap the
    // namespace. The class registry that `handle` owns is the part that must
    // not move, and that one is locked down.
    ctx.define_value(&global, "rlx", Value::Object(namespace), PropFlags::C_W);
}

/// Hand a context the filesystem functions (`rlx.readFile`, `readText`,
/// `writeFile`, `fileExists`, `fileSize`, `env`).
///
/// Separate from [`install`] on purpose: a `Runtime` is sandboxed until an
/// embedder says otherwise. The `rlx-js` CLI calls this, since a script run
/// from a shell already has the user's authority.
pub fn install_fs(ctx: &mut Context) {
    let namespace = match ctx.global_value("rlx") {
        Value::Object(obj) => obj,
        _ => return,
    };
    api::fs::install(ctx, &namespace);
    // Recorded on the hidden registry so every other path-taking function
    // (openGguf, loadTokenizer, writeGguf, Trainer.save, …) can check it.
    handle::allow_fs(ctx);
}

/// A JavaScript context with the RLX API installed.
pub struct Runtime {
    ctx: Context,
    /// How long [`Runtime::eval`] will drive timers before giving up. Bounded so
    /// a script that schedules a timer an hour out cannot wedge the host.
    budget: std::time::Duration,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    pub fn new() -> Self {
        let mut ctx = Context::new();
        install(&mut ctx);
        Self {
            ctx,
            budget: std::time::Duration::from_secs(30),
        }
    }

    /// Cap how long `eval` will wait on timers. Default 30 s.
    pub fn set_event_loop_budget(&mut self, budget: std::time::Duration) {
        self.budget = budget;
    }

    /// Evaluate `source`, run the event loop to quiescence, and return the
    /// completion value as a string.
    ///
    /// A promise is **awaited**, not stringified. `(async () => 1)()` used to
    /// come back as `"[object Promise]"`, which made every async script look
    /// like it had returned nothing useful.
    ///
    /// Errors come back already formatted — message plus stack — because a
    /// `Value` holding an exception is useless to a caller that has dropped the
    /// context it belongs to.
    pub fn eval(&mut self, source: &str) -> Result<String, String> {
        let value = self
            .ctx
            .eval(source)
            .map_err(|e| self.ctx.format_exception(&e))?;
        self.settle(value)
    }

    /// Drain microtasks and timers, then unwrap a promise completion value.
    fn settle(&mut self, value: Value) -> Result<String, String> {
        let quiet = api::timers::drain(&mut self.ctx, self.budget)
            .map_err(|e| self.ctx.format_exception(&e))?;
        match promise_state(&value) {
            None => self
                .ctx
                .to_rust_string(&value)
                .map_err(|e| self.ctx.format_exception(&e)),
            Some(PromiseState::Fulfilled(inner)) => self
                .ctx
                .to_rust_string(&inner)
                .map_err(|e| self.ctx.format_exception(&e)),
            Some(PromiseState::Rejected(error)) => Err(self.ctx.format_exception(&error)),
            Some(PromiseState::Pending) => Err(if quiet {
                // Nothing left to run and still pending: the promise is waiting
                // on something this runtime cannot supply, and saying so beats
                // returning "[object Promise]".
                "the script returned a promise that will never settle — nothing \
                 is left on the microtask or timer queue to resolve it"
                    .to_string()
            } else {
                format!(
                    "the script returned a promise still pending after the {:?} \
                     event-loop budget; raise it with `set_event_loop_budget`",
                    self.budget
                )
            }),
        }
    }

    /// Read and evaluate a file. The path is used for error messages.
    pub fn eval_file(&mut self, path: &std::path::Path) -> Result<String, String> {
        let source =
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.eval(&source)
    }

    /// Cap how long a script may run, in VM instructions. Without this a
    /// `while (true) {}` in a config file hangs the host.
    pub fn set_instruction_budget(&mut self, instructions: Option<u64>) {
        self.ctx.set_instruction_budget(instructions);
    }

    /// Allow this runtime's scripts to read and write files.
    ///
    /// Off by default — see [`install_fs`].
    pub fn allow_filesystem(&mut self) {
        install_fs(&mut self.ctx);
    }

    /// The underlying context, for embedders adding their own globals.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }
}
