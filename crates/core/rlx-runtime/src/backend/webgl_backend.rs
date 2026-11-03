// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_ir::OpKind;
use rlx_webgl::{Plan, build_plan, runnable_ops};
use std::collections::HashMap;

pub struct WebglBackend;

struct WebglExecutable {
    plan: Plan,
    params: HashMap<String, Vec<f32>>,
    /// this executable's plan, to the shared backend's caches
    #[cfg(target_family = "wasm")]
    id: u64,
}

/// One WebGL2 backend a thread (a context, its programs compiled once),
/// shared by every executable: a model run a layer at a time compiles a
/// graph a layer, and a context each was ~35 programs compiled again a
/// layer and contexts piling up until a phone's tab was killed.
/// The thread's WebGL2 backend (one context, shared with every compiled
/// graph), for work outside a graph — [`rlx_webgl::exec_gl::ResidentStore`].
#[cfg(target_family = "wasm")]
pub fn webgl_backend() -> Result<std::rc::Rc<rlx_webgl::exec_gl::GlBackend>, String> {
    shared::backend()
}

/// A fresh plan id for [`webgl_backend`]'s caches.
#[cfg(target_family = "wasm")]
pub fn webgl_plan_id() -> u64 {
    shared::next_id()
}

#[cfg(target_family = "wasm")]
mod shared {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use rlx_webgl::exec_gl::GlBackend;

    thread_local! {
        static GL: RefCell<Option<Rc<GlBackend>>> = const { RefCell::new(None) };
        static NEXT: Cell<u64> = const { Cell::new(1) };
    }

    pub fn backend() -> Result<Rc<GlBackend>, String> {
        GL.with(|g| {
            if let Some(gl) = g.borrow().as_ref() {
                return Ok(gl.clone());
            }
            let gl = Rc::new(
                GlBackend::new()
                    .map_err(|e| format!("WebGL2 unavailable in this browser: {}", e.0))?,
            );
            *g.borrow_mut() = Some(gl.clone());
            Ok(gl)
        })
    }

    /// The backend, if there is one yet.
    pub fn existing() -> Option<Rc<GlBackend>> {
        GL.with(|g| g.borrow().clone())
    }

    /// The context was lost: make a new one next time.
    pub fn drop_backend() {
        GL.with(|g| g.borrow_mut().take());
    }

    pub fn next_id() -> u64 {
        NEXT.with(|n| n.replace(n.get() + 1))
    }
}

impl Backend for WebglBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        // Native kernels plus the rlx-cpu host fallback (`build_plan` still
        // lowers natively whatever it can).
        runnable_ops()
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let _ = options;
        if let Some(reason) = crate::browser_support::collective_block_reason(&graph) {
            panic!("{reason}");
        }
        let plan =
            build_plan(&graph).unwrap_or_else(|e| panic!("rlx-webgl compile failed: {}", e.0));
        // Natively `OpenGl` runs the plan on the CPU, where no texture limit
        // applies — so check the one browsers impose, or a graph that passes
        // every native test fails in the tab. `RLX_WEBGL_MAX_TEXTURE` sets it.
        #[cfg(not(target_family = "wasm"))]
        {
            let limit = rlx_ir::env::parse_or("RLX_WEBGL_MAX_TEXTURE", 16384usize);
            if let Some((origin, rows, cols)) = plan.largest_texture()
                && rows.max(cols) > limit
            {
                panic!(
                    "rlx-webgl: {origin} needs a {rows} x {cols} texture, over the {limit} \
                     MAX_TEXTURE_SIZE of desktop browsers (graph {:?})",
                    graph.name
                );
            }
        }
        Box::new(WebglExecutable {
            plan,
            params: HashMap::new(),
            #[cfg(target_family = "wasm")]
            id: shared::next_id(),
        })
    }
}

impl ExecutableGraph for WebglExecutable {
    fn capabilities(&self) -> crate::ExecutableCapabilities {
        crate::ExecutableCapabilities {
            clone: true,
            ..crate::ExecutableCapabilities::NONE
        }
    }

    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params.insert(name.to_string(), data.to_vec());
        // The GL side caches param textures across runs; a new value must
        // be uploaded again.
        #[cfg(target_family = "wasm")]
        if let Some(gl) = shared::existing() {
            gl.forget_param_of(self.id, name);
        }
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.try_run(inputs).unwrap_or_else(|e| panic!("{e}"))
    }

    fn try_run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        let mut named: Vec<(&str, &[f32])> = inputs.to_vec();
        for (name, data) in &self.params {
            named.push((name.as_str(), data.as_slice()));
        }
        #[cfg(target_family = "wasm")]
        {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let r = gl.run(&self.plan, &named);
            // a lost context stays lost: a new one next run
            if r.is_err() && gl.context_lost() {
                shared::drop_backend();
            }
            r.map_err(|e| format!("rlx-webgl run failed: {}", e.0))
        }
        #[cfg(not(target_family = "wasm"))]
        {
            rlx_webgl::run_cpu(&self.plan, &named)
                .map_err(|e| format!("rlx-webgl run_cpu failed: {}", e.0))
        }
    }

    #[cfg(target_family = "wasm")]
    fn try_submit(&mut self, inputs: &[(&str, &[f32])]) -> Option<Result<(), String>> {
        let mut named: Vec<(&str, &[f32])> = inputs.to_vec();
        for (name, data) in &self.params {
            named.push((name.as_str(), data.as_slice()));
        }
        Some((|| {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let r = gl.submit(self.id, &self.plan, &named);
            if r.is_err() && gl.context_lost() {
                shared::drop_backend();
            }
            r.map_err(|e| format!("rlx-webgl run failed: {}", e.0))
        })())
    }

    #[cfg(target_family = "wasm")]
    fn try_submit_half(
        &mut self,
        inputs: Vec<(String, Vec<u16>)>,
    ) -> Option<Result<Vec<(String, Vec<u16>)>, String>> {
        let named: Vec<(&str, &[f32])> = self
            .params
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        Some((|| {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let r = gl.submit_half(self.id, &self.plan, &named, inputs);
            if r.is_err() && gl.context_lost() {
                shared::drop_backend();
            }
            r.map_err(|e| format!("rlx-webgl run failed: {}", e.0))
        })())
    }

    #[cfg(target_family = "wasm")]
    fn try_collect(&mut self) -> Result<Vec<Vec<f32>>, String> {
        let gl = shared::existing().ok_or("rlx-webgl: no backend to collect from")?;
        let r = gl.collect(self.id);
        if r.is_err() && gl.context_lost() {
            shared::drop_backend();
        }
        r.map_err(|e| format!("rlx-webgl run failed: {}", e.0))
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(WebglExecutable {
            plan: self.plan.clone(),
            params: self.params.clone(),
            #[cfg(target_family = "wasm")]
            id: shared::next_id(),
        })
    }
}
