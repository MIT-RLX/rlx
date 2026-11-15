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
    params: HashMap<String, HostParam>,
    /// this executable's plan, to the shared backend's caches
    #[cfg(target_family = "wasm")]
    id: u64,
}

/// A param's value as kept here: the GL side computes in f32, but holds a
/// param as a texture once uploaded, so a half one stays half (2 B a value
/// in the tab's memory, not 4) and is widened only when it must be uploaded
/// (its first run, a lost context).
#[derive(Clone)]
enum HostParam {
    F32(Vec<f32>),
    /// F16 or BF16 bytes.
    Half(Vec<u8>, rlx_ir::DType),
}

impl WebglExecutable {
    /// The params for a run: each f32 one as it is, and each half one
    /// widened — of those, only the ones in `upload` when given (the others
    /// are textures already, and are not read).
    fn run_params(&self, upload: Option<&[&str]>) -> Vec<(&str, std::borrow::Cow<'_, [f32]>)> {
        let upload: Option<std::collections::HashSet<&str>> =
            upload.map(|u| u.iter().copied().collect());
        self.params
            .iter()
            .filter_map(|(name, p)| match p {
                HostParam::F32(v) => Some((name.as_str(), std::borrow::Cow::Borrowed(&v[..]))),
                HostParam::Half(_, _)
                    if upload.as_ref().is_some_and(|u| !u.contains(name.as_str())) =>
                {
                    None
                }
                HostParam::Half(b, dtype) => Some((
                    name.as_str(),
                    std::borrow::Cow::Owned(super::widen_bytes_to_f32(b, *dtype)),
                )),
            })
            .collect()
    }

    /// The GL side caches param textures across runs; a new value must be
    /// uploaded again.
    fn forget_uploaded(&self, name: &str) {
        #[cfg(target_family = "wasm")]
        if let Some(gl) = shared::existing() {
            gl.forget_param_of(self.id, name);
        }
        let _ = name;
    }

    /// The params for the shared backend's next run of this plan: each half
    /// one it has still to upload is widened and uploaded now, one at a time
    /// (widening them all for the run held the whole model in f32 at once,
    /// and a tab's memory never shrinks back); the f32 ones as they are.
    #[cfg(target_family = "wasm")]
    fn gl_params(
        &self,
        gl: &rlx_webgl::exec_gl::GlBackend,
    ) -> Result<Vec<(&str, std::borrow::Cow<'_, [f32]>)>, String> {
        for name in gl.params_to_upload(self.id, &self.plan) {
            if let Some(HostParam::Half(b, dtype)) = self.params.get(name) {
                gl.upload_param(&self.plan, name, &super::widen_bytes_to_f32(b, *dtype))
                    .map_err(|e| format!("rlx-webgl upload of {name} failed: {}", e.0))?;
            }
        }
        Ok(self.run_params(Some(&[])))
    }
}

/// Its plan's textures go with it.
#[cfg(target_family = "wasm")]
impl Drop for WebglExecutable {
    fn drop(&mut self) {
        if let Some(gl) = shared::existing() {
            gl.drop_plan(self.id);
        }
    }
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
        let t0 = web_time_now();
        let plan =
            build_plan(&graph).unwrap_or_else(|e| panic!("rlx-webgl compile failed: {}", e.0));
        // `RLX_WEBGL_PLAN_STATS=1`: what planning cost (time, steps, index tables)
        if rlx_ir::env::flag("RLX_WEBGL_PLAN_STATS") {
            let tables: Vec<String> = plan
                .index_table_stats()
                .iter()
                .map(|(k, n, m)| {
                    format!(
                        "{k} {:.1} MB (largest {:.1} MB)",
                        *n as f64 * 4e-6,
                        *m as f64 * 4e-6
                    )
                })
                .collect();
            let ops: Vec<String> = plan
                .steps_by_origin()
                .iter()
                .take(6)
                .map(|(o, n)| format!("{o} {n}"))
                .collect();
            eprintln!(
                "rlx-webgl plan {:?}: {} nodes -> {} steps, {} slots, {:.0} ms; tables: {}; steps by op: {}",
                graph.name,
                graph.len(),
                plan.steps.len(),
                plan.slot_dims.len(),
                web_time_now() - t0,
                tables.join(", "),
                ops.join(", ")
            );
        }
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

    /// Half-precision (F16 / BF16) weights kept half ([`HostParam`]), and
    /// widened to the f32 the GL side computes in only to be uploaded; other
    /// float types widened now (the default panics for anything but F32).
    fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: rlx_ir::DType) {
        match dtype {
            rlx_ir::DType::F32 => {
                let f = rlx_ir::bytes::decode_le::<f32>(data);
                self.set_param(name, f.as_ref());
            }
            rlx_ir::DType::F16 | rlx_ir::DType::BF16 => {
                self.params
                    .insert(name.to_string(), HostParam::Half(data.to_vec(), dtype));
                self.forget_uploaded(name);
            }
            _ => self.set_param(name, &super::widen_bytes_to_f32(data, dtype)),
        }
    }

    fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params
            .insert(name.to_string(), HostParam::F32(data.to_vec()));
        self.forget_uploaded(name);
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.try_run(inputs).unwrap_or_else(|e| panic!("{e}"))
    }

    fn try_run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        let mut named: Vec<(&str, &[f32])> = inputs.to_vec();
        #[cfg(target_family = "wasm")]
        {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let params = self.gl_params(&gl)?;
            named.extend(params.iter().map(|(n, d)| (*n, &d[..])));
            let r = gl.run(&self.plan, &named);
            // a lost context stays lost: a new one next run
            if r.is_err() && gl.context_lost() {
                shared::drop_backend();
            }
            r.map_err(|e| format!("rlx-webgl run failed: {}", e.0))
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let params = self.run_params(None);
            named.extend(params.iter().map(|(n, d)| (*n, &d[..])));
            rlx_webgl::run_cpu(&self.plan, &named)
                .map_err(|e| format!("rlx-webgl run_cpu failed: {}", e.0))
        }
    }

    #[cfg(target_family = "wasm")]
    fn try_submit(&mut self, inputs: &[(&str, &[f32])]) -> Option<Result<(), String>> {
        Some((|| {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let params = self.gl_params(&gl)?;
            let mut named: Vec<(&str, &[f32])> = inputs.to_vec();
            named.extend(params.iter().map(|(n, d)| (*n, &d[..])));
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
        Some((|| {
            let gl = shared::backend()?;
            gl.use_plan(self.id);
            let params = self.gl_params(&gl)?;
            let named: Vec<(&str, &[f32])> = params.iter().map(|(n, d)| (*n, &d[..])).collect();
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

/// Milliseconds on a monotonic clock (natively; the browser has no
/// `Instant`, and its plan stats report 0 ms).
fn web_time_now() -> f64 {
    #[cfg(not(target_family = "wasm"))]
    {
        use std::sync::OnceLock;
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1e3
    }
    #[cfg(target_family = "wasm")]
    {
        0.0
    }
}
