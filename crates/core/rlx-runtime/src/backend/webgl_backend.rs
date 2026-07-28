// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use rlx_ir::OpKind;
use rlx_webgl::{Plan, build_plan, supported_ops};
use std::collections::HashMap;

pub struct WebglBackend;

struct WebglExecutable {
    plan: Plan,
    params: HashMap<String, Vec<f32>>,
    #[cfg(target_arch = "wasm32")]
    gl: Option<rlx_webgl::exec_gl::GlBackend>,
}

impl Backend for WebglBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        supported_ops()
    }

    fn compile(&self, graph: Graph, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        let _ = options;
        if let Some(reason) = crate::browser_support::collective_block_reason(&graph) {
            panic!("{reason}");
        }
        let plan =
            build_plan(&graph).unwrap_or_else(|e| panic!("rlx-webgl compile failed: {}", e.0));
        Box::new(WebglExecutable {
            plan,
            params: HashMap::new(),
            #[cfg(target_arch = "wasm32")]
            gl: None,
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
    }

    fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let mut named: Vec<(&str, &[f32])> = inputs.to_vec();
        for (name, data) in &self.params {
            named.push((name.as_str(), data.as_slice()));
        }
        #[cfg(target_arch = "wasm32")]
        {
            let gl = self.gl.get_or_insert_with(|| {
                rlx_webgl::exec_gl::GlBackend::new()
                    .expect("WebGL2 backend unavailable in this browser")
            });
            gl.run(&self.plan, &named)
                .unwrap_or_else(|e| panic!("rlx-webgl run failed: {}", e.0))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            rlx_webgl::run_cpu(&self.plan, &named)
                .unwrap_or_else(|e| panic!("rlx-webgl run_cpu failed: {}", e.0))
        }
    }

    fn clone_box(&self) -> Box<dyn ExecutableGraph> {
        Box::new(WebglExecutable {
            plan: self.plan.clone(),
            params: self.params.clone(),
            #[cfg(target_arch = "wasm32")]
            gl: None,
        })
    }
}
