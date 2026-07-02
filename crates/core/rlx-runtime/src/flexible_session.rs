// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Session that defers backend choice until compile time.

use rlx_driver::Device;
use rlx_ir::{Graph, GraphModule, HirModule};
use rlx_opt::PrecisionPolicy;

use crate::compiled::CompiledGraph;
use crate::device_policy::{DevicePolicy, resolve_device};
use crate::precision::Precision;
use crate::session::Session;

/// Compile-time settings without a fixed [`Device`].
///
/// Pick the backend per graph via [`Self::compile_resolved`] or
/// [`Self::compile_on`].
pub struct FlexibleSession {
    device_policy: DevicePolicy,
    precision: Precision,
    op_policy: Option<PrecisionPolicy>,
}

impl Default for FlexibleSession {
    fn default() -> Self {
        Self::new()
    }
}

impl FlexibleSession {
    pub fn new() -> Self {
        Self {
            device_policy: DevicePolicy::default(),
            precision: Precision::F32,
            op_policy: None,
        }
    }

    pub fn from_env() -> Self {
        Self {
            device_policy: DevicePolicy::from_env(),
            ..Self::new()
        }
    }

    pub fn with_device_policy(mut self, policy: DevicePolicy) -> Self {
        self.device_policy = policy;
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_op_policy(mut self, policy: PrecisionPolicy) -> Self {
        self.op_policy = Some(policy);
        self
    }

    pub fn device_policy(&self) -> &DevicePolicy {
        &self.device_policy
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    fn session_on(&self, device: Device) -> Session {
        let mut s = Session::new_with_precision(device, self.precision);
        if let Some(p) = &self.op_policy {
            s = s.with_policy(p.clone());
        }
        s
    }

    pub fn compile_on(&self, graph: Graph, device: Device) -> Result<CompiledGraph, String> {
        Ok(self.session_on(device).compile(graph))
    }

    pub fn compile_with_on(
        &self,
        graph: Graph,
        device: Device,
        options: &crate::CompileOptions,
    ) -> Result<CompiledGraph, String> {
        Ok(self.session_on(device).compile_with(graph, options))
    }

    pub fn compile_resolved(
        &self,
        graph: Graph,
        hint: Option<Device>,
    ) -> Result<CompiledGraph, String> {
        let device = resolve_device(&graph, hint, &self.device_policy)?;
        self.compile_on(graph, device)
    }

    pub fn compile_resolved_with(
        &self,
        graph: Graph,
        hint: Option<Device>,
        options: &crate::CompileOptions,
    ) -> Result<CompiledGraph, String> {
        let device = resolve_device(&graph, hint, &self.device_policy)?;
        self.compile_with_on(graph, device, options)
    }
}

impl FlexibleSession {
    pub fn compile_hir_on(
        &self,
        hir: HirModule,
        device: Device,
    ) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        self.session_on(device).compile_hir(hir)
    }

    pub fn compile_module_on(
        &self,
        module: GraphModule,
        device: Device,
    ) -> Result<CompiledGraph, rlx_ir::hir::LowerError> {
        self.session_on(device).compile_module(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    #[test]
    fn compile_resolved_picks_cpu() {
        let mut g = Graph::new("id");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        g.set_outputs(vec![x]);
        let session = FlexibleSession::new().with_device_policy(DevicePolicy::only([Device::Cpu]));
        let compiled = session.compile_resolved(g, None).expect("compile");
        assert_eq!(compiled.device(), Device::Cpu);
    }

    #[test]
    fn compile_resolved_with_matches_compile() {
        let mut g = Graph::new("id");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        let y = g.input("y", Shape::new(&[2], DType::F32));
        let s = g.add_node(
            rlx_ir::Op::Binary(rlx_ir::op::BinaryOp::Add),
            vec![x, y],
            Shape::new(&[2], DType::F32),
        );
        g.set_outputs(vec![s]);

        let session = FlexibleSession::new().with_device_policy(DevicePolicy::only([Device::Cpu]));
        let g1 = g.clone();
        let g2 = g;
        let a = session.compile_resolved(g1, None).expect("compile");
        let b = session
            .compile_resolved_with(g2, None, &crate::CompileOptions::new())
            .expect("compile_with");
        assert_eq!(a.device(), b.device());
    }
}
