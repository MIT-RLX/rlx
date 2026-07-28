// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serving-oriented wrapper: warm-all backends, fallback execution, throttle-aware re-warm.

use rlx_driver::Device;
use rlx_ir::{DType, Graph};

use crate::device_parse::device_label;
use crate::device_policy::{DeviceFallbackError, DevicePolicy};
use crate::graph_devices::GraphDevices;
use crate::hwinfo::HwSnapshot;

/// Production helper: compile all viable backends up front, run with fallback chain.
pub struct DeviceRouter {
    runner: GraphDevices,
    rebench_on_throttle: bool,
}

impl DeviceRouter {
    pub fn new(graph: Graph, policy: DevicePolicy) -> Result<Self, String> {
        let mut runner = GraphDevices::with_policy(graph, policy);
        runner.warm_all()?;
        Ok(Self {
            runner,
            rebench_on_throttle: true,
        })
    }

    pub fn from_env(graph: Graph) -> Result<Self, String> {
        Self::new(graph, DevicePolicy::from_env())
    }

    pub fn with_rebench_on_throttle(mut self, enabled: bool) -> Self {
        self.rebench_on_throttle = enabled;
        self
    }

    pub fn set_rebench_on_throttle(&mut self, enabled: bool) {
        self.rebench_on_throttle = enabled;
    }

    pub fn runner(&self) -> &GraphDevices {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut GraphDevices {
        &mut self.runner
    }

    fn maybe_rewarm(&mut self) -> Result<(), String> {
        if self.rebench_on_throttle && HwSnapshot::collect().is_throttled() {
            self.runner.invalidate_cache();
            self.runner.warm_all()?;
        }
        Ok(())
    }

    /// Run on explicit device.
    pub fn run_on(
        &mut self,
        device: Device,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.maybe_rewarm()?;
        self.runner.run(device, inputs)
    }

    /// Resolve hint / env / benchmark policy, then execute.
    pub fn run(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<Device>,
    ) -> Result<(Device, Vec<Vec<f32>>), String> {
        self.maybe_rewarm()?;
        let device = self.runner.resolve_with_inputs(hint, inputs)?;
        let out = self.runner.run(device, inputs)?;
        Ok((device, out))
    }

    /// Fallback chain (`RLX_DEVICE_CHAIN` or explicit).
    pub fn run_chain(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<Device>,
    ) -> Result<(Device, Vec<Vec<f32>>), DeviceFallbackError> {
        self.maybe_rewarm()?;
        self.runner.run_chain(hint, inputs)
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.runner.set_param(name, data);
    }

    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.runner.set_param_typed(name, data, dtype);
    }

    pub fn devices(&self) -> Vec<&'static str> {
        self.runner
            .devices()
            .iter()
            .map(|d| device_label(*d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    #[test]
    fn router_runs_identity_on_cpu() {
        let mut g = Graph::new("id");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        g.set_outputs(vec![x]);
        let mut router = DeviceRouter::new(g, DevicePolicy::only([Device::Cpu])).expect("router");
        let (dev, out) = router
            .run(&[("x", &[1.0, 2.0, 3.0, 4.0])], None)
            .expect("run");
        assert_eq!(dev, Device::Cpu);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }
}
