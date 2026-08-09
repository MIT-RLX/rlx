// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-backend execution — compile once per device, run on any of them.

use std::collections::HashMap;

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Op};

use crate::compiled::CompiledGraph;
use crate::cost::fastest_device_for_with_policy;
use crate::device_bench::{DeviceBenchResult, benchmark_devices, warm_all};
use crate::device_ext::is_available;
use crate::device_policy::{
    DeviceCandidate, DeviceFallbackError, DevicePickStrategy, DevicePolicy, device_chain_from_env,
    device_report, devices_for_with_policy, resolve_device, resolve_device_chain,
};
use crate::session::Session;

/// Param names declared in `graph` (`Op::Param`).
pub fn graph_param_names(graph: &Graph) -> Vec<String> {
    graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[derive(Debug, Clone)]
enum CachedParam {
    F32(Vec<f32>),
    Typed { bytes: Vec<u8>, dtype: DType },
}

fn apply_cached_params(compiled: &mut CompiledGraph, params: &HashMap<String, CachedParam>) {
    for (name, param) in params {
        match param {
            CachedParam::F32(data) => compiled.set_param(name, data),
            CachedParam::Typed { bytes, dtype } => compiled.set_param_typed(name, bytes, *dtype),
        }
    }
}

/// A graph plus lazy per-device compiled executables.
pub struct GraphDevices {
    graph: Graph,
    policy: DevicePolicy,
    pick: DevicePickStrategy,
    supported: Vec<Device>,
    params: HashMap<String, CachedParam>,
    benchmark_winner: Option<Device>,
    cache: HashMap<Device, CompiledGraph>,
}

impl GraphDevices {
    pub fn new(graph: Graph) -> Self {
        Self::with_policy(graph, DevicePolicy::default())
    }

    pub fn with_policy(graph: Graph, policy: DevicePolicy) -> Self {
        let pick = policy.pick_strategy();
        let supported = devices_for_with_policy(&graph, &policy);
        Self {
            graph,
            policy,
            pick,
            supported,
            params: HashMap::new(),
            benchmark_winner: None,
            cache: HashMap::new(),
        }
    }

    pub fn from_env(graph: Graph) -> Self {
        Self::with_policy(graph, DevicePolicy::from_env())
    }

    pub fn policy(&self) -> &DevicePolicy {
        &self.policy
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn devices(&self) -> &[Device] {
        &self.supported
    }

    pub fn report(&self) -> Vec<DeviceCandidate> {
        device_report(&self.graph, &self.policy)
    }

    pub fn fastest(&self) -> Device {
        fastest_device_for_with_policy(&self.graph, &self.policy)
    }

    pub fn resolve(&self, hint: Option<Device>) -> Result<Device, String> {
        resolve_device(&self.graph, hint, &self.policy)
    }

    /// Resolve using `RLX_DEVICE_CHAIN` when set, else [`Self::resolve`].
    pub fn resolve_chain(&self, hint: Option<Device>) -> Result<Device, String> {
        if let Some(device) = hint {
            return self.resolve(Some(device));
        }
        let chain = device_chain_from_env();
        if chain.is_empty() {
            return self.resolve(None);
        }
        resolve_device_chain(&self.graph, &chain, &self.policy)
    }

    /// Upload a param to every cached executor and future compilations.
    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.params
            .insert(name.to_string(), CachedParam::F32(data.to_vec()));
        for compiled in self.cache.values_mut() {
            compiled.set_param(name, data);
        }
    }

    /// Typed param upload — mirrored to all cached backends.
    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: DType) {
        self.params.insert(
            name.to_string(),
            CachedParam::Typed {
                bytes: data.to_vec(),
                dtype,
            },
        );
        for compiled in self.cache.values_mut() {
            compiled.set_param_typed(name, data, dtype);
        }
    }

    /// Re-apply stored params to all cached backends (after manual cache changes).
    pub fn sync_params_to_all(&mut self) {
        for compiled in self.cache.values_mut() {
            apply_cached_params(compiled, &self.params);
        }
    }

    /// Hint → env → cost model, or micro-benchmark when policy requests it.
    pub fn resolve_with_inputs(
        &mut self,
        hint: Option<Device>,
        inputs: &[(&str, &[f32])],
    ) -> Result<Device, String> {
        if hint.is_some() {
            return self.resolve(hint);
        }
        match self.pick {
            DevicePickStrategy::CostModel => self.resolve(None),
            DevicePickStrategy::Benchmark { runs } => {
                if let Some(device) = self.benchmark_winner {
                    return Ok(device);
                }
                let ranked = self.benchmark(inputs, runs)?;
                let device = ranked
                    .first()
                    .map(|r| r.device)
                    .unwrap_or_else(|| self.fastest());
                self.benchmark_winner = Some(device);
                Ok(device)
            }
        }
    }

    pub fn compile(&mut self, device: Device) -> Result<&mut CompiledGraph, String> {
        Self::ensure_supported(&self.supported, device)?;
        if !self.cache.contains_key(&device) {
            let mut compiled = Session::new(device).compile(self.graph.clone());
            apply_cached_params(&mut compiled, &self.params);
            self.cache.insert(device, compiled);
        }
        Ok(self.cache.get_mut(&device).expect("just inserted"))
    }

    pub fn compile_fastest(&mut self) -> Result<&mut CompiledGraph, String> {
        self.compile(self.fastest())
    }

    pub fn compile_resolved(&mut self, hint: Option<Device>) -> Result<&mut CompiledGraph, String> {
        self.compile(self.resolve(hint)?)
    }

    pub fn compile_chain(&mut self, hint: Option<Device>) -> Result<&mut CompiledGraph, String> {
        self.compile(self.resolve_chain(hint)?)
    }

    pub fn warm_all(&mut self) -> Result<Vec<Device>, String> {
        warm_all(self)
    }

    pub fn benchmark(
        &mut self,
        inputs: &[(&str, &[f32])],
        runs: usize,
    ) -> Result<Vec<DeviceBenchResult>, String> {
        benchmark_devices(self, inputs, runs)
    }

    pub fn run(
        &mut self,
        device: Device,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.compile(device)?.run(inputs))
    }

    pub fn run_resolved(
        &mut self,
        hint: Option<Device>,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.compile_resolved(hint)?.run(inputs))
    }

    pub fn run_fastest(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.compile_fastest()?.run(inputs))
    }

    /// Try `chain` in order until one backend compiles and runs successfully.
    pub fn run_try(
        &mut self,
        chain: &[Device],
        inputs: &[(&str, &[f32])],
    ) -> Result<(Device, Vec<Vec<f32>>), DeviceFallbackError> {
        let viable: Vec<Device> = self.devices().to_vec();
        let mut attempts = Vec::new();
        for &device in chain {
            if !viable.contains(&device) {
                attempts.push((device, "not viable for this graph under policy".into()));
                continue;
            }
            match self.run(device, inputs) {
                Ok(value) => return Ok((device, value)),
                Err(err) => attempts.push((device, err)),
            }
        }
        if attempts.is_empty() {
            attempts.push((Device::Cpu, "empty fallback chain".into()));
        }
        Err(DeviceFallbackError { attempts })
    }

    /// Like [`Self::run_try`] using `RLX_DEVICE_CHAIN` when set.
    pub fn run_chain(
        &mut self,
        hint: Option<Device>,
        inputs: &[(&str, &[f32])],
    ) -> Result<(Device, Vec<Vec<f32>>), DeviceFallbackError> {
        if let Some(device) = hint {
            self.run(device, inputs)
                .map(|v| (device, v))
                .map_err(|e| DeviceFallbackError {
                    attempts: vec![(device, e)],
                })
        } else {
            let chain = device_chain_from_env();
            if chain.is_empty() {
                let device = self.resolve(None).map_err(|e| DeviceFallbackError {
                    attempts: vec![(Device::Cpu, e)],
                })?;
                self.run(device, inputs)
                    .map(|v| (device, v))
                    .map_err(|e| DeviceFallbackError {
                        attempts: vec![(device, e)],
                    })
            } else {
                self.run_try(&chain, inputs)
            }
        }
    }

    pub fn compile_resolved_with_inputs(
        &mut self,
        hint: Option<Device>,
        inputs: &[(&str, &[f32])],
    ) -> Result<&mut CompiledGraph, String> {
        let device = self.resolve_with_inputs(hint, inputs)?;
        self.compile(device)
    }

    pub fn run_resolved_with_inputs(
        &mut self,
        hint: Option<Device>,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<Vec<f32>>, String> {
        Ok(self.compile_resolved_with_inputs(hint, inputs)?.run(inputs))
    }

    pub fn invalidate_cache(&mut self) {
        self.cache.clear();
        self.benchmark_winner = None;
        self.supported = devices_for_with_policy(&self.graph, &self.policy);
    }

    pub fn set_policy(&mut self, policy: DevicePolicy) {
        self.policy = policy.clone();
        self.pick = policy.pick_strategy();
        self.invalidate_cache();
    }

    fn ensure_supported(supported: &[Device], device: Device) -> Result<(), String> {
        if !is_available(device) {
            return Err(format!(
                "device {device} is not available — enable the matching Cargo feature"
            ));
        }
        if !supported.contains(&device) {
            return Err(format!(
                "device {device} cannot lower this graph under the active policy"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, Shape};

    fn identity_graph() -> Graph {
        let mut g = Graph::new("id");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        g.set_outputs(vec![x]);
        g
    }

    #[test]
    fn set_param_applies_to_new_compile() {
        let mut g = Graph::new("p");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        let w = g.param("w", Shape::new(&[2], DType::F32));
        let y = g.binary(
            rlx_ir::op::BinaryOp::Add,
            x,
            w,
            Shape::new(&[2], DType::F32),
        );
        g.set_outputs(vec![y]);

        let mut runner = GraphDevices::new(g);
        runner.set_param("w", &[1.0, 2.0]);
        let out = runner.run(Device::Cpu, &[("x", &[3.0, 4.0])]).unwrap();
        assert_eq!(out[0], vec![4.0, 6.0]);
    }

    #[test]
    fn run_on_cpu_roundtrip() {
        let mut runner = GraphDevices::new(identity_graph());
        let out = runner
            .run(Device::Cpu, &[("x", &[1.0, 2.0, 3.0, 4.0])])
            .expect("cpu run");
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn run_try_falls_back_to_cpu() {
        let mut runner = GraphDevices::new(identity_graph());
        // Lead with a device THIS BUILD genuinely cannot run, so the fallback
        // path is exercised on every host. Hard-coding `Device::Cuda` only
        // passed where CUDA was absent — on a CUDA host `run_try` correctly
        // selects Cuda and an unconditional `== Cpu` assertion fails.
        let chain: Vec<Device> = match [Device::Cuda, Device::Rocm, Device::Vulkan, Device::Metal]
            .into_iter()
            .find(|d| !crate::is_available(*d))
        {
            Some(unavailable) => vec![unavailable, Device::Cpu],
            None => vec![Device::Cpu],
        };
        let (dev, out) = runner
            .run_try(&chain, &[("x", &[1.0, 2.0, 3.0, 4.0])])
            .expect("fallback");
        assert_eq!(dev, Device::Cpu);
        assert_eq!(out[0][0], 1.0);
    }
}
