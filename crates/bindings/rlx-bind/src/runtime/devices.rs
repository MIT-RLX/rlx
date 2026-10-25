// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Devices: what this build has, which suits a graph, and running one graph
//! across several (`Runner` over `GraphDevices`, `Router` over `DeviceRouter`).
//!
//! A device policy is `"all"`, `"env"` (`RLX_DEVICE*`), or
//! `{only, deny, prefer, benchmark_runs}` — lists of device names.

use rlx_ir::Graph;
use rlx_runtime::{Device, DevicePolicy, DeviceRouter, GraphDevices};
use serde_json::{Value, json};

use super::opts::Opts;
use super::session::Compiled;
use crate::panics::catch;
use crate::{ApiError, ApiResult};

/// Labels of the devices compiled into this build.
pub fn available() -> Vec<&'static str> {
    rlx_runtime::available_devices()
        .into_iter()
        .map(rlx_runtime::device_label)
        .collect()
}

/// Whether `name` is a device this build can run. An unknown name is not an
/// error here: the question was "can I use it", and the answer is no.
pub fn is_available(name: &str) -> bool {
    rlx_runtime::parse_device(name).is_ok_and(rlx_runtime::is_available)
}

/// A device name, parsed (whether or not it is in this build).
pub fn parse(name: &str) -> ApiResult<Device> {
    rlx_runtime::parse_device(name).map_err(|e| ApiError::type_(e.to_string()))
}

pub fn label(d: Device) -> &'static str {
    rlx_runtime::device_label(d)
}

/// The backends manifest (features, kernels, capabilities) as JSON text.
pub fn manifest_json() -> String {
    rlx_runtime::BackendsManifest::json().to_string()
}

fn device_list(v: &Value, what: &str) -> ApiResult<Vec<Device>> {
    let Value::Array(items) = v else {
        return Err(ApiError::type_(format!(
            "{what}: expected a list of device names"
        )));
    };
    items
        .iter()
        .map(|item| {
            let name = item.as_str().ok_or_else(|| {
                ApiError::type_(format!("{what}: expected a list of device names"))
            })?;
            parse(name).map_err(|e| ApiError::type_(format!("{what}: {e}")))
        })
        .collect()
}

/// A device policy from `"all"`, `"env"`, `{only, deny, prefer, benchmark_runs}`
/// or null (all).
pub fn policy(v: &Value) -> ApiResult<DevicePolicy> {
    match v {
        Value::Null => Ok(DevicePolicy::all()),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "env" => Ok(DevicePolicy::from_env()),
            "all" => Ok(DevicePolicy::all()),
            other => Err(ApiError::type_(format!(
                "device policy must be 'env', 'all' or an object, got '{other}'"
            ))),
        },
        Value::Object(_) => {
            let o = Opts::new("device policy", v)?;
            let mut p = match o.get("only") {
                Some(list) => DevicePolicy::only(device_list(list, "device policy only")?),
                None => DevicePolicy::all(),
            };
            if let Some(list) = o.get("deny") {
                p = p.with_deny(device_list(list, "device policy deny")?);
            }
            if let Some(list) = o.get("prefer") {
                p = p.with_prefer(device_list(list, "device policy prefer")?);
            }
            if let Some(runs) = o.u64("benchmark_runs")? {
                p = p.with_benchmark_pick(runs as usize);
            }
            o.finish()?;
            Ok(p)
        }
        other => Err(ApiError::type_(format!(
            "device policy must be 'env', 'all' or an object, got {}",
            super::opts::type_name(other)
        ))),
    }
}

/// The device the cost model expects to run `graph` fastest.
pub fn fastest_for(graph: &Graph, policy: Option<&DevicePolicy>) -> &'static str {
    let d = match policy {
        Some(p) => rlx_runtime::cost::fastest_device_for_with_policy(graph, p),
        None => rlx_runtime::fastest_device_for(graph),
    };
    label(d)
}

/// One row per candidate device: availability, graph support, the blocker.
pub fn report(graph: &Graph, policy: &DevicePolicy) -> Value {
    let rows: Vec<Value> = rlx_runtime::device_report(graph, policy)
        .into_iter()
        .map(|c| {
            json!({
                "device": c.label,
                "available": c.available,
                "registered": c.registered,
                "supports_graph": c.supports_graph,
                "recommended": c.recommended,
                "blocker": c.blocker,
                "capabilities": c.capabilities,
            })
        })
        .collect();
    Value::Array(rows)
}

fn optional(hint: Option<&str>) -> ApiResult<Option<Device>> {
    hint.map(parse).transpose()
}

/// One graph, compiled lazily per device — the binding form of `GraphDevices`.
pub struct Runner {
    pub inner: GraphDevices,
    pub output_shapes: Vec<Vec<usize>>,
}

impl Runner {
    pub fn new(graph: Graph, policy: DevicePolicy) -> Self {
        let (output_shapes, _) = Compiled::outputs_of(&graph);
        Self {
            inner: GraphDevices::with_policy(graph, policy),
            output_shapes,
        }
    }

    /// A runner with the device policy from `RLX_DEVICE*`.
    pub fn from_env(graph: Graph) -> Self {
        let (output_shapes, _) = Compiled::outputs_of(&graph);
        Self {
            inner: GraphDevices::from_env(graph),
            output_shapes,
        }
    }

    pub fn devices(&self) -> Vec<&'static str> {
        self.inner.devices().iter().map(|d| label(*d)).collect()
    }

    pub fn fastest(&self) -> &'static str {
        label(self.inner.fastest())
    }

    /// The device a run with `hint` would use.
    pub fn resolve(&mut self, hint: Option<&str>) -> ApiResult<&'static str> {
        let hint = optional(hint)?;
        self.inner
            .resolve(hint)
            .map(label)
            .map_err(ApiError::runtime)
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }

    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> ApiResult<()> {
        let dtype = crate::enums::dtype(dtype)?;
        self.inner.set_param_typed(name, data, dtype);
        Ok(())
    }

    /// Set several f32 parameters; returns how many.
    pub fn set_params(&mut self, params: &[(&str, &[f32])]) -> usize {
        for (name, data) in params {
            self.inner.set_param(name, data);
        }
        params.len()
    }

    /// Compile every viable device now; returns which.
    pub fn warm_all(&mut self) -> ApiResult<Vec<&'static str>> {
        let devices = catch("warm_all", || self.inner.warm_all())?.map_err(ApiError::runtime)?;
        Ok(devices.into_iter().map(label).collect())
    }

    /// Time each device on `inputs`: `[{device, compile_ms, exec_ms}]`.
    pub fn benchmark(&mut self, inputs: &[(&str, &[f32])], runs: usize) -> ApiResult<Value> {
        let results = catch("benchmark", || self.inner.benchmark(inputs, runs))?
            .map_err(ApiError::runtime)?;
        Ok(Value::Array(
            results
                .into_iter()
                .map(|r| {
                    json!({
                        "device": r.label,
                        "compile_ms": r.compile_ns as f64 / 1e6,
                        "exec_ms": r.median_exec_ns as f64 / 1e6,
                    })
                })
                .collect(),
        ))
    }

    /// Run on the resolved device; returns it with the outputs.
    pub fn run(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<&str>,
    ) -> ApiResult<(&'static str, Vec<Vec<f32>>)> {
        let hint = optional(hint)?;
        let device = self.inner.resolve(hint).map_err(ApiError::runtime)?;
        let outs = catch("run", || self.inner.run(device, inputs))?.map_err(ApiError::runtime)?;
        Ok((label(device), outs))
    }

    /// Run on exactly `device`, no resolution.
    pub fn run_on(&mut self, device: &str, inputs: &[(&str, &[f32])]) -> ApiResult<Vec<Vec<f32>>> {
        let device = parse(device)?;
        catch("run_on", || self.inner.run(device, inputs))?.map_err(ApiError::runtime)
    }

    /// Try each device of `chain` in order until one runs; returns which.
    pub fn run_try(
        &mut self,
        chain: &[String],
        inputs: &[(&str, &[f32])],
    ) -> ApiResult<(&'static str, Vec<Vec<f32>>)> {
        let devices: Vec<Device> = chain.iter().map(|d| parse(d)).collect::<ApiResult<_>>()?;
        let (d, outs) = catch("run_try", || self.inner.run_try(&devices, inputs))?
            .map_err(|e| ApiError::runtime(e.to_string()))?;
        Ok((label(d), outs))
    }

    /// Run on `hint` (or the resolved device), falling back down the policy.
    pub fn run_chain(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<&str>,
    ) -> ApiResult<(&'static str, Vec<Vec<f32>>)> {
        let hint = optional(hint)?;
        let (d, outs) = catch("run_chain", || self.inner.run_chain(hint, inputs))?
            .map_err(|e| ApiError::runtime(e.to_string()))?;
        Ok((label(d), outs))
    }

    /// Resolve with the inputs in hand (input-shape-aware device choice), then run.
    pub fn run_with_inputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<&str>,
    ) -> ApiResult<Vec<Vec<f32>>> {
        let hint = optional(hint)?;
        catch("run_with_inputs", || {
            self.inner.run_resolved_with_inputs(hint, inputs)
        })?
        .map_err(ApiError::runtime)
    }

    /// Candidates for this graph: `[{device, available, supports_graph, recommended, blocker}]`.
    pub fn report(&self) -> Value {
        Value::Array(
            self.inner
                .report()
                .into_iter()
                .map(|c| {
                    json!({
                        "device": c.label,
                        "available": c.available,
                        "supports_graph": c.supports_graph,
                        "recommended": c.recommended,
                        "blocker": c.blocker,
                    })
                })
                .collect(),
        )
    }
}

/// Route each run to a device by policy, falling back on failure.
pub struct Router {
    pub inner: DeviceRouter,
    pub output_shapes: Vec<Vec<usize>>,
}

impl Router {
    pub fn new(graph: Graph, policy: DevicePolicy) -> ApiResult<Self> {
        let (output_shapes, _) = Compiled::outputs_of(&graph);
        let inner =
            catch("Router", || DeviceRouter::new(graph, policy))?.map_err(ApiError::runtime)?;
        Ok(Self {
            inner,
            output_shapes,
        })
    }

    /// A router with the device policy from `RLX_DEVICE*`.
    pub fn from_env(graph: Graph) -> ApiResult<Self> {
        let (output_shapes, _) = Compiled::outputs_of(&graph);
        let inner =
            catch("Router", || DeviceRouter::from_env(graph))?.map_err(ApiError::runtime)?;
        Ok(Self {
            inner,
            output_shapes,
        })
    }

    pub fn devices(&self) -> Vec<&'static str> {
        self.inner.devices()
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.inner.set_param(name, data);
    }

    /// Set several f32 parameters; returns how many.
    pub fn set_params(&mut self, params: &[(&str, &[f32])]) -> usize {
        for (name, data) in params {
            self.inner.set_param(name, data);
        }
        params.len()
    }

    pub fn set_param_typed(&mut self, name: &str, data: &[u8], dtype: &str) -> ApiResult<()> {
        let dtype = crate::enums::dtype(dtype)?;
        self.inner.set_param_typed(name, data, dtype);
        Ok(())
    }

    /// Run on exactly `device`, no policy.
    pub fn run_on(&mut self, device: &str, inputs: &[(&str, &[f32])]) -> ApiResult<Vec<Vec<f32>>> {
        let device = parse(device)?;
        catch("run_on", || self.inner.run_on(device, inputs))?.map_err(ApiError::runtime)
    }

    pub fn set_rebench_on_throttle(&mut self, enabled: bool) {
        self.inner.set_rebench_on_throttle(enabled);
    }

    /// Run on the policy's device (or `hint`).
    pub fn run(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<&str>,
    ) -> ApiResult<(&'static str, Vec<Vec<f32>>)> {
        let hint = optional(hint)?;
        let (d, outs) =
            catch("run", || self.inner.run(inputs, hint))?.map_err(ApiError::runtime)?;
        Ok((label(d), outs))
    }

    /// Run, falling back down the policy's chain when a device fails.
    pub fn run_chain(
        &mut self,
        inputs: &[(&str, &[f32])],
        hint: Option<&str>,
    ) -> ApiResult<(&'static str, Vec<Vec<f32>>)> {
        let hint = optional(hint)?;
        let (d, outs) = catch("run_chain", || self.inner.run_chain(inputs, hint))?
            .map_err(|e| ApiError::runtime(e.to_string()))?;
        Ok((label(d), outs))
    }
}
