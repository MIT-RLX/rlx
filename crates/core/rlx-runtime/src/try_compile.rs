// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`Session::try_compile`]: compile, or say why not, without panicking.
//!
//! [`Session::compile`] panics on an invalid graph, an unavailable device, or
//! an op the backend cannot lower, and the panic message is whatever the
//! failing layer printed. This checks the same things in order and returns a
//! [`CompileError`] that names the node:
//!
//! 1. the IR verifier ([`rlx_ir::verify_all`]);
//! 2. the device is compiled in and present;
//! 3. every op lowers on that device after the backend's own rewrites
//!    ([`crate::first_unsupported_op_with_options`]);
//! 4. the backend compile itself — a panic there becomes
//!    [`CompileError::Backend`].

use std::panic::{AssertUnwindSafe, catch_unwind};

use rlx_ir::{Graph, NodeId, OpKind, VerifyError};

use crate::{CompileOptions, CompiledGraph, Device, Session};

/// Why a graph did not compile.
#[derive(Debug)]
pub enum CompileError {
    /// The graph fails IR verification.
    Invalid(Vec<VerifyError>),
    /// The device is not built into this binary, or not present on this host.
    DeviceUnavailable {
        device: Device,
        compiled_in: bool,
        /// Why a built-in device is absent, when the backend can say.
        reason: Option<String>,
    },
    /// An op the device cannot lower, even after rewriting.
    Unsupported {
        device: Device,
        node: NodeId,
        op: OpKind,
        name: Option<String>,
    },
    /// The backend panicked while compiling.
    Backend(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Invalid(errs) => {
                write!(f, "rlx: the graph is invalid ({} error(s))", errs.len())?;
                for e in errs.iter().take(5) {
                    write!(f, "\n  {e}")?;
                }
                if errs.len() > 5 {
                    write!(f, "\n  … {} more", errs.len() - 5)?;
                }
                Ok(())
            }
            CompileError::DeviceUnavailable {
                device,
                compiled_in,
                reason,
            } => {
                if *compiled_in {
                    write!(
                        f,
                        "rlx: {device:?} is built in but not available on this host"
                    )?;
                    match reason {
                        Some(r) => write!(f, ": {r}"),
                        None => Ok(()),
                    }
                } else {
                    write!(
                        f,
                        "rlx: {device:?} is not built into this binary — enable its cargo \
                         feature (e.g. `--features {}`)",
                        device.as_arg()
                    )
                }
            }
            CompileError::Unsupported {
                device,
                node,
                op,
                name,
            } => {
                write!(f, "rlx: {device:?} cannot lower {op:?} (node {node:?}")?;
                if let Some(n) = name {
                    write!(f, " `{n}`")?;
                }
                write!(
                    f,
                    "); try another device, or `cargo rlx check` for the dispatch report"
                )
            }
            CompileError::Backend(msg) => write!(f, "rlx: backend compile failed: {msg}"),
        }
    }
}

impl std::error::Error for CompileError {}

fn unavailable(device: Device) -> CompileError {
    let compiled_in = crate::feature_compiled(device);
    CompileError::DeviceUnavailable {
        device,
        compiled_in,
        reason: compiled_in
            .then(|| crate::device_ext::unavailable_reason(device))
            .flatten(),
    }
}

impl Session {
    /// [`Self::new`] that returns [`CompileError::DeviceUnavailable`] instead
    /// of panicking when `device` is not built in or not present.
    pub fn try_new(device: Device) -> Result<Self, CompileError> {
        if !crate::is_available(device) {
            return Err(unavailable(device));
        }
        Ok(Self::new(device))
    }

    /// [`Self::compile`] that returns a [`CompileError`] instead of panicking.
    ///
    /// ```no_run
    /// # use rlx_runtime::{Session, Device};
    /// # let graph = rlx_ir::Graph::new("g");
    /// match Session::new(Device::Metal).try_compile(graph) {
    ///     Ok(mut c) => { /* c.run(..) */ }
    ///     Err(e) => eprintln!("{e}"),
    /// }
    /// ```
    pub fn try_compile(&self, graph: Graph) -> Result<CompiledGraph, CompileError> {
        self.try_compile_with(graph, &CompileOptions::default())
    }

    /// [`Self::compile_with`] that returns a [`CompileError`] instead of
    /// panicking.
    pub fn try_compile_with(
        &self,
        graph: Graph,
        options: &CompileOptions,
    ) -> Result<CompiledGraph, CompileError> {
        // `verify_all`: the full checker the compile pipeline runs after each
        // pass. Run up front, its errors are about the caller's graph instead
        // of being attributed to whichever pass ran first.
        let errs = rlx_ir::verify_all(&graph);
        if !errs.is_empty() {
            return Err(CompileError::Invalid(errs));
        }
        let device = self.device();
        if !crate::is_available(device) {
            return Err(unavailable(device));
        }
        if let Some((idx, op)) = crate::first_unsupported_op_with_options(device, &graph, options) {
            let node = &graph.nodes()[idx];
            return Err(CompileError::Unsupported {
                device,
                node: node.id,
                op: op.kind(),
                name: node.name.clone(),
            });
        }
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = catch_unwind(AssertUnwindSafe(|| self.compile_with(graph, options)));
        std::panic::set_hook(prev);
        r.map_err(|p| {
            CompileError::Backend(
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic payload>".into()),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, GraphExt, Op, Shape};

    fn ok_graph() -> Graph {
        let mut g = Graph::new("ok");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        let y = g.relu(x);
        g.set_outputs(vec![y]);
        g
    }

    #[test]
    fn compiles_a_valid_graph() {
        let mut c = Session::new(Device::Cpu).try_compile(ok_graph()).unwrap();
        assert_eq!(
            c.run(&[("x", &[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])])[0][1],
            2.0
        );
    }

    #[test]
    fn reports_an_invalid_graph() {
        let mut g = Graph::new("bad");
        let x = g.input("x", Shape::new(&[2, 3], DType::F32));
        let w = g.param("w", Shape::new(&[4, 5], DType::F32));
        // Inner dims disagree; the declared shape is a lie the verifier catches.
        let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[2, 5], DType::F32));
        g.set_outputs(vec![y]);
        let e = Session::new(Device::Cpu)
            .try_compile(g)
            .err()
            .expect("must fail");
        assert!(matches!(e, CompileError::Invalid(_)), "{e}");
        assert!(
            e.to_string().starts_with("rlx: the graph is invalid"),
            "{e}"
        );
    }

    #[test]
    fn reports_a_device_not_built_in() {
        let _ = ok_graph;
        // TPU is never in this crate's CPU-only test build.
        if crate::feature_compiled(Device::Tpu) {
            return;
        }
        let e = Session::try_new(Device::Tpu).err().expect("must fail");
        assert!(e.to_string().contains("not built into this binary"), "{e}");
    }
}
