// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Runtime support for code emitted by [`crate::emit_codegen`].
//!
//! The codegen emitter produces statements that reference a builder binding
//! `b` (a [`GraphBuilder`]), the free function [`shape_from_meta`], and the
//! `opts` import options. Those symbols live here so emitted modules are real,
//! compilable Rust — the [`crate::emit_codegen::emit_graph_source`] assembler
//! stitches a preamble that brings them into scope.

use std::collections::HashMap;

use anyhow::{Context, Result};
use rlx_ir::hir::{HirModule, HirMut, HirNodeId};
use rlx_ir::{DType, HirGraphExt, Op, Shape};

use crate::lower::{ImportOptions, resolve_shape};

// Re-export the support crates so generated modules need only depend on
// `rlx-onnx-import` + `rlx-ir` by path — no registry resolution required, which
// keeps the codegen compile harness usable offline.
pub use ::anyhow;
pub use ::serde_json;

/// Resolve an output-meta JSON blob (`{"shape": [...], "dtype": "..."}`) to a
/// concrete [`Shape`]. Mirrors the import path's shape resolution so a generated
/// module and direct import agree on every node's output shape.
pub fn shape_from_meta(meta: &serde_json::Value, opts: &ImportOptions) -> Shape {
    resolve_shape(meta, opts).unwrap_or_else(|_| Shape::new(&[1], DType::F32))
}

/// Mutable graph being assembled by emitted code. Holds the [`HirModule`] under
/// construction, the float initializer table (`params`), and a name→node map so
/// `b.tensor("name")` resolves operands across statements.
pub struct GraphBuilder {
    pub hir: HirModule,
    pub params: HashMap<String, Vec<f32>>,
    pub i64_params: HashMap<String, Vec<i64>>,
    env: HashMap<String, HirNodeId>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            hir: HirModule::new("onnx_codegen"),
            params: HashMap::new(),
            i64_params: HashMap::new(),
            env: HashMap::new(),
        }
    }

    /// Register a graph input and bind it to `name`.
    pub fn input(&mut self, name: &str, shape: Shape) -> HirNodeId {
        let id = {
            let mut m = HirMut::new(&mut self.hir);
            m.input(name, shape)
        };
        self.env.insert(name.to_string(), id);
        id
    }

    /// Register an F32 constant initializer and bind it to `name`.
    pub fn constant_f32(&mut self, name: &str, data: Vec<f32>, dims: &[usize]) -> HirNodeId {
        let shape = Shape::new(dims, DType::F32);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let id = {
            let mut m = HirMut::new(&mut self.hir);
            m.add_node(Op::Constant { data: bytes }, vec![], shape)
        };
        self.params.insert(name.to_string(), data);
        self.env.insert(name.to_string(), id);
        id
    }

    /// Register an I64 constant initializer and bind it to `name`.
    pub fn constant_i64(&mut self, name: &str, data: Vec<i64>, dims: &[usize]) -> HirNodeId {
        let shape = Shape::new(dims, DType::I64);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let id = {
            let mut m = HirMut::new(&mut self.hir);
            m.add_node(Op::Constant { data: bytes }, vec![], shape)
        };
        self.i64_params.insert(name.to_string(), data);
        self.env.insert(name.to_string(), id);
        id
    }

    /// Look up a previously bound tensor by name (used by emitted operand refs).
    pub fn tensor(&self, name: &str) -> Result<HirNodeId> {
        self.env
            .get(name)
            .copied()
            .with_context(|| format!("emit_runtime: tensor not bound: {name}"))
    }

    /// Bind a node id to an output name (the assembler calls this after each
    /// emitted node body so downstream `b.tensor(..)` lookups succeed).
    pub fn bind(&mut self, name: &str, id: HirNodeId) {
        self.env.insert(name.to_string(), id);
    }

    /// One `(name, num_inputs, attrs_len, actual_inputs)` row per `Op::Custom`
    /// node, in build order — the inspection surface the compile harness asserts
    /// against to prove emitted wiring is correct.
    pub fn custom_summary(&self) -> Vec<(String, u32, usize, usize)> {
        use rlx_ir::hir::HirOp;
        self.hir
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                HirOp::Mir(Op::Custom {
                    name,
                    num_inputs,
                    attrs,
                }) => Some((name.clone(), *num_inputs, attrs.len(), n.inputs.len())),
                _ => None,
            })
            .collect()
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}
