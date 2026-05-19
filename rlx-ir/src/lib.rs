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

//! RLX Tensor IR — the intermediate representation for the RLX ML compiler.
//!
//! This IR is:
//! - **Standalone**: no runtime, no backend, no framework coupling
//! - **Serializable**: graphs can be saved/loaded for AOT compilation
//! - **Optimizable**: designed for pattern-matching fusion and buffer planning
//!
//! The IR has three levels:
//! - [`Graph`]: a DAG of tensor operations (like XLA's HloModule)
//! - [`Node`]: a single operation with typed inputs/outputs
//! - [`Op`]: the operation kind with parameters

pub mod async_copy;
pub mod const_check;
pub mod dtype;
pub mod graph;
pub mod infer;
pub mod layout;
pub mod measure;
pub mod op;
pub mod op_registry;
pub mod ops;
pub mod perfetto;
pub mod pretty;
pub mod quant;
pub mod rng;
pub mod shape;
pub mod target;
pub mod verify;

pub use async_copy::{AsyncCopy, BarrierToken, DoubleBuffer, SyncCopy};
pub use dtype::{DType, Element, ElementSubtype};
pub use graph::{Graph, Node, NodeId};
pub use infer::GraphExt;
pub use layout::{Coord2, Ragged, ShapeTuple, Strides2, Strides3, Tile2, Tile3};
pub use measure::{CacheBuster, Tick, time_ns};
pub use op::{Op, OpKind};
pub use op_registry::{
    JvpContext, OpExtension, OpRegistry, VjpContext, VmapContext, global_registry, lookup_op,
    register_op,
};
pub use pretty::{pretty_print, pretty_stats};
pub use quant::{QuantMap, QuantScheme};
pub use rng::Philox4x32;
pub use shape::{Dim, DimBinding, Shape};
