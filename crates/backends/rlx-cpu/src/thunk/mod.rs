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

//! Thunks — pre-compiled kernel dispatch with zero per-call overhead.
//!
//! At compile time, the graph is lowered into a flat `Vec<Thunk>` where each
//! thunk holds pre-computed arena offsets, dimensions, and kernel type.
//! At runtime, the executor just iterates thunks and calls kernels directly.

// Edition 2024: bodies of `unsafe fn` are safe by default; `sl`/`sl_mut` stay `unsafe fn`.
#![allow(unsafe_op_in_unsafe_fn)]
//! No match dispatch, no HashMap lookup, no dimension computation.

use crate::arena::Arena;
use crate::op_registry::CpuKernel;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{Graph, NodeId, Op, Shape};
use std::collections::HashMap;
use std::sync::Arc;

mod types;
pub use types::*;

mod compile_dispatch;
pub use compile_dispatch::*;

mod exec_dispatch;
pub use exec_dispatch::*;

mod ops;
pub use ops::*;

mod scan;
pub use scan::*;

mod host_op;
pub use host_op::*;

mod indexing_host;
pub use indexing_host::*;

#[cfg(test)]
mod tests;
