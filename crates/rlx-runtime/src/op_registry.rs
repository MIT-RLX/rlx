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

//! Op registry — re-exported from `rlx-ir`.
//!
//! The registry was promoted to `rlx-ir` once `Op::Custom` landed: the
//! IR layer needs to dispatch through it during shape inference, so
//! the registry can no longer live above the IR. This module is kept
//! as a thin re-export for backward compatibility with downstream
//! code that imported from `rlx_runtime::op_registry`.

pub use rlx_ir::op_registry::{
    OpExtension, OpRegistry, VjpContext, global_registry, lookup_op, register_op,
};
