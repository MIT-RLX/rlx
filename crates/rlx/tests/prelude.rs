// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `use rlx::prelude::*` next to the caller's own glob imports.
//!
//! A prelude is imported with `*`, usually beside other `*` imports. Two
//! globs naming the same item make every use of it ambiguous (E0659), so a
//! prelude must not export names as generic as `var` or `set`: it used to
//! re-export `rlx::ir::env::{var, set, unset, flag}`, and this file did not
//! compile. Settings stay reachable as `env::…`.

mod caller {
    pub fn var() -> u32 {
        1
    }
    pub fn set() -> u32 {
        2
    }
    pub fn flag() -> u32 {
        4
    }
    pub fn unset() -> u32 {
        8
    }
}

use caller::*;
use rlx::prelude::*;

#[test]
fn generic_names_stay_the_callers() {
    assert_eq!(var() + set() + flag() + unset(), 15);
    // The settings module is still in the prelude.
    assert!(env::var("RLX_PRELUDE_TEST_UNSET_KEY").is_none());
}

#[test]
fn result_keeps_std_two_parameter_form() {
    fn std_style() -> Result<u8, std::num::ParseIntError> {
        "7".parse::<u8>()
    }
    fn anyhow_style() -> Result<Device> {
        Ok(parse_device("cpu")?)
    }
    assert_eq!(std_style().unwrap(), 7);
    assert_eq!(anyhow_style().unwrap(), Device::Cpu);
}

#[test]
fn compiled_graph_introspection_is_in_scope() {
    // `CompiledGraph::{inputs, params, outputs}` return `TensorInfo`.
    let info = TensorInfo {
        name: "x".into(),
        dtype: DType::F32,
        shape: Some(vec![1, 4]),
    };
    assert_eq!(info.shape.as_deref(), Some(&[1, 4][..]));
}
