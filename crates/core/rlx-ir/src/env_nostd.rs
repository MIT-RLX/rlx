// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `no_std` stand-in for the `env` module: there is no process environment, so
//! every `RLX_*` switch reads as unset.
use crate::prelude::*;

pub fn var(_key: &str) -> Option<String> {
    None
}

pub fn flag(_key: &str) -> bool {
    false
}
