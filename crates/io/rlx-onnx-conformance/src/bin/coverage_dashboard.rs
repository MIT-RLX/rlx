// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use rlx_onnx_conformance::backend_runner::coverage_dashboard;

fn main() {
    println!("{}", coverage_dashboard());
}
