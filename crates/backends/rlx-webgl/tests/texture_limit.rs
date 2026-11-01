// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plans keep every texture within `RLX_WEBGL_MAX_TEXTURE` (a browser's
//! `MAX_TEXTURE_SIZE`; 4096–8192 on some phones), not only a desktop's 16384:
//! tall slots widen to at most that many columns. They widened to 8192
//! regardless, so on a 4096 device no tensor past 8192 rows ran at all. The
//! switch is process-global, so this file is one test.

use rlx_ir::{DType, Graph, Shape};

#[test]
fn widened_slots_keep_to_the_texture_limit() {
    // a volume of 6 channels, 14.7 M texels: within 4096² = 16.8 M
    let dims = [1, 6, 96, 160, 160];
    let build = || {
        let mut g = Graph::new("tall");
        let sh = Shape::new(&dims, DType::F32);
        let x = g.input("x", sh.clone());
        let y = g.activation(rlx_ir::op::Activation::Relu, x, sh.clone());
        let z = g.activation(rlx_ir::op::Activation::Elu, y, sh);
        g.set_outputs(vec![z]);
        g
    };
    for side in [16384usize, 8192, 4096] {
        rlx_ir::env::set("RLX_WEBGL_MAX_TEXTURE", side.to_string());
        let plan = rlx_webgl::build_plan(&build()).expect("plan");
        let (origin, rows, cols) = plan.largest_texture().unwrap();
        eprintln!("  limit {side}: largest {rows} x {cols} ({origin})");
        assert!(
            rows.max(cols) <= side,
            "limit {side}: {origin} is {rows} x {cols}"
        );
    }
    rlx_ir::env::clear_overrides();
}
