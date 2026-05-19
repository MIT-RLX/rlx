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

//! Repro: Conv2dBackwardInput / Conv2dBackwardWeight produce correct
//! outputs in `MlxMode::Lazy` but diverge wildly under `MlxMode::Compiled`.
//! This isolates the issue from the bench harness so we can iterate on
//! a fix.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

fn close(got: &[f32], want: &[f32], tol: f32) -> (bool, f32) {
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    (got.len() == want.len() && max_abs <= tol, max_abs)
}

#[test]
fn conv2d_backward_input_compiled_mode_matches_lazy() {
    let n = 2usize;
    let ci = 4usize;
    let h = 6usize;
    let co = 3usize;
    let k = 3usize;
    let h_out = h; // s=1, p=1, k=3 → h_out=h

    let mut g = Graph::new("conv_bwd_in");
    let dy = g.input("dy", Shape::new(&[n, co, h_out, h_out], DType::F32));
    let w = g.input("w", Shape::new(&[co, ci, k, k], DType::F32));
    let dx = g.conv2d_backward_input(
        dy,
        w,
        Shape::new(&[n, ci, h, h], DType::F32),
        vec![k, k],
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
        1,
    );
    g.set_outputs(vec![dx]);

    let dys: Vec<f32> = (0..n * co * h_out * h_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();
    let ws: Vec<f32> = (0..co * ci * k * k)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();

    let mut lazy = MlxExecutable::compile_with_mode(g.clone(), MlxMode::Lazy);
    let lazy_out = lazy
        .run(&[("dy", &dys), ("w", &ws)])
        .into_iter()
        .next()
        .unwrap();

    let mut compiled = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
    let compiled_out = compiled
        .run(&[("dy", &dys), ("w", &ws)])
        .into_iter()
        .next()
        .unwrap();

    assert_eq!(
        lazy_out.len(),
        compiled_out.len(),
        "lazy_out.len()={} compiled_out.len()={}",
        lazy_out.len(),
        compiled_out.len()
    );
    let (ok, max_abs) = close(&compiled_out, &lazy_out, 1e-4);
    assert!(
        ok,
        "Compiled-mode Conv2dBackwardInput diverges from lazy: max_abs={max_abs:.3e}\n\
         lazy[0..8]     = {:?}\n\
         compiled[0..8] = {:?}",
        &lazy_out[0..8.min(lazy_out.len())],
        &compiled_out[0..8.min(compiled_out.len())]
    );
}

#[test]
fn conv2d_backward_weight_compiled_mode_matches_lazy() {
    let n = 2usize;
    let ci = 4usize;
    let h = 6usize;
    let co = 3usize;
    let k = 3usize;
    let h_out = h;

    let mut g = Graph::new("conv_bwd_w");
    let x = g.input("x", Shape::new(&[n, ci, h, h], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, co, h_out, h_out], DType::F32));
    let dw = g.conv2d_backward_weight(
        x,
        dy,
        Shape::new(&[co, ci, k, k], DType::F32),
        vec![k, k],
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
        1,
    );
    g.set_outputs(vec![dw]);

    let xs: Vec<f32> = (0..n * ci * h * h)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let dys: Vec<f32> = (0..n * co * h_out * h_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();

    let mut lazy = MlxExecutable::compile_with_mode(g.clone(), MlxMode::Lazy);
    let lazy_out = lazy
        .run(&[("x", &xs), ("dy", &dys)])
        .into_iter()
        .next()
        .unwrap();

    let mut compiled = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
    let compiled_out = compiled
        .run(&[("x", &xs), ("dy", &dys)])
        .into_iter()
        .next()
        .unwrap();

    let (ok, max_abs) = close(&compiled_out, &lazy_out, 1e-4);
    assert!(
        ok,
        "Compiled-mode Conv2dBackwardWeight diverges from lazy: max_abs={max_abs:.3e}\n\
         lazy[0..8]     = {:?}\n\
         compiled[0..8] = {:?}",
        &lazy_out[0..8.min(lazy_out.len())],
        &compiled_out[0..8.min(compiled_out.len())]
    );
}

#[test]
fn conv2d_backward_input_compiled_mode_second_call_stable() {
    // If the compiled trace captures stale data on first call, the
    // second call (different inputs) should diverge differently or
    // even stay frozen at the first call's output.
    let n = 2usize;
    let ci = 4usize;
    let h = 6usize;
    let co = 3usize;
    let k = 3usize;
    let h_out = h;

    let mut g = Graph::new("conv_bwd_in");
    let dy = g.input("dy", Shape::new(&[n, co, h_out, h_out], DType::F32));
    let w = g.input("w", Shape::new(&[co, ci, k, k], DType::F32));
    let dx = g.conv2d_backward_input(
        dy,
        w,
        Shape::new(&[n, ci, h, h], DType::F32),
        vec![k, k],
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
        1,
    );
    g.set_outputs(vec![dx]);

    let dys_1: Vec<f32> = (0..n * co * h_out * h_out)
        .map(|i| 0.07 * (i as f32) - 0.3)
        .collect();
    let ws_1: Vec<f32> = (0..co * ci * k * k)
        .map(|i| 0.05 * (i as f32) - 0.4)
        .collect();
    let dys_2: Vec<f32> = (0..n * co * h_out * h_out)
        .map(|i| 0.13 * (i as f32) + 0.1)
        .collect();
    let ws_2: Vec<f32> = (0..co * ci * k * k)
        .map(|i| 0.02 * (i as f32) + 0.5)
        .collect();

    // Lazy: ground truth for both inputs.
    let mut lazy = MlxExecutable::compile_with_mode(g.clone(), MlxMode::Lazy);
    let lazy_1 = lazy
        .run(&[("dy", &dys_1), ("w", &ws_1)])
        .into_iter()
        .next()
        .unwrap();
    let lazy_2 = lazy
        .run(&[("dy", &dys_2), ("w", &ws_2)])
        .into_iter()
        .next()
        .unwrap();

    // Compiled: compare against lazy for each input.
    let mut compiled = MlxExecutable::compile_with_mode(g, MlxMode::Compiled);
    let comp_1 = compiled
        .run(&[("dy", &dys_1), ("w", &ws_1)])
        .into_iter()
        .next()
        .unwrap();
    let comp_2 = compiled
        .run(&[("dy", &dys_2), ("w", &ws_2)])
        .into_iter()
        .next()
        .unwrap();

    let (ok1, max1) = close(&comp_1, &lazy_1, 1e-4);
    let (ok2, max2) = close(&comp_2, &lazy_2, 1e-4);
    let (frozen, _) = close(&comp_1, &comp_2, 1e-9);

    assert!(
        ok1 && ok2,
        "Compiled-mode diverges. call1 max_abs={max1:.3e}, call2 max_abs={max2:.3e}, \
         compiled outputs identical across calls = {frozen}"
    );
}
