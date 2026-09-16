// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Reshape(Narrow(x))` must lower and stay numerically correct on CoreML.
//!
//! MPSGraph mis-lowers this one shape: it folds the reshape onto the SLICE'S
//! PARENT and drops the offset, then fails module verification and aborts the
//! process from MPSGraphExecutable — a SIGABRT no caller can catch. rlx used to
//! refuse the graph outright, which cost six ports their CoreML support
//! (`rlx-bbb`, `rlx-brainbert`, `rlx-cbramod`, `rlx-decode`,
//! `rlx-devicepassport`, `rlx-duin`).
//!
//! It now lowers as a `gather` of `[start, start+len)`, which is not foldable
//! that way — the offsets live in an index tensor, so the reshape has nothing to
//! absorb.
//!
//! SCOPE, stated honestly: this pins that the gather lowering returns the RIGHT
//! SLICE — if the index math regresses, the numbers move and this fails. It does
//! NOT reproduce the original SIGABRT: disabling the substitution leaves these
//! two-op graphs passing, because the miscompile needs more surrounding graph
//! than a synthetic case carries. The six ports that hit it in the wild
//! (`rlx-bbb`, `rlx-brainbert`, `rlx-cbramod`, `rlx-decode`,
//! `rlx-devicepassport`, `rlx-duin`) are what cover the abort itself, via their
//! own parity tests.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn case(lead: usize, last: usize, start: usize, len: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if rlx_ir::env::skip_unless_device("ane", true, rlx_runtime::is_available(Device::Ane)) {
        eprintln!("skip: CoreML/ANE unavailable");
        return None;
    }
    let mut g = Graph::new("reshape_narrow");
    let x = g.input("x", Shape::new(&[lead, last], DType::F32));
    let n = g.narrow_(x, 1, start, len);
    // Re-split the leading extent and leave the sliced axis trailing — the exact
    // configuration `narrow_nodes_feeding_reshape` detects.
    let y = g.reshape_(n, vec![(lead / 2) as i64, 2, len as i64]);
    g.set_outputs(vec![y]);

    let xs: Vec<f32> = (0..lead * last).map(|i| i as f32).collect();
    let run = |d: Device| -> Vec<f32> {
        Session::new(d)
            .compile(g.clone())
            .run(&[("x", xs.as_slice())])
            .remove(0)
    };
    Some((run(Device::Cpu), run(Device::Ane)))
}

#[test]
fn reshape_of_narrow_keeps_the_offset() {
    // (lead, last, start, len): `len != last`, start != 0 — the miscompile shape.
    for (lead, last, start, len) in [(4usize, 8usize, 4usize, 4usize), (6, 10, 3, 5)] {
        let Some((cpu, coreml)) = case(lead, last, start, len) else {
            return;
        };
        assert_eq!(cpu.len(), coreml.len(), "length mismatch");
        let d = cpu
            .iter()
            .zip(&coreml)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("lead={lead} last={last} start={start} len={len}: max_abs={d:.3e}");
        assert!(
            d <= 1e-5,
            "CoreML returned a different slice than CPU (max_abs {d}) — the \
             reshape/slice fold dropped the offset again"
        );
    }
}
