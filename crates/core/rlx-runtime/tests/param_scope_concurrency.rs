// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two models compiled at the same time on two threads, each under its own
//! param-sharing scope, with a large weight of the same name and size but
//! different values. The scope is process-wide and a shared weight is matched
//! by name, dtype and size, so without serialization one thread's compile can
//! land in the other's scope and map the other model's weight pages.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

const K: usize = 256;
const N: usize = 256;

fn graph() -> Graph {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("param_scope_concurrency");
    let x = g.input("x", f(&[1, K]));
    let w = g.param("w", f(&[K, N]));
    let y = g.add_node(Op::MatMul, vec![x, w], f(&[1, N]));
    g.set_outputs(vec![y]);
    g
}

fn weight(salt: usize) -> Vec<f32> {
    (0..K * N)
        .map(|i| (((i * 31 + salt * 977) % 101) as f32 - 50.0) / 101.0)
        .collect()
}

fn expected(x: &[f32], w: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|n| (0..K).map(|k| x[k] * w[k * N + n]).sum())
        .collect()
}

#[test]
fn concurrent_scopes_keep_their_own_weights() {
    let x: Vec<f32> = (0..K).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let pid = std::process::id();
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let x = x.clone();
            std::thread::spawn(move || {
                let w = weight(t);
                let want = expected(&x, &w);
                let label = format!("scope-race-{t}:{pid}");
                for round in 0..24 {
                    let mut c = rlx_runtime::with_param_sharing_scope(&label, || {
                        Session::new(Device::Cpu).compile(graph())
                    });
                    if !c.param_already_set("w") {
                        c.set_param("w", &w);
                    }
                    let got = c.run(&[("x", &x)]).remove(0);
                    let d = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(d < 1e-3, "thread {t} round {round}: off by {d}");
                }
            })
        })
        .collect();
    for t in threads {
        t.join().expect("thread panicked");
    }
}
