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

//! vmap (batched function transform) over linalg custom ops.

#![cfg(feature = "cpu")]

use rlx_ir::{DType, Graph, Shape};
use rlx_opt::vmap::vmap;
use rlx_runtime::{Device, Session};

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn vmap_diag_extract_batches_correctly() {
    rlx_linalg::register();
    let n = 3;
    let batch = 2;

    let mut g = Graph::new("vm_de");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let d = rlx_linalg::diag_extract(&mut g, a);
    g.set_outputs(vec![d]);

    let vg = vmap(&g, &["a"], batch);
    let mut c = Session::new(Device::Cpu).compile(vg);

    // Two 3×3 matrices in a batch.
    let a_data = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // batch 1
        10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0,
    ];
    let outs = c.run_typed(&[("a", &f64s_to_bytes(&a_data), DType::F64)]);
    let d_got = bytes_to_f64s(&outs[0].0);
    let want = [1.0, 5.0, 9.0, 10.0, 20.0, 30.0];
    assert_eq!(d_got.len(), batch * n);
    for i in 0..(batch * n) {
        assert!(
            (d_got[i] - want[i]).abs() < 1e-12,
            "vmap[diag][{i}]={} want {}",
            d_got[i],
            want[i]
        );
    }
}

#[test]
fn vmap_diag_set_batches_correctly() {
    rlx_linalg::register();
    let n = 3;
    let batch = 2;
    let mut g = Graph::new("vm_ds");
    let v = g.input("v", Shape::new(&[n], DType::F64));
    let m = rlx_linalg::diag_set(&mut g, v);
    g.set_outputs(vec![m]);

    let vg = vmap(&g, &["v"], batch);
    let mut c = Session::new(Device::Cpu).compile(vg);

    let v_data = vec![
        // batch 0
        2.0, 3.0, 5.0, // batch 1
        7.0, 11.0, 13.0,
    ];
    let outs = c.run_typed(&[("v", &f64s_to_bytes(&v_data), DType::F64)]);
    let m_got = bytes_to_f64s(&outs[0].0);
    assert_eq!(m_got.len(), batch * n * n);
    let want = vec![
        // batch 0
        2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, // batch 1
        7.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 13.0,
    ];
    for i in 0..(batch * n * n) {
        assert!(
            (m_got[i] - want[i]).abs() < 1e-12,
            "vmap[ds][{i}]={} want {}",
            m_got[i],
            want[i]
        );
    }
}

#[test]
fn vmap_trace_via_composition() {
    // trace = diag_extract + sum. If diag_extract has a vmap rule
    // (which it does) and Op::Reduce::Sum has the built-in vmap rule
    // (axes shifted by 1), trace should be batchable end-to-end.
    rlx_linalg::register();
    let n = 3;
    let batch = 2;
    let mut g = Graph::new("vm_tr");
    let a = g.input("a", Shape::new(&[n, n], DType::F64));
    let t = rlx_linalg::trace(&mut g, a);
    g.set_outputs(vec![t]);

    let vg = vmap(&g, &["a"], batch);
    let mut c = Session::new(Device::Cpu).compile(vg);

    let a_data = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        // batch 1: trace = 1+5+9 = 15 and 2+4+6 = 12
        1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 6.0,
    ];
    let outs = c.run_typed(&[("a", &f64s_to_bytes(&a_data), DType::F64)]);
    let t_got = bytes_to_f64s(&outs[0].0);
    assert!(
        (t_got[0] - 15.0).abs() < 1e-12,
        "vmap[trace][0]={}",
        t_got[0]
    );
    assert!(
        (t_got[1] - 11.0).abs() < 1e-12,
        "vmap[trace][1]={}",
        t_got[1]
    );
}

#[test]
fn vmap_unrelated_op_without_rule_panics_clearly() {
    // A custom op without a vmap rule should panic with a message
    // that mentions the op name. Exercise the failure path on
    // `logdet` (no vmap rule) — but the panic is the test condition.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rlx_linalg::register();
        let n = 3;
        let mut g = Graph::new("vm_logdet_panic");
        let a = g.input("a", Shape::new(&[n, n], DType::F64));
        let l = rlx_linalg::logdet(&mut g, a);
        g.set_outputs(vec![l]);
        let _ = vmap(&g, &["a"], 2);
    }));
    assert!(result.is_err(), "expected vmap to panic on logdet");
}
