// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `taps::NodeDigests` bisects to the first node whose value differs: the
//! caller's outputs are untouched, two runs of one input agree everywhere,
//! a changed input diverges first at the node that reads it, the text form
//! round-trips, and a subset taps only its nodes.

use std::rc::Rc;

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{Device, Session, taps};

fn graph() -> (Graph, [NodeId; 4]) {
    let f = DType::F32;
    let mut g = Graph::new("digests");
    let x = g.input("x", Shape::new(&[4, 8], f));
    let w = g.input("w", Shape::new(&[8, 3], f));
    let a = g.relu(x); // reads only x
    let wt = g.transpose_(w, vec![1, 0]);
    let at = g.transpose_(a, vec![1, 0]);
    let b = g.mm(wt, at); // [3, 4]
    let c = g.tanh(b);
    g.set_outputs(vec![c]);
    (g, [a, b, c, x])
}

fn run(tap: Rc<taps::NodeDigests>, x: &[f32]) -> Vec<f32> {
    let w: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
    taps::with_tap(tap, || {
        let (g, _) = graph();
        let mut c = Session::new(Device::Cpu).compile(g);
        c.finalize_params();
        c.run(&[("x", x), ("w", &w)]).remove(0)
    })
}

#[test]
fn digests_bisect_to_the_first_differing_node() {
    let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 4.0).collect();
    let (d1, d2) = (
        Rc::new(taps::NodeDigests::default()),
        Rc::new(taps::NodeDigests::default()),
    );
    let y1 = run(d1.clone(), &x);
    let y2 = run(d2.clone(), &x);
    assert_eq!(y1, y2);
    assert_eq!(y1.len(), 12, "the caller still sees only its own output");
    let (a, b) = (d1.digests(), d2.digests());
    assert!(a.len() >= 3, "{a:?}");
    assert_eq!(taps::first_divergence(&a, &b), None);

    // a different input: the first divergence is the first node reading it
    let mut x2 = x.clone();
    x2[31] += 1.0;
    let d3 = Rc::new(taps::NodeDigests::default());
    run(d3.clone(), &x2);
    let c = d3.digests();
    let (i, p, q) = taps::first_divergence(&a, &c).expect("a changed input diverges");
    let (_, [relu, ..]) = graph();
    assert_eq!((p.node, q.node), (relu, relu), "first at {i}: {p} vs {q}");

    // the text form round-trips
    let back = taps::NodeDigests::parse(&d1.to_text()).unwrap();
    assert_eq!(back, a);

    // a subset taps only its nodes
    let (_, [_, mm, _, _]) = graph();
    let d4 = Rc::new(taps::NodeDigests::subset([mm]));
    run(d4.clone(), &x);
    let s = d4.digests();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].node, mm);
    assert!(a.iter().any(|d| d.node == mm && d.hash == s[0].hash));
}

/// Two graphs of the same width compiled first, then run in the other order
/// (and one twice): each run is attributed to the graph that ran, which a
/// pairing by width alone gets wrong. On every backend at hand: the compile
/// tag is a constant output.
#[test]
fn runs_pair_with_the_compile_that_ran() {
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut devices = vec![Device::Cpu];
    #[cfg(feature = "gpu")]
    if rlx_runtime::is_available(Device::Gpu) {
        devices.push(Device::Gpu);
    }
    for device in devices {
        let make = |scale: f32| {
            let mut g = Graph::new("pair");
            let x = g.input("x", Shape::new(&[8], DType::F32));
            let s = g.add_node(
                rlx_ir::Op::Constant {
                    data: [scale; 8].iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
                vec![],
                Shape::new(&[8], DType::F32),
            );
            let y = g.mul(x, s);
            let z = g.relu(y);
            g.set_outputs(vec![z]);
            g
        };
        let x: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let d = Rc::new(taps::NodeDigests::default());
        let (ya, yb) = taps::with_tap(d.clone(), || {
            let mut a = Session::new(device).compile(make(2.0));
            let mut b = Session::new(device).compile(make(-3.0));
            a.finalize_params();
            b.finalize_params();
            let yb = b.run(&[("x", &x)]).remove(0);
            let ya = a.run(&[("x", &x)]).remove(0);
            b.run(&[("x", &x)]);
            (ya, yb)
        });
        assert_eq!(ya.len(), 8, "{device:?}: caller's outputs untouched");
        assert_ne!(ya, yb);
        let runs: Vec<usize> = d.digests().iter().map(|g| g.compile).collect();
        let per = runs.len() / 3;
        assert!(per >= 1, "{device:?}: {runs:?}");
        assert!(
            runs[..per].iter().all(|&k| k == 1),
            "{device:?} first run is b: {runs:?}"
        );
        assert!(
            runs[per..2 * per].iter().all(|&k| k == 0),
            "{device:?} second is a: {runs:?}"
        );
        assert!(
            runs[2 * per..].iter().all(|&k| k == 1),
            "{device:?} third is b: {runs:?}"
        );
        // b's two runs digest the same
        let dg = d.digests();
        assert_eq!(
            dg[..per].iter().map(|g| g.hash).collect::<Vec<_>>(),
            dg[2 * per..].iter().map(|g| g.hash).collect::<Vec<_>>()
        );
    }
}
