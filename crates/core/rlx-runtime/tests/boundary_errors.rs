// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Caller mistakes at the `CompiledGraph` boundary are reported by name on
//! every backend, instead of computing on zeros (see `rlx_runtime::io_check`).

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::device_ext::available_devices;
use rlx_runtime::{CompileError, IoError, Session};

fn graph() -> Graph {
    let mut g = Graph::new("mlp");
    let x = g.input("tokens", Shape::new(&[2, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 3], DType::F32));
    let h = g.mm(x, w);
    let y = g.relu(h);
    g.set_outputs(vec![y]);
    g
}

const X: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

#[test]
fn mistakes_are_named_on_every_device() {
    let _gpu = common::serialize_gpu();
    for dev in available_devices() {
        let mut c = Session::new(dev).try_compile(graph()).expect("compile");
        assert_eq!(c.inputs()[0].name, "tokens");
        assert_eq!(c.params()[0].shape, Some(vec![4, 3]));

        let e = c.try_run(&[("token", &X)]).unwrap_err();
        assert_eq!(
            e,
            IoError::UnknownInput {
                name: "token".into(),
                closest: vec!["tokens".into()]
            },
            "{dev:?}"
        );
        let e = c.try_run(&[("tokens", &X[..5])]).unwrap_err();
        assert!(
            matches!(
                e,
                IoError::WrongLength {
                    got: 5,
                    want: 8,
                    ..
                }
            ),
            "{dev:?}: {e}"
        );

        assert!(c.check_bindings(&["tokens"]).is_err(), "w is not set yet");
        c.set_param("w", &[0.5; 12]);
        assert!(c.check_bindings(&["tokens"]).is_ok());
        let y = c.try_run(&[("tokens", &X)]).unwrap();
        assert!((y[0][0] - 5.0).abs() < 1e-5, "{dev:?}: {:?}", y[0]);

        let short = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.set_param("w", &[0.5; 5]);
        }));
        assert!(
            short.is_err(),
            "{dev:?}: a 5-value upload to a [4, 3] param must panic"
        );
    }
}

#[test]
fn invalid_graph_is_an_error_not_a_panic() {
    let mut g = Graph::new("bad");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[4, 5], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::MatMul,
        vec![x, w],
        Shape::new(&[2, 5], DType::F32),
    );
    g.set_outputs(vec![y]);
    let e = Session::new(rlx_runtime::Device::Cpu)
        .try_compile(g)
        .err()
        .unwrap();
    assert!(matches!(e, CompileError::Invalid(_)), "{e}");
}

/// An invalid input graph is reported as the input's fault, not as broken by
/// whichever optimization pass happened to run first (debug builds verify
/// after every pass).
#[test]
fn invalid_input_is_not_blamed_on_a_pass() {
    if !cfg!(debug_assertions) {
        return;
    }
    let mut g = Graph::new("bad");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[4, 5], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::MatMul,
        vec![x, w],
        Shape::new(&[2, 5], DType::F32),
    );
    let z = g.relu(y);
    g.set_outputs(vec![z]);
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(|| Session::new(rlx_runtime::Device::Cpu).compile(g));
    std::panic::set_hook(prev);
    let msg = match r {
        Ok(_) => panic!("an invalid graph compiled"),
        Err(p) => p.downcast_ref::<String>().cloned().unwrap_or_default(),
    };
    assert!(
        msg.contains("the input graph (already invalid before any pass"),
        "{msg}"
    );
}

/// A param passed to `run` takes effect. Backends ignored it, so a training
/// loop that fed its updated weights through `run` computed with stale ones.
#[test]
fn param_passed_to_run_is_uploaded() {
    let mut c = Session::new(rlx_runtime::Device::Cpu)
        .try_compile(graph())
        .expect("compile");
    let a = c.run(&[("tokens", &X), ("w", &[1.0; 12])]);
    let b = c.run(&[("tokens", &X), ("w", &[2.0; 12])]);
    assert_eq!(a[0][0], 10.0);
    assert_eq!(b[0][0], 20.0, "the second weights were ignored");
    let e = c.try_run(&[("tokens", &X), ("w", &[1.0; 5])]).unwrap_err();
    assert!(
        matches!(e, IoError::WrongLength { kind: "param", .. }),
        "{e}"
    );
}

/// Names meant to read zeros can be declared so; `check_bindings` then passes.
#[test]
fn allow_unbound_accepts_intended_zeros() {
    let mut c = Session::new(rlx_runtime::Device::Cpu)
        .try_compile(graph())
        .expect("compile");
    assert!(c.check_bindings(&[]).is_err());
    c.allow_unbound(&["tokens", "w"]);
    assert!(c.check_bindings(&[]).is_ok());
}

/// Output names are read off the graph as given, so they stay right after
/// fusion renumbers and folds the nodes behind them.
#[test]
fn outputs_are_found_by_name_on_every_device() {
    let _gpu = common::serialize_gpu();
    let mut g = Graph::new("two");
    let x = g.input("tokens", Shape::new(&[2, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 3], DType::F32));
    let h = g.mm(x, w);
    let y = g.relu(h);
    g.set_named_outputs([("probs", y), ("hidden", h)]);
    let w_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.25 - 1.5).collect();
    for dev in available_devices() {
        let mut c = Session::new(dev).try_compile(g.clone()).expect("compile");
        let outs = c.outputs();
        assert_eq!(outs[1].name, "hidden", "{dev:?}");
        assert_eq!(outs[0].shape, Some(vec![2, 3]));
        c.set_param("w", &w_data);
        let r = c.run(&[("tokens", &X)]);
        let hidden = &r[c.output_index("hidden").unwrap()];
        let probs = &r[c.output_index("probs").unwrap()];
        assert!(hidden.iter().any(|v| *v < 0.0), "{dev:?}: {hidden:?}");
        for (p, h) in probs.iter().zip(hidden) {
            assert!(
                (p - h.max(0.0)).abs() < 1e-4,
                "{dev:?}: {probs:?} vs {hidden:?}"
            );
        }
        assert_eq!(
            c.output_index("hiden").unwrap_err().to_string(),
            "rlx: the graph has no output named `hiden` (did you mean `hidden`?)"
        );
    }
}
