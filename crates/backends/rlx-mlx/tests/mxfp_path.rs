//! Host (default) vs `native-mxfp` — both produce finite mxfp4 DequantMatMul.
//!
//! ```text
//! cargo test -p rlx-mlx --test mxfp_path
//! cargo test -p rlx-mlx --features native-mxfp --test mxfp_path
//! ```

#![cfg(rlx_mlx_host)]

use rlx_ir::{DType, Graph, Op, Shape, quant::QuantScheme};
use rlx_mlx::lower::first_host_eval_op;
use rlx_mlx::{MlxExecutable, MlxMode};

fn tiny_mxfp4_graph() -> Graph {
    let n = 4usize;
    let k = 32usize;
    let m = 1usize;
    let mut g = Graph::new("mxfp4_path");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.add_node(
        Op::Param {
            name: "lin.weight".into(),
        },
        vec![],
        Shape::new(&[n * k / 2], DType::U8),
    );
    let sc = g.add_node(
        Op::Param {
            name: "lin.scales".into(),
        },
        vec![],
        Shape::new(&[n, 1], DType::U8),
    );
    let zp = g.add_node(
        Op::Param {
            name: "lin.biases".into(),
        },
        vec![],
        Shape::new(&[4], DType::U8),
    );
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::MlxMxfp4 { group_size: 32 },
        },
        vec![x, w, sc, zp],
        Shape::new(&[m, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn mxfp4_dequant_matmul_runs() {
    let g = tiny_mxfp4_graph();
    #[cfg(not(feature = "native-mxfp"))]
    {
        assert!(
            first_host_eval_op(&g).is_some(),
            "default path should host-eval mxfp"
        );
    }
    #[cfg(feature = "native-mxfp")]
    {
        let label = first_host_eval_op(&g);
        assert!(
            label.is_none() || label.is_some_and(|s| !s.contains("mxfp")),
            "native-mxfp should not force mxfp host-eval, got {label:?}"
        );
    }

    let mut exe = MlxExecutable::compile_with_mode(g, MlxMode::Lazy);
    let n = 4usize;
    let k = 32usize;
    let w_bytes = vec![0u8; n * k / 2];
    let scales = vec![64u8; n];
    let biases = vec![0u8; 4];
    exe.set_param_typed("lin.weight", &w_bytes, DType::U8);
    exe.set_param_typed("lin.scales", &scales, DType::U8);
    exe.set_param_typed("lin.biases", &biases, DType::U8);
    let x = vec![0.1f32; k];
    let outs = exe.run(&[("x", &x)]);
    assert_eq!(outs.len(), 1);
    let y = &outs[0];
    assert_eq!(y.len(), n);
    assert!(y.iter().all(|v| v.is_finite()));
}
