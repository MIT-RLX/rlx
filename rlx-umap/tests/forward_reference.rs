//! Compare compiled MLP forward against a naive CPU reference.

use rlx_driver::Device;
use rlx_runtime::Session;
use rlx_umap::encoder::mlp::{ModelSpec, build_forward_graph, init_model_weights};
use rlx_umap::register;
use rlx_umap::weights::WeightStore;

fn mlp_reference(x: &[f32], w: &WeightStore, spec: &ModelSpec) -> Vec<f32> {
    let n = spec.n;
    let mut h = x.to_vec();
    let mut in_d = spec.input_dim;
    for (li, &hd) in spec.hidden.iter().enumerate() {
        let ww = w.get(&format!("umap_w{li}")).unwrap();
        let bb = w.get(&format!("umap_b{li}")).unwrap();
        let mut out = vec![0.0f32; n * hd];
        for i in 0..n {
            for j in 0..hd {
                let mut s = bb[j];
                for k in 0..in_d {
                    s += h[i * in_d + k] * ww[k * hd + j];
                }
                out[i * hd + j] = s.max(0.0);
            }
        }
        h = out;
        in_d = hd;
    }
    let ww = w.get("umap_w_out").unwrap();
    let bb = w.get("umap_b_out").unwrap();
    let d_out = spec.output_dim;
    let mut out = vec![0.0f32; n * d_out];
    for i in 0..n {
        for j in 0..d_out {
            let mut s = bb[j];
            for k in 0..in_d {
                s += h[i * in_d + k] * ww[k * d_out + j];
            }
            out[i * d_out + j] = s;
        }
    }
    out
}

#[test]
fn compiled_forward_matches_reference() {
    register();
    let n = 8;
    let d = 4;
    let spec = ModelSpec {
        n,
        input_dim: d,
        output_dim: 2,
        hidden: vec![6],
    };
    let weights = init_model_weights(&spec, 7);
    let x: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.03).sin()).collect();
    for (name, t) in &weights.0 {
        let m = t.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        eprintln!("param {name} len={} max_abs={m}", t.len());
    }

    let want = mlp_reference(&x, &weights, &spec);
    eprintln!(
        "reference max_abs={}",
        want.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    );

    let (g, _, _) = build_forward_graph(&spec);
    let mut exec = Session::new(Device::Cpu).compile(g);
    weights.apply(&mut exec);
    let got = exec.run(&[("x", &x)]).remove(0);

    assert_eq!(got.len(), want.len());
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "max_err={max_err} got[0..4]={:?} want[0..4]={:?}",
        &got[..4],
        &want[..4]
    );
    assert!(max_err < 1e-4, "max_err={max_err}");
}
