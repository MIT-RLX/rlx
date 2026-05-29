// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use rlx_driver::Device;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Session;
use rlx_umap::{
    compare_knn, cosine_knn_graph, cosine_pairwise_reference, knn_forward_packed,
    max_pairwise_error, pairwise_cosine_graph, register, unpack_knn_packed,
};

#[test]
fn cosine_pairwise_graph_matches_reference() {
    register();
    let n = 32;
    let d = 8;
    let _k = 5;
    let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.13).sin()).collect();
    let ref_pw = cosine_pairwise_reference(&data, n, d);

    let mut g = Graph::new("pw");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let pw = pairwise_cosine_graph(&mut g, x, n);
    g.set_outputs(vec![pw]);

    let mut exe = Session::new(Device::Cpu).compile(g);
    let got = exe.run(&[("x", &data)]).remove(0);
    assert!(max_pairwise_error(&ref_pw, &got) < 1e-4);
}

#[test]
fn cosine_knn_end_to_end_parity() {
    register();
    let n = 48;
    let d = 16;
    let k = 8;
    let data: Vec<f32> = (0..n * d).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();

    let ref_pw = cosine_pairwise_reference(&data, n, d);

    let mut g_pw = Graph::new("pw_only");
    let x = g_pw.input("x", Shape::new(&[n, d], DType::F32));
    let pw = pairwise_cosine_graph(&mut g_pw, x, n);
    g_pw.set_outputs(vec![pw]);
    let got_pw = Session::new(Device::Cpu)
        .compile(g_pw)
        .run(&[("x", &data)])
        .remove(0);
    let pw_err = max_pairwise_error(&ref_pw, &got_pw);
    assert!(
        pw_err < 1e-3,
        "cosine pairwise graph max err {pw_err} too large for k-NN parity"
    );

    let mut ref_packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&ref_pw, n, k, &mut ref_packed);
    let (ref_idx, ref_dist) = unpack_knn_packed(&ref_packed, n, k);
    let mut got_packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&got_pw, n, k, &mut got_packed);
    let (host_idx, host_dist) = unpack_knn_packed(&got_packed, n, k);

    let mut g = Graph::new("e2e");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let (idx, dist) = cosine_knn_graph(&mut g, x, n, k as u32);
    g.set_outputs(vec![idx, dist]);

    let mut exe = Session::new(Device::Cpu).compile(g);
    let outs = exe.run(&[("x", &data)]);
    let host_vs_ref = compare_knn(&ref_idx, &ref_dist, &host_idx, &host_dist, n, k);
    assert_eq!(
        host_vs_ref.index_match_rate, 1.0,
        "k-NN on standalone-graph pairwise must match reference"
    );

    let e2e_vs_host = compare_knn(&host_idx, &host_dist, &outs[0], &outs[1], n, k);
    assert_eq!(
        e2e_vs_host.index_match_rate, 1.0,
        "fused e2e graph k-NN must match standalone-graph k-NN (pw_err={pw_err:.2e})"
    );

    let e2e_vs_ref = compare_knn(&ref_idx, &ref_dist, &outs[0], &outs[1], n, k);
    assert_eq!(
        e2e_vs_ref.index_match_rate, 1.0,
        "e2e k-NN must match reference"
    );
    assert!(e2e_vs_ref.max_dist_error < 1e-4);
}
