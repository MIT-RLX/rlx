//! k-NN in the UMAP fit loop: device-fused graph vs CPU reference.

use rlx_driver::Device;
use rlx_umap::config::Metric;
use rlx_umap::encoder::knn::{
    build_knn_edges, knn_edge_match_rate, knn_index_match_rate, knn_indices_cpu,
    knn_indices_device_fused,
};
use rlx_umap::register;

fn test_data(n: usize, d: usize) -> Vec<f32> {
    (0..n * d)
        .map(|i| ((i as f32 * 0.13).sin() * 0.5 + 0.5).clamp(0.0, 1.0))
        .collect()
}

#[test]
fn fused_cpu_matches_reference_indices() {
    register();
    let n = 64;
    let d = 16;
    let k = 15;
    let data = test_data(n, d);

    for metric in [Metric::Euclidean, Metric::Cosine] {
        let ref_idx = knn_indices_cpu(&data, n, d, k, &metric);
        let fused =
            knn_indices_device_fused(&data, n, d, k, &metric, Device::Cpu).expect("fused cpu knn");
        let rate = knn_index_match_rate(&ref_idx, &fused, n, k);
        assert!(
            rate >= 1.0 - 1e-9,
            "{metric:?}: fused CPU index match {rate}"
        );

        let ref_edges = build_knn_edges(&data, n, d, k, &metric, Device::Cpu);
        let fused_edges = build_knn_edges(&data, n, d, k, &metric, Device::Cpu);
        let edge_rate = knn_edge_match_rate(&ref_edges, &fused_edges);
        assert!(
            edge_rate >= 1.0 - 1e-9,
            "{metric:?}: edge parity {edge_rate}"
        );
    }
}

#[cfg(any(feature = "metal", feature = "mlx", feature = "gpu", feature = "cuda"))]
mod device_knn {
    use super::*;
    use rlx_runtime::device_ext;
    use rlx_umap::encoder::knn::{knn_indices_cpu, knn_indices_for_device, pairwise_matrix_cpu};
    use rlx_umap::max_pairwise_error;

    macro_rules! device_knn_parity {
        ($name:ident, $dev:expr, $metric:expr) => {
            #[test]
            fn $name() {
                register();
                if !device_ext::is_available($dev) {
                    eprintln!("skip: {:?} not available", $dev);
                    return;
                }
                let n = 64;
                let d = 16;
                let k = 15;
                let data = test_data(n, d);
                let metric = $metric;

                let ref_pw = pairwise_matrix_cpu(&data, n, d, &metric);
                let _ = &ref_pw;
                let ref_idx = knn_indices_cpu(&data, n, d, k, &metric);

                let fused_idx = knn_indices_for_device(&data, n, d, k, &metric, $dev);

                let idx_rate = knn_index_match_rate(&ref_idx, &fused_idx, n, k);
                assert!(
                    idx_rate >= 1.0 - 1e-9,
                    "{:?} {metric:?} index match {idx_rate}",
                    $dev
                );

                let ref_edges = build_knn_edges(&data, n, d, k, &metric, Device::Cpu);
                let dev_edges = build_knn_edges(&data, n, d, k, &metric, $dev);
                let edge_rate = knn_edge_match_rate(&ref_edges, &dev_edges);
                assert!(
                    edge_rate >= 1.0 - 1e-9,
                    "{:?} {metric:?} edge Jaccard {edge_rate}",
                    $dev
                );

                use rlx_ir::{DType, Shape};
                use rlx_runtime::Session;
                use rlx_umap::{pairwise_cosine_graph, pairwise_euclidean_graph};
                let mut g = rlx_ir::Graph::new("pw");
                let x = g.input("x", Shape::new(&[n, d], DType::F32));
                let pw = match metric {
                    Metric::Cosine => pairwise_cosine_graph(&mut g, x, n),
                    _ => pairwise_euclidean_graph(&mut g, x, n),
                };
                g.set_outputs(vec![pw]);
                let dev_pw = Session::new($dev).compile(g).run(&[("x", &data)]).remove(0);
                let pw_err = max_pairwise_error(&ref_pw, &dev_pw);
                let pw_tol = match $dev {
                    Device::Cuda if matches!(metric, Metric::Cosine) => 2e-3_f32,
                    Device::Cuda => 6e-2_f32,
                    _ if matches!(metric, Metric::Cosine) => 1e-3_f32,
                    _ => 2e-3_f32,
                };
                assert!(
                    pw_err < pw_tol,
                    "{:?} {metric:?} pairwise max err {pw_err} (tol {pw_tol})",
                    $dev
                );
            }
        };
    }

    #[cfg(feature = "metal")]
    device_knn_parity!(metal_fit_knn_euclidean, Device::Metal, Metric::Euclidean);

    #[cfg(feature = "metal")]
    device_knn_parity!(metal_fit_knn_cosine, Device::Metal, Metric::Cosine);

    #[cfg(feature = "gpu")]
    device_knn_parity!(wgpu_fit_knn_euclidean, Device::Gpu, Metric::Euclidean);

    #[cfg(feature = "mlx")]
    device_knn_parity!(mlx_fit_knn_euclidean, Device::Mlx, Metric::Euclidean);

    #[cfg(feature = "mlx")]
    device_knn_parity!(mlx_fit_knn_cosine, Device::Mlx, Metric::Cosine);

    #[cfg(feature = "cuda")]
    device_knn_parity!(cuda_fit_knn_euclidean, Device::Cuda, Metric::Euclidean);

    #[cfg(feature = "cuda")]
    device_knn_parity!(cuda_fit_knn_cosine, Device::Cuda, Metric::Cosine);
}
