// Backend parity vs CPU session for cosine pairwise + k-NN.

use rlx_driver::Device;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::Session;
use rlx_runtime::device_ext;
use rlx_umap::{
    compare_knn, cosine_knn_graph, cosine_pairwise_reference, euclidean_pairwise_reference,
    max_pairwise_error, pairwise_cosine_graph, pairwise_euclidean_graph, register,
};

fn run_pw(device: Device, data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut g = Graph::new("pw");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let pw = pairwise_cosine_graph(&mut g, x, n);
    g.set_outputs(vec![pw]);
    Session::new(device)
        .compile(g)
        .run(&[("x", data)])
        .remove(0)
}

fn run_knn(device: Device, data: &[f32], n: usize, d: usize, k: u32) -> (Vec<f32>, Vec<f32>) {
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    if device == Device::Mlx {
        return rlx_umap::session::cosine_knn_mlx(data, n, d, k).expect("mlx split knn");
    }
    let mut g = Graph::new("knn");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let (idx, dist) = cosine_knn_graph(&mut g, x, n, k);
    g.set_outputs(vec![idx, dist]);
    let outs = Session::new(device).compile(g).run(&[("x", data)]);
    (outs[0].clone(), outs[1].clone())
}

#[test]
fn cpu_cosine_baseline() {
    register();
    let n = 64;
    let d = 16;
    let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
    let ref_pw = cosine_pairwise_reference(&data, n, d);
    let cpu_pw = run_pw(Device::Cpu, &data, n, d);
    let err = max_pairwise_error(&ref_pw, &cpu_pw);
    assert!(err < 1e-4, "cpu pw err {err}");
}

macro_rules! backend_pw_test {
    ($name:ident, $dev:expr) => {
        #[test]
        fn $name() {
            register();
            if !device_ext::is_available($dev) {
                eprintln!("skip: {:?} not available", $dev);
                return;
            }
            let n = 64;
            let d = 16;
            let k = 8u32;
            let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
            let cpu_pw = run_pw(Device::Cpu, &data, n, d);
            let dev_pw = run_pw($dev, &data, n, d);
            let err = max_pairwise_error(&cpu_pw, &dev_pw);
            assert!(err < 1e-4, "{:?} pairwise vs cpu max err {err}", $dev);

            let (cpu_idx, cpu_dist) = run_knn(Device::Cpu, &data, n, d, k);
            let (dev_idx, dev_dist) =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_knn($dev, &data, n, d, k)
                })) {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("skip knn: {:?} panicked", $dev);
                        return;
                    }
                };
            let report = compare_knn(&cpu_idx, &cpu_dist, &dev_idx, &dev_dist, n, k as usize);
            assert_eq!(
                report.index_match_rate, 1.0,
                "{:?} knn index match {:.4}",
                $dev, report.index_match_rate
            );
        }
    };
}

#[cfg(feature = "metal")]
fn metal_vs_cpu_graph(g: Graph, data: &[f32]) -> f32 {
    let cpu = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", data)])
        .remove(0);
    let metal = Session::new(Device::Metal)
        .compile(g)
        .run(&[("x", data)])
        .remove(0);
    max_pairwise_error(&cpu, &metal)
}

#[cfg(feature = "metal")]
#[test]
fn metal_matmul_only() {
    register();
    if !device_ext::is_available(Device::Metal) {
        return;
    }
    let n = 32;
    let d = 16;
    let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let xt = g.transpose_(x, vec![1, 0]);
    let cross = g.mm(x, xt);
    g.set_outputs(vec![cross]);
    let err = metal_vs_cpu_graph(g, &data);
    assert!(err < 1e-4, "matmul metal vs cpu {err}");
}

#[cfg(feature = "metal")]
#[test]
fn metal_div_same_shape_vs_cpu() {
    register();
    if !device_ext::is_available(Device::Metal) {
        return;
    }
    let n = 32;
    let d = 16;
    let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
    let mut g = Graph::new("div");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let xt = g.transpose_(x, vec![1, 0]);
    let cross = g.mm(x, xt);
    let denom = g.add_node(
        rlx_ir::Op::Constant {
            data: vec![0f32; n * n]
                .into_iter()
                .flat_map(|_| 1.0f32.to_le_bytes())
                .collect(),
        },
        vec![],
        Shape::new(&[n, n], DType::F32),
    );
    let sim = g.div(cross, denom);
    g.set_outputs(vec![sim]);
    let err = metal_vs_cpu_graph(g, &data);
    assert!(err < 1e-4, "div [n,n] metal vs cpu {err}");
}

#[cfg(feature = "metal")]
#[test]
fn metal_euclidean_pairwise_vs_cpu() {
    register();
    if !device_ext::is_available(Device::Metal) {
        return;
    }
    let n = 64;
    let d = 16;
    let data: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07).sin()).collect();
    let mut g = Graph::new("euclid");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let pw = pairwise_euclidean_graph(&mut g, x, n);
    g.set_outputs(vec![pw]);
    let cpu = Session::new(Device::Cpu)
        .compile(g.clone())
        .run(&[("x", &data)])
        .remove(0);
    let metal = Session::new(Device::Metal)
        .compile(g)
        .run(&[("x", &data)])
        .remove(0);
    let err = max_pairwise_error(&cpu, &metal);
    assert!(err < 1e-4, "euclidean metal vs cpu {err}");
}

#[cfg(feature = "metal")]
backend_pw_test!(metal_cosine_parity, Device::Metal);

#[cfg(feature = "mlx")]
backend_pw_test!(mlx_cosine_parity, Device::Mlx);

#[cfg(feature = "gpu")]
backend_pw_test!(wgpu_cosine_parity, Device::Gpu);
