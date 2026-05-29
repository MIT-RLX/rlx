// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Global k-NN graph construction for the UMAP fit loop.
//!
//! - **CPU** (`Device::Cpu`): host reference pairwise + `rlx_cpu::umap_knn` (fast path).
//! - **GPU backends** (`Metal`, `Gpu`, `Mlx`, …): fused `pairwise → umap.knn` on device.
//! - **CUDA fit k-NN**: host reference pairwise + CPU k-NN by default (cuBLAS pairwise can
//!   reorder neighbours vs reference). Set `RLX_UMAP_CUDA_FUSED_KNN=1` for the fused GPU path.

use rlx_driver::Device;

use crate::config::Metric;
use crate::pack::unpack_knn_packed;
use crate::pairwise::{cosine_pairwise_reference, euclidean_pairwise_reference};

/// Positive k-NN edges `(head, tail)` (up to `n * k`).
pub fn build_knn_edges(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
    device: Device,
) -> Vec<(usize, usize)> {
    assert!(k < n, "k ({k}) must be < n ({n})");

    #[cfg(feature = "nn-descent")]
    const NN_DESCENT_THRESHOLD: usize = 10_000;
    #[cfg(feature = "nn-descent")]
    if n >= NN_DESCENT_THRESHOLD {
        return knn_edges_nn_descent(data_f32, n, d, k);
    }

    let idx = if device == Device::Cpu {
        knn_indices_cpu(data_f32, n, d, k, metric)
    } else {
        #[cfg(feature = "bench")]
        {
            knn_indices_for_device(data_f32, n, d, k, metric, device)
        }
        #[cfg(not(feature = "bench"))]
        {
            let _ = device;
            knn_indices_cpu(data_f32, n, d, k, metric)
        }
    };

    edges_from_indices(&idx, n, k)
}

/// CPU reference indices (flat row-major `[n, k]`).
pub fn knn_indices_cpu(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
) -> Vec<f32> {
    let pairwise = pairwise_matrix_cpu(data_f32, n, d, metric);
    knn_indices_from_pairwise(&pairwise, n, k)
}

/// Pairwise distance matrix on host (reference for parity tests).
pub fn pairwise_matrix_cpu(data_f32: &[f32], n: usize, d: usize, metric: &Metric) -> Vec<f32> {
    match metric {
        Metric::Cosine => cosine_pairwise_reference(data_f32, n, d),
        Metric::Euclidean | Metric::Manhattan | Metric::Minkowski => {
            euclidean_pairwise_reference(data_f32, n, d)
        }
    }
}

/// k-NN column indices from a row-major `[n, n]` pairwise matrix.
pub fn knn_indices_from_pairwise(pairwise: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut packed = vec![0f32; n * 2 * k];
    rlx_cpu::umap_knn::knn_forward_packed(pairwise, n, k, &mut packed);
    let (idx, _) = unpack_knn_packed(&packed, n, k);
    idx
}

/// k-NN indices for a non-CPU training device (fused graph, or MLX hybrid).
#[cfg(feature = "bench")]
pub fn knn_indices_for_device(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
    device: Device,
) -> Vec<f32> {
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    if device == Device::Mlx {
        return knn_indices_mlx(data_f32, n, d, k, metric).unwrap_or_else(|e| {
            if cfg!(debug_assertions) {
                eprintln!("[rlx-umap] MLX k-NN fallback to CPU: {e}");
            }
            knn_indices_cpu(data_f32, n, d, k, metric)
        });
    }
    // cuBLAS pairwise can differ slightly from the host reference and reorder k-NN
    // neighbours; fit uses reference pairwise + CPU k-NN for strict parity.
    #[cfg(feature = "cuda")]
    if device == Device::Cuda
        && !std::env::var("RLX_UMAP_CUDA_FUSED_KNN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    {
        return knn_indices_cpu(data_f32, n, d, k, metric);
    }
    knn_indices_device_fused(data_f32, n, d, k, metric, device).unwrap_or_else(|e| {
        if cfg!(debug_assertions) {
            eprintln!("[rlx-umap] device k-NN fallback to CPU: {e}");
        }
        knn_indices_cpu(data_f32, n, d, k, metric)
    })
}

/// MLX: pairwise on device, `umap.knn` on CPU (`mlx::compile` cannot host-eval k-NN).
#[cfg(all(feature = "bench", feature = "mlx", target_os = "macos"))]
pub fn knn_indices_mlx(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
) -> Result<Vec<f32>, String> {
    Ok(knn_mlx_hybrid(data_f32, n, d, k, metric)?.0)
}

/// MLX hybrid k-NN: `(indices, distances)` flat `[n, k]` each.
#[cfg(all(feature = "bench", feature = "mlx", target_os = "macos"))]
pub fn knn_mlx_hybrid(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Session, device_ext};

    if !device_ext::is_available(Device::Mlx) {
        return Err("MLX device not available".into());
    }

    crate::register();

    let mut g_pw = Graph::new("mlx_fit_pw");
    let x = g_pw.input("x", Shape::new(&[n, d], DType::F32));
    let pw = match metric {
        Metric::Cosine => crate::graph::pairwise_cosine_graph(&mut g_pw, x, n),
        Metric::Euclidean | Metric::Manhattan | Metric::Minkowski => {
            crate::graph::pairwise_euclidean_graph(&mut g_pw, x, n)
        }
    };
    g_pw.set_outputs(vec![pw]);
    let pw_mat = Session::new(Device::Mlx)
        .compile(g_pw)
        .run(&[("x", data_f32)])
        .into_iter()
        .next()
        .ok_or_else(|| "mlx pairwise output missing".to_string())?;

    let mut g_knn = Graph::new("mlx_fit_knn_cpu");
    let pw_in = g_knn.input("pairwise", Shape::new(&[n, n], DType::F32));
    let packed = crate::graph::knn_graph(&mut g_knn, pw_in, k as u32);
    let (idx, dist) = crate::graph::split_knn_packed(&mut g_knn, packed, k as u32);
    g_knn.set_outputs(vec![idx, dist]);
    let outs = Session::new(Device::Cpu)
        .compile(g_knn)
        .run(&[("pairwise", &pw_mat)]);
    if outs.len() != 2 {
        return Err(format!(
            "mlx cpu knn: expected 2 outputs, got {}",
            outs.len()
        ));
    }
    Ok((outs[0].clone(), outs[1].clone()))
}

/// Fused `x → pairwise → umap.knn` on `device` (Metal / wgpu when enabled).
#[cfg(feature = "bench")]
pub fn knn_indices_device_fused(
    data_f32: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: &Metric,
    device: Device,
) -> Result<Vec<f32>, String> {
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Session, device_ext};

    if !device_ext::is_available(device) {
        return Err(format!("device {device:?} not available"));
    }

    crate::register();

    let mut g = Graph::new("umap_fit_knn");
    let x = g.input("x", Shape::new(&[n, d], DType::F32));
    let pw = match metric {
        Metric::Cosine => crate::graph::pairwise_cosine_graph(&mut g, x, n),
        Metric::Euclidean | Metric::Manhattan | Metric::Minkowski => {
            crate::graph::pairwise_euclidean_graph(&mut g, x, n)
        }
    };
    let packed = crate::graph::knn_graph(&mut g, pw, k as u32);
    let (idx, _) = crate::graph::split_knn_packed(&mut g, packed, k as u32);
    g.set_outputs(vec![idx]);

    let outs = Session::new(device).compile(g).run(&[("x", data_f32)]);
    outs.into_iter()
        .next()
        .ok_or_else(|| "knn index output missing".into())
}

/// Fraction of `(row, slot)` k-NN indices that match between two flat `[n, k]` layouts.
pub fn knn_index_match_rate(ref_idx: &[f32], got_idx: &[f32], n: usize, k: usize) -> f64 {
    assert_eq!(ref_idx.len(), n * k);
    assert_eq!(got_idx.len(), n * k);
    let mut matches = 0usize;
    for i in 0..n * k {
        if (ref_idx[i] as i32) == (got_idx[i] as i32) {
            matches += 1;
        }
    }
    matches as f64 / (n * k) as f64
}

/// Edge-list parity: same directed edges `(head, tail)` ignoring order.
pub fn knn_edge_match_rate(ref_edges: &[(usize, usize)], got_edges: &[(usize, usize)]) -> f64 {
    use std::collections::HashSet;
    let a: HashSet<_> = ref_edges.iter().copied().collect();
    let b: HashSet<_> = got_edges.iter().copied().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(&b).count();
    let union = a.union(&b).count();
    inter as f64 / union as f64
}

#[cfg(feature = "nn-descent")]
fn knn_edges_nn_descent(data_f32: &[f32], n: usize, d: usize, k: usize) -> Vec<(usize, usize)> {
    let (idx, _) = crate::nn_descent::nn_descent(data_f32, n, d, k);
    edges_from_indices_i32(&idx, n, k)
}

fn edges_from_indices(idx: &[f32], n: usize, k: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            let neighbor = idx[i * k + j] as usize;
            if neighbor < n {
                edges.push((i, neighbor));
            }
        }
    }
    edges
}

#[cfg(feature = "nn-descent")]
fn edges_from_indices_i32(idx: &[i32], n: usize, k: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            let neighbor = idx[i * k + j] as usize;
            if neighbor < n {
                edges.push((i, neighbor));
            }
        }
    }
    edges
}
