// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Sparse UMAP training loop (port of fast-umap `train_sparse`).

use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rlx_driver::Device;
use rlx_runtime::Session;

use crate::adam::AdamState;
use crate::config::TrainingConfig;
use crate::encoder::knn::build_knn_edges;
use crate::encoder::mlp::ModelSpec;
use crate::model::CompiledUmap;
use crate::utils::{f64_to_f32, normalize_data_f64};
use crate::weights::WeightStore;

const EDGE_BATCH_COUNT: usize = 16;
/// Max positive edges per training step (must match `FittedUmap::load` compile caps).
pub const MAX_POS_EDGES_PER_EPOCH: usize = 50_000;
const LOSS_READBACK_INTERVAL: usize = 5;

struct EdgeBatch {
    head: Vec<f32>,
    tail: Vec<f32>,
    n_pos: usize,
    n_neg: usize,
}

/// Per-epoch training progress (fast-umap compatible).
#[derive(Debug, Clone)]
pub struct EpochProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f64,
    pub best_loss: f64,
    pub elapsed_secs: f64,
}

pub struct TrainResult {
    pub weights: WeightStore,
    pub losses: Vec<f64>,
    pub best_loss: f64,
    pub n_pos: usize,
    pub n_neg: usize,
    pub compiled: CompiledUmap,
}

/// Train parametric UMAP on normalized f32 data (`n × d` row-major).
pub fn train_sparse(
    device: Device,
    data: &mut [f64],
    n: usize,
    d: usize,
    spec: &ModelSpec,
    config: &TrainingConfig,
    metric: &crate::config::Metric,
    exit_rx: Option<Receiver<()>>,
    on_progress: Option<&dyn Fn(EpochProgress)>,
) -> TrainResult {
    assert!(n > config.k_neighbors, "n must be > k_neighbors");
    let k = config.k_neighbors;
    let neg_rate = config.neg_sample_rate;

    normalize_data_f64(data, n, d);
    let data_f32 = f64_to_f32(data);

    if config.verbose {
        eprintln!("[rlx-umap] Computing global k-NN (k={k}, n={n}, device={device:?}) …");
    }
    let knn_start = Instant::now();
    let all_pos_edges = build_knn_edges(&data_f32, n, d, k, metric, device);
    let n_all_edges = all_pos_edges.len();
    let n_pos = n_all_edges.min(MAX_POS_EDGES_PER_EPOCH);
    let subsampling = n_pos < n_all_edges;
    let n_neg = (n_pos * neg_rate).max(n);

    let mut rng = StdRng::seed_from_u64(9999);
    let fused_batches: Vec<EdgeBatch> = (0..EDGE_BATCH_COUNT)
        .map(|_| {
            sample_edge_batch(
                &all_pos_edges,
                n_all_edges,
                n_pos,
                n_neg,
                n,
                subsampling,
                &mut rng,
            )
        })
        .collect();

    if config.verbose {
        eprintln!(
            "[rlx-umap] k-NN done in {:.2}s — {n_all_edges} edges, batch {n_pos}+{n_neg}",
            knn_start.elapsed().as_secs_f64()
        );
    }

    let session = Session::new(device);
    let mut compiled = CompiledUmap::compile(&session, spec, n_pos, n_neg);
    let mut weights = crate::encoder::init_model_weights(spec, 9999);
    compiled.set_weights(&weights);

    #[cfg(feature = "pca")]
    if config.pca_warmstart {
        pca_warmstart(
            &session,
            spec,
            &mut compiled,
            &mut weights,
            &data_f32,
            config.verbose,
        );
    }

    let mut adam = AdamState::new_like(&weights);
    let mut best_loss = f64::INFINITY;
    let mut best_weights = weights.clone();
    let mut losses = Vec::new();
    let start = Instant::now();
    let mut epochs_without_improvement = 0i32;

    let ka = [config.kernel_a];
    let kb = [config.kernel_b];
    let rep = [config.repulsion_strength];

    for epoch in 0..config.epochs {
        if let Some(rx) = &exit_rx {
            if rx.try_recv().is_ok() {
                if config.verbose {
                    eprintln!(
                        "[rlx-umap] Interrupted — restoring best model (epoch {epoch}, loss {best_loss:.6})"
                    );
                }
                break;
            }
        }

        let batch = &fused_batches[epoch % EDGE_BATCH_COUNT];
        let outs = compiled.train.run(&[
            ("x", &data_f32),
            ("edge_h", &batch.head),
            ("edge_t", &batch.tail),
            ("kernel_a", &ka),
            ("kernel_b", &kb),
            ("repulsion", &rep),
            ("d_output", &[1.0f32]),
        ]);

        let mut grads = WeightStore::default();
        for (slot, out) in compiled.train_meta.params.iter().zip(outs.iter().skip(1)) {
            grads.0.insert(slot.name.clone(), out.clone());
        }

        adam.step(
            &mut weights,
            &grads,
            config.learning_rate,
            config.beta1,
            config.beta2,
            config.penalty,
            1e-8,
        );
        compiled.set_weights(&weights);

        let should_read = epoch % LOSS_READBACK_INTERVAL == 0 || epoch + 1 == config.epochs;
        if should_read {
            let loss = outs.first().and_then(|v| v.first()).copied().unwrap_or(0.0) as f64;

            if !loss.is_finite() {
                eprintln!(
                    "[rlx-umap] WARNING: loss became {loss} at epoch {} — stopping early.",
                    epoch + 1
                );
                break;
            }

            losses.push(loss);
            if loss < best_loss {
                best_loss = loss;
                best_weights = weights.clone();
                epochs_without_improvement = 0;
                if let Some(min_l) = config.min_desired_loss {
                    if loss <= min_l {
                        break;
                    }
                }
            } else {
                epochs_without_improvement += LOSS_READBACK_INTERVAL as i32;
            }

            if config.verbose {
                eprintln!(
                    "[rlx-umap] epoch {}/{} loss={loss:.6} best={best_loss:.6}",
                    epoch + 1,
                    config.epochs
                );
            }

            if let Some(cb) = on_progress {
                cb(EpochProgress {
                    epoch: epoch + 1,
                    total_epochs: config.epochs,
                    loss,
                    best_loss,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                });
            }
        }

        if let Some(patience) = config.patience {
            if epochs_without_improvement >= patience {
                if config.verbose {
                    eprintln!(
                        "[rlx-umap] Early stopping — no improvement for {patience} epochs (best {best_loss:.6})"
                    );
                }
                break;
            }
        }

        if let Some(timeout) = config.timeout {
            if start.elapsed().as_secs() >= timeout {
                if config.verbose {
                    eprintln!("[rlx-umap] Timeout ({timeout}s) at epoch {}", epoch + 1);
                }
                break;
            }
        }

        if config.cooldown_ms > 0 {
            thread::sleep(Duration::from_millis(config.cooldown_ms));
        }
    }

    compiled.set_weights(&best_weights);

    TrainResult {
        weights: best_weights,
        losses,
        best_loss,
        n_pos,
        n_neg,
        compiled,
    }
}

#[cfg(feature = "pca")]
fn pca_warmstart(
    session: &Session,
    spec: &ModelSpec,
    compiled: &mut CompiledUmap,
    weights: &mut WeightStore,
    data_f32: &[f32],
    verbose: bool,
) {
    use crate::encoder::pca_warmstart::build_pca_warmstart_graph;
    use crate::pca::pca;

    const PCA_EPOCHS: usize = 10;
    const PCA_LR: f64 = 1e-2;

    let (projected, _, _) = pca(data_f32, spec.n, spec.input_dim, spec.output_dim);
    let meta = build_pca_warmstart_graph(spec);
    let mut pca_exec = session.compile(meta.backward);
    weights.apply(&mut pca_exec);

    if verbose {
        eprintln!("[rlx-umap] PCA warm-start ({PCA_EPOCHS} epochs) …");
    }

    let mut adam = AdamState::new_like(weights);
    for _ in 0..PCA_EPOCHS {
        let outs = pca_exec.run(&[
            ("x", data_f32),
            ("pca_target", &projected),
            ("d_output", &[1.0f32]),
        ]);
        let mut grads = WeightStore::default();
        for (slot, out) in meta.params.iter().zip(outs.iter().skip(1)) {
            grads.0.insert(slot.name.clone(), out.clone());
        }
        adam.step(weights, &grads, PCA_LR, 0.9, 0.999, 0.0, 1e-8);
        weights.apply(&mut pca_exec);
    }
    compiled.set_weights(weights);

    if verbose {
        eprintln!("[rlx-umap] PCA warm-start complete");
    }
}

fn sample_edge_batch(
    all_pos: &[(usize, usize)],
    n_all: usize,
    n_pos: usize,
    n_neg: usize,
    n: usize,
    subsampling: bool,
    rng: &mut StdRng,
) -> EdgeBatch {
    let pos_sample: Vec<(usize, usize)> = if subsampling {
        let mut indices: Vec<usize> = (0..n_all).collect();
        indices.shuffle(rng);
        indices
            .into_iter()
            .take(n_pos)
            .map(|i| all_pos[i])
            .collect()
    } else {
        all_pos.to_vec()
    };

    let mut head = Vec::with_capacity(n_pos + n_neg);
    let mut tail = Vec::with_capacity(n_pos + n_neg);

    for &(h, t) in &pos_sample {
        head.push(h as f32);
        tail.push(t as f32);
    }
    if let Some(&(h, t)) = pos_sample.last() {
        while head.len() < n_pos {
            head.push(h as f32);
            tail.push(t as f32);
        }
    }
    for _ in 0..n_neg {
        let i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n.saturating_sub(1));
        if j >= i {
            j += 1;
        }
        head.push(i as f32);
        tail.push(j as f32);
    }

    EdgeBatch {
        head,
        tail,
        n_pos,
        n_neg,
    }
}
