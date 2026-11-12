// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident e-means: the training state never leaves the device.
//!
//! [`DeviceEmeans`] compiles one [`resident_step_graph`] per run and binds the
//! dataset, the centroids, the masses and the last-pass accumulators as GPU
//! handles; each step's outputs feed those handles in place. A step uploads
//! only the batch's row indices, their weights and `α` (`2B + 1` floats), and
//! reads nothing back. The host reads:
//!
//! - the `k` masses every [`CHECK_UNUSED_EVERY`] steps, for the revival check
//!   (and the centroids only when a cluster is actually revived);
//! - the `[k, d+1]` accumulator and the objective once per epoch;
//! - the labels during the last epoch, when assignments are captured.
//!
//! The schedule, the revival decisions and their RNG stream are the host
//! trainer's ([`Emeans`]), so a device run splits the same clusters as a host
//! run up to f32 summation order. A backend without GPU handles (the CPU)
//! runs the same graph with the state passed in and out each step.
//!
//! The tail batch of an epoch is padded to `B` with zero-weight rows, so one
//! compiled graph serves every step.

use rlx_runtime::{CompileOptions, CompiledGraph, Device, Precision, Session};

use super::*;

/// Input names of [`resident_step_graph`].
pub mod resident_io {
    /// The whole training set `[n, d]` (bound once).
    pub const DATA: &str = "data";
    /// Row indices of this batch into `data`, `[B]` (padding repeats row 0).
    pub const ROWS: &str = "rows";
    /// Weight of each batch row, `[B, 1]`: 1 for a real row, 0 for padding.
    pub const WEIGHT: &str = "weight";
    /// Centroid positions `[k, d]`.
    pub const CENTROIDS: &str = "centroids";
    /// EMA masses `[k]`.
    pub const USAGE: &str = "usage";
    /// Scheduled step `α`, `[1]`.
    pub const ALPHA: &str = "alpha";
    /// Running per-cluster sums and counts of the epoch, `[k, d+1]`.
    pub const ACC: &str = "acc";
    /// Running objective of the epoch, `[1]`.
    pub const OBJECTIVE: &str = "objective";
}

const OUT_CENTROIDS: usize = 0;
const OUT_USAGE: usize = 1;
const OUT_ACC: usize = 2;
const OUT_OBJECTIVE: usize = 3;
const OUT_LABELS: usize = 4;
const N_OUTPUTS: usize = 5;

/// One e-means step that gathers its batch from a resident dataset.
///
/// Inputs: see [`resident_io`]. Outputs, in order: `centroids'[k, d]`,
/// `usage'[k]`, `acc + stats [k, d+1]`, `objective + Σ‖x − c‖² [1]`,
/// `labels[B]`. The first four are meant to feed their inputs in place.
pub fn resident_step_graph(n: usize, b: usize, d: usize, k: usize, assign: DType) -> Graph {
    use rlx_ir::infer::GraphExt as _;
    let f = DType::F32;
    let mut g = Graph::new("emeans_resident_step");
    let data = g.input(resident_io::DATA, Shape::new(&[n, d], f));
    let rows = g.input(resident_io::ROWS, Shape::new(&[b], f));
    // A host-made weight rather than an in-graph `iota < valid` mask: it is
    // plain data under every number format (a comparison has no f64 form).
    let w = g.input(resident_io::WEIGHT, Shape::new(&[b, 1], f));
    let c = g.input(resident_io::CENTROIDS, Shape::new(&[k, d], f));
    let usage = g.input(resident_io::USAGE, Shape::new(&[k], f));
    let alpha = g.input(resident_io::ALPHA, Shape::new(&[1], f));
    let acc = g.input(resident_io::ACC, Shape::new(&[k, d + 1], f));
    let objective = g.input(resident_io::OBJECTIVE, Shape::new(&[1], f));

    let x = g.gather_(data, rows, 0);
    let s = build_step(&mut g, x, Some(w), c, usage, alpha, assign);
    let acc_new = g.add(acc, s.stats);
    let objective_new = g.add(objective, s.objective);
    g.set_outputs(vec![s.centroids, s.usage, acc_new, objective_new, s.labels]);
    g
}

/// Where the training state lives between steps.
enum State {
    /// GPU handles fed in place by the graph.
    Resident,
    /// Host copies passed in and out each step (no GPU-handle support).
    Host {
        data: Vec<f32>,
        acc: Vec<f32>,
        objective: [f32; 1],
    },
}

/// The compile options [`DeviceEmeans::new`] uses on `device`: the defaults,
/// except that the XDNA NPU gets [`Precision::BF16`]. Its default matmul is
/// per-channel INT8; the bf16 kernel is exact for bf16-rounded operands and
/// cut the assignment's excess distance over the exact nearest centroid from
/// 7.9e-6 to 1.1e-6 on overlapping blobs.
pub fn device_options(device: Device) -> CompileOptions {
    let mut options = CompileOptions::new();
    if device == Device::Xdna {
        options.precision = Precision::BF16;
    }
    options
}

/// Device-resident e-means trainer (see the [module docs](self)).
pub struct DeviceEmeans {
    device: Device,
    options: CompileOptions,
    assign: DType,
    em: Emeans,
    resident: Option<bool>,
}

impl DeviceEmeans {
    /// A trainer for `k` clusters of dimension `d` on `device`, f32 assignment,
    /// compiled with [`device_options`].
    pub fn new(device: Device, d: usize, k: usize, cfg: EmeansConfig) -> Self {
        Self::with_options(device, d, k, cfg, device_options(device), DType::F32)
    }

    /// [`Self::new`] with compile options (number format, …) and the
    /// assignment precision of [`step_graph_with`].
    pub fn with_options(
        device: Device,
        d: usize,
        k: usize,
        cfg: EmeansConfig,
        options: CompileOptions,
        assign: DType,
    ) -> Self {
        Self {
            device,
            options,
            assign,
            em: Emeans::new(d, k, cfg),
            resident: None,
        }
    }

    /// The run's schedule, statistics and closing centroids.
    pub fn emeans(&self) -> &Emeans {
        &self.em
    }

    /// Whether the last run kept its state on the device (`None` before a run).
    pub fn resident(&self) -> Option<bool> {
        self.resident
    }

    /// Train on `x` (`[n, d]`); same contract and initialization as
    /// [`Emeans::train`].
    pub fn train(&mut self, x: &[f32], init: Option<&[f32]>) -> (Vec<f32>, Option<Vec<u32>>) {
        let (d, k) = (self.em.d, self.em.k);
        assert_eq!(x.len() % d, 0, "e-means: rows must be [n, d]");
        let n = x.len() / d;
        let init = match init {
            Some(c) => c.to_vec(),
            None => initial_centroids(x, d, k, &self.em.cfg),
        };
        self.em.begin(n, &init);
        let recipe = self.em.running_recipe().clone();
        let b = recipe.batch_size;
        let session = Session::new(self.device);
        let mut graph =
            session.compile_with(resident_step_graph(n, b, d, k, self.assign), &self.options);

        let zero_acc = vec![0f32; k * (d + 1)];
        let mut state = bind_resident(&mut graph, x, &self.em, &zero_acc);
        self.resident = Some(matches!(state, State::Resident));

        let capture = self.em.cfg.capture_assignments;
        let mut labels = capture.then(|| vec![u32::MAX; n]);
        // The host copy is newer than the device's fed outputs right after a
        // revival rebinds them (a handle read returns the last output).
        let mut host_fresh = false;
        let mut rows = vec![0f32; b];
        let mut weight = vec![0f32; b];
        for (epoch, &epoch_rows) in recipe.epoch_rows.iter().enumerate() {
            let last = self.em.last_epoch();
            let order = RandomState::new(epoch_seed(self.em.cfg.seed, epoch)).permutation(n);
            for idx in order[..epoch_rows].chunks(b) {
                rows.fill(0.0);
                for (r, &i) in rows.iter_mut().zip(idx) {
                    *r = i as f32;
                }
                for (j, w) in weight.iter_mut().enumerate() {
                    *w = if j < idx.len() { 1.0 } else { 0.0 };
                }
                let alpha = [self.em.current_alpha() as f32];
                let want_labels = capture && last;
                let got = step(
                    &mut graph,
                    &mut state,
                    &mut self.em,
                    &rows,
                    &weight,
                    &alpha,
                    want_labels,
                );
                if let (Some(out), Some(got)) = (labels.as_mut(), got) {
                    for (&i, &l) in idx.iter().zip(&got) {
                        out[i] = l as u32;
                    }
                }
                host_fresh = (self.em.step + 1).is_multiple_of(CHECK_UNUSED_EVERY)
                    && revive(&mut graph, &state, &mut self.em);
                self.em.step += 1;
                self.em.epoch_rows_done += idx.len();
            }
            end_epoch(&mut graph, &mut state, &mut self.em, &zero_acc);
        }
        if !host_fresh {
            sync_model(&graph, &state, &mut self.em);
        }
        (self.em.finalize().to_vec(), labels)
    }
}

/// Bind every piece of state as a fed GPU handle, or fall back to host state.
fn bind_resident(graph: &mut CompiledGraph, x: &[f32], em: &Emeans, zero_acc: &[f32]) -> State {
    use resident_io::*;
    let fed = [
        (CENTROIDS, em.centroids.as_slice(), OUT_CENTROIDS),
        (USAGE, em.usage.as_slice(), OUT_USAGE),
        (ACC, zero_acc, OUT_ACC),
        (OBJECTIVE, &[0.0][..], OUT_OBJECTIVE),
    ];
    let resident = graph.capabilities().gpu_handles
        && graph.bind_gpu_handle(DATA, x)
        && fed.iter().all(|(name, init, out)| {
            graph.bind_gpu_handle(name, init) && graph.set_gpu_handle_feed(name, *out)
        });
    if resident {
        State::Resident
    } else {
        State::Host {
            data: x.to_vec(),
            acc: zero_acc.to_vec(),
            objective: [0.0],
        }
    }
}

/// `outs[i]` whether the backend returned every slot or only the read ones.
fn pick(mut outs: Vec<Vec<f32>>, i: usize, read: &[usize]) -> Vec<f32> {
    if outs.len() == N_OUTPUTS {
        std::mem::take(&mut outs[i])
    } else {
        let at = read.iter().position(|&r| r == i).expect("output was read");
        std::mem::take(&mut outs[at])
    }
}

/// Run one step; returns the batch's labels when asked for.
fn step(
    graph: &mut CompiledGraph,
    state: &mut State,
    em: &mut Emeans,
    rows: &[f32],
    weight: &[f32],
    alpha: &[f32; 1],
    want_labels: bool,
) -> Option<Vec<f32>> {
    use resident_io::*;
    match state {
        State::Resident => {
            let read: &[usize] = if want_labels { &[OUT_LABELS] } else { &[] };
            let outs = graph.run_read_outputs(
                &[(ROWS, rows), (WEIGHT, weight), (ALPHA, alpha)],
                Some(read),
            );
            want_labels.then(|| pick(outs, OUT_LABELS, read))
        }
        State::Host {
            data,
            acc,
            objective,
        } => {
            let mut outs = graph.run(&[
                (DATA, data),
                (ROWS, rows),
                (WEIGHT, weight),
                (CENTROIDS, &em.centroids),
                (USAGE, &em.usage),
                (ALPHA, alpha),
                (ACC, acc),
                (OBJECTIVE, objective),
            ]);
            em.centroids = std::mem::take(&mut outs[OUT_CENTROIDS]);
            em.usage = std::mem::take(&mut outs[OUT_USAGE]);
            *acc = std::mem::take(&mut outs[OUT_ACC]);
            objective[0] = outs[OUT_OBJECTIVE][0];
            want_labels.then(|| std::mem::take(&mut outs[OUT_LABELS]))
        }
    }
}

fn read(graph: &CompiledGraph, name: &str) -> Vec<f32> {
    graph
        .read_gpu_handle(name)
        .unwrap_or_else(|| panic!("e-means: backend lost resident `{name}`"))
}

/// The host trainer's revival check over the device's masses: reads `k`
/// floats, and the centroids only when something is split. True when it
/// split (and rebound the state).
fn revive(graph: &mut CompiledGraph, state: &State, em: &mut Emeans) -> bool {
    let resident = matches!(state, State::Resident);
    if resident {
        em.usage = read(graph, resident_io::USAGE);
    }
    let splits = em.plan_splits();
    if splits.is_empty() {
        return false;
    }
    if resident {
        em.centroids = read(graph, resident_io::CENTROIDS);
    }
    for (src, dst) in splits {
        em.split(src, dst);
    }
    if resident {
        let ok = graph.bind_gpu_handle(resident_io::CENTROIDS, &em.centroids)
            && graph.bind_gpu_handle(resident_io::USAGE, &em.usage);
        assert!(ok, "e-means: backend refused to rebind revived state");
    }
    true
}

/// Fold the epoch's device accumulators into the host trainer, then reset them.
fn end_epoch(graph: &mut CompiledGraph, state: &mut State, em: &mut Emeans, zero_acc: &[f32]) {
    let (acc, objective) = match state {
        State::Resident => {
            let acc = read(graph, resident_io::ACC);
            let objective = read(graph, resident_io::OBJECTIVE)[0];
            let ok = graph.bind_gpu_handle(resident_io::ACC, zero_acc)
                && graph.bind_gpu_handle(resident_io::OBJECTIVE, &[0.0]);
            assert!(ok, "e-means: backend refused to reset the accumulators");
            (acc, objective)
        }
        State::Host { acc, objective, .. } => {
            let out = (std::mem::replace(acc, zero_acc.to_vec()), objective[0]);
            objective[0] = 0.0;
            out
        }
    };
    let d = em.d;
    em.objective = objective as f64;
    for (j, row) in acc.chunks_exact(d + 1).enumerate() {
        em.pass_counts[j] = row[d] as f64;
        if let Some(ps) = em.pass_sums.as_mut() {
            ps[j * d..(j + 1) * d].copy_from_slice(&row[..d]);
        }
    }
    em.end_epoch();
}

/// Copy the device's final positions and masses into the host trainer.
fn sync_model(graph: &CompiledGraph, state: &State, em: &mut Emeans) {
    if matches!(state, State::Resident) {
        em.centroids = read(graph, resident_io::CENTROIDS);
        em.usage = read(graph, resident_io::USAGE);
    }
}
