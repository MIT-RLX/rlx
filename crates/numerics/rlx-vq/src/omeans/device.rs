// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! o-means on an rlx device: the two-level assignment, the statistics and the
//! EMA run in compiled graphs over device-resident state.
//!
//! [`DeviceOmeans`] keeps [`Omeans`]' schedule, revival, cell maintenance and
//! closing mean on the host, and moves every per-row computation to the
//! device:
//!
//! - [`probe_graph`] scores rows against the cell centres and keeps each
//!   row's `nprobe` nearest cells (`TopK`). A row's probes depend only on the
//!   cell centres and on which cells are empty, never on the centroids, so
//!   they are computed for all `n` rows at once, and again only when the
//!   centres move: at the first build and at each refit with Lloyd iterations
//!   (every half pass after the first).
//! - [`nearest_cell_graph`] files the centroids under their nearest cell
//!   centre for each cell build and refit (every [`CHECK_UNUSED_EVERY`] steps
//!   during the first pass).
//! - [`tiled_step_graph`] runs one training step. Each probed cell's rows are
//!   cut into pages of [`TILE_ROWS`] and its centroids into blocks of
//!   [`TILE_CENTROIDS`]; every (page, block) pair of a cell is one unit, and
//!   the units run as one batched matmul. Each row then takes the best of its
//!   candidate units (`gather` + `argmin`), checks the split links of the
//!   winner, and the step ends with e-means' scatter statistics and EMA, as
//!   [`crate::emeans::device::DeviceEmeans`] runs them.
//!
//! The host builds a step's unit and candidate index arrays from the probes and
//! the cell membership alone, uploads them with the batch's row ids, and reads
//! nothing back except the masses at each revival check, the centroids when
//! something is split or the cells are refit, and the objective once per
//! epoch. Distances are taken relative to the dataset mean: the argmin is the
//! same for any shift, but far from the origin `‖c‖²` and `2x·c` dwarf the
//! gaps between nearby centroids.

use rlx_ir::infer::GraphExt as _;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};

use super::*;
use crate::emeans::build_update;
use crate::emeans::device::device_options;

/// Rows per unit page.
pub const TILE_ROWS: usize = 128;
/// Centroids per unit block.
pub const TILE_CENTROIDS: usize = 64;
/// Added to the norm of an empty cell or a padding centroid, so it never wins
/// while a real candidate exists.
const PAD_NORM: f32 = 1e30;
/// A probe pass scores at most this many (row, cell) pairs per run.
const PROBE_PAIRS: usize = 1 << 22;

/// Input names of [`probe_graph`] and [`tiled_step_graph`].
pub mod io {
    /// The whole training set `[n, d]` (bound once when resident).
    pub const DATA: &str = "data";
    /// Row ids into `data`, f32-encoded.
    pub const ROWS: &str = "rows";
    /// `[1, d]` subtracted from rows, cell centres and centroids.
    pub const SHIFT: &str = "shift";
    /// Cell centres `[nlist, d]`.
    pub const CENTRES: &str = "centres";
    /// `[nlist]`: 0 for a cell with centroids, a huge value for an empty one.
    pub const CELL_BIAS: &str = "cell_bias";
    /// Weight of each batch row `[B, 1]`: 1 for a real row, 0 for padding.
    pub const WEIGHT: &str = "weight";
    /// Scheduled step `α`, `[1]`.
    pub const ALPHA: &str = "alpha";
    /// Weight of this batch in the closing mean, `[1]` (0 if not part of it).
    pub const OMEGA: &str = "omega";
    /// Centroids `[k, d]` (fed in place).
    pub const CENTROIDS: &str = "centroids";
    /// EMA masses `[k]` (fed in place).
    pub const USAGE: &str = "usage";
    /// Closing accumulator `Σ ω·[sums | counts]`, `[k, d+1]` (fed in place).
    pub const ACC: &str = "acc";
    /// Objective of the epoch, `[1]` (fed in place).
    pub const OBJECTIVE: &str = "objective";
    /// Dataset row of each unit row, `[U · TILE_ROWS]`.
    pub const UNIT_ROWS: &str = "unit_rows";
    /// Centroid of each unit column, `[U · TILE_CENTROIDS]` (`k` = padding).
    pub const UNIT_CENTROIDS: &str = "unit_centroids";
    /// `u · TILE_CENTROIDS` for every unit row, `[U · TILE_ROWS]` (constant).
    pub const UNIT_BASE: &str = "unit_base";
    /// Unit rows of each batch row, `[B · M]` (`U · TILE_ROWS` = padding).
    pub const CANDIDATES: &str = "candidates";
    /// `r · M` for every batch row, `[B]` (constant).
    pub const CANDIDATE_BASE: &str = "candidate_base";
    /// `3r` for every batch row, `[B]` (constant).
    pub const LINK_BASE: &str = "link_base";
    /// Latest split child of every centroid, or the centroid itself, `[k]`.
    pub const CHILD: &str = "child";
}

/// Probes of `rows` dataset rows: output `[rows, nprobe]`, the cells in
/// ascending order of `‖g‖² − 2x·g` (f32-encoded, ties to the lower cell).
pub fn probe_graph(n: usize, rows: usize, d: usize, nlist: usize, nprobe: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("omeans_probe");
    let data = g.input(io::DATA, Shape::new(&[n, d], f));
    let ids = g.input(io::ROWS, Shape::new(&[rows], f));
    let centres = g.input(io::CENTRES, Shape::new(&[nlist, d], f));
    let bias = g.input(io::CELL_BIAS, Shape::new(&[nlist], f));
    let shift = g.input(io::SHIFT, Shape::new(&[1, d], f));
    let x = g.gather_(data, ids, 0);
    let xs = g.sub(x, shift);
    let gs = g.sub(centres, shift);
    let gs_sq = g.mul(gs, gs);
    let gn = g.sum(gs_sq, vec![1], false);
    let gn = g.add(gn, bias);
    let gn_row = g.reshape_(gn, vec![1, nlist as i64]);
    let gs_t = g.transpose_(gs, vec![1, 0]);
    let cross = g.mm(xs, gs_t);
    let two = g.constant(2.0, f);
    let two_cross = g.mul(cross, two);
    // TopK keeps the largest: rank by −score = 2x·g − ‖g‖².
    let neg_score = g.sub(two_cross, gn_row);
    let probes = g.add_node(
        Op::TopK { k: nprobe },
        vec![neg_score],
        Shape::new(&[rows, nprobe], f),
    );
    g.set_outputs(vec![probes]);
    g
}

/// Nearest cell centre of each of `k` centroids: output `[k]` (f32-encoded,
/// ties to the lower cell). Inputs [`io::CENTROIDS`], [`io::CENTRES`],
/// [`io::SHIFT`]. Used for the cell builds and refits.
pub fn nearest_cell_graph(k: usize, d: usize, nlist: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("omeans_nearest_cell");
    let c = g.input(io::CENTROIDS, Shape::new(&[k, d], f));
    let centres = g.input(io::CENTRES, Shape::new(&[nlist, d], f));
    let shift = g.input(io::SHIFT, Shape::new(&[1, d], f));
    let cs = g.sub(c, shift);
    let gs = g.sub(centres, shift);
    let gs_sq = g.mul(gs, gs);
    let gn = g.sum(gs_sq, vec![1], false);
    let gn_row = g.reshape_(gn, vec![1, nlist as i64]);
    let gs_t = g.transpose_(gs, vec![1, 0]);
    let cross = g.mm(cs, gs_t);
    let neg_two = g.constant(-2.0, f);
    let neg_two_cross = g.mul(cross, neg_two);
    let score = g.add(gn_row, neg_two_cross);
    let cell = g.argmin(score, 1, false, Shape::new(&[k], f));
    g.set_outputs(vec![cell]);
    g
}

const OUT_CENTROIDS: usize = 0;
const OUT_USAGE: usize = 1;
const OUT_ACC: usize = 2;
const OUT_OBJECTIVE: usize = 3;
const OUT_LABELS: usize = 4;
const N_OUTPUTS: usize = 5;

/// `‖c − μ‖² − 2(x − μ)·(c − μ)` of every shifted row `xs[B, d]` against
/// centroid `ids[B]` of the shifted table `cs[k, d]` with norms `cn[k]`, as
/// `[B, 1]`.
fn link_score(g: &mut Graph, xs: NodeId, cs: NodeId, cn: NodeId, ids: NodeId) -> NodeId {
    let b = g.shape(xs).dim(0).unwrap_static();
    let c = g.gather_(cs, ids, 0);
    let prod = g.mul(xs, c);
    let cross = g.sum(prod, vec![1], false);
    let norm = g.gather_(cn, ids, 0);
    let neg_two = g.constant(-2.0, DType::F32);
    let neg_two_cross = g.mul(cross, neg_two);
    let s = g.add(norm, neg_two_cross);
    g.reshape_(s, vec![b as i64, 1])
}

/// One o-means step over `units` tiles and `m` candidate unit rows per batch
/// row. Inputs: see [`io`]. Outputs, in order: `centroids'[k, d]`,
/// `usage'[k]`, `acc + ω·stats [k, d+1]`, `objective + Σ‖x − c‖² [1]`,
/// `labels[B]`. The first four are meant to feed their inputs in place.
pub fn tiled_step_graph(n: usize, b: usize, d: usize, k: usize, units: usize, m: usize) -> Graph {
    let f = DType::F32;
    let (tr, tc) = (TILE_ROWS, TILE_CENTROIDS);
    let mut g = Graph::new("omeans_tiled_step");
    let data = g.input(io::DATA, Shape::new(&[n, d], f));
    let rows = g.input(io::ROWS, Shape::new(&[b], f));
    let w = g.input(io::WEIGHT, Shape::new(&[b, 1], f));
    let alpha = g.input(io::ALPHA, Shape::new(&[1], f));
    let omega = g.input(io::OMEGA, Shape::new(&[1], f));
    let shift = g.input(io::SHIFT, Shape::new(&[1, d], f));
    let c = g.input(io::CENTROIDS, Shape::new(&[k, d], f));
    let usage = g.input(io::USAGE, Shape::new(&[k], f));
    let acc = g.input(io::ACC, Shape::new(&[k, d + 1], f));
    let objective = g.input(io::OBJECTIVE, Shape::new(&[1], f));
    let unit_rows = g.input(io::UNIT_ROWS, Shape::new(&[units * tr], f));
    let unit_cents = g.input(io::UNIT_CENTROIDS, Shape::new(&[units * tc], f));
    let unit_base = g.input(io::UNIT_BASE, Shape::new(&[units * tr], f));
    let cand = g.input(io::CANDIDATES, Shape::new(&[b * m], f));
    let cand_base = g.input(io::CANDIDATE_BASE, Shape::new(&[b], f));
    let link_base = g.input(io::LINK_BASE, Shape::new(&[b], f));
    let child = g.input(io::CHILD, Shape::new(&[k], f));

    // Shifted centroids and norms, plus one padding centroid (id k).
    let cs = g.sub(c, shift);
    let cs_sq = g.mul(cs, cs);
    let cn = g.sum(cs_sq, vec![1], false);
    // −2·c, gathered per unit, so the score is one matmul plus the norms.
    let neg_two = g.constant(-2.0, f);
    let cs_m2 = g.mul(cs, neg_two);
    let zero_row = g.full(&[1, d], 0.0, f);
    let cs_ext = g.concat_(vec![cs_m2, zero_row], 0);
    let pad_norm = g.full(&[1], PAD_NORM, f);
    let cn_ext = g.concat_(vec![cn, pad_norm], 0);

    // Units: [U, TR, d] rows against [U, d, TC] centroids.
    let xu = g.gather_(data, unit_rows, 0);
    let xu = g.sub(xu, shift);
    let xu = g.reshape_(xu, vec![units as i64, tr as i64, d as i64]);
    let cu = g.gather_(cs_ext, unit_cents, 0);
    let cu = g.reshape_(cu, vec![units as i64, tc as i64, d as i64]);
    let cu_t = g.transpose_(cu, vec![0, 2, 1]);
    let nu = g.gather_(cn_ext, unit_cents, 0);
    let nu = g.reshape_(nu, vec![units as i64, 1, tc as i64]);
    let neg_two_cross = g.mm(xu, cu_t);
    let score = g.add(nu, neg_two_cross);
    let unit_out = Shape::new(&[units, tr], f);
    // A min-reduce rather than a gather at the argmin: the flat score index
    // `i · TC + slot` passes 2^24 (exact in f32) beyond 2048 units.
    let best_u = g.reduce(score, ReduceOp::Min, vec![2], false, unit_out.clone());
    let best_u = g.reshape_(best_u, vec![(units * tr) as i64]);
    let slot_u = g.argmin(score, 2, false, unit_out);
    let slot_u = g.reshape_(slot_u, vec![(units * tr) as i64]);
    let at = g.add(unit_base, slot_u);
    let label_u = g.gather_(unit_cents, at, 0);

    // Each batch row: best of its candidate unit rows (padding → PAD_NORM).
    let pad_best = g.full(&[1], PAD_NORM, f);
    let best_ext = g.concat_(vec![best_u, pad_best], 0);
    let zero = g.full(&[1], 0.0, f);
    let label_ext = g.concat_(vec![label_u, zero], 0);
    let cb = g.gather_(best_ext, cand, 0);
    let cb = g.reshape_(cb, vec![b as i64, m as i64]);
    let cl = g.gather_(label_ext, cand, 0);
    let pick = g.argmin(cb, 1, false, Shape::new(&[b], f));
    let best = g.reduce(cb, ReduceOp::Min, vec![1], true, Shape::new(&[b, 1], f));
    let at = g.add(cand_base, pick);
    let label = g.gather_(cl, at, 0);

    // Split links: the winner's child and grandchild. argmin over
    // [best, s(c1), s(c2)] keeps the first minimum, which is the reference's
    // sequential strict comparison.
    let x = g.gather_(data, rows, 0);
    let xs = g.sub(x, shift);
    let c1 = g.gather_(child, label, 0);
    let c2 = g.gather_(child, c1, 0);
    let s1 = link_score(&mut g, xs, cs, cn, c1);
    let s2 = link_score(&mut g, xs, cs, cn, c2);
    let s3 = g.concat_(vec![best, s1, s2], 1);
    let label_col = g.reshape_(label, vec![b as i64, 1]);
    let c1_col = g.reshape_(c1, vec![b as i64, 1]);
    let c2_col = g.reshape_(c2, vec![b as i64, 1]);
    let l3 = g.concat_(vec![label_col, c1_col, c2_col], 1);
    let l3 = g.reshape_(l3, vec![(3 * b) as i64]);
    let pick = g.argmin(s3, 1, false, Shape::new(&[b], f));
    let at = g.add(link_base, pick);
    let labels = g.gather_(l3, at, 0);

    let s = build_update(&mut g, x, Some(w), c, usage, alpha, labels);
    let weighted = g.mul(s.stats, omega);
    let acc_new = g.add(acc, weighted);
    let objective_new = g.add(objective, s.objective);
    g.set_outputs(vec![s.centroids, s.usage, acc_new, objective_new, labels]);
    g
}

/// How the device path ran.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceReport {
    /// Probe passes over all `n` rows (one per change of the cell centres).
    pub probe_passes: usize,
    /// Compilations of [`tiled_step_graph`] (the units or candidates outgrew
    /// their capacity).
    pub step_compiles: usize,
    /// Final unit capacity `U`.
    pub units: usize,
    /// Final candidates per row `M`.
    pub candidates: usize,
    /// Row-centroid products the units computed, padding included; compare
    /// with [`OmeansReport::exact_distance_evals`].
    pub unit_evals: u64,
    /// Whether the training state stayed on the device (`false` on backends
    /// without GPU handles, which receive it with every run).
    pub resident: bool,
    /// Wall time of the probe passes.
    pub probe_seconds: f64,
    /// Wall time of the step graph runs.
    pub step_seconds: f64,
    /// Wall time of building the unit and candidate arrays.
    pub tile_seconds: f64,
    /// Wall time of cell builds and refits, reads of the centroids included.
    pub cell_seconds: f64,
    /// Wall time of the revival checks, reads and rebinds included.
    pub revival_seconds: f64,
}

/// Bind the `data` input once, when the backend keeps GPU handles.
fn bind_data(graph: &mut CompiledGraph, data: &[f32]) -> bool {
    graph.capabilities().gpu_handles && graph.bind_gpu_handle(io::DATA, data)
}

/// Unit and candidate arrays of one batch.
struct Tiles {
    units: usize,
    m: usize,
    unit_rows: Vec<f32>,
    unit_cents: Vec<f32>,
    /// `[B, m]` unit-row positions, `u32::MAX` = padding.
    cand: Vec<u32>,
    /// Centroids the probed cells hold, summed over rows.
    evals: u64,
}

/// Cut the batch's (row, probed cell) pairs into units and list each row's
/// candidate unit rows, by probe rank then block.
fn tile_batch(
    idx: &[usize],
    probes: &[u32],
    nprobe: usize,
    index: &CentroidIndex,
    k: usize,
) -> Tiles {
    let (tr, tc) = (TILE_ROWS, TILE_CENTROIDS);
    let g = index.nlist();
    let sizes = index.cell_sizes();
    let blocks: Vec<usize> = sizes.iter().map(|s| s.div_ceil(tc)).collect();
    let mut rows_in = vec![0usize; g];
    let mut m = 1;
    for &i in idx {
        let mut per_row = 0;
        for &c in &probes[i * nprobe..(i + 1) * nprobe] {
            rows_in[c as usize] += 1;
            per_row += blocks[c as usize];
        }
        m = m.max(per_row);
    }
    // First unit of each cell; a cell's units run page-major.
    let mut first = vec![0usize; g + 1];
    for c in 0..g {
        first[c + 1] = first[c] + rows_in[c].div_ceil(tr) * blocks[c];
    }
    let units = first[g];
    let mut unit_rows = vec![0f32; units * tr];
    let mut unit_cents = vec![k as f32; units * tc];
    for c in 0..g {
        let members = &index.members[index.offsets[c]..index.offsets[c + 1]];
        for page in 0..rows_in[c].div_ceil(tr) {
            for (bl, chunk) in members.chunks(tc).enumerate() {
                let u = first[c] + page * blocks[c] + bl;
                for (q, &j) in chunk.iter().enumerate() {
                    unit_cents[u * tc + q] = j as f32;
                }
            }
        }
    }
    let mut cand = vec![u32::MAX; idx.len() * m];
    let mut fill = vec![0usize; g];
    let mut evals = 0u64;
    for (r, &i) in idx.iter().enumerate() {
        let mut slot = 0;
        for &c in &probes[i * nprobe..(i + 1) * nprobe] {
            let c = c as usize;
            let (page, o) = (fill[c] / tr, fill[c] % tr);
            fill[c] += 1;
            for bl in 0..blocks[c] {
                let u = first[c] + page * blocks[c] + bl;
                unit_rows[u * tr + o] = i as f32;
                cand[r * m + slot] = (u * tr + o) as u32;
                slot += 1;
            }
            evals += sizes[c] as u64;
        }
    }
    Tiles {
        units,
        m,
        unit_rows,
        unit_cents,
        cand,
        evals,
    }
}

/// Every row's probes, recomputed when the centres or empty cells change.
#[derive(Default)]
struct Probes {
    stage: Option<(CompiledGraph, bool, (usize, usize, usize))>,
    probes: Vec<u32>,
    key: Vec<f32>,
}

impl Probes {
    #[allow(clippy::too_many_arguments)]
    fn refresh(
        &mut self,
        session: &Session,
        options: &CompileOptions,
        data: &[f32],
        d: usize,
        shift: &[f32],
        index: &CentroidIndex,
        nprobe: usize,
        report: &mut DeviceReport,
    ) {
        let n = data.len() / d;
        let g = index.nlist();
        let bias: Vec<f32> = index
            .centre_norms
            .iter()
            .map(|&v| if v == EMPTY_NORM { PAD_NORM } else { 0.0 })
            .collect();
        let mut key = index.centres.clone();
        key.extend_from_slice(&bias);
        key.push(nprobe as f32);
        if key == self.key {
            return;
        }
        let t0 = std::time::Instant::now();
        let rows = n.min((PROBE_PAIRS / g).max(1));
        let shape = (rows, g, nprobe);
        if self.stage.as_ref().is_none_or(|(_, _, s)| *s != shape) {
            let mut graph = session.compile_with(probe_graph(n, rows, d, g, nprobe), options);
            let resident = bind_data(&mut graph, data);
            self.stage = Some((graph, resident, shape));
        }
        let (graph, resident, _) = self.stage.as_mut().expect("compiled above");
        self.probes = vec![0; n * nprobe];
        let mut ids = vec![0f32; rows];
        for start in (0..n).step_by(rows) {
            let len = rows.min(n - start);
            ids.fill(0.0);
            for (j, v) in ids[..len].iter_mut().enumerate() {
                *v = (start + j) as f32;
            }
            let mut inputs: Vec<(&str, &[f32])> = vec![
                (io::ROWS, &ids),
                (io::CENTRES, &index.centres),
                (io::CELL_BIAS, &bias),
                (io::SHIFT, shift),
            ];
            let out = if *resident {
                graph.run_read_outputs(&inputs, Some(&[0]))
            } else {
                inputs.push((io::DATA, data));
                graph.run(&inputs)
            };
            for (p, &c) in self.probes[start * nprobe..(start + len) * nprobe]
                .iter_mut()
                .zip(&out[0])
            {
                *p = c as u32;
            }
        }
        self.key = key;
        report.probe_passes += 1;
        report.probe_seconds += t0.elapsed().as_secs_f64();
    }
}

/// Centroids, masses, closing accumulator and objective.
type State = (Vec<f32>, Vec<f32>, Vec<f32>, [f32; 1]);

/// The compiled step and the training state it owns.
struct Step {
    graph: CompiledGraph,
    shape: (usize, usize),
    resident: bool,
    /// Host copies; authoritative when `!resident`, or right after a rebind
    /// (a handle read returns the last output until the next run).
    centroids: Vec<f32>,
    usage: Vec<f32>,
    acc: Vec<f32>,
    objective: [f32; 1],
    host_fresh: bool,
    /// Constant inputs, passed with every run when not resident.
    unit_base: Vec<f32>,
    cand_base: Vec<f32>,
    link_base: Vec<f32>,
}

impl Step {
    #[allow(clippy::too_many_arguments)]
    fn compile(
        session: &Session,
        options: &CompileOptions,
        data: &[f32],
        b: usize,
        d: usize,
        k: usize,
        shape: (usize, usize),
        state: State,
    ) -> Self {
        let n = data.len() / d;
        let (units, m) = shape;
        // Every index the graph gathers with is f32-encoded.
        let largest = [
            n,
            k + 1,
            units * TILE_ROWS + 1,
            units * TILE_CENTROIDS,
            3 * b,
            b * m,
        ];
        assert!(
            largest.iter().all(|&v| v <= 1 << 24),
            "o-means: an index of the tiled step passes 2^24 ({largest:?})"
        );
        let mut graph = session.compile_with(tiled_step_graph(n, b, d, k, units, m), options);
        let unit_base: Vec<f32> = (0..units * TILE_ROWS)
            .map(|i| ((i / TILE_ROWS) * TILE_CENTROIDS) as f32)
            .collect();
        let cand_base: Vec<f32> = (0..b).map(|r| (r * m) as f32).collect();
        let link_base: Vec<f32> = (0..b).map(|r| (3 * r) as f32).collect();
        let (centroids, usage, acc, objective) = state;
        let fed = [
            (io::CENTROIDS, &centroids[..], OUT_CENTROIDS),
            (io::USAGE, &usage[..], OUT_USAGE),
            (io::ACC, &acc[..], OUT_ACC),
            (io::OBJECTIVE, &objective[..], OUT_OBJECTIVE),
        ];
        let resident = bind_data(&mut graph, data)
            && graph.bind_gpu_handle(io::UNIT_BASE, &unit_base)
            && graph.bind_gpu_handle(io::CANDIDATE_BASE, &cand_base)
            && graph.bind_gpu_handle(io::LINK_BASE, &link_base)
            && fed.iter().all(|(name, init, out)| {
                graph.bind_gpu_handle(name, init) && graph.set_gpu_handle_feed(name, *out)
            });
        Self {
            graph,
            shape,
            resident,
            centroids,
            usage,
            acc,
            objective,
            host_fresh: true,
            unit_base,
            cand_base,
            link_base,
        }
    }

    fn read(&self, name: &str) -> Vec<f32> {
        if !self.resident || self.host_fresh {
            return match name {
                io::CENTROIDS => self.centroids.clone(),
                io::USAGE => self.usage.clone(),
                io::ACC => self.acc.clone(),
                _ => self.objective.to_vec(),
            };
        }
        self.graph
            .read_gpu_handle(name)
            .unwrap_or_else(|| panic!("o-means: backend lost resident `{name}`"))
    }

    /// The current state, from the device when it is newer.
    fn state(&self) -> State {
        (
            self.read(io::CENTROIDS),
            self.read(io::USAGE),
            self.read(io::ACC),
            [self.read(io::OBJECTIVE)[0]],
        )
    }

    /// Replace one piece of state.
    fn write(&mut self, name: &str, value: Vec<f32>) {
        if self.resident {
            // After a rebind, reads of the other handles would return the
            // last output; make the host copies current first.
            if !self.host_fresh {
                (self.centroids, self.usage, self.acc, self.objective) = self.state();
                self.host_fresh = true;
            }
            assert!(
                self.graph.bind_gpu_handle(name, &value),
                "o-means: backend refused to rebind `{name}`"
            );
        }
        match name {
            io::CENTROIDS => self.centroids = value,
            io::USAGE => self.usage = value,
            io::ACC => self.acc = value,
            _ => self.objective = [value[0]],
        }
    }

    /// Run one step; returns the batch's labels when asked for.
    fn run(
        &mut self,
        data: &[f32],
        per_step: &[(&str, &[f32])],
        want_labels: bool,
    ) -> Option<Vec<f32>> {
        if self.resident {
            let read: &[usize] = if want_labels { &[OUT_LABELS] } else { &[] };
            let mut outs = self.graph.run_read_outputs(per_step, Some(read));
            self.host_fresh = false;
            return want_labels.then(|| {
                let at = if outs.len() == N_OUTPUTS {
                    OUT_LABELS
                } else {
                    0
                };
                std::mem::take(&mut outs[at])
            });
        }
        let mut inputs: Vec<(&str, &[f32])> = per_step.to_vec();
        inputs.extend_from_slice(&[
            (io::DATA, data),
            (io::CENTROIDS, &self.centroids),
            (io::USAGE, &self.usage),
            (io::ACC, &self.acc),
            (io::OBJECTIVE, &self.objective),
            (io::UNIT_BASE, &self.unit_base),
            (io::CANDIDATE_BASE, &self.cand_base),
            (io::LINK_BASE, &self.link_base),
        ]);
        let mut outs = self.graph.run(&inputs);
        self.centroids = std::mem::take(&mut outs[OUT_CENTROIDS]);
        self.usage = std::mem::take(&mut outs[OUT_USAGE]);
        self.acc = std::mem::take(&mut outs[OUT_ACC]);
        self.objective = [outs[OUT_OBJECTIVE][0]];
        want_labels.then(|| std::mem::take(&mut outs[OUT_LABELS]))
    }
}

/// The compile options [`DeviceOmeans::new`] uses on `device`: e-means'
/// [`device_options`] with strict FP32 matmuls. CUDA otherwise multiplies in
/// TF32, whose 10-bit inputs picked a centroid 0.1% farther than the nearest
/// for 3 of 540 rows; pass `exact_f32_matmul = false` through
/// [`DeviceOmeans::with_options`] to trade that for speed.
pub fn omeans_options(device: Device) -> CompileOptions {
    device_options(device).with_exact_f32_matmul(true)
}

/// o-means with the training state on an rlx device (see the
/// [module docs](self)).
pub struct DeviceOmeans {
    device: Device,
    options: CompileOptions,
    om: Omeans,
    report: DeviceReport,
}

impl DeviceOmeans {
    /// A trainer for `k` clusters of dimension `d` on `device`, compiled with
    /// [`omeans_options`].
    pub fn new(device: Device, d: usize, k: usize, cfg: OmeansConfig) -> Self {
        Self::with_options(device, d, k, cfg, omeans_options(device))
    }

    /// [`Self::new`] with explicit compile options.
    pub fn with_options(
        device: Device,
        d: usize,
        k: usize,
        cfg: OmeansConfig,
        options: CompileOptions,
    ) -> Self {
        Self {
            device,
            options,
            om: Omeans::new(d, k, cfg),
            report: DeviceReport::default(),
        }
    }

    /// The run's schedule, counters, centroids and final index.
    pub fn omeans(&self) -> &Omeans {
        &self.om
    }

    /// How the device path ran.
    pub fn device_report(&self) -> &DeviceReport {
        &self.report
    }

    /// Train on `x` (`[n, d]`); same contract and initialization as
    /// [`Omeans::train`].
    pub fn train(&mut self, x: &[f32], init: Option<&[f32]>) -> (Vec<f32>, Option<Vec<u32>>) {
        let (d, k) = (self.om.d, self.om.k);
        assert!(self.om.schedule.is_none(), "o-means: one run per object");
        assert_eq!(x.len() % d, 0, "o-means: rows must be [n, d]");
        let n = x.len() / d;
        // Row ids, centroid ids and unit positions travel as f32.
        assert!(n < 1 << 24, "o-means: the device path needs n < 2^24 rows");
        let sched = schedule(n, d, k, &self.om.cfg);
        let recipe = sched.recipe.clone();
        let b = recipe.batch_size;
        let init = match init {
            Some(c) => {
                assert_eq!(c.len(), k * d, "o-means: initial centroids must be [k, d]");
                c.to_vec()
            }
            None => initial_centroids(x, d, k, &self.om.cfg.emeans),
        };
        let seed = self.om.cfg.emeans.seed;
        let mut split_rng = RandomState::new(seed);
        let cell_seed = (seed as u64 * 7919 + 17) as u32;
        let last = recipe.epoch_rows.len() - 1;
        let mut report = DeviceReport::default();
        let mut counts = OmeansReport::default();

        let mut mean = vec![0f64; d];
        for row in x.chunks_exact(d) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += *v as f64;
            }
        }
        let shift: Vec<f32> = mean.iter().map(|m| (m / n as f64) as f32).collect();
        let session = Session::new(self.device);
        let mut probes = Probes::default();
        let mut cell_graph: Option<CompiledGraph> = None;
        let mut index: Option<CentroidIndex> = None;
        let mut step: Option<Step> = None;
        let mut pending: Option<State> =
            Some((init, vec![INIT_USAGE; k], vec![0f32; k * (d + 1)], [0.0]));
        let mut labels = self
            .om
            .cfg
            .emeans
            .capture_assignments
            .then(|| vec![u32::MAX; n]);
        let (mut done, mut since_fit, mut t) = (0usize, 0usize, 0usize);
        let mut rows = vec![0f32; b];
        let mut weight = vec![0f32; b];

        for (epoch, &epoch_rows) in recipe.epoch_rows.iter().enumerate() {
            let order = RandomState::new(epoch_seed(seed, epoch)).permutation(n);
            for idx in order[..epoch_rows].chunks(b) {
                let t0 = std::time::Instant::now();
                let first_pass = done < n;
                let every = if first_pass {
                    CHECK_UNUSED_EVERY
                } else {
                    sched.rebuild_every
                };
                let fit = match index {
                    None => Some(FIRST_BUILD_ITERS),
                    Some(_) if since_fit >= every => Some(if first_pass {
                        FIRST_PASS_REBUILD_ITERS
                    } else {
                        REBUILD_ITERS
                    }),
                    Some(_) => None,
                };
                if let Some(iters) = fit {
                    let centroids = match (&step, &pending) {
                        (Some(s), _) => s.read(io::CENTROIDS),
                        (None, Some(st)) => st.0.clone(),
                        (None, None) => unreachable!("state lives in the step or pending"),
                    };
                    let mut nearest = |centres: &[f32], rows: &[f32]| -> Vec<usize> {
                        let graph = cell_graph.get_or_insert_with(|| {
                            session
                                .compile_with(nearest_cell_graph(k, d, sched.nlist), &self.options)
                        });
                        graph.run(&[
                            (io::CENTROIDS, rows),
                            (io::CENTRES, centres),
                            (io::SHIFT, &shift),
                        ])[0]
                            .iter()
                            .map(|&c| c as usize)
                            .collect()
                    };
                    match index.as_mut() {
                        None => {
                            index = Some(CentroidIndex::build_with(
                                &centroids,
                                d,
                                sched.nlist,
                                iters,
                                cell_seed,
                                &mut nearest,
                            ))
                        }
                        Some(ix) => ix.refit_with(&centroids, iters, &mut nearest),
                    }
                    counts.distance_evals += ((iters + 1) * k * sched.nlist) as u64;
                    counts.rebuilds += 1;
                    since_fit = 0;
                }
                since_fit += 1;
                report.cell_seconds += t0.elapsed().as_secs_f64();
                let ix = index.as_ref().expect("built above");
                probes.refresh(
                    &session,
                    &self.options,
                    x,
                    d,
                    &shift,
                    ix,
                    sched.nprobe,
                    &mut report,
                );

                let t0 = std::time::Instant::now();
                let tiles = tile_batch(idx, &probes.probes, sched.nprobe, ix, k);
                let fits = step
                    .as_ref()
                    .is_some_and(|s| s.shape.0 >= tiles.units && s.shape.1 >= tiles.m);
                if !fits {
                    // Grow with a quarter of headroom and never shrink, so
                    // the capacities settle within a few steps.
                    let old = step.as_ref().map_or((1, 1), |s| s.shape);
                    let grow = |want: usize, had: usize| (want + want / 4).max(had);
                    let shape = (grow(tiles.units, old.0), grow(tiles.m, old.1));
                    let state = match step.as_ref() {
                        Some(s) => s.state(),
                        None => pending.take().expect("initial state"),
                    };
                    step = Some(Step::compile(
                        &session,
                        &self.options,
                        x,
                        b,
                        d,
                        k,
                        shape,
                        state,
                    ));
                    report.step_compiles += 1;
                    (report.units, report.candidates) = shape;
                }
                let s = step.as_mut().expect("compiled above");
                let (units, m) = s.shape;
                let mut unit_rows = tiles.unit_rows;
                unit_rows.resize(units * TILE_ROWS, 0.0);
                let mut unit_cents = tiles.unit_cents;
                unit_cents.resize(units * TILE_CENTROIDS, k as f32);
                let pad = (units * TILE_ROWS) as f32;
                let mut cand = vec![pad; b * m];
                for (r, row) in tiles.cand.chunks(tiles.m).enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        if v != u32::MAX {
                            cand[r * m + j] = v as f32;
                        }
                    }
                }
                rows.fill(0.0);
                for (r, &i) in rows.iter_mut().zip(idx) {
                    *r = i as f32;
                }
                for (j, w) in weight.iter_mut().enumerate() {
                    *w = if j < idx.len() { 1.0 } else { 0.0 };
                }
                let child: Vec<f32> = ix
                    .child
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| if c < 0 { j as f32 } else { c as f32 })
                    .collect();
                done += idx.len();
                let omega = sched.closing_weight(n, epoch, done).unwrap_or(0.0) as f32;
                let alpha = scheduled_alpha(recipe.alpha0, t, recipe.total_steps) as f32;
                counts.distance_evals += tiles.evals + (idx.len() * sched.nlist) as u64;
                counts.exact_distance_evals += (idx.len() * k) as u64;
                report.unit_evals += (tiles.units * TILE_ROWS * TILE_CENTROIDS) as u64;
                report.tile_seconds += t0.elapsed().as_secs_f64();

                let t0 = std::time::Instant::now();
                let want = epoch == last && labels.is_some();
                let got = s.run(
                    x,
                    &[
                        (io::ROWS, &rows),
                        (io::WEIGHT, &weight),
                        (io::ALPHA, &[alpha]),
                        (io::OMEGA, &[omega]),
                        (io::SHIFT, &shift),
                        (io::UNIT_ROWS, &unit_rows),
                        (io::UNIT_CENTROIDS, &unit_cents),
                        (io::CANDIDATES, &cand),
                        (io::CHILD, &child),
                    ],
                    want,
                );
                report.step_seconds += t0.elapsed().as_secs_f64();
                if let (Some(out), Some(got)) = (labels.as_mut(), got) {
                    for (&i, &l) in idx.iter().zip(&got) {
                        out[i] = l as u32;
                    }
                }

                if sched.checks_after(t) {
                    let t0 = std::time::Instant::now();
                    let mut usage = s.read(io::USAGE);
                    let splits = plan_splits(&usage, &mut split_rng, recipe.unused_threshold);
                    if !splits.is_empty() {
                        let mut centroids = s.read(io::CENTROIDS);
                        let mut acc = sched.recency_close.then(|| s.read(io::ACC));
                        let ix = index.as_mut().expect("built above");
                        for &(src, dst) in &splits {
                            split_pair(&mut centroids, &mut usage, d, src, dst);
                            if let Some(acc) = acc.as_mut() {
                                // Donor and child share the donor's closing mass.
                                for v in 0..=d {
                                    acc[src * (d + 1) + v] *= 0.5;
                                    acc[dst * (d + 1) + v] = acc[src * (d + 1) + v];
                                }
                            }
                            ix.record_split(src, dst);
                        }
                        counts.n_splits += splits.len();
                        s.write(io::CENTROIDS, centroids);
                        s.write(io::USAGE, usage);
                        if let Some(acc) = acc {
                            s.write(io::ACC, acc);
                        }
                    }
                    report.revival_seconds += t0.elapsed().as_secs_f64();
                }
                t += 1;
            }
            let s = step.as_mut().expect("every epoch has a step");
            counts
                .epoch_objectives
                .push(s.read(io::OBJECTIVE)[0] as f64);
            s.write(io::OBJECTIVE, vec![0.0]);
        }

        let s = step.as_ref().expect("every epoch has a step");
        let (mut centroids, usage, acc, _) = s.state();
        report.resident = s.resident;
        // Closing mean: a centroid without closing mass keeps its EMA position.
        let min_mass = if sched.recency_close {
            MIN_CLOSING_MASS as f32
        } else {
            0.0
        };
        for (c, a) in centroids.chunks_exact_mut(d).zip(acc.chunks_exact(d + 1)) {
            if a[d] > min_mass {
                for (v, sum) in c.iter_mut().zip(&a[..d]) {
                    *v = sum / a[d];
                }
            }
        }
        let mut ix = index.expect("built in the first step");
        ix.refit(&centroids, REBUILD_ITERS);
        self.om.centroids = centroids.clone();
        self.om.usage = usage;
        self.om.index = Some(ix);
        self.om.schedule = Some(sched);
        self.om.report = counts;
        self.report = report;
        (centroids, labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tiled step on the CPU backend with `α = 0` (state unchanged) labels
    /// a batch as an exact f64 two-level search does, split links included.
    /// Labels may differ only where two candidates (cells or centroids) tie to
    /// within f32 rounding.
    #[test]
    fn tiled_step_labels_match_exact_search() {
        check_tiled_step(3000, 12, 300, 8, 3, 5);
    }

    /// The same at the shapes of a large-k run: over 2048 units, so the
    /// `[U, 128, 64]` score tensor needs more than 65535 workgroups of 256.
    #[test]
    fn tiled_step_labels_match_exact_search_large() {
        check_tiled_step(40_000, 16, 8192, 32, 12, 6);
    }

    fn check_tiled_step(n: usize, d: usize, k: usize, nlist: usize, nprobe: usize, stride: usize) {
        let mut s = 7u64;
        let mut uniform = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        // Far from the origin, so the shifted frame matters.
        let x: Vec<f32> = (0..n * d).map(|_| 50.0 + 4.0 * uniform()).collect();
        let centroids = x[..k * d].to_vec();
        let mut index = CentroidIndex::build(&centroids, d, nlist, 4, 3);
        index.record_split(5, 9);
        index.record_split(5, 11);
        let idx: Vec<usize> = (k..n).step_by(stride).collect();
        let b = idx.len();

        let sq = |a: &[f32], c: &[f32]| -> f64 {
            a.iter()
                .zip(c)
                .map(|(u, v)| (*u as f64 - *v as f64).powi(2))
                .sum()
        };
        let exact =
            |i: usize, j: u32| sq(&x[i * d..(i + 1) * d], &centroids[j as usize * d..][..d]);
        // (label, distance, gap from the last probed cell to the next one).
        let reference: Vec<(u32, f64, f64)> = idx
            .iter()
            .map(|&i| {
                let xi = &x[i * d..(i + 1) * d];
                let mut cells: Vec<(f64, usize)> = (0..nlist)
                    .filter(|&c| index.offsets[c + 1] > index.offsets[c])
                    .map(|c| (sq(xi, &index.centres[c * d..(c + 1) * d]), c))
                    .collect();
                cells.sort_by(|a, b| a.0.total_cmp(&b.0));
                let gap = cells
                    .get(nprobe)
                    .map_or(f64::INFINITY, |next| next.0 - cells[nprobe - 1].0);
                let mut best = (u32::MAX, f64::INFINITY);
                for &(_, c) in &cells[..nprobe] {
                    for &j in &index.members[index.offsets[c]..index.offsets[c + 1]] {
                        let e = exact(i, j);
                        if e < best.1 {
                            best = (j, e);
                        }
                    }
                }
                let link = |j: u32| match index.child[j as usize] {
                    c if c < 0 => j,
                    c => c as u32,
                };
                let c1 = link(best.0);
                let c2 = link(c1);
                for c in [c1, c2] {
                    let e = exact(i, c);
                    if e < best.1 {
                        best = (c, e);
                    }
                }
                (best.0, best.1, gap)
            })
            .collect();

        let mut failed = Vec::new();
        for dev in rlx_runtime::available_devices() {
            let session = Session::new(dev);
            let options = omeans_options(dev);
            let shift = vec![50.0f32; d];
            let mut report = DeviceReport::default();
            let mut probes = Probes::default();
            probes.refresh(
                &session,
                &options,
                &x,
                d,
                &shift,
                &index,
                nprobe,
                &mut report,
            );
            let tiles = tile_batch(&idx, &probes.probes, nprobe, &index, k);
            let state = (
                centroids.clone(),
                vec![1.0; k],
                vec![0.0; k * (d + 1)],
                [0.0],
            );
            let shape = (tiles.units, tiles.m);
            let mut step = Step::compile(&session, &options, &x, b, d, k, shape, state);
            let pad = (tiles.units * TILE_ROWS) as f32;
            let cand: Vec<f32> = tiles
                .cand
                .iter()
                .map(|&v| if v == u32::MAX { pad } else { v as f32 })
                .collect();
            let rows: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
            let child: Vec<f32> = index
                .child
                .iter()
                .enumerate()
                .map(|(j, &c)| if c < 0 { j as f32 } else { c as f32 })
                .collect();
            let got = step
                .run(
                    &x,
                    &[
                        (io::ROWS, &rows),
                        (io::WEIGHT, &vec![1.0; b]),
                        (io::ALPHA, &[0.0]),
                        (io::OMEGA, &[1.0]),
                        (io::SHIFT, &shift),
                        (io::UNIT_ROWS, &tiles.unit_rows),
                        (io::UNIT_CENTROIDS, &tiles.unit_cents),
                        (io::CANDIDATES, &cand),
                        (io::CHILD, &child),
                    ],
                    true,
                )
                .expect("labels");
            let mut bad = 0;
            for ((&i, r), &g) in idx.iter().zip(&reference).zip(&got) {
                let g = g as u32;
                if g == r.0 {
                    continue;
                }
                // A different label is fine only on a near-tie: of the centroid
                // (same distance) or of the last probe (a different cell probed).
                let e = exact(i, g);
                let tie = (e - r.1).abs() <= 1e-4 * r.1 || r.2 <= 1e-3 * r.1.max(1.0);
                if !tie {
                    bad += 1;
                    if bad == 1 {
                        eprintln!(
                            "{dev:?}: row {i}: label {g} at {e}, exact search {} at {}",
                            r.0, r.1
                        );
                    }
                }
            }
            // Each row's candidates cover exactly its probed cells' centroids.
            let covered: u64 = (0..b)
                .map(|r| {
                    tiles.cand[r * tiles.m..(r + 1) * tiles.m]
                        .iter()
                        .filter(|&&v| v != u32::MAX)
                        .map(|&v| {
                            let u = v as usize / TILE_ROWS;
                            tiles.unit_cents[u * TILE_CENTROIDS..(u + 1) * TILE_CENTROIDS]
                                .iter()
                                .filter(|&&c| (c as usize) < k)
                                .count() as u64
                        })
                        .sum::<u64>()
                })
                .sum();
            assert_eq!(covered, tiles.evals);
            // The step folded the batch into the closing accumulator (ω = 1).
            let acc = step.read(io::ACC);
            let rows_counted: f32 = acc.chunks_exact(d + 1).map(|a| a[d]).sum();
            eprintln!(
                "{dev:?}: {} units, {bad} of {b} labels off, {rows_counted} rows counted, \
                 resident {}",
                tiles.units, step.resident
            );
            // The XDNA NPU has no f32 matmul: its bf16 kernel reorders cells and
            // centroids that tie to within bf16 rounding (the e-means matrix
            // gives it the same allowance).
            let allowed = if dev == Device::Xdna { b / 200 } else { 0 };
            if bad > allowed || rows_counted as usize != b {
                failed.push(dev);
            }
        }
        assert!(failed.is_empty(), "wrong on {failed:?}");
    }
}
