// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Config-driven micro-bench harness (plan #13).
//!
//! One bench binary that reads every `bench-configs/*.toml` at
//! startup and emits a criterion group per file. Each `[[shape]]`
//! row in a config becomes one timing in that group.
//!
//! Adding shapes: edit a TOML.
//! Adding kernels: edit the `match cfg.kernel { ... }` below
//! plus drop a new TOML.
//!
//! Borrowed from MAX's `benchmarks/autotune/test.yaml` +
//! `kbench.py` pattern. The Rust spelling uses TOML (no extra
//! dep beyond what's already in the workspace) + criterion
//! programmatically.

use criterion::{Criterion, black_box};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchFile {
    kernel: String,
    shape: Vec<Shape>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    label: String,
    // sgemm fields
    #[serde(default)]
    m: Option<usize>,
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    n: Option<usize>,
    // cumsum / general 2D fields
    #[serde(default)]
    rows: Option<usize>,
    #[serde(default)]
    cols: Option<usize>,
}

fn configs_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir).join("bench-configs")
}

fn load_configs() -> Vec<(String, BenchFile)> {
    let dir = configs_dir();
    let mut out = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|_| panic!("bench-configs dir not found: {}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).expect("read toml");
        let parsed: BenchFile =
            toml::from_str(&raw).unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()));
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        out.push((name, parsed));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn random_buf(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = rlx_ir::Philox4x32::new(seed);
    let mut v = vec![0f32; len];
    rng.fill_normal(&mut v);
    v
}

fn run_sgemm_group(c: &mut Criterion, group_name: &str, cfg: &BenchFile) {
    let mut g = c.benchmark_group(group_name);
    for s in &cfg.shape {
        let m = s.m.expect("sgemm shape missing `m`");
        let k = s.k.expect("sgemm shape missing `k`");
        let n = s.n.expect("sgemm shape missing `n`");
        let a = random_buf(m * k, 1);
        let b = random_buf(k * n, 2);
        let mut c_buf = vec![0f32; m * n];
        g.bench_function(&s.label, |bencher| {
            bencher.iter(|| {
                rlx_cpu::blas::sgemm(black_box(&a), black_box(&b), black_box(&mut c_buf), m, k, n);
            });
        });
    }
    g.finish();
}

fn run_cumsum_group(c: &mut Criterion, group_name: &str, cfg: &BenchFile) {
    use rlx_cpu::arena::Arena;
    use rlx_cpu::thunk::{compile_thunks, execute_thunks};
    use rlx_ir::{DType, Graph, Shape as IrShape};

    let mut g = c.benchmark_group(group_name);
    for s in &cfg.shape {
        let rows = s.rows.expect("cumsum shape missing `rows`");
        let cols = s.cols.expect("cumsum shape missing `cols`");
        let f = DType::F32;
        let mut graph = Graph::new("tuned-cumsum");
        let x = graph.input("x", IrShape::new(&[rows, cols], f));
        let cs = graph.cumsum(x, -1, false, IrShape::new(&[rows, cols], f));
        graph.set_outputs(vec![cs]);
        let plan = rlx_opt::memory::plan_memory(&graph);
        let mut arena = Arena::from_plan(plan);
        let sched = compile_thunks(&graph, &arena);
        // Seed input.
        let in_off = arena.byte_offset(x);
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(in_off) as *mut f32;
            let mut rng = rlx_ir::Philox4x32::new(11);
            for i in 0..(rows * cols) {
                *p.add(i) = rng.next_f32();
            }
        }
        g.bench_function(&s.label, |bencher| {
            bencher.iter(|| {
                execute_thunks(black_box(&sched), arena.raw_buf_mut());
            });
        });
    }
    g.finish();
}

fn run_all(c: &mut Criterion) {
    let configs = load_configs();
    if configs.is_empty() {
        eprintln!("[tuned] no bench-configs/*.toml found; nothing to run");
        return;
    }
    for (name, cfg) in &configs {
        let group = format!("tuned/{name}");
        match cfg.kernel.as_str() {
            "sgemm" => run_sgemm_group(c, &group, cfg),
            "cumsum" => run_cumsum_group(c, &group, cfg),
            other => eprintln!("[tuned] unknown kernel `{other}` in {name}.toml; skipping"),
        }
    }
}

fn main() {
    // Manually drive criterion (mirrors what criterion_main!
    // generates) so we don't need a static benches list.
    let mut criterion = Criterion::default().configure_from_args();
    run_all(&mut criterion);
    criterion.final_summary();
}
