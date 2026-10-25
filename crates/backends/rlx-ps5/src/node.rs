// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The PS5 as an RLX dist node: joins the mesh and serves whatever stage the
//! coordinator ships, on the console's CPU.
//!
//! A payload starts with no arguments or environment, so the settings come
//! from `/data/rlx-ps5/node.conf`: `KEY=VALUE` lines with the same keys as
//! `NodeConfig::from_env` (`RANK`, `WORLD`, `PEERS`, `TOPOLOGY`, `DISCOVER`,
//! `DISCOVER_HOST`, …); a variable already in the environment wins, so the
//! same binary runs on a host for checks. For example, as rank 1 of 2 next to
//! `cargo run -p rlx-ffi --example node_coordinator -- --world 2 --peers …`:
//!
//! ```text
//! RANK=1
//! WORLD=2
//! PEERS=192.168.1.10:29500,192.168.1.50:29501
//! ```
//!
//! The outcome goes to `/data/rlx-ps5/node.txt` as well as stdout.

use rlx_runtime::dist::node::{NodeConfig, serve_worker};

const CONF: &str = "/data/rlx-ps5/node.conf";

fn main() {
    let conf = std::fs::read_to_string(CONF).unwrap_or_default();
    for line in conf
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        if let Some((k, v)) = line.split_once('=')
            && rlx_ir::env::read_os(k.trim()).is_none()
        {
            // SAFETY: single-threaded here — nothing has spawned a thread yet.
            unsafe { std::env::set_var(k.trim(), v.trim()) };
        }
    }
    // The checks above cached these keys as unset; read them anew.
    rlx_ir::env::reload();
    let workers = rlx_ps5::init();
    let msg = run(workers);
    println!("{msg}");
    let _ = std::fs::create_dir_all("/data/rlx-ps5");
    let _ = std::fs::write("/data/rlx-ps5/node.txt", format!("{msg}\n"));
    rlx_ps5::exit(i32::from(!msg.starts_with("ok")));
}

fn run(workers: usize) -> String {
    if std::env::var_os("PEERS").is_none() && std::env::var_os("DISCOVER").is_none() {
        return format!(
            "error: no peers — put RANK, WORLD and PEERS (or DISCOVER=1 + DISCOVER_HOST) in {CONF}"
        );
    }
    let cfg = match NodeConfig::from_env() {
        Ok(c) => c,
        Err(e) => return format!("error: config: {e}"),
    };
    println!(
        "rlx-ps5 node: rank {}/{} {:?}, {workers} CPU worker(s), joining…",
        cfg.rank, cfg.world, cfg.topology
    );
    let group = match cfg.connect() {
        Ok(g) => g,
        Err(e) => return format!("error: connect: {e}"),
    };
    match serve_worker(&group, |_uri| Vec::new()) {
        Ok(r) => format!(
            "ok: rank {} on {} ({}), {} activation(s)",
            r.rank,
            r.device.name(),
            r.platform,
            r.activations
        ),
        Err(e) => format!("error: {e}"),
    }
}
