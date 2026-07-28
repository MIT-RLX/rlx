// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Optional: real mlx-community checkpoint → one packed Linear on Metal.
//!
//! - Always runs when `$RLX_HF_CACHE` (or `~/.cache/rlx/hf`) already has the
//!   repo (`config.json` / `.rlx_fetch_ok`).
//! - Downloads only when `RLX_HF_MLX=1` and the cache is cold.
//! - Skips cleanly when offline with an empty cache.

#![cfg(target_os = "macos")]

use rlx_mlx_io::{
    DEFAULT_HF_MLX_REPO, build_parallel_dequant_graph, collect_packed_linears,
    fetch_default_mlx_community, fetch_ok, hf_cache_dir, load_path, param_bindings_for,
    write_fetch_ok,
};
use rlx_runtime::{Device, Session};

#[test]
fn metal_hf_mlx_one_linear() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let repo = std::env::var("RLX_HF_MLX_REPO").unwrap_or_else(|_| DEFAULT_HF_MLX_REPO.into());
    let cached = hf_cache_dir().join(repo.replace('/', "--"));
    let allow_download = std::env::var("RLX_HF_MLX").ok().as_deref() == Some("1");
    let warm = fetch_ok(&cached);

    if !warm && !allow_download {
        eprintln!(
            "skip metal_hf_mlx_one_linear: no cache at {} — set RLX_HF_MLX=1 to download",
            cached.display()
        );
        return;
    }

    let dir = if warm && !allow_download {
        // Prefer cache-only: avoid network when CI mounts a warm cache.
        cached
    } else {
        match fetch_default_mlx_community() {
            Ok(d) => {
                let _ = write_fetch_ok(&d);
                d
            }
            Err(e) => {
                if warm {
                    eprintln!("fetch refresh failed ({e:#}); using existing cache");
                    cached
                } else {
                    eprintln!("skip metal_hf_mlx_one_linear: fetch failed: {e:#}");
                    return;
                }
            }
        }
    };

    let mut w = match load_path(&dir) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("skip: load failed: {e:#}");
            return;
        }
    };
    let mut linears = match collect_packed_linears(&mut w) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            eprintln!("skip: no packed linears");
            return;
        }
        Err(e) => {
            eprintln!("skip: collect failed: {e:#}");
            return;
        }
    };
    linears.sort_by_key(|b| b.packed.w_q.len());
    let one = vec![linears.remove(0)];
    let g = build_parallel_dequant_graph("hf_mlx_one", &one, 2).unwrap();
    let k = one[0].packed.out_shape[1];
    let x: Vec<f32> = (0..2 * k).map(|i| ((i as f32) * 0.017).sin()).collect();

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        for b in &one {
            for (name, bytes, dt) in param_bindings_for(b) {
                c.set_param_typed(&name, &bytes, dt);
            }
        }
        let in_name = format!(
            "{}_x",
            one[0]
                .name
                .chars()
                .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
                .collect::<String>()
        );
        c.run(&[(&in_name, x.as_slice())]).remove(0)
    };
    let cpu = run(Device::Cpu);
    let metal = run(Device::Metal);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 5e-3,
        "HF mlx Metal vs CPU max_abs={max_abs} layer={}",
        one[0].name
    );
}
