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

//! Bucketed decode SDPA: Lq=1, Lk=past_upper+1, `MaskKind::Custom` [B, Lk].

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};
use std::sync::{Mutex, MutexGuard};

/// Metal command queues and MPS caches are process-global; serialize these tests.
static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct MetalTestGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl MetalTestGuard {
    fn new() -> Self {
        Self(METAL_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for MetalTestGuard {
    fn drop(&mut self) {
        rlx_metal::device::drain_command_queue();
        rlx_metal::mps_blas::invalidate_caches();
    }
}

fn build_decode_attn(b: usize, lq: usize, lk: usize, nh: usize, dh: usize) -> Graph {
    let f = DType::F32;
    let hs = nh * dh;
    let mut g = Graph::new("decode_custom_mask_attn");
    let q = g.input("q", Shape::new(&[b, lq, hs], f));
    let k = g.input("k", Shape::new(&[b, lk, hs], f));
    let v = g.input("v", Shape::new(&[b, lk, hs], f));
    let mask = g.input("mask", Shape::new(&[b, lk], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, lq, hs], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn build_causal_decode_attn(b: usize, lq: usize, lk: usize, nh: usize, dh: usize) -> Graph {
    let f = DType::F32;
    let hs = nh * dh;
    let mut g = Graph::new("decode_causal_attn");
    let q = g.input("q", Shape::new(&[b, lq, hs], f));
    let k = g.input("k", Shape::new(&[b, lk, hs], f));
    let v = g.input("v", Shape::new(&[b, lk, hs], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, lq, hs], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn metal_bucketed_decode_custom_mask_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    // Talker-ish GQA decode: 16 Q heads, 8 KV heads, head_dim 128.
    let (b, lq, lk, nh, dh) = (1, 1, 33, 16, 128);
    let hs = nh * dh;
    let nq = b * lq * hs;
    let nk = b * lk * hs;

    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin()).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.013).cos()).collect();
    let v: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.011).sin()).collect();

    // past_seq=17, upper=32 → mask ones at [0..17) and index 32.
    let past_seq = 17usize;
    let upper = 32usize;
    let mut mask = vec![0f32; lk];
    for (i, slot) in mask.iter_mut().enumerate() {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }

    let graph = build_decode_attn(b, lq, lk, nh, dh);
    let inputs = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
        ("mask", mask.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(graph.clone());
    let metal_out = metal.run(&inputs).remove(0);

    let mut cpu = Session::new(Device::Cpu).compile(graph);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("bucketed decode attn metal vs cpu max_abs={d:.6}");
    eprintln!("cpu[:4]   = {:?}", &cpu_out[..4]);
    eprintln!("metal[:4] = {:?}", &metal_out[..4]);
    assert!(
        d < 1e-3,
        "bucketed decode custom-mask attention diverged (max_abs={d})"
    );

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Empty past + new KV row (decode first step after empty context).
#[test]
fn metal_empty_past_kv_concat_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, nkv, dh) = (1, 8, 128);
    let kv_dim = nkv * dh;
    let f = DType::F32;

    let mut g = Graph::new("empty_past_concat");
    let past_k = g.input("past_k", Shape::new(&[b, 0, kv_dim], f));
    let k_new = g.input("k_new", Shape::new(&[b, 1, kv_dim], f));
    let k_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_k, k_new],
        Shape::new(&[b, 1, kv_dim], f),
    );
    g.set_outputs(vec![k_cat]);

    let k_new_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let inputs = [("past_k", &[][..]), ("k_new", k_new_data.as_slice())];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("empty past kv concat max_abs={d:.6}");
    assert!(d < 1e-5, "empty past concat diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

#[test]
fn metal_decode_rope_single_row_hd128_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, seq, nh, dh) = (1, 1, 16, 128);
    let hs = nh * dh;
    let half = dh / 2;
    let f = DType::F32;

    let mut g = Graph::new("decode_rope_single_row");
    let x = g.input("x", Shape::new(&[b, seq, hs], f));
    let cos = g.input("cos", Shape::new(&[1, half], f));
    let sin = g.input("sin", Shape::new(&[1, half], f));
    let y = g.add_node(
        rlx_ir::Op::Rope {
            head_dim: dh,
            n_rot: dh,
            style: rlx_ir::RopeStyle::NeoX,
        },
        vec![x, cos, sin],
        Shape::new(&[b, seq, hs], f),
    );
    g.set_outputs(vec![y]);

    let n = b * seq * hs;
    let x_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();
    let cos_data: Vec<f32> = (0..half).map(|i| ((i as f32) * 0.31).cos()).collect();
    let sin_data: Vec<f32> = (0..half).map(|i| ((i as f32) * 0.29).sin()).collect();
    let inputs = [
        ("x", x_data.as_slice()),
        ("cos", cos_data.as_slice()),
        ("sin", sin_data.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("decode single-row rope hd128 max_abs={d:.6}");
    assert!(d < 1e-5, "decode single-row rope diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

#[test]
fn metal_causal_decode_attn_hd128_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, lq, lk, nh, dh) = (1, 1, 7, 16, 128);
    let hs = nh * dh;
    let nq = b * lq * hs;
    let nk = b * lk * hs;

    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin()).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.013).cos()).collect();
    let v: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.011).sin()).collect();

    let graph = build_causal_decode_attn(b, lq, lk, nh, dh);
    let inputs = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(graph.clone());
    let metal_out = metal.run(&inputs).remove(0);

    let mut cpu = Session::new(Device::Cpu).compile(graph);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("causal decode attn hd128 metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-3, "causal decode attention diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

#[test]
fn metal_concat_then_bucketed_attn_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, upper, lq, nh, dh) = (1, 32, 1, 16, 128);
    let lk = upper + 1;
    let hs = nh * dh;

    let mut g = Graph::new("concat_attn");
    let f = DType::F32;
    let past_k = g.input("past_k", Shape::new(&[b, upper, hs], f));
    let k_new = g.input("k_new", Shape::new(&[b, 1, hs], f));
    let past_v = g.input("past_v", Shape::new(&[b, upper, hs], f));
    let v_new = g.input("v_new", Shape::new(&[b, 1, hs], f));
    let q = g.input("q", Shape::new(&[b, lq, hs], f));
    let mask = g.input("mask", Shape::new(&[b, lk], f));

    let k_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_k, k_new],
        Shape::new(&[b, lk, hs], f),
    );
    let v_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_v, v_new],
        Shape::new(&[b, lk, hs], f),
    );
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nh,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k_cat, v_cat, mask],
        Shape::new(&[b, lq, hs], f),
    );
    g.set_outputs(vec![y]);

    let n_past = b * upper * hs;
    let n_new = b * hs;
    let past_k_data: Vec<f32> = (0..n_past).map(|i| (i as f32 * 1e-4).sin()).collect();
    let k_new_data: Vec<f32> = (0..n_new).map(|i| (i as f32 * 2e-4).cos()).collect();
    let past_v_data: Vec<f32> = (0..n_past).map(|i| (i as f32 * 3e-4).sin()).collect();
    let v_new_data: Vec<f32> = (0..n_new).map(|i| (i as f32 * 4e-4).cos()).collect();
    let q_data: Vec<f32> = (0..b * lq * hs).map(|i| (i as f32 * 5e-4).sin()).collect();
    let past_seq = 17usize;
    let mut mask = vec![0f32; lk];
    for (i, slot) in mask.iter_mut().enumerate() {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }

    let inputs = [
        ("past_k", past_k_data.as_slice()),
        ("k_new", k_new_data.as_slice()),
        ("past_v", past_v_data.as_slice()),
        ("v_new", v_new_data.as_slice()),
        ("q", q_data.as_slice()),
        ("mask", mask.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("concat+repeat_kv+attn metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-3, "concat bucketed attn diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Qwen3 `per_head_rms`: reshape → RMS → reshape (talker q/k norm path).
#[test]
fn metal_per_head_rms_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, seq, nh, dh) = (1, 1, 16, 128);
    let hs = nh * dh;
    let f = DType::F32;
    let mut g = Graph::new("per_head_rms");
    let x = g.input("x", Shape::new(&[b, seq, hs], f));
    let gamma = g.input("gamma", Shape::new(&[dh], f));
    let beta = g.input("beta", Shape::new(&[dh], f));
    let flat = g.reshape(x, vec![nh as i64, dh as i64], Shape::new(&[nh, dh], f));
    let normed = g.rms_norm(flat, gamma, beta, 1e-6);
    let out = g.reshape(
        normed,
        vec![b as i64, seq as i64, hs as i64],
        Shape::new(&[b, seq, hs], f),
    );
    g.set_outputs(vec![out]);

    let n = b * seq * hs;
    let x_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
    let gamma_data: Vec<f32> = (0..dh).map(|i| 1.0 + (i as f32) * 1e-4).collect();
    let beta_data = vec![0f32; dh];
    let inputs = [
        ("x", x_data.as_slice()),
        ("gamma", gamma_data.as_slice()),
        ("beta", beta_data.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("per_head_rms metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-4, "per_head_rms diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Single last-axis narrow on bucketed K tensor.
#[test]
fn metal_gqa_narrow_last_axis_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, lk, nkv, dh) = (1, 17, 8, 128);
    let kv_hs = nkv * dh;
    let mut g = Graph::new("narrow_kv");
    let f = DType::F32;
    let k = g.input("k", Shape::new(&[b, lk, kv_hs], f));
    let slice = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 2,
            start: 2 * dh,
            len: dh,
        },
        vec![k],
        Shape::new(&[b, lk, dh], f),
    );
    g.set_outputs(vec![slice]);

    let n = b * lk * kv_hs;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
    let inputs = [("k", data.as_slice())];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("narrow last axis metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-5, "narrow diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Concat two distinct narrow slices (no duplicate src segments).
#[test]
fn metal_gqa_concat_two_narrows_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, lk, nkv, dh) = (1, 17, 8, 128);
    let kv_hs = nkv * dh;
    let mut g = Graph::new("concat_two_narrows");
    let f = DType::F32;
    let k = g.input("k", Shape::new(&[b, lk, kv_hs], f));
    let s0 = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 2,
            start: 0,
            len: dh,
        },
        vec![k],
        Shape::new(&[b, lk, dh], f),
    );
    let s1 = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 2,
            start: dh,
            len: dh,
        },
        vec![k],
        Shape::new(&[b, lk, dh], f),
    );
    let out = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        vec![s0, s1],
        Shape::new(&[b, lk, 2 * dh], f),
    );
    g.set_outputs(vec![out]);

    let n = b * lk * kv_hs;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
    let inputs = [("k", data.as_slice())];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("concat two narrows metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-5, "concat two narrows diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Concat same narrow slice twice (repeat_kv pattern).
#[test]
fn metal_gqa_concat_duplicate_narrow_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, lk, nkv, dh) = (1, 17, 8, 128);
    let kv_hs = nkv * dh;
    let mut g = Graph::new("concat_dup_narrow");
    let f = DType::F32;
    let k = g.input("k", Shape::new(&[b, lk, kv_hs], f));
    let s0 = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 2,
            start: 0,
            len: dh,
        },
        vec![k],
        Shape::new(&[b, lk, dh], f),
    );
    let out = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        vec![s0, s0],
        Shape::new(&[b, lk, 2 * dh], f),
    );
    g.set_outputs(vec![out]);

    let n = b * lk * kv_hs;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
    let inputs = [("k", data.as_slice())];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("concat duplicate narrow metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-5, "concat duplicate narrow diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

fn build_repeat_kv_concat_graph(
    b: usize,
    lk: usize,
    nkv: usize,
    dh: usize,
    group: usize,
) -> (Graph, usize) {
    let kv_hs = nkv * dh;
    let q_hs = nkv * group * dh;
    let mut g = Graph::new("repeat_kv_concat");
    let f = DType::F32;
    let k_cat = g.input("k_cat", Shape::new(&[b, lk, kv_hs], f));
    let mut k_pieces = Vec::new();
    for h in 0..nkv {
        let off = h * dh;
        let k_slice = g.add_node(
            rlx_ir::Op::Narrow {
                axis: 2,
                start: off,
                len: dh,
            },
            vec![k_cat],
            Shape::new(&[b, lk, dh], f),
        );
        for _ in 0..group {
            k_pieces.push(k_slice);
        }
    }
    let k_rep = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        k_pieces,
        Shape::new(&[b, lk, q_hs], f),
    );
    g.set_outputs(vec![k_rep]);
    (g, b * lk * kv_hs)
}

/// `repeat_kv` narrow+concat only (no SDPA).
#[test]
fn metal_gqa_repeat_kv_concat_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, upper, nq, nkv, dh) = (1, 16, 16, 8, 128);
    let lk = upper + 1;
    let group = nq / nkv;

    for nkv_try in [1usize, 2, 4, 8] {
        let (g, n) = build_repeat_kv_concat_graph(b, lk, nkv_try, dh, group);
        let k_data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
        let inputs = [("k_cat", k_data.as_slice())];
        let mut metal = Session::new(Device::Metal).compile(g.clone());
        let metal_out = metal.run(&inputs).remove(0);
        let mut cpu = Session::new(Device::Cpu).compile(g);
        let cpu_out = cpu.run(&inputs).remove(0);
        let d = max_abs(&cpu_out, &metal_out);
        eprintln!(
            "repeat_kv concat nkv={nkv_try} segs={} max_abs={d:.6}",
            nkv_try * group
        );
        if d >= 1e-5 {
            eprintln!(
                "  cpu[:4]={:?} metal[:4]={:?}",
                &cpu_out[..4],
                &metal_out[..4]
            );
        }
    }

    let (g, n) = build_repeat_kv_concat_graph(b, lk, nkv, dh, group);
    let k_data: Vec<f32> = (0..n).map(|i| (i as f32 * 1e-4).sin()).collect();
    let inputs = [("k_cat", k_data.as_slice())];
    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);
    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("gqa repeat_kv concat talker nkv=8 max_abs={d:.6}");
    assert!(d < 1e-5, "repeat_kv concat diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Bucketed K concat on axis 1 with GQA `kv_hs` (talker-shaped).
#[test]
fn metal_gqa_kv_concat_axis1_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, upper, nkv, dh) = (1, 16, 8, 128);
    let lk = upper + 1;
    let kv_hs = nkv * dh;
    let mut g = Graph::new("gqa_kv_concat");
    let f = DType::F32;
    let past_k = g.input("past_k", Shape::new(&[b, upper, kv_hs], f));
    let k_new = g.input("k_new", Shape::new(&[b, 1, kv_hs], f));
    let k_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_k, k_new],
        Shape::new(&[b, lk, kv_hs], f),
    );
    g.set_outputs(vec![k_cat]);

    let n_past = b * upper * kv_hs;
    let past: Vec<f32> = (0..n_past).map(|i| (i as f32 * 1e-4).sin()).collect();
    let new_k: Vec<f32> = (0..b * kv_hs).map(|i| (i as f32 * 2e-4).cos()).collect();
    let inputs = [("past_k", past.as_slice()), ("k_new", new_k.as_slice())];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("gqa kv axis1 concat metal vs cpu max_abs={d:.6}");
    assert!(d < 1e-5, "kv axis1 concat diverged (max_abs={d})");

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}

/// Talker GQA: K/V are `n_kv * head_dim`, `repeat_kv` expands to `n_q * head_dim`.
#[test]
fn metal_gqa_repeat_kv_bucketed_attn_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let _guard = MetalTestGuard::new();
    rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");

    let (b, upper, lq, nq, nkv, dh) = (1, 16, 1, 16, 8, 128);
    let lk = upper + 1;
    let q_hs = nq * dh;
    let kv_hs = nkv * dh;
    let group = nq / nkv;

    let mut g = Graph::new("gqa_repeat_kv_attn");
    let f = DType::F32;
    let past_k = g.input("past_k", Shape::new(&[b, upper, kv_hs], f));
    let k_new = g.input("k_new", Shape::new(&[b, 1, kv_hs], f));
    let past_v = g.input("past_v", Shape::new(&[b, upper, kv_hs], f));
    let v_new = g.input("v_new", Shape::new(&[b, 1, kv_hs], f));
    let q = g.input("q", Shape::new(&[b, lq, q_hs], f));
    let mask = g.input("mask", Shape::new(&[b, lk], f));

    let k_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_k, k_new],
        Shape::new(&[b, lk, kv_hs], f),
    );
    let v_cat = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![past_v, v_new],
        Shape::new(&[b, lk, kv_hs], f),
    );

    let mut k_pieces = Vec::new();
    let mut v_pieces = Vec::new();
    for h in 0..nkv {
        let off = h * dh;
        let k_slice = g.add_node(
            rlx_ir::Op::Narrow {
                axis: 2,
                start: off,
                len: dh,
            },
            vec![k_cat],
            Shape::new(&[b, lk, dh], f),
        );
        let v_slice = g.add_node(
            rlx_ir::Op::Narrow {
                axis: 2,
                start: off,
                len: dh,
            },
            vec![v_cat],
            Shape::new(&[b, lk, dh], f),
        );
        for _ in 0..group {
            k_pieces.push(k_slice);
            v_pieces.push(v_slice);
        }
    }
    let k_rep = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        k_pieces,
        Shape::new(&[b, lk, q_hs], f),
    );
    let v_rep = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        v_pieces,
        Shape::new(&[b, lk, q_hs], f),
    );

    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: nq,
            head_dim: dh,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k_rep, v_rep, mask],
        Shape::new(&[b, lq, q_hs], f),
    );
    g.set_outputs(vec![y]);

    let n_past = b * upper * kv_hs;
    let n_new = b * kv_hs;
    let past_k_data: Vec<f32> = (0..n_past).map(|i| (i as f32 * 1e-4).sin()).collect();
    let k_new_data: Vec<f32> = (0..n_new).map(|i| (i as f32 * 2e-4).cos()).collect();
    let past_v_data: Vec<f32> = (0..n_past).map(|i| (i as f32 * 3e-4).sin()).collect();
    let v_new_data: Vec<f32> = (0..n_new).map(|i| (i as f32 * 4e-4).cos()).collect();
    let q_data: Vec<f32> = (0..b * lq * q_hs)
        .map(|i| (i as f32 * 5e-4).sin())
        .collect();
    let past_seq = 13usize;
    let mut mask_data = vec![0f32; lk];
    for (i, slot) in mask_data.iter_mut().enumerate() {
        *slot = if i < past_seq || i == upper { 1.0 } else { 0.0 };
    }

    let inputs = [
        ("past_k", past_k_data.as_slice()),
        ("k_new", k_new_data.as_slice()),
        ("past_v", past_v_data.as_slice()),
        ("v_new", v_new_data.as_slice()),
        ("q", q_data.as_slice()),
        ("mask", mask_data.as_slice()),
    ];

    let mut metal = Session::new(Device::Metal).compile(g.clone());
    let metal_out = metal.run(&inputs).remove(0);
    let mut cpu = Session::new(Device::Cpu).compile(g);
    let cpu_out = cpu.run(&inputs).remove(0);

    let d = max_abs(&cpu_out, &metal_out);
    eprintln!("gqa repeat_kv bucketed attn metal vs cpu max_abs={d:.6}");
    assert!(
        d < 1e-3,
        "gqa repeat_kv bucketed attn diverged (max_abs={d})"
    );

    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
}
