// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// REVE-shaped transformer block: patch embed + one encoder layer.
// Catches MPSGraph vs thunk drift (see reve-rs layer-1 bisect).

#![cfg(target_os = "macos")]

use rlx_ir::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const KEY_ZEROS: &str = "__reve.zeros_embed";
const KEY_SCALE: &str = "__reve.attn_head_scale";

fn randn(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Patch embed + one REVE transformer block (manual attention, GEGLU).
fn build_patch_plus_one_block() -> Graph {
    let (b, s, patch, d, heads, dh, mlp) = (1usize, 176, 200, 512, 8, 64, 1362);
    let inner = heads * dh;
    let mut g = Graph::new("reve_block");

    let zeros = g.param(KEY_ZEROS, Shape::new(&[d], DType::F32));
    let scale = g.param(KEY_SCALE, Shape::new(&[1], DType::F32));

    let patches = g.input("patches", Shape::new(&[b, s, patch], DType::F32));
    let pos = g.input("pos_embed", Shape::new(&[b, s, d], DType::F32));
    let pe_w = g.param(
        "to_patch_embedding.0.weight",
        Shape::new(&[patch, d], DType::F32),
    );
    let pe_b = g.param("to_patch_embedding.0.bias", Shape::new(&[d], DType::F32));
    let x0 = g.mm(patches, pe_w);
    let patch_emb = g.add(x0, pe_b);
    let mut h = g.add(patch_emb, pos);

    // Layer 0
    let an_g = g.param(
        "transformer.layers.0.0.norm.weight",
        Shape::new(&[d], DType::F32),
    );
    let xn = g.rms_norm(h, an_g, zeros, 1e-6);
    let wq = g.param(
        "transformer.layers.0.0.to_q.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wk = g.param(
        "transformer.layers.0.0.to_k.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wv = g.param(
        "transformer.layers.0.0.to_v.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wo = g.param(
        "transformer.layers.0.0.to_out.weight",
        Shape::new(&[inner, d], DType::F32),
    );
    let q = g.mm(xn, wq);
    let k = g.mm(xn, wk);
    let v = g.mm(xn, wv);
    let bh = (b * heads) as i64;
    let q4 = g.reshape_(q, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let q_bhsd = g.transpose_(q4, vec![0, 2, 1, 3]);
    let k_bhsd = g.transpose_(k4, vec![0, 2, 1, 3]);
    let v_bhsd = g.transpose_(v4, vec![0, 2, 1, 3]);
    let q3 = g.reshape_(q_bhsd, vec![bh, s as i64, dh as i64]);
    let k3 = g.reshape_(k_bhsd, vec![bh, s as i64, dh as i64]);
    let v3 = g.reshape_(v_bhsd, vec![bh, s as i64, dh as i64]);
    let k_t = g.transpose_(k3, vec![0, 2, 1]);
    let scores0 = g.mm(q3, k_t);
    let scores = g.mul(scores0, scale);
    let w = g.sm(scores, 2);
    let attn = g.mm(w, v3);
    let attn_bhsd = g.reshape_(attn, vec![b as i64, heads as i64, s as i64, dh as i64]);
    let attn_bshd = g.transpose_(attn_bhsd, vec![0, 2, 1, 3]);
    let attn_flat = g.reshape_(attn_bshd, vec![b as i64, s as i64, inner as i64]);
    let attn_out = g.mm(attn_flat, wo);
    h = g.add(h, attn_out);

    let fn_g = g.param(
        "transformer.layers.0.1.net.0.weight",
        Shape::new(&[d], DType::F32),
    );
    let hn = g.rms_norm(h, fn_g, zeros, 1e-6);
    let wg = g.param(
        "transformer.layers.0.1.net.1.w_gate.weight",
        Shape::new(&[d, mlp], DType::F32),
    );
    let wu = g.param(
        "transformer.layers.0.1.net.1.w_up.weight",
        Shape::new(&[d, mlp], DType::F32),
    );
    let w2 = g.param(
        "transformer.layers.0.1.net.3.weight",
        Shape::new(&[mlp, d], DType::F32),
    );
    let gates = g.mm(hn, wg);
    let up = g.mm(hn, wu);
    let g_act = g.gelu(gates);
    let h_geglu = g.mul(g_act, up);
    let ff = g.mm(h_geglu, w2);
    h = g.add(h, ff);

    g.set_outputs(vec![h]);
    g
}

fn run_block(
    dev: Device,
    g: Graph,
    params: &[(&str, &[f32])],
    inputs: &[(&str, &[f32])],
) -> Vec<f32> {
    let mut c = Session::new(dev).compile(g);
    for (n, d) in params {
        c.set_param(n, d);
    }
    c.run(inputs).remove(0)
}

#[test]
fn metal_patch_embed_mpsgraph_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");
    let (b, s, patch, d) = (1usize, 176, 200, 512);
    let mut g = Graph::new("patch");
    let patches = g.input("patches", Shape::new(&[b, s, patch], DType::F32));
    let pos = g.input("pos_embed", Shape::new(&[b, s, d], DType::F32));
    let pe_w = g.param(
        "to_patch_embedding.0.weight",
        Shape::new(&[patch, d], DType::F32),
    );
    let pe_b = g.param("to_patch_embedding.0.bias", Shape::new(&[d], DType::F32));
    let x0 = g.mm(patches, pe_w);
    let patch_emb = g.add(x0, pe_b);
    let h = g.add(patch_emb, pos);
    g.set_outputs(vec![h]);

    let patches_in = vec![0f32; b * s * patch];
    let pos_in = vec![0f32; b * s * d];
    let pe_w = randn(patch * d, 1);
    let pe_b = randn(d, 2);
    let params = [
        ("to_patch_embedding.0.weight", pe_w.as_slice()),
        ("to_patch_embedding.0.bias", pe_b.as_slice()),
    ];
    let inputs = [
        ("patches", patches_in.as_slice()),
        ("pos_embed", pos_in.as_slice()),
    ];
    let cpu = run_block(Device::Cpu, g.clone(), &params, &inputs);
    let metal = run_block(Device::Metal, g, &params, &inputs);
    let drift = max_abs(&cpu, &metal);
    eprintln!("patch embed mpsgraph max_abs={drift:.6}");
    assert!(drift < 1e-5, "patch embed drift {drift}");
}

#[test]
fn metal_reve_attn_subgraph_mpsgraph_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");

    let (b, s, d, heads, dh, inner) = (1usize, 176, 512, 8, 64, 512);
    let bh = (b * heads) as i64;
    let mut g = Graph::new("attn_sub");
    let zeros = g.param(KEY_ZEROS, Shape::new(&[d], DType::F32));
    let scale = g.param(KEY_SCALE, Shape::new(&[1], DType::F32));
    let h = g.input("h", Shape::new(&[b, s, d], DType::F32));
    let an_g = g.param(
        "transformer.layers.0.0.norm.weight",
        Shape::new(&[d], DType::F32),
    );
    let xn = g.rms_norm(h, an_g, zeros, 1e-6);
    let wq = g.param(
        "transformer.layers.0.0.to_q.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wk = g.param(
        "transformer.layers.0.0.to_k.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wv = g.param(
        "transformer.layers.0.0.to_v.weight",
        Shape::new(&[d, inner], DType::F32),
    );
    let wo = g.param(
        "transformer.layers.0.0.to_out.weight",
        Shape::new(&[inner, d], DType::F32),
    );
    let q = g.mm(xn, wq);
    let k = g.mm(xn, wk);
    let v = g.mm(xn, wv);
    let q4 = g.reshape_(q, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, heads as i64, dh as i64]);
    let q_bhsd = g.transpose_(q4, vec![0, 2, 1, 3]);
    let k_bhsd = g.transpose_(k4, vec![0, 2, 1, 3]);
    let v_bhsd = g.transpose_(v4, vec![0, 2, 1, 3]);
    let q3 = g.reshape_(q_bhsd, vec![bh, s as i64, dh as i64]);
    let k3 = g.reshape_(k_bhsd, vec![bh, s as i64, dh as i64]);
    let v3 = g.reshape_(v_bhsd, vec![bh, s as i64, dh as i64]);
    let k_t = g.transpose_(k3, vec![0, 2, 1]);
    let scores0 = g.mm(q3, k_t);
    let scores = g.mul(scores0, scale);
    let w = g.sm(scores, 2);
    let attn = g.mm(w, v3);
    let attn_bhsd = g.reshape_(attn, vec![b as i64, heads as i64, s as i64, dh as i64]);
    let attn_bshd = g.transpose_(attn_bhsd, vec![0, 2, 1, 3]);
    let attn_flat = g.reshape_(attn_bshd, vec![b as i64, s as i64, inner as i64]);
    let attn_out = g.mm(attn_flat, wo);
    let y = g.add(h, attn_out);
    g.set_outputs(vec![y]);

    let h_in = randn(b * s * d, 20);
    let zeros = vec![0f32; d];
    let scale = vec![(64f32).powf(-0.5)];
    let an_g = randn(d, 3);
    let wq = randn(d * inner, 4);
    let wk = randn(d * inner, 5);
    let wv = randn(d * inner, 6);
    let wo = randn(inner * d, 7);
    let params = [
        (KEY_ZEROS, zeros.as_slice()),
        (KEY_SCALE, scale.as_slice()),
        ("transformer.layers.0.0.norm.weight", an_g.as_slice()),
        ("transformer.layers.0.0.to_q.weight", wq.as_slice()),
        ("transformer.layers.0.0.to_k.weight", wk.as_slice()),
        ("transformer.layers.0.0.to_v.weight", wv.as_slice()),
        ("transformer.layers.0.0.to_out.weight", wo.as_slice()),
    ];
    let inputs = [("h", h_in.as_slice())];
    let cpu = run_block(Device::Cpu, g.clone(), &params, &inputs);
    let metal = run_block(Device::Metal, g, &params, &inputs);
    let drift = max_abs(&cpu, &metal);
    eprintln!("attn subgraph mpsgraph max_abs={drift:.6}");
    assert!(drift < 1e-2, "attn subgraph drift {drift}");
}

#[test]
fn metal_reve_ffn_subgraph_mpsgraph_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");

    let (b, s, d, mlp) = (1usize, 176, 512, 1362);
    let mut g = Graph::new("ffn_sub");
    let zeros = g.param(KEY_ZEROS, Shape::new(&[d], DType::F32));
    let h = g.input("h", Shape::new(&[b, s, d], DType::F32));
    let fn_g = g.param(
        "transformer.layers.0.1.net.0.weight",
        Shape::new(&[d], DType::F32),
    );
    let hn = g.rms_norm(h, fn_g, zeros, 1e-6);
    let wg = g.param(
        "transformer.layers.0.1.net.1.w_gate.weight",
        Shape::new(&[d, mlp], DType::F32),
    );
    let wu = g.param(
        "transformer.layers.0.1.net.1.w_up.weight",
        Shape::new(&[d, mlp], DType::F32),
    );
    let w2 = g.param(
        "transformer.layers.0.1.net.3.weight",
        Shape::new(&[mlp, d], DType::F32),
    );
    let gates = g.mm(hn, wg);
    let up = g.mm(hn, wu);
    let g_act = g.gelu(gates);
    let h_geglu = g.mul(g_act, up);
    let ff = g.mm(h_geglu, w2);
    let y = g.add(h, ff);
    g.set_outputs(vec![y]);

    let h_in = randn(b * s * d, 30);
    let zeros = vec![0f32; d];
    let fn_g = randn(d, 8);
    let wg = randn(d * mlp, 9);
    let wu = randn(d * mlp, 10);
    let w2 = randn(mlp * d, 11);
    let params = [
        (KEY_ZEROS, zeros.as_slice()),
        ("transformer.layers.0.1.net.0.weight", fn_g.as_slice()),
        ("transformer.layers.0.1.net.1.w_gate.weight", wg.as_slice()),
        ("transformer.layers.0.1.net.1.w_up.weight", wu.as_slice()),
        ("transformer.layers.0.1.net.3.weight", w2.as_slice()),
    ];
    let inputs = [("h", h_in.as_slice())];
    let cpu = run_block(Device::Cpu, g.clone(), &params, &inputs);
    let metal = run_block(Device::Metal, g, &params, &inputs);
    let drift = max_abs(&cpu, &metal);
    eprintln!("ffn subgraph mpsgraph max_abs={drift:.6}");
    assert!(drift < 1e-2, "ffn subgraph drift {drift}");
}

#[test]
fn metal_geglu_core_mpsgraph_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");

    let (b, s, d, mlp) = (1usize, 176, 512, 1362);
    let mut g = Graph::new("geglu_core");
    let x = g.input("x", Shape::new(&[b, s, d], DType::F32));
    let wg = g.param("wg", Shape::new(&[d, mlp], DType::F32));
    let wu = g.param("wu", Shape::new(&[d, mlp], DType::F32));
    let gates = g.mm(x, wg);
    let up = g.mm(x, wu);
    let g_act = g.gelu(gates);
    let y = g.mul(g_act, up);
    g.set_outputs(vec![y]);

    let x_in = randn(b * s * d, 40);
    let wg = randn(d * mlp, 41);
    let wu = randn(d * mlp, 42);
    let params = [("wg", wg.as_slice()), ("wu", wu.as_slice())];
    let inputs = [("x", x_in.as_slice())];
    let cpu = run_block(Device::Cpu, g.clone(), &params, &inputs);
    let metal = run_block(Device::Metal, g, &params, &inputs);
    let drift = max_abs(&cpu, &metal);
    eprintln!("geglu core mpsgraph max_abs={drift:.6}");
    assert!(drift < 5e-3, "geglu core drift {drift}");
}

#[test]
fn metal_reve_block_mpsgraph_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    rlx_ir::env::unset("RLX_DISABLE_MPSGRAPH");
    rlx_ir::env::set("RLX_MPSGRAPH_FORCE", "1");

    let g = build_patch_plus_one_block();
    let (b, s, patch, d, inner, mlp) = (1usize, 176, 200, 512, 512, 1362);

    let patches = vec![0f32; b * s * patch];
    let pos = vec![0f32; b * s * d];
    let zeros = vec![0f32; d];
    let scale = vec![(dh_f()).powf(-0.5)];
    let pe_w = randn(patch * d, 1);
    let pe_b = randn(d, 2);
    let an_g = randn(d, 3);
    let wq = randn(d * inner, 4);
    let wk = randn(d * inner, 5);
    let wv = randn(d * inner, 6);
    let wo = randn(inner * d, 7);
    let fn_g = randn(d, 8);
    let wg = randn(d * mlp, 9);
    let wu = randn(d * mlp, 10);
    let w2 = randn(mlp * d, 11);

    let params: Vec<(&str, &[f32])> = vec![
        (KEY_ZEROS, &zeros),
        (KEY_SCALE, &scale),
        ("to_patch_embedding.0.weight", &pe_w),
        ("to_patch_embedding.0.bias", &pe_b),
        ("transformer.layers.0.0.norm.weight", &an_g),
        ("transformer.layers.0.0.to_q.weight", &wq),
        ("transformer.layers.0.0.to_k.weight", &wk),
        ("transformer.layers.0.0.to_v.weight", &wv),
        ("transformer.layers.0.0.to_out.weight", &wo),
        ("transformer.layers.0.1.net.0.weight", &fn_g),
        ("transformer.layers.0.1.net.1.w_gate.weight", &wg),
        ("transformer.layers.0.1.net.1.w_up.weight", &wu),
        ("transformer.layers.0.1.net.3.weight", &w2),
    ];
    let inputs: Vec<(&str, &[f32])> = vec![("patches", &patches), ("pos_embed", &pos)];

    let cpu = run_block(Device::Cpu, g.clone(), &params, &inputs);
    let metal = run_block(Device::Metal, g, &params, &inputs);
    let drift = max_abs(&cpu, &metal);
    eprintln!("reve block mpsgraph max_abs={drift:.6}");
    assert!(drift < 5e-3, "MPSGraph drift {drift}");
}

fn dh_f() -> f32 {
    64.0
}
