// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! The CPU attention-block fusion (`Thunk::FusedAttnBlock`) reimplements RoPE
//! inline. It must honor the RoPE *pairing style* (NeoX rotate-half vs GPT-J
//! interleaved) captured from the fused `Op::Rope`, otherwise a GPT-J model
//! would be silently rotated NeoX-style by the fused path. This is a standalone
//! integration test (its own process) so it can toggle `RLX_FUSE_ATTN_THRESHOLD`
//! to compare the fused kernel against the unfused base RoPE kernel without
//! racing the lib tests.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{Thunk, compile_thunks, execute_thunks};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, Graph, Op, Shape};

const S: usize = 5;
const D: usize = 8;
const NH: usize = 2;
const DH: usize = 4;

/// QKV proj → narrow×3 → RoPE(style)×2 → causal attention → out proj.
/// The single QKV matmul + narrows are what trigger the `FusedAttnBlock` fusion.
fn build(style: RopeStyle) -> Graph {
    let f = DType::F32;
    let half = DH / 2;
    let mut g = Graph::new("rope_style_fuse");
    let hidden = g.input("hidden", Shape::new(&[S, D], f));
    let wqkv = g.input("wqkv", Shape::new(&[D, 3 * D], f));
    let wo = g.input("wo", Shape::new(&[D, D], f));
    let cos = g.input("cos", Shape::new(&[S, half], f));
    let sin = g.input("sin", Shape::new(&[S, half], f));
    let qkv = g.matmul(hidden, wqkv, Shape::new(&[S, 3 * D], f));
    let q = g.narrow_(qkv, 1, 0, D);
    let k = g.narrow_(qkv, 1, D, D);
    let v = g.narrow_(qkv, 1, 2 * D, D);
    let q3 = g.reshape(q, vec![1, S as i64, D as i64], Shape::new(&[1, S, D], f));
    let k3 = g.reshape(k, vec![1, S as i64, D as i64], Shape::new(&[1, S, D], f));
    let v3 = g.reshape(v, vec![1, S as i64, D as i64], Shape::new(&[1, S, D], f));
    let qr = g.rope_styled(q3, cos, sin, DH, style);
    let kr = g.rope_styled(k3, cos, sin, DH, style);
    let attn = g.attention_kind(
        qr,
        kr,
        v3,
        NH,
        DH,
        MaskKind::Causal,
        Shape::new(&[1, S, D], f),
    );
    let a2 = g.reshape(attn, vec![S as i64, D as i64], Shape::new(&[S, D], f));
    let out = g.matmul(a2, wo, Shape::new(&[S, D], f));
    g.set_outputs(vec![out]);
    g
}

fn leaf(name: &str, n: usize) -> Vec<f32> {
    let mut h = 1469598103934665603u64;
    for b in name.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    (0..n)
        .map(|i| {
            let z = h.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            (((z >> 40) & 0xffff) as f32 / 65535.0 - 0.5) * 0.6
        })
        .collect()
}

fn run(g: &Graph, threshold: &str, want_fused: bool) -> Vec<f32> {
    // Decided at compile time; this test owns its process so the env is safe.
    unsafe {
        std::env::set_var("RLX_FUSE_ATTN_THRESHOLD", threshold);
    }
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(g, &arena);
    let fused = sched
        .thunks
        .iter()
        .any(|t| matches!(t, Thunk::FusedAttnBlock { .. }));
    assert_eq!(
        fused, want_fused,
        "FusedAttnBlock present={fused}, want={want_fused} (threshold={threshold})"
    );
    let half = DH / 2;
    for node in g.nodes() {
        let name = match &node.op {
            Op::Input { name } => name.clone(),
            _ => continue,
        };
        // Real rope tables for cos/sin so the rotation is non-trivial (small
        // random sin would make every pairing ≈ identity → styles indistinct).
        let data = if name == "cos" || name == "sin" {
            let mut t = vec![0f32; S * half];
            for pos in 0..S {
                for i in 0..half {
                    let freq = 1.0f32 / 10000f32.powf(2.0 * i as f32 / DH as f32);
                    let a = pos as f32 * freq;
                    t[pos * half + i] = if name == "cos" { a.cos() } else { a.sin() };
                }
            }
            t
        } else {
            leaf(&name, node.shape.num_elements().unwrap())
        };
        let off = arena.byte_offset(node.id);
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
            for (i, &val) in data.iter().enumerate() {
                *p.add(i) = val;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let off = arena.byte_offset(g.outputs[0]);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..S * D).map(|i| *p.add(i)).collect()
    }
}

#[test]
fn fused_attn_block_honors_rope_style() {
    let g_gptj = build(RopeStyle::GptJ);
    let fused = run(&g_gptj, "64", true); // batch·seq=5 ≤ 64 → FusedAttnBlock
    let unfused = run(&g_gptj, "0", false); // no fusion → base RoPE kernel
    unsafe {
        std::env::remove_var("RLX_FUSE_ATTN_THRESHOLD");
    }
    let max_abs = fused
        .iter()
        .zip(&unfused)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-5,
        "fused GPT-J attention diverged from unfused base kernel: max_abs={max_abs}"
    );

    // Non-vacuous: the fused path must actually apply GPT-J pairing, i.e. differ
    // from NeoX on the same weights.
    let g_neox = build(RopeStyle::NeoX);
    let fused_neox = run(&g_neox, "64", true);
    unsafe {
        std::env::remove_var("RLX_FUSE_ATTN_THRESHOLD");
    }
    let style_diff = fused
        .iter()
        .zip(&fused_neox)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        style_diff > 1e-4,
        "fused GPT-J and NeoX outputs identical — style not honored (diff={style_diff})"
    );
}
