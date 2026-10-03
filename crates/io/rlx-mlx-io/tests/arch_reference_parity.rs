// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Numerical parity for the llama-family graph builders.
//!
//! The other tests here assert shapes. That is how a graph that computes the
//! wrong thing passes: every tensor is the right size and every value is
//! wrong. This file computes the same forward pass independently, in plain
//! `f32`, and compares.
//!
//! The reference shares only `dequant_affine_f32` with the code under test —
//! that primitive has its own coverage — and re-derives the transformer
//! maths from the weights.

use rlx_ir::QuantScheme;
use rlx_mlx_io::{
    MlxArchConfig, MlxPackedLinear, PackedLinearBinding, build_llama_like_decode_dyn,
    build_llama_like_decode_masked, build_llama_like_decode_masked_embedded,
    build_llama_like_prefill_kv, build_llama_like_prefill_kv_embedded, decode_keep_mask,
    decode_rope_row, dequant_affine_f32, param_bindings_for,
};
use rlx_runtime::{Device, Session};

const GS: usize = 32;
const EPS: f32 = 1e-5;

fn affine_pack(n: usize, k: usize) -> MlxPackedLinear {
    affine_pack_seeded(n, k, 0)
}

fn affine_pack_seeded(n: usize, k: usize, seed: usize) -> MlxPackedLinear {
    let n_groups = k / GS;
    MlxPackedLinear {
        w_q: (0..n * (k / 2))
            .map(|i| ((i * 37 + 11 + seed * 53) % 256) as u8)
            .collect(),
        scales: (0..n * n_groups)
            .flat_map(|i| (0.02f32 + 0.001 * (i % 7) as f32).to_le_bytes())
            .collect(),
        biases: (0..n * n_groups)
            .flat_map(|i| (-0.01f32 + 0.001 * (i % 5) as f32).to_le_bytes())
            .collect(),
        scheme: QuantScheme::MlxAffine {
            bits: 4,
            group_size: GS as u32,
        },
        out_shape: vec![n, k],
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Dense `[n, k]` weights for a packed linear, via the tested primitive.
fn dense(p: &MlxPackedLinear) -> Vec<f32> {
    let (n, k) = (p.out_shape[0], p.out_shape[1]);
    let QuantScheme::MlxAffine { bits, group_size } = p.scheme else {
        panic!("test uses affine packing");
    };
    dequant_affine_f32(
        &p.w_q,
        &f32s(&p.scales),
        &f32s(&p.biases),
        bits as u32,
        group_size,
        n,
        k / group_size as usize,
    )
    .unwrap()
}

/// Per-layer RMSNorm gain. Distinct per layer so a graph that reused layer
/// 0's weights everywhere could not pass the multi-layer tests; the reference
/// reads the same function, so both sides stay in step.
fn ln_gain(h: usize, layer: usize) -> Vec<f32> {
    (0..h)
        .map(|i| 1.0 + 0.01 * ((i + layer) % 5) as f32)
        .collect()
}

/// Per-layer QK-norm gain (Qwen3 only), over `head_dim`.
fn qk_gain(hd: usize, layer: usize) -> Vec<f32> {
    (0..hd)
        .map(|i| 1.0 + 0.02 * ((i + layer) % 3) as f32)
        .collect()
}

struct Model {
    arch: MlxArchConfig,
    linears: Vec<PackedLinearBinding>,
    embed: Vec<f32>,
}

fn tiny(model_type: &str, nh: usize, nkv: usize) -> Model {
    tiny_l(model_type, nh, nkv, 1)
}

fn tiny_l(model_type: &str, nh: usize, nkv: usize, layers: usize) -> Model {
    let h = 64usize;
    let inter = 128usize;
    let hd = h / nh;
    let vocab = 32usize;
    let arch = MlxArchConfig {
        model_type: model_type.into(),
        vocab_size: vocab,
        hidden_size: h,
        intermediate_size: inter,
        num_hidden_layers: layers,
        num_attention_heads: nh,
        num_key_value_heads: nkv,
        rms_norm_eps: EPS,
        rope_theta: 10_000.0,
        max_position_embeddings: 128,
        head_dim: Some(hd),
    };
    // Per-layer weights must differ, or a graph that reads layer 0's weights
    // for every layer would still pass. `affine_pack` is seeded by layer.
    let linears = (0..layers)
        .flat_map(|l| {
            [
                ("self_attn.q_proj", nh * hd, h),
                ("self_attn.k_proj", nkv * hd, h),
                ("self_attn.v_proj", nkv * hd, h),
                ("self_attn.o_proj", h, nh * hd),
                ("mlp.gate_proj", inter, h),
                ("mlp.up_proj", inter, h),
                ("mlp.down_proj", h, inter),
            ]
            .into_iter()
            .map(move |(sfx, n, k)| PackedLinearBinding {
                name: format!("model.layers.{l}.{sfx}"),
                packed: affine_pack_seeded(n, k, l),
            })
        })
        .collect();
    let embed = (0..vocab * h)
        .map(|i| 0.01f32 * ((i % 7) as f32 + 1.0))
        .collect();
    Model {
        arch,
        linears,
        embed,
    }
}

fn bind(c: &mut rlx_runtime::CompiledGraph, m: &Model) {
    for b in &m.linears {
        for (name, bytes, dt) in param_bindings_for(b) {
            c.set_param_typed(&name, &bytes, dt);
        }
    }
    let h = m.arch.hidden_size;
    let hd = m.arch.head_dim();
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    c.set_param_typed(
        "model.embed_tokens.weight",
        &bytes(&m.embed),
        rlx_ir::DType::F32,
    );
    c.set_param_typed("lm_head.weight", &bytes(&m.embed), rlx_ir::DType::F32);

    let zeros = bytes(&vec![0.0f32; h]);
    c.set_param_typed(
        "model.norm.weight",
        &bytes(&vec![1.0f32; h]),
        rlx_ir::DType::F32,
    );
    c.set_param_typed("model.norm.bias_zero", &zeros, rlx_ir::DType::F32);
    for l in 0..m.arch.num_hidden_layers {
        let gain = ln_gain(h, l);
        for which in ["input_layernorm", "post_attention_layernorm"] {
            let key = format!("model.layers.{l}.{which}");
            c.set_param_typed(&format!("{key}.weight"), &bytes(&gain), rlx_ir::DType::F32);
            c.set_param_typed(&format!("{key}.bias_zero"), &zeros, rlx_ir::DType::F32);
        }
        if m.arch.uses_qk_norm() {
            let hd_gain = qk_gain(hd, l);
            let hd_zeros = bytes(&vec![0.0f32; hd]);
            for which in ["q_norm", "k_norm"] {
                let key = format!("model.layers.{l}.self_attn.{which}.weight");
                c.set_param_typed(&key, &bytes(&hd_gain), rlx_ir::DType::F32);
                c.set_param_typed(&format!("{key}.bias_zero"), &hd_zeros, rlx_ir::DType::F32);
            }
        }
    }
}

// ── reference maths ────────────────────────────────────────────────────

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// `y[i] = sum_j x[j] * w[i, j]`, with `w` row-major `[n, k]`.
fn matvec(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0..k).map(|j| x[j] * w[i * k + j]).sum())
        .collect()
}

/// NeoX / half-split RoPE on one head: pairs `(i, i + hd/2)`.
fn rope_neox(head: &mut [f32], pos: usize, theta: f64) {
    let hd = head.len();
    let half = hd / 2;
    for i in 0..half {
        let inv = 1.0 / theta.powf(2.0 * i as f64 / hd as f64);
        let ang = pos as f64 * inv;
        let (s, c) = (ang.sin() as f32, ang.cos() as f32);
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * c - b * s;
        head[i + half] = a * s + b * c;
    }
}

/// Reference K and V for every position, shaped `[seq, nkv, hd]` flattened.
fn reference_kv(m: &Model, tokens: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let a = &m.arch;
    let (h, hd, nkv) = (a.hidden_size, a.head_dim(), a.num_key_value_heads);
    let wk = dense(&m.linears[1].packed);
    let wv = dense(&m.linears[2].packed);
    let g1 = ln_gain(h, 0);
    let gq = qk_gain(hd, 0);

    let mut k_out = Vec::new();
    let mut v_out = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        let x = &m.embed[tok as usize * h..(tok as usize + 1) * h];
        let n1 = rmsnorm(x, &g1, a.rms_norm_eps);
        let mut k = matvec(&n1, &wk, nkv * hd, h);
        let v = matvec(&n1, &wv, nkv * hd, h);
        for head in 0..nkv {
            let slice = &mut k[head * hd..(head + 1) * hd];
            if a.uses_qk_norm() {
                let normed = rmsnorm(slice, &gq, a.rms_norm_eps);
                slice.copy_from_slice(&normed);
            }
            rope_neox(slice, pos, a.rope_theta as f64);
        }
        k_out.extend_from_slice(&k);
        v_out.extend_from_slice(&v);
    }
    (k_out, v_out)
}

fn compare(label: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let first_bad = got.iter().zip(want).position(|(g, w)| (g - w).abs() > tol);
    let (worst, at) =
        got.iter()
            .zip(want)
            .enumerate()
            .fold((0.0f32, 0usize), |(worst, at), (i, (g, w))| {
                let d = (g - w).abs();
                if d > worst { (d, i) } else { (worst, at) }
            });
    if first_bad.is_none() {
        return;
    }
    // A breakdown beats a single maximum: where the error starts, and
    // whether it is a constant ratio (a scale factor) or structural.
    let head: Vec<String> = (0..8.min(got.len()))
        .map(|i| format!("[{i}] {:.5}/{:.5}", got[i], want[i]))
        .collect();
    let ratios: Vec<String> = (0..6.min(got.len()))
        .filter(|i| want[*i].abs() > 1e-6)
        .map(|i| format!("{:.4}", got[i] / want[i]))
        .collect();
    panic!(
        "{label}: first mismatch at {:?}, worst {worst:.6} at {at} \
         (graph {:.6} vs ref {:.6})\n  first values graph/ref: {}\n  ratios: {}",
        first_bad,
        got[at],
        want[at],
        head.join("  "),
        ratios.join(" ")
    );
}

// ── tests ──────────────────────────────────────────────────────────────

/// V skips RoPE entirely, so if V matches and K does not, the projection and
/// norm are fine and the fault is in the rotation.
#[test]
fn prefill_v_matches_reference() {
    let m = tiny("llama", 4, 2);
    let tokens = [3u32, 7, 1];
    let g =
        build_llama_like_prefill_kv("v", &m.arch, &m.linears, 1, tokens.len(), Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tb: Vec<u8> = tokens
        .iter()
        .flat_map(|t| (*t as f32).to_le_bytes())
        .collect();
    let outs = c.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);
    let (_, want_v) = reference_kv(&m, &tokens);
    compare("V", &f32s(&outs[2].0), &want_v, 1e-3);
}

/// One KV head means RoPE has no per-head stride to get wrong.
#[test]
fn prefill_k_matches_reference_single_head() {
    let m = tiny("llama", 1, 1);
    let tokens = [3u32, 7];
    let g =
        build_llama_like_prefill_kv("k1", &m.arch, &m.linears, 1, tokens.len(), Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tb: Vec<u8> = tokens
        .iter()
        .flat_map(|t| (*t as f32).to_le_bytes())
        .collect();
    let outs = c.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);
    let (want_k, _) = reference_kv(&m, &tokens);
    compare("K (1 head)", &f32s(&outs[1].0), &want_k, 1e-3);
}

/// K and V come out of the layer after embedding → RMSNorm → projection →
/// (QK-norm) → RoPE. Checking them isolates the first half of attention from
/// everything downstream.
#[test]
fn prefill_kv_matches_reference_llama() {
    let m = tiny("llama", 4, 2);
    let tokens = [3u32, 7, 1];
    let seq = tokens.len();

    let g = build_llama_like_prefill_kv("ref", &m.arch, &m.linears, 1, seq, Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tok_bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|t| (*t as f32).to_le_bytes())
        .collect();
    let outs = c.run_typed(&[("tokens", tok_bytes.as_slice(), rlx_ir::DType::F32)]);

    let (want_k, want_v) = reference_kv(&m, &tokens);
    compare("K", &f32s(&outs[1].0), &want_k, 1e-3);
    compare("V", &f32s(&outs[2].0), &want_v, 1e-3);
}

#[test]
fn prefill_kv_matches_reference_qwen3_with_qk_norm() {
    let m = tiny("qwen3", 4, 2);
    let tokens = [5u32, 2];
    let seq = tokens.len();

    let g = build_llama_like_prefill_kv("ref", &m.arch, &m.linears, 1, seq, Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tok_bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|t| (*t as f32).to_le_bytes())
        .collect();
    let outs = c.run_typed(&[("tokens", tok_bytes.as_slice(), rlx_ir::DType::F32)]);

    let (want_k, want_v) = reference_kv(&m, &tokens);
    compare("K", &f32s(&outs[1].0), &want_k, 1e-3);
    compare("V", &f32s(&outs[2].0), &want_v, 1e-3);
}

/// The foundational check: does the graph's `DequantMatMul` agree with the
/// standalone `dequant_affine_f32` on the same packed bytes?
///
/// Every projection in the model is this operation. If the two disagree —
/// on nibble order, group layout, or orientation — every downstream value is
/// wrong while staying plausibly scaled, which is exactly the symptom.
#[test]
fn dequant_matmul_agrees_with_standalone_dequant() {
    // One group per row, two, and three: group indexing is where a stride
    // bug hides, and a single-group case cannot see it.
    for k in [GS, GS * 2, GS * 3] {
        check_dequant_matmul(8, k);
    }
}

fn check_dequant_matmul(n: usize, k: usize) {
    use rlx_ir::{DType, Graph, Op, Shape};

    let packed = affine_pack(n, k);
    let w = dense(&packed);

    // x · Wᵀ through the graph.
    let mut g = Graph::new("dq_one");
    let x = g.input("x", Shape::new(&[1, k], DType::F32));
    let n_groups = k / GS;
    let wq = g.param("w.weight", Shape::new(&[packed.w_q.len()], DType::U8));
    let sc = g.param("w.scales", Shape::new(&[n, n_groups], packed.scale_dtype()));
    let bi = g.param("w.biases", Shape::new(&[n, n_groups], packed.bias_dtype()));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: packed.scheme,
        },
        vec![x, wq, sc, bi],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut c = Session::new(Device::Cpu).compile(g);
    for (name, bytes, dt) in param_bindings_for(&PackedLinearBinding {
        name: "w".into(),
        packed: packed.clone(),
    }) {
        c.set_param_typed(&name, &bytes, dt);
    }

    let xs: Vec<f32> = (0..k).map(|i| 0.1 * (i % 5) as f32 - 0.2).collect();
    let xb: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = c.run_typed(&[("x", xb.as_slice(), DType::F32)]);

    let want = matvec(&xs, &w, n, k);
    compare(
        &format!(
            "DequantMatMul vs dequant_affine_f32 (k={k}, {} groups)",
            k / GS
        ),
        &f32s(&outs[0].0),
        &want,
        1e-3,
    );
}

/// RMSNorm convention: `x * rsqrt(mean(x²) + eps) * weight`, no `1 +` on the
/// weight (that is Gemma's variant) and no bias term beyond the zero one the
/// graph declares.
#[test]
fn rmsnorm_convention_matches_reference() {
    use rlx_ir::{DType, Graph, Op, Shape};

    let h = 64usize;
    let mut g = Graph::new("rms");
    let x = g.input("x", Shape::new(&[1, h], DType::F32));
    let w = g.param("w", Shape::new(&[h], DType::F32));
    let b = g.param("b", Shape::new(&[h], DType::F32));
    let y = g.add_node(
        Op::RmsNorm { axis: -1, eps: EPS },
        vec![x, w, b],
        Shape::new(&[1, h], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let weight: Vec<f32> = (0..h).map(|i| 0.5 + 0.01 * i as f32).collect();
    c.set_param_typed("w", &bytes(&weight), DType::F32);
    c.set_param_typed("b", &bytes(&vec![0.0f32; h]), DType::F32);

    let xs: Vec<f32> = (0..h).map(|i| 0.01 * ((i % 7) as f32 + 1.0)).collect();
    let outs = c.run_typed(&[("x", bytes(&xs).as_slice(), DType::F32)]);

    compare(
        "RmsNorm",
        &f32s(&outs[0].0),
        &rmsnorm(&xs, &weight, EPS),
        1e-5,
    );
}

/// Smallest possible case: one token, one head, one layer. No layout
/// ambiguity, no RoPE rotation at position 0 (angle 0 ⇒ identity), so K is
/// just `rmsnorm(embed[tok]) · W_kᵀ`. If this disagrees, the composition of
/// gather → norm → projection is wrong, not any of the pieces.
#[test]
fn single_token_single_head_k_is_norm_times_weights() {
    let m = tiny("llama", 1, 1);
    let tok = 3u32;
    let a = &m.arch;
    let (h, hd) = (a.hidden_size, a.head_dim());

    let g = build_llama_like_prefill_kv("one", a, &m.linears, 1, 1, Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tb = (tok as f32).to_le_bytes().to_vec();
    let outs = c.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);
    let got_k = f32s(&outs[1].0);

    // Reference, computed inline so every stage is visible.
    let x = &m.embed[tok as usize * h..(tok as usize + 1) * h];
    let n1 = rmsnorm(x, &ln_gain(h, 0), a.rms_norm_eps);
    let wk = dense(&m.linears[1].packed);
    let want = matvec(&n1, &wk, hd, h); // nkv*hd == hd for one head

    eprintln!("embed[{tok}][0..4] = {:?}", &x[..4]);
    eprintln!("n1[0..4]           = {:?}", &n1[..4]);
    eprintln!("wk[0][0..4]        = {:?}", &wk[..4]);
    eprintln!("graph K[0..4]      = {:?}", &got_k[..4]);
    eprintln!("ref   K[0..4]      = {:?}", &want[..4]);

    compare("K (1 token, 1 head)", &got_k, &want, 1e-3);
}

/// The one link the other tests leave unchecked: does the embedding gather
/// return the row the token names?
#[test]
fn embedding_gather_returns_the_right_row() {
    use rlx_ir::{DType, Graph, Op, Shape};

    let (vocab, h) = (32usize, 64usize);
    let embed: Vec<f32> = (0..vocab * h)
        .map(|i| 0.01 * ((i % 7) as f32 + 1.0))
        .collect();

    let mut g = Graph::new("gather");
    let tokens = g.input("tokens", Shape::new(&[1, 1], DType::F32));
    let w = g.param("embed", Shape::new(&[vocab, h], DType::F32));
    let flat = g.add_node(
        Op::Reshape { new_shape: vec![1] },
        vec![tokens],
        Shape::new(&[1], DType::F32),
    );
    let out = g.add_node(
        Op::Gather { axis: 0 },
        vec![w, flat],
        Shape::new(&[1, h], DType::F32),
    );
    g.set_outputs(vec![out]);

    let mut c = Session::new(Device::Cpu).compile(g);
    let bytes: Vec<u8> = embed.iter().flat_map(|x| x.to_le_bytes()).collect();
    c.set_param_typed("embed", &bytes, DType::F32);

    for tok in [0u32, 1, 3, 7, 31] {
        let tb = (tok as f32).to_le_bytes().to_vec();
        let outs = c.run_typed(&[("tokens", tb.as_slice(), DType::F32)]);
        let got = f32s(&outs[0].0);
        let want = &embed[tok as usize * h..(tok as usize + 1) * h];
        compare(&format!("gather row {tok}"), &got, want, 1e-6);
    }
}

/// When the graph and the reference disagree, say *which* embedding row the
/// graph behaved as though it read. A mismatch that resolves to a specific
/// wrong row is an indexing bug; one that resolves to none is arithmetic.
#[test]
fn identify_which_row_the_layer_used() {
    let m = tiny("llama", 1, 1);
    let tok = 3u32;
    let a = &m.arch;
    let (h, hd) = (a.hidden_size, a.head_dim());

    let g = build_llama_like_prefill_kv("which", a, &m.linears, 1, 1, Some(1)).unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind(&mut c, &m);
    let tb = (tok as f32).to_le_bytes().to_vec();
    let got = f32s(&c.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)])[1].0);

    let wk = dense(&m.linears[1].packed);
    let ones = vec![1.0f32; h];
    let mut best = (f32::INFINITY, usize::MAX);
    for row in 0..a.vocab_size {
        let x = &m.embed[row * h..(row + 1) * h];
        let cand = matvec(&rmsnorm(x, &ones, a.rms_norm_eps), &wk, hd, h);
        let err: f32 = cand
            .iter()
            .zip(&got)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f32::max);
        if err < best.0 {
            best = (err, row);
        }
    }
    eprintln!(
        "graph output best matches embedding row {} (max error {:.6}); token was {tok}",
        best.1, best.0
    );
    assert_eq!(
        best.1, tok as usize,
        "the layer read row {} for token {tok}",
        best.1
    );
}

/// Decode has to *continue* prefill: one decode step on top of the KV for
/// `tokens[..n]` must give the logits that a prefill over `tokens[..n + 1]`
/// gives at its last row. Prefill is pinned to the hand-written reference
/// above, so this transfers that guarantee to the decode graph — RoPE
/// position, KV append order, attention over past+new — without needing a
/// second reference implementation of the whole step.
#[test]
fn decode_step_continues_prefill() {
    let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (mt, nh, nkv) in [("llama", 4, 2), ("llama", 1, 1), ("qwen3", 4, 2)] {
        let m = tiny(mt, nh, nkv);
        let tokens = [3u32, 7, 1, 5];
        let n = tokens.len() - 1;
        let vocab = m.arch.vocab_size;

        // Ground truth: prefill the whole sequence, take the last row.
        let gf = build_llama_like_prefill_kv("full", &m.arch, &m.linears, 1, tokens.len(), Some(1))
            .unwrap();
        let mut cf = Session::new(Device::Cpu).compile(gf);
        bind(&mut cf, &m);
        let tb_full: Vec<u8> = tokens
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let of = cf.run_typed(&[("tokens", tb_full.as_slice(), rlx_ir::DType::F32)]);
        let full = f32s(&of[0].0);
        let want = &full[n * vocab..(n + 1) * vocab];

        // Prefill the prefix and keep its KV.
        let gp = build_llama_like_prefill_kv("pre", &m.arch, &m.linears, 1, n, Some(1)).unwrap();
        let mut cp = Session::new(Device::Cpu).compile(gp);
        bind(&mut cp, &m);
        let tb: Vec<u8> = tokens[..n]
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let op = cp.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);
        let (pk, pv) = (op[1].0.clone(), op[2].0.clone());

        // One decode step at position n.
        let gd = build_llama_like_decode_dyn("dec", &m.arch, &m.linears, 1, n, Some(1)).unwrap();
        let mut cd = Session::new(Device::Cpu).compile(gd);
        bind(&mut cd, &m);
        let (cos, sin) = decode_rope_row(&m.arch, n);
        let (cb, sb) = (fb(&cos), fb(&sin));
        let tok = (tokens[n] as f32).to_le_bytes().to_vec();
        let od = cd.run_typed(&[
            ("token", tok.as_slice(), rlx_ir::DType::F32),
            ("rope_cos", cb.as_slice(), rlx_ir::DType::F32),
            ("rope_sin", sb.as_slice(), rlx_ir::DType::F32),
            ("past_k_0", pk.as_slice(), rlx_ir::DType::F32),
            ("past_v_0", pv.as_slice(), rlx_ir::DType::F32),
        ]);
        compare(
            &format!("{mt} nh={nh} nkv={nkv} decode logits"),
            &f32s(&od[0].0),
            want,
            2e-3,
        );
    }
}

/// The engine does not run decode at exact capacity — it pads the KV cache to
/// a bucket and masks the slack. That is the path that actually generates
/// tokens, so it needs the same guarantee: padded + masked decode must equal
/// prefill's last row. The padding is filled with large values, so a mask that
/// leaks shows up as a gross mismatch rather than a rounding error.
#[test]
fn masked_decode_step_continues_prefill() {
    let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (mt, nh, nkv) in [("llama", 4, 2), ("qwen3", 4, 2)] {
        for capacity in [4usize, 8, 16] {
            let m = tiny(mt, nh, nkv);
            let tokens = [3u32, 7, 1, 5];
            let real = tokens.len() - 1;
            let vocab = m.arch.vocab_size;
            let (hd, row) = (m.arch.head_dim(), nkv * m.arch.head_dim());

            let gf =
                build_llama_like_prefill_kv("full", &m.arch, &m.linears, 1, tokens.len(), Some(1))
                    .unwrap();
            let mut cf = Session::new(Device::Cpu).compile(gf);
            bind(&mut cf, &m);
            let tbf: Vec<u8> = tokens
                .iter()
                .flat_map(|t| (*t as f32).to_le_bytes())
                .collect();
            let of = cf.run_typed(&[("tokens", tbf.as_slice(), rlx_ir::DType::F32)]);
            let full = f32s(&of[0].0);
            let want = &full[real * vocab..(real + 1) * vocab];

            let gp =
                build_llama_like_prefill_kv("pre", &m.arch, &m.linears, 1, real, Some(1)).unwrap();
            let mut cp = Session::new(Device::Cpu).compile(gp);
            bind(&mut cp, &m);
            let tb: Vec<u8> = tokens[..real]
                .iter()
                .flat_map(|t| (*t as f32).to_le_bytes())
                .collect();
            let op = cp.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);

            // Pad each cache to `capacity` rows; slack is deliberately loud.
            let pad = |src: &[u8]| -> Vec<u8> {
                let mut v = f32s(src);
                assert_eq!(v.len(), real * row, "prefill KV row stride");
                v.resize(capacity * row, 1_000.0);
                fb(&v)
            };
            let (pk, pv) = (pad(&op[1].0), pad(&op[2].0));

            let gd =
                build_llama_like_decode_masked("decm", &m.arch, &m.linears, 1, capacity, Some(1))
                    .unwrap();
            let mut cd = Session::new(Device::Cpu).compile(gd);
            bind(&mut cd, &m);
            let (cos, sin) = decode_rope_row(&m.arch, real);
            let (cb, sb) = (fb(&cos), fb(&sin));
            let mb = fb(&decode_keep_mask(real, capacity));
            let tok = (tokens[real] as f32).to_le_bytes().to_vec();
            let od = cd.run_typed(&[
                ("token", tok.as_slice(), rlx_ir::DType::F32),
                ("rope_cos", cb.as_slice(), rlx_ir::DType::F32),
                ("rope_sin", sb.as_slice(), rlx_ir::DType::F32),
                ("mask", mb.as_slice(), rlx_ir::DType::F32),
                ("past_k_0", pk.as_slice(), rlx_ir::DType::F32),
                ("past_v_0", pv.as_slice(), rlx_ir::DType::F32),
            ]);
            let _ = hd;
            compare(
                &format!("{mt} nh={nh} cap={capacity} masked decode logits"),
                &f32s(&od[0].0),
                want,
                2e-3,
            );
        }
    }
}

/// Everything above runs one layer, so a graph that crossed layer indices —
/// fed `past_k_1` into layer 0, or emitted the caches out of order — would
/// pass. The engine runs 28. Per-layer weights and norm gains differ, so any
/// crossing changes the logits.
#[test]
fn multi_layer_decode_continues_multi_layer_prefill() {
    let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (mt, layers) in [("llama", 2usize), ("llama", 3), ("qwen3", 3)] {
        let (nh, nkv) = (4usize, 2usize);
        let m = tiny_l(mt, nh, nkv, layers);
        let tokens = [3u32, 7, 1, 5];
        let real = tokens.len() - 1;
        let vocab = m.arch.vocab_size;
        let row = nkv * m.arch.head_dim();
        let capacity = 8usize;

        let gf = build_llama_like_prefill_kv(
            "mfull",
            &m.arch,
            &m.linears,
            1,
            tokens.len(),
            Some(layers),
        )
        .unwrap();
        let mut cf = Session::new(Device::Cpu).compile(gf);
        bind(&mut cf, &m);
        let tbf: Vec<u8> = tokens
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let of = cf.run_typed(&[("tokens", tbf.as_slice(), rlx_ir::DType::F32)]);
        assert_eq!(of.len(), 1 + 2 * layers, "prefill output count");
        let full = f32s(&of[0].0);
        let want = &full[real * vocab..(real + 1) * vocab];

        let gp = build_llama_like_prefill_kv("mpre", &m.arch, &m.linears, 1, real, Some(layers))
            .unwrap();
        let mut cp = Session::new(Device::Cpu).compile(gp);
        bind(&mut cp, &m);
        let tb: Vec<u8> = tokens[..real]
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let op = cp.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);

        let pad = |src: &[u8]| -> Vec<u8> {
            let mut v = f32s(src);
            assert_eq!(v.len(), real * row, "KV row stride");
            v.resize(capacity * row, 1_000.0);
            fb(&v)
        };
        let caches: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|l| (pad(&op[1 + 2 * l].0), pad(&op[2 + 2 * l].0)))
            .collect();

        let gd =
            build_llama_like_decode_masked("mdec", &m.arch, &m.linears, 1, capacity, Some(layers))
                .unwrap();
        let mut cd = Session::new(Device::Cpu).compile(gd);
        bind(&mut cd, &m);
        let (cos, sin) = decode_rope_row(&m.arch, real);
        let (cb, sb) = (fb(&cos), fb(&sin));
        let mb = fb(&decode_keep_mask(real, capacity));
        let tok = (tokens[real] as f32).to_le_bytes().to_vec();
        let names: Vec<(String, String)> = (0..layers)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();
        let mut inputs: Vec<(&str, &[u8], rlx_ir::DType)> = vec![
            ("token", tok.as_slice(), rlx_ir::DType::F32),
            ("rope_cos", cb.as_slice(), rlx_ir::DType::F32),
            ("rope_sin", sb.as_slice(), rlx_ir::DType::F32),
            ("mask", mb.as_slice(), rlx_ir::DType::F32),
        ];
        for (l, (kn, vn)) in names.iter().enumerate() {
            inputs.push((kn.as_str(), caches[l].0.as_slice(), rlx_ir::DType::F32));
            inputs.push((vn.as_str(), caches[l].1.as_slice(), rlx_ir::DType::F32));
        }
        let od = cd.run_typed(&inputs);
        compare(
            &format!("{mt} L={layers} multi-layer decode logits"),
            &f32s(&od[0].0),
            want,
            2e-3,
        );
    }
}

/// Two decode steps, driven exactly the way an engine must drive them. One
/// step cannot catch a cache that is written back to the wrong row: the error
/// only shows on the *next* step, when the row it should have filled is read
/// as context. This pins the protocol — the appended row comes back at index
/// `capacity`, and belongs at index `real` — which is the part an engine has
/// to get right and the graph cannot enforce.
#[test]
fn successive_decode_steps_match_prefill() {
    let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (mt, layers) in [("llama", 1usize), ("llama", 2), ("qwen3", 2)] {
        let (nh, nkv) = (4usize, 2usize);
        let m = tiny_l(mt, nh, nkv, layers);
        let tokens = [3u32, 7, 1, 5, 2];
        let prefix = 2usize; // prefill this many, decode the rest
        let vocab = m.arch.vocab_size;
        let row = nkv * m.arch.head_dim();
        let capacity = 8usize;

        // Ground truth for every step: one prefill over the whole sequence.
        let gf = build_llama_like_prefill_kv(
            "sfull",
            &m.arch,
            &m.linears,
            1,
            tokens.len(),
            Some(layers),
        )
        .unwrap();
        let mut cf = Session::new(Device::Cpu).compile(gf);
        bind(&mut cf, &m);
        let tbf: Vec<u8> = tokens
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let full = f32s(&cf.run_typed(&[("tokens", tbf.as_slice(), rlx_ir::DType::F32)])[0].0);

        // Prefill the prefix into a padded cache.
        let gp = build_llama_like_prefill_kv("spre", &m.arch, &m.linears, 1, prefix, Some(layers))
            .unwrap();
        let mut cp = Session::new(Device::Cpu).compile(gp);
        bind(&mut cp, &m);
        let tb: Vec<u8> = tokens[..prefix]
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let op = cp.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);
        let mut caches: Vec<(Vec<f32>, Vec<f32>)> = (0..layers)
            .map(|l| {
                let mut k = f32s(&op[1 + 2 * l].0);
                let mut v = f32s(&op[2 + 2 * l].0);
                k.resize(capacity * row, 1_000.0);
                v.resize(capacity * row, 1_000.0);
                (k, v)
            })
            .collect();

        let gd =
            build_llama_like_decode_masked("sdec", &m.arch, &m.linears, 1, capacity, Some(layers))
                .unwrap();
        let mut cd = Session::new(Device::Cpu).compile(gd);
        bind(&mut cd, &m);
        let names: Vec<(String, String)> = (0..layers)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();

        for real in prefix..tokens.len() {
            let (cos, sin) = decode_rope_row(&m.arch, real);
            let (cb, sb) = (fb(&cos), fb(&sin));
            let mb = fb(&decode_keep_mask(real, capacity));
            let tok = (tokens[real] as f32).to_le_bytes().to_vec();
            let bufs: Vec<(Vec<u8>, Vec<u8>)> =
                caches.iter().map(|(k, v)| (fb(k), fb(v))).collect();
            let mut inputs: Vec<(&str, &[u8], rlx_ir::DType)> = vec![
                ("token", tok.as_slice(), rlx_ir::DType::F32),
                ("rope_cos", cb.as_slice(), rlx_ir::DType::F32),
                ("rope_sin", sb.as_slice(), rlx_ir::DType::F32),
                ("mask", mb.as_slice(), rlx_ir::DType::F32),
            ];
            for (l, (kn, vn)) in names.iter().enumerate() {
                inputs.push((kn.as_str(), bufs[l].0.as_slice(), rlx_ir::DType::F32));
                inputs.push((vn.as_str(), bufs[l].1.as_slice(), rlx_ir::DType::F32));
            }
            let od = cd.run_typed(&inputs);

            compare(
                &format!("{mt} L={layers} decode at pos {real}"),
                &f32s(&od[0].0),
                &full[real * vocab..(real + 1) * vocab],
                2e-3,
            );

            // Fold the appended row (at `capacity`) into slot `real`.
            for l in 0..layers {
                let k_out = f32s(&od[1 + 2 * l].0);
                let v_out = f32s(&od[2 + 2 * l].0);
                assert_eq!(k_out.len(), (capacity + 1) * row, "appended cache length");
                let src = capacity * row;
                let dst = real * row;
                caches[l].0[dst..dst + row].copy_from_slice(&k_out[src..src + row]);
                caches[l].1[dst..dst + row].copy_from_slice(&v_out[src..src + row]);
            }
        }
    }
}

/// The `EmbedSource::Rows` graphs must compute exactly what the `Table` graphs
/// compute — they only move the embedding lookup out to the caller, so the
/// logits are the same numbers or the refactor is wrong. Checked for prefill
/// and for masked decode, since both grew a variant.
#[test]
fn embedded_input_graphs_match_the_table_graphs() {
    let fb = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    for (mt, layers) in [("llama", 2usize), ("qwen3", 2)] {
        let m = tiny_l(mt, 4, 2, layers);
        let tokens = [3u32, 7, 1];
        let h = m.arch.hidden_size;
        let seq = tokens.len();
        let row = 2 * m.arch.head_dim();
        let capacity = 8usize;

        // Caller-side lookup: exactly the rows the table graph would gather.
        let rows: Vec<f32> = tokens
            .iter()
            .flat_map(|t| m.embed[*t as usize * h..(*t as usize + 1) * h].to_vec())
            .collect();

        // ── prefill ────────────────────────────────────────────────────
        let gt =
            build_llama_like_prefill_kv("t", &m.arch, &m.linears, 1, seq, Some(layers)).unwrap();
        let mut ct = Session::new(Device::Cpu).compile(gt);
        bind(&mut ct, &m);
        let tb: Vec<u8> = tokens
            .iter()
            .flat_map(|t| (*t as f32).to_le_bytes())
            .collect();
        let want = ct.run_typed(&[("tokens", tb.as_slice(), rlx_ir::DType::F32)]);

        let ge =
            build_llama_like_prefill_kv_embedded("e", &m.arch, &m.linears, 1, seq, Some(layers))
                .unwrap();
        let mut ce = Session::new(Device::Cpu).compile(ge);
        bind(&mut ce, &m);
        let rb = fb(&rows);
        let got = ce.run_typed(&[("embeddings", rb.as_slice(), rlx_ir::DType::F32)]);

        assert_eq!(got.len(), want.len(), "{mt}: prefill output count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            compare(
                &format!("{mt} prefill embedded output {i}"),
                &f32s(&g.0),
                &f32s(&w.0),
                1e-6,
            );
        }

        // ── masked decode ──────────────────────────────────────────────
        let pad = |src: &[u8]| -> Vec<u8> {
            let mut v = f32s(src);
            v.resize(capacity * row, 1_000.0);
            fb(&v)
        };
        let caches: Vec<(Vec<u8>, Vec<u8>)> = (0..layers)
            .map(|l| (pad(&want[1 + 2 * l].0), pad(&want[2 + 2 * l].0)))
            .collect();
        let (cos, sin) = decode_rope_row(&m.arch, seq);
        let (cb, sb) = (fb(&cos), fb(&sin));
        let mb = fb(&decode_keep_mask(seq, capacity));
        let next = 5u32;
        let tok = (next as f32).to_le_bytes().to_vec();
        let nrow = fb(&m.embed[next as usize * h..(next as usize + 1) * h]);
        let names: Vec<(String, String)> = (0..layers)
            .map(|l| (format!("past_k_{l}"), format!("past_v_{l}")))
            .collect();

        let run = |c: &mut rlx_runtime::CompiledGraph, first: (&str, &[u8])| {
            let mut inputs: Vec<(&str, &[u8], rlx_ir::DType)> = vec![
                (first.0, first.1, rlx_ir::DType::F32),
                ("rope_cos", cb.as_slice(), rlx_ir::DType::F32),
                ("rope_sin", sb.as_slice(), rlx_ir::DType::F32),
                ("mask", mb.as_slice(), rlx_ir::DType::F32),
            ];
            for (l, (kn, vn)) in names.iter().enumerate() {
                inputs.push((kn.as_str(), caches[l].0.as_slice(), rlx_ir::DType::F32));
                inputs.push((vn.as_str(), caches[l].1.as_slice(), rlx_ir::DType::F32));
            }
            c.run_typed(&inputs)
        };

        let gdt =
            build_llama_like_decode_masked("dt", &m.arch, &m.linears, 1, capacity, Some(layers))
                .unwrap();
        let mut cdt = Session::new(Device::Cpu).compile(gdt);
        bind(&mut cdt, &m);
        let dwant = run(&mut cdt, ("token", tok.as_slice()));

        let gde = build_llama_like_decode_masked_embedded(
            "de",
            &m.arch,
            &m.linears,
            1,
            capacity,
            Some(layers),
        )
        .unwrap();
        let mut cde = Session::new(Device::Cpu).compile(gde);
        bind(&mut cde, &m);
        let dgot = run(&mut cde, ("embeddings", nrow.as_slice()));

        assert_eq!(dgot.len(), dwant.len(), "{mt}: decode output count");
        for (i, (g, w)) in dgot.iter().zip(&dwant).enumerate() {
            compare(
                &format!("{mt} decode embedded output {i}"),
                &f32s(&g.0),
                &f32s(&w.0),
                1e-6,
            );
        }
    }
}
