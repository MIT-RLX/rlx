// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Decode graph (past K/V concat) builds and runs on CPU.

use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{
    MlxArchConfig, MlxPackedLinear, PackedLinearBinding, build_llama_like_decode,
    build_llama_like_prefill, param_bindings_for,
};
use rlx_runtime::{Device, Session};

fn affine_pack(n: usize, k: usize) -> MlxPackedLinear {
    let gs = 32usize;
    let n_groups = k / gs;
    let w_q: Vec<u8> = (0..n * (k / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let scales: Vec<u8> = (0..n * n_groups)
        .flat_map(|i| (0.02f32 + 0.001 * (i % 7) as f32).to_le_bytes())
        .collect();
    let biases: Vec<u8> = (0..n * n_groups)
        .flat_map(|i| (-0.01f32 + 0.001 * (i % 5) as f32).to_le_bytes())
        .collect();
    MlxPackedLinear {
        w_q,
        scales,
        biases,
        scheme: QuantScheme::MlxAffine {
            bits: 4,
            group_size: gs as u32,
        },
        out_shape: vec![n, k],
    }
}

fn lin(name: &str, n: usize, k: usize) -> PackedLinearBinding {
    PackedLinearBinding {
        name: name.into(),
        packed: affine_pack(n, k),
    }
}

fn tiny_arch() -> (MlxArchConfig, Vec<PackedLinearBinding>) {
    let h = 64usize;
    let inter = 128usize;
    let nh = 4usize;
    let nkv = 2usize;
    let hd = h / nh;
    let arch = MlxArchConfig {
        model_type: "llama".into(),
        vocab_size: 32,
        hidden_size: h,
        intermediate_size: inter,
        num_hidden_layers: 1,
        num_attention_heads: nh,
        num_key_value_heads: nkv,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 128,
        head_dim: Some(hd),
    };
    let prefix = "model.layers.0";
    let linears = vec![
        lin(&format!("{prefix}.self_attn.q_proj"), nh * hd, h),
        lin(&format!("{prefix}.self_attn.k_proj"), nkv * hd, h),
        lin(&format!("{prefix}.self_attn.v_proj"), nkv * hd, h),
        lin(&format!("{prefix}.self_attn.o_proj"), h, nh * hd),
        lin(&format!("{prefix}.mlp.gate_proj"), inter, h),
        lin(&format!("{prefix}.mlp.up_proj"), inter, h),
        lin(&format!("{prefix}.mlp.down_proj"), h, inter),
    ];
    (arch, linears)
}

fn bind_common(
    c: &mut rlx_runtime::CompiledGraph,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
) {
    for b in linears {
        for (name, bytes, dt) in param_bindings_for(b) {
            c.set_param_typed(&name, &bytes, dt);
        }
    }
    let h = arch.hidden_size;
    let emb: Vec<u8> = (0..arch.vocab_size * h)
        .flat_map(|i| (0.01f32 * ((i % 7) as f32 + 1.0)).to_le_bytes())
        .collect();
    c.set_param_typed("model.embed_tokens.weight", &emb, rlx_ir::DType::F32);
    let ones: Vec<u8> = (0..h).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let zeros: Vec<u8> = (0..h).flat_map(|_| 0.0f32.to_le_bytes()).collect();
    for name in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ] {
        c.set_param_typed(name, &ones, rlx_ir::DType::F32);
    }
    for name in [
        "model.layers.0.input_layernorm.bias_zero",
        "model.layers.0.post_attention_layernorm.bias_zero",
        "model.norm.bias_zero",
    ] {
        c.set_param_typed(name, &zeros, rlx_ir::DType::F32);
    }
    let head: Vec<u8> = (0..arch.vocab_size * h)
        .flat_map(|i| (0.02f32 * ((i % 5) as f32 + 1.0)).to_le_bytes())
        .collect();
    c.set_param_typed("lm_head.weight", &head, rlx_ir::DType::F32);
}

#[test]
fn tiny_llama_decode_cpu() {
    let (arch, linears) = tiny_arch();
    let batch = 1usize;
    let past_len = 2usize;
    let g = build_llama_like_decode(
        "tiny_dec",
        &arch,
        &linears,
        batch,
        past_len,
        past_len,
        Some(1),
    )
    .unwrap();
    let mut c = Session::new(Device::Cpu).compile(g);
    bind_common(&mut c, &arch, &linears);

    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let past_elems = batch * past_len * nkv * hd;
    let past_k = vec![0.01f32; past_elems];
    let past_v = vec![0.02f32; past_elems];
    let past_k_b: Vec<u8> = past_k.iter().flat_map(|x| x.to_le_bytes()).collect();
    let past_v_b: Vec<u8> = past_v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let token_bytes = 3f32.to_le_bytes().to_vec();
    let outs = c.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::F32),
        ("past_k_0", past_k_b.as_slice(), rlx_ir::DType::F32),
        ("past_v_0", past_v_b.as_slice(), rlx_ir::DType::F32),
    ]);
    // logits + new_k + new_v
    assert_eq!(outs.len(), 3);
    assert_eq!(outs[0].1, rlx_ir::DType::F32);
    let logits_len = outs[0].0.len() / 4;
    assert_eq!(logits_len, batch * arch.vocab_size);
    let kv_len = outs[1].0.len() / 4;
    assert_eq!(kv_len, batch * (past_len + 1) * nkv * hd);
}

#[test]
fn tiny_llama_prefill_still_builds() {
    let (arch, linears) = tiny_arch();
    let g = build_llama_like_prefill("tiny_pf", &arch, &linears, 1, 2, Some(1)).unwrap();
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, rlx_ir::Op::Rope { .. }))
    );
}

/// The dynamic-RoPE decode graph must produce the same logits as the
/// position-baked one. That equivalence is the whole point: a generation
/// loop can then compile once per KV length instead of once per token.
#[test]
fn dyn_rope_decode_matches_baked_constants() {
    use rlx_mlx_io::{build_llama_like_decode_dyn, decode_rope_row};

    let (arch, linears) = tiny_arch();
    let batch = 1usize;
    let past_len = 2usize;
    let position = past_len;

    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let past_elems = batch * past_len * nkv * hd;
    let past_k_b: Vec<u8> = vec![0.01f32; past_elems]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let past_v_b: Vec<u8> = vec![0.02f32; past_elems]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let token_bytes = 3f32.to_le_bytes().to_vec();

    let baked =
        build_llama_like_decode("baked", &arch, &linears, batch, past_len, position, Some(1))
            .unwrap();
    let mut cb = Session::new(Device::Cpu).compile(baked);
    bind_common(&mut cb, &arch, &linears);
    let out_baked = cb.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::F32),
        ("past_k_0", past_k_b.as_slice(), rlx_ir::DType::F32),
        ("past_v_0", past_v_b.as_slice(), rlx_ir::DType::F32),
    ]);

    let dynamic =
        build_llama_like_decode_dyn("dyn", &arch, &linears, batch, past_len, Some(1)).unwrap();
    let mut cd = Session::new(Device::Cpu).compile(dynamic);
    bind_common(&mut cd, &arch, &linears);
    let (cos, sin) = decode_rope_row(&arch, position);
    let cos_b: Vec<u8> = cos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let sin_b: Vec<u8> = sin.iter().flat_map(|x| x.to_le_bytes()).collect();
    let out_dyn = cd.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::F32),
        ("rope_cos", cos_b.as_slice(), rlx_ir::DType::F32),
        ("rope_sin", sin_b.as_slice(), rlx_ir::DType::F32),
        ("past_k_0", past_k_b.as_slice(), rlx_ir::DType::F32),
        ("past_v_0", past_v_b.as_slice(), rlx_ir::DType::F32),
    ]);

    assert_eq!(out_baked.len(), out_dyn.len());
    for (i, (a, b)) in out_baked.iter().zip(out_dyn.iter()).enumerate() {
        assert_eq!(a.0.len(), b.0.len(), "output {i} length");
        let fa: Vec<f32> =
            a.0.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        let fb: Vec<f32> =
            b.0.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        for (j, (x, y)) in fa.iter().zip(fb.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5,
                "output {i} element {j}: baked {x} vs dyn {y}"
            );
        }
    }
}

/// A padded cache plus a keep-mask must give the same logits as an
/// exact-length cache. This is what lets a generation loop compile once per
/// bucket instead of once per token, so the equivalence is load-bearing.
#[test]
fn masked_padded_decode_matches_exact_length() {
    use rlx_mlx_io::{build_llama_like_decode_masked, decode_keep_mask, decode_rope_row};

    let (arch, linears) = tiny_arch();
    let batch = 1usize;
    let real_past = 2usize;
    let capacity = 5usize; // bucket larger than the real cache
    let position = real_past;

    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let (cos, sin) = decode_rope_row(&arch, position);
    let cos_b: Vec<u8> = cos.iter().flat_map(|x| x.to_le_bytes()).collect();
    let sin_b: Vec<u8> = sin.iter().flat_map(|x| x.to_le_bytes()).collect();
    let token_bytes = 3f32.to_le_bytes().to_vec();

    // Distinct values per row so padding cannot be mistaken for real data.
    let row = |i: usize, base: f32| vec![base + i as f32 * 0.1; nkv * hd];
    let exact_k: Vec<f32> = (0..real_past).flat_map(|i| row(i, 0.01)).collect();
    let exact_v: Vec<f32> = (0..real_past).flat_map(|i| row(i, 0.02)).collect();

    // Padding filled with values that would visibly corrupt the result if
    // the mask were ignored.
    let mut padded_k = exact_k.clone();
    let mut padded_v = exact_v.clone();
    for _ in real_past..capacity {
        padded_k.extend(std::iter::repeat_n(9.0f32, nkv * hd));
        padded_v.extend(std::iter::repeat_n(-9.0f32, nkv * hd));
    }

    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

    // Reference: exact-length cache, no mask.
    let g_exact = rlx_mlx_io::build_llama_like_decode_dyn(
        "exact",
        &arch,
        &linears,
        batch,
        real_past,
        Some(1),
    )
    .unwrap();
    let mut c_exact = Session::new(Device::Cpu).compile(g_exact);
    bind_common(&mut c_exact, &arch, &linears);
    let out_exact = c_exact.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::F32),
        ("rope_cos", cos_b.as_slice(), rlx_ir::DType::F32),
        ("rope_sin", sin_b.as_slice(), rlx_ir::DType::F32),
        ("past_k_0", bytes(&exact_k).as_slice(), rlx_ir::DType::F32),
        ("past_v_0", bytes(&exact_v).as_slice(), rlx_ir::DType::F32),
    ]);

    // Under test: padded cache + keep-mask.
    let g_mask =
        build_llama_like_decode_masked("masked", &arch, &linears, batch, capacity, Some(1))
            .unwrap();
    let mut c_mask = Session::new(Device::Cpu).compile(g_mask);
    bind_common(&mut c_mask, &arch, &linears);
    let mask = decode_keep_mask(real_past, capacity);
    let out_mask = c_mask.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::F32),
        ("rope_cos", cos_b.as_slice(), rlx_ir::DType::F32),
        ("rope_sin", sin_b.as_slice(), rlx_ir::DType::F32),
        ("mask", bytes(&mask).as_slice(), rlx_ir::DType::F32),
        ("past_k_0", bytes(&padded_k).as_slice(), rlx_ir::DType::F32),
        ("past_v_0", bytes(&padded_v).as_slice(), rlx_ir::DType::F32),
    ]);

    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    // Only the logits are comparable: the KV outputs differ in length by
    // construction, since one is padded.
    let a = f32s(&out_exact[0].0);
    let b = f32s(&out_mask[0].0);
    assert_eq!(a.len(), b.len(), "logit count");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-4,
            "logit {i}: exact {x} vs masked {y} — padding leaked through the mask"
        );
    }
}

/// The prefill-with-KV graph must emit per-layer K/V alongside the logits,
/// and its logits must match the plain prefill graph.
#[test]
fn prefill_with_kv_matches_plain_prefill_and_emits_cache() {
    use rlx_mlx_io::{build_llama_like_prefill, build_llama_like_prefill_kv};

    let (arch, linears) = tiny_arch();
    let batch = 1usize;
    let seq = 3usize;
    let tokens: Vec<u8> = [1f32, 2.0, 3.0]
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .collect();

    let g_plain = build_llama_like_prefill("plain", &arch, &linears, batch, seq, Some(1)).unwrap();
    let mut c_plain = Session::new(Device::Cpu).compile(g_plain);
    bind_common(&mut c_plain, &arch, &linears);
    let out_plain = c_plain.run_typed(&[("tokens", tokens.as_slice(), rlx_ir::DType::F32)]);

    let g_kv = build_llama_like_prefill_kv("kv", &arch, &linears, batch, seq, Some(1)).unwrap();
    let mut c_kv = Session::new(Device::Cpu).compile(g_kv);
    bind_common(&mut c_kv, &arch, &linears);
    let out_kv = c_kv.run_typed(&[("tokens", tokens.as_slice(), rlx_ir::DType::F32)]);

    // logits + k_0 + v_0 for the single layer under test.
    assert_eq!(out_kv.len(), 3, "expected logits plus one layer's K and V");
    assert_eq!(out_plain[0].0, out_kv[0].0, "logits must be unchanged");

    let kv_elems = batch * seq * arch.num_key_value_heads * arch.head_dim();
    assert_eq!(out_kv[1].0.len() / 4, kv_elems, "K shape");
    assert_eq!(out_kv[2].0.len() / 4, kv_elems, "V shape");
}
