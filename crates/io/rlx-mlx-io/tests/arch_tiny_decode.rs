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
    let token_bytes = 3i32.to_le_bytes().to_vec();
    let outs = c.run_typed(&[
        ("token", token_bytes.as_slice(), rlx_ir::DType::I32),
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
