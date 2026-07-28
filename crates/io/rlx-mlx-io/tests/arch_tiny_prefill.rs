// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tiny Llama-like mlx-lm graph builds and runs on CPU.

use rlx_ir::quant::QuantScheme;
use rlx_mlx_io::{
    MlxArchConfig, MlxPackedLinear, PackedLinearBinding, build_llama_like_prefill,
    param_bindings_for,
};
use rlx_runtime::{Device, Session};

fn affine_pack(n: usize, k: usize) -> MlxPackedLinear {
    let gs = 32usize;
    assert!(k.is_multiple_of(gs));
    let n_groups = k / gs;
    let w_q: Vec<u8> = (0..n * (k / 2))
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let scales: Vec<u8> = (0..n * n_groups)
        .flat_map(|i| {
            let s = 0.02f32 + 0.001 * (i % 7) as f32;
            s.to_le_bytes()
        })
        .collect();
    let biases: Vec<u8> = (0..n * n_groups)
        .flat_map(|i| {
            let b = -0.01f32 + 0.001 * (i % 5) as f32;
            b.to_le_bytes()
        })
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

#[test]
fn tiny_llama_prefill_cpu() {
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
    let batch = 1usize;
    let seq = 2usize;
    let g = build_llama_like_prefill("tiny_llama", &arch, &linears, batch, seq, Some(1)).unwrap();

    let mut c = Session::new(Device::Cpu).compile(g);
    for b in &linears {
        for (name, bytes, dt) in param_bindings_for(b) {
            c.set_param_typed(&name, &bytes, dt);
        }
    }
    // Dense params (embed / norms / lm_head) — zeros are fine for a shape check.
    let emb = vec![0.01f32; arch.vocab_size * h];
    let emb_b: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
    c.set_param_typed("model.embed_tokens.weight", &emb_b, rlx_ir::DType::F32);
    let ones = vec![1.0f32; h];
    let ones_b: Vec<u8> = ones.iter().flat_map(|x| x.to_le_bytes()).collect();
    let zeros = vec![0.0f32; h];
    let zeros_b: Vec<u8> = zeros.iter().flat_map(|x| x.to_le_bytes()).collect();
    for name in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ] {
        c.set_param_typed(name, &ones_b, rlx_ir::DType::F32);
    }
    for name in [
        "model.layers.0.input_layernorm.bias_zero",
        "model.layers.0.post_attention_layernorm.bias_zero",
        "model.norm.bias_zero",
    ] {
        c.set_param_typed(name, &zeros_b, rlx_ir::DType::F32);
    }
    let head = vec![0.02f32; arch.vocab_size * h];
    let head_b: Vec<u8> = head.iter().flat_map(|x| x.to_le_bytes()).collect();
    c.set_param_typed("lm_head.weight", &head_b, rlx_ir::DType::F32);

    let tokens_i32: Vec<i32> = (0..batch * seq)
        .map(|i| (i % arch.vocab_size) as i32)
        .collect();
    let token_bytes: Vec<u8> = tokens_i32.iter().flat_map(|t| t.to_le_bytes()).collect();
    let out = c
        .run_typed(&[("tokens", token_bytes.as_slice(), rlx_ir::DType::I32)])
        .remove(0);
    assert_eq!(out.1, rlx_ir::DType::F32);
    let logits: Vec<f32> = out
        .0
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(logits.len(), batch * seq * arch.vocab_size);
    assert!(logits.iter().any(|v| v.is_finite()));
}
