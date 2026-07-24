//! keep-packed + arch config embeds Llama prefill graph (not parallel dequant).

use std::collections::HashMap;
use std::fs;

use rlx_ir::Op;
use rlx_pkg::{MlxImportOptions, mlx_to_rlxp};
use safetensors::tensor::{Dtype, TensorView};

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn import_mlx_keep_packed_embeds_llama_graph() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{
  "model_type": "llama",
  "vocab_size": 32,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "max_position_embeddings": 128,
  "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}
}"#,
    )
    .unwrap();

    let h = 64usize;
    let nh = 4usize;
    let nkv = 2usize;
    let hd = h / nh;
    let inter = 128usize;
    let gs = 32usize;

    let mut owned: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();
    let mut push = |name: &str, n: usize, k: usize| {
        let ng = k / gs;
        let w: Vec<u8> = (0..n * (k / 2)).map(|i| ((i * 3) % 256) as u8).collect();
        let s = f32_bytes(
            &(0..n * ng)
                .map(|i| 0.01 * (i + 1) as f32)
                .collect::<Vec<_>>(),
        );
        let b = f32_bytes(&vec![0.0f32; n * ng]);
        let wlen = w.len();
        owned.push((format!("{name}.weight"), w, vec![wlen], Dtype::U8));
        owned.push((format!("{name}.scales"), s, vec![n, ng], Dtype::F32));
        owned.push((format!("{name}.biases"), b, vec![n, ng], Dtype::F32));
    };
    let p = "model.layers.0";
    push(&format!("{p}.self_attn.q_proj"), nh * hd, h);
    push(&format!("{p}.self_attn.k_proj"), nkv * hd, h);
    push(&format!("{p}.self_attn.v_proj"), nkv * hd, h);
    push(&format!("{p}.self_attn.o_proj"), h, nh * hd);
    push(&format!("{p}.mlp.gate_proj"), inter, h);
    push(&format!("{p}.mlp.up_proj"), inter, h);
    push(&format!("{p}.mlp.down_proj"), h, inter);

    let emb = f32_bytes(&vec![0.01f32; 32 * h]);
    owned.push((
        "model.embed_tokens.weight".into(),
        emb,
        vec![32, h],
        Dtype::F32,
    ));

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, data, shape, dt) in &owned {
        tensors.insert(
            name.clone(),
            TensorView::new(*dt, shape.clone(), data).unwrap(),
        );
    }
    safetensors::serialize_to_file(&tensors, None, &dir.path().join("model.safetensors")).unwrap();

    let out = dir.path().join("arch.rlxp");
    let opts = MlxImportOptions {
        dequant_to_f32: false,
        include_graph: true,
        graph_batch: 1,
        graph_seq: 2,
        graph_num_layers: Some(1),
        ..Default::default()
    };
    mlx_to_rlxp(dir.path(), &out, &opts).unwrap();

    let pkg = rlx_pkg::Package::open(&out).unwrap();
    let g = pkg.graph().unwrap();
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Attention { .. })),
        "expected Attention in keep-packed arch graph"
    );
    assert!(
        g.nodes().iter().any(|n| matches!(n.op, Op::Rope { .. })),
        "expected Rope in keep-packed arch graph"
    );
}
