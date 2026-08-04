// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural tests for the NeMo Conformer encoder graph builder. These use a
//! *synthetic* state dict (name → shape) so the architecture-to-graph mapping
//! is exercised end-to-end without a multi-gigabyte `.nemo` on disk. Numerical
//! parity against a reference NeMo forward pass is a separate concern (needs a
//! bundled checkpoint + expected output); here we assert the graph is complete
//! and self-consistent under the IR's independent shape inference.

use std::collections::BTreeMap;

use rlx_ir::{Op, verify_all};
use rlx_nemo::{EncoderOpts, NemoConfig, build_nemo_encoder_graph};

/// Build a synthetic Conformer state dict (`name → shape`) for the given
/// geometry. Mirrors the real NeMo `ConformerEncoder` parameter names.
fn synth_conformer(
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    d_ff: usize,
    conv_kernel: usize,
    batch_norm: bool,
) -> BTreeMap<String, Vec<usize>> {
    let dk = d_model / n_heads;
    let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut put = |k: String, s: Vec<usize>| {
        m.insert(k, s);
    };
    for i in 0..n_layers {
        let p = format!("encoder.layers.{i}");
        for ffn in ["feed_forward1", "feed_forward2"] {
            put(format!("{p}.{ffn}.linear1.weight"), vec![d_ff, d_model]);
            put(format!("{p}.{ffn}.linear1.bias"), vec![d_ff]);
            put(format!("{p}.{ffn}.linear2.weight"), vec![d_model, d_ff]);
            put(format!("{p}.{ffn}.linear2.bias"), vec![d_model]);
        }
        for norm in [
            "norm_feed_forward1",
            "norm_feed_forward2",
            "norm_self_att",
            "norm_conv",
            "norm_out",
        ] {
            put(format!("{p}.{norm}.weight"), vec![d_model]);
            put(format!("{p}.{norm}.bias"), vec![d_model]);
        }
        for lin in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            put(
                format!("{p}.self_attn.{lin}.weight"),
                vec![d_model, d_model],
            );
            put(format!("{p}.self_attn.{lin}.bias"), vec![d_model]);
        }
        put(
            format!("{p}.self_attn.linear_pos.weight"),
            vec![d_model, d_model],
        );
        put(format!("{p}.self_attn.pos_bias_u"), vec![n_heads, dk]);
        put(format!("{p}.self_attn.pos_bias_v"), vec![n_heads, dk]);

        put(
            format!("{p}.conv.pointwise_conv1.weight"),
            vec![2 * d_model, d_model, 1],
        );
        put(format!("{p}.conv.pointwise_conv1.bias"), vec![2 * d_model]);
        put(
            format!("{p}.conv.depthwise_conv.weight"),
            vec![d_model, 1, conv_kernel],
        );
        put(format!("{p}.conv.depthwise_conv.bias"), vec![d_model]);
        put(
            format!("{p}.conv.pointwise_conv2.weight"),
            vec![d_model, d_model, 1],
        );
        put(format!("{p}.conv.pointwise_conv2.bias"), vec![d_model]);
        put(format!("{p}.conv.batch_norm.weight"), vec![d_model]);
        put(format!("{p}.conv.batch_norm.bias"), vec![d_model]);
        if batch_norm {
            put(format!("{p}.conv.batch_norm.running_mean"), vec![d_model]);
            put(format!("{p}.conv.batch_norm.running_var"), vec![d_model]);
        }
    }
    m
}

/// Add a `dw_striding` `pre_encode` front-end (subsampling factor 4, N=2) to a
/// state dict: input conv `1→C`, then a depthwise + pointwise block, then the
/// `out` Linear `C·F' → d_model`.
fn add_dw_striding(
    m: &mut BTreeMap<String, Vec<usize>>,
    d_model: usize,
    conv_channels: usize,
    feat_in: usize,
) {
    let c = conv_channels;
    // conv.0: Conv2d(1 → C, 3×3, stride 2)
    m.insert("pre_encode.conv.0.weight".into(), vec![c, 1, 3, 3]);
    m.insert("pre_encode.conv.0.bias".into(), vec![c]);
    // conv.2: depthwise Conv2d(C → C, 3×3, stride 2, groups=C)
    m.insert("pre_encode.conv.2.weight".into(), vec![c, 1, 3, 3]);
    m.insert("pre_encode.conv.2.bias".into(), vec![c]);
    // conv.3: pointwise Conv2d(C → C, 1×1)
    m.insert("pre_encode.conv.3.weight".into(), vec![c, c, 1, 1]);
    m.insert("pre_encode.conv.3.bias".into(), vec![c]);
    // out Linear: C · calc_length(feat_in, N=2) → d_model.
    // calc_length: L → floor((L-1)/2)+1, twice.
    let l1 = (feat_in - 1) / 2 + 1;
    let f_out = (l1 - 1) / 2 + 1;
    m.insert("pre_encode.out.weight".into(), vec![d_model, c * f_out]);
    m.insert("pre_encode.out.bias".into(), vec![d_model]);
}

fn cfg(d_model: usize, n_layers: usize, n_heads: usize) -> NemoConfig {
    let yaml =
        format!("encoder:\n  d_model: {d_model}\n  n_layers: {n_layers}\n  n_heads: {n_heads}\n");
    NemoConfig::from_yaml_bytes(yaml.as_bytes()).unwrap()
}

#[test]
fn builds_full_conformer_and_verifies() {
    let (d, l, h, ff, k) = (64, 3, 4, 256, 9);
    let shapes = synth_conformer(d, l, h, ff, k, true);
    let cfg = cfg(d, l, h);
    let opts = EncoderOpts {
        name: "test_conformer".into(),
        batch: 2,
        seq_len: 16,
        ..EncoderOpts::default()
    };

    let g = build_nemo_encoder_graph(&cfg, &shapes, &opts)
        .expect("build ok")
        .expect("recognized as conformer");

    // The IR's independent structural + shape inference must accept the graph.
    let errs = verify_all(&g);
    assert!(errs.is_empty(), "verify_all errors: {errs:?}");

    // Output is the encoder hidden state [B, T, D].
    let out = *g.outputs.last().unwrap();
    let got: Vec<usize> = g
        .shape(out)
        .dims()
        .iter()
        .map(|x| x.unwrap_static())
        .collect();
    assert_eq!(got, vec![opts.batch, opts.seq_len, d]);

    // Every Param binds a real weight name from the checkpoint.
    for n in g.nodes() {
        if let Op::Param { name } = &n.op {
            assert!(
                shapes.contains_key(name),
                "graph binds unknown weight {name}"
            );
        }
    }

    // Structural landmarks of the Conformer path are present per layer.
    let count = |pred: &dyn Fn(&Op) -> bool| g.nodes().iter().filter(|n| pred(&n.op)).count();
    let convs = count(&|o| matches!(o, Op::Conv { .. }));
    let bns = count(&|o| matches!(o, Op::BatchNormInference { .. }));
    let pads = count(&|o| matches!(o, Op::Pad { .. }));
    let softmaxes = count(&|o| matches!(o, Op::Softmax { .. }));
    assert_eq!(convs, l, "one depthwise conv per layer");
    assert_eq!(bns, l, "one batch-norm per layer");
    assert_eq!(pads, l, "one rel_shift pad per layer");
    assert_eq!(softmaxes, l, "one attention softmax per layer");
}

#[test]
fn layer_norm_conv_variant_selected_without_running_stats() {
    let (d, l, h, ff, k) = (32, 1, 2, 128, 5);
    let shapes = synth_conformer(d, l, h, ff, k, /* batch_norm = */ false);
    let cfg = cfg(d, l, h);
    let opts = EncoderOpts {
        seq_len: 8,
        ..EncoderOpts::default()
    };

    let g = build_nemo_encoder_graph(&cfg, &shapes, &opts)
        .unwrap()
        .expect("conformer");
    assert!(verify_all(&g).is_empty());

    let bns = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::BatchNormInference { .. }))
        .count();
    let lns = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::LayerNorm { .. }))
        .count();
    assert_eq!(bns, 0, "no batch-norm when running stats absent");
    // 5 block LayerNorms + the conv-module LayerNorm = 6 for a single layer.
    assert_eq!(lns, 6, "conv module falls back to LayerNorm");
}

#[test]
fn dims_recovered_from_weights_without_config() {
    // Empty-ish config: geometry must be recovered from weight shapes.
    let (d, l, h, ff, k) = (48, 2, 6, 192, 7);
    let shapes = synth_conformer(d, l, h, ff, k, true);
    let empty = NemoConfig::from_yaml_bytes(b"sample_rate: 16000\n").unwrap();
    let opts = EncoderOpts {
        seq_len: 12,
        ..EncoderOpts::default()
    };

    let g = build_nemo_encoder_graph(&empty, &shapes, &opts)
        .unwrap()
        .expect("conformer recovered from weights");
    assert!(verify_all(&g).is_empty());
    // n_layers recovered by counting norm_out → one softmax per layer.
    let softmaxes = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Softmax { .. }))
        .count();
    assert_eq!(softmaxes, l);
}

#[test]
fn builds_encoder_with_conv_subsampling_frontend() {
    let (d, l, h, ff, k) = (64, 2, 4, 256, 9);
    let conv_channels = 32;
    let feat_in = 80;
    let mut shapes = synth_conformer(d, l, h, ff, k, true);
    add_dw_striding(&mut shapes, d, conv_channels, feat_in);

    // Config carries feat_in so the front-end can size the mel input.
    let yaml = format!(
        "preprocessor:\n  features: {feat_in}\nencoder:\n  d_model: {d}\n  n_layers: {l}\n  n_heads: {h}\n"
    );
    let cfg = NemoConfig::from_yaml_bytes(yaml.as_bytes()).unwrap();

    let mel_frames = 200;
    let opts = EncoderOpts {
        batch: 1,
        mel_frames: Some(mel_frames),
        ..EncoderOpts::default()
    };
    let g = build_nemo_encoder_graph(&cfg, &shapes, &opts)
        .unwrap()
        .expect("conformer with subsampling");
    assert!(verify_all(&g).is_empty(), "{:?}", verify_all(&g));

    // Input is mel features [B, frames, feat_in].
    let inp = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Input { .. }))
        .unwrap();
    let in_dims: Vec<usize> = g
        .shape(inp.id)
        .dims()
        .iter()
        .map(|x| x.unwrap_static())
        .collect();
    assert_eq!(in_dims, vec![1, mel_frames, feat_in]);

    // Output length T = calc_length(frames, N=2) = floor floor.
    let t = {
        let l1 = (mel_frames - 1) / 2 + 1;
        (l1 - 1) / 2 + 1
    };
    let out = *g.outputs.last().unwrap();
    let out_dims: Vec<usize> = g
        .shape(out)
        .dims()
        .iter()
        .map(|x| x.unwrap_static())
        .collect();
    assert_eq!(out_dims, vec![1, t, d]);

    // Three subsampling convs (input + depthwise + pointwise) plus the depthwise
    // conv inside each of the l layers.
    let convs = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Conv { .. }))
        .count();
    assert_eq!(convs, 3 + l);
}

#[test]
fn non_conformer_checkpoint_returns_none() {
    let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    shapes.insert("decoder.embedding.weight".into(), vec![1000, 512]);
    let cfg = NemoConfig::from_yaml_bytes(b"sample_rate: 22050\n").unwrap();
    let g = build_nemo_encoder_graph(&cfg, &shapes, &EncoderOpts::default()).unwrap();
    assert!(g.is_none(), "not a conformer → None");
}
