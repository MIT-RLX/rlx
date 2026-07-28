// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use rlx_ir::Shape;
use rlx_onnx_import::layout;

#[test]
fn expand_output_dims_matches_onnx_broadcast() {
    assert_eq!(
        layout::expand_output_dims(&[1, 128], &[8, 1, 1]),
        Some(vec![8, 1, 128])
    );
}

#[test]
fn bidir_merge_skips_fc_split_rank3() {
    let fc = Shape::new(&[1, 1024, 1], rlx_ir::DType::F32);
    assert!(layout::bidir_lstm_merge_reshape_dims(&fc).is_none());
}

#[test]
fn prefer_seq_first_keeps_static_meta_when_channels_differ() {
    let eval = Shape::new(&[8, 1, 128], rlx_ir::DType::F32);
    let meta = Shape::new(&[1, 8, 640], rlx_ir::DType::F32);
    let out = layout::prefer_seq_first_expand_target(&eval, &meta);
    assert_eq!(out.dims(), meta.dims());
}
