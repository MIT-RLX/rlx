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
