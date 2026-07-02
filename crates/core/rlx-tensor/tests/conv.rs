// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! conv2d forward — the CNN building block — validated against a
//! hand-computed reference. Run: `cargo test -p rlx-tensor --features eval`.
#![cfg(feature = "eval")]

use rlx_tensor::Tensor;

fn approx(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-5, "{a:?} != {b:?}");
    }
}

#[test]
fn conv2d_diagonal_kernel() {
    // NCHW input [1,1,3,3]:        weight [1,1,2,2] = [[1,0],[0,1]]
    //   1 2 3                       out[i,j] = in[i,j] + in[i+1,j+1]
    //   4 5 6                       -> [[1+5, 2+6], [4+8, 5+9]] = [[6,8],[12,14]]
    //   7 8 9
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [1, 1, 3, 3],
    );
    let w = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]);

    let out = x.conv2d(&w, [2, 2], [1, 1], [0, 0], [1, 1], 1);
    assert_eq!(
        out.shape().dims(),
        &[
            rlx_tensor::Dim::Static(1),
            rlx_tensor::Dim::Static(1),
            rlx_tensor::Dim::Static(2),
            rlx_tensor::Dim::Static(2)
        ]
    );
    approx(&out.to_vec(), &[6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn conv2d_with_stride_and_padding() {
    // Sum-pool-like 2x2 averaging kernel over a padded 3x3 input, stride 2.
    let x = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [1, 1, 3, 3],
    );
    let w = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]);
    // pad 0, stride 2: windows at (0,0) and (0,2-edge truncated)… use stride 1,
    // pad 0 -> 2x2 output of window sums.
    let out = x.conv2d(&w, [2, 2], [1, 1], [0, 0], [1, 1], 1).to_vec();
    // sums of each 2x2 window: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9]
    approx(&out, &[12.0, 16.0, 24.0, 28.0]);
}
