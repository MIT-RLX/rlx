// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end MNIST: run the embedded test image through the full
//! INT8 forward pass and check we predict the correct digit.

use rlx_cortexm::model::{SCRATCH_LEN, infer};
use rlx_cortexm::model_weights::{TEST_IMAGE, TEST_LABEL};

#[test]
fn predicts_test_image_correctly() {
    let mut a = vec![0i8; SCRATCH_LEN];
    let mut b = vec![0i8; SCRATCH_LEN];
    let pred = infer(TEST_IMAGE, &mut a, &mut b);
    assert_eq!(
        pred as u8, TEST_LABEL,
        "predicted {pred} but the label is {TEST_LABEL}"
    );
}
