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
