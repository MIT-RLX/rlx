// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `conv_output` against torch 2.10's own NNPACK (`parity/export_torch_nnpack.py`).

use serde_json::Value;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|h| f32::from_bits(u32::from_str_radix(h.as_str().unwrap(), 16).unwrap()))
        .collect()
}

#[test]
fn matches_torch_nnpack_bit_for_bit() {
    let p =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/torch_nnpack.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 9);
    for c in cases {
        let u = |k: &str| c[k].as_u64().unwrap() as usize;
        let pad: Vec<usize> = c["pad"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let (input, kernel, bias, want) = (
            floats(&c["input"]),
            floats(&c["kernel"]),
            floats(&c["bias"]),
            floats(&c["output"]),
        );
        let got = rlx_nnpack::conv_output(
            u("batch"),
            u("in_ch"),
            u("out_ch"),
            u("in_h"),
            u("in_w"),
            [pad[0], pad[1], pad[2], pad[3]],
            u("k_h"),
            u("k_w"),
            &input,
            &kernel,
            &bias,
        )
        .unwrap();
        assert_eq!(got.len(), want.len());
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            bad,
            0,
            "b{} ic{} oc{} {}x{} k{}x{}: {bad} of {} outputs differ",
            u("batch"),
            u("in_ch"),
            u("out_ch"),
            u("in_h"),
            u("in_w"),
            u("k_h"),
            u("k_w"),
            want.len()
        );
    }
}

#[test]
fn refuses_what_nnpack_refuses() {
    use rlx_nnpack::status::*;
    let z = vec![0.0f32; 4096];
    let call = |b, ic, oc, h, w, pad, kh, kw| {
        rlx_nnpack::conv_output(b, ic, oc, h, w, pad, kh, kw, &z, &z, &z).err()
    };
    assert_eq!(call(0, 1, 1, 4, 20, [0; 4], 1, 9), Some(INVALID_BATCH_SIZE));
    assert_eq!(
        call(1, 0, 1, 4, 20, [0; 4], 1, 9),
        Some(INVALID_INPUT_CHANNELS)
    );
    assert_eq!(
        call(1, 1, 0, 4, 20, [0; 4], 1, 9),
        Some(INVALID_OUTPUT_CHANNELS)
    );
    assert_eq!(call(1, 1, 1, 0, 20, [0; 4], 1, 9), Some(INVALID_INPUT_SIZE));
    assert_eq!(
        call(1, 1, 1, 4, 20, [0, 9, 0, 0], 1, 9),
        Some(INVALID_INPUT_PADDING),
        "padding must be below the kernel width"
    );
    assert_eq!(
        call(1, 1, 1, 4, 20, [1, 0, 0, 0], 1, 9),
        Some(INVALID_INPUT_PADDING),
        "and below its height"
    );
    assert_eq!(
        call(1, 1, 1, 4, 8, [0; 4], 3, 5),
        Some(UNSUPPORTED_ALGORITHM),
        "auto picks ft8x8/wt8x8 here"
    );
    assert_eq!(
        call(1, 1, 1, 4, 40, [0; 4], 1, 17),
        Some(UNSUPPORTED_ALGORITHM)
    );
    assert!(call(1, 1, 1, 1, 40, [0; 4], 1, 9).is_none());
}
