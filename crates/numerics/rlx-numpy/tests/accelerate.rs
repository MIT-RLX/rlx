// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `accelerate::sgemm_nn` against Apple Accelerate's `sgemm_` itself
//! (`parity/export_accelerate_sgemm.py`): every measured regime, bit for bit.

#[test]
fn sgemm_model_matches_accelerate() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/accelerate_sgemm.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let f32s = |x: &serde_json::Value| -> Vec<f32> {
        x.as_array()
            .unwrap()
            .iter()
            .map(|u| f32::from_bits(u.as_u64().unwrap() as u32))
            .collect()
    };
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() > 100);
    for c in cases {
        let (m, n) = (
            c["m"].as_u64().unwrap() as usize,
            c["n"].as_u64().unwrap() as usize,
        );
        let beta = c["beta"].as_f64().unwrap() as f32;
        let (a, b, want) = (f32s(&c["a"]), f32s(&c["b"]), f32s(&c["c"]));
        let mut got = vec![0.0f32; m * n];
        rlx_numpy::accelerate::sgemm_nn(m, n, 9, 1.0, &a, m, &b, 9, beta, &mut got, m);
        let bad = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(bad, 0, "m{m} n{n} beta{beta}: {bad} of {} differ", m * n);
    }
}

#[test]
#[should_panic(expected = "modelled for k = 9")]
fn refuses_unmeasured_k() {
    let mut c = vec![0.0f32; 4];
    rlx_numpy::accelerate::sgemm_nn(2, 2, 3, 1.0, &[0.0; 6], 2, &[0.0; 6], 3, 0.0, &mut c, 2);
}
