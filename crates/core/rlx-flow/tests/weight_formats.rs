// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The file-backed `WeightSource`s (`--features formats`): safetensors
//! dtypes and shards, and a GGUF tensor served as F32 and as packed blocks —
//! through a real flow, where both must give the same projection.
#![cfg(feature = "formats")]

use rlx_flow::formats::{GgufSource, SafetensorsSource};
use rlx_flow::{ModelFlow, WeightSource, WeightSourceExt};
use rlx_gguf::{GgmlType, GgufWriter};
use rlx_ir::{DType, Shape};
use rlx_runtime::{Device, Session};

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rlx-flow-formats-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A safetensors file from `(name, dtype, shape, bytes)`.
fn write_st(path: &std::path::Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|&x| f(x)).collect()
}

#[test]
fn safetensors_dtypes_shards_and_errors() {
    let d = scratch("st");
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    write_st(
        &d.join("model-00001-of-00002.safetensors"),
        &[
            ("model.w", "F32", vec![2, 3], le(&w, f32::to_le_bytes)),
            (
                "model.h",
                "F16",
                vec![2],
                le(
                    &[half::f16::from_f32(0.5), half::f16::from_f32(-2.0)],
                    half::f16::to_le_bytes,
                ),
            ),
        ],
    );
    write_st(
        &d.join("model-00002-of-00002.safetensors"),
        &[(
            "model.b",
            "BF16",
            vec![1],
            le(&[half::bf16::from_f32(3.0)], half::bf16::to_le_bytes),
        )],
    );
    let mut src = SafetensorsSource::open(&d).unwrap();
    assert_eq!(src.keys(), ["model.b", "model.h", "model.w"]);
    assert_eq!(src.info("model.h"), Some(("F16", &[2usize][..])));
    // Non-consuming, transposed on request.
    for _ in 0..2 {
        assert_eq!(
            src.take("model.w", true).unwrap(),
            (vec![1., 4., 2., 5., 3., 6.], vec![3, 2])
        );
    }
    assert_eq!(src.take("model.h", false).unwrap().0, vec![0.5, -2.0]);
    assert_eq!(src.take("model.b", false).unwrap().0, vec![3.0]);
    // A miss names the key the checkpoint does have.
    let err = src.take("w", false).unwrap_err().to_string();
    assert!(err.contains("closest: model.w"), "{err}");
    let err = src.take("layers.model.w", false).unwrap_err().to_string();
    assert!(err.contains("closest: model.w"), "{err}");
    // Through a prefix, keys come back stripped.
    let p = SafetensorsSource::open(&d).unwrap().with_prefix("model.");
    assert_eq!(p.keys(), ["b", "h", "w"]);

    // The same name in two shards is refused.
    write_st(
        &d.join("model-00003.safetensors"),
        &[("model.b", "F32", vec![1], le(&[1f32], f32::to_le_bytes))],
    );
    assert!(SafetensorsSource::open(&d).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn gguf_packed_and_f32_give_the_same_projection() {
    let (k, n, b) = (64usize, 4usize, 2usize);
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.05)
        .collect();
    let d = scratch("gguf");
    let path = d.join("m.gguf");
    let mut gw = GgufWriter::new();
    // GGML order: innermost (in = k) first.
    gw.add_tensor_bytes(
        "blk.0.w",
        vec![k, n],
        GgmlType::Q8_0,
        rlx_gguf::quantize(&wf, GgmlType::Q8_0).unwrap(),
    )
    .unwrap();
    gw.write_to_path(&path).unwrap();

    let mut f32_src = GgufSource::open(&path).unwrap();
    let (data, shape) = f32_src.take("blk.0.w", false).unwrap();
    assert_eq!(shape, vec![n, k], "outermost-first like safetensors");
    assert!(
        f32_src.take_packed("blk.0.w").unwrap().is_none(),
        "packing is opt-in"
    );
    let mut packed_src = GgufSource::open(&path).unwrap().packed(true);
    let p = packed_src
        .take_packed("blk.0.w")
        .unwrap()
        .expect("Q8_0 packs");
    assert_eq!((p.in_dim, p.out_dim), (k, n));
    assert_eq!(p.scheme, rlx_ir::quant::QuantScheme::GgufQ8_0);

    let x: Vec<f32> = (0..b * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let run = |src: &mut dyn WeightSource| -> (Vec<f32>, bool) {
        // The builder asks for HF-style names; the file has GGUF ones.
        let mut src = src.renamed(|key| key.replace("layers.0.", "blk.0."));
        let built = ModelFlow::new("proj")
            .input("x", Shape::new(&[b, k], DType::F32))
            .linear("layers.0.w", true)
            .build(&mut src)
            .unwrap();
        let typed = built.typed_params.clone();
        let (g, params) = built.into_graph_parts().unwrap();
        let dequant = g
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, rlx_ir::Op::DequantMatMul { .. }));
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, v) in &params {
            c.set_param(name, v);
        }
        for (name, bytes, dt) in &typed {
            c.set_param_typed(name, bytes, *dt);
        }
        (c.run(&[("x", &x)]).remove(0), dequant)
    };
    let (y_f32, f32_dequant) = run(&mut f32_src);
    let (y_packed, packed_dequant) = run(&mut packed_src);
    assert!(
        !f32_dequant && packed_dequant,
        "packed weights must reach the graph"
    );
    // Reference: x · Wᵀ over the dequantized weights.
    for (j, (a, p)) in y_f32.iter().zip(&y_packed).enumerate() {
        let (r, c) = (j / n, j % n);
        let want: f32 = (0..k).map(|i| x[r * k + i] * data[c * k + i]).sum();
        assert!((a - want).abs() < 1e-4, "f32 [{j}] {a} vs {want}");
        assert!((p - want).abs() < 1e-3, "packed [{j}] {p} vs {want}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn open_picks_the_format_by_path() {
    let d = scratch("open");
    write_st(
        &d.join("a.safetensors"),
        &[("x", "F32", vec![1], le(&[7f32], f32::to_le_bytes))],
    );
    let mut s = rlx_flow::formats::open(d.join("a.safetensors")).unwrap();
    assert_eq!(s.take("x", false).unwrap().0, vec![7.0]);
    assert!(rlx_flow::formats::open(d.join("a.bin")).is_err());
    let _ = std::fs::remove_dir_all(&d);
}

/// FP8 safetensors (`F8_E4M3` / `F8_E5M2`, the dtypes HF FP8 checkpoints
/// use): every one of the 256 codes decodes like `rlx_ir`'s codec, and a few
/// codes match their OCP values written out by hand.
#[test]
fn safetensors_fp8_decodes_every_code() {
    use rlx_ir::quant::ScaledFormat;
    let d = scratch("fp8");
    let codes: Vec<u8> = (0..=255).collect();
    write_st(
        &d.join("fp8.safetensors"),
        &[
            ("e4m3", "F8_E4M3", vec![16, 16], codes.clone()),
            ("e5m2", "F8_E5M2", vec![256], codes.clone()),
        ],
    );
    let mut s = SafetensorsSource::open(d.join("fp8.safetensors")).unwrap();
    for (key, fmt) in [
        ("e4m3", ScaledFormat::F8E4M3),
        ("e5m2", ScaledFormat::F8E5M2),
    ] {
        let (got, shape) = s.take(key, false).unwrap();
        assert_eq!(shape.iter().product::<usize>(), 256);
        for (c, v) in got.iter().enumerate() {
            let want = fmt.decode(c as u8);
            assert!(
                v.to_bits() == want.to_bits() || (v.is_nan() && want.is_nan()),
                "{key} code {c:#04x}: {v} vs {want}"
            );
        }
    }
    let (e4, _) = s.take("e4m3", false).unwrap();
    assert_eq!(e4[0x38], 1.0);
    assert_eq!(e4[0x7e], 448.0, "E4M3 max finite");
    assert_eq!(e4[0xb8], -1.0);
    assert_eq!(e4[0x01], 2f32.powi(-9), "smallest subnormal");
    assert!(e4[0x7f].is_nan() && e4[0xff].is_nan());
    let (e5, _) = s.take("e5m2", false).unwrap();
    assert_eq!(e5[0x3c], 1.0);
    assert_eq!(e5[0x7b], 57344.0, "E5M2 max finite");
    assert_eq!(e5[0x7c], f32::INFINITY);
    assert!(e5[0x7d].is_nan());
    let _ = std::fs::remove_dir_all(&d);
}

/// An HF block-FP8 checkpoint (DeepSeek-V3 / Qwen3-FP8 layout): an
/// `F8_E4M3` weight plus a `weight_scale_inv` grid of 128×128 blocks. `take`
/// returns `decode(code) * scale[i/128][j/128]`, transposed when asked; the
/// block size comes from `config.json`; `fp8_scales(false)` gives raw codes.
#[test]
fn safetensors_fp8_applies_block_scales() {
    use rlx_ir::quant::ScaledFormat;
    let d = scratch("fp8-block");
    let (rows, cols) = (200usize, 130usize); // ragged 2×2 grid of 128-blocks
    let codes: Vec<u8> = (0..rows * cols).map(|k| (k % 0x7e) as u8).collect();
    let scale = [0.5f32, 2.0, 4.0, 0.25];
    write_st(
        &d.join("model.safetensors"),
        &[
            ("l.weight", "F8_E4M3", vec![rows, cols], codes.clone()),
            (
                "l.weight_scale_inv",
                "F32",
                vec![2, 2],
                le(&scale, f32::to_le_bytes),
            ),
        ],
    );
    let want = |i: usize, j: usize| {
        ScaledFormat::F8E4M3.decode(codes[i * cols + j]) * scale[(i / 128) * 2 + j / 128]
    };
    let mut s = SafetensorsSource::open(d.join("model.safetensors")).unwrap();
    let (w, shape) = s.take("l.weight", false).unwrap();
    assert_eq!(shape, vec![rows, cols]);
    let (wt, tshape) = s.take("l.weight", true).unwrap();
    assert_eq!(tshape, vec![cols, rows]);
    for i in 0..rows {
        for j in 0..cols {
            assert_eq!(w[i * cols + j], want(i, j), "({i}, {j})");
            assert_eq!(wt[j * rows + i], want(i, j), "transposed ({i}, {j})");
        }
    }

    let mut raw = SafetensorsSource::open(d.join("model.safetensors"))
        .unwrap()
        .fp8_scales(false);
    let (r, _) = raw.take("l.weight", false).unwrap();
    assert_eq!(r[cols + 1], ScaledFormat::F8E4M3.decode(codes[cols + 1]));

    // A 100×65 block from config.json makes the same 2×2 grid tile
    // differently; inference alone would pick 128.
    std::fs::write(
        d.join("config.json"),
        r#"{"quantization_config":{"quant_method":"fp8","weight_block_size":[100,65]}}"#,
    )
    .unwrap();
    let mut s = SafetensorsSource::open(&d).unwrap();
    let (w, _) = s.take("l.weight", false).unwrap();
    let (i, j) = (150, 70);
    assert_eq!(
        w[i * cols + j],
        ScaledFormat::F8E4M3.decode(codes[i * cols + j]) * scale[2 + 1]
    );
    let (i, j) = (110, 64);
    assert_eq!(
        w[i * cols + j],
        ScaledFormat::F8E4M3.decode(codes[i * cols + j]) * scale[2]
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// An FP8 checkpoint through a real flow under `WeightStorage::Auto`: the
/// weight lands as a BF16 param (2 B/weight, not f32), with and without
/// `half_bytes` (bytes straight from the source vs the f32 path the backend
/// narrows at bind), and both give `x · bf16(decode·scale)ᵀ`.
#[test]
fn fp8_weight_is_stored_bf16_through_a_flow() {
    use rlx_flow::WeightStorage;
    use rlx_ir::quant::ScaledFormat;
    let (k, n, b) = (256usize, 3usize, 2usize);
    let codes: Vec<u8> = (0..n * k)
        .map(|i| ((i * 7) % 0x7e) as u8 ^ ((i as u8 & 1) << 7))
        .collect();
    let scale = [0.37f32, 1.9]; // 1×2 grid of 128-blocks along k
    let d = scratch("fp8-flow");
    let path = d.join("m.safetensors");
    write_st(
        &path,
        &[
            ("w", "F8_E4M3", vec![n, k], codes.clone()),
            (
                "w_scale_inv",
                "F32",
                vec![1, 2],
                le(&scale, f32::to_le_bytes),
            ),
            (
                "h",
                "BF16",
                vec![2, 3],
                le(&[1.5f32, -2.0, 0.25, 3.0, -0.5, 8.0], |v| {
                    half::bf16::from_f32(v).to_le_bytes()
                }),
            ),
        ],
    );
    // bf16-rounded scaled weights, [n, k].
    let w: Vec<f32> = (0..n * k)
        .map(|i| {
            let v = ScaledFormat::F8E4M3.decode(codes[i]) * scale[(i % k) / 128];
            half::bf16::from_f32(v).to_f32()
        })
        .collect();

    let src = SafetensorsSource::open(&path).unwrap();
    assert_eq!(src.stored_dtype("w"), Some(DType::BF16));
    let mut plain = src;
    assert!(
        plain.take_half("w", false).unwrap().is_none(),
        "half_bytes is opt-in"
    );
    let mut halves = SafetensorsSource::open(&path).unwrap().half_bytes(true);

    // take_half: FP8 → scaled bf16 bytes (and transposed); BF16 → its bytes.
    let (hb, hs) = halves.take_half("w", true).unwrap().unwrap();
    assert_eq!(hs, vec![k, n]);
    for c in 0..k {
        for r in 0..n {
            let got = half::bf16::from_le_bytes([hb[2 * (c * n + r)], hb[2 * (c * n + r) + 1]]);
            assert_eq!(got.to_f32(), w[r * k + c], "wᵀ[{c}][{r}]");
        }
    }
    let (bb, _) = halves.take_half("h", false).unwrap().unwrap();
    assert_eq!(bb[..2], half::bf16::from_f32(1.5).to_le_bytes());

    let x: Vec<f32> = (0..b * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let run = |src: &mut dyn WeightSource, half_bytes: bool| -> Vec<f32> {
        let built = ModelFlow::new("proj")
            .with_weight_storage(WeightStorage::Auto)
            .input("x", Shape::new(&[b, k], DType::F32))
            .linear("w", true)
            .build(src)
            .unwrap();
        let typed = built.typed_params.clone();
        assert_eq!(
            typed
                .iter()
                .any(|(name, _, dt)| name == "w" && *dt == DType::BF16),
            half_bytes,
            "bf16 bytes come from the source only with half_bytes"
        );
        let (g, params) = built.into_graph_parts().unwrap();
        assert_eq!(params.contains_key("w"), !half_bytes);
        let wp = g
            .nodes()
            .iter()
            .find(|nd| matches!(&nd.op, rlx_ir::Op::Param { name } if name == "w"))
            .expect("w param");
        assert_eq!(wp.shape.dtype(), DType::BF16, "FP8 weight stored as bf16");
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, v) in &params {
            c.set_param(name, v);
        }
        for (name, bytes, dt) in &typed {
            c.set_param_typed(name, bytes, *dt);
        }
        c.run(&[("x", &x)]).remove(0)
    };
    for (label, y) in [
        ("f32 path", run(&mut plain, false)),
        ("half_bytes", run(&mut halves, true)),
    ] {
        for (j, got) in y.iter().enumerate() {
            let (r, c) = (j / n, j % n);
            let want: f32 = (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum();
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{label} [{j}] {got} vs {want}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// `fp8_packed`: an `F8_E4M3` weight with a 128×128 `weight_scale_inv` grid
/// reaches the graph as one `Fp8E4m3B128` `DequantMatMul` blob (1 B/weight +
/// scales) and gives the same projection as the unpacked f32 path. A scale
/// that changes inside a 128-element run along `in` stays unpacked.
#[test]
fn fp8_packed_weight_runs_as_dequant_matmul() {
    use rlx_ir::quant::{QuantScheme, ScaledFormat};
    let (k, n, b) = (384usize, 200usize, 3usize); // 2×3 grid, ragged rows
    let finite: Vec<u8> = (0..=255u8).filter(|c| c & 0x7f != 0x7f).collect();
    let codes: Vec<u8> = (0..n * k)
        .map(|i| finite[(i * 11) % finite.len()])
        .collect();
    let scale: Vec<f32> = (0..6).map(|t| 0.01 * (t as f32 + 1.5)).collect();
    let d = scratch("fp8-packed");
    let path = d.join("m.safetensors");
    write_st(
        &path,
        &[
            ("w", "F8_E4M3", vec![n, k], codes.clone()),
            (
                "w_scale_inv",
                "F32",
                vec![2, 3],
                le(&scale, f32::to_le_bytes),
            ),
        ],
    );
    let w: Vec<f32> = (0..n * k)
        .map(|t| {
            let (i, j) = (t / k, t % k);
            ScaledFormat::F8E4M3.decode(codes[t]) * scale[(i / 128) * 3 + j / 128]
        })
        .collect();

    let x: Vec<f32> = (0..b * k)
        .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.05)
        .collect();
    let run = |src: &mut dyn WeightSource| -> (Vec<f32>, Option<QuantScheme>, usize) {
        let built = ModelFlow::new("proj")
            .input("x", Shape::new(&[b, k], DType::F32))
            .linear("w", true)
            .build(src)
            .unwrap();
        let typed = built.typed_params.clone();
        let blob = typed.iter().map(|(_, bytes, _)| bytes.len()).sum();
        let (g, params) = built.into_graph_parts().unwrap();
        let scheme = g.nodes().iter().find_map(|nd| match nd.op {
            rlx_ir::Op::DequantMatMul { scheme } => Some(scheme),
            _ => None,
        });
        let mut c = Session::new(Device::Cpu).compile(g);
        for (name, v) in &params {
            c.set_param(name, v);
        }
        for (name, bytes, dt) in &typed {
            c.set_param_typed(name, bytes, *dt);
        }
        (c.run(&[("x", &x)]).remove(0), scheme, blob)
    };
    let mut packed = SafetensorsSource::open(&path).unwrap().fp8_packed(true);
    assert!(packed.has_packed("w"));
    let (yp, scheme, blob) = run(&mut packed);
    assert_eq!(scheme, Some(QuantScheme::Fp8E4m3B128));
    assert_eq!(blob, n * (k / 128) * 132, "1 B per weight + an f32 per 128");
    let (yf, unpacked, _) = run(&mut SafetensorsSource::open(&path).unwrap());
    assert_eq!(unpacked, None);
    for (j, (p, f)) in yp.iter().zip(&yf).enumerate() {
        let (r, c) = (j / n, j % n);
        let want: f32 = (0..k).map(|i| x[r * k + i] * w[c * k + i]).sum();
        let tol = 1e-4 * want.abs().max(1.0);
        assert!((p - want).abs() <= tol, "packed [{j}] {p} vs {want}");
        assert!((f - want).abs() <= tol, "f32 [{j}] {f} vs {want}");
    }

    // A 64-wide block along `in` can't be one scale per 128: served unpacked.
    let path2 = d.join("narrow.safetensors");
    write_st(
        &path2,
        &[
            ("w", "F8_E4M3", vec![n, k], codes.clone()),
            (
                "w_scale_inv",
                "F32",
                vec![2, 6],
                le(&[1.0f32; 12], f32::to_le_bytes),
            ),
        ],
    );
    let mut narrow = SafetensorsSource::open(&path2).unwrap().fp8_packed(true);
    assert!(!narrow.has_packed("w"));
    assert!(narrow.take_packed("w").unwrap().is_none());
    let _ = std::fs::remove_dir_all(&d);
}

/// `F8_E5M2` packs as `Fp8E5m2B128` (per-channel scale), and its scale tensor
/// is folded into the weight, so `keys()` no longer lists it.
#[test]
fn fp8_e5m2_packs_and_hides_its_scale() {
    use rlx_ir::quant::{QuantScheme, ScaledFormat};
    let (k, n) = (256usize, 6usize);
    let finite: Vec<u8> = (0..=255u8)
        .filter(|&c| ScaledFormat::F8E5M2.decode(c).is_finite())
        .collect();
    let codes: Vec<u8> = (0..n * k).map(|i| finite[(i * 5) % finite.len()]).collect();
    let scale: Vec<f32> = (0..n).map(|r| 1e-3 * (r as f32 + 1.0)).collect();
    let d = scratch("fp8-e5m2");
    let path = d.join("m.safetensors");
    write_st(
        &path,
        &[
            ("w", "F8_E5M2", vec![n, k], codes.clone()),
            ("w_scale", "F32", vec![n, 1], le(&scale, f32::to_le_bytes)),
            ("b", "F32", vec![2], le(&[1.0f32, 2.0], f32::to_le_bytes)),
        ],
    );
    let mut src = SafetensorsSource::open(&path).unwrap().fp8_packed(true);
    assert_eq!(src.keys(), vec!["b".to_string(), "w".to_string()]);
    let p = src.take_packed("w").unwrap().expect("E5M2 packs");
    assert_eq!(
        (p.scheme, p.in_dim, p.out_dim),
        (QuantScheme::Fp8E5m2B128, k, n)
    );
    let w = rlx_ir::lowp_codec::fp8_b128::dequant_scheme(p.scheme, &p.w_q, n * k).unwrap();
    let (f, _) = src.take("w", false).unwrap();
    assert_eq!(w, f, "packed and unpacked reads agree");
    for (t, v) in w.iter().enumerate() {
        assert_eq!(*v, ScaledFormat::F8E5M2.decode(codes[t]) * scale[t / k]);
    }
    let raw = SafetensorsSource::open(&path).unwrap().fp8_scales(false);
    assert!(
        raw.keys().contains(&"w_scale".to_string()),
        "raw mode keeps the scale visible"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// A 3-D `[E, out, in]` FP8 expert stack through `take_packed_bank`, with a
/// per-expert 128×128 grid and with one per-tensor scale: the packed bank
/// dequantizes to exactly what the unpacked read (`apply_weight_scales`) gives.
#[test]
fn fp8_expert_bank_packs_like_the_unpacked_read() {
    use rlx_ir::quant::QuantScheme;
    let (ex, n, k) = (3usize, 130usize, 256usize);
    let finite: Vec<u8> = (0..=255u8).filter(|c| c & 0x7f != 0x7f).collect();
    let codes: Vec<u8> = (0..ex * n * k)
        .map(|i| finite[(i * 13) % finite.len()])
        .collect();
    let grid: Vec<f32> = (0..ex * 2 * 2).map(|t| 0.01 * (t as f32 + 1.0)).collect();
    let d = scratch("fp8-bank");
    let path = d.join("m.safetensors");
    write_st(
        &path,
        &[
            ("experts", "F8_E4M3", vec![ex, n, k], codes.clone()),
            (
                "experts_scale_inv",
                "F32",
                vec![ex, 2, 2],
                le(&grid, f32::to_le_bytes),
            ),
            ("tensor", "F8_E4M3", vec![ex, n, k], codes.clone()),
            (
                "tensor_scale",
                "F32",
                vec![],
                le(&[0.125f32], f32::to_le_bytes),
            ),
        ],
    );
    let mut src = SafetensorsSource::open(&path).unwrap().fp8_packed(true);
    for key in ["experts", "tensor"] {
        assert!(
            src.take_packed(key).unwrap().is_none(),
            "rank 3 is a bank, not a linear"
        );
        let b = src.take_packed_bank(key).unwrap().expect("FP8 bank packs");
        assert_eq!(
            (b.scheme, b.num_experts, b.out_dim, b.in_dim),
            (QuantScheme::Fp8E4m3B128, ex, n, k)
        );
        let w = rlx_ir::lowp_codec::fp8_b128::dequant_scheme(b.scheme, &b.w_q, ex * n * k).unwrap();
        let (f, shape) = src.take(key, false).unwrap();
        assert_eq!(shape, vec![ex, n, k]);
        assert_eq!(w, f, "{key}: packed bank vs unpacked read");
    }
    let _ = std::fs::remove_dir_all(&d);
}
