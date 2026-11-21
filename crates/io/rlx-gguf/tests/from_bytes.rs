// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GgufFile::from_bytes` (the in-place parse a wasm32 page needs) must read
//! exactly what `from_reader` does: same metadata, same tensors, same bytes.

use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};

fn sample() -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.set_arch("llama");
    w.set_meta("general.name", MetaValue::String("from-bytes".into()));
    let a: Vec<u8> = (0..64u32)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    let b: Vec<u8> = (0..32u16)
        .flat_map(|i| (0x3f80u16 + i).to_le_bytes())
        .collect();
    w.add_tensor_bytes("a", vec![8, 8], GgmlType::F32, a)
        .unwrap();
    w.add_tensor_bytes("b", vec![32], GgmlType::BF16, b)
        .unwrap();
    let mut out = std::io::Cursor::new(Vec::new());
    w.write(&mut out).unwrap();
    out.into_inner()
}

#[test]
fn from_bytes_matches_from_reader() {
    let bytes = sample();
    let by_reader = GgufFile::from_reader(&mut std::io::Cursor::new(bytes.clone())).unwrap();
    let in_place = GgufFile::from_bytes(bytes).unwrap();
    assert_eq!(by_reader.metadata.len(), in_place.metadata.len());
    for name in ["a", "b"] {
        let (x, y) = (by_reader.get(name).unwrap(), in_place.get(name).unwrap());
        assert_eq!(x.shape, y.shape, "{name} shape");
        assert_eq!(
            by_reader.tensor_bytes(x).unwrap(),
            in_place.tensor_bytes(y).unwrap(),
            "{name} bytes"
        );
    }
    let (b, _) = in_place.dequant_f32("b").unwrap();
    assert_eq!(b[0], 1.0, "bf16 0x3f80 is 1.0");
}

#[test]
fn from_bytes_rejects_a_truncated_header() {
    let bytes = sample();
    assert!(GgufFile::from_bytes(bytes[..16].to_vec()).is_err());
}

/// `retain` keeps exactly the chosen tensors, byte for byte, and frees the rest.
#[test]
fn retain_keeps_only_the_chosen_tensors() {
    let bytes = sample();
    let full = GgufFile::from_bytes(bytes.clone()).unwrap();
    let mut part = GgufFile::from_bytes(bytes).unwrap();
    part.retain(|name| name == "b").unwrap();
    assert!(part.get("a").is_none());
    let (x, y) = (full.get("b").unwrap(), part.get("b").unwrap());
    assert_eq!(full.tensor_bytes(x).unwrap(), part.tensor_bytes(y).unwrap());
    assert_eq!(
        full.dequant_f32("b").unwrap().0,
        part.dequant_f32("b").unwrap().0
    );
}

/// `retain_map` renames what it keeps (a pipeline stage renumbering its
/// blocks) and refuses to merge two tensors into one name.
#[test]
fn retain_map_renames_and_rejects_collisions() {
    let bytes = sample();
    let full = GgufFile::from_bytes(bytes.clone()).unwrap();
    let mut part = GgufFile::from_bytes(bytes.clone()).unwrap();
    part.retain_map(|name| (name == "a").then(|| "z".to_string()))
        .unwrap();
    assert!(part.get("a").is_none() && part.get("b").is_none());
    let z = part.get("z").unwrap();
    assert_eq!(z.name, "z");
    assert_eq!(
        full.tensor_bytes(full.get("a").unwrap()).unwrap(),
        part.tensor_bytes(z).unwrap()
    );
    let mut clash = GgufFile::from_bytes(bytes).unwrap();
    assert!(clash.retain_map(|_| Some("same".to_string())).is_err());
}

/// A header plus only the byte ranges `tensor_ranges` names reads the chosen
/// tensors exactly as the whole file does — the partial download a pipeline
/// stage makes.
#[test]
fn tensor_ranges_and_chunks_rebuild_a_subset() {
    let bytes = sample();
    let full = GgufFile::from_bytes(bytes.clone()).unwrap();
    let header = GgufFile::header_from_bytes(&bytes).unwrap();
    for (want, gap) in [(vec!["b"], 0u64), (vec!["a", "b"], 0), (vec!["a", "b"], 64)] {
        let keep = |n: &str| want.contains(&n);
        let ranges = header.tensor_ranges(keep, gap).unwrap();
        if gap > 0 {
            assert_eq!(ranges.len(), 1, "neighbours merge");
        }
        let chunks: Vec<(u64, Vec<u8>)> = ranges
            .iter()
            .map(|r| (r.start, bytes[r.start as usize..r.end as usize].to_vec()))
            .collect();
        let part = GgufFile::header_from_bytes(&bytes)
            .unwrap()
            .with_tensor_chunks(keep, &chunks)
            .unwrap();
        assert_eq!(part.tensors.len(), want.len());
        for name in &want {
            assert_eq!(
                full.dequant_f32(name).unwrap().0,
                part.dequant_f32(name).unwrap().0,
                "{name}"
            );
        }
    }
    // A chunk set that misses a tensor is an error, not zeros.
    let header = GgufFile::header_from_bytes(&bytes).unwrap();
    assert!(header.with_tensor_chunks(|_| true, &[]).is_err());
}

/// `begin_partial` + `feed` in uneven, out-of-order pieces assembles the
/// same tensors as the whole file; a missing piece is an error.
#[test]
fn partial_assembly_from_uneven_pieces() {
    let bytes = sample();
    let full = GgufFile::from_bytes(bytes.clone()).unwrap();
    let keep = |n: &str| n == "a" || n == "b";
    let mut p = GgufFile::header_from_bytes(&bytes)
        .unwrap()
        .begin_partial(keep)
        .unwrap();
    // 7-byte pieces, last to first.
    let mut pieces: Vec<(u64, &[u8])> = bytes
        .chunks(7)
        .enumerate()
        .map(|(i, c)| ((i * 7) as u64, c))
        .collect();
    pieces.reverse();
    for (start, c) in pieces {
        p.feed(start, c);
    }
    assert_eq!(p.missing(), 0);
    let part = p.finish().unwrap();
    for name in ["a", "b"] {
        assert_eq!(
            full.dequant_f32(name).unwrap().0,
            part.dequant_f32(name).unwrap().0,
            "{name}"
        );
    }
    let mut short = GgufFile::header_from_bytes(&bytes)
        .unwrap()
        .begin_partial(keep)
        .unwrap();
    short.feed(0, &bytes[..bytes.len() - 10]);
    assert!(short.missing() > 0);
    assert!(short.finish().is_err());
}

/// Renaming every tensor keeps the data in place (no rebuild).
#[test]
fn retain_map_renames_in_place_when_keeping_all() {
    let bytes = sample();
    let full = GgufFile::from_bytes(bytes.clone()).unwrap();
    let mut f = GgufFile::from_bytes(bytes).unwrap();
    f.retain_map(|n| Some(format!("x.{n}"))).unwrap();
    assert!(f.get("a").is_none());
    assert_eq!(
        full.dequant_f32("a").unwrap().0,
        f.dequant_f32("x.a").unwrap().0
    );
    assert_eq!(
        full.dequant_f32("b").unwrap().0,
        f.dequant_f32("x.b").unwrap().0
    );
}

/// A one-tensor-plus-sentinel GGUF holding `data` as `dtype`, with the
/// tensor's on-disk type id rewritten to `id` (to build the files the
/// header cannot tell apart on its own).
fn q2_file(dtype: GgmlType, n: usize, data: Vec<u8>, id: u32) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.set_arch("q2test");
    w.add_tensor_bytes("w".to_string(), vec![n], dtype, data)
        .unwrap();
    // A successor, so `w` has a measurable span.
    w.add_tensor_bytes("z".to_string(), vec![32], GgmlType::F32, vec![0u8; 128])
        .unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    w.write(&mut buf).unwrap();
    let mut bytes = buf.into_inner();
    // Tensor info: name (u64 len + bytes), n_dims u32, dims u64…, type u32.
    let pat = [
        &1u64.to_le_bytes()[..],
        b"w",
        &1u32.to_le_bytes(),
        &(n as u64).to_le_bytes(),
    ]
    .concat();
    let at = bytes
        .windows(pat.len())
        .position(|win| win == pat.as_slice())
        .expect("tensor info")
        + pat.len();
    bytes[at..at + 4].copy_from_slice(&id.to_le_bytes());
    bytes
}

#[test]
fn q2_0_type_42_is_ggml_group_64() {
    // ggml `dequantize_row_q2_0`: code (qs[j/4] >> 2(j%4)) & 3, value (q-1)·d.
    let n = 256;
    // Ternary values: exact under ggml's `d = amax` encoder.
    let src: Vec<f32> = (0..n).map(|i| ((i * 7) % 3) as f32 - 1.0).collect();
    let q = rlx_gguf::quantize::quantize(&src, GgmlType::Q2_0).unwrap();
    assert_eq!(q.len(), n / 64 * 18);
    let f = GgufFile::from_bytes(q2_file(GgmlType::Q2_0, n, q.clone(), 42)).unwrap();
    assert_eq!(f.get("w").unwrap().dtype, GgmlType::Q2_0);
    let got = f.dequant_f32("w").unwrap().0;
    for (j, g) in got.iter().enumerate() {
        let b = &q[j / 64 * 18..];
        let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        let code = (b[2 + (j % 64) / 4] >> (((j % 64) % 4) * 2)) & 3;
        assert_eq!(*g, (code as i32 - 1) as f32 * d, "element {j}");
    }
    assert_eq!(got, src);
}

#[test]
fn old_group_128_type_42_files_read_as_pq2_0() {
    // Ternary Bonsai v1 wrote group-128 blocks under id 42. Large enough that
    // the two layouts' sizes differ by more than the alignment padding.
    let n = 4096;
    let src: Vec<f32> = (0..n).map(|i| ((i * 3) % 3) as f32 - 1.0).collect();
    let q = rlx_gguf::quantize::quantize(&src, GgmlType::PQ2_0).unwrap();
    assert_eq!(q.len(), n / 128 * 34);
    let f = GgufFile::from_bytes(q2_file(GgmlType::PQ2_0, n, q, 42)).unwrap();
    assert_eq!(f.get("w").unwrap().dtype, GgmlType::PQ2_0);
    assert_eq!(f.dequant_f32("w").unwrap().0, src);
}
