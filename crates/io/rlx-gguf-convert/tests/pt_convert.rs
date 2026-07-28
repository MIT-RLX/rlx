// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end: hand-assemble a PyTorch `torch.save` checkpoint (the ZIP +
//! pickle layout `torch.save` actually emits) in a tempdir, convert it to
//! GGUF at Q4_K, parse the GGUF back, and verify the reconstruction stays
//! close to the originals.

#![cfg(feature = "pt")]

use std::io::Write;

use rlx_gguf_convert::{Converter, Scheme};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn synth(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s as i32) as f32 / 2.147e9) * 2.0
        })
        .collect()
}

// ── torch.save ZIP + pickle synthesis ────────────────────────────────

fn push_int(p: &mut Vec<u8>, v: usize) {
    if v < 256 {
        p.extend_from_slice(&[b'K', v as u8]);
    } else if v < 65536 {
        p.push(b'M');
        p.extend_from_slice(&(v as u16).to_le_bytes());
    } else {
        p.push(b'J');
        p.extend_from_slice(&(v as i32).to_le_bytes());
    }
}

fn binunicode(p: &mut Vec<u8>, s: &str) {
    p.push(b'X');
    p.extend_from_slice(&(s.len() as u32).to_le_bytes());
    p.extend_from_slice(s.as_bytes());
}

/// Contiguous row-major stride for a shape.
fn contig_stride(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    stride
}

/// Build the `data.pkl` for a flat state dict of f32 (`FloatStorage`)
/// tensors. Storage `i` holds tensor `i`'s contiguous data at key `i`.
fn pickle_state_dict(tensors: &[(&str, Vec<usize>)]) -> Vec<u8> {
    let mut p: Vec<u8> = vec![0x80, 0x02]; // PROTO 2
    p.push(b'}'); // EMPTY_DICT
    p.push(b'('); // MARK (for SETITEMS)

    for (i, (name, shape)) in tensors.iter().enumerate() {
        binunicode(&mut p, name); // key
        let numel: usize = shape.iter().product();
        let stride = contig_stride(shape);

        // value = _rebuild_tensor_v2(storage, 0, size, stride, False, None)
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
        p.push(b'('); // MARK (args)

        // storage = persistent_load(("storage", FloatStorage, key, "cpu", numel))
        p.push(b'('); // MARK (persid tuple)
        binunicode(&mut p, "storage");
        p.extend_from_slice(b"ctorch\nFloatStorage\n");
        binunicode(&mut p, &i.to_string()); // storage key
        binunicode(&mut p, "cpu");
        push_int(&mut p, numel);
        p.push(b't'); // TUPLE
        p.push(b'Q'); // BINPERSID -> Storage

        push_int(&mut p, 0); // storage_offset

        p.push(b'('); // size tuple
        for &d in shape {
            push_int(&mut p, d);
        }
        p.push(b't');

        p.push(b'('); // stride tuple
        for &d in &stride {
            push_int(&mut p, d);
        }
        p.push(b't');

        p.push(0x89); // requires_grad = False
        p.push(b'N'); // backward_hooks = None

        p.push(b't'); // TUPLE -> args
        p.push(b'R'); // REDUCE -> Tensor
    }

    p.push(b'u'); // SETITEMS
    p.push(b'.'); // STOP
    p
}

/// STORED (uncompressed) ZIP — the layout `torch.save` emits. CRCs are
/// left zero; the reader keys off central directory + local headers.
fn zip_stored(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    let mut offsets = Vec::new();

    for (name, data) in files {
        offsets.push(out.len() as u32);
        let nb = name.as_bytes();
        out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
        out.extend_from_slice(&[20, 0]);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 0]); // STORED
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&[0, 0, 0, 0]); // crc
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(nb);
        out.extend_from_slice(data);
    }

    let cd_offset = out.len() as u32;
    for ((name, data), &off) in files.iter().zip(&offsets) {
        let nb = name.as_bytes();
        central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
        central.extend_from_slice(&[20, 0]);
        central.extend_from_slice(&[20, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]); // STORED
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]); // crc
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&off.to_le_bytes());
        central.extend_from_slice(nb);
    }
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);

    out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&[0, 0]);
    let n = files.len() as u16;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

fn write_pt(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> tempfile::NamedTempFile {
    let metas: Vec<(&str, Vec<usize>)> = tensors.iter().map(|(n, s, _)| (*n, s.clone())).collect();
    let mut files: Vec<(&str, Vec<u8>)> = vec![("archive/data.pkl", pickle_state_dict(&metas))];
    // Leak the storage-entry names so they can borrow as &str for the zip
    // builder (test-only; the process exits right after).
    for (i, (_, _, data)) in tensors.iter().enumerate() {
        let name: &'static str = Box::leak(format!("archive/data/{i}").into_boxed_str());
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        files.push((name, bytes));
    }
    let zip = zip_stored(&files);
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&zip).unwrap();
    f.flush().unwrap();
    f
}

// ── tests ─────────────────────────────────────────────────────────────

#[test]
fn pt_to_gguf_q4_k() {
    let w = synth(512, 1);
    let b = vec![0.1f32, -0.2, 0.3, -0.4];

    let pt = write_pt(&[("w", vec![2, 256], w.clone()), ("b", vec![4], b.clone())]);

    let gguf_path = tempfile::NamedTempFile::new().unwrap();
    let report = Converter::from_pt(pt.path())
        .unwrap()
        .default_scheme(Scheme::Q4_K)
        .skip_quant_for(|_, shape| shape.len() < 2)
        .architecture("test")
        .write_gguf(gguf_path.path())
        .unwrap();
    assert_eq!(report.tensors, 2);
    assert!(report.output_bytes > 0);

    let parsed = rlx_gguf::GgufFile::from_path(gguf_path.path()).unwrap();
    let (w_out, w_shape) = parsed.dequant_f32("w").unwrap();
    assert_eq!(w_shape, vec![2, 256]);
    assert!(cosine(&w, &w_out) > 0.99, "w cosine {}", cosine(&w, &w_out));

    let (b_out, b_shape) = parsed.dequant_f32("b").unwrap();
    assert_eq!(b_shape, vec![4]);
    for (a, c) in b.iter().zip(&b_out) {
        assert!((a - c).abs() < 1e-3, "bias mismatch {a} vs {c}");
    }
}

#[test]
fn pt_to_gguf_multi_tensor_names() {
    // A small two-layer state dict — exercises dotted keys and several
    // storages converting together.
    let pt = write_pt(&[
        ("layers.0.weight", vec![2, 256], synth(512, 7)),
        ("layers.1.weight", vec![2, 256], synth(512, 9)),
    ]);
    let gguf_path = tempfile::NamedTempFile::new().unwrap();
    let report = Converter::from_pt(pt.path())
        .unwrap()
        .default_scheme(Scheme::Q6_K)
        .write_gguf(gguf_path.path())
        .unwrap();
    assert_eq!(report.tensors, 2);

    let parsed = rlx_gguf::GgufFile::from_path(gguf_path.path()).unwrap();
    assert!(parsed.get("layers.0.weight").is_some());
    assert!(parsed.get("layers.1.weight").is_some());
}
