// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `import-pt` packs a torch.save checkpoint into `.rlxp`.

use std::io::Write;

use rlx_pkg::{Package, PtImportOptions, pt_to_rlxp};

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
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&[0, 0, 0, 0]);
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
        central.extend_from_slice(&[20, 0, 20, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&[0, 0, 0, 0]);
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        central.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        central.extend_from_slice(&off.to_le_bytes());
        central.extend_from_slice(nb);
    }
    let cd_size = central.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
    out.extend_from_slice(&[0, 0, 0, 0]);
    let n = files.len() as u16;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    out
}

fn binunicode(s: &str) -> Vec<u8> {
    let mut v = vec![b'X'];
    v.extend_from_slice(&(s.len() as u32).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v
}

fn pickle_one_tensor() -> Vec<u8> {
    let mut p: Vec<u8> = vec![0x80, 0x02];
    p.push(b'}');
    p.push(b'(');
    p.extend(binunicode("w"));
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
    p.push(b'(');
    p.push(b'(');
    p.extend(binunicode("storage"));
    p.extend_from_slice(b"ctorch\nFloatStorage\n");
    p.extend(binunicode("0"));
    p.extend(binunicode("cpu"));
    p.extend_from_slice(&[b'K', 4]);
    p.push(b't');
    p.push(b'Q');
    p.extend_from_slice(&[b'K', 0]);
    p.push(b'(');
    p.extend_from_slice(&[b'K', 2, b'K', 2]);
    p.push(b't');
    p.push(b'(');
    p.extend_from_slice(&[b'K', 2, b'K', 1]);
    p.push(b't');
    p.push(0x89);
    p.push(b'N');
    p.push(b't');
    p.push(b'R');
    p.push(b'u');
    p.push(b'.');
    p
}

#[test]
fn import_pt_to_rlxp() {
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let mut storage = Vec::new();
    for &v in &values {
        storage.extend_from_slice(&v.to_le_bytes());
    }
    let zip = zip_stored(&[
        ("archive/data.pkl", pickle_one_tensor()),
        ("archive/data/0", storage),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let pt = dir.path().join("w.pt");
    std::fs::File::create(&pt).unwrap().write_all(&zip).unwrap();

    let out = dir.path().join("out.rlxp");
    pt_to_rlxp(&pt, &out, &PtImportOptions::default()).unwrap();
    let pack = Package::open(&out).unwrap();
    let idx = pack.weights_index().expect("weights");
    assert!(idx.names().any(|n| n == "w"));
    let bytes = pack.tensor_bytes("w").unwrap();
    assert_eq!(bytes.len(), 16);
}
