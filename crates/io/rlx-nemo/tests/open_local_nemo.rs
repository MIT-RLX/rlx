// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end smoke test against a real `.nemo` if one is present in the
//! local HuggingFace cache. Skips (passes) when no file is found so CI on
//! a clean machine stays green. Set `RLX_NEMO_TEST_FILE` to point at any
//! `.nemo` explicitly.

use std::path::PathBuf;

use rlx_nemo::NemoModel;

fn locate_nemo() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("RLX_NEMO_TEST_FILE") {
        let pb = PathBuf::from(p);
        return pb.is_file().then_some(pb);
    }
    // The nano-codec .nemo ships the same torch.save layout as the ASR model.
    let home = std::env::var("HOME").ok()?;
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let pat = "models--nvidia--nemo-nano-codec-22khz-0.6kbps-12.5fps";
    let snaps = hub.join(pat).join("snapshots");
    let entries = std::fs::read_dir(&snaps).ok()?;
    for e in entries.flatten() {
        let dir = e.path();
        if let Ok(files) = std::fs::read_dir(&dir) {
            for f in files.flatten() {
                let p = f.path();
                if p.extension().and_then(|x| x.to_str()) == Some("nemo") {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[test]
fn open_and_read_real_nemo() {
    let Some(path) = locate_nemo() else {
        eprintln!("no local .nemo found; skipping (set RLX_NEMO_TEST_FILE to run)");
        return;
    };
    eprintln!("opening {}", path.display());

    let model = NemoModel::open(&path).expect("open .nemo");
    assert!(!model.is_empty(), "checkpoint should contain tensors");
    eprintln!("loaded {} tensors", model.len());

    // Config must parse and expose at least one scalar.
    assert!(
        model.config().get("sample_rate").is_some() || model.config().root().is_mapping(),
        "config should parse as a mapping"
    );

    // Read a handful of tensors fully and sanity-check them.
    let names = model.names();
    assert!(!names.is_empty());
    let mut checked = 0;
    for name in names.iter().take(8) {
        let t = model.tensor(name).expect("read tensor");
        let numel: usize = t.shape.iter().product();
        assert_eq!(t.data.len(), numel, "{name}: data len matches shape");
        assert!(
            t.data.iter().all(|x| x.is_finite()),
            "{name}: all values finite"
        );
        checked += 1;
    }
    assert_eq!(checked, names.len().min(8));
    eprintln!("verified {checked} tensors decode to finite, correctly-shaped f32");
}
