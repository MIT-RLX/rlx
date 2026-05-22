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
//! Strict f32 parity vs `slang-splat/tools/parity_baseline.py` → `reference_cpu.py`.

use rlx_splat::{
    assert_parity_exact, core::make_parity_scene, parity_camera, parity_tiny_render_params,
    reference::render_reference, PARITY_BACKGROUND,
};

#[test]
fn tiny_render_matches_slang_splat_reference_cpu() {
    let root = std::env::var("SLANG_SPLAT_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../slang-splat")
        });
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../slang-splat-rs/tools/parity_baseline.py");
    if !root.is_dir() {
        eprintln!("skip: SLANG_SPLAT_ROOT / sibling slang-splat not found at {}", root.display());
        return;
    }
    if !script.is_file() {
        eprintln!("skip: parity_baseline.py missing at {}", script.display());
        return;
    }

    let tmp = std::env::temp_dir().join("rlx-splat-python-parity");
    let status = std::process::Command::new("python3")
        .arg(&script)
        .arg("--slang-splat-root")
        .arg(&root)
        .arg("--case")
        .arg("tiny_render_seed5")
        .arg("--output-dir")
        .arg(&tmp)
        .status()
        .expect("python3");
    assert!(status.success(), "parity_baseline.py failed");

    let blob_path = tmp.join("tiny_render_seed5.f32");
    let bytes = std::fs::read(&blob_path).expect("read baseline blob");
    assert_eq!(bytes.len(), 64 * 64 * 4 * 4);
    let expected: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let scene = make_parity_scene();
    let actual = render_reference(
        &scene,
        &parity_camera(),
        PARITY_BACKGROUND,
        &parity_tiny_render_params(),
    );

    assert_parity_exact(&actual, &expected).unwrap_or_else(|e| {
        let (mad, idx) = rlx_splat::max_abs_diff(&actual, &expected);
        panic!("{e}; max_abs={mad:.9e} @ {idx} rust={} py={}", actual[idx], expected[idx]);
    });
}
