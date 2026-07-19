// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Full round-trip validation of the ONNX codegen emitter: emit a standalone
// Rust program for a graph that exercises every indexing/contraction op, then
// compile it with `cargo` and run it. A green run proves the emitted text is
// valid Rust *and* builds a structurally-correct graph (op name, declared
// `num_inputs`, attr-blob length, and actual operand wiring all checked).
//
// Gated `#[ignore]` because it shells out to `cargo` (first run compiles the
// rlx crates into a throwaway target dir, which is slow). Run on demand:
//
//     cargo test -p rlx-onnx-import --test codegen_compile -- --ignored --nocapture

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use rlx_onnx_import::BundleNode;
use rlx_onnx_import::emit_codegen::{ConstSpec, GraphSpec, emit_graph_source};

fn node(
    op: &str,
    inputs: &[&str],
    out: &str,
    out_shape: &[i64],
    attrs: &[(&str, serde_json::Value)],
) -> BundleNode {
    BundleNode {
        name: format!("/{op}_0"),
        op: op.to_string(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs: vec![out.to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        output_meta: vec![serde_json::json!({"shape": out_shape, "dtype": "f32"})],
    }
}

/// Build the six-op graph spec covering GatherND/OneHot/NonZero/CumProd/Einsum/
/// ScatterND. GatherND/ScatterND are first-class; the rest stay Custom.
fn six_op_nodes() -> Vec<BundleNode> {
    vec![
        node(
            "GatherND",
            &["data_g", "idx_g"],
            "y_g",
            &[2],
            &[("batch_dims", serde_json::json!(0))],
        ),
        node(
            "OneHot",
            &["idx_oh", "depth_oh", "val_oh"],
            "y_oh",
            &[2, 3],
            &[("axis", serde_json::json!(-1))],
        ),
        node("NonZero", &["x_nz"], "y_nz", &[2, 3], &[]),
        node("CumProd", &["x_cp", "axis_cp"], "y_cp", &[2, 3], &[]),
        node(
            "Einsum",
            &["a_es", "b_es"],
            "y_es",
            &[2, 2],
            &[("equation", serde_json::json!("ij,jk->ik"))],
        ),
        node(
            "ScatterND",
            &["data_s", "idx_s", "upd_s"],
            "y_s",
            &[4, 4],
            &[],
        ),
    ]
}

fn meta(shape: &[i64], dtype: &str) -> serde_json::Value {
    serde_json::json!({"shape": shape, "dtype": dtype})
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_crate(name: &str) -> String {
    // crates/rlx-onnx-import -> crates/<name>
    crate_dir()
        .parent()
        .unwrap()
        .join(name)
        .to_string_lossy()
        .replace('\\', "/")
}

#[test]
#[ignore = "shells out to cargo; run with --ignored for full codegen validation"]
fn emitted_codegen_compiles_and_runs() {
    let nodes = six_op_nodes();
    let spec = GraphSpec {
        inputs: vec![
            ("data_g".into(), meta(&[2, 2], "f32")),
            ("x_nz".into(), meta(&[2, 3], "f32")),
            ("x_cp".into(), meta(&[2, 3], "f32")),
            ("a_es".into(), meta(&[2, 3], "f32")),
            ("b_es".into(), meta(&[3, 2], "f32")),
            ("data_s".into(), meta(&[4, 4], "f32")),
        ],
        consts: vec![
            ConstSpec::I64 {
                name: "idx_g".into(),
                data: vec![0, 0, 1, 1],
                dims: vec![2, 2],
            },
            ConstSpec::I64 {
                name: "idx_oh".into(),
                data: vec![1, 2],
                dims: vec![2],
            },
            ConstSpec::I64 {
                name: "depth_oh".into(),
                data: vec![3],
                dims: vec![1],
            },
            ConstSpec::F32 {
                name: "val_oh".into(),
                data: vec![0.0, 1.0],
                dims: vec![2],
            },
            ConstSpec::I64 {
                name: "axis_cp".into(),
                data: vec![1],
                dims: vec![1],
            },
            ConstSpec::I64 {
                name: "idx_s".into(),
                data: vec![0, 2],
                dims: vec![2, 1],
            },
            ConstSpec::F32 {
                name: "upd_s".into(),
                data: vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0],
                dims: vec![2, 4],
            },
        ],
        nodes: &nodes,
    };

    let source = emit_graph_source(&spec);
    // Sanity: the assembler must have routed every op through Op::Custom.
    assert_eq!(source.matches("Op::Custom").count(), 6, "source:\n{source}");

    // Lay down a throwaway, workspace-detached cargo project that path-depends
    // on the rlx crates (offline-safe — anyhow/serde_json arrive via re-export).
    let dir = std::env::temp_dir().join(format!("rlx_codegen_harness_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let cargo_toml = format!(
        "[package]\nname = \"rlx_codegen_harness\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n\
         [[bin]]\nname = \"harness\"\npath = \"src/main.rs\"\n\n\
         [dependencies]\n\
         rlx-onnx-import = {{ path = \"{}\" }}\n\
         rlx-ir = {{ path = \"{}\" }}\n\n\
         [workspace]\n",
        workspace_crate("rlx-onnx-import"),
        workspace_crate("rlx-ir"),
    );
    std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();
    std::fs::write(dir.join("src/main.rs"), &source).unwrap();

    let out = Command::new(env!("CARGO"))
        .args(["run", "--quiet"])
        .current_dir(&dir)
        // Isolated target dir so we never poison the workspace build cache.
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("failed to spawn cargo");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "generated program failed to compile/run\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n--- source ---\n{source}"
    );

    // name -> (num_inputs, attrs_len, actual_inputs) parsed from the run output.
    let got: HashMap<String, (u32, usize, usize)> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("CUSTOM "))
        .filter_map(|rest| {
            let p: Vec<&str> = rest.split_whitespace().collect();
            match p.as_slice() {
                [name, ni, al, ic] => Some((
                    name.to_string(),
                    (ni.parse().ok()?, al.parse().ok()?, ic.parse().ok()?),
                )),
                _ => None,
            }
        })
        .collect();

    // (op name, num_inputs, attrs_len, actual operand count) the compiled,
    // executed codegen must have produced.
    let expected = [
        ("onnx.OneHot", 3u32, 4usize, 3usize), // axis i32
        ("onnx.NonZero", 1, 0, 1),
        ("onnx.CumProd", 2, 6, 2), // [axis_i32, exclusive, reverse]; data+axis
        ("onnx.Einsum", 2, 9, 2),  // "ij,jk->ik" == 9 bytes
                                   // GatherND / ScatterND are first-class IR ops, not listed in CUSTOM summary.
    ];
    for (name, ni, al, ic) in expected {
        let actual = got
            .get(name)
            .unwrap_or_else(|| panic!("missing {name} in output:\n{stdout}"));
        assert_eq!(actual, &(ni, al, ic), "{name} mismatch (got {actual:?})");
    }
    assert!(
        !got.contains_key("onnx.ScatterND") && !got.contains_key("onnx.GatherND"),
        "ScatterND/GatherND must not emit Custom; got {got:?}"
    );
    assert_eq!(
        got.len(),
        expected.len(),
        "unexpected extra custom ops: {got:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
    let _ = Path::new(".");
}
