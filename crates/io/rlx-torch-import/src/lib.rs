// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! PyTorch (`torch.export`) → RLX importer.
//!
//! The Python front-end (`pyrlx.torch_import`) exports a `torch.nn.Module` to a
//! neutral `torch-ir.json` (Core ATen graph, concrete shapes) + a
//! `weights.safetensors`. This crate maps that graph **directly onto RLX ops**
//! (no ONNX in the loop): [`lower::lower`] turns each aten node into the shared
//! [`call::Call`] vocabulary, which two walkers consume — [`hir_build`] builds a
//! live `HirModule` to run/verify, and [`emit`] prints a standalone RLX crate.
//!
//! Pipeline: `torch-ir.json` → [`lower`] → [`call::Lowered`] → { runnable
//! bundle (serialized HIR + weights), generated crate }, verified against
//! PyTorch via [`verify`].

pub mod call;
pub mod emit;
pub mod emit_styles;
pub mod hir_build;
pub mod ir;
pub mod lower;
pub mod nodeop;
pub mod onnx;
pub mod run;
pub mod verify;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub use call::Lowered;
pub use ir::TorchIr;
pub use verify::ParityReport;

/// Read + parse `torch-ir.json` from a directory.
pub fn load_ir(dir: &Path) -> Result<TorchIr> {
    let path = dir.join("torch-ir.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let ir: TorchIr =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        ir.format == "rlx-torch-ir",
        "unexpected IR format {:?}",
        ir.format
    );
    Ok(ir)
}

/// Metadata for a runnable bundle (`bundle/meta.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMeta {
    pub name: String,
    pub inputs: Vec<BundleIo>,
    pub output_count: usize,
    /// Zero-filled params to synthesize at load (key, numel).
    pub zero_params: Vec<(String, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleIo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Emit a self-contained runnable bundle: serialized HIR + weights + meta.
pub fn emit_bundle(src_dir: &Path, bundle_dir: &Path, lo: &Lowered) -> Result<()> {
    std::fs::create_dir_all(bundle_dir)?;
    let hir = hir_build::build_hir(lo)?;
    let json = rlx_ir::hir_to_json(&hir).context("serializing HIR")?;
    std::fs::write(bundle_dir.join("model.hir.json"), json)?;
    std::fs::copy(
        src_dir.join("weights.safetensors"),
        bundle_dir.join("weights.safetensors"),
    )
    .context("copying weights into bundle")?;
    let meta = BundleMeta {
        name: lo.name.clone(),
        inputs: lo
            .inputs
            .iter()
            .map(|i| BundleIo {
                name: i.name.clone(),
                shape: i.shape.clone(),
                dtype: format!("{:?}", i.dtype).to_lowercase(),
            })
            .collect(),
        output_count: lo.outputs.len(),
        zero_params: lo
            .zero_params
            .iter()
            .map(|z| (z.key.clone(), z.shape.iter().product()))
            .collect(),
    };
    std::fs::write(
        bundle_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}

/// Run a bundle emitted by [`emit_bundle`] on CPU with the given inputs.
#[cfg(feature = "runtime")]
pub fn run_bundle(bundle_dir: &Path, inputs: &[(String, Vec<f32>)]) -> Result<Vec<Vec<f32>>> {
    run_bundle_on(bundle_dir, inputs, rlx_runtime::Device::Cpu)
}

/// Run a bundle on the given device.
#[cfg(feature = "runtime")]
pub fn run_bundle_on(
    bundle_dir: &Path,
    inputs: &[(String, Vec<f32>)],
    device: rlx_runtime::Device,
) -> Result<Vec<Vec<f32>>> {
    let hir = rlx_ir::hir_from_json(&std::fs::read_to_string(bundle_dir.join("model.hir.json"))?)
        .context("deserializing HIR")?;
    let meta: BundleMeta =
        serde_json::from_str(&std::fs::read_to_string(bundle_dir.join("meta.json"))?)?;
    // All params bound as f32 (integer params are f32 nodes + a cast in the HIR).
    let weights = run::load_safetensors_f32(&bundle_dir.join("weights.safetensors"))?;
    let mut params: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for (k, (data, _)) in &weights {
        params.insert(k.clone(), data.clone());
    }
    for (k, numel) in &meta.zero_params {
        params.insert(k.clone(), vec![0.0f32; *numel]);
    }
    run::run_cpu(hir, &params, &[], inputs, device)
}

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub emit_bundle: bool,
    pub emit_crate: bool,
    pub verify: bool,
    pub crate_name: Option<String>,
    pub rlx_root: PathBuf,
    /// Authoring layer the generated crate targets (graph / tensor / flow).
    pub emit_style: emit_styles::EmitStyle,
    /// Device the parity check runs on ("cpu", "cuda", "metal", …).
    pub device: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConvertReport {
    pub model: String,
    pub num_inputs: usize,
    pub num_params: usize,
    pub num_instrs: usize,
    pub bundle_dir: Option<String>,
    pub crate_dir: Option<String>,
    pub parity: Option<ParityReport>,
}

/// The full import: lower → (bundle, crate) → verify. `dir` holds
/// `torch-ir.json` + `weights.safetensors` (+ `reference.safetensors` if
/// verifying).
pub fn convert(dir: &Path, opts: &ConvertOptions) -> Result<ConvertReport> {
    let ir = load_ir(dir)?;
    let lo = lower::lower(&ir)?;

    let mut report = ConvertReport {
        model: lo.name.clone(),
        num_inputs: lo.inputs.len(),
        num_params: lo.params.len() + lo.zero_params.len(),
        num_instrs: lo.instrs.len(),
        bundle_dir: None,
        crate_dir: None,
        parity: None,
    };

    if opts.emit_bundle {
        let bundle_dir = dir.join("bundle");
        emit_bundle(dir, &bundle_dir, &lo)?;
        report.bundle_dir = Some(bundle_dir.display().to_string());
    }

    if opts.emit_crate {
        let crate_name = opts
            .crate_name
            .clone()
            .unwrap_or_else(|| format!("rlx-{}", sanitize_crate(&lo.name)));
        let crate_dir = dir.join(&crate_name);
        emit::emit_crate(
            &crate_dir,
            &lo,
            &crate_name,
            &opts.rlx_root,
            opts.emit_style,
        )?;
        report.crate_dir = Some(crate_dir.display().to_string());
    }

    if opts.verify {
        #[cfg(feature = "runtime")]
        {
            let dev = run::parse_device(&opts.device)?;
            let hir = hir_build::build_hir(&lo)?;
            report.parity = Some(verify::verify(dir, &lo, hir, dev)?);
        }
        #[cfg(not(feature = "runtime"))]
        {
            anyhow::bail!("--verify requires the `runtime` feature");
        }
    }

    Ok(report)
}

fn sanitize_crate(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;

    // A tiny linear model: out = bias + x @ w  (addmm), all f32.
    const LINEAR_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "linear",
      "inputs":  [{"id": "x", "shape": [1, 2], "dtype": "f32"}],
      "weights": [
        {"id": "p_w", "key": "w", "shape": [2, 2], "dtype": "f32", "kind": "param"},
        {"id": "p_b", "key": "b", "shape": [2],    "dtype": "f32", "kind": "param"}
      ],
      "nodes": [
        {"id": "addmm", "op": "aten.addmm.default",
         "args": [{"ref": "p_b"}, {"ref": "x"}, {"ref": "p_w"}],
         "out": {"shape": [1, 2], "dtype": "f32"}}
      ],
      "outputs": [{"ref": "addmm", "shape": [1, 2], "dtype": "f32"}]
    }"#;

    fn linear_params() -> std::collections::HashMap<String, Vec<f32>> {
        // w = [[1,2],[3,4]], b = [10, 20]
        let mut p = std::collections::HashMap::new();
        p.insert("w".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        p.insert("b".to_string(), vec![10.0, 20.0]);
        p
    }

    #[test]
    fn lower_build_run_linear() {
        let ir: TorchIr = serde_json::from_str(LINEAR_IR).unwrap();
        let lo = lower::lower(&ir).unwrap();
        assert_eq!(lo.inputs.len(), 1);
        assert_eq!(lo.params.len(), 2);
        // addmm expands to mm + add.
        assert_eq!(lo.instrs.len(), 2);

        let hir = hir_build::build_hir(&lo).unwrap();
        let params = linear_params();
        // x = [5, 6]; x@w = [5*1+6*3, 5*2+6*4] = [23, 34]; + b = [33, 54]
        let out = run::run_cpu(
            hir,
            &params,
            &[],
            &[("x".to_string(), vec![5.0, 6.0])],
            rlx_runtime::Device::Cpu,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
        assert!((out[0][0] - 33.0).abs() < 1e-4, "got {:?}", out[0]);
        assert!((out[0][1] - 54.0).abs() < 1e-4, "got {:?}", out[0]);
    }

    #[test]
    fn hir_json_roundtrip_runs_identically() {
        let ir: TorchIr = serde_json::from_str(LINEAR_IR).unwrap();
        let lo = lower::lower(&ir).unwrap();
        let params = linear_params();

        let hir = hir_build::build_hir(&lo).unwrap();
        let direct = run::run_cpu(
            hir,
            &params,
            &[],
            &[("x".to_string(), vec![5.0, 6.0])],
            rlx_runtime::Device::Cpu,
        )
        .unwrap();

        // Serialize → deserialize (the bundle's core) and re-run.
        let hir2 = hir_build::build_hir(&lo).unwrap();
        let json = rlx_ir::hir_to_json(&hir2).unwrap();
        let restored = rlx_ir::hir_from_json(&json).unwrap();
        let round = run::run_cpu(
            restored,
            &params,
            &[],
            &[("x".to_string(), vec![5.0, 6.0])],
            rlx_runtime::Device::Cpu,
        )
        .unwrap();

        assert_eq!(direct, round);
    }

    #[test]
    fn unsupported_op_is_reported() {
        let ir_json = r#"{
          "format": "rlx-torch-ir", "version": 1, "model_name": "m",
          "inputs": [{"id": "x", "shape": [2], "dtype": "f32"}],
          "weights": [],
          "nodes": [{"id": "n", "op": "aten.some_exotic_op.default",
                     "args": [{"ref": "x"}], "out": {"shape": [2], "dtype": "f32"}}],
          "outputs": [{"ref": "n", "shape": [2], "dtype": "f32"}]
        }"#;
        let ir: TorchIr = serde_json::from_str(ir_json).unwrap();
        let err = lower::lower(&ir).unwrap_err().to_string();
        assert!(err.contains("some_exotic_op"), "err was: {err}");
    }

    fn run_ir(ir_json: &str, params: &[(&str, Vec<f32>)], input: (&str, Vec<f32>)) -> Vec<f32> {
        let ir: TorchIr = serde_json::from_str(ir_json).unwrap();
        let lo = lower::lower(&ir).unwrap();
        let hir = hir_build::build_hir(&lo).unwrap();
        let pmap: std::collections::HashMap<String, Vec<f32>> = params
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let out = run::run_cpu(
            hir,
            &pmap,
            &[],
            &[(input.0.to_string(), input.1)],
            rlx_runtime::Device::Cpu,
        )
        .unwrap();
        out.into_iter().next().unwrap()
    }

    // native_group_norm (multi-output) → Op::GroupNorm, output via getitem[0].
    const GN_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "gn",
      "inputs":  [{"id": "x", "shape": [1, 2, 1, 2], "dtype": "f32"}],
      "weights": [
        {"id": "p_g", "key": "gn_gamma", "shape": [2], "dtype": "f32", "kind": "param"},
        {"id": "p_b", "key": "gn_beta",  "shape": [2], "dtype": "f32", "kind": "param"}
      ],
      "nodes": [
        {"id": "gn", "op": "aten.native_group_norm.default",
         "args": [{"ref": "x"}, {"ref": "p_g"}, {"ref": "p_b"},
                  {"int": 1}, {"int": 2}, {"int": 2}, {"int": 1}, {"float": 1e-5}],
         "out": [{"shape": [1,2,1,2], "dtype": "f32"},
                 {"shape": [1,1], "dtype": "f32"},
                 {"shape": [1,1], "dtype": "f32"}]},
        {"id": "gi", "op": "_getitem", "args": [{"ref": "gn"}, {"int": 0}],
         "out": {"shape": [1,2,1,2], "dtype": "f32"}}
      ],
      "outputs": [{"ref": "gi", "shape": [1,2,1,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn native_group_norm_matches_reference() {
        // 1 group over [1,2,3,4]: mean=2.5, var=1.25, inv=1/sqrt(1.25+eps).
        let out = run_ir(
            GN_IR,
            &[("gn_gamma", vec![1.0, 1.0]), ("gn_beta", vec![0.0, 0.0])],
            ("x", vec![1.0, 2.0, 3.0, 4.0]),
        );
        let inv = 1.0f32 / (1.25f32 + 1e-5).sqrt();
        let want = [-1.5 * inv, -0.5 * inv, 0.5 * inv, 1.5 * inv];
        assert_eq!(out.len(), 4);
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "group_norm {out:?} vs {want:?}");
        }
    }

    // upsample_nearest2d 2× → single ResizeNearest2x (exact pixel doubling).
    const UP2_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "up2",
      "inputs":  [{"id": "x", "shape": [1, 1, 2, 2], "dtype": "f32"}],
      "weights": [],
      "nodes": [
        {"id": "up", "op": "aten.upsample_nearest2d.default",
         "args": [{"ref": "x"}, {"list": [{"int": 4}, {"int": 4}]}],
         "out": {"shape": [1,1,4,4], "dtype": "f32"}}
      ],
      "outputs": [{"ref": "up", "shape": [1,1,4,4], "dtype": "f32"}]
    }"#;

    #[test]
    fn upsample_nearest_2x_doubles_pixels() {
        let out = run_ir(UP2_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0]));
        #[rustfmt::skip]
        let want = vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(out, want);
    }

    // upsample_nearest2d.vec 4× → chained ResizeNearest2x (two 2× steps).
    const UP4_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "up4",
      "inputs":  [{"id": "x", "shape": [1, 1, 1, 1], "dtype": "f32"}],
      "weights": [],
      "nodes": [
        {"id": "up", "op": "aten.upsample_nearest2d.vec",
         "args": [{"ref": "x"}, {"none": true}, {"list": [{"float": 4.0}, {"float": 4.0}]}],
         "out": {"shape": [1,1,4,4], "dtype": "f32"}}
      ],
      "outputs": [{"ref": "up", "shape": [1,1,4,4], "dtype": "f32"}]
    }"#;

    #[test]
    fn upsample_nearest_4x_chains_two_steps() {
        let lo = lower::lower(&serde_json::from_str::<TorchIr>(UP4_IR).unwrap()).unwrap();
        // one intermediate (2×) + one final (4×) ResizeNearest2x instruction.
        assert_eq!(lo.instrs.len(), 2, "4× should chain two 2× resizes");
        let out = run_ir(UP4_IR, &[], ("x", vec![5.0]));
        assert_eq!(out, vec![5.0; 16]);
    }

    // constant_pad_nd on the last axis: prepend 1 + append 2 columns of 7.
    const PAD_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "pad",
      "inputs":  [{"id": "x", "shape": [1, 1, 2, 2], "dtype": "f32"}],
      "weights": [],
      "nodes": [
        {"id": "pd", "op": "aten.constant_pad_nd.default",
         "args": [{"ref": "x"}, {"list": [{"int": 1}, {"int": 2}]}, {"float": 7.0}],
         "out": {"shape": [1,1,2,5], "dtype": "f32"}}
      ],
      "outputs": [{"ref": "pd", "shape": [1,1,2,5], "dtype": "f32"}]
    }"#;

    #[test]
    fn constant_pad_nd_last_axis_fills_constant() {
        // [[1,2],[3,4]] pad (1,2) with 7 → [[7,1,2,7,7],[7,3,4,7,7]].
        let out = run_ir(PAD_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(out, vec![7.0, 1.0, 2.0, 7.0, 7.0, 7.0, 3.0, 4.0, 7.0, 7.0]);
    }

    fn run_ir_n(ir_json: &str, inputs: &[(&str, Vec<f32>)]) -> Vec<f32> {
        let ir: TorchIr = serde_json::from_str(ir_json).unwrap();
        let lo = lower::lower(&ir).unwrap();
        let hir = hir_build::build_hir(&lo).unwrap();
        let ins: Vec<(String, Vec<f32>)> = inputs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        let out = run::run_cpu(
            hir,
            &Default::default(),
            &[],
            &ins,
            rlx_runtime::Device::Cpu,
        )
        .unwrap();
        out.into_iter().next().unwrap()
    }

    // split.Tensor into equal chunks; second piece reached via getitem[1].
    const SPLIT_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "split",
      "inputs":  [{"id": "x", "shape": [1, 4], "dtype": "f32"}], "weights": [],
      "nodes": [
        {"id": "sp", "op": "aten.split.Tensor", "args": [{"ref": "x"}, {"int": 2}, {"int": 1}],
         "out": [{"shape": [1,2], "dtype": "f32"}, {"shape": [1,2], "dtype": "f32"}]},
        {"id": "g1", "op": "_getitem", "args": [{"ref": "sp"}, {"int": 1}],
         "out": {"shape": [1,2], "dtype": "f32"}}],
      "outputs": [{"ref": "g1", "shape": [1,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn split_tensor_second_piece() {
        // [1,2,3,4] split(2, dim=1) → ([1,2],[3,4]); piece 1 = [3,4].
        let out = run_ir(SPLIT_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(out, vec![3.0, 4.0]);
    }

    // baddbmm on the beta=0 attention path, with an `empty` bias (→ zeros).
    const BADDBMM_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "badd",
      "inputs":  [{"id": "a", "shape": [1,2,2], "dtype": "f32"},
                  {"id": "b", "shape": [1,2,2], "dtype": "f32"}],
      "weights": [],
      "nodes": [
        {"id": "e", "op": "aten.empty.memory_format",
         "args": [{"list": [{"int": 1}, {"int": 2}, {"int": 2}]}],
         "out": {"shape": [1,2,2], "dtype": "f32"}},
        {"id": "bd", "op": "aten.baddbmm.default",
         "args": [{"ref": "e"}, {"ref": "a"}, {"ref": "b"}],
         "kwargs": {"beta": {"float": 0.0}, "alpha": {"float": 2.0}},
         "out": {"shape": [1,2,2], "dtype": "f32"}}],
      "outputs": [{"ref": "bd", "shape": [1,2,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn baddbmm_beta0_alpha_scaled_matmul() {
        // a=[[1,2],[3,4]] @ b=I = a; alpha=2, beta=0 → 2*a = [[2,4],[6,8]].
        let out = run_ir_n(
            BADDBMM_IR,
            &[
                ("a", vec![1.0, 2.0, 3.0, 4.0]),
                ("b", vec![1.0, 0.0, 0.0, 1.0]),
            ],
        );
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    // ones_like → constant of 1s, then a real op consumes it.
    const ONES_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "ones",
      "inputs":  [{"id": "x", "shape": [2], "dtype": "f32"}], "weights": [],
      "nodes": [
        {"id": "o", "op": "aten.ones_like.default", "args": [{"ref": "x"}],
         "out": {"shape": [2], "dtype": "f32"}},
        {"id": "y", "op": "aten.add.Tensor", "args": [{"ref": "x"}, {"ref": "o"}],
         "out": {"shape": [2], "dtype": "f32"}}],
      "outputs": [{"ref": "y", "shape": [2], "dtype": "f32"}]
    }"#;

    // Dynamic leading (batch) dim: input axis 0 marked dynamic (symbol 0 = BATCH).
    const DYN_BATCH_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "dynlin",
      "inputs":  [{"id": "x", "shape": [2, 4], "dtype": "f32", "dynamic": [0, -1]}],
      "weights": [{"id": "p_w", "key": "w", "shape": [4, 2], "dtype": "f32", "kind": "param"}],
      "nodes": [{"id": "mm", "op": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "p_w"}],
                 "out": {"shape": [2, 2], "dtype": "f32"}}],
      "outputs": [{"ref": "mm", "shape": [2, 2], "dtype": "f32"}]
    }"#;

    #[test]
    fn dynamic_batch_runs_at_multiple_sizes() {
        // Import once (dynamic batch), run at batch 2 and 4 by binding sym 0.
        // W picks columns so out[r] = [x[r,0]+x[r,2], x[r,1]+x[r,3]].
        let lo = lower::lower(&serde_json::from_str::<TorchIr>(DYN_BATCH_IR).unwrap()).unwrap();
        assert!(lo.inputs[0].is_dynamic(), "input axis 0 should be dynamic");
        let params: std::collections::HashMap<String, Vec<f32>> = [(
            "w".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )]
        .into_iter()
        .collect();
        for batch in [2usize, 4usize] {
            let hir = hir_build::build_hir(&lo).unwrap();
            let x: Vec<f32> = (0..batch * 4).map(|i| i as f32).collect();
            let binding = rlx_ir::DimBinding::from_pairs(&[(0, batch)]);
            let out = run::run_dynamic(
                hir,
                &params,
                &[("x".to_string(), x.clone())],
                rlx_runtime::Device::Cpu,
                &binding,
            )
            .unwrap();
            assert_eq!(out[0].len(), batch * 2, "batch {batch} output size");
            for r in 0..batch {
                let (e0, e1) = (x[r * 4] + x[r * 4 + 2], x[r * 4 + 1] + x[r * 4 + 3]);
                assert!(
                    (out[0][r * 2] - e0).abs() < 1e-4 && (out[0][r * 2 + 1] - e1).abs() < 1e-4,
                    "batch {batch} row {r}: got {:?} want [{e0},{e1}]",
                    &out[0][r * 2..r * 2 + 2]
                );
            }
        }
    }

    // aten `sum.dim_IntList` with an empty dim list reduces over ALL dims (→ a
    // scalar) — how a bare `Tensor.sum()` decomposes. Reducing *nothing* (the old
    // behavior) left a `[2,3]` and broke downstream broadcasts (e.g. `+ eye`).
    const SUM_ALL_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "sumall",
      "inputs":  [{"id": "x", "shape": [2, 3], "dtype": "f32"}],
      "weights": [],
      "nodes": [{"id": "s", "op": "aten.sum.dim_IntList", "args": [{"ref": "x"}, {"list": []}],
                 "out": {"shape": [], "dtype": "f32"}}],
      "outputs": [{"ref": "s", "shape": [], "dtype": "f32"}]
    }"#;

    #[test]
    fn sum_empty_dims_reduces_all() {
        let out = run_ir(SUM_ALL_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert_eq!(out.len(), 1, "empty-dim sum should reduce to a scalar");
        assert!(
            (out[0] - 21.0).abs() < 1e-5,
            "sum of 1..6 = 21, got {out:?}"
        );
    }

    #[test]
    fn ones_like_fills_ones() {
        // x + ones_like(x): [5,7] + [1,1] = [6,8].
        let out = run_ir(ONES_IR, &[], ("x", vec![5.0, 7.0]));
        assert_eq!(out, vec![6.0, 8.0]);
    }

    // pixel_shuffle r=2 on [1,4,1,2] → [1,1,2,4] (pure reshape+permute).
    const PS_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "ps",
      "inputs":  [{"id": "x", "shape": [1, 4, 1, 2], "dtype": "f32"}], "weights": [],
      "nodes": [
        {"id": "ps", "op": "aten.pixel_shuffle.default", "args": [{"ref": "x"}, {"int": 2}],
         "out": {"shape": [1,1,2,4], "dtype": "f32"}}],
      "outputs": [{"ref": "ps", "shape": [1,1,2,4], "dtype": "f32"}]
    }"#;

    #[test]
    fn pixel_shuffle_rearranges() {
        // channels [0,1|2,3|4,5|6,7] (each 1×2) → interleaved r×r spatial blocks.
        let out = run_ir(
            PS_IR,
            &[],
            ("x", vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        );
        assert_eq!(out, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    // upsample_bilinear2d 2× align_corners=False → resize_bilinear2d (separable).
    const BILIN_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "bl",
      "inputs": [{"id": "x", "shape": [1, 1, 2, 2], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "bl", "op": "aten.upsample_bilinear2d.default",
                 "args": [{"ref": "x"}, {"list": [{"int": 4}, {"int": 4}]}, {"bool": false}],
                 "out": {"shape": [1,1,4,4], "dtype": "f32"}}],
      "outputs": [{"ref": "bl", "shape": [1,1,4,4], "dtype": "f32"}]
    }"#;

    #[test]
    fn upsample_bilinear_align_corners_false() {
        // torch F.interpolate([[1,2],[3,4]], size=(4,4), mode="bilinear",
        //   align_corners=False) — hand-computed separable reference.
        let out = run_ir(BILIN_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0]));
        #[rustfmt::skip]
        let want = vec![
            1.0,  1.25, 1.75, 2.0,
            1.5,  1.75, 2.25, 2.5,
            2.5,  2.75, 3.25, 3.5,
            3.0,  3.25, 3.75, 4.0,
        ];
        assert_eq!(out.len(), 16);
        for (g, w) in out.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "bilinear {out:?} vs {want:?}");
        }
    }

    // upsample_bicubic2d 4→7 (W only), align_corners=True → resize_bicubic2d.
    const BICUBIC_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "bc",
      "inputs": [{"id": "x", "shape": [1, 1, 1, 4], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "bc", "op": "aten.upsample_bicubic2d.default",
                 "args": [{"ref": "x"}, {"list": [{"int": 1}, {"int": 7}]}, {"bool": true}],
                 "out": {"shape": [1,1,1,7], "dtype": "f32"}}],
      "outputs": [{"ref": "bc", "shape": [1,1,1,7], "dtype": "f32"}]
    }"#;

    #[test]
    fn upsample_bicubic_align_corners_true() {
        // torch F.interpolate([0,1,2,3], size=7, mode="bicubic", align_corners=True):
        // interior points reproduce the ramp; edges use the a=-0.75 kernel + clamp.
        let out = run_ir(BICUBIC_IR, &[], ("x", vec![0.0, 1.0, 2.0, 3.0]));
        let want = [0.0, 0.40625, 1.0, 1.5, 2.0, 2.59375, 3.0];
        assert_eq!(out.len(), 7);
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "bicubic {out:?} vs {want:?}");
        }
    }

    // _upsample_bilinear2d_aa 4→2 (W only), align_corners=False — antialiased.
    const BILIN_AA_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "aa",
      "inputs": [{"id": "x", "shape": [1, 1, 1, 4], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "aa", "op": "aten._upsample_bilinear2d_aa.default",
                 "args": [{"ref": "x"}, {"list": [{"int": 1}, {"int": 2}]}, {"bool": false}],
                 "out": {"shape": [1,1,1,2], "dtype": "f32"}}],
      "outputs": [{"ref": "aa", "shape": [1,1,1,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn upsample_bilinear_aa_downsample() {
        // torch antialias bilinear [1,2,3,4] → 2: widened triangle → [12/7, 23/7].
        let out = run_ir(BILIN_AA_IR, &[], ("x", vec![1.0, 2.0, 3.0, 4.0]));
        let want = [12.0f32 / 7.0, 23.0f32 / 7.0];
        assert_eq!(out.len(), 2);
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "aa {out:?} vs {want:?}");
        }
    }

    // grid_sampler_2d on a 2×2 image; mode/pad set via `{MODE}`/`{PAD}` codes.
    fn gridsample_ir(mode: i64, pad: i64, ac: bool, wo: usize) -> String {
        format!(
            r#"{{
      "format": "rlx-torch-ir", "version": 1, "model_name": "gs",
      "inputs": [{{"id": "x", "shape": [1,1,2,2], "dtype": "f32"}},
                 {{"id": "g", "shape": [1,1,{wo},2], "dtype": "f32"}}], "weights": [],
      "nodes": [{{"id": "gs", "op": "aten.grid_sampler_2d.default",
                 "args": [{{"ref": "x"}}, {{"ref": "g"}}, {{"int": {mode}}}, {{"int": {pad}}}, {{"bool": {ac}}}],
                 "out": {{"shape": [1,1,1,{wo}], "dtype": "f32"}}}}],
      "outputs": [{{"ref": "gs", "shape": [1,1,1,{wo}], "dtype": "f32"}}]
    }}"#
        )
    }

    #[test]
    fn grid_sample_bilinear_zeros() {
        // image [[1,2],[3,4]] (align_corners=True). Sample (0,0)→center=2.5,
        // (-1,-1)→pixel(0,0)=1, (3,3)→all corners OOB→0 (zeros padding).
        let ir = gridsample_ir(0, 0, true, 3);
        let out = run_ir_n(
            &ir,
            &[
                ("x", vec![1.0, 2.0, 3.0, 4.0]),
                ("g", vec![0.0, 0.0, -1.0, -1.0, 3.0, 3.0]),
            ],
        );
        let want = [2.5f32, 1.0, 0.0];
        for (o, w) in out.iter().zip(want) {
            assert!((o - w).abs() < 1e-5, "bilinear/zeros {out:?} vs {want:?}");
        }
    }

    #[test]
    fn grid_sample_nearest() {
        // (-1,-1)→pixel(0,0)=1; (1,1)→pixel(1,1)=4; (0.4,0.4)→round→pixel(1,1)=4.
        let ir = gridsample_ir(1, 0, true, 3);
        let out = run_ir_n(
            &ir,
            &[
                ("x", vec![1.0, 2.0, 3.0, 4.0]),
                ("g", vec![-1.0, -1.0, 1.0, 1.0, 0.4, 0.4]),
            ],
        );
        let want = [1.0f32, 4.0, 4.0];
        for (o, w) in out.iter().zip(want) {
            assert!((o - w).abs() < 1e-5, "nearest {out:?} vs {want:?}");
        }
    }

    #[test]
    fn grid_sample_border() {
        // (3,3) with border padding clamps coord to (1,1) → pixel(1,1)=4.
        let ir = gridsample_ir(0, 1, true, 1);
        let out = run_ir_n(
            &ir,
            &[("x", vec![1.0, 2.0, 3.0, 4.0]), ("g", vec![3.0, 3.0])],
        );
        assert!((out[0] - 4.0).abs() < 1e-5, "border {out:?}");
    }

    #[test]
    fn grid_sample_reflection() {
        // align_corners=True reflection (span=size-1=1): (3,3)→coord 2.0→reflect→0
        // →pixel(0,0)=1; (2,2)→1.5→0.5→center=2.5; (-3,-3)→-1.0→reflect→1→pixel(1,1)=4.
        let ir = gridsample_ir(0, 2, true, 3);
        let out = run_ir_n(
            &ir,
            &[
                ("x", vec![1.0, 2.0, 3.0, 4.0]),
                ("g", vec![3.0, 3.0, 2.0, 2.0, -3.0, -3.0]),
            ],
        );
        let want = [1.0f32, 2.5, 4.0];
        for (o, w) in out.iter().zip(want) {
            assert!((o - w).abs() < 1e-5, "reflection {out:?} vs {want:?}");
        }
    }

    // bicubic on a 4×4 constant image → constant (partition of unity end-to-end).
    const BICUBIC_GS_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "gsb",
      "inputs": [{"id": "x", "shape": [1,1,4,4], "dtype": "f32"},
                 {"id": "g", "shape": [1,1,2,2], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "gs", "op": "aten.grid_sampler_2d.default",
                 "args": [{"ref": "x"}, {"ref": "g"}, {"int": 2}, {"int": 1}, {"bool": true}],
                 "out": {"shape": [1,1,1,2], "dtype": "f32"}}],
      "outputs": [{"ref": "gs", "shape": [1,1,1,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn grid_sample_bicubic_constant() {
        // bicubic weights sum to 1 → a constant image resamples to the same const
        // (border padding keeps partition-of-unity even near the edge).
        let out = run_ir_n(
            BICUBIC_GS_IR,
            &[("x", vec![5.0; 16]), ("g", vec![-0.3, 0.2, 0.5, -0.1])],
        );
        for o in &out {
            assert!((o - 5.0).abs() < 1e-4, "bicubic const {out:?}");
        }
    }

    // Run one IR/input on a named device (for cross-backend parity checks).
    #[cfg(any(feature = "metal", feature = "mlx"))]
    fn run_ir_on(ir_json: &str, input: (&str, Vec<f32>), device: &str) -> Vec<f32> {
        let ir: TorchIr = serde_json::from_str(ir_json).unwrap();
        let lo = lower::lower(&ir).unwrap();
        let hir = hir_build::build_hir(&lo).unwrap();
        let dev = run::parse_device(device).unwrap();
        run::run_cpu(
            hir,
            &Default::default(),
            &[],
            &[(input.0.to_string(), input.1)],
            dev,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
    }

    #[cfg(any(feature = "metal", feature = "mlx"))]
    fn run_ir_on_n(ir_json: &str, inputs: &[(&str, Vec<f32>)], device: &str) -> Vec<f32> {
        let ir: TorchIr = serde_json::from_str(ir_json).unwrap();
        let lo = lower::lower(&ir).unwrap();
        let hir = hir_build::build_hir(&lo).unwrap();
        let dev = run::parse_device(device).unwrap();
        let ins: Vec<(String, Vec<f32>)> = inputs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        run::run_cpu(hir, &Default::default(), &[], &ins, dev)
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    #[cfg(feature = "metal")]
    #[test]
    fn grid_sample_metal_matches_cpu() {
        // All four modes on Apple GPU must match CPU (proves native execution).
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![0.3f32, -0.2, -0.7, 0.6, 1.4, -1.4];
        for (mode, pad) in [(0i64, 0i64), (1, 0), (0, 1), (0, 2)] {
            let ir = gridsample_ir(mode, pad, false, 3);
            let cpu = run_ir_on_n(&ir, &[("x", x.clone()), ("g", g.clone())], "cpu");
            let gpu = run_ir_on_n(&ir, &[("x", x.clone()), ("g", g.clone())], "metal");
            for (a, b) in gpu.iter().zip(&cpu) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "metal mode={mode} pad={pad} {gpu:?} vs {cpu:?}"
                );
            }
        }
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn grid_sample_mlx_matches_cpu() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![0.3f32, -0.2, -0.7, 0.6, 1.4, -1.4];
        for (mode, pad) in [(0i64, 0i64), (1, 0), (0, 1), (0, 2)] {
            let ir = gridsample_ir(mode, pad, false, 3);
            let cpu = run_ir_on_n(&ir, &[("x", x.clone()), ("g", g.clone())], "cpu");
            let gpu = run_ir_on_n(&ir, &[("x", x.clone()), ("g", g.clone())], "mlx");
            for (a, b) in gpu.iter().zip(&cpu) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "mlx mode={mode} pad={pad} {gpu:?} vs {cpu:?}"
                );
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn bilinear_metal_matches_cpu() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let cpu = run_ir_on(BILIN_IR, ("x", x.clone()), "cpu");
        let gpu = run_ir_on(BILIN_IR, ("x", x), "metal");
        for (a, b) in gpu.iter().zip(&cpu) {
            assert!((a - b).abs() < 1e-4, "metal {gpu:?} vs cpu {cpu:?}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn bicubic_metal_matches_cpu() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0];
        let cpu = run_ir_on(BICUBIC_IR, ("x", x.clone()), "cpu");
        let gpu = run_ir_on(BICUBIC_IR, ("x", x), "metal");
        for (a, b) in gpu.iter().zip(&cpu) {
            assert!((a - b).abs() < 1e-4, "metal {gpu:?} vs cpu {cpu:?}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn bilinear_aa_metal_matches_cpu() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let cpu = run_ir_on(BILIN_AA_IR, ("x", x.clone()), "cpu");
        let gpu = run_ir_on(BILIN_AA_IR, ("x", x), "metal");
        for (a, b) in gpu.iter().zip(&cpu) {
            assert!((a - b).abs() < 1e-4, "metal {gpu:?} vs cpu {cpu:?}");
        }
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn bilinear_mlx_matches_cpu() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let cpu = run_ir_on(BILIN_IR, ("x", x.clone()), "cpu");
        let gpu = run_ir_on(BILIN_IR, ("x", x), "mlx");
        for (a, b) in gpu.iter().zip(&cpu) {
            assert!((a - b).abs() < 1e-4, "mlx {gpu:?} vs cpu {cpu:?}");
        }
    }

    // _scaled_dot_product_flash_attention (multi-output) must match the public op.
    const SDPA_PUB_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "sd",
      "inputs": [{"id": "q", "shape": [1,1,2,2], "dtype": "f32"},
                 {"id": "k", "shape": [1,1,2,2], "dtype": "f32"},
                 {"id": "v", "shape": [1,1,2,2], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "sd", "op": "aten.scaled_dot_product_attention.default",
                 "args": [{"ref": "q"}, {"ref": "k"}, {"ref": "v"}],
                 "out": {"shape": [1,1,2,2], "dtype": "f32"}}],
      "outputs": [{"ref": "sd", "shape": [1,1,2,2], "dtype": "f32"}]
    }"#;
    const SDPA_FLASH_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "fa",
      "inputs": [{"id": "q", "shape": [1,1,2,2], "dtype": "f32"},
                 {"id": "k", "shape": [1,1,2,2], "dtype": "f32"},
                 {"id": "v", "shape": [1,1,2,2], "dtype": "f32"}], "weights": [],
      "nodes": [
        {"id": "fa", "op": "aten._scaled_dot_product_flash_attention.default",
         "args": [{"ref": "q"}, {"ref": "k"}, {"ref": "v"}],
         "out": [{"shape": [1,1,2,2], "dtype": "f32"}, {"shape": [1,1,2], "dtype": "f32"}]},
        {"id": "g0", "op": "_getitem", "args": [{"ref": "fa"}, {"int": 0}],
         "out": {"shape": [1,1,2,2], "dtype": "f32"}}],
      "outputs": [{"ref": "g0", "shape": [1,1,2,2], "dtype": "f32"}]
    }"#;

    #[test]
    fn sdpa_flash_variant_matches_public() {
        let qkv = [
            ("q", vec![1.0, 0.0, 0.0, 1.0]),
            ("k", vec![1.0, 0.0, 0.0, 1.0]),
            ("v", vec![1.0, 2.0, 3.0, 4.0]),
        ];
        let pubv = run_ir_n(SDPA_PUB_IR, &qkv);
        let flash = run_ir_n(SDPA_FLASH_IR, &qkv);
        assert_eq!(pubv.len(), 4);
        for (a, b) in flash.iter().zip(&pubv) {
            assert!((a - b).abs() < 1e-6, "flash {flash:?} vs public {pubv:?}");
        }
    }

    const LRELU_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "lr",
      "inputs": [{"id": "x", "shape": [4], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "lr", "op": "aten.leaky_relu.default",
                 "args": [{"ref": "x"}, {"float": 0.1}], "out": {"shape": [4], "dtype": "f32"}}],
      "outputs": [{"ref": "lr", "shape": [4], "dtype": "f32"}]
    }"#;

    #[test]
    fn leaky_relu_negative_slope() {
        // max(x, 0.1·x): [-2,-1,0,3] → [-0.2,-0.1,0,3].
        let out = run_ir(LRELU_IR, &[], ("x", vec![-2.0, -1.0, 0.0, 3.0]));
        let want = [-0.2f32, -0.1, 0.0, 3.0];
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "leaky_relu {out:?} vs {want:?}");
        }
    }

    const HSWISH_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "hw",
      "inputs": [{"id": "x", "shape": [4], "dtype": "f32"}], "weights": [],
      "nodes": [{"id": "hw", "op": "aten.hardswish.default", "args": [{"ref": "x"}],
                 "out": {"shape": [4], "dtype": "f32"}}],
      "outputs": [{"ref": "hw", "shape": [4], "dtype": "f32"}]
    }"#;

    #[test]
    fn hardswish_matches_reference() {
        // x·clamp(x/6+0.5,0,1): [-4,0,3,1] → [0,0,3,2/3].
        let out = run_ir(HSWISH_IR, &[], ("x", vec![-4.0, 0.0, 3.0, 1.0]));
        let want = [0.0f32, 0.0, 3.0, 2.0 / 3.0];
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "hardswish {out:?} vs {want:?}");
        }
    }

    // Mask produced by a compare (`x > 2` → bool), the realistic masked_fill path.
    const MFILL_IR: &str = r#"{
      "format": "rlx-torch-ir", "version": 1, "model_name": "mf",
      "inputs": [{"id": "x", "shape": [4], "dtype": "f32"},
                 {"id": "t", "shape": [4], "dtype": "f32"}], "weights": [],
      "nodes": [
        {"id": "m", "op": "aten.gt.Tensor", "args": [{"ref": "x"}, {"ref": "t"}],
         "out": {"shape": [4], "dtype": "bool"}},
        {"id": "mf", "op": "aten.masked_fill.Scalar",
         "args": [{"ref": "x"}, {"ref": "m"}, {"float": 9.0}],
         "out": {"shape": [4], "dtype": "f32"}}],
      "outputs": [{"ref": "mf", "shape": [4], "dtype": "f32"}]
    }"#;

    #[test]
    fn masked_fill_scalar() {
        // mask = x > 2 = [F,F,T,T]; masked_fill(x, mask, 9) = [1,2,9,9].
        let out = run_ir_n(
            MFILL_IR,
            &[
                ("x", vec![1.0, 2.0, 3.0, 4.0]),
                ("t", vec![2.0, 2.0, 2.0, 2.0]),
            ],
        );
        assert_eq!(out, vec![1.0, 2.0, 9.0, 9.0]);
    }
}
