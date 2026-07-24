// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Import MLX weight dumps into an `.rlxp` package.

use crate::tier::StorageTier;
use crate::write::{ContainerKind, PackedWeight, WriteOptions, write_package};
use anyhow::{Context, Result};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Shape};
use rlx_mlx_io::{
    PackedLinearBinding, build_llama_like_prefill, build_parallel_dequant_graph,
    collect_packed_linears, load_path,
};
use std::path::Path;

/// Options for [`mlx_to_rlxp`].
#[derive(Debug, Clone)]
pub struct MlxImportOptions {
    pub container: ContainerKind,
    pub include_graph: bool,
    pub compress_sidecars: bool,
    pub auto_tier: bool,
    /// When true (default), dequantize affine/mxfp layers to f32 before packing.
    /// When false, keep `{base}.weight/.scales/.biases` as packed scheme rows
    /// for `Op::DequantMatMul`.
    pub dequant_to_f32: bool,
    /// Prefill batch / seq when embedding a Llama-like graph (`include_graph`
    /// + keep-packed + arch in config). Defaults: batch=1, seq=1.
    pub graph_batch: usize,
    pub graph_seq: usize,
    /// Cap decoder layers in the embedded graph (`None` = all).
    pub graph_num_layers: Option<usize>,
}

impl Default for MlxImportOptions {
    fn default() -> Self {
        Self {
            container: ContainerKind::Flat,
            include_graph: false,
            compress_sidecars: true,
            auto_tier: true,
            dequant_to_f32: true,
            graph_batch: 1,
            graph_seq: 1,
            graph_num_layers: None,
        }
    }
}

fn push_packed_linear(weights: &mut Vec<PackedWeight>, b: &PackedLinearBinding) {
    let n = b.packed.out_shape[0];
    let n_groups = b.packed.n_groups().max(1);
    // Logical `[n, k]` for DequantMatMul; packed bytes length is `data.len()`.
    weights.push(PackedWeight {
        name: format!("{}.weight", b.name),
        shape: b.packed.out_shape.clone(),
        scheme: b.packed.scheme.to_string(),
        layout: "mlx_nk".into(),
        data: b.packed.w_q.clone(),
        rank: None,
        tier: StorageTier::Hot,
    });
    weights.push(PackedWeight {
        name: format!("{}.scales", b.name),
        shape: vec![n, n_groups],
        scheme: if matches!(b.packed.scheme, QuantScheme::MlxAffine { .. }) {
            "f32".into()
        } else {
            "u8".into()
        },
        layout: "row_major".into(),
        data: b.packed.scales.clone(),
        rank: None,
        tier: StorageTier::Hot,
    });
    if matches!(b.packed.scheme, QuantScheme::MlxAffine { .. }) {
        weights.push(PackedWeight {
            name: format!("{}.biases", b.name),
            shape: vec![n, n_groups],
            scheme: "f32".into(),
            layout: "row_major".into(),
            data: b.packed.biases.clone(),
            rank: None,
            tier: StorageTier::Hot,
        });
    }
}

fn push_remaining_dense(
    loaded: &mut rlx_mlx_io::MlxWeights,
    weights: &mut Vec<PackedWeight>,
) -> Result<()> {
    for name in loaded.logical_keys() {
        if let Ok((data, shape)) = loaded.take_dense_f32(&name) {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for x in &data {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            weights.push(PackedWeight {
                name,
                shape,
                scheme: "f32".into(),
                layout: "row_major".into(),
                data: bytes,
                rank: None,
                tier: StorageTier::Hot,
            });
        }
    }
    Ok(())
}

/// Convert an MLX weight path (dir / `.safetensors` / `.npz` / `.npy`) → `.rlxp`.
pub fn mlx_to_rlxp(
    mlx_path: impl AsRef<Path>,
    out: impl AsRef<Path>,
    opts: &MlxImportOptions,
) -> Result<()> {
    let mlx_path = mlx_path.as_ref();
    let mut loaded =
        load_path(mlx_path).with_context(|| format!("open mlx {}", mlx_path.display()))?;
    let config_bytes = loaded
        .config
        .raw
        .as_ref()
        .map(serde_json::to_vec_pretty)
        .transpose()?;

    let gname = mlx_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mlx")
        .to_string();

    let mut weights = Vec::new();
    let graph = if opts.dequant_to_f32 {
        let shaped = loaded.into_shaped_f32()?;
        for (name, (shape, data)) in shaped {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for x in &data {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            weights.push(PackedWeight {
                name,
                shape,
                scheme: "f32".into(),
                layout: "row_major".into(),
                data: bytes,
                rank: None,
                tier: StorageTier::Hot,
            });
        }
        if opts.include_graph {
            let mut g = Graph::new(&gname);
            let s = Shape::new(&[1], DType::F32);
            let x = g.input("x", s);
            g.set_outputs(vec![x]);
            g
        } else {
            Graph::new(&gname)
        }
    } else {
        let linears = collect_packed_linears(&mut loaded)?;
        for b in &linears {
            push_packed_linear(&mut weights, b);
        }
        push_remaining_dense(&mut loaded, &mut weights)?;
        if opts.include_graph && !linears.is_empty() {
            if let Some(arch) = loaded.config.arch.as_ref() {
                build_llama_like_prefill(
                    &gname,
                    arch,
                    &linears,
                    opts.graph_batch.max(1),
                    opts.graph_seq.max(1),
                    opts.graph_num_layers,
                )?
            } else {
                build_parallel_dequant_graph(&gname, &linears, 1)?
            }
        } else if opts.include_graph {
            let mut g = Graph::new(&gname);
            let s = Shape::new(&[1], DType::F32);
            let x = g.input("x", s);
            g.set_outputs(vec![x]);
            g
        } else {
            Graph::new(&gname)
        }
    };

    let mut wopts = WriteOptions {
        name: gname,
        producer: Some("rlx-pkg/mlx_import".into()),
        container: opts.container,
        compress_sidecars: opts.compress_sidecars,
        include_graph: opts.include_graph,
        write_checksums: true,
        ..WriteOptions::default()
    };

    if opts.auto_tier {
        crate::auto_tier::apply_auto_tier(&mut weights, &Default::default());
    }

    if let Some(bytes) = config_bytes {
        wopts
            .sidecars
            .push(("config.json".into(), "application/json".into(), bytes));
    }

    write_package(out.as_ref(), &graph, &weights, &wopts)
        .with_context(|| format!("write {}", out.as_ref().display()))?;
    Ok(())
}
