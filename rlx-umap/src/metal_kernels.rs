// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Metal host-side kernels for `umap.knn` (unified memory roundtrip).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use rlx_ir::{DType, Shape};
use rlx_metal::op_registry::{MetalKernel, register_metal_kernel};

use crate::knn_attrs::KnnAttrs;
use crate::ops::UMAP_KNN;
use rlx_cpu::umap_knn::knn_forward_packed;

#[derive(Debug)]
struct KnnForwardMetal;

impl MetalKernel for KnnForwardMetal {
    fn name(&self) -> &str {
        UMAP_KNN
    }

    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        let pairwise = unsafe { typed_f32(inputs[0].0, inputs[0].1, "pairwise")? };
        let out = unsafe { typed_f32_mut(output.0, output.1, "packed")? };
        let n = inputs[0].1.dim(0).unwrap_static();
        let n2 = inputs[0].1.dim(1).unwrap_static();
        if n != n2 {
            return Err(format!(
                "umap.knn: expected square [{n}, {n}], got [{n}, {n2}]"
            ));
        }
        let k = KnnAttrs::decode(attrs)?.k as usize;
        if out.len() != n * 2 * k {
            return Err(format!(
                "umap.knn: output len {} != n*2*k = {}",
                out.len(),
                n * 2 * k
            ));
        }
        knn_forward_packed(pairwise, n, k, out);
        Ok(())
    }
}

unsafe fn typed_f32<'a>(bytes: &'a [u8], shape: &Shape, role: &str) -> Result<&'a [f32], String> {
    if shape.dtype() != DType::F32 {
        return Err(format!("{role}: expected F32, got {:?}", shape.dtype()));
    }
    let n = shape
        .num_elements()
        .ok_or_else(|| format!("{role}: dynamic shape"))?;
    let need = n * 4;
    if bytes.len() < need {
        return Err(format!("{role}: buffer {} < {} elements", bytes.len(), n));
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), n) })
}

unsafe fn typed_f32_mut<'a>(
    bytes: &'a mut [u8],
    shape: &Shape,
    role: &str,
) -> Result<&'a mut [f32], String> {
    if shape.dtype() != DType::F32 {
        return Err(format!("{role}: expected F32, got {:?}", shape.dtype()));
    }
    let n = shape
        .num_elements()
        .ok_or_else(|| format!("{role}: dynamic shape"))?;
    let need = n * 4;
    if bytes.len() < need {
        return Err(format!("{role}: buffer {} < {} elements", bytes.len(), n));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), n) })
}

pub fn register_metal_kernels() {
    register_metal_kernel(Arc::new(KnnForwardMetal));
}
