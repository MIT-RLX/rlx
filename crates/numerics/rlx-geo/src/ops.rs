// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX custom-op registration for geometry.
//!
//! * `geo.delaunay` — input `points [n, 2] I32`; output `[2n, 3] I32` triangle
//!   indices (a fixed upper-bound buffer; unused rows are `[-1, -1, -1]`, since
//!   a triangulation has fewer than `2n` triangles). CPU kernel only for now.
//! * `geo.voronoi_grid` — input `sites [n, 2] I32`, attrs `{width, height}`;
//!   output `[height, width] I32` nearest-site labels (`-1` where empty). CPU
//!   kernel + native wgpu kernel (feature `gpu`).

use std::sync::Arc;

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::{DType, OpExtension, Shape, register_op};

use crate::triangulate::triangulate;
use crate::voronoi::voronoi_grid_exact;

pub const GEO_DELAUNAY: &str = "geo.delaunay";
pub const GEO_VORONOI_GRID: &str = "geo.voronoi_grid";

/// Register every rlx-geo IR extension + kernel. Call once per process.
pub fn register_geo_ops() {
    register_op(Arc::new(DelaunayExt));
    register_cpu_kernel(Arc::new(DelaunayCpu));

    register_op(Arc::new(VoronoiGridExt));
    register_cpu_kernel(Arc::new(VoronoiGridCpu));

    #[cfg(feature = "gpu")]
    crate::wgpu_kernels::register_wgpu_geo_kernels();
}

// ---------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------

/// `geo.voronoi_grid` attributes: the output grid dimensions.
#[derive(Clone, Copy, Debug)]
pub struct VoronoiGridAttrs {
    pub width: u32,
    pub height: u32,
}

impl VoronoiGridAttrs {
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(8);
        v.extend_from_slice(&self.width.to_le_bytes());
        v.extend_from_slice(&self.height.to_le_bytes());
        v
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err(format!(
                "geo.voronoi_grid: attrs need 8 bytes (width,height), got {}",
                bytes.len()
            ));
        }
        Ok(VoronoiGridAttrs {
            width: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            height: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------------
// geo.delaunay
// ---------------------------------------------------------------------------

struct DelaunayExt;

impl OpExtension for DelaunayExt {
    fn name(&self) -> &str {
        GEO_DELAUNAY
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let n = inputs[0].dim(0).unwrap_static();
        // Fixed upper bound: a triangulation has < 2n triangles.
        Shape::new(&[2 * n, 3], DType::I32)
    }
}

struct DelaunayCpu;

impl CpuKernel for DelaunayCpu {
    fn name(&self) -> &str {
        GEO_DELAUNAY
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let pts = inputs[0].expect_i32("geo.delaunay points")?;
        let n = inputs[0].shape().dim(0).unwrap_static();
        if pts.len() != n * 2 {
            return Err(format!(
                "geo.delaunay: points must be [n,2] I32 (len {}), got {}",
                n * 2,
                pts.len()
            ));
        }
        let points: Vec<[i32; 2]> = pts.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let tris = triangulate(&points);

        let out = output.expect_i32_mut("geo.delaunay triangles")?;
        if out.len() != 2 * n * 3 {
            return Err(format!(
                "geo.delaunay: output must be [2n,3] I32 (len {}), got {}",
                2 * n * 3,
                out.len()
            ));
        }
        if tris.len() > 2 * n {
            return Err(format!(
                "geo.delaunay: {} triangles exceeds capacity {}",
                tris.len(),
                2 * n
            ));
        }
        for (k, t) in tris.iter().enumerate() {
            out[3 * k] = t[0] as i32;
            out[3 * k + 1] = t[1] as i32;
            out[3 * k + 2] = t[2] as i32;
        }
        for slot in out[3 * tris.len()..].iter_mut() {
            *slot = -1; // sentinel: unused triangle row
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// geo.voronoi_grid
// ---------------------------------------------------------------------------

struct VoronoiGridExt;

impl OpExtension for VoronoiGridExt {
    fn name(&self) -> &str {
        GEO_VORONOI_GRID
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, _inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let a = VoronoiGridAttrs::decode(attrs).expect("geo.voronoi_grid attrs");
        Shape::new(&[a.height as usize, a.width as usize], DType::I32)
    }
}

struct VoronoiGridCpu;

impl CpuKernel for VoronoiGridCpu {
    fn name(&self) -> &str {
        GEO_VORONOI_GRID
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = VoronoiGridAttrs::decode(attrs)?;
        let sites_i32 = inputs[0].expect_i32("geo.voronoi_grid sites")?;
        let sites: Vec<[i32; 2]> = sites_i32.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        let labels = voronoi_grid_exact(&sites, a.width, a.height);

        let out = output.expect_i32_mut("geo.voronoi_grid labels")?;
        if out.len() != labels.len() {
            return Err(format!(
                "geo.voronoi_grid: output len {} != width*height {}",
                out.len(),
                labels.len()
            ));
        }
        for (o, &l) in out.iter_mut().zip(labels.iter()) {
            *o = l as i32; // u32::MAX -> -1 sentinel (empty cell)
        }
        Ok(())
    }
}
