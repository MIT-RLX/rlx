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

// RLX — versatile ML compiler + runtime.
//
// Pageable or pinned host staging for faster H2D/D2H on the CUDA run hot path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError, PinnedHostSlice};

/// Host-side f32 buffer used for input upload / output download.
pub enum F32HostSlot {
    Pageable(Vec<f32>),
    Pinned(PinnedHostSlice<f32>),
}

impl F32HostSlot {
    pub fn new(ctx: &Arc<CudaContext>, len: usize, pinned: bool) -> Self {
        if pinned {
            Self::Pinned(
                unsafe { ctx.alloc_pinned::<f32>(len) }
                    .unwrap_or_else(|e| panic!("rlx-cuda: pinned host alloc failed: {e}")),
            )
        } else {
            Self::Pageable(vec![0.0f32; len])
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Pageable(v) => v.len(),
            Self::Pinned(p) => p.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_from_host(&mut self, data: &[f32]) {
        match self {
            Self::Pageable(v) => {
                debug_assert!(data.len() <= v.len());
                v[..data.len()].copy_from_slice(data);
            }
            Self::Pinned(p) => {
                debug_assert!(data.len() <= p.len());
                let dst = p
                    .as_mut_slice()
                    .expect("rlx-cuda: pinned input staging unavailable");
                dst[..data.len()].copy_from_slice(data);
            }
        }
    }

    pub fn htod(
        &self,
        stream: &Arc<CudaStream>,
        dst: &mut cudarc::driver::CudaViewMut<f32>,
        len: usize,
    ) -> Result<(), DriverError> {
        debug_assert!(len <= self.len());
        match self {
            Self::Pageable(v) => stream.memcpy_htod(&v[..len], dst),
            Self::Pinned(p) => stream.memcpy_htod(p, dst),
        }
    }

    pub fn dtoh(
        &mut self,
        stream: &Arc<CudaStream>,
        src: &cudarc::driver::CudaView<f32>,
    ) -> Result<(), DriverError> {
        match self {
            Self::Pageable(v) => stream.memcpy_dtoh(src, v.as_mut_slice()),
            Self::Pinned(p) => stream.memcpy_dtoh(src, p),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Pageable(v) => v.as_slice(),
            Self::Pinned(p) => p.as_slice().expect("rlx-cuda: pinned output read failed"),
        }
    }

    pub fn copy_into(&self, dst: &mut [f32]) {
        let src = self.as_slice();
        debug_assert!(dst.len() <= src.len());
        dst.copy_from_slice(&src[..dst.len()]);
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}
