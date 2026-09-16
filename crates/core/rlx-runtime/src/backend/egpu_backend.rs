// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! External GPU backend adapter (`Device::Egpu`) — wraps `rlx-egpu`.
//!
//! Discovery + transport seam only. `rlx-egpu` finds an unclaimed display-class
//! device on a PCIe tunnel and, with its `dext` feature, reaches config
//! space, BARs, and DMA memory through a driver extension. Device bring-up
//! (firmware load, memory controller, rings, page tables) is not implemented, so
//! there is no execution path: `supported_ops` is empty and `compile` surfaces
//! `rlx_egpu`'s diagnostic rather than falling back to the CPU.
use super::*;
use rlx_ir::OpKind;

pub struct EgpuBackend;

impl Backend for EgpuBackend {
    fn supported_ops(&self) -> &'static [OpKind] {
        rlx_egpu::SUPPORTED_OPS
    }

    fn compile(&self, _graph: Graph, _options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        panic!("EgpuBackend: {}", rlx_egpu::diagnostic());
    }

    fn compile_lir(&self, lir: LirModule, options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        self.compile(lir.into_graph(), options)
    }
}
