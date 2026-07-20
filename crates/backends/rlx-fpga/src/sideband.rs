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

//! Resolve Graph-bound scalar Inputs into [`SidebandSpec`] soft ports.

use rlx_ir::{DType, Graph, Op};

use crate::codegen::io_ports::sanitize_port;
use crate::export_config::{IoConfig, SidebandSpec};

/// Merge `bind.sideband_inputs` into `io.sidebands` (widths from tensor shapes).
///
/// Sideband Inputs must be scalar or a single element (`numel == 1`) and must
/// not be the primary activation Input.
pub fn resolve_graph_sidebands(g: &Graph, io: &mut IoConfig) -> Result<(), String> {
    let primary = io.bind.input.clone().or_else(|| {
        g.nodes().iter().find_map(|n| match &n.op {
            Op::Input { name } => Some(name.clone()),
            _ => None,
        })
    });

    for name in io.bind.sideband_inputs.clone() {
        if primary.as_deref() == Some(name.as_str()) {
            return Err(format!(
                "rlx-fpga: sideband_input {name:?} is also the activation input — \
                 pick a different bind.input or remove it from sideband_inputs"
            ));
        }
        if io
            .sidebands
            .iter()
            .any(|s| s.name == name || sanitize_port(&name) == s.name)
        {
            continue; // explicit SidebandSpec wins
        }
        let id = g.input_id(&name).ok_or_else(|| {
            format!("rlx-fpga: sideband_input {name:?} does not match any Op::Input")
        })?;
        let shape = g.shape(id);
        let numel = shape.num_elements().unwrap_or(0);
        if numel != 1 {
            return Err(format!(
                "rlx-fpga: sideband_input {name:?} has {numel} elements; \
                 sidebands must be scalar (numel=1)"
            ));
        }
        let bits = match shape.dtype() {
            DType::Bool => 1u8,
            other => {
                let b = (other.size_bytes() * 8) as u8;
                b.clamp(1, 64)
            }
        };
        let signed = matches!(
            shape.dtype(),
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::F16
                | DType::BF16
                | DType::F32
                | DType::F64
                | DType::C64
                | DType::C128
        );
        io.sidebands.push(SidebandSpec {
            name: sanitize_port(&name),
            bits,
            signed,
            echo: true,
        });
    }

    for sb in &mut io.sidebands {
        sb.name = sanitize_port(&sb.name);
        sb.bits = sb.bits.clamp(1, 64);
    }

    let mut seen = std::collections::HashSet::new();
    for sb in &io.sidebands {
        if !seen.insert(sb.name.clone()) {
            return Err(format!(
                "rlx-fpga: duplicate sideband port name {:?}",
                sb.name
            ));
        }
    }
    Ok(())
}
