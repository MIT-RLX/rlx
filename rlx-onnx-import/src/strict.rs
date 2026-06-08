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

//! Fail-loud import validation.

use anyhow::{Result, bail};

use crate::lower::{ImportOptions, ImportReport};

/// When [`ImportOptions::strict`] is set, reject imports that used stubs or left ops unsupported.
pub fn validate_strict_import(opts: &ImportOptions, report: &ImportReport) -> Result<()> {
    if !opts.strict {
        return Ok(());
    }
    if !report.unsupported.is_empty() {
        bail!(
            "strict ONNX import: unsupported ops {:?} ({} nodes skipped)",
            report.unsupported,
            report.skipped
        );
    }
    if report.stubbed > 0 {
        let sample: Vec<_> = report.stubbed_nodes.iter().take(8).cloned().collect();
        bail!(
            "strict ONNX import: {} stubbed node(s), e.g. {sample:?}",
            report.stubbed
        );
    }
    Ok(())
}
