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

//! ONNX compile / optimization level → native [`CompileOptions`].

use rlx_runtime::CompileOptions;

/// Compile tier for native RLX execution (mirrors ORT graph opt levels 0–3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnnxCompileLevel {
    /// No DCE / constant folding.
    Level0,
    /// DCE only.
    Level1,
    /// Default RLX pipeline.
    #[default]
    Level2,
    /// Default pipeline (same as level 2 today; reserved for heavier fusion).
    Level3,
}

impl OnnxCompileLevel {
    pub fn from_u8(n: u8) -> Self {
        match n {
            0 => Self::Level0,
            1 => Self::Level1,
            2 => Self::Level2,
            _ => Self::Level3,
        }
    }

    pub fn to_compile_options(self) -> CompileOptions {
        match self {
            Self::Level0 => CompileOptions::new()
                .with_dce(false)
                .with_constant_folding(false),
            Self::Level1 => CompileOptions::new().with_constant_folding(false),
            Self::Level2 | Self::Level3 => CompileOptions::default(),
        }
    }
}
