// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
