// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Flow-level extension wiring — applies [`rlx_ir::hir_extension`] after build.

use rlx_ir::hir::HirModule;
use rlx_ir::{apply_hir_extensions, apply_hir_extensions_named};

/// Names of HIR extensions to apply when building this flow (empty = all registered).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlowExtensionPlan {
    pub enabled: Vec<String>,
    pub apply_all: bool,
}

impl FlowExtensionPlan {
    pub fn all() -> Self {
        Self {
            enabled: Vec::new(),
            apply_all: true,
        }
    }

    pub fn only(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            enabled: names.into_iter().map(Into::into).collect(),
            apply_all: false,
        }
    }

    pub fn apply(&self, hir: &mut HirModule) {
        if self.apply_all {
            apply_hir_extensions(hir);
        } else {
            let refs: Vec<&str> = self.enabled.iter().map(String::as_str).collect();
            apply_hir_extensions_named(hir, &refs);
        }
    }
}
