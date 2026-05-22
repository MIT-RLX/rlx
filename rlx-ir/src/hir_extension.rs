// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Retroactive HIR extensions (Slang `extension` declarations).
//!
//! Third-party or arch-specific crates register transforms that run on a built
//! [`HirModule`] before lower — without editing core block definitions.

use std::sync::{OnceLock, RwLock};

use crate::hir::HirModule;

/// Transform applied after model flow build, before MIR lower.
pub type HirExtensionFn = fn(&mut HirModule);

struct Registry {
    entries: Vec<(&'static str, HirExtensionFn)>,
}

impl Registry {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();

fn registry() -> &'static RwLock<Registry> {
    REGISTRY.get_or_init(|| RwLock::new(Registry::new()))
}

/// Register a named extension (call from `init` or model crate startup).
pub fn register_hir_extension(name: &'static str, f: HirExtensionFn) {
    let mut reg = registry().write().expect("hir extension registry");
    if reg.entries.iter().any(|(n, _)| *n == name) {
        reg.entries.retain(|(n, _)| *n != name);
    }
    reg.entries.push((name, f));
}

/// Registered extension names in registration order.
pub fn registered_hir_extensions() -> Vec<&'static str> {
    registry()
        .read()
        .expect("hir extension registry")
        .entries
        .iter()
        .map(|(n, _)| *n)
        .collect()
}

/// Apply all registered extensions in order.
pub fn apply_hir_extensions(hir: &mut HirModule) {
    let fns: Vec<HirExtensionFn> = registry()
        .read()
        .expect("hir extension registry")
        .entries
        .iter()
        .map(|(_, f)| *f)
        .collect();
    for f in fns {
        f(hir);
    }
}

/// Apply only extensions whose names appear in `names`.
pub fn apply_hir_extensions_named(hir: &mut HirModule, names: &[&str]) {
    let reg = registry().read().expect("hir extension registry");
    for (name, f) in &reg.entries {
        if names.contains(name) {
            f(hir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirMut;
    use crate::{DType, HirModule, Shape};

    fn tag_outputs(hir: &mut HirModule) {
        if let Some(id) = hir.outputs.first().copied() {
            hir.node_mut(id).name = Some("extended".into());
        }
    }

    #[test]
    fn extension_runs_on_module() {
        register_hir_extension("test_tag", tag_outputs);
        let mut hir = HirModule::new("ext");
        let mut gb = HirMut::new(&mut hir);
        let x = gb.input("x", Shape::new(&[2], DType::F32));
        hir.set_outputs(vec![x]);
        apply_hir_extensions_named(&mut hir, &["test_tag"]);
        let out = hir.node(hir.outputs[0]);
        assert_eq!(out.name.as_deref(), Some("extended"));
    }
}
