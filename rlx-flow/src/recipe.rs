// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Composable model recipes — arch presets that remain patchable.

use crate::flow::ModelFlow;

/// Assemble a [`ModelFlow`] from config — use for arch-specific presets (LLaMA, Qwen, FLUX, …).
///
/// Recipes return an unbuilt flow so callers can still `.raw_stage()`, `.custom()`, or
/// `.patch()` before `build()`.
pub trait ModelRecipe {
    fn name(&self) -> &str;
    fn assemble(&self) -> ModelFlow;
}

impl<F> ModelRecipe for F
where
    F: Fn() -> ModelFlow,
{
    fn name(&self) -> &str {
        "closure_recipe"
    }

    fn assemble(&self) -> ModelFlow {
        self()
    }
}
