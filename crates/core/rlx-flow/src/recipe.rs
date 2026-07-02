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
