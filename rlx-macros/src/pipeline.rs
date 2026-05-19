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

//! Compile-time pipeline scheduler (plan #11).
//!
//! Borrowed from MAX's `max/kernels/src/pipeline/` `comptime`
//! scheduler that replaced 800 lines of hand-written schedule
//! tables with a dependency graph + cost model. The Rust spelling
//! is a proc macro: declare stages and their `depends_on` lists,
//! the macro topologically sorts them at compile time, detects
//! cycles, and emits a const array describing the schedule.
//!
//! Zero runtime cost: a runtime "execute the pipeline" walk is
//! just iterating a const slice.
//!
//! Syntax:
//! ```ignore
//! use rlx_macros::pipeline_schedule;
//!
//! pipeline_schedule! {
//!     name: AttentionBlock,
//!     stages: {
//!         qkv_proj   => [],
//!         narrow_q   => [qkv_proj],
//!         narrow_k   => [qkv_proj],
//!         narrow_v   => [qkv_proj],
//!         rope_q     => [narrow_q],
//!         rope_k     => [narrow_k],
//!         attention  => [rope_q, rope_k, narrow_v],
//!         out_proj   => [attention],
//!     }
//! }
//!
//! // Emits:
//! //   pub struct AttentionBlock;
//! //   impl AttentionBlock {
//! //     pub const ORDER: &'static [&'static str] = &[...];
//! //     pub const DEPS:  &'static [(&'static str, &'static [&'static str])] = &[...];
//! //   }
//! ```

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::{HashMap, HashSet};
use syn::{
    Ident, Result,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token,
};

/// One stage line: `name => [dep1, dep2, ...]`.
struct Stage {
    name: Ident,
    deps: Vec<Ident>,
}

impl Parse for Stage {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<token::FatArrow>()?;
        let content;
        syn::bracketed!(content in input);
        let deps: Punctuated<Ident, token::Comma> =
            content.parse_terminated(Ident::parse, token::Comma)?;
        Ok(Self {
            name,
            deps: deps.into_iter().collect(),
        })
    }
}

/// Whole macro input: `name: Foo, stages: { stage_a => [...], ... }`.
struct PipelineInput {
    name: Ident,
    stages: Vec<Stage>,
}

impl Parse for PipelineInput {
    fn parse(input: ParseStream) -> Result<Self> {
        // `name: Foo,`
        let name_kw: Ident = input.parse()?;
        if name_kw != "name" {
            return Err(syn::Error::new(
                name_kw.span(),
                "expected `name: <Ident>` as the first field",
            ));
        }
        input.parse::<token::Colon>()?;
        let name: Ident = input.parse()?;
        input.parse::<token::Comma>()?;

        // `stages: { ... }`
        let stages_kw: Ident = input.parse()?;
        if stages_kw != "stages" {
            return Err(syn::Error::new(
                stages_kw.span(),
                "expected `stages: { ... }` as the second field",
            ));
        }
        input.parse::<token::Colon>()?;
        let braced;
        syn::braced!(braced in input);
        let parsed: Punctuated<Stage, token::Comma> =
            braced.parse_terminated(Stage::parse, token::Comma)?;
        Ok(Self {
            name,
            stages: parsed.into_iter().collect(),
        })
    }
}

/// Topologically sort `stages` (Kahn's algorithm). Errors out at
/// compile time on cycles or unknown deps.
fn topo_sort(stages: &[Stage]) -> std::result::Result<Vec<String>, String> {
    let names: Vec<String> = stages.iter().map(|s| s.name.to_string()).collect();
    let name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();

    // Validate dep names exist.
    for s in stages {
        for d in &s.deps {
            if !name_set.contains(d.to_string().as_str()) {
                return Err(format!(
                    "stage `{}` depends on `{}` which is not declared",
                    s.name, d
                ));
            }
        }
    }

    // Build indegree + adjacency.
    let mut indeg: HashMap<String, usize> = names.iter().map(|n| (n.clone(), 0)).collect();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for s in stages {
        let me = s.name.to_string();
        for d in &s.deps {
            let dep = d.to_string();
            adj.entry(dep).or_default().push(me.clone());
            *indeg.get_mut(&me).unwrap() += 1;
        }
    }

    // Kahn's: iterate ready set in deterministic (declaration-order) order.
    let mut order: Vec<String> = Vec::with_capacity(stages.len());
    let mut ready: Vec<String> = names
        .iter()
        .filter(|n| indeg[n.as_str()] == 0)
        .cloned()
        .collect();
    while let Some(n) = ready.pop() {
        order.push(n.clone());
        if let Some(succs) = adj.get(&n) {
            for s in succs {
                let d = indeg.get_mut(s).unwrap();
                *d -= 1;
                if *d == 0 {
                    ready.push(s.clone());
                }
            }
        }
    }

    if order.len() != stages.len() {
        let unresolved: Vec<&str> = indeg
            .iter()
            .filter(|(_, d)| **d > 0)
            .map(|(n, _)| n.as_str())
            .collect();
        return Err(format!(
            "pipeline has a dependency cycle through: {}",
            unresolved.join(", ")
        ));
    }

    Ok(order)
}

pub fn pipeline_schedule_impl(input: TokenStream2) -> TokenStream2 {
    let parsed: PipelineInput = match syn::parse2(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error(),
    };
    let order = match topo_sort(&parsed.stages) {
        Ok(o) => o,
        Err(msg) => {
            return quote! { compile_error!(#msg); };
        }
    };

    let name = parsed.name;
    let order_lits: Vec<_> = order.iter().map(|s| quote! { #s }).collect();

    // For DEPS, preserve the user-declared dependency lists keyed by
    // stage name (for tooling / visualization / debug).
    let dep_pairs: Vec<TokenStream2> = parsed
        .stages
        .iter()
        .map(|s| {
            let stage_str = s.name.to_string();
            let deps: Vec<_> = s
                .deps
                .iter()
                .map(|d| {
                    let s = d.to_string();
                    quote! { #s }
                })
                .collect();
            // Note: emit a fixed-size array reference, not a slice. The
            // `: &'static [&'static str]` field type coerces it.
            if deps.is_empty() {
                quote! { (#stage_str, &[]) }
            } else {
                quote! { (#stage_str, &[#(#deps),*]) }
            }
        })
        .collect();

    quote! {
        /// Auto-generated pipeline schedule (plan #11).
        ///
        /// `ORDER` is the topologically-sorted stage execution
        /// order (declaration order broken ties). `DEPS` preserves
        /// the original (stage, deps) pairs for debug tooling.
        pub struct #name;

        impl #name {
            pub const ORDER: &'static [&'static str] = &[
                #(#order_lits),*
            ];

            pub const DEPS: &'static [(&'static str, &'static [&'static str])] = &[
                #(#dep_pairs),*
            ];
        }
    }
}
