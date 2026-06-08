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

//! `#[rlx_runner]` proc-macro: per-family LM runner registration + bin main.
//!
//! Replaces the per-crate boilerplate in `rlx-<family>/src/bin/*.rs`
//! (8 lines of `fn main()` that just call `cli::run`) plus the
//! manual `register_cli(...)` call in `rlx-models::bin::rlx_run`.
//! Wires a runner into [`rlx_runtime::auto_runner_name`] via
//! `inventory`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{
    ItemFn, LitStr, Token, parse::Parse, parse::ParseStream, parse_macro_input,
    punctuated::Punctuated,
};

/// One `key = "value"` entry in the macro input.
struct Field {
    key: syn::Ident,
    _eq: Token![=],
    value: FieldValue,
}

enum FieldValue {
    Str(LitStr),
    StrList(Vec<LitStr>),
}

impl Parse for Field {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: syn::Ident = input.parse()?;
        let _eq: Token![=] = input.parse()?;
        let value = if input.peek(syn::token::Bracket) {
            let content;
            syn::bracketed!(content in input);
            let items: Punctuated<LitStr, Token![,]> = Punctuated::parse_terminated(&content)?;
            FieldValue::StrList(items.into_iter().collect())
        } else {
            FieldValue::Str(input.parse()?)
        };
        Ok(Field { key, _eq, value })
    }
}

struct RunnerArgs {
    fields: Vec<Field>,
}

impl Parse for RunnerArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let items: Punctuated<Field, Token![,]> = Punctuated::parse_terminated(input)?;
        Ok(RunnerArgs {
            fields: items.into_iter().collect(),
        })
    }
}

impl RunnerArgs {
    fn get_str(&self, key: &str) -> syn::Result<String> {
        for f in &self.fields {
            if f.key == key {
                if let FieldValue::Str(s) = &f.value {
                    return Ok(s.value());
                }
                return Err(syn::Error::new(
                    f.key.span(),
                    format!("`{key}` must be a string"),
                ));
            }
        }
        Err(syn::Error::new(
            Span::call_site(),
            format!("missing required field `{key}` (expected `{key} = \"...\"`)"),
        ))
    }

    fn get_str_list(&self, key: &str) -> Vec<String> {
        for f in &self.fields {
            if f.key == key {
                if let FieldValue::StrList(items) = &f.value {
                    return items.iter().map(|s| s.value()).collect();
                }
            }
        }
        Vec::new()
    }
}

/// `register_lm_runner!{ family = "qwen3", description = "Qwen 3 LM",
/// arches = ["qwen3", "qwen3moe"] }`
///
/// Registers the family in `rlx_runtime::ModelRegistration` so that
/// `auto_runner_name(arch, path)` can route a weights file to this
/// runner without the caller hardcoding the family.
pub(crate) fn register_lm_runner_impl(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as RunnerArgs);

    let family = match args.get_str("family") {
        Ok(s) => s,
        Err(e) => return e.to_compile_error().into(),
    };
    let description = args
        .get_str("description")
        .unwrap_or_else(|_| family.clone());
    let arches = args.get_str_list("arches");

    let arch_arms = if arches.is_empty() {
        // No arches listed → match by file extension only (`.gguf` /
        // `.safetensors`). Caller can override `matches`.
        quote! { let _ = (arch, path); false }
    } else {
        let lits = arches.iter().map(|a| {
            let a_lc = a.to_ascii_lowercase();
            quote! { #a_lc }
        });
        quote! {
            let _ = path;
            [#(#lits),*].iter().any(|a: &&str| *a == arch)
        }
    };

    let family_lit = LitStr::new(&family, Span::call_site());
    let description_lit = LitStr::new(&description, Span::call_site());

    let expanded = quote! {
        const _: () = {
            fn _matches(arch: &str, path: &::std::path::Path) -> bool {
                #arch_arms
            }
            ::rlx_runtime::lm::inventory::submit! {
                ::rlx_runtime::lm::ModelRegistration {
                    family: #family_lit,
                    description: #description_lit,
                    matches: _matches,
                }
            }
        };
    };
    TokenStream::from(expanded)
}

/// `#[rlx_runner_main(cli::run, "rlx-qwen3")]` — replaces the
/// 8-line `fn main()` in every `rlx-<family>/src/bin/*.rs`.
///
/// The attribute goes on an `extern crate` placeholder or any item;
/// only the attribute args matter. We emit a `fn main()` that
/// forwards `std::env::args().skip(1)` to the runner.
///
/// Usage:
/// ```ignore
/// // src/bin/rlx_qwen3.rs
/// rlx_macros::rlx_runner_main!(rlx_qwen3::cli::run, "rlx-qwen3");
/// ```
pub(crate) fn rlx_runner_main_impl(input: TokenStream) -> TokenStream {
    let parsed = match syn::parse::<MainArgs>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };
    let path = parsed.path;
    let name_lit = parsed.name;

    let expanded = quote! {
        fn main() -> ::std::process::ExitCode {
            let args: ::std::vec::Vec<::std::string::String> =
                ::std::env::args().skip(1).collect();
            match #path(&args) {
                ::std::result::Result::Ok(()) => ::std::process::ExitCode::SUCCESS,
                ::std::result::Result::Err(e) => {
                    ::std::eprintln!("{}: {:#}", #name_lit, e);
                    ::std::process::ExitCode::FAILURE
                }
            }
        }
    };
    TokenStream::from(expanded)
}

struct MainArgs {
    path: syn::Path,
    name: LitStr,
}

impl Parse for MainArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path: syn::Path = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let name: LitStr = input.parse()?;
        Ok(MainArgs { path, name })
    }
}

#[allow(dead_code)]
fn unused() -> TokenStream2 {
    quote! {}
}

#[allow(dead_code)]
fn _check_item_fn(_f: ItemFn) {}
