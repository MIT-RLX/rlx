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

//! RLX proc macros for AOT model compilation.
//!
//! `#[rlx_model]` transforms a function that uses the RLX tracing API
//! into an optimized, cached, zero-overhead execution path.
//!
//! # Usage
//! ```rust,ignore
//! use rlx_macros::rlx_model;
//! use rlx_runtime::trace::*;
//!
//! #[rlx_model]
//! fn my_encoder(t: &Tracer) -> Vec<TracedTensor> {
//!     let x = t.input("x", &[4, 15, 384], DType::F32);
//!     let w = t.param("w", &[384, 1536], DType::F32);
//!     let b = t.param("b", &[1536], DType::F32);
//!     let out = t.matmul(x, w);
//!     let out = (out + b).gelu();
//!     vec![out]
//! }
//!
//! // Generated: my_encoder_compiled() returns a cached CompiledGraph
//! // that's built once and reused on every call.
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

mod lm_runner;
mod pipeline;

/// Compile-time pipeline scheduler (plan #11). See `pipeline_schedule_impl`
/// in this crate's private `pipeline` module for the full grammar.
///
/// ```ignore
/// pipeline_schedule! {
///     name: AttentionBlock,
///     stages: {
///         qkv_proj => [],
///         narrow_q => [qkv_proj],
///         attention => [narrow_q],
///     }
/// }
/// ```
///
/// Emits a unit struct + `ORDER`/`DEPS` const slices, with
/// topological sort + cycle detection at compile time.
#[proc_macro]
pub fn pipeline_schedule(item: TokenStream) -> TokenStream {
    pipeline::pipeline_schedule_impl(item.into()).into()
}

/// AOT compilation macro for RLX models.
///
/// Wraps a tracing function with a `static OnceCell` cache that:
/// 1. On first call: traces the function → builds IR graph → fuses → compiles thunks
/// 2. On subsequent calls: executes pre-compiled thunks (zero overhead)
///
/// The original function becomes the "graph builder". A new `_compiled` function
/// is generated that manages the cache and execution.
///
/// # Opt-in self-check
/// `#[rlx_model(check)]` injects a call to
/// [`rlx_runtime::check::model_self_check`] right after the graph is traced, so
/// building the model surfaces shape/dtype, backend-dispatch, missed-fusion and
/// numeric findings on stderr. It runs on the CPU reference backend by default;
/// tune with `RLX_CHECK` (`off` / `all` / `strict`). No extra dependency — the
/// generated code already routes through `::rlx_runtime`.
#[proc_macro_attribute]
pub fn rlx_model(attr: TokenStream, item: TokenStream) -> TokenStream {
    // `#[rlx_model(check)]` opts this model into the post-trace self-check.
    let want_check = attr
        .to_string()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|w| w == "check");
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_block = &input_fn.block;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;

    // Generate the compiled version name
    let compiled_name = syn::Ident::new(&format!("{fn_name}_compiled"), fn_name.span());

    // The graph builder function name (original, kept for debugging)
    let builder_name = syn::Ident::new(&format!("{fn_name}_build_graph"), fn_name.span());

    // Optional post-trace self-check (see `#[rlx_model(check)]`).
    let check_hook = if want_check {
        quote! { ::rlx_runtime::check::model_self_check(stringify!(#fn_name), &graph); }
    } else {
        quote! {}
    };

    let expanded = quote! {
        /// Graph builder (the original function — builds IR graph via tracing).
        fn #builder_name(#fn_inputs) #fn_output {
            #fn_block
        }

        /// Compiled model — traces once, caches, executes with zero overhead.
        ///
        /// Returns a reference to the cached `CompiledGraph`. Call `.run()` or
        /// `.run_raw()` to execute.
        #fn_vis fn #compiled_name() -> &'static ::std::sync::Mutex<::rlx_runtime::CompiledGraph> {
            use ::std::sync::{Mutex, OnceLock};

            static COMPILED: OnceLock<Mutex<::rlx_runtime::CompiledGraph>> = OnceLock::new();

            COMPILED.get_or_init(|| {
                // Trace the function to build the IR graph
                let graph = ::rlx_runtime::trace::trace(stringify!(#fn_name), |t| {
                    #builder_name(t)
                });

                // Opt-in `#[rlx_model(check)]` self-check (no-op otherwise).
                #check_hook

                // Compile: fuse → memory plan → thunks
                let session = ::rlx_runtime::Session::new(::rlx_runtime::Device::Cpu);
                let compiled = session.compile(graph);

                Mutex::new(compiled)
            })
        }

        // Keep original function accessible for debugging
        #[allow(dead_code)]
        #input_fn
    };

    TokenStream::from(expanded)
}

/// Register a per-family LM runner so [`rlx_runtime::auto_runner_name`]
/// can route a weights file to it.
///
/// ```ignore
/// rlx_macros::register_lm_runner! {
///     family = "qwen3",
///     description = "Qwen 3 LM",
///     arches = ["qwen3", "qwen3moe"]
/// }
/// ```
///
/// Backed by `inventory` at startup; no per-bin `register_cli` call
/// is needed once each family invokes this macro at the crate root.
#[proc_macro]
pub fn register_lm_runner(input: TokenStream) -> TokenStream {
    lm_runner::register_lm_runner_impl(input)
}

/// `fn main()` for a per-family runner binary. Replaces the 8-line
/// boilerplate at the top of every `rlx-<family>/src/bin/rlx_*.rs`.
///
/// ```ignore
/// // src/bin/rlx_qwen3.rs
/// rlx_macros::rlx_runner_main!(rlx_qwen3::cli::run, "rlx-qwen3");
/// ```
#[proc_macro]
pub fn rlx_runner_main(input: TokenStream) -> TokenStream {
    lm_runner::rlx_runner_main_impl(input)
}
