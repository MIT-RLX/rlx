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
#[proc_macro_attribute]
pub fn rlx_model(_attr: TokenStream, item: TokenStream) -> TokenStream {
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
