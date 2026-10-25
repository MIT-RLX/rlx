// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fails when rlx-ir gains a public graph builder the op table does not reach.
//!
//! Every `pub fn (&mut self, …)` on `impl Graph` and every `GraphExt` method is
//! a builder. Each must either be called from a row of the table
//! (`src/table.rs`) or appear in [`NOT_BOUND`] with the reason it is not part
//! of the binding surface. Adding an op to the IR therefore forces a decision
//! here — bind it (one row, every language gets it) or say why not.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

/// IR builders deliberately left off the binding surface, with the reason.
const NOT_BOUND: &[(&str, &str)] = &[
    // Backward kernels: autodiff emits these; scripts reach them via grad/jvp.
    ("activation_backward", "backward kernel"),
    ("ada_layer_norm_backward", "backward kernel"),
    ("attention_backward", "backward kernel"),
    ("attention_backward_all", "backward kernel"),
    ("batch_norm_inference_backward_beta", "backward kernel"),
    ("batch_norm_inference_backward_gamma", "backward kernel"),
    ("batch_norm_inference_backward_input", "backward kernel"),
    ("complex_norm_sq_backward", "backward kernel"),
    ("conv2d_backward_input", "backward kernel"),
    ("conv2d_backward_weight", "backward kernel"),
    ("conv3d_backward_input", "backward kernel"),
    ("conv3d_backward_weight", "backward kernel"),
    ("cumsum_backward", "backward kernel"),
    ("gated_residual_backward", "backward kernel"),
    ("gather_backward", "backward kernel"),
    ("group_norm_backward_beta", "backward kernel"),
    ("group_norm_backward_gamma", "backward kernel"),
    ("group_norm_backward_input", "backward kernel"),
    ("layer_norm_backward_gamma", "backward kernel"),
    ("layer_norm_backward_input", "backward kernel"),
    ("log_mel_backward", "backward kernel"),
    ("maxpool2d_backward", "backward kernel"),
    ("maxpool3d_backward", "backward kernel"),
    ("relu_backward", "backward kernel"),
    ("rms_norm_backward_beta", "backward kernel"),
    ("rms_norm_backward_gamma", "backward kernel"),
    ("rms_norm_backward_input", "backward kernel"),
    ("rope_backward", "backward kernel"),
    ("scan_backward", "backward kernel"),
    ("scan_backward_with_checkpoints", "backward kernel"),
    ("scan_backward_xs", "backward kernel"),
    ("scan_backward_xs_with_checkpoints", "backward kernel"),
    ("softmax_cross_entropy_backward", "backward kernel"),
    // Gaussian splatting is an out-of-tree seam (`rlx-splat` registers the
    // kernels); the IR keeps it, the bindings do not expose it.
    ("gaussian_splat_prepare", "splat seam, not bound"),
    ("gaussian_splat_rasterize", "splat seam, not bound"),
    ("gaussian_splat_render", "splat seam, not bound"),
    ("gaussian_splat_render_backward", "splat seam, not bound"),
    ("gaussian_splat_render_decomposed", "splat seam, not bound"),
    ("gaussian_splat_render_meta", "splat seam, not bound"),
    // Lowering / backend-internal forms the compiler produces, not authoring ops.
    ("conv3d_im2col", "lowering form of conv3d"),
    ("fft_butterfly_stage", "lowering form of fft"),
    ("fft_q", "quantized FFT lowering stage"),
    ("fused_swiglu_ffn", "fusion output; author with swiglu_ffn"),
    (
        "interpolate3d_nearest",
        "covered by interpolate3d(mode=\"nearest\")",
    ),
    ("linear_fused", "fusion output; author with linear_bias_act"),
    ("partitioned_conv", "lowering form of partitioned_conv1d"),
    (
        "partitioned_conv1d_gemm",
        "lowering form of partitioned_conv1d",
    ),
    (
        "synth_matmul_qcodebook",
        "quantized-codebook lowering of synth_matmul",
    ),
    ("spd_exp_map_graph", "graph-composed form of spd_exp_map"),
    ("spd_expm_batch", "covered by spd_matrix_fn(kind=\"expm\")"),
    (
        "spd_invsqrt_batched",
        "covered by spd_matrix_fn(kind=\"invsqrtm\")",
    ),
    (
        "spd_invsqrtm_batch",
        "covered by spd_matrix_fn(kind=\"invsqrtm\")",
    ),
    (
        "spd_invsqrtm_batch_graph",
        "graph-composed form of spd_matrix_fn",
    ),
    ("spd_log_map_graph", "graph-composed form of spd_log_map"),
    ("spd_logeig_batched", "covered by logeig"),
    ("spd_logm_batch", "covered by spd_matrix_fn(kind=\"logm\")"),
    (
        "spd_logm_batch_graph",
        "graph-composed form of spd_matrix_fn",
    ),
    (
        "spd_matrix_fn_batch_graph",
        "graph-composed form of spd_matrix_fn",
    ),
    ("spd_poweig_batched", "eigen-power lowering stage"),
    ("spd_reeig_batched", "covered by reeig"),
    (
        "spd_sqrt_batched",
        "covered by spd_matrix_fn(kind=\"sqrtm\")",
    ),
    (
        "spd_sqrtm_batch",
        "covered by spd_matrix_fn(kind=\"sqrtm\")",
    ),
    (
        "spd_sqrtm_batch_graph",
        "graph-composed form of spd_matrix_fn",
    ),
    // Graph surgery, not op construction.
    ("append_node", "raw node insertion"),
    ("node_mut", "raw node access"),
    ("nodes_mut", "raw node access"),
    ("set_inputs", "raw input rewiring"),
    ("set_vars", "raw graph-state rewiring"),
    ("set_sym_attrs", "raw graph-state rewiring"),
    ("set_sym_defs", "raw graph-state rewiring"),
    (
        "splice",
        "graph composition; api.idl fn `splice` (runtime::compose)",
    ),
    ("lane_mask_like_", "batched-lane bookkeeping for vmap"),
    ("reset_lanes_", "batched-lane bookkeeping for vmap"),
    ("reset_lanes_many_", "batched-lane bookkeeping for vmap"),
    ("reset_lanes_to_", "batched-lane bookkeeping for vmap"),
    // Output names: `set_outputs` + `name_output` (both rows) say the same.
    ("set_named_outputs", "set_outputs + name_output"),
    ("set_output_names", "name_output, once per output"),
    // `FftAlgo` (rounding model) has no binding kind yet; `fft_norm` is the
    // default algorithm in every language.
    (
        "fft_algo",
        "FftAlgo is not a binding kind yet; fft_norm covers the default",
    ),
];

/// Builders the table reaches under another IR spelling (the `GraphExt`
/// shorthand wraps the explicit-shape builder, or vice versa).
const SPELLED_OTHERWISE: &[(&str, &str)] = &[
    ("attention", "attention_ (GraphExt)"),
    ("concat", "concat_ (GraphExt)"),
    ("constant", "try_constant"),
    ("cummax", "cummax_ (GraphExt)"),
    ("cumprod", "cumprod_ (GraphExt)"),
    ("custom_op", "try_custom_op / custom_op_packed"),
    ("eq", "eq_ row via Op::Compare"),
    ("gather", "gather_ (GraphExt)"),
    ("ln", "log"),
    ("lt", "lt_ row via Op::Compare"),
    ("max_pool2d", "pool2d(ReduceOp::Max)"),
    ("avg_pool2d", "pool2d(ReduceOp::Mean)"),
    ("reshape", "reshape_ (GraphExt)"),
    ("rope", "rope_styled"),
    ("rope_n", "rope_n_styled"),
    ("scaled_grouped_matmul", "scaled_grouped_matmul_bias"),
    ("slice_", "slice_ (GraphExt)"),
    ("sm", "softmax row"),
    ("softmax", "sm (GraphExt)"),
    ("zeros", "full"),
];

/// Public `&mut self` methods of `impl Graph` / `impl GraphExt for Graph`.
struct Builders(BTreeSet<String>);

impl<'a> Visit<'a> for Builders {
    fn visit_item_impl(&mut self, i: &'a syn::ItemImpl) {
        let on_graph = matches!(&*i.self_ty, syn::Type::Path(p)
            if p.path.segments.last().is_some_and(|s| s.ident == "Graph"));
        if !on_graph {
            return;
        }
        for item in &i.items {
            let syn::ImplItem::Fn(f) = item else { continue };
            let public = matches!(f.vis, syn::Visibility::Public(_)) || i.trait_.is_some();
            let mut_self = matches!(f.sig.inputs.first(),
                Some(syn::FnArg::Receiver(r)) if r.mutability.is_some());
            if public && mut_self {
                self.0.insert(f.sig.ident.to_string());
            }
        }
    }
}

/// Every `<recv>.method(` call in the table — what the rows reach.
struct Calls(BTreeSet<String>);

impl<'a> Visit<'a> for Calls {
    fn visit_expr_method_call(&mut self, m: &'a syn::ExprMethodCall) {
        self.0.insert(m.method.to_string());
        syn::visit::visit_expr_method_call(self, m);
    }

    fn visit_macro(&mut self, m: &'a syn::Macro) {
        // The table is a macro body: parse its rows' blocks out of the tokens.
        let tokens = m.tokens.clone();
        let mut stack: Vec<proc_macro2::TokenStream> = vec![tokens];
        while let Some(ts) = stack.pop() {
            for tt in ts {
                if let proc_macro2::TokenTree::Group(gr) = tt {
                    if let Ok(block) = syn::parse2::<syn::Block>(quote_block(&gr)) {
                        self.visit_block(&block);
                    }
                    stack.push(gr.stream());
                }
            }
        }
    }
}

/// `gr` as a `{ … }` block, with macro-definition `$` tokens dropped so
/// `$crate::shapes::same(…)` parses as `crate::shapes::same(…)`.
fn quote_block(gr: &proc_macro2::Group) -> proc_macro2::TokenStream {
    let mut ts = proc_macro2::TokenStream::new();
    ts.extend([proc_macro2::TokenTree::Group(proc_macro2::Group::new(
        proc_macro2::Delimiter::Brace,
        strip_dollar(gr.stream()),
    ))]);
    ts
}

fn strip_dollar(ts: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    ts.into_iter()
        .filter(|tt| !matches!(tt, proc_macro2::TokenTree::Punct(p) if p.as_char() == '$'))
        .map(|tt| match tt {
            proc_macro2::TokenTree::Group(g) => proc_macro2::TokenTree::Group(
                proc_macro2::Group::new(g.delimiter(), strip_dollar(g.stream())),
            ),
            other => other,
        })
        .collect()
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("read rlx-ir src") {
        let p = e.unwrap().path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_ir_builder_is_bound_or_excluded() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ir_src = here.join("../../core/rlx-ir/src");
    let mut files = Vec::new();
    rs_files(&ir_src, &mut files);
    let mut builders = Builders(BTreeSet::new());
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        if let Ok(file) = syn::parse_file(&src) {
            builders.visit_file(&file);
        }
    }
    assert!(
        builders.0.len() > 150,
        "found only {} IR builders — did rlx-ir move?",
        builders.0.len()
    );

    let table = std::fs::read_to_string(here.join("src/table.rs")).unwrap();
    let mut calls = Calls(BTreeSet::new());
    calls.visit_file(&syn::parse_file(&table).unwrap());

    let excluded: BTreeSet<&str> = NOT_BOUND
        .iter()
        .chain(SPELLED_OTHERWISE)
        .map(|(n, _)| *n)
        .collect();
    let missing: Vec<&String> = builders
        .0
        .iter()
        .filter(|b| !calls.0.contains(*b) && !excluded.contains(b.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "rlx-ir builders with no op-table row: {missing:?}\n\
         Add a row to crates/bindings/rlx-bind/src/table.rs (every language gets \
         it), or list it in NOT_BOUND in this file with the reason."
    );

    let stale: Vec<&str> = excluded
        .iter()
        .filter(|n| !builders.0.contains(**n))
        .copied()
        .collect();
    assert!(
        stale.is_empty(),
        "NOT_BOUND / SPELLED_OTHERWISE name builders rlx-ir no longer has: {stale:?}"
    );
}
