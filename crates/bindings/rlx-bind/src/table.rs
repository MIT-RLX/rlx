// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The** graph-builder table. Every language binding is generated from it.
//!
//! [`for_each_graph_op!`] hands every row to a callback macro. In this crate
//! the callbacks build the typed core functions ([`crate::ops`]) and the
//! reflected [`manifest`](crate::manifest()); rlx-js and pyrlx expand it into
//! their native methods; `rlx-bindgen` prints the C ABI, Swift and `.pyi` from
//! the manifest. Adding an op is adding a row here — every language gets it.
//!
//! # Row grammar
//!
//! ```text
//! /// docs
//! name [js = "jsName", py_also = "alias"]                 // optional renames
//!     (arg: kind = default, … ; { opt: kind = default, … }) // `; {…}` = options
//!     -> ret                                                // optional, default `node`
//!     => |g| { body };
//! ```
//!
//! * `name` is the canonical snake_case name — Python's spelling. JS
//!   camelCases it (a trailing `_` dropped: `where_` → `where`).
//! * Argument names are Python's keyword names. JS reads arguments by
//!   position, so names never constrain it.
//! * The `{…}` group is trailing **options**: one object in JS
//!   (`conv2d(x, w, {stride: 2})`), keyword-or-positional parameters in
//!   Python, flattened parameters in C and Swift. `name | alias` lets JS read
//!   a legacy key.
//! * Defaults are neutral literals so every generator can print them.
//! * The body runs under the panic guard, inside a closure returning
//!   `ApiResult`: `?` and [`bail!`](crate::bail) work, and arguments arrive
//!   decoded (see [`crate::kinds`]).
//! * Renames: `js`, `py`, `c`, `swift` replace a language's name; `js_also`,
//!   `py_also` add a second one.

/// Expand the op table into `$cb! { [extra…] rows… }`.
#[macro_export]
macro_rules! for_each_graph_op {
    ($cb:path $(, $($extra:tt)*)?) => {
        $cb! { [$($($extra)*)?]

        // ── graph structure ─────────────────────────────────

        /// A runtime input. `dtype` defaults to f32.
        input(name: str, shape: usizes, dtype: dtype = "f32") => |g| {
            g.input(name, ::rlx_ir::Shape::new(&shape, dtype))
        };
        /// A trainable / loadable weight, filled by `set_params` at run time.
        param(name: str, shape: usizes, dtype: dtype = "f32") => |g| {
            g.param(name, ::rlx_ir::Shape::new(&shape, dtype))
        };
        /// A rank-0 literal that broadcasts in binary ops.
        constant(value: f64, dtype: dtype = "f32") => |g| {
            g.try_constant(value, dtype).map_err($crate::ApiError::range)?
        };
        /// A constant tensor of zeros.
        zeros(dims: usizes, dtype: dtype = "f32") => |g| { g.full(&dims, 0.0, dtype) };
        /// A constant tensor filled with `value`.
        full(dims: usizes, value: f32, dtype: dtype = "f32") => |g| { g.full(&dims, value, dtype) };
        /// Mark the graph's outputs, in order.
        set_outputs(outputs: nodes) -> unit => |g| { g.set_outputs(outputs) };
        /// The output node ids, in order.
        outputs() -> nodes => |g| { g.outputs.clone() };
        /// Name output `index`, so a compiled graph finds it by name.
        name_output(index: usize, name: str) -> unit => |g| { g.name_output(index, name) };
        /// Each output's name, in order; "" for an unnamed one.
        output_names() -> strings => |g| {
            (0..g.outputs.len())
                .map(|i| g.output_name(i).unwrap_or_default().to_string())
                .collect::<Vec<_>>()
        };
        /// `(dims, dtype)` of a node; a dynamic dim reads back as `-1`.
        shape_of(id: node) -> shape => |g| { g.shape(id).clone() };
        node_count() -> usize => |g| { g.nodes().len() };
        name() -> string => |g| { g.name.clone() };
        /// Every `Param` name, in creation order — what `set_params` must cover.
        param_names() -> strings => |g| {
            g.nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    ::rlx_ir::Op::Param { name } => Some(name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        /// Distinct op kinds present, sorted.
        op_kinds() -> strings => |g| {
            let mut kinds: Vec<String> =
                g.nodes().iter().map(|n| format!("{:?}", n.op.kind())).collect();
            kinds.sort();
            kinds.dedup();
            kinds
        };

        // ── linear algebra ──────────────────────────────────

        /// Shape-inferred matmul.
        matmul(lhs: node, rhs: node) => |g| { g.mm(lhs, rhs) };
        /// Matmul with an explicit output shape, for a layout inference would not pick.
        matmul_with_shape(lhs: node, rhs: node, out_shape: usizes, dtype: dtype = "f32") => |g| {
            g.matmul(lhs, rhs, ::rlx_ir::Shape::new(&out_shape, dtype))
        };
        /// `x = A⁻¹·b`; `x` takes `b`'s shape unless `out_shape` is given.
        dense_solve(a: node, b: node, out_shape: opt_usizes = None, dtype: dtype = "f32") => |g| {
            let shape = match out_shape {
                Some(d) => ::rlx_ir::Shape::new(&d, dtype),
                None => $crate::shapes::same(g, b),
            };
            g.dense_solve(a, b, shape)
        };
        batched_dense_solve(a: node, b: node, out_shape: opt_usizes = None, dtype: dtype = "f32") => |g| {
            let shape = match out_shape {
                Some(d) => ::rlx_ir::Shape::new(&d, dtype),
                None => $crate::shapes::same(g, b),
            };
            g.batched_dense_solve(a, b, shape)
        };
        cholesky(a: node) => |g| { let s = $crate::shapes::same(g, a); g.cholesky(a, s) };
        det(a: node) => |g| {
            let s = ::rlx_ir::Shape::scalar(g.shape(a).dtype());
            g.det(a, s)
        };
        logdet(a: node) => |g| {
            let s = ::rlx_ir::Shape::scalar(g.shape(a).dtype());
            g.logdet(a, s)
        };
        /// Symmetric eigendecomposition → `(eigenvalues, eigenvectors)`.
        eigh(a: node) -> pair => |g| { g.eigh(a) };
        /// Batched symmetric eigendecomposition → `(eigenvalues, eigenvectors)`.
        eigh_batch(a: node) -> pair => |g| { g.eigh_batch(a) };
        /// One factor of the thin QR of `a [m, n]`: `Q [m, k]` or `R [k, n]`.
        qr(a: node, part: qr_part = "q") => |g| {
            let (m, n) = $crate::shapes::matrix_dims(g, a, "qr")?;
            let k = m.min(n);
            let dt = g.shape(a).dtype();
            let s = match part {
                ::rlx_ir::op::QrPart::Q => ::rlx_ir::Shape::new(&[m, k], dt),
                ::rlx_ir::op::QrPart::R => ::rlx_ir::Shape::new(&[k, n], dt),
            };
            g.qr(a, part, s)
        };
        /// One factor of the thin SVD: `"u"`, `"s"` or `"vt"`.
        svd(a: node, part: svd_part = "s") => |g| {
            let (m, n) = $crate::shapes::matrix_dims(g, a, "svd")?;
            let k = m.min(n);
            let dt = g.shape(a).dtype();
            let s = match part {
                ::rlx_ir::op::SvdPart::U => ::rlx_ir::Shape::new(&[m, k], dt),
                ::rlx_ir::op::SvdPart::S => ::rlx_ir::Shape::new(&[k], dt),
                ::rlx_ir::op::SvdPart::Vt => ::rlx_ir::Shape::new(&[k, n], dt),
            };
            g.svd(a, part, s)
        };
        /// Solve `A·X = B` (or `Aᵀ·X = B`) for triangular `A`.
        triangular_solve(a: node, b: node, lower: bool = true, transpose: bool = false) => |g| {
            let s = $crate::shapes::same(g, b);
            g.triangular_solve(a, b, lower, transpose, s)
        };
        /// `|z|²` of a complex tensor, as a real tensor.
        complex_norm_sq(z: node) => |g| { g.complex_norm_sq(z) };

        // ── elementwise ─────────────────────────────────────

        add(a: node, b: node) => |g| { g.add(a, b) };
        sub(a: node, b: node) => |g| { g.sub(a, b) };
        mul(a: node, b: node) => |g| { g.mul(a, b) };
        div(a: node, b: node) => |g| { g.div(a, b) };
        /// Any binary op by name (`"pow"`, `"atan2"`, `"bit_and"`, …) or symbol.
        binary(op: binop, lhs: node, rhs: node) => |g| {
            use ::rlx_ir::op::BinaryOp;
            match op {
                BinaryOp::Add => g.add(lhs, rhs),
                BinaryOp::Sub => g.sub(lhs, rhs),
                BinaryOp::Mul => g.mul(lhs, rhs),
                BinaryOp::Div => g.div(lhs, rhs),
                other => {
                    let s = ::rlx_ir::shape::binary_shape(g.shape(lhs), g.shape(rhs))
                        .map_err(|e| $crate::ApiError::type_(format!("binary({other:?}) shape: {e}")))?;
                    g.binary(other, lhs, rhs, s)
                }
            }
        };
        /// Any activation by name — every variant of the IR's `Activation`.
        activation(kind: activation, input: node) => |g| {
            let s = $crate::shapes::same(g, input);
            g.activation(kind, input, s)
        };
        gelu(x: node) => |g| { g.gelu(x) };
        /// Tanh-approximation GELU.
        gelu_approx(x: node) => |g| { g.gelu_approx(x) };
        silu(x: node) => |g| { g.silu(x) };
        relu(x: node) => |g| { g.relu(x) };
        exp(x: node) => |g| { g.exp(x) };
        sqrt(x: node) => |g| { g.sqrt(x) };
        neg(x: node) => |g| { g.neg(x) };
        tanh(x: node) => |g| { g.tanh(x) };
        sigmoid(x: node) => |g| { g.sigmoid(x) };
        log(x: node) => |g| { g.log(x) };
        abs(x: node) => |g| { g.abs(x) };
        recip(x: node) => |g| { g.recip(x) };
        rsqrt(x: node) => |g| { g.rsqrt(x) };
        /// Complex conjugate — the Wirtinger-AD primitive.
        conjugate(z: node) => |g| { g.conjugate(z) };
        clamp(x: node, min: f32, max: f32) => |g| { g.clamp_(x, min, max) };
        /// Identity forward, zero backward (`stop_gradient` / `detach`).
        stop_gradient(x: node) => |g| { g.stop_gradient(x) };
        /// `compare("lt" | "<" | …, a, b)` → a Bool tensor.
        compare(op: cmp_op, lhs: node, rhs: node) => |g| { $crate::ops::compare_nodes(g, op, lhs, rhs)? };
        eq_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Eq, a, b)? };
        ne_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Ne, a, b)? };
        lt_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Lt, a, b)? };
        le_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Le, a, b)? };
        gt_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Gt, a, b)? };
        ge_(a: node, b: node) => |g| { $crate::ops::compare_nodes(g, ::rlx_ir::op::CmpOp::Ge, a, b)? };
        /// `cond ? a : b`, elementwise.
        where_(cond: node, a: node, b: node) => |g| {
            let s = $crate::shapes::same(g, a);
            g.add_node(::rlx_ir::Op::Where, vec![cond, a, b], s)
        };

        // ── reduction / ordering ────────────────────────────

        reduce(input: node, op: reduce_op, axes: usizes, keep_dim: bool = false) => |g| {
            let s = ::rlx_ir::shape::reduce_shape(g.shape(input), &axes, keep_dim)
                .map_err(|e| $crate::ApiError::type_(format!("reduce shape: {e}")))?;
            g.reduce(input, op, axes, keep_dim, s)
        };
        sum(x: node, axes: usizes, keep_dim: bool = false) => |g| { g.sum(x, axes, keep_dim) };
        mean(x: node, axes: usizes, keep_dim: bool = false) => |g| { g.mean(x, axes, keep_dim) };
        softmax(input: node, axis: i32 = -1) => |g| { g.sm(input, axis) };
        cumsum(input: node, axis: i32, exclusive: bool = false) => |g| {
            let s = $crate::shapes::same(g, input);
            g.cumsum(input, axis, exclusive, s)
        };
        cumprod(x: node, axis: i32, exclusive: bool = false) => |g| { g.cumprod_(x, axis, exclusive) };
        cummax(x: node, axis: i32, exclusive: bool = false) => |g| { g.cummax_(x, axis, exclusive) };
        argmax(x: node, axis: usize, keep_dim: bool = false) => |g| {
            let s = $crate::shapes::reduced(g, x, axis, keep_dim)?;
            g.argmax(x, axis, keep_dim, s)
        };
        argmin(x: node, axis: usize, keep_dim: bool = false) => |g| {
            let s = $crate::shapes::reduced(g, x, axis, keep_dim)?;
            g.argmin(x, axis, keep_dim, s)
        };
        sort(x: node, axis: usize, descending: bool = false) => |g| {
            let s = $crate::shapes::same(g, x);
            g.sort(x, axis, descending, s)
        };
        argsort(x: node, axis: usize, descending: bool = false) => |g| {
            let s = $crate::shapes::same(g, x);
            g.argsort(x, axis, descending, s)
        };
        histogram(x: node, bins: usize, min: f32, max: f32) => |g| { g.histogram(x, bins, min, max) };

        // ── shape ───────────────────────────────────────────

        /// `-1` infers one axis.
        reshape(x: node, new_shape: i64s) => |g| { g.reshape_(x, new_shape) };
        transpose(x: node, perm: usizes) => |g| { g.transpose_(x, perm) };
        narrow(x: node, axis: usize, start: usize, length: usize) => |g| { g.narrow_(x, axis, start, length) };
        concat(inputs: nodes, axis: usize) => |g| { g.concat_(inputs, axis) };
        gather(table: node, indices: node, axis: usize) => |g| { g.gather_(table, indices, axis) };
        gather_elements(data: node, indices: node, axis: i32) => |g| { g.gather_elements(data, indices, axis) };
        cast(x: node, to: dtype) => |g| { g.cast(x, to) };
        /// Strided slice; `step` may be negative.
        slice(x: node, axis: usize, start: usize, length: usize, step: i64 = 1) => |g| {
            g.slice_(x, axis, start, length, step)
        };
        tile(x: node, reps: usizes) => |g| { g.tile_(x, reps) };
        /// Keep one triangle, zero the other.
        trilu(x: node, upper: bool, diagonal: i64 = 0) => |g| { g.trilu_(x, upper, diagonal) };
        roll(x: node, shifts: i64s, dims: usizes) => |g| { g.roll_(x, shifts, dims) };
        reverse(x: node, axes: usizes) => |g| { g.reverse(x, axes) };
        /// One `[before, after]` per axis; `mode` is constant / reflect / replicate / circular.
        pad(x: node, pads: pads, mode: str = "constant", value: f32 = 0.0) => |g| {
            let mode = $crate::enums::pad_mode(&mode, value)?;
            g.pad_(x, pads, mode)
        };
        /// Broadcast to `dims` (`-1` keeps an axis) without a copy.
        expand(x: node, dims: i64s) => |g| {
            let s = ::rlx_ir::shape::expand_shape(g.shape(x), &dims)
                .map_err(|e| $crate::ApiError::type_(format!("expand shape: {e}")))?;
            g.add_node(::rlx_ir::Op::Expand { target_shape: dims }, vec![x], s)
        };
        /// ONNX ScatterND; output shape == `data`.
        scatter_nd(data: node, indices: node, updates: node, reduction: scatter_red = "none") => |g| {
            g.scatter_nd(data, indices, updates, reduction)
        };
        /// ONNX ScatterElements along `axis`.
        scatter_elements(data: node, indices: node, updates: node, axis: i32, reduction: scatter_red = "none") => |g| {
            g.scatter_elements(data, indices, updates, axis, reduction)
        };
        /// ONNX GatherND → `indices[:-1] + data[batch_dims + k:]`.
        gather_nd(data: node, indices: node, batch_dims: i32 = 0) => |g| {
            let d = $crate::shapes::static_dims(g, data, "gather_nd data")?;
            let i = $crate::shapes::static_dims(g, indices, "gather_nd indices")?;
            let Some(&k) = i.last() else {
                $crate::bail!("gather_nd: indices must have rank ≥ 1");
            };
            let from = batch_dims.max(0) as usize + k;
            if from > d.len() {
                $crate::bail!(
                    "gather_nd: batch_dims + indices.shape[-1] = {from} exceeds data rank {}",
                    d.len()
                );
            }
            let mut out: Vec<usize> = i[..i.len() - 1].to_vec();
            out.extend_from_slice(&d[from..]);
            let s = ::rlx_ir::Shape::new(&out, g.shape(data).dtype());
            g.gather_nd(data, indices, batch_dims, s)
        };
        /// Nearest-neighbour ×2 upsample of an NCHW map.
        resize_nearest_2x(x: node) => |g| {
            let d = $crate::shapes::static_dims(g, x, "resize_nearest_2x")?;
            if d.len() != 4 {
                $crate::bail!("resize_nearest_2x: expected a 4-D NCHW input");
            }
            let s = ::rlx_ir::Shape::new(&[d[0], d[1], d[2] * 2, d[3] * 2], g.shape(x).dtype());
            g.add_node(::rlx_ir::Op::ResizeNearest2x, vec![x], s)
        };

        // ── normalization ───────────────────────────────────

        layer_norm(input: node, gamma: node, beta: node, axis: i32 = -1, eps: f32 = 1e-5) => |g| {
            let s = $crate::shapes::same(g, input);
            g.layer_norm(input, gamma, beta, axis, eps, s)
        };
        rms_norm(input: node, gamma: node, beta: node, eps: f32 = 1e-5) => |g| {
            g.rms_norm(input, gamma, beta, eps)
        };
        /// LayerNorm over the channel axis of an NCHW map.
        layer_norm2d(input: node, gamma: node, beta: node, eps: f32 = 1e-5) => |g| {
            g.layer_norm2d(input, gamma, beta, eps)
        };
        group_norm(input: node, gamma: node, beta: node, num_groups: usize, eps: f32 = 1e-5) => |g| {
            g.group_norm(input, gamma, beta, num_groups, eps)
        };
        /// Inference BatchNorm with frozen running statistics.
        batch_norm_inference [js = "batchNorm"]
            (x: node, gamma: node, beta: node, running_mean: node, running_var: node, eps: f32 = 1e-5) => |g| {
            g.batch_norm_inference(x, gamma, beta, running_mean, running_var, eps)
        };
        /// Training BatchNorm over batch statistics.
        batch_norm_training(x: node, gamma: node, beta: node, eps: f32 = 1e-5) => |g| {
            g.batch_norm_training(x, gamma, beta, eps)
        };
        /// DiT adaLN-Zero: `norm(x)·(1+scale)+shift`.
        ada_layer_norm(x: node, scale: node, shift: node, norm: ada_norm = "layer_norm", eps: f32 = 1e-6) => |g| {
            g.ada_layer_norm(x, scale, shift, norm, eps)
        };
        /// DiT gated residual: `x + gate·y`.
        gated_residual(x: node, y: node, gate: node) => |g| { g.gated_residual(x, y, gate) };
        /// `LayerNorm(x + residual (+ bias))` in one kernel.
        fused_residual_layer_norm [py_also = "fused_residual_ln"]
            (x: node, residual: node, gamma: node, beta: node, eps: f32 = 1e-5, bias: opt_node = None) => |g| {
            let s = $crate::shapes::same(g, x);
            g.fused_residual_ln(x, residual, bias, gamma, beta, eps, s)
        };
        /// `RmsNorm(x + residual (+ bias))` in one kernel.
        fused_residual_rms_norm(x: node, residual: node, gamma: node, beta: node, eps: f32 = 1e-5, bias: opt_node = None) => |g| {
            let s = $crate::shapes::same(g, x);
            g.fused_residual_rms_norm(x, residual, bias, gamma, beta, eps, s)
        };

        // ── attention / position ────────────────────────────

        /// Explicit mask tensor, `[batch, key_len]` only — a per-query mask is a bias.
        attention(q: node, k: node, v: node, mask: node, num_heads: usize, head_dim: usize) => |g| {
            g.attention_(q, k, v, mask, num_heads, head_dim)
        };
        /// Kernel-synthesized mask: `"none"`, `"causal"`, `"sliding:<N>"`.
        attention_kind(q: node, k: node, v: node, num_heads: usize, head_dim: usize, mask_kind: mask = "causal") => |g| {
            let s = ::rlx_ir::shape::attention_shape(g.shape(q));
            g.attention_kind(q, k, v, num_heads, head_dim, mask_kind, s)
        };
        /// Additive per-head, per-query bias `[B, H, Sq, Sk]`.
        attention_bias(q: node, k: node, v: node, bias: node, num_heads: usize, head_dim: usize) => |g| {
            let s = ::rlx_ir::shape::attention_shape(g.shape(q));
            g.attention_bias(q, k, v, bias, num_heads, head_dim, s)
        };
        /// Attention whose value head width differs from the query/key width (MLA).
        attention_kind_vdim(q: node, k: node, v: node, num_heads: usize, head_dim: usize, v_head_dim: usize,
                            mask_kind: mask = "causal") => |g| {
            let s = ::rlx_ir::shape::attention_shape_vdim(g.shape(q), num_heads, head_dim, v_head_dim);
            g.attention_kind_vdim(q, k, v, num_heads, head_dim, v_head_dim, mask_kind, s)
        };
        /// Attention with a score `scale` and logit `softcap` (Gemma-2, Grok).
        /// `bias` selects the per-query path, `mask` the `[B, Sk]` mask tensor;
        /// otherwise the kernel synthesizes `mask_kind`.
        attention_opts(q: node, k: node, v: node, num_heads: usize, head_dim: usize;
                       { mask: opt_node = None, bias: opt_node = None, mask_kind: mask = "causal",
                         scale: opt_f32 = None, softcap: opt_f32 = None }) => |g| {
            let s = ::rlx_ir::shape::attention_shape(g.shape(q));
            match (bias, mask) {
                (Some(b), _) => g.attention_bias_opts(q, k, v, b, num_heads, head_dim, s, scale, softcap),
                (None, Some(m)) => g.attention_opts(q, k, v, m, num_heads, head_dim, s, scale, softcap),
                (None, None) => g.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, s, scale, softcap),
            }
        };
        /// RoPE. `n_rot` defaults to the full head (partial rotary's table
        /// stride is `n_rot/2`); `style` is `"neox"` (HF) or `"gptj"` (GGUF).
        rope [py_also = "rope_n"]
            (x: node, cos: node, sin: node, head_dim: usize, n_rot: opt_usize = None, style: rope_style = "neox") => |g| {
            match n_rot {
                None => g.rope_styled(x, cos, sin, head_dim, style),
                Some(n) => g.rope_n_styled(x, cos, sin, head_dim, n, style),
            }
        };
        /// Axial 2-D RoPE (SAM / ViT image tokens).
        axial_rope2d(x: node, end_x: usize, end_y: usize, head_dim: usize, num_heads: usize,
                     theta: f32 = 10000.0, repeat_factor: usize = 1) => |g| {
            g.axial_rope2d(x, end_x, end_y, head_dim, num_heads, theta, repeat_factor)
        };

        // ── convolution / pooling / resampling ──────────────

        /// NCHW convolution, output shape inferred.
        conv2d(input: node, weight: node;
               { kernel_size: pair = [1, 1], stride: pair = [1, 1], padding: pair = [0, 0],
                 dilation: pair = [1, 1], groups: usize = 1 }) => |g| {
            g.conv2d(input, weight, kernel_size, stride, padding, dilation, groups)
        };
        /// `x [N,Cin,L]`, `w [Cin,Cout/groups,K]`.
        conv_transpose1d(x: node, w: node, kernel: usize, stride: usize = 1, padding: usize = 0,
                         dilation: usize = 1, output_padding: usize = 0, groups: usize = 1) => |g| {
            g.conv_transpose1d(x, w, kernel, stride, padding, dilation, output_padding, groups)
        };
        conv_transpose2d(input: node, weight: node;
                         { kernel_size: pair = [1, 1], stride: pair = [1, 1], padding: pair = [0, 0],
                           dilation: pair = [1, 1], output_padding: pair = [0, 0], groups: usize = 1 }) => |g| {
            g.conv_transpose2d(input, weight, kernel_size, stride, padding, dilation, output_padding, groups)
        };
        /// `x [N,C,D,H,W]`; the kernel size is read from `w`.
        conv3d(x: node, w: node;
               { stride: triple = [1, 1, 1], padding: triple = [0, 0, 0], dilation: triple = [1, 1, 1],
                 groups: usize = 1 }) => |g| {
            g.conv3d(x, w, stride, padding, dilation, groups)
        };
        conv_transpose3d(x: node, w: node;
                         { stride: triple = [1, 1, 1], padding: triple = [0, 0, 0], dilation: triple = [1, 1, 1],
                           output_padding: triple = [0, 0, 0], groups: usize = 1 }) => |g| {
            g.conv_transpose3d(x, w, stride, padding, dilation, output_padding, groups)
        };
        /// The patch matrix a GEMM-lowered conv multiplies.
        im2col(x: node;
               { kernel_size: pair = [1, 1], stride: pair = [1, 1], padding: pair = [0, 0],
                 dilation: pair = [1, 1] }) => |g| {
            g.im2col(x, kernel_size, stride, padding, dilation)
        };
        /// `stride` defaults to the kernel size.
        max_pool2d(x: node; { kernel_size: pair = [2, 2], stride: opt_pair = None, padding: pair = [0, 0] }) => |g| {
            g.pool2d(x, ::rlx_ir::op::ReduceOp::Max, kernel_size, stride.unwrap_or(kernel_size), padding)
        };
        avg_pool2d(x: node; { kernel_size: pair = [2, 2], stride: opt_pair = None, padding: pair = [0, 0] }) => |g| {
            g.pool2d(x, ::rlx_ir::op::ReduceOp::Mean, kernel_size, stride.unwrap_or(kernel_size), padding)
        };
        pool2d(x: node, op: reduce_op; { kernel_size: pair = [2, 2], stride: opt_pair = None, padding: pair = [0, 0] }) => |g| {
            g.pool2d(x, op, kernel_size, stride.unwrap_or(kernel_size), padding)
        };
        /// INT8 NCHW convolution; output `i8`.
        q_conv2d(x: node, w: node, bias: node, x_zp: i32, w_zp: i32, out_zp: i32, mult: f32;
                 { kernel_size: pair = [1, 1], stride: pair = [1, 1], padding: pair = [0, 0],
                   dilation: pair = [1, 1], groups: usize = 1 }) => |g| {
            let xd = $crate::shapes::static_dims(g, x, "q_conv2d x")?;
            let wd = $crate::shapes::static_dims(g, w, "q_conv2d w")?;
            if xd.len() != 4 || wd.len() != 4 {
                $crate::bail!("q_conv2d: x and w must be 4-D (NCHW / OIHW)");
            }
            let spatial = |i: usize| {
                (xd[2 + i] + 2 * padding[i]).saturating_sub(dilation[i] * (kernel_size[i] - 1) + 1)
                    / stride[i]
                    + 1
            };
            let s = ::rlx_ir::Shape::new(&[xd[0], wd[0], spatial(0), spatial(1)], ::rlx_ir::DType::I8);
            g.q_conv2d(x, w, bias, kernel_size.to_vec(), stride.to_vec(), padding.to_vec(),
                       dilation.to_vec(), groups, x_zp, w_zp, out_zp, mult, s)
        };
        interpolate1d(x: node, length: usize, mode: interp = "nearest") => |g| { g.interpolate1d(x, length, mode) };
        /// `x [N,C,D,H,W]` → `size = [d, h, w]`.
        interpolate3d(x: node, size: usizes, mode: interp = "nearest", align_corners: bool = false) => |g| {
            let size = $crate::kinds::exactly::<3, usize>(size, "interpolate3d size")?;
            g.interpolate3d(x, size, mode, align_corners)
        };

        // ── blocks / fused matmul ───────────────────────────

        /// `x·W (+ bias)`.
        linear_bias(x: node, w: node, bias: opt_node = None) => |g| { g.linear_bias(x, w, bias) };
        /// `x·W (+ bias)`, optionally followed by an activation.
        linear_bias_act(x: node, w: node, bias: opt_node = None, activation: opt_activation = None) => |g| {
            g.linear_bias_act(x, w, bias, activation)
        };
        /// `down(silu(x·gate) · (x·up))`.
        swiglu_ffn(x: node, up: node, gate: node, down: node) => |g| { g.swiglu_ffn(x, up, gate, down) };
        /// Two projections of one input sharing its load → `(x·w1, x·w2)`.
        shared_matmul_pair(x: node, w1: node, w2: node) -> pair => |g| { g.shared_matmul_pair(x, w1, w2) };
        fused_matmul_bias_act(input: node, weight: node, bias: node, out_shape: usizes, dtype: dtype,
                              activation: opt_activation = None) => |g| {
            g.fused_matmul_bias_act(input, weight, bias, activation, ::rlx_ir::Shape::new(&out_shape, dtype))
        };
        /// `x·W + scale·(x·A)·B`.
        lora_matmul(x: node, w: node, a: node, b: node, scale: f32, out_shape: usizes, dtype: dtype = "f32") => |g| {
            g.lora_matmul(x, w, a, b, scale, ::rlx_ir::Shape::new(&out_shape, dtype))
        };
        /// Fused softmax + cross-entropy against a dense target distribution.
        softmax_cross_entropy(logits: node, targets: node) => |g| { g.softmax_cross_entropy(logits, targets) };
        /// Softmax cross-entropy against integer class labels.
        softmax_cross_entropy_with_logits(logits: node, labels: node) => |g| {
            g.softmax_cross_entropy_with_logits(logits, labels)
        };

        // ── quantization ────────────────────────────────────

        quantize(x: node, scale: f32, zero_point: i32 = 0) => |g| { g.quantize(x, scale, zero_point) };
        dequantize(x: node, scale: f32, zero_point: i32 = 0) => |g| { g.dequantize(x, scale, zero_point) };
        quantize_per_channel(x: node, axis: usize, scales: f32s, zero_points: i64s) => |g| {
            g.quantize_per_channel(x, axis, scales, zero_points.into_iter().map(|z| z as i32).collect())
        };
        dequantize_per_channel(x: node, axis: usize, scales: f32s, zero_points: i64s) => |g| {
            g.dequantize_per_channel(x, axis, scales, zero_points.into_iter().map(|z| z as i32).collect())
        };
        /// Native FP8/FP6/FP4 GEMM; `rhs` is K-last (`[n, k]`).
        scaled_matmul(lhs: node, rhs: node, format: scaled_fmt = "f8e4m3", layout: scale_layout = "per_tensor") => |g| {
            g.scaled_matmul(lhs, rhs, format, layout)
        };
        /// Native low-precision GEMM with a bias epilogue.
        scaled_matmul_bias(lhs: node, rhs: node, bias: opt_node = None, format: scaled_fmt = "f8e4m3",
                           layout: scale_layout = "per_tensor") => |g| {
            g.scaled_matmul_bias(lhs, rhs, bias, format, layout)
        };
        /// MoE grouped GEMM over a block-scaled expert bank (MXFP4).
        scaled_grouped_matmul(x: node, weight: node, expert_idx: node, format: scaled_fmt = "f4e2m1",
                              layout: scale_layout = "mx", bias: opt_node = None) => |g| {
            g.scaled_grouped_matmul_bias(x, weight, expert_idx, bias, format, layout)
        };
        /// `x` → `(codes (U8), scales)` in a minifloat format.
        scaled_quantize(x: node, format: scaled_fmt = "f8e4m3", layout: scale_layout = "per_tensor") -> pair => |g| {
            g.scaled_quantize(x, format, layout)
        };
        scaled_dequantize(codes: node, scale: node, format: scaled_fmt = "f8e4m3", layout: scale_layout = "per_tensor") => |g| {
            g.scaled_dequantize(codes, scale, format, layout)
        };
        /// Dequantize-in-the-loop matmul. `scheme`: `"int8:32"`, `"mlx_affine:4:64"`, `"Q4_K"`, ….
        dequant_matmul(x: node, w_q: node, scale: node, zp: node, scheme: quant_scheme) => |g| {
            let s = if scheme.is_gguf() {
                ::rlx_ir::shape::dequant_matmul_shape(g.shape(x), g.shape(w_q))
                    .map_err(|e| $crate::ApiError::type_(format!("dequant_matmul shape: {e}")))?
            } else {
                $crate::shapes::matmul(g, x, w_q, "dequant_matmul")?
            };
            g.dequant_matmul(x, w_q, scale, zp, scheme, s)
        };
        /// Matmul against a GGUF-packed byte blob (`[out, in]` order), `n` outputs.
        dequant_matmul_packed(x: node, packed: node, scheme: quant_scheme, n: usize) => |g| {
            let s = $crate::shapes::last_axis(g, x, n);
            g.dequant_matmul_packed(x, packed, scheme, s)
        };
        /// MoE grouped matmul against one packed expert bank.
        dequant_grouped_matmul_packed(x: node, bank: node, expert_idx: node, scheme: quant_scheme, n: usize) => |g| {
            let s = $crate::shapes::last_axis(g, x, n);
            g.dequant_grouped_matmul_packed(x, bank, expert_idx, scheme, s)
        };
        /// NVFP4 block matmul (group 16, FP8 block scales, f32 global scale).
        dequant_matmul_nvfp4(x: node, w_q: node, block_scales: node, global_scale: node, n: usize) => |g| {
            let s = $crate::shapes::last_axis(g, x, n);
            g.dequant_matmul_nvfp4(x, w_q, block_scales, global_scale, s)
        };
        /// INT8-arithmetic matmul: i8 in, i32 bias, i8 out; `mult = x_scale·w_scale/out_scale`.
        q_matmul(x: node, w: node, bias: node, x_zp: i32, w_zp: i32, out_zp: i32, mult: f32) => |g| {
            let s = $crate::shapes::matmul(g, x, w, "q_matmul")?.with_dtype(::rlx_ir::DType::I8);
            g.q_matmul(x, w, bias, x_zp, w_zp, out_zp, mult, s)
        };
        /// Matmul whose weight is synthesized from a learned VQ codebook in-loop.
        synth_matmul(x: node, indices: node, codebook: node, entry_dim: u32, num_entries: u32) => |g| {
            let m = $crate::shapes::static_dims(g, x, "synth_matmul x")?;
            let n = $crate::shapes::static_dims(g, indices, "synth_matmul indices")?;
            if m.is_empty() || n.is_empty() {
                $crate::bail!("synth_matmul: x and indices must have rank ≥ 1");
            }
            let s = ::rlx_ir::Shape::new(&[m[0], n[0]], g.shape(x).dtype());
            let kind = ::rlx_ir::SynthKind::Codebook { entry_dim, num_entries };
            g.synth_matmul(x, indices, codebook, kind, s)
        };
        /// KAN spline activation (RBF basis over `[grid_min, grid_max]`).
        spline_activation(x: node, coeff: node, num_basis: u32, grid_min: f32 = -1.0, grid_max: f32 = 1.0) => |g| {
            g.spline_activation(x, coeff, num_basis, grid_min, grid_max)
        };
        /// Nearest-codebook assignment → `(indices, quantized)`.
        vector_quantize(x: node, codebook: node, metric: vq_metric = "l2") -> pair => |g| {
            g.vector_quantize(x, codebook, metric)
        };
        /// One index tensor per stage, then the summed reconstruction last.
        residual_vq(x: node, codebooks: nodes, metric: vq_metric = "l2") -> nodes => |g| {
            let (mut codes, quantized) = g.residual_vq(x, &codebooks, metric);
            codes.push(quantized);
            codes
        };

        // ── recurrent / state-space / control flow ──────────

        /// Packed-weight LSTM → `[batch, seq, D·hidden]`; `h0`/`c0` are zero.
        lstm(x: node, w_ih: node, w_hh: node, bias: node, hidden_size: usize, num_layers: usize = 1,
             bidirectional: bool = false) => |g| {
            let s = $crate::shapes::recurrent(g, x, hidden_size, bidirectional);
            g.lstm(x, w_ih, w_hh, bias, hidden_size, num_layers, bidirectional, s)
        };
        /// LSTM threading decode state: `h0`/`c0` `[L·D, batch, hidden]` in, written back.
        lstm_carry(x: node, w_ih: node, w_hh: node, bias: node, h0: node, c0: node, hidden_size: usize,
                   num_layers: usize = 1, bidirectional: bool = false) => |g| {
            let s = $crate::shapes::recurrent(g, x, hidden_size, bidirectional);
            g.lstm_carry(x, w_ih, w_hh, bias, h0, c0, hidden_size, num_layers, bidirectional, s)
        };
        gru(x: node, w_ih: node, w_hh: node, b_ih: node, b_hh: node, hidden_size: usize,
            num_layers: usize = 1, bidirectional: bool = false, h0: opt_node = None) => |g| {
            let s = $crate::shapes::recurrent(g, x, hidden_size, bidirectional);
            g.gru(x, w_ih, w_hh, b_ih, b_hh, h0, hidden_size, num_layers, bidirectional, s)
        };
        /// Elman RNN; `relu` selects the activation (else tanh).
        rnn(x: node, w_ih: node, w_hh: node, bias: node, hidden_size: usize, num_layers: usize = 1,
            bidirectional: bool = false, relu: bool = false, h0: opt_node = None) => |g| {
            let s = $crate::shapes::recurrent(g, x, hidden_size, bidirectional);
            g.rnn(x, w_ih, w_hh, bias, h0, hidden_size, num_layers, bidirectional, relu, s)
        };
        /// Mamba selective scan: x/delta `[b,s,h]`, a `[h,n]`, b/c `[b,s,n]`.
        selective_scan(x: node, delta: node, a: node, b: node, c: node, state_size: usize) => |g| {
            let s = $crate::shapes::same(g, x);
            g.selective_scan(x, delta, a, b, c, state_size, s)
        };
        /// Mamba-2 / SSD scalar-decay scan over `x [B,S,H,P]`.
        mamba2(x: node, dt: node, a: node, b: node, c: node, head_dim: usize, state_size: usize) => |g| {
            let s = $crate::shapes::same(g, x);
            g.mamba2(x, dt, a, b, c, head_dim, state_size, s)
        };
        /// Gated DeltaNet: q/k/v `[b,s,h,n]`, gate/beta `[b,s,h]` (gate `[b,s,h,n]` with `per_channel`).
        gated_delta_net(q: node, k: node, v: node, gate: node, beta: node, state_size: usize,
                        per_channel: bool = false) => |g| {
            let s = $crate::shapes::same(g, v);
            if per_channel {
                g.gated_delta_net_pc(q, k, v, gate, beta, state_size, s)
            } else {
                g.gated_delta_net(q, k, v, gate, beta, state_size, s)
            }
        };
        /// Gated DeltaNet threading `state [b,h,n,n]` for decode.
        gated_delta_net_carry(q: node, k: node, v: node, gate: node, beta: node, state: node,
                              state_size: usize, per_channel: bool = false) => |g| {
            let s = $crate::shapes::same(g, v);
            if per_channel {
                g.gated_delta_net_carry_pc(q, k, v, gate, beta, state, state_size, s)
            } else {
                g.gated_delta_net_carry(q, k, v, gate, beta, state, state_size, s)
            }
        };
        /// `scan(init, body, length)`. `body`'s inputs are, in creation order,
        /// the carry, each broadcast, then each per-step slice of `xs`; its one
        /// output is the next carry. The body graph is **consumed**.
        /// `trajectory` returns every carry stacked (`[length, …init]`).
        scan(init: node, body: graph, length: u32;
             { xs: nodes = [], bcasts: nodes = [], checkpoints: u32 = 0, trajectory: bool = false }) => |g| {
            if trajectory && (!bcasts.is_empty() || checkpoints > 0) {
                $crate::bail!("scan: `trajectory` does not combine with `bcasts` or `checkpoints`");
            }
            if checkpoints > 0 && (!xs.is_empty() || !bcasts.is_empty()) {
                $crate::bail!("scan: `checkpoints` applies to a carry-only scan (no `xs` / `bcasts`)");
            }
            if trajectory {
                if xs.is_empty() {
                    g.scan_trajectory(init, body, length)
                } else {
                    g.scan_trajectory_with_xs(init, &xs, body, length)
                }
            } else if !bcasts.is_empty() {
                g.scan_with_bcasts_and_xs(init, &bcasts, &xs, body, length)
            } else if !xs.is_empty() {
                g.scan_with_xs(init, &xs, body, length)
            } else if checkpoints > 0 {
                g.scan_checkpointed(init, body, length, checkpoints)
            } else {
                g.scan(init, body, length)
            }
        };
        /// A sub-graph with optional hand-written AD rules (`custom_vjp` /
        /// `custom_jvp`). Every body graph passed in is consumed.
        custom_fn(inputs: nodes, fwd_body: graph, vjp_body: opt_graph = None, jvp_body: opt_graph = None) => |g| {
            g.custom_fn(inputs, fwd_body, vjp_body, jvp_body)
        };
        /// One node per output of `fwd_body`, which is consumed.
        custom_fn_multi(inputs: nodes, fwd_body: graph) -> nodes => |g| {
            let handle = g.custom_fn_multi(inputs, fwd_body);
            (0..handle.n_outputs()).map(|i| handle.output(g, i)).collect::<Vec<_>>()
        };
        /// An op registered through `rlx_ir::register_op`. `attrs` is its
        /// serialized attribute bytes; `out_shape` declares a packed output.
        custom_op(name: str, inputs: nodes, attrs: bytes = [], out_shape: opt_usizes = None, dtype: dtype = "f32") => |g| {
            match out_shape {
                Some(d) => g.custom_op_packed(name, attrs, inputs, ::rlx_ir::Shape::new(&d, dtype)),
                None => g
                    .try_custom_op(name, attrs, inputs)
                    .map_err(|e| $crate::ApiError::type_(format!("custom_op: {e}")))?,
            }
        };
        /// Fused logits → token id, so sampling never leaves the device.
        sample(logits: node;
               { top_k: usize = 0, top_p: f32 = 1.0, temperature: f32 = 1.0, seed: u64 = 0,
                 output_shape | shape: opt_usizes = None, dtype: dtype = "i32" }) => |g| {
            let Some(dims) = output_shape else {
                $crate::bail!("sample: `output_shape` is required");
            };
            g.sample(logits, top_k, top_p, temperature, seed, ::rlx_ir::Shape::new(&dims, dtype))
        };

        // ── FFT ─────────────────────────────────────────────

        fft(x: node, inverse: bool = false) => |g| { g.fft(x, inverse) };
        ifft(x: node) => |g| { g.fft(x, true) };
        fft_norm(x: node, inverse: bool = false, norm: fft_norm = "backward") => |g| { g.fft_norm(x, inverse, norm) };
        /// Real signal → full spectrum `(re, im)`.
        fft_real(x: node, norm: fft_norm = "backward") -> pair => |g| { g.fft_real(x, norm) };
        /// Real-input FFT → `(re, im)` half spectrum, which `irfft` takes back.
        rfft(x: node, norm: fft_norm = "backward") -> pair => |g| { g.rfft(x, norm) };
        irfft(re: node, im: node, n: usize, norm: fft_norm = "backward") => |g| { g.irfft(re, im, n, norm) };
        /// Non-power-of-two real FFT → half spectrum `(re, im)`.
        rfft_exact(x: node, n: usize, norm: fft_norm = "backward") -> pair => |g| { g.rfft_exact(x, n, norm) };
        rfft_exact_mag(x: node, n: usize, norm: fft_norm = "backward") => |g| { g.rfft_exact_mag(x, n, norm) };
        psd(re: node, im: node) => |g| { g.psd(re, im) };
        /// Real signal straight to PSD.
        psd_real(x: node, norm: fft_norm = "backward") => |g| { g.psd_real(x, norm) };
        stft(x: node, frame_len: usize, hop: usize, norm: fft_norm = "backward") => |g| {
            g.stft(x, frame_len, hop, norm)
        };
        fftfreq(n: usize) => |g| { g.fftfreq_tensor(n) };
        rfftfreq(n: usize) => |g| { g.rfftfreq_tensor(n) };
        /// Complex FFT along one axis.
        fft_axis(x: node, axis: usize, inverse: bool = false) => |g| { g.fft_axis(x, axis, inverse) };
        /// Complex N-D FFT over `axes`.
        fftn(x: node, axes: usizes, inverse: bool = false) => |g| { g.fftn(x, &axes, inverse) };
        ifftn(x: node, axes: usizes) => |g| { g.ifftn(x, &axes) };
        /// Linear convolution of two real signals via an `n_fft`-point FFT.
        fft_conv1d(a: node, b: node, n_fft: usize, norm: fft_norm = "backward") => |g| { g.fft_conv1d(a, b, n_fft, norm) };
        /// Inverse FFT of a full `(re, im)` spectrum → real signal.
        ifft_spectrum(re: node, im: node, norm: fft_norm = "backward") => |g| { g.ifft_spectrum(re, im, norm) };
        /// Row-batched real FFT → `(re, im)`.
        fft_batch_real(x: node, norm: fft_norm = "backward") -> pair => |g| { g.fft_batch_real(x, norm) };
        /// Interleaved complex spectrum → `(re, im)`.
        split_spectrum(spectrum: node) -> pair => |g| { g.split_spectrum(spectrum) };
        /// Zero-pad the last axis up to the next power of two.
        pad_last_axis_to_pow2(x: node) => |g| { g.pad_last_axis_to_pow2(x) };

        // ── signal processing ───────────────────────────────

        /// Analytic signal → `(re, im)` (`scipy.signal.hilbert`).
        hilbert(x: node) -> pair => |g| { g.hilbert(x) };
        envelope(x: node) => |g| { g.envelope(x) };
        instantaneous_phase(x: node) => |g| { g.instantaneous_phase(x) };
        /// FIR convolution with fixed `taps`; `mode` as NumPy, plus `"causal"`.
        fir_conv1d(x: node, taps: f32s, mode: fir_mode = "full") => |g| { g.fir_conv1d(x, &taps, mode) };
        /// Zero-phase forward-backward FIR (`filtfilt`).
        fir_filtfilt(x: node, taps: f32s) => |g| { g.fir_filtfilt(x, &taps) };
        /// One second-order section: `b = [b0, b1, b2]`, `a = [a0, a1, a2]`.
        biquad(x: node, b: f32x3, a: f32x3) => |g| { g.biquad(x, b, a) };
        /// Direct-form IIR `lfilter(b, a, x)`.
        iirfilt(x: node, b: f32s, a: f32s) => |g| { g.iirfilt(x, &b, &a) };
        /// IIR approximated by its first `taps` impulse-response samples.
        iir_as_fir(x: node, b: f32s, a: f32s, taps: usize, mode: fir_mode = "causal") => |g| {
            g.iir_as_fir(x, &b, &a, taps, mode)
        };
        /// Cascaded second-order sections, SciPy row layout `[b0,b1,b2,a0,a1,a2]`.
        sosfilt(x: node, sos: sos) => |g| { g.sosfilt(x, &sos) };
        /// Zero-phase SOS filtering.
        sosfiltfilt(x: node, sos: sos) => |g| { g.sosfiltfilt(x, &sos) };
        /// Polyphase rational resampling by `up / down` with FIR `taps`.
        resample_poly(x: node, up: usize, down: usize, taps: f32s) => |g| { g.resample_poly(x, up, down, &taps) };
        /// Uniformly-partitioned convolution with an impulse-response tensor.
        partitioned_conv1d(x: node, ir: node, block: usize) => |g| { g.partitioned_conv1d(x, ir, block) };
        /// Convolution reverb with a fixed impulse response.
        conv_reverb(x: node, ir: f32s, block: usize) => |g| { g.conv_reverb(x, &ir, block) };
        /// Framed, windowed magnitude (or `power`) spectrogram; `log` takes `ln`.
        spectrogram(x: node, frame_len: usize, hop: usize, window: window = "hann", power: bool = true,
                    log: bool = false) => |g| {
            g.spectrogram(x, frame_len, hop, window, power, log)
        };
        /// Mel filterbank applied to a power spectrum, then `log`.
        log_mel(spectrum: node, filters: node) => |g| { g.log_mel(spectrum, filters) };
        /// Top-`k` Welch PSD peaks over `n_segments` segments.
        welch_peaks(spectrum: node, k: usize, n_segments: usize) => |g| { g.welch_peaks(spectrum, k, n_segments) };
        /// One power per `[lo, hi]` band.
        band_power(x: node, sample_rate: f32, bands: bands) => |g| { g.band_power(x, sample_rate, &bands) };
        /// Gaussian differential entropy per `[lo, hi]` band.
        differential_entropy(x: node, sample_rate: f32, bands: bands) => |g| {
            g.differential_entropy(x, sample_rate, &bands)
        };

        // ── SPD manifold ────────────────────────────────────

        /// `W·X·Wᵀ` — the SPDNet bilinear map.
        bimap(w: node, x: node) => |g| { g.bimap(w, x) };
        /// Rectify eigenvalues below `eps` (SPDNet ReEig).
        reeig(x: node, eps: f32) => |g| { g.reeig(x, eps) };
        /// Matrix log via eigendecomposition (SPDNet LogEig).
        logeig(x: node, eps: f32) => |g| { g.logeig(x, eps) };
        spd_log_map(base: node, x: node) => |g| { g.spd_log_map(base, x) };
        spd_exp_map(base: node, v: node) => |g| { g.spd_exp_map(base, v) };
        spd_parallel_transport(from_: node, to: node, v: node) => |g| { g.spd_parallel_transport(from_, to, v) };
        /// Batched `logm` / `expm` / `sqrtm` / `invsqrtm` over `[batch, n, n]`.
        spd_matrix_fn(x: node, kind: spd_fn) => |g| { g.spd_matrix_fn_batch(x, kind) };
        /// Riemannian batch norm → `(normalized, batch_mean)`.
        spd_batch_norm(x: node, gamma: node, eps: f32, iters: u32 = 20, tol: f32 = 1e-6) -> pair => |g| {
            g.spd_batch_norm(x, gamma, eps, iters, tol)
        };
        /// The transport step of SPD batch norm against a given `mean`.
        spd_batch_norm_transport(x: node, mean: node, gamma: node, eps: f32) => |g| {
            g.spd_batch_norm_transport(x, mean, gamma, eps)
        };
        /// The Fréchet mean under the affine-invariant metric.
        spd_karcher_mean(x: node, iters: u32 = 20, tol: f32 = 1e-6, weights: opt_node = None) => |g| {
            match weights {
                Some(w) => g.spd_karcher_mean_weighted(x, w, iters, tol),
                None => g.spd_karcher_mean(x, iters, tol),
            }
        };
        /// ReEig by Jacobi sweeps (no LAPACK) — runs on every backend.
        spectral_reeig(x: node, sweeps: u32, eps: f64) => |g| { g.spectral_reeig(x, sweeps, eps) };
        spectral_logeig(x: node, sweeps: u32, eps: f64) => |g| { g.spectral_logeig(x, sweeps, eps) };
        }
    };
}
