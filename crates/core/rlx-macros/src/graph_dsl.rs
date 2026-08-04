// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx! { … }` graph DSL — the parser, semantic checker, and code generator.
//!
//! This proc macro is the engine behind the public `rlx_tensor::rlx!`
//! declarative macro. It is invoked as `__rlx_build!` from a tiny
//! `macro_rules!` wrapper that first pulls the handful of names this macro's
//! output references (`GraphScope`, `shape!`, `DType`, `MaskKind`) into a
//! block-local scope via `$crate::…`. That split is deliberate:
//!
//! * the `macro_rules!` wrapper owns crate-path hygiene (`$crate` resolves
//!   transitively, so `rlx!` works whether reached as `rlx_tensor::rlx!`, the
//!   umbrella `rlx::rlx!`, or from a downstream model crate), and
//! * this proc macro owns *parsing, lowering, checking, and codegen* — a real
//!   Pratt parser, a lowering pass (`f(x)` sugar resolution, `fn` inlining,
//!   `repeat` unrolling), a semantic pass reporting precise spanned errors
//!   (including a best-effort static matmul shape check), and **path-free**
//!   `Tensor` method/operator output resolved against the wrapper's `use`.
//!
//! # Grammar
//! ```text
//! program := ( "graph" STRING ";" )?  item*
//! item    := decl | fndef
//! fndef   := "fn" IDENT "(" (IDENT ("," IDENT)*)? ")" "{" ("let" IDENT "=" expr ";")+ "}"
//! decl    := "input"  IDENT ":" shape ";"
//!          | "param"  IDENT ":" shape ";"
//!          | "const"  IDENT "=" (scalar | array) ":" DTYPE ";"
//!          | "bind"   bindname ("," bindname)* ";"   // adopt outer Tensor(s)
//!            where bindname := IDENT             // scalar `Tensor`
//!                            | IDENT "[" "]"      // collection `Vec<Tensor>`/&[Tensor]
//!          | "let"    IDENT "=" expr ";"
//!          | "repeat" INT   "{" decl* "}"        // unrolled INT times (macro-time)
//!          | "repeat" EXPR  "{" decl* "}"        // runtime `for _ in 0..EXPR` loop
//!          | "scan" IDENT "=" expr "for" INT "{" ("let" IDENT "=" expr ";")+ "}"
//!          | ("out"|"output") IDENT ("," IDENT)* ";"
//! scalar  := ("-")? LIT
//! array   := "[" (numeric-literal nested arrays) "]"     // e.g. [[0,1],[1,0]]
//! shape   := "[" <tokens forwarded verbatim to shape!> "]"   // a dim is `?`,
//!            an int literal, OR any in-scope `usize` expr (runtime dim)
//! expr    := term (INFIX term)*
//! INFIX   := "==" "!=" "<" "<=" ">" ">=" "+" "-" "*" "/" "%" "@" "**"
//! term    := "-" term | "(" expr ")" | NUMBER | IDENT
//!          | IDENT "(" expr ("," expr)* ")"           // f(x) sugar / fn call
//!          | term "." IDENT "(" <Rust exprs> ")"      // escape hatch
//! ```
//!
//! In an escape-hatch `.method(args)`, a bare identifier is a binding
//! reference — it's validated and auto-borrowed (`k` → `&k`). Any other arg
//! (literal, enum path, `&x`, call, closure) is raw Rust; wrap an external
//! value as `(value)`, or prefix it with `~` (`~eps`, `~num_heads`), to pass it
//! through by value — the escape a *scalar* argument given as an identifier or
//! const needs so it reaches an `f32`/`usize`/enum parameter instead of `&f32`.
//! This lets a full config-driven block — `q.attention(k, v, ~nh, ~dh,
//! MaskKind::Causal)`, `h.layer_norm(g, b, ~eps)` — be written in the DSL.
//!
//! # `f(x)` sugar
//! * `f(x)`               → `x.f()` (any no-extra-arg `Tensor` method)
//! * `matmul(a,b)`/`mm`   → `a @ b`
//! * `maximum(a,b)` `minimum(a,b)` `pow(a,b)` `atan2(a,b)` `rem(a,b)` — binary
//!   tensor ops; a scalar second operand is promoted.
//! * `clamp(x, lo, hi)`   → `x.clamp(lo, hi)` (`lo`/`hi` scalar)
//! * `select(c,t,f)` / `where_(c,t,f)` → `c.where_(&t, &f)`
//! * a user `fn name(…)`  → inlined at the call site
//!
//! # Precedence
//! Tightest → loosest: postfix `.method(…)` > unary `-` > `**` > (`@` `*` `/`
//! `%`) > (`+` `-`) > (`== != < <= > >=`). `@` shares a band with `* /` and is
//! left-associative (NumPy-style); `**` is right-associative. Elementwise ops
//! broadcast; a scalar literal is promoted (`x * 2.0`, `x ** 2`, `x > 0.0`).
//!
//! # Config-driven structure (runtime dims / depth / adopted tensors)
//! The block form is not limited to compile-time constants — a whole model can
//! be sized and shaped from a runtime `Config`:
//!
//! * **Runtime shape dims.** A `shape` axis is forwarded verbatim to `shape!`,
//!   which accepts any `usize` *expression*, so `input x: [bt, d];`
//!   `param w: [d, ff];` size inputs/params from in-scope values. An integer
//!   literal stays literal, `?`/`?N` stays a dynamic axis; a bare ident / field
//!   access / arithmetic is a runtime dim (`Dim::Static(expr)`).
//! * **Runtime `repeat EXPR { … }`.** When the count is not an integer literal
//!   it lowers to a Rust `for _ in 0..(EXPR) { … }` loop (parsed
//!   brace-free so the body `{` is not misread), so the layer/step count is a
//!   config value. A body `let` that rebinds an already-declared binding is
//!   *loop-carried* (threaded through a fresh `__rlx_carry_*` cell and
//!   re-exposed after the loop), mirroring the shadowing of a literal unroll;
//!   fresh body names are per-iteration locals. Literal `repeat N` /
//!   `repeat i in a..b` still unroll at macro-expansion time.
//! * **`bind t;` / `bind a, b;`.** Adopt an outer-scope Rust `Tensor` variable
//!   of the same name (a baked constant, a codebook param) as an in-scope
//!   binding — auto-borrowed in method args, cloned in expressions — so
//!   `x.synth_matmul(idx, cb, ~ed, ~ne)` reads `idx`/`cb` from outer scope. (The
//!   no-new-grammar alternative is the `(&t)` / `~&t` reference escape.)
//! * **`bind cb[], idx[];` (indexed collection).** The `[]` marker adopts an
//!   outer *collection* — a `Vec<Tensor>` / `&[Tensor]` — instead of a single
//!   tensor. Inside a `repeat i in a..b`, `cb[i]` / `idx[i]` adopt the i-th
//!   element (the loop index flows into the outer `Vec` access), in both
//!   method-arg (`x.synth_matmul(idx[i], cb[i], …)`, auto-borrowed to
//!   `&(idx[i])`) and expression (`h @ cb[i]`) position. This writes a
//!   per-layer-distinct-codebook synthesis transformer as one block.
//! * **`layers[i].field` (struct-collection field).** A collection index may
//!   carry a trailing `.field` chain (`layers[i].cb`, `layers[i].a.b`), so a
//!   whole `Vec<LayerParams>` struct is adopted with ONE `bind layers[];` and
//!   each per-layer tensor is `layers[i].<field>` — killing the ~40 separate
//!   collection binds a synth transformer would otherwise need. Lowers to
//!   `&(layers[i].field)` (method-arg) / `(layers[i].field).clone()` (expr),
//!   with the repeat index substituted per iteration. Only field access (not a
//!   method call) may follow the index; `w[i].m(…)` still parses as a method.

use proc_macro2::{Group, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::{ToTokens, quote};
use std::collections::{HashMap, HashSet};
use syn::{
    Ident, Lit, LitStr, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Entry point for the `__rlx_build!` proc macro.
pub fn rlx_build_impl(input: TokenStream) -> TokenStream {
    let dsl = match syn::parse2::<GraphDsl>(input) {
        Ok(dsl) => dsl,
        Err(e) => return e.to_compile_error(),
    };
    let dsl = match dsl.lower() {
        Ok(dsl) => dsl,
        Err(e) => return e.to_compile_error(),
    };
    if let Err(e) = dsl.check() {
        return e.to_compile_error();
    }
    dsl.codegen()
}

/// Entry point for the `__rlx_expr!` proc macro — the "Rust bridge". Evaluates a
/// SINGLE `rlx!`-grammar expression where every identifier is an in-scope Rust
/// `Tensor` variable (not a declared graph binding), returning the resulting
/// `Tensor`. Lets ordinary Rust `for` loops / config drive graph structure while
/// the DSL expresses the per-step math:
///
/// ```ignore
/// let mut h = embed;
/// for l in &layers {
///     h = rlx_expr!(gelu(linear(h, l.w1, l.b1)));   // h, l.w1 … are Rust values
/// }
/// ```
pub fn rlx_expr_impl(input: TokenStream) -> TokenStream {
    let parser = |s: ParseStream| parse_expr(s, 0);
    let expr = match syn::parse::Parser::parse2(parser, input) {
        Ok(e) => e,
        Err(e) => return e.to_compile_error(),
    };
    // Resolve `f(x)` sugar / builtins (no fn defs or hoisting in an expr).
    let funcs = HashMap::new();
    let mut ctx = Ctx {
        funcs: &funcs,
        counter: 0,
        active: Vec::new(),
        collection_binds: HashSet::new(),
    };
    let mut hoist = Vec::new();
    let lowered = match lower_expr(expr, &mut ctx, &mut hoist) {
        Ok(e) => e,
        Err(e) => return e.to_compile_error(),
    };
    // Every identifier names a Rust `Tensor`, so treat them all as bindings for
    // bare-ident auto-borrow in escape-hatch method args.
    let mut vars = HashSet::new();
    collect_idents(&lowered, &mut vars);
    lowered.emit(&vars)
}

/// Collect every identifier referenced in an expression (Var/index base and
/// bare-ident escape-hatch args), so `rlx_expr!` auto-borrows them like `rlx!`.
fn collect_idents(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Var(id) => {
            out.insert(id.to_string());
        }
        Expr::Num(_) => {}
        Expr::Neg(x) => collect_idents(x, out),
        Expr::Bin(_, a, b, _) | Expr::MatMul(a, b, _) | Expr::Cmp(_, a, b, _) => {
            collect_idents(a, out);
            collect_idents(b, out);
        }
        Expr::MethodDsl { recv, args, .. } => {
            collect_idents(recv, out);
            for (a, _) in args {
                collect_idents(a, out);
            }
        }
        Expr::Linear { x, w, b, .. } => {
            collect_idents(x, out);
            collect_idents(w, out);
            collect_idents(b, out);
        }
        Expr::Index { base, index, .. } => {
            out.insert(base.to_string());
            collect_idents(index, out);
        }
        // A collection base names an outer Rust `Vec<Tensor>`, not a graph
        // binding, so it isn't collected as a bare tensor ident.
        Expr::AdoptIndex { .. } => {}
        Expr::Method { recv, args, .. } => {
            collect_idents(recv, out);
            for a in args {
                if let syn::Expr::Path(p) = a {
                    if p.qself.is_none()
                        && p.path.leading_colon.is_none()
                        && p.path.segments.len() == 1
                    {
                        out.insert(p.path.segments[0].ident.to_string());
                    }
                }
            }
        }
        Expr::Call { .. } => {}
    }
}

// ── AST ──────────────────────────────────────────────────────────────────

struct GraphDsl {
    name: Option<LitStr>,
    stmts: Vec<Stmt>,
}

#[derive(Clone)]
enum Stmt {
    Decl {
        kind: DeclKind,
        name: Ident,
        /// `param w[N]: …` declares a family `w_0..w_{N-1}` (expanded in `lower`).
        family: Option<usize>,
        /// `param w @ "ir.name": …` overrides the IR/`set_param` name (a `{i}`
        /// placeholder is filled with the family index). Defaults to the ident.
        ir_name: Option<LitStr>,
        shape: TokenStream,
    },
    Const {
        name: Ident,
        value: ConstVal,
    },
    Let {
        name: Ident,
        expr: Expr,
    },
    /// `let (a, b, …) = expr;` — bind each element of a `Vec`-producing
    /// expression (e.g. `split(x, axis, n)` / `x.chunk(..)`).
    LetTuple {
        names: Vec<Ident>,
        expr: Expr,
    },
    Out {
        names: Vec<Ident>,
    },
    /// `tap a, b;` — expose intermediates as extra graph outputs (debugging).
    Tap {
        names: Vec<Ident>,
    },
    /// A reusable subgraph template — collected and inlined during `lower`.
    Fn(FnDef),
    /// `repeat N { … }` — unrolled `N` times during `lower`.
    Repeat {
        count: usize,
        body: Vec<Stmt>,
    },
    /// `repeat i in start..end { … }` — unrolled with the index `i` bound to
    /// each value (usable in family indexing `w[i]` and `@"…{i}…"` keys).
    RepeatIndexed {
        var: Ident,
        start: usize,
        end: usize,
        body: Vec<Stmt>,
    },
    /// `scan carry = init for N { … }` — a compact `Op::Scan` loop (one body
    /// graph, not unrolled). The body is a sequence of `let`s; the last binds
    /// the next carry. Outer bindings referenced in the body are passed as
    /// scan broadcasts.
    Scan {
        carry: Ident,
        init: Expr,
        length: usize,
        body: Vec<(Ident, Expr)>,
    },
    /// `repeat <expr> { … }` where `<expr>` is a RUNTIME `usize` value (an
    /// identifier / field access / `(expr)`), NOT an integer literal. Unlike the
    /// literal `repeat N`/`repeat i in …` (unrolled at macro-expansion time),
    /// this lowers to a Rust `for _ in 0..(<expr>) { … }` loop, so a
    /// config-driven layer/step count works. Body `let`s that rebind an
    /// already-declared binding are threaded as loop-carried values across
    /// iterations (mirroring the shadowing of a literal unroll).
    RepeatRuntime {
        count: TokenStream,
        body: Vec<Stmt>,
    },
    /// `repeat i in start..end { … }` where `start`/`end` are RUNTIME `usize`
    /// expressions (not both integer literals). Unlike the literal
    /// [`RepeatIndexed`](Stmt::RepeatIndexed) (unrolled at macro-expansion), this
    /// lowers to a Rust `for i in (start)..(end) { … }` loop with the index `i`
    /// live in scope — so a config-driven layer count can still index per-layer
    /// `bind`-collections (`cb[i]`) and thread loop-carried rebinds, exactly like
    /// the literal indexed form but at runtime. (Parameter *families* `w[i]`
    /// still need the literal form, since each is a distinct declared binding.)
    RepeatIndexedRuntime {
        var: Ident,
        start: TokenStream,
        end: TokenStream,
        body: Vec<Stmt>,
    },
    /// `bind t;` / `bind a, b;` — adopt an outer-scope Rust `Tensor` variable of
    /// the same name as an in-scope graph binding, so a pre-built constant/param
    /// tensor (e.g. baked u8 quantization indices, or a codebook param) can be
    /// referenced by bare name — auto-borrowed in method args and cloned in
    /// expressions — exactly like a declared `input`/`param`.
    ///
    /// The `[]` marker (`bind cb[], idx[];`) adopts an outer *collection*
    /// (`Vec<Tensor>` / `&[Tensor]`) instead: inside a `repeat i in a..b`, the
    /// element `cb[i]` is adopted per iteration. Each entry carries a flag —
    /// `true` = collection.
    Bind {
        names: Vec<(Ident, bool)>,
    },
}

#[derive(Clone)]
enum ConstVal {
    Scalar {
        neg: bool,
        value: Lit,
        dtype: Ident,
    },
    Array {
        data: Vec<f64>,
        dims: Vec<usize>,
        dtype: Ident,
    },
}

#[derive(Clone)]
enum DeclKind {
    Input,
    Param,
}

#[derive(Clone)]
struct FnDef {
    name: Ident,
    params: Vec<Ident>,
    /// Body is a sequence of `let name = expr;`; the last binds the return.
    body: Vec<(Ident, Expr)>,
}

#[derive(Clone)]
enum Expr {
    /// A previously-declared tensor binding.
    Var(Ident),
    /// A numeric literal (a scalar operand).
    Num(TokenStream),
    /// Unary negation.
    Neg(Box<Expr>),
    /// Elementwise binary op (`+ - * /`); the `Span` is the operator token.
    Bin(BinKind, Box<Expr>, Box<Expr>, Span),
    /// Matrix multiply (`@` / `matmul`); the `Span` is the operator token.
    MatMul(Box<Expr>, Box<Expr>, Span),
    /// Elementwise comparison (`== != < <= > >=`) → `Bool` tensor.
    Cmp(CmpKind, Box<Expr>, Box<Expr>, Span),
    /// HF-style linear `linear(x, w, b)` → `x · Wᵀ + b`, optionally with a fused
    /// activation folded in from a wrapping `gelu(linear(..))` etc.
    Linear {
        x: Box<Expr>,
        w: Box<Expr>,
        b: Box<Expr>,
        /// `Activation` variant ident (e.g. `Gelu`), or `None`.
        act: Option<Ident>,
        /// Span of the `linear` call, for shape-mismatch diagnostics.
        span: Span,
    },
    /// `base[index]` — a parameter-family element, resolved to `base_<k>` during
    /// `repeat i in 0..N` unrolling (or a literal index). A trailing `.field`
    /// chain (`fields`, usually empty) is only valid on a `bind` collection base
    /// (`layers[i].w`), routed to [`Expr::AdoptIndex`] at lowering. `inner` is an
    /// optional trailing `[s]` after the field chain — selecting an element of a
    /// `Vec<Tensor>` *field* (`layers[i].cb[s]`), with `s` an inner `repeat`
    /// index; only valid alongside a non-empty `fields`, and likewise only on a
    /// `bind` collection base.
    Index {
        base: Ident,
        index: Box<Expr>,
        fields: Vec<Ident>,
        inner: Option<Box<Expr>>,
    },
    /// `base[index]` (optionally `.field.chain`, optionally a trailing `[inner]`)
    /// where `base` is a `bind`-adopted outer *collection* (`Vec<Tensor>`/
    /// `&[Tensor]`, or a `Vec<Struct>` when `fields` selects a tensor field, or a
    /// `Vec<Tensor>` field further indexed by `inner`) — emits
    /// `base[index].field…[inner].clone()` (the indexed Tensor is adopted into the
    /// graph like a scalar `bind`). `index`/`inner` are Rust index tokens (a
    /// literal from `repeat i in …`, or an outer/inner `usize` loop var).
    AdoptIndex {
        base: Ident,
        index: TokenStream,
        fields: Vec<Ident>,
        inner: Option<TokenStream>,
    },
    /// Unresolved `f(args)` — resolved by `lower` into a method call, builtin
    /// sugar, or a `fn` inline.
    Call {
        name: Ident,
        args: Vec<Expr>,
        /// Optional per-argument name (`blk(x: a)`); honored only for `fn` calls.
        arg_names: Vec<Option<Ident>>,
    },
    /// Method call whose args are DSL sub-expressions (produced by sugar). Each
    /// arg is emitted as `&(arg)` (`Ref`) or `arg` (`Scalar`).
    MethodDsl {
        recv: Box<Expr>,
        method: Ident,
        args: Vec<(Expr, ArgMode)>,
    },
    /// Escape-hatch method call — args are raw Rust expressions.
    Method {
        recv: Box<Expr>,
        name: Ident,
        args: Vec<syn::Expr>,
    },
}

#[derive(Clone, Copy)]
enum ArgMode {
    /// A tensor operand — emit `&(expr)`.
    Ref,
    /// A scalar operand — emit the bare `expr` (an `f64`). The target methods
    /// take `impl Into<BinaryRhs>` (or a plain `f64`, e.g. `clamp`), so a
    /// scalar is accepted directly: `maximum(x, 2.0)`, `x ** 2`, `x > 0.0`.
    Scalar,
    /// A raw config literal emitted verbatim (no `as f64` cast), so Rust infers
    /// the target type — e.g. a `usize` axis/head-dim (`rope(x,c,s,64)`) or a
    /// gather axis. Only meaningful for a numeric-literal `expr`.
    Raw,
}

impl ArgMode {
    /// For a binary-op operand: pass a scalar by value, borrow a tensor.
    fn for_operand(e: &Expr) -> Self {
        if e.is_scalar() {
            ArgMode::Scalar
        } else {
            ArgMode::Ref
        }
    }
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinKind {
    fn as_char(self) -> char {
        match self {
            BinKind::Add => '+',
            BinKind::Sub => '-',
            BinKind::Mul => '*',
            BinKind::Div => '/',
        }
    }

    /// The operator as a single token carrying `span`, so a type error on the
    /// lowered expression points back at the source operator.
    fn spanned(self, span: Span) -> TokenStream {
        let mut p = Punct::new(self.as_char(), Spacing::Alone);
        p.set_span(span);
        TokenStream::from(TokenTree::Punct(p))
    }
}

#[derive(Clone, Copy)]
enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpKind {
    fn method(self) -> &'static str {
        match self {
            CmpKind::Eq => "eq",
            CmpKind::Ne => "ne",
            CmpKind::Lt => "lt",
            CmpKind::Le => "le",
            CmpKind::Gt => "gt",
            CmpKind::Ge => "ge",
        }
    }

    /// The equivalent comparison with operands swapped (`a < b` ⇔ `b > a`),
    /// used to lower a scalar-on-the-left comparison to a tensor method.
    fn swapped(self) -> CmpKind {
        match self {
            CmpKind::Lt => CmpKind::Gt,
            CmpKind::Gt => CmpKind::Lt,
            CmpKind::Le => CmpKind::Ge,
            CmpKind::Ge => CmpKind::Le,
            CmpKind::Eq => CmpKind::Eq,
            CmpKind::Ne => CmpKind::Ne,
        }
    }
}

// ── Parsing ──────────────────────────────────────────────────────────────

impl Parse for GraphDsl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Optional leading `graph "name";`.
        let mut name = None;
        if peek_kw(input, "graph") {
            input.parse::<Ident>()?; // `graph`
            name = Some(input.parse::<LitStr>()?);
            input.parse::<Token![;]>()?;
        }

        let mut stmts = Vec::new();
        while !input.is_empty() {
            stmts.push(parse_stmt(input)?);
        }
        Ok(GraphDsl { name, stmts })
    }
}

/// True if the next token is the identifier `kw` (a contextual keyword).
fn peek_kw(input: ParseStream, kw: &str) -> bool {
    input
        .fork()
        .parse::<Ident>()
        .map(|id| id == kw)
        .unwrap_or(false)
}

/// Parse a `{ stmt* }` block body into a statement list.
fn parse_braced_stmts(input: ParseStream) -> syn::Result<Vec<Stmt>> {
    let body_buf;
    syn::braced!(body_buf in input);
    let mut body = Vec::new();
    while !body_buf.is_empty() {
        body.push(parse_stmt(&body_buf)?);
    }
    Ok(body)
}

fn parse_stmt(input: ParseStream) -> syn::Result<Stmt> {
    if input.peek(Token![let]) {
        input.parse::<Token![let]>()?;
        // Tuple destructuring: `let (a, b, …) = expr;`.
        if input.peek(syn::token::Paren) {
            let pat;
            syn::parenthesized!(pat in input);
            let names: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&pat)?;
            input.parse::<Token![=]>()?;
            let expr = parse_expr(input, 0)?;
            input.parse::<Token![;]>()?;
            return Ok(Stmt::LetTuple {
                names: names.into_iter().collect(),
                expr,
            });
        }
        let name = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let expr = parse_expr(input, 0)?;
        input.parse::<Token![;]>()?;
        return Ok(Stmt::Let { name, expr });
    }

    if input.peek(Token![const]) {
        return parse_const(input);
    }

    if input.peek(Token![fn]) {
        return parse_fn(input);
    }

    // Contextual-keyword statements: input / param / out / output / repeat.
    let kw = input.parse::<Ident>()?;
    match kw.to_string().as_str() {
        "input" | "param" => {
            let name = input.parse::<Ident>()?;
            // Optional family size: `param w[N]: …`.
            let family = if input.peek(syn::token::Bracket) {
                let fc;
                syn::bracketed!(fc in input);
                Some(fc.parse::<syn::LitInt>()?.base10_parse::<usize>()?)
            } else {
                None
            };
            // Optional IR-name override: `param w @ "ir.name": …`.
            let ir_name = if input.peek(Token![@]) {
                input.parse::<Token![@]>()?;
                Some(input.parse::<LitStr>()?)
            } else {
                None
            };
            input.parse::<Token![:]>()?;
            let content;
            syn::bracketed!(content in input);
            let shape = content.parse::<TokenStream>()?;
            input.parse::<Token![;]>()?;
            let kind = if kw == "input" {
                DeclKind::Input
            } else {
                DeclKind::Param
            };
            Ok(Stmt::Decl {
                kind,
                name,
                family,
                ir_name,
                shape,
            })
        }
        "out" | "output" => {
            let mut names = vec![input.parse::<Ident>()?];
            while input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                names.push(input.parse::<Ident>()?);
            }
            input.parse::<Token![;]>()?;
            Ok(Stmt::Out { names })
        }
        "tap" => {
            let mut names = vec![input.parse::<Ident>()?];
            while input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                names.push(input.parse::<Ident>()?);
            }
            input.parse::<Token![;]>()?;
            Ok(Stmt::Tap { names })
        }
        "repeat" => {
            // `repeat i in start..end { … }` (indexed). Literal bounds unroll at
            // macro-expansion (`RepeatIndexed`); a runtime bound on either side
            // emits a real `for i in start..end` loop (`RepeatIndexedRuntime`).
            if input.peek(Ident) && input.peek2(Token![in]) {
                let var = input.parse::<Ident>()?;
                input.parse::<Token![in]>()?;
                let range = input.call(syn::Expr::parse_without_eager_brace)?;
                let body = parse_braced_stmts(input)?;
                let syn::Expr::Range(r) = range else {
                    return Err(syn::Error::new_spanned(
                        range,
                        "`repeat i in …` expects a `start..end` range",
                    ));
                };
                let (Some(lo), Some(hi)) = (r.start.as_deref(), r.end.as_deref()) else {
                    return Err(syn::Error::new_spanned(
                        &r,
                        "`repeat i in start..end` needs both bounds",
                    ));
                };
                // Both bounds integer literals → unroll at macro time (also the
                // only form that can index a parameter *family* `w[i]`).
                if let (Some(s), Some(e)) = (usize_lit(lo), usize_lit(hi)) {
                    return Ok(Stmt::RepeatIndexed {
                        var,
                        start: s,
                        end: e,
                        body,
                    });
                }
                return Ok(Stmt::RepeatIndexedRuntime {
                    var,
                    start: quote!(#lo),
                    end: quote!(#hi),
                    body,
                });
            }
            // Literal `repeat N { … }` — unrolled `N` times at macro-expansion.
            if input.peek(syn::LitInt) && input.peek2(syn::token::Brace) {
                let count = input.parse::<syn::LitInt>()?.base10_parse::<usize>()?;
                let body = parse_braced_stmts(input)?;
                return Ok(Stmt::Repeat { count, body });
            }
            // Runtime `repeat <expr> { … }` — a Rust `for _ in 0..<expr>` loop.
            // `parse_without_eager_brace` stops before the body `{`, so a bare
            // ident / field access / arithmetic count is not misread as a struct
            // literal.
            let count_expr = input.call(syn::Expr::parse_without_eager_brace)?;
            let body = parse_braced_stmts(input)?;
            Ok(Stmt::RepeatRuntime {
                count: quote!(#count_expr),
                body,
            })
        }
        "bind" => {
            // Each name is a scalar `bind t` or a collection `bind cb[]` (empty
            // brackets mark an outer `Vec<Tensor>`/`&[Tensor]`, indexed `cb[i]`).
            let mut names = Vec::new();
            loop {
                let id = input.parse::<Ident>()?;
                let is_collection = if input.peek(syn::token::Bracket) {
                    let content;
                    syn::bracketed!(content in input);
                    if !content.is_empty() {
                        return Err(content.error(
                            "`bind name[]` takes EMPTY brackets — it marks an outer \
                             collection (`Vec<Tensor>`/`&[Tensor]`), indexed `name[i]`",
                        ));
                    }
                    true
                } else {
                    false
                };
                names.push((id, is_collection));
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
            input.parse::<Token![;]>()?;
            Ok(Stmt::Bind { names })
        }
        "scan" => {
            let carry = input.parse::<Ident>()?;
            input.parse::<Token![=]>()?;
            let init = parse_expr(input, 0)?;
            input.parse::<Token![for]>()?;
            let length = input.parse::<syn::LitInt>()?.base10_parse::<usize>()?;
            let body_buf;
            syn::braced!(body_buf in input);
            let mut body = Vec::new();
            while !body_buf.is_empty() {
                body_buf.parse::<Token![let]>()?;
                let ln = body_buf.parse::<Ident>()?;
                body_buf.parse::<Token![=]>()?;
                let e = parse_expr(&body_buf, 0)?;
                body_buf.parse::<Token![;]>()?;
                body.push((ln, e));
            }
            if body.is_empty() {
                return Err(syn::Error::new(
                    carry.span(),
                    format!(
                        "scan `{carry}` has an empty body — it must end in a `let` to \
                             produce the next carry"
                    ),
                ));
            }
            Ok(Stmt::Scan {
                carry,
                init,
                length,
                body,
            })
        }
        other => Err(syn::Error::new(
            kw.span(),
            format!(
                "unknown rlx! statement `{other}` — expected one of \
                 `input`, `param`, `const`, `bind`, `let`, `fn`, `repeat`, `scan`, `tap`, `out`"
            ),
        )),
    }
}

fn parse_const(input: ParseStream) -> syn::Result<Stmt> {
    input.parse::<Token![const]>()?;
    let name = input.parse::<Ident>()?;
    input.parse::<Token![=]>()?;

    // Array constant: `const w = [[0, 1], [1, 0]] : F32;`
    if input.peek(syn::token::Bracket) {
        let arr = input.parse::<syn::Expr>()?;
        input.parse::<Token![:]>()?;
        let dtype = input.parse::<Ident>()?;
        input.parse::<Token![;]>()?;
        let (data, dims) = extract_array(&arr)?;
        if data.is_empty() {
            return Err(syn::Error::new(
                name.span(),
                "empty array `const` is not allowed",
            ));
        }
        return Ok(Stmt::Const {
            name,
            value: ConstVal::Array { data, dims, dtype },
        });
    }

    // Scalar constant: `const eps = 1e-6 : F32;`
    let neg = input.peek(Token![-]);
    if neg {
        input.parse::<Token![-]>()?;
    }
    let value = input.parse::<Lit>()?;
    input.parse::<Token![:]>()?;
    let dtype = input.parse::<Ident>()?;
    input.parse::<Token![;]>()?;
    Ok(Stmt::Const {
        name,
        value: ConstVal::Scalar { neg, value, dtype },
    })
}

fn parse_fn(input: ParseStream) -> syn::Result<Stmt> {
    input.parse::<Token![fn]>()?;
    let name = input.parse::<Ident>()?;
    let params_buf;
    syn::parenthesized!(params_buf in input);
    let params: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&params_buf)?;
    let body_buf;
    syn::braced!(body_buf in input);
    let mut body = Vec::new();
    while !body_buf.is_empty() {
        body_buf.parse::<Token![let]>()?;
        let ln = body_buf.parse::<Ident>()?;
        body_buf.parse::<Token![=]>()?;
        let e = parse_expr(&body_buf, 0)?;
        body_buf.parse::<Token![;]>()?;
        body.push((ln, e));
    }
    if body.is_empty() {
        return Err(syn::Error::new(
            name.span(),
            format!("fn `{name}` has an empty body — it must end in a `let` to return"),
        ));
    }
    Ok(Stmt::Fn(FnDef {
        name,
        params: params.into_iter().collect(),
        body,
    }))
}

/// Recursively extract row-major values + dims from a nested numeric array
/// literal (`[[0, 1], [1, 0]]`). Rejects ragged arrays and non-literal entries.
fn extract_array(e: &syn::Expr) -> syn::Result<(Vec<f64>, Vec<usize>)> {
    if let syn::Expr::Array(a) = e {
        let mut rows: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
        for el in &a.elems {
            rows.push(extract_array(el)?);
        }
        if rows.is_empty() {
            return Ok((vec![], vec![0]));
        }
        let sub = rows[0].1.clone();
        for (i, r) in rows.iter().enumerate() {
            if r.1 != sub {
                return Err(syn::Error::new(
                    a.elems[i].span(),
                    "ragged array `const` — every row must have the same shape",
                ));
            }
        }
        let mut data = Vec::new();
        for r in &rows {
            data.extend_from_slice(&r.0);
        }
        let mut dims = vec![rows.len()];
        dims.extend(sub);
        Ok((data, dims))
    } else if let Some(v) = lit_to_f64(e) {
        Ok((vec![v], vec![]))
    } else {
        Err(syn::Error::new(
            e.span(),
            "array `const` entries must be numeric literals (e.g. `1.0`, `-2`, `3`)",
        ))
    }
}

/// A numeric literal (allowing a leading unary `-` and paren/group wrappers).
fn lit_to_f64(e: &syn::Expr) -> Option<f64> {
    match e {
        syn::Expr::Lit(l) => match &l.lit {
            syn::Lit::Int(i) => i.base10_parse::<f64>().ok(),
            syn::Lit::Float(f) => f.base10_parse::<f64>().ok(),
            _ => None,
        },
        syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => lit_to_f64(&u.expr).map(|v| -v),
        syn::Expr::Group(g) => lit_to_f64(&g.expr),
        syn::Expr::Paren(p) => lit_to_f64(&p.expr),
        _ => None,
    }
}

use syn::spanned::Spanned as _;

/// An infix operator. `@` shares the `* / %` band (NumPy-style); `**` is
/// tighter and right-associative; comparisons are the loosest.
#[derive(Clone, Copy)]
enum InfixOp {
    Bin(BinKind),
    Mat,
    Mod,
    Pow,
    Cmp(CmpKind),
}

/// `(op, left_bp, right_bp)`. Left-associative ops use `right_bp = left_bp + 1`;
/// right-associative `**` uses `right_bp = left_bp`.
fn peek_infix(input: ParseStream) -> Option<(InfixOp, u8, u8)> {
    if input.peek(Token![*]) && input.peek2(Token![*]) {
        Some((InfixOp::Pow, 7, 7))
    } else if input.peek(Token![+]) {
        Some((InfixOp::Bin(BinKind::Add), 3, 4))
    } else if input.peek(Token![-]) {
        Some((InfixOp::Bin(BinKind::Sub), 3, 4))
    } else if input.peek(Token![*]) {
        Some((InfixOp::Bin(BinKind::Mul), 5, 6))
    } else if input.peek(Token![/]) {
        Some((InfixOp::Bin(BinKind::Div), 5, 6))
    } else if input.peek(Token![%]) {
        Some((InfixOp::Mod, 5, 6))
    } else if input.peek(Token![@]) {
        Some((InfixOp::Mat, 5, 6))
    } else if input.peek(Token![==]) {
        Some((InfixOp::Cmp(CmpKind::Eq), 1, 2))
    } else if input.peek(Token![!=]) {
        Some((InfixOp::Cmp(CmpKind::Ne), 1, 2))
    } else if input.peek(Token![<=]) {
        Some((InfixOp::Cmp(CmpKind::Le), 1, 2))
    } else if input.peek(Token![>=]) {
        Some((InfixOp::Cmp(CmpKind::Ge), 1, 2))
    } else if input.peek(Token![<]) {
        Some((InfixOp::Cmp(CmpKind::Lt), 1, 2))
    } else if input.peek(Token![>]) {
        Some((InfixOp::Cmp(CmpKind::Gt), 1, 2))
    } else {
        None
    }
}

/// A `syn::Expr` that is a bare unsuffixed/`usize` integer literal → its value,
/// else `None`. Distinguishes a literal `repeat i in 0..3` (macro-time unroll)
/// from a runtime `repeat i in 0..n` (`for` loop).
fn usize_lit(e: &syn::Expr) -> Option<usize> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: syn::Lit::Int(i),
        ..
    }) = e
    {
        return i.base10_parse::<usize>().ok();
    }
    None
}

/// Pratt parser. `min_bp` is the minimum binding power that keeps a binary
/// operator in this sub-expression (precedence climbing).
fn parse_expr(input: ParseStream, min_bp: u8) -> syn::Result<Expr> {
    let mut lhs = parse_prefix(input)?;

    while let Some((op, l_bp, r_bp)) = peek_infix(input) {
        if l_bp < min_bp {
            break;
        }
        let span = input.span(); // the operator token's span
        consume_infix(input, op)?;
        let rhs = parse_expr(input, r_bp)?;
        lhs = build_infix(op, lhs, rhs, span);
    }

    Ok(lhs)
}

fn build_infix(op: InfixOp, lhs: Expr, rhs: Expr, span: Span) -> Expr {
    match op {
        InfixOp::Mat => Expr::MatMul(Box::new(lhs), Box::new(rhs), span),
        InfixOp::Bin(kind) => Expr::Bin(kind, Box::new(lhs), Box::new(rhs), span),
        InfixOp::Cmp(c) => Expr::Cmp(c, Box::new(lhs), Box::new(rhs), span),
        InfixOp::Mod => {
            let mode = ArgMode::for_operand(&rhs);
            Expr::MethodDsl {
                recv: Box::new(lhs),
                method: Ident::new("rem", span),
                args: vec![(rhs, mode)],
            }
        }
        InfixOp::Pow => {
            let mode = ArgMode::for_operand(&rhs);
            Expr::MethodDsl {
                recv: Box::new(lhs),
                method: Ident::new("pow", span),
                args: vec![(rhs, mode)],
            }
        }
    }
}

fn consume_infix(input: ParseStream, op: InfixOp) -> syn::Result<()> {
    match op {
        InfixOp::Pow => {
            input.parse::<Token![*]>()?;
            input.parse::<Token![*]>()?;
            Ok(())
        }
        InfixOp::Mat => input.parse::<Token![@]>().map(drop),
        InfixOp::Mod => input.parse::<Token![%]>().map(drop),
        InfixOp::Bin(BinKind::Add) => input.parse::<Token![+]>().map(drop),
        InfixOp::Bin(BinKind::Sub) => input.parse::<Token![-]>().map(drop),
        InfixOp::Bin(BinKind::Mul) => input.parse::<Token![*]>().map(drop),
        InfixOp::Bin(BinKind::Div) => input.parse::<Token![/]>().map(drop),
        InfixOp::Cmp(CmpKind::Eq) => input.parse::<Token![==]>().map(drop),
        InfixOp::Cmp(CmpKind::Ne) => input.parse::<Token![!=]>().map(drop),
        InfixOp::Cmp(CmpKind::Le) => input.parse::<Token![<=]>().map(drop),
        InfixOp::Cmp(CmpKind::Ge) => input.parse::<Token![>=]>().map(drop),
        InfixOp::Cmp(CmpKind::Lt) => input.parse::<Token![<]>().map(drop),
        InfixOp::Cmp(CmpKind::Gt) => input.parse::<Token![>]>().map(drop),
    }
}

fn parse_prefix(input: ParseStream) -> syn::Result<Expr> {
    if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        let operand = parse_prefix(input)?;
        return Ok(Expr::Neg(Box::new(operand)));
    }
    parse_postfix(input)
}

fn parse_postfix(input: ParseStream) -> syn::Result<Expr> {
    let mut e = parse_atom(input)?;
    while input.peek(Token![.]) {
        input.parse::<Token![.]>()?;
        let name = input.parse::<Ident>()?;
        let content;
        syn::parenthesized!(content in input);
        // Escape hatch: method args are real Rust expressions, parsed as
        // `syn::Expr` (not raw tokens) so comma-splitting stays correct even with
        // turbofish, closures, or nested generics. A leading `~` marks an arg as
        // pass-BY-VALUE — it opts that arg out of the bare-identifier auto-borrow
        // (`k` → `&k`) so a scalar variable/const (`~eps`, `~num_heads`) reaches a
        // by-value method parameter (`f32`/`usize`/enum) instead of `&f32`. `~`
        // isn't a Rust prefix operator, so it's peeled here before `syn` parses
        // the arg; the peeled arg is then wrapped like the documented `(value)`
        // escape so every downstream pass forwards it verbatim.
        let args = parse_method_args(&content)?;
        e = Expr::Method {
            recv: Box::new(e),
            name,
            args,
        };
    }
    Ok(e)
}

/// Parse a comma-separated escape-hatch method-arg list, honoring a leading `~`
/// (pass-by-value marker) on any argument. Mirrors `Punctuated::parse_terminated`
/// (trailing comma allowed) but peels the `~` first, since it is not a valid
/// Rust prefix and would otherwise fail `syn::Expr` parsing.
fn parse_method_args(content: ParseStream) -> syn::Result<Vec<syn::Expr>> {
    let mut args = Vec::new();
    while !content.is_empty() {
        let by_value = content.peek(Token![~]);
        if by_value {
            content.parse::<Token![~]>()?;
        }
        let expr = content.parse::<syn::Expr>()?;
        args.push(if by_value { by_value_arg(expr) } else { expr });
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        } else {
            break;
        }
    }
    if !content.is_empty() {
        return Err(content.error("unexpected tokens after method arguments"));
    }
    Ok(args)
}

/// Wrap a `~expr` by-value arg so every later pass treats it as raw Rust: a
/// parenthesised expression is never a bare single-segment path, so it dodges the
/// auto-borrow (`method_args`), the unknown-binding check (`check_method_arg`),
/// and the free-var/ident collectors — exactly the documented `(value)` escape,
/// reached via the terser `~value`. Semantically `(expr)` == `expr` as an arg.
fn by_value_arg(expr: syn::Expr) -> syn::Expr {
    syn::Expr::Paren(syn::ExprParen {
        attrs: Vec::new(),
        paren_token: syn::token::Paren::default(),
        expr: Box::new(expr),
    })
}

fn parse_atom(input: ParseStream) -> syn::Result<Expr> {
    // Parenthesised sub-expression.
    if input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in input);
        let inner = parse_expr(&content, 0)?;
        if !content.is_empty() {
            return Err(content.error("unexpected tokens after expression"));
        }
        return Ok(inner);
    }

    // Numeric literal (scalar operand).
    if input.peek(syn::LitInt) || input.peek(syn::LitFloat) {
        let lit = input.parse::<Lit>()?;
        return Ok(Expr::Num(quote!(#lit)));
    }

    // Identifier: a bare var, a function-call, or a family index `w[i]`.
    if input.peek(Ident) {
        let id = input.parse::<Ident>()?;
        if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let (args, arg_names) = parse_arg_list(&content)?;
            return make_call(id, args, arg_names);
        }
        if input.peek(syn::token::Bracket) {
            let ic;
            syn::bracketed!(ic in input);
            let index = parse_expr(&ic, 0)?;
            // A trailing *bare* field chain `[i].a.b` selects a tensor field of a
            // `bind` collection element (`layers[i].w`). Stop at a `.method(` —
            // `!peek3(Paren)` keeps method calls (`w[i].gelu()`) for `parse_postfix`.
            let mut fields = Vec::new();
            while input.peek(Token![.]) && input.peek2(Ident) && !input.peek3(syn::token::Paren) {
                input.parse::<Token![.]>()?;
                fields.push(input.parse::<Ident>()?);
            }
            // An optional trailing `[s]` indexes a `Vec<Tensor>` *field* of the
            // collection element (`layers[i].cb[s]`), where `s` is an inner
            // `repeat s in …` index. Only meaningful after a field chain (the
            // element is a struct), so gate on a non-empty `fields`.
            let inner = if !fields.is_empty() && input.peek(syn::token::Bracket) {
                let jc;
                syn::bracketed!(jc in input);
                Some(Box::new(parse_expr(&jc, 0)?))
            } else {
                None
            };
            return Ok(Expr::Index {
                base: id,
                index: Box::new(index),
                fields,
                inner,
            });
        }
        return Ok(Expr::Var(id));
    }

    Err(input.error("expected a tensor expression (a binding name, `f(x)`, `a @ b`, or `(…)`)"))
}

/// Arguments to `f(…)` sugar are DSL expressions (so `gelu(x @ w + b)` works),
/// unlike `.method(…)` escape-hatch args which are raw Rust. Each may be named
/// (`blk(x: a, w: b)`) — only user `fn` calls honor the names.
fn parse_arg_list(input: ParseStream) -> syn::Result<(Vec<Expr>, Vec<Option<Ident>>)> {
    let mut args = Vec::new();
    let mut names = Vec::new();
    if input.is_empty() {
        return Ok((args, names));
    }
    loop {
        // `ident : expr` is a named arg (a single `:`, not the `::` of a path).
        let named = if input.peek(Ident) && input.peek2(Token![:]) {
            let n = input.parse::<Ident>()?;
            input.parse::<Token![:]>()?;
            Some(n)
        } else {
            None
        };
        args.push(parse_expr(input, 0)?);
        names.push(named);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }
    Ok((args, names))
}

/// Map `f(args)` to an AST node. `matmul`/`mm` are lowered to `@` here; every
/// other call becomes an unresolved [`Expr::Call`] that `lower` turns into
/// builtin sugar, a no-extra-arg method, or a `fn` inline.
fn make_call(id: Ident, mut args: Vec<Expr>, arg_names: Vec<Option<Ident>>) -> syn::Result<Expr> {
    let name = id.to_string();
    if name == "matmul" || name == "mm" {
        if args.len() != 2 {
            return Err(syn::Error::new(
                id.span(),
                format!("`{name}(a, b)` takes exactly two operands"),
            ));
        }
        if arg_names.iter().any(Option::is_some) {
            return Err(syn::Error::new(
                id.span(),
                "`matmul` takes positional operands",
            ));
        }
        let b = args.pop().unwrap();
        let a = args.pop().unwrap();
        return Ok(Expr::MatMul(Box::new(a), Box::new(b), id.span()));
    }
    Ok(Expr::Call {
        name: id,
        args,
        arg_names,
    })
}

// ── Lowering: resolve calls, inline fns, unroll repeats ────────────────────

/// Threaded lowering state: the `fn` table, a fresh-name counter, and the
/// active-inline stack (for recursion detection). Bundled so the lowering
/// functions take one `&mut Ctx` instead of three parameters. The hoist buffer
/// (`out`) stays a separate argument because it's swapped per nested scope
/// (a scan/fn body hoists into its own buffer, not the outer graph).
struct Ctx<'a> {
    funcs: &'a HashMap<String, FnDef>,
    counter: u32,
    active: Vec<String>,
    /// Names `bind`-adopted as outer *collections* (`bind cb[]`), so an
    /// `Expr::Index` on one lowers to `cb[i].clone()` (adopt the indexed Tensor)
    /// rather than to a param-family element `cb_i`.
    collection_binds: HashSet<String>,
}

impl GraphDsl {
    /// Rewrite the statement list into a flat sequence of `Decl`/`Const`/`Let`/
    /// `Out` — resolving `f(x)` sugar, inlining `fn` bodies, and unrolling
    /// `repeat` blocks. After this, no `Call`/`Fn`/`Repeat` nodes remain.
    fn lower(self) -> syn::Result<Self> {
        // Collect fn definitions (visible regardless of source order).
        let mut funcs: HashMap<String, FnDef> = HashMap::new();
        let mut rest: Vec<Stmt> = Vec::new();
        for s in self.stmts {
            if let Stmt::Fn(f) = s {
                if funcs.insert(f.name.to_string(), f.clone()).is_some() {
                    return Err(syn::Error::new(
                        f.name.span(),
                        format!("rlx! fn `{}` is defined more than once", f.name),
                    ));
                }
            } else {
                rest.push(s);
            }
        }

        // Names adopted as outer collections (`bind cb[]`), visible everywhere.
        let mut collection_binds: HashSet<String> = HashSet::new();
        for s in &rest {
            if let Stmt::Bind { names } = s {
                for (id, is_coll) in names {
                    if *is_coll {
                        collection_binds.insert(id.to_string());
                    }
                }
            }
        }

        let mut ctx = Ctx {
            funcs: &funcs,
            counter: 0,
            active: Vec::new(),
            collection_binds,
        };
        let mut out: Vec<Stmt> = Vec::new();
        lower_stmts(rest, &mut ctx, &mut out)?;
        Ok(GraphDsl {
            name: self.name,
            stmts: out,
        })
    }
}

fn lower_stmts(stmts: Vec<Stmt>, ctx: &mut Ctx, out: &mut Vec<Stmt>) -> syn::Result<()> {
    for s in stmts {
        match s {
            Stmt::Let { name, expr } => {
                let e = lower_expr(expr, ctx, out)?;
                out.push(Stmt::Let { name, expr: e });
            }
            Stmt::LetTuple { names, expr } => {
                let e = lower_expr(expr, ctx, out)?;
                out.push(Stmt::LetTuple { names, expr: e });
            }
            Stmt::Repeat { count, body } => {
                validate_loop_body(&body, "repeat")?;
                for _ in 0..count {
                    lower_stmts(body.clone(), ctx, out)?;
                }
            }
            Stmt::RepeatRuntime { count, body } => {
                validate_loop_body(&body, "repeat")?;
                // Lower the body into its OWN buffer so any fn-inline hoists stay
                // inside the loop, then keep the wrapper for `for`-loop codegen.
                let mut body_buf: Vec<Stmt> = Vec::new();
                lower_stmts(body, ctx, &mut body_buf)?;
                out.push(Stmt::RepeatRuntime {
                    count,
                    body: body_buf,
                });
            }
            Stmt::RepeatIndexed {
                var,
                start,
                end,
                body,
            } => {
                validate_loop_body(&body, "repeat")?;
                for k in start..end {
                    let lit = proc_macro2::Literal::usize_unsuffixed(k);
                    let mut subst = HashMap::new();
                    subst.insert(var.to_string(), Expr::Num(quote!(#lit)));
                    let substituted: Vec<Stmt> = body
                        .iter()
                        .cloned()
                        .map(|s| subst_stmt(s, &subst))
                        .collect();
                    lower_stmts(substituted, ctx, out)?;
                }
            }
            Stmt::RepeatIndexedRuntime {
                var,
                start,
                end,
                body,
            } => {
                validate_loop_body(&body, "repeat")?;
                // The runtime index `i` stays symbolic (a real `for` var), so the
                // body is lowered ONCE — `cb[i]` resolves to an `AdoptIndex` keyed
                // on `i`, not a per-iteration literal.
                let mut body_buf: Vec<Stmt> = Vec::new();
                lower_stmts(body, ctx, &mut body_buf)?;
                out.push(Stmt::RepeatIndexedRuntime {
                    var,
                    start,
                    end,
                    body: body_buf,
                });
            }
            Stmt::Decl {
                kind,
                name,
                family: Some(count),
                ir_name,
                shape,
            } => {
                // Expand a family `w[N]` into `w_0 … w_{N-1}`, filling any `{i}`
                // in the IR-name template with the element index.
                for k in 0..count {
                    let ename = Ident::new(&format!("{name}_{k}"), name.span());
                    let eir = ir_name
                        .as_ref()
                        .map(|s| LitStr::new(&s.value().replace("{i}", &k.to_string()), s.span()));
                    out.push(Stmt::Decl {
                        kind: kind.clone(),
                        name: ename,
                        family: None,
                        ir_name: eir,
                        shape: shape.clone(),
                    });
                }
            }
            Stmt::Scan {
                carry,
                init,
                length,
                body,
            } => {
                // The carry init is outer code (its fn-inlines hoist to `out`).
                let init = lower_expr(init, ctx, out)?;
                // The body is lowered into its OWN buffer so any fn-inline hoists
                // stay inside the scan body, not the outer graph.
                let mut body_buf: Vec<Stmt> = Vec::new();
                for (n, e) in body {
                    let le = lower_expr(e, ctx, &mut body_buf)?;
                    body_buf.push(Stmt::Let { name: n, expr: le });
                }
                let new_body = body_buf
                    .into_iter()
                    .map(|s| match s {
                        Stmt::Let { name, expr } => (name, expr),
                        _ => unreachable!("scan body lowers to lets only"),
                    })
                    .collect();
                out.push(Stmt::Scan {
                    carry,
                    init,
                    length,
                    body: new_body,
                });
            }
            Stmt::Fn(f) => {
                // Nested fn defs aren't collected; reject them clearly.
                return Err(syn::Error::new(
                    f.name.span(),
                    "`fn` must be defined at graph top level, not nested",
                ));
            }
            other => out.push(other),
        }
    }
    Ok(())
}

/// A `repeat`/`repeat i in …` body may only re-bind `let`s (or nest loops) —
/// declarations/outputs inside would collide or escape each iteration.
fn validate_loop_body(body: &[Stmt], kw: &str) -> syn::Result<()> {
    for b in body {
        let bad = match b {
            Stmt::Let { .. }
            | Stmt::LetTuple { .. }
            | Stmt::Repeat { .. }
            | Stmt::RepeatIndexed { .. }
            | Stmt::RepeatIndexedRuntime { .. }
            | Stmt::RepeatRuntime { .. } => None,
            Stmt::Decl { name, .. } | Stmt::Const { name, .. } => Some((
                name.span(),
                format!(
                    "cannot declare an `input`/`param`/`const` inside `{kw}` \
                     (its name would collide each iteration) — declare it outside"
                ),
            )),
            Stmt::Bind { names } => Some((
                names[0].0.span(),
                format!(
                    "`bind` is not allowed inside `{kw}` — adopt outer tensors \
                     before the loop"
                ),
            )),
            Stmt::Out { names } | Stmt::Tap { names } => Some((
                names[0].span(),
                format!("`out`/`tap` is not allowed inside `{kw}` — put it after the loop"),
            )),
            Stmt::Fn(f) => Some((f.name.span(), format!("`fn` is not allowed inside `{kw}`"))),
            Stmt::Scan { carry, .. } => {
                Some((carry.span(), format!("`scan` is not allowed inside `{kw}`")))
            }
        };
        if let Some((span, msg)) = bad {
            return Err(syn::Error::new(span, msg));
        }
    }
    Ok(())
}

/// Substitute a `repeat i in …` loop variable (→ a literal) through a body
/// statement, so `w[i]` / expressions see the concrete index before lowering.
fn subst_stmt(s: Stmt, subst: &HashMap<String, Expr>) -> Stmt {
    let empty = HashMap::new();
    match s {
        Stmt::Let { name, expr } => Stmt::Let {
            name,
            expr: subst_expr(expr, subst, &empty),
        },
        Stmt::LetTuple { names, expr } => Stmt::LetTuple {
            names,
            expr: subst_expr(expr, subst, &empty),
        },
        Stmt::Repeat { count, body } => Stmt::Repeat {
            count,
            body: body.into_iter().map(|b| subst_stmt(b, subst)).collect(),
        },
        Stmt::RepeatIndexed {
            var,
            start,
            end,
            body,
        } => Stmt::RepeatIndexed {
            var,
            start,
            end,
            body: body.into_iter().map(|b| subst_stmt(b, subst)).collect(),
        },
        Stmt::RepeatRuntime { count, body } => Stmt::RepeatRuntime {
            count: subst_tokens(count, subst),
            body: body.into_iter().map(|b| subst_stmt(b, subst)).collect(),
        },
        Stmt::RepeatIndexedRuntime {
            var,
            start,
            end,
            body,
        } => Stmt::RepeatIndexedRuntime {
            var,
            start: subst_tokens(start, subst),
            end: subst_tokens(end, subst),
            body: body.into_iter().map(|b| subst_stmt(b, subst)).collect(),
        },
        Stmt::Scan {
            carry,
            init,
            length,
            body,
        } => Stmt::Scan {
            carry,
            init: subst_expr(init, subst, &empty),
            length,
            body: body
                .into_iter()
                .map(|(n, e)| (n, subst_expr(e, subst, &empty)))
                .collect(),
        },
        other => other,
    }
}

/// Substitute a `repeat i in …` loop variable (→ literal tokens) through a raw
/// token stream — used for a runtime `repeat <expr>` count that mentions an
/// enclosing indexed-loop variable (`repeat i in 0..2 { repeat i { … } }`).
fn subst_tokens(ts: TokenStream, subst: &HashMap<String, Expr>) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(id) => match subst.get(&id.to_string()) {
                Some(Expr::Num(t)) => out.extend(t.clone()),
                _ => out.extend(std::iter::once(TokenTree::Ident(id))),
            },
            TokenTree::Group(g) => {
                let inner = subst_tokens(g.stream(), subst);
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    g.delimiter(),
                    inner,
                ))));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

fn lower_expr(e: Expr, ctx: &mut Ctx, out: &mut Vec<Stmt>) -> syn::Result<Expr> {
    Ok(match e {
        Expr::Var(_) | Expr::Num(_) => e,
        Expr::Neg(x) => Expr::Neg(Box::new(lower_expr(*x, ctx, out)?)),
        Expr::Bin(k, a, b, s) => Expr::Bin(
            k,
            Box::new(lower_expr(*a, ctx, out)?),
            Box::new(lower_expr(*b, ctx, out)?),
            s,
        ),
        Expr::MatMul(a, b, s) => Expr::MatMul(
            Box::new(lower_expr(*a, ctx, out)?),
            Box::new(lower_expr(*b, ctx, out)?),
            s,
        ),
        Expr::Cmp(c, a, b, s) => Expr::Cmp(
            c,
            Box::new(lower_expr(*a, ctx, out)?),
            Box::new(lower_expr(*b, ctx, out)?),
            s,
        ),
        Expr::MethodDsl { recv, method, args } => {
            let recv = Box::new(lower_expr(*recv, ctx, out)?);
            let mut largs = Vec::with_capacity(args.len());
            for (a, m) in args {
                largs.push((lower_expr(a, ctx, out)?, m));
            }
            Expr::MethodDsl {
                recv,
                method,
                args: largs,
            }
        }
        Expr::Method { recv, name, args } => Expr::Method {
            recv: Box::new(lower_expr(*recv, ctx, out)?),
            name,
            args,
        },
        Expr::Linear { x, w, b, act, span } => Expr::Linear {
            x: Box::new(lower_expr(*x, ctx, out)?),
            w: Box::new(lower_expr(*w, ctx, out)?),
            b: Box::new(lower_expr(*b, ctx, out)?),
            act,
            span,
        },
        Expr::Index {
            base,
            index,
            fields,
            inner,
        } => {
            let idx = lower_expr(*index, ctx, out)?;
            if ctx.collection_binds.contains(&base.to_string()) {
                // `bind cb[]` collection element → adopt the indexed Tensor
                // (optionally a `.field` of a `Vec<Struct>` element, then an
                // optional `[s]` into that field's `Vec<Tensor>`).
                let inner = match inner {
                    Some(e) => {
                        let li = lower_expr(*e, ctx, out)?;
                        Some(index_tokens(&li)?)
                    }
                    None => None,
                };
                Expr::AdoptIndex {
                    base,
                    index: index_tokens(&idx)?,
                    fields,
                    inner,
                }
            } else if fields.is_empty() && inner.is_none() {
                resolve_index(&base, &idx)?
            } else {
                return Err(syn::Error::new(
                    base.span(),
                    format!(
                        "`{base}[…].{}…` field/element access is only valid on a \
                         `bind`-adopted collection (`bind {base}[];`), not a param family",
                        fields.first().map(|f| f.to_string()).unwrap_or_default()
                    ),
                ));
            }
        }
        Expr::AdoptIndex {
            base,
            index,
            fields,
            inner,
        } => Expr::AdoptIndex {
            base,
            index,
            fields,
            inner,
        },
        Expr::Call {
            name,
            args,
            arg_names,
        } => {
            let mut largs = Vec::with_capacity(args.len());
            for a in args {
                largs.push(lower_expr(a, ctx, out)?);
            }
            resolve_call(name, largs, arg_names, ctx, out)?
        }
    })
}

/// Render a collection index (`cb[index]`) as Rust index tokens: a literal (from
/// `repeat i in …`) or a bare `usize` binding/var. Rejects computed/tensor
/// indices, which don't index a `Vec<Tensor>`.
fn index_tokens(index: &Expr) -> syn::Result<TokenStream> {
    match index {
        Expr::Num(t) => Ok(t.clone()),
        Expr::Var(id) => Ok(quote!(#id)),
        _ => Err(syn::Error::new(
            Span::call_site(),
            "a `bind`-collection index must be a literal or the `repeat i in …` \
             index (it indexes an outer `Vec<Tensor>`)",
        )),
    }
}

/// Resolve `base[index]` to the family element `base_<k>` — the index must be a
/// constant literal by now (a `repeat i in …` variable is substituted first).
fn resolve_index(base: &Ident, index: &Expr) -> syn::Result<Expr> {
    if let Expr::Num(t) = index {
        if let Ok(k) = t.to_string().trim().parse::<usize>() {
            return Ok(Expr::Var(Ident::new(&format!("{base}_{k}"), base.span())));
        }
    }
    Err(syn::Error::new(
        base.span(),
        format!(
            "`{base}[…]` index must be a constant — a literal, or the loop \
             variable of an enclosing `repeat i in start..end`"
        ),
    ))
}

/// Map a DSL activation-sugar name to its `Activation` variant ident, so
/// `act(linear(..))` can fold the activation into the fused linear op.
fn activation_variant(name: &str) -> Option<&'static str> {
    Some(match name {
        "relu" => "Relu",
        "gelu" => "Gelu",
        "gelu_approx" => "GeluApprox",
        "silu" => "Silu",
        "tanh" => "Tanh",
        "sigmoid" => "Sigmoid",
        _ => return None,
    })
}

fn resolve_call(
    name: Ident,
    args: Vec<Expr>,
    arg_names: Vec<Option<Ident>>,
    ctx: &mut Ctx,
    out: &mut Vec<Stmt>,
) -> syn::Result<Expr> {
    let nm = name.to_string();

    // 1. A user-defined fn takes precedence — inline it.
    if let Some(f) = ctx.funcs.get(&nm) {
        if f.params.len() != args.len() {
            return Err(syn::Error::new(
                name.span(),
                format!(
                    "fn `{nm}` takes {} argument(s), but {} were given",
                    f.params.len(),
                    args.len()
                ),
            ));
        }
        if ctx.active.iter().any(|a| a == &nm) {
            return Err(syn::Error::new(
                name.span(),
                format!("recursive rlx! fn `{nm}` is not supported (fn bodies are inlined)"),
            ));
        }
        // Reorder named args (`blk(w: a, x: b)`) to the fn's parameter order.
        let args = reorder_named_args(f, args, arg_names)?;
        return inline_fn(f, args, ctx, out);
    }

    // Named args are only meaningful for `fn` calls.
    if let Some(n) = arg_names.iter().flatten().next() {
        return Err(syn::Error::new(
            n.span(),
            format!("named argument `{n}:` is only allowed on a user `fn` call, not `{nm}(…)`"),
        ));
    }

    // 2. Builtin multi-arg sugar.
    let n = args.len();
    match (nm.as_str(), n) {
        ("maximum" | "minimum" | "atan2" | "rem" | "pow", 2) => {
            let mut it = args.into_iter();
            let recv = it.next().unwrap();
            let rhs = it.next().unwrap();
            let mode = ArgMode::for_operand(&rhs);
            Ok(Expr::MethodDsl {
                recv: Box::new(recv),
                method: Ident::new(&nm, name.span()),
                args: vec![(rhs, mode)],
            })
        }
        ("clamp", 3) => {
            let mut it = args.into_iter();
            let recv = it.next().unwrap();
            let lo = it.next().unwrap();
            let hi = it.next().unwrap();
            for b in [&lo, &hi] {
                if !b.is_scalar() {
                    return Err(syn::Error::new(
                        name.span(),
                        "`clamp(x, lo, hi)` bounds must be scalar literals",
                    ));
                }
            }
            Ok(Expr::MethodDsl {
                recv: Box::new(recv),
                method: Ident::new("clamp", name.span()),
                args: vec![(lo, ArgMode::Scalar), (hi, ArgMode::Scalar)],
            })
        }
        ("select" | "where_", 3) => {
            let mut it = args.into_iter();
            let cond = it.next().unwrap();
            let t = it.next().unwrap();
            let f = it.next().unwrap();
            // A scalar branch (`select(mask, x, 0.0)`) is promoted by `where_`.
            let tm = ArgMode::for_operand(&t);
            let fm = ArgMode::for_operand(&f);
            Ok(Expr::MethodDsl {
                recv: Box::new(cond),
                method: Ident::new("where_", name.span()),
                args: vec![(t, tm), (f, fm)],
            })
        }
        ("matmul_t" | "mm_t", 2) => {
            let mut it = args.into_iter();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(a),
                method: Ident::new("matmul_t", name.span()),
                args: vec![(b, ArgMode::Ref)],
            })
        }
        // Fused softmax cross-entropy against a dense (soft / one-hot) target
        // distribution: `cross_entropy(logits, targets)` → per-row loss `[N]`.
        // Pair with `mean(…)` for a scalar training loss.
        ("cross_entropy", 2) => {
            let mut it = args.into_iter();
            let logits = it.next().unwrap();
            let targets = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(logits),
                method: Ident::new("cross_entropy", name.span()),
                args: vec![(targets, ArgMode::Ref)],
            })
        }
        // Fused softmax cross-entropy with integer class labels:
        // `softmax_cross_entropy(logits, labels)` → per-row loss `[N]`.
        ("softmax_cross_entropy", 2) => {
            let mut it = args.into_iter();
            let logits = it.next().unwrap();
            let labels = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(logits),
                method: Ident::new("softmax_cross_entropy_with_logits", name.span()),
                args: vec![(labels, ArgMode::Ref)],
            })
        }
        // HF linear `x·Wᵀ + b` → fused matmul+bias (activation folded later).
        ("linear", 3) => {
            let mut it = args.into_iter();
            let x = it.next().unwrap();
            let w = it.next().unwrap();
            let b = it.next().unwrap();
            Ok(Expr::Linear {
                x: Box::new(x),
                w: Box::new(w),
                b: Box::new(b),
                act: None,
                span: name.span(),
            })
        }
        // Embedding lookup: `embed(table, ids)` → `table.gather(ids, 0)`.
        ("embed", 2) => {
            let mut it = args.into_iter();
            let table = it.next().unwrap();
            let ids = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(table),
                method: Ident::new("gather", name.span()),
                args: vec![(ids, ArgMode::Ref), (Expr::Num(quote!(0)), ArgMode::Raw)],
            })
        }
        // Rotary embedding: `rope(x, cos, sin, head_dim)`.
        ("rope", 4) => {
            let mut it = args.into_iter();
            let x = it.next().unwrap();
            let cos = it.next().unwrap();
            let sin = it.next().unwrap();
            let hd = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(x),
                method: Ident::new("rope", name.span()),
                args: vec![(cos, ArgMode::Ref), (sin, ArgMode::Ref), (hd, ArgMode::Raw)],
            })
        }
        // `split(x, axis, n)` → `x.chunk(axis, n)` (a `Vec` — bind with a tuple
        // `let (a, b, …) = split(…)`).
        ("split", 3) => {
            let mut it = args.into_iter();
            let x = it.next().unwrap();
            let axis = it.next().unwrap();
            let n = it.next().unwrap();
            Ok(Expr::MethodDsl {
                recv: Box::new(x),
                method: Ident::new("chunk", name.span()),
                args: vec![(axis, ArgMode::Raw), (n, ArgMode::Raw)],
            })
        }
        // Reduce-to-scalar sugar: `mean(x)` / `sum(x)` collapse EVERY axis to a
        // scalar (rank-0), unlike the axis-taking `x.mean(axes, keep)` method.
        // Makes a loss reduction one word: `mean(cross_entropy(logits, tgt))`.
        ("mean", 1) => {
            let recv = args.into_iter().next().unwrap();
            Ok(Expr::Method {
                recv: Box::new(recv),
                name: Ident::new("mean_all", name.span()),
                args: Vec::new(),
            })
        }
        ("sum", 1) => {
            let recv = args.into_iter().next().unwrap();
            Ok(Expr::Method {
                recv: Box::new(recv),
                name: Ident::new("sum_all", name.span()),
                args: Vec::new(),
            })
        }
        // 3. Single-arg method sugar: `f(x)` → `x.f()`, with one peephole —
        // `act(linear(..))` folds the activation into the fused linear op.
        (_, 1) => {
            let recv = args.into_iter().next().unwrap();
            match (activation_variant(&nm), recv) {
                (
                    Some(variant),
                    Expr::Linear {
                        x,
                        w,
                        b,
                        act: None,
                        span,
                    },
                ) => Ok(Expr::Linear {
                    x,
                    w,
                    b,
                    act: Some(Ident::new(variant, name.span())),
                    span,
                }),
                (_, recv) => Ok(Expr::Method {
                    recv: Box::new(recv),
                    name,
                    args: Vec::new(),
                }),
            }
        }
        (_, n) => Err(syn::Error::new(
            name.span(),
            format!(
                "`{nm}(…)` with {n} argument(s) has no builtin sugar and no matching `fn` — \
                 call it as a method `x.{nm}(…)`, or define `fn {nm}(…)`"
            ),
        )),
    }
}

/// Reorder `blk(w: a, x: b)` named call arguments to the fn's parameter order.
/// All-positional passes through unchanged; mixing positional and named, an
/// unknown/duplicate/missing name, are errors.
fn reorder_named_args(
    f: &FnDef,
    args: Vec<Expr>,
    arg_names: Vec<Option<Ident>>,
) -> syn::Result<Vec<Expr>> {
    if arg_names.iter().all(Option::is_none) {
        return Ok(args);
    }
    if let Some(unnamed_after) = arg_names.iter().flatten().next() {
        if arg_names.iter().any(Option::is_none) {
            return Err(syn::Error::new(
                unnamed_after.span(),
                "a `fn` call mixes positional and named arguments — name all or none",
            ));
        }
    }
    let param_set: HashSet<String> = f.params.iter().map(|p| p.to_string()).collect();
    let mut by_name: HashMap<String, Expr> = HashMap::new();
    for (nm, a) in arg_names.into_iter().zip(args.into_iter()) {
        let nm = nm.unwrap();
        if !param_set.contains(&nm.to_string()) {
            return Err(syn::Error::new(
                nm.span(),
                format!("fn `{}` has no parameter `{nm}`", f.name),
            ));
        }
        if by_name.insert(nm.to_string(), a).is_some() {
            return Err(syn::Error::new(
                nm.span(),
                format!("argument `{nm}` given twice"),
            ));
        }
    }
    f.params
        .iter()
        .map(|p| {
            by_name.remove(&p.to_string()).ok_or_else(|| {
                syn::Error::new(
                    p.span(),
                    format!("missing argument `{p}` in call to fn `{}`", f.name),
                )
            })
        })
        .collect()
}

/// Inline `f(args)`: emit its body `let`s (renamed to fresh names, with params
/// substituted and locals renamed) into `out`, and return a reference to the
/// last one.
fn inline_fn(f: &FnDef, args: Vec<Expr>, ctx: &mut Ctx, out: &mut Vec<Stmt>) -> syn::Result<Expr> {
    ctx.active.push(f.name.to_string());

    // `subst` rewrites DSL Var positions (param → arg expr, local → fresh Var).
    // `rename` rewrites bare-ident references inside escape-hatch raw args
    // (param → arg's ident when it's a simple `Var`, local → fresh ident).
    let mut subst: HashMap<String, Expr> = HashMap::new();
    let mut rename: HashMap<String, Ident> = HashMap::new();
    for (p, a) in f.params.iter().zip(args.into_iter()) {
        if let Expr::Var(id) = &a {
            rename.insert(p.to_string(), id.clone());
        }
        subst.insert(p.to_string(), a);
    }

    let mut last: Option<Ident> = None;
    for (lname, lexpr) in &f.body {
        let substituted = subst_expr(lexpr.clone(), &subst, &rename);
        let lowered = lower_expr(substituted, ctx, out)?;
        let id = ctx.counter;
        ctx.counter += 1;
        let fresh = Ident::new(&format!("__rlx_{lname}_{id}"), lname.span());
        out.push(Stmt::Let {
            name: fresh.clone(),
            expr: lowered,
        });
        subst.insert(lname.to_string(), Expr::Var(fresh.clone()));
        rename.insert(lname.to_string(), fresh.clone());
        last = Some(fresh);
    }

    ctx.active.pop();
    Ok(Expr::Var(
        last.expect("fn body is non-empty (checked at parse)"),
    ))
}

fn subst_expr(e: Expr, subst: &HashMap<String, Expr>, rename: &HashMap<String, Ident>) -> Expr {
    match e {
        Expr::Var(id) => subst.get(&id.to_string()).cloned().unwrap_or(Expr::Var(id)),
        Expr::Num(t) => Expr::Num(t),
        Expr::Neg(x) => Expr::Neg(Box::new(subst_expr(*x, subst, rename))),
        Expr::Bin(k, a, b, s) => Expr::Bin(
            k,
            Box::new(subst_expr(*a, subst, rename)),
            Box::new(subst_expr(*b, subst, rename)),
            s,
        ),
        Expr::MatMul(a, b, s) => Expr::MatMul(
            Box::new(subst_expr(*a, subst, rename)),
            Box::new(subst_expr(*b, subst, rename)),
            s,
        ),
        Expr::Cmp(c, a, b, s) => Expr::Cmp(
            c,
            Box::new(subst_expr(*a, subst, rename)),
            Box::new(subst_expr(*b, subst, rename)),
            s,
        ),
        Expr::Call {
            name,
            args,
            arg_names,
        } => Expr::Call {
            name,
            args: args
                .into_iter()
                .map(|a| subst_expr(a, subst, rename))
                .collect(),
            arg_names,
        },
        Expr::MethodDsl { recv, method, args } => Expr::MethodDsl {
            recv: Box::new(subst_expr(*recv, subst, rename)),
            method,
            args: args
                .into_iter()
                .map(|(a, m)| (subst_expr(a, subst, rename), m))
                .collect(),
        },
        Expr::Method { recv, name, args } => Expr::Method {
            recv: Box::new(subst_expr(*recv, subst, rename)),
            name,
            // Raw escape-hatch args see BOTH the fn-inline rename (`k`→fresh) and
            // the `repeat i in …` index substitution (`i`→literal), so an indexed
            // collection arg `cb[i]` resolves to `cb[0]`, `cb[1]`, ….
            args: args
                .into_iter()
                .map(|a| subst_rename_syn_expr(a, subst, rename))
                .collect(),
        },
        Expr::Linear { x, w, b, act, span } => Expr::Linear {
            x: Box::new(subst_expr(*x, subst, rename)),
            w: Box::new(subst_expr(*w, subst, rename)),
            b: Box::new(subst_expr(*b, subst, rename)),
            act,
            span,
        },
        Expr::Index {
            base,
            index,
            fields,
            inner,
        } => Expr::Index {
            base,
            index: Box::new(subst_expr(*index, subst, rename)),
            fields,
            inner: inner.map(|e| Box::new(subst_expr(*e, subst, rename))),
        },
        Expr::AdoptIndex {
            base,
            index,
            fields,
            inner,
        } => Expr::AdoptIndex {
            base,
            index: subst_rename_tokens(index, subst, rename),
            fields,
            inner: inner.map(|t| subst_rename_tokens(t, subst, rename)),
        },
    }
}

/// Substitute inside a raw (escape-hatch) argument: apply the fn-inline `rename`
/// (bare `k`→fresh ident) AND the `repeat i in …` index map (`i`→literal), so a
/// fn body's `q.attention(k, …)` still resolves after locals are renamed and an
/// indexed collection arg `cb[i]` picks up the concrete loop index.
fn subst_rename_syn_expr(
    a: syn::Expr,
    subst: &HashMap<String, Expr>,
    rename: &HashMap<String, Ident>,
) -> syn::Expr {
    if subst.is_empty() && rename.is_empty() {
        return a;
    }
    let ts = subst_rename_tokens(a.to_token_stream(), subst, rename);
    syn::parse2::<syn::Expr>(ts).unwrap_or(a)
}

/// Token rewrite for raw args: an ident is replaced by its `rename` target, else
/// (if `subst` maps it to a `Num`/`Var`) by that literal/ident, else left as-is.
fn subst_rename_tokens(
    ts: TokenStream,
    subst: &HashMap<String, Expr>,
    rename: &HashMap<String, Ident>,
) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in ts {
        match tt {
            TokenTree::Ident(id) => {
                let key = id.to_string();
                if let Some(r) = rename.get(&key) {
                    out.extend(std::iter::once(TokenTree::Ident(r.clone())));
                } else {
                    match subst.get(&key) {
                        Some(Expr::Num(t)) => out.extend(t.clone()),
                        Some(Expr::Var(v)) => {
                            out.extend(std::iter::once(TokenTree::Ident(v.clone())))
                        }
                        _ => out.extend(std::iter::once(TokenTree::Ident(id))),
                    }
                }
            }
            TokenTree::Group(g) => {
                let inner = subst_rename_tokens(g.stream(), subst, rename);
                out.extend(std::iter::once(TokenTree::Group(Group::new(
                    g.delimiter(),
                    inner,
                ))));
            }
            other => out.extend(std::iter::once(other)),
        }
    }
    out
}

/// Collect the outer bindings a `scan` body references — every identifier that
/// is neither the carry nor a body-local `let` name — in first-seen order.
/// These become the scan's broadcast inputs (held constant across iterations).
/// Unknown/typo names are already rejected by `check` before this runs.
fn collect_free_vars(body: &[(Ident, Expr)], carry: &str) -> Vec<Ident> {
    let mut local: HashSet<String> = HashSet::new();
    local.insert(carry.to_string());
    let mut order: Vec<Ident> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (name, expr) in body {
        collect_expr_free(expr, &local, &mut order, &mut seen);
        local.insert(name.to_string());
    }
    order
}

fn collect_expr_free(
    e: &Expr,
    local: &HashSet<String>,
    order: &mut Vec<Ident>,
    seen: &mut HashSet<String>,
) {
    match e {
        Expr::Var(id) => push_free(id, local, order, seen),
        Expr::Num(_) => {}
        Expr::Neg(x) => collect_expr_free(x, local, order, seen),
        Expr::Bin(_, a, b, _) | Expr::MatMul(a, b, _) | Expr::Cmp(_, a, b, _) => {
            collect_expr_free(a, local, order, seen);
            collect_expr_free(b, local, order, seen);
        }
        Expr::Call { args, .. } => {
            for a in args {
                collect_expr_free(a, local, order, seen);
            }
        }
        Expr::MethodDsl { recv, args, .. } => {
            collect_expr_free(recv, local, order, seen);
            for (a, _) in args {
                collect_expr_free(a, local, order, seen);
            }
        }
        Expr::Linear { x, w, b, .. } => {
            collect_expr_free(x, local, order, seen);
            collect_expr_free(w, local, order, seen);
            collect_expr_free(b, local, order, seen);
        }
        Expr::Index { base, index, .. } => {
            push_free(base, local, order, seen);
            collect_expr_free(index, local, order, seen);
        }
        // Collection base is an outer Rust `Vec<Tensor>`, not a scan broadcast.
        Expr::AdoptIndex { .. } => {}
        Expr::Method { recv, args, .. } => {
            collect_expr_free(recv, local, order, seen);
            // Only top-level bare-ident raw args are binding references (same
            // rule as auto-borrow); everything else is opaque raw Rust.
            for a in args {
                if let syn::Expr::Path(p) = a {
                    if p.qself.is_none()
                        && p.path.leading_colon.is_none()
                        && p.path.segments.len() == 1
                        && p.path.segments[0].arguments.is_none()
                    {
                        push_free(&p.path.segments[0].ident, local, order, seen);
                    }
                }
            }
        }
    }
}

fn push_free(
    id: &Ident,
    local: &HashSet<String>,
    order: &mut Vec<Ident>,
    seen: &mut HashSet<String>,
) {
    let s = id.to_string();
    if !local.contains(&s) && seen.insert(s) {
        order.push(id.clone());
    }
}

// ── Semantic checks ──────────────────────────────────────────────────────

impl GraphDsl {
    /// Resolve bindings and reject the mistakes that would otherwise surface as
    /// cryptic type errors in generated code: unknown names, matmul on a
    /// scalar, `let`s that don't produce a tensor, and (best-effort) matmul
    /// shape mismatches.
    fn check(&self) -> syn::Result<()> {
        // Every declared binding (for out-of-order `out` resolution).
        let mut declared: HashSet<String> = HashSet::new();
        for s in &self.stmts {
            match s {
                Stmt::Decl { name, .. } | Stmt::Const { name, .. } | Stmt::Let { name, .. } => {
                    declared.insert(name.to_string());
                }
                Stmt::LetTuple { names, .. } => {
                    for n in names {
                        declared.insert(n.to_string());
                    }
                }
                Stmt::Scan { carry, .. } => {
                    declared.insert(carry.to_string());
                }
                Stmt::Bind { names } => {
                    for (n, _) in names {
                        declared.insert(n.to_string());
                    }
                }
                _ => {}
            }
        }

        // Names visible *so far* (a `let` may only reference earlier bindings).
        let mut in_scope: HashSet<String> = HashSet::new();
        let mut has_output = false;

        for s in &self.stmts {
            match s {
                Stmt::Decl { name, .. } | Stmt::Const { name, .. } => {
                    in_scope.insert(name.to_string());
                }
                Stmt::Let { name, expr } => {
                    check_expr(expr, &in_scope)?;
                    if expr.is_scalar() {
                        return Err(syn::Error::new(
                            name.span(),
                            format!(
                                "`let {name}` binds a scalar expression, but a `let` must \
                                 produce a tensor — declare it with `const {name} = … : F32;` \
                                 or combine it with a tensor"
                            ),
                        ));
                    }
                    in_scope.insert(name.to_string());
                }
                Stmt::LetTuple { names, expr } => {
                    check_expr(expr, &in_scope)?;
                    for n in names {
                        in_scope.insert(n.to_string());
                    }
                }
                Stmt::Out { names } => {
                    has_output = true;
                    for n in names {
                        if !declared.contains(&n.to_string()) {
                            return Err(syn::Error::new(
                                n.span(),
                                format!("unknown output binding `{n}`"),
                            ));
                        }
                    }
                }
                Stmt::Scan {
                    carry, init, body, ..
                } => {
                    check_expr(init, &in_scope)?;
                    // The body sees the carry plus everything in scope; its own
                    // `let`s accumulate as it goes.
                    let mut body_scope = in_scope.clone();
                    body_scope.insert(carry.to_string());
                    for (n, e) in body {
                        check_expr(e, &body_scope)?;
                        body_scope.insert(n.to_string());
                    }
                    if let Some((n, last)) = body.last() {
                        if last.is_scalar() {
                            return Err(syn::Error::new(
                                n.span(),
                                "the last `let` of a `scan` body must produce a tensor (the \
                                 next carry), not a scalar",
                            ));
                        }
                    }
                    in_scope.insert(carry.to_string());
                }
                Stmt::Tap { names } => {
                    for n in names {
                        if !declared.contains(&n.to_string()) {
                            return Err(syn::Error::new(
                                n.span(),
                                format!("unknown tap binding `{n}`"),
                            ));
                        }
                    }
                }
                Stmt::Bind { names } => {
                    // Adopts outer Rust `Tensor`(s) / collections of the same name
                    // — visible from here on, like an `input`/`param`.
                    for (n, _) in names {
                        in_scope.insert(n.to_string());
                    }
                }
                Stmt::RepeatRuntime { body, .. } => {
                    // The body is a scoped block: its `let`s see everything in
                    // scope plus each other; new body-locals do NOT escape (only
                    // rebound outer bindings are threaded, already in scope).
                    let mut body_scope = in_scope.clone();
                    check_block_scope(body, &mut body_scope)?;
                }
                Stmt::RepeatIndexedRuntime { body, .. } => {
                    // Same scoping as `RepeatRuntime`; the index var is a `usize`
                    // (only used in `cb[i]`), never a tensor binding, so it isn't
                    // added to the tensor scope.
                    let mut body_scope = in_scope.clone();
                    check_block_scope(body, &mut body_scope)?;
                }
                Stmt::Fn(_) | Stmt::Repeat { .. } | Stmt::RepeatIndexed { .. } => {
                    // Lowered away before `check`.
                }
            }
        }

        // A graph needs at least one output: an explicit `out`, or a `let` /
        // `scan` to fall back on.
        if !has_output
            && !self.stmts.iter().any(|s| {
                matches!(
                    s,
                    Stmt::Let { .. }
                        | Stmt::Scan { .. }
                        | Stmt::RepeatRuntime { .. }
                        | Stmt::RepeatIndexedRuntime { .. }
                )
            })
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "rlx! graph has no output — add `out <name>;` or at least one `let`",
            ));
        }

        // Best-effort static matmul shape check.
        self.check_shapes()?;
        Ok(())
    }

    /// A conservative pass that flags a matmul whose operands are both known to
    /// be static 2-D with incompatible inner dimensions. It infers only through
    /// bindings and matmuls (everything else is "unknown" and skipped), so it
    /// never produces a false positive — it just catches the common
    /// `input @ weight` and matmul-chain mismatches at expansion time.
    fn check_shapes(&self) -> syn::Result<()> {
        let mut env: HashMap<String, Option<Vec<Option<u64>>>> = HashMap::new();
        for s in &self.stmts {
            match s {
                Stmt::Decl { name, shape, .. } => {
                    env.insert(name.to_string(), parse_shape(shape));
                }
                Stmt::Const { name, value } => {
                    let sh = match value {
                        ConstVal::Array { dims, .. } => {
                            Some(dims.iter().map(|d| Some(*d as u64)).collect())
                        }
                        ConstVal::Scalar { .. } => Some(vec![]),
                    };
                    env.insert(name.to_string(), sh);
                }
                Stmt::Let { name, expr } => {
                    let sh = shape_check(expr, &env)?;
                    env.insert(name.to_string(), sh);
                }
                Stmt::Scan {
                    carry, init, body, ..
                } => {
                    // The carry keeps the init's shape across iterations; check
                    // the body's matmuls with the carry bound in a local env.
                    let carry_sh = shape_check(init, &env)?;
                    let mut body_env = env.clone();
                    body_env.insert(carry.to_string(), carry_sh.clone());
                    for (n, e) in body {
                        let sh = shape_check(e, &body_env)?;
                        body_env.insert(n.to_string(), sh);
                    }
                    env.insert(carry.to_string(), carry_sh);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Scope-check the body of a runtime `repeat` (post-lowering, so only `let` /
/// `let (…)` / nested `repeat <expr>` remain): each statement sees `scope` plus
/// the bindings before it; a nested loop gets its own cloned scope.
fn check_block_scope(body: &[Stmt], scope: &mut HashSet<String>) -> syn::Result<()> {
    for s in body {
        match s {
            Stmt::Let { name, expr } => {
                check_expr(expr, scope)?;
                if expr.is_scalar() {
                    return Err(syn::Error::new(
                        name.span(),
                        format!(
                            "`let {name}` inside `repeat` binds a scalar, but a `let` must \
                             produce a tensor"
                        ),
                    ));
                }
                scope.insert(name.to_string());
            }
            Stmt::LetTuple { names, expr } => {
                check_expr(expr, scope)?;
                for n in names {
                    scope.insert(n.to_string());
                }
            }
            Stmt::RepeatRuntime { body, .. } | Stmt::RepeatIndexedRuntime { body, .. } => {
                let mut inner = scope.clone();
                check_block_scope(body, &mut inner)?;
            }
            // Nothing else survives lowering + `validate_loop_body`.
            _ => {}
        }
    }
    Ok(())
}

fn check_expr(e: &Expr, scope: &HashSet<String>) -> syn::Result<()> {
    match e {
        Expr::Var(id) => {
            if scope.contains(&id.to_string()) {
                Ok(())
            } else {
                Err(syn::Error::new(
                    id.span(),
                    format!("unknown binding `{id}` — declare it with `input`/`param`/`let` first"),
                ))
            }
        }
        Expr::Num(_) => Ok(()),
        Expr::Neg(x) => check_expr(x, scope),
        Expr::Bin(_, a, b, _) => {
            check_expr(a, scope)?;
            check_expr(b, scope)
        }
        Expr::Cmp(_, a, b, span) => {
            check_expr(a, scope)?;
            check_expr(b, scope)?;
            if a.is_scalar() && b.is_scalar() {
                return Err(syn::Error::new(
                    *span,
                    "comparing two scalars does not produce a tensor",
                ));
            }
            Ok(())
        }
        Expr::MatMul(a, b, span) => {
            check_expr(a, scope)?;
            check_expr(b, scope)?;
            if a.is_scalar() || b.is_scalar() {
                return Err(syn::Error::new(
                    *span,
                    "matmul `@` requires tensor operands, not a scalar",
                ));
            }
            Ok(())
        }
        Expr::MethodDsl { recv, args, .. } => {
            check_expr(recv, scope)?;
            for (a, _) in args {
                check_expr(a, scope)?;
            }
            Ok(())
        }
        Expr::Method { recv, args, .. } => {
            check_expr(recv, scope)?;
            for a in args {
                check_method_arg(a, scope)?;
            }
            Ok(())
        }
        Expr::Linear { x, w, b, .. } => {
            check_expr(x, scope)?;
            check_expr(w, scope)?;
            check_expr(b, scope)
        }
        Expr::AdoptIndex { base, .. } => {
            if scope.contains(&base.to_string()) {
                Ok(())
            } else {
                Err(syn::Error::new(
                    base.span(),
                    format!(
                        "unknown `bind`-collection `{base}` — declare it with `bind {base}[];`"
                    ),
                ))
            }
        }
        Expr::Index { base, .. } => Err(syn::Error::new(
            base.span(),
            "internal error: unresolved family index survived lowering",
        )),
        Expr::Call { name, .. } => Err(syn::Error::new(
            name.span(),
            "internal error: unresolved call survived lowering",
        )),
    }
}

/// A **bare identifier** in an escape-hatch method arg is a binding reference
/// (auto-borrowed at codegen), so a bare name that resolves to no binding is
/// almost always a typo — reject it with a hint. Everything else — a literal,
/// an enum path (`MaskKind::Causal`), `&x`, a call, a closure, or a
/// parenthesised group — is raw Rust and left to the compiler, which is also
/// the documented escape: wrap an external value as `(value)` — or prefix it
/// with `~value` — to pass a scalar by value and opt out of the check.
fn check_method_arg(arg: &syn::Expr, scope: &HashSet<String>) -> syn::Result<()> {
    if let syn::Expr::Path(p) = arg {
        if p.qself.is_none() && p.path.leading_colon.is_none() && p.path.segments.len() == 1 {
            let seg = &p.path.segments[0];
            if seg.arguments.is_none() && !scope.contains(&seg.ident.to_string()) {
                let id = &seg.ident;
                return Err(syn::Error::new(
                    id.span(),
                    format!(
                        "unknown binding `{id}` in method argument — declare it first, or \
                         wrap it as `({id})` to pass an external value through raw"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Parse a `[dims]` token stream into a best-effort static shape: `Some(n)` for
/// an integer-literal axis, `None` for a dynamic / expression axis. Returns
/// `None` if the shape can't be parsed at all (treated as "unknown").
fn parse_shape(ts: &TokenStream) -> Option<Vec<Option<u64>>> {
    let parser = |input: ParseStream| -> syn::Result<Vec<Option<u64>>> {
        // Optional `DType;` prefix.
        if input.peek(Ident) && input.peek2(Token![;]) {
            input.parse::<Ident>()?;
            input.parse::<Token![;]>()?;
        }
        let mut dims = Vec::new();
        while !input.is_empty() {
            if input.peek(Token![?]) {
                input.parse::<Token![?]>()?;
                if input.peek(syn::LitInt) {
                    input.parse::<syn::LitInt>()?;
                }
                dims.push(None);
            } else {
                let e: syn::Expr = input.parse()?;
                let v = match &e {
                    syn::Expr::Lit(l) => match &l.lit {
                        syn::Lit::Int(i) => i.base10_parse::<u64>().ok(),
                        _ => None,
                    },
                    _ => None,
                };
                dims.push(v);
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        Ok(dims)
    };
    syn::parse::Parser::parse2(parser, ts.clone()).ok()
}

/// Recursively infer an expression's static shape and error on a definite
/// matmul mismatch. Returns `None` when the shape isn't statically known.
fn shape_check(
    e: &Expr,
    env: &HashMap<String, Option<Vec<Option<u64>>>>,
) -> syn::Result<Option<Vec<Option<u64>>>> {
    match e {
        Expr::Var(id) => Ok(env.get(&id.to_string()).cloned().flatten()),
        Expr::Num(_) => Ok(Some(vec![])),
        Expr::Neg(x) => shape_check(x, env),
        Expr::MatMul(a, b, span) => {
            let sa = shape_check(a, env)?;
            let sb = shape_check(b, env)?;
            if let (Some(sa), Some(sb)) = (&sa, &sb) {
                if sa.len() == 2 && sb.len() == 2 {
                    if let (Some(k1), Some(k2)) = (sa[1], sb[0]) {
                        if k1 != k2 {
                            return Err(syn::Error::new(
                                *span,
                                format!(
                                    "matmul shape mismatch: inner dimensions differ ({k1} ≠ {k2}) \
                                     — left is `[…, {k1}]`, right is `[{k2}, …]`"
                                ),
                            ));
                        }
                    }
                    return Ok(Some(vec![sa[0], sb[1]]));
                }
            }
            Ok(None)
        }
        Expr::Bin(_, a, b, _) => {
            shape_check(a, env)?;
            shape_check(b, env)?;
            Ok(None)
        }
        Expr::Cmp(_, a, b, _) => {
            shape_check(a, env)?;
            shape_check(b, env)?;
            Ok(None)
        }
        Expr::MethodDsl { recv, args, .. } => {
            shape_check(recv, env)?;
            for (a, _) in args {
                shape_check(a, env)?;
            }
            Ok(None)
        }
        Expr::Method { recv, .. } => {
            shape_check(recv, env)?;
            Ok(None)
        }
        Expr::Linear { x, w, b, span, .. } => {
            let sx = shape_check(x, env)?;
            let sw = shape_check(w, env)?;
            shape_check(b, env)?;
            // `linear(x, w, b)` = `x·Wᵀ+b`: x is `[.., in]`, W is `[out, in]`,
            // so contract on `in` and the result's last dim becomes `out`.
            if let (Some(sx), Some(sw)) = (&sx, &sw) {
                if sw.len() == 2 && !sx.is_empty() {
                    if let (Some(x_in), Some(w_in)) = (sx[sx.len() - 1], sw[1]) {
                        if x_in != w_in {
                            return Err(syn::Error::new(
                                *span,
                                format!(
                                    "linear shape mismatch: input's last dim {x_in} ≠ the \
                                     weight's in-dim {w_in} (a `linear` weight is `[out, in]`)"
                                ),
                            ));
                        }
                    }
                    let mut out = sx.clone();
                    *out.last_mut().unwrap() = sw[0];
                    return Ok(Some(out));
                }
            }
            Ok(None)
        }
        Expr::Index { .. } | Expr::Call { .. } | Expr::AdoptIndex { .. } => Ok(None),
    }
}

// ── Code generation ──────────────────────────────────────────────────────

impl GraphDsl {
    fn codegen(&self) -> TokenStream {
        let scope = Ident::new("__rlx_scope", Span::mixed_site());
        let name = match &self.name {
            Some(s) => quote!(#s),
            None => quote!("rlx_graph"),
        };

        // Declared tensor bindings — used to auto-borrow bare-ident args.
        let mut vars: HashSet<String> = HashSet::new();
        for s in &self.stmts {
            match s {
                Stmt::Decl { name, .. } | Stmt::Const { name, .. } | Stmt::Let { name, .. } => {
                    vars.insert(name.to_string());
                }
                Stmt::LetTuple { names, .. } => {
                    for n in names {
                        vars.insert(n.to_string());
                    }
                }
                Stmt::Scan { carry, .. } => {
                    vars.insert(carry.to_string());
                }
                Stmt::Bind { names } => {
                    // Scalar binds auto-borrow as bare idents; collection binds
                    // auto-borrow their indexed element `cb[i]` — both keyed on
                    // membership in `vars`.
                    for (n, _) in names {
                        vars.insert(n.to_string());
                    }
                }
                _ => {}
            }
        }

        let mut body = Vec::new();
        let mut outs: Vec<Ident> = Vec::new();
        let mut taps: Vec<Ident> = Vec::new();
        let mut last_let: Option<Ident> = None;
        // Fresh-name counter for runtime-`repeat` loop-carry cells.
        let mut repeat_uid: u32 = 0;

        for s in &self.stmts {
            match s {
                Stmt::Decl {
                    kind,
                    name,
                    ir_name,
                    shape,
                    ..
                } => {
                    // The IR / `set_param` name is the `@"…"` override if given,
                    // else the binding ident.
                    let sname = match ir_name {
                        Some(s) => s.clone(),
                        None => LitStr::new(&name.to_string(), name.span()),
                    };
                    let ctor = match kind {
                        DeclKind::Input => quote!(input),
                        DeclKind::Param => quote!(param),
                    };
                    body.push(quote! {
                        let #name = #scope.#ctor(#sname, shape!(#shape));
                    });
                }
                Stmt::Const { name, value } => match value {
                    ConstVal::Scalar { neg, value, dtype } => {
                        let sign = if *neg { quote!(-) } else { quote!() };
                        body.push(quote! {
                            let #name = #scope.constant((#sign #value) as f64, DType::#dtype);
                        });
                    }
                    ConstVal::Array { data, dims, dtype } => {
                        let data = data.iter().map(|v| quote!(#v));
                        let dims = dims.iter().map(|d| quote!(#d));
                        body.push(quote! {
                            let #name = #scope.constant_nd(
                                vec![ #(#data),* ], vec![ #(#dims),* ], DType::#dtype);
                        });
                    }
                },
                Stmt::Let { name, expr } => {
                    let e = expr.emit(&vars);
                    body.push(quote! { let #name = #e; });
                    last_let = Some(name.clone());
                }
                Stmt::LetTuple { names, expr } => {
                    // Bind each element of a `Vec`-producing expression.
                    let e = expr.emit(&vars);
                    let tmp = Ident::new("__rlx_tuple", Span::mixed_site());
                    body.push(quote! { let #tmp = #e; });
                    for (i, n) in names.iter().enumerate() {
                        body.push(quote! { let #n = #tmp[#i].clone(); });
                    }
                    last_let = names.last().cloned();
                }
                Stmt::Scan {
                    carry,
                    init,
                    length,
                    body: steps,
                } => {
                    let init_e = init.emit(&vars);
                    let len = *length as u32;
                    // Outer bindings referenced in the body become scan
                    // broadcasts (held constant across iterations).
                    let free = collect_free_vars(steps, &carry.to_string());
                    let bcast_refs = free.iter().map(|f| quote!(&#f));
                    // Emit the body with the carry, broadcasts, and body locals
                    // in scope for auto-borrow.
                    let mut body_vars = vars.clone();
                    body_vars.insert(carry.to_string());
                    for f in &free {
                        body_vars.insert(f.to_string());
                    }
                    for (n, _) in steps {
                        body_vars.insert(n.to_string());
                    }
                    let rebinds = free.iter().enumerate().map(|(i, f)| {
                        quote! { let #f = __rlx_bcasts[#i].clone(); }
                    });
                    let step_lets = steps.iter().map(|(n, e)| {
                        let ee = e.emit(&body_vars);
                        quote! { let #n = #ee; }
                    });
                    let next = &steps.last().expect("scan body non-empty").0;
                    body.push(quote! {
                        let #carry = #scope.scan_block(
                            &(#init_e),
                            &[ #(#bcast_refs),* ],
                            #len,
                            |__rlx_carry, __rlx_bcasts| {
                                let #carry = __rlx_carry.clone();
                                #(#rebinds)*
                                #(#step_lets)*
                                #next.clone()
                            },
                        );
                    });
                    last_let = Some(carry.clone());
                }
                Stmt::RepeatRuntime { count, body: rbody } => {
                    // Emit a Rust `for` loop; rebound outer bindings are threaded
                    // as loop-carried values and re-exposed after the loop.
                    let (ts, last_carry) =
                        emit_repeat_runtime(count, rbody, &vars, &mut repeat_uid);
                    body.push(ts);
                    if let Some(v) = last_carry {
                        last_let = Some(v);
                    }
                }
                Stmt::RepeatIndexedRuntime {
                    var,
                    start,
                    end,
                    body: rbody,
                } => {
                    let (ts, last_carry) =
                        emit_repeat_indexed_runtime(var, start, end, rbody, &vars, &mut repeat_uid);
                    body.push(ts);
                    if let Some(v) = last_carry {
                        last_let = Some(v);
                    }
                }
                // `bind` adopts an outer Rust `Tensor` var of the same name — it
                // already exists in scope, so nothing is emitted.
                Stmt::Bind { .. } => {}
                Stmt::Out { names } => outs.extend(names.iter().cloned()),
                Stmt::Tap { names } => taps.extend(names.iter().cloned()),
                Stmt::Fn(_) | Stmt::Repeat { .. } | Stmt::RepeatIndexed { .. } => {}
            }
        }

        if outs.is_empty() {
            if let Some(l) = last_let {
                outs.push(l);
            }
        }
        // `tap`ped intermediates are appended as extra outputs (after the main
        // output), for debugging.
        outs.extend(taps);

        let out_ids = outs.iter().map(|o| quote!(#o.id()));

        // Note: `unused_variables` is *not* silenced — a declared-but-unused
        // input/param is dead weight and worth a warning.
        quote! {{
            #[allow(unused_mut, clippy::let_and_return)]
            let mut #scope = GraphScope::new(#name);
            #(#body)*
            #scope.set_outputs([ #(#out_ids),* ]);
            #scope.finish()
        }}
    }
}

/// The `let`-bound idents introduced directly by a loop body (this nesting
/// level only — nested `repeat`s manage their own carries). Used both to detect
/// loop-carried bindings and to extend the auto-borrow var set inside the body.
fn collect_level_let_idents(body: &[Stmt]) -> Vec<Ident> {
    let mut out = Vec::new();
    for s in body {
        match s {
            Stmt::Let { name, .. } => out.push(name.clone()),
            Stmt::LetTuple { names, .. } => out.extend(names.iter().cloned()),
            _ => {}
        }
    }
    out
}

/// Emit the body statements of a runtime `repeat` (post-lowering: only `let` /
/// `let (…)` / nested runtime `repeat` remain) as a `Vec` of Rust statements,
/// with `vars` driving bare-ident auto-borrow.
fn emit_stmts_block(body: &[Stmt], vars: &HashSet<String>, uid: &mut u32) -> Vec<TokenStream> {
    let mut out = Vec::new();
    for s in body {
        match s {
            Stmt::Let { name, expr } => {
                let e = expr.emit(vars);
                out.push(quote! { let #name = #e; });
            }
            Stmt::LetTuple { names, expr } => {
                let e = expr.emit(vars);
                let tmp = Ident::new("__rlx_tuple", Span::mixed_site());
                out.push(quote! { let #tmp = #e; });
                for (i, n) in names.iter().enumerate() {
                    out.push(quote! { let #n = #tmp[#i].clone(); });
                }
            }
            Stmt::RepeatRuntime { count, body } => {
                let (ts, _) = emit_repeat_runtime(count, body, vars, uid);
                out.push(ts);
            }
            Stmt::RepeatIndexedRuntime {
                var,
                start,
                end,
                body,
            } => {
                let (ts, _) = emit_repeat_indexed_runtime(var, start, end, body, vars, uid);
                out.push(ts);
            }
            _ => {}
        }
    }
    out
}

/// Emit a runtime `repeat <count> { … }` as a Rust `for _ in 0..(<count>)` loop.
///
/// A body `let` that rebinds a binding already in `vars` (an `input`/`param`/
/// outer `let`/`bind`) is *loop-carried*: it threads through iterations, exactly
/// like the shadowing of a literal `repeat` unroll. Each carried binding gets a
/// fresh mutable cell (`__rlx_carry_<name>_<uid>`, uid-suffixed so nested loops
/// carrying the same name don't collide): seeded from the outer value, re-read
/// at the top of every iteration, written back at the bottom, and re-exposed
/// after the loop. Returns the loop tokens and the last carried ident (the
/// default graph output if no explicit `out`).
fn emit_repeat_runtime(
    count: &TokenStream,
    body: &[Stmt],
    vars: &HashSet<String>,
    uid: &mut u32,
) -> (TokenStream, Option<Ident>) {
    let level = collect_level_let_idents(body);

    // Carried = a body-rebound name that already exists in the outer scope.
    let mut seen: HashSet<String> = HashSet::new();
    let mut carried: Vec<Ident> = Vec::new();
    for id in &level {
        let s = id.to_string();
        if vars.contains(&s) && seen.insert(s) {
            carried.push(id.clone());
        }
    }

    let my = *uid;
    *uid += 1;
    let cells: Vec<Ident> = carried
        .iter()
        .map(|v| Ident::new(&format!("__rlx_carry_{v}_{my}"), Span::mixed_site()))
        .collect();

    // Body auto-borrow set: outer vars plus every name bound at this level.
    let mut inner_vars = vars.clone();
    for id in &level {
        inner_vars.insert(id.to_string());
    }
    let body_ts = emit_stmts_block(body, &inner_vars, uid);

    let pre = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { let mut #c = #v.clone(); });
    let entry = carried.iter().zip(&cells).map(|(v, c)| {
        quote! { #[allow(unused_variables)] let #v = #c.clone(); }
    });
    let write = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { #c = #v.clone(); });
    let expose = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { let #v = #c; });

    let ts = quote! {
        #(#pre)*
        for _ in 0..(#count) {
            #(#entry)*
            #(#body_ts)*
            #(#write)*
        }
        #(#expose)*
    };
    (ts, carried.last().cloned())
}

/// Emit `repeat i in start..end { … }` (runtime bounds) as a Rust
/// `for i in (start)..(end) { … }` loop. Identical loop-carry threading to
/// [`emit_repeat_runtime`], but the index `var` is live in the body — so a
/// per-layer `bind`-collection access `cb[i]` picks element `i` each iteration.
fn emit_repeat_indexed_runtime(
    var: &Ident,
    start: &TokenStream,
    end: &TokenStream,
    body: &[Stmt],
    vars: &HashSet<String>,
    uid: &mut u32,
) -> (TokenStream, Option<Ident>) {
    let level = collect_level_let_idents(body);

    let mut seen: HashSet<String> = HashSet::new();
    let mut carried: Vec<Ident> = Vec::new();
    for id in &level {
        let s = id.to_string();
        if vars.contains(&s) && seen.insert(s) {
            carried.push(id.clone());
        }
    }

    let my = *uid;
    *uid += 1;
    let cells: Vec<Ident> = carried
        .iter()
        .map(|v| Ident::new(&format!("__rlx_carry_{v}_{my}"), Span::mixed_site()))
        .collect();

    let mut inner_vars = vars.clone();
    for id in &level {
        inner_vars.insert(id.to_string());
    }
    let body_ts = emit_stmts_block(body, &inner_vars, uid);

    let pre = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { let mut #c = #v.clone(); });
    let entry = carried.iter().zip(&cells).map(|(v, c)| {
        quote! { #[allow(unused_variables)] let #v = #c.clone(); }
    });
    let write = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { #c = #v.clone(); });
    let expose = carried
        .iter()
        .zip(&cells)
        .map(|(v, c)| quote! { let #v = #c; });

    let ts = quote! {
        #(#pre)*
        for #var in (#start)..(#end) {
            #(#entry)*
            #(#body_ts)*
            #(#write)*
        }
        #(#expose)*
    };
    (ts, carried.last().cloned())
}

impl Expr {
    /// A sub-expression is a *scalar* if it is built purely from numeric
    /// literals — those get promoted (`x * 2.0`), the rest stay tensors.
    fn is_scalar(&self) -> bool {
        match self {
            Expr::Num(_) => true,
            Expr::Neg(e) => e.is_scalar(),
            Expr::Bin(_, a, b, _) => a.is_scalar() && b.is_scalar(),
            Expr::Cmp(_, a, b, _) => a.is_scalar() && b.is_scalar(),
            _ => false,
        }
    }

    fn emit(&self, vars: &HashSet<String>) -> TokenStream {
        match self {
            // A bare binding is cloned (a cheap graph-handle refcount bump)
            // so it stays usable in later statements.
            Expr::Var(id) => quote!(#id.clone()),
            Expr::Num(t) => quote!((#t) as f64),
            Expr::Neg(e) => {
                let x = e.emit(vars);
                if e.is_scalar() {
                    quote!(-(#x))
                } else {
                    quote!((#x).neg())
                }
            }
            Expr::MatMul(a, b, span) => {
                let aa = a.emit(vars);
                let bb = b.emit(vars);
                // Method ident carries the operator span so a shape/type error
                // localises to the `@`.
                let matmul = Ident::new("matmul", *span);
                quote!((#aa).#matmul(&(#bb)))
            }
            Expr::Bin(kind, a, b, span) => {
                let op = kind.spanned(*span);
                let aa = a.emit(vars);
                let bb = b.emit(vars);
                match (a.is_scalar(), b.is_scalar()) {
                    // scalar `op` tensor — the left-scalar impls need `&Tensor`.
                    (true, false) => quote!((#aa) #op (&(#bb))),
                    // everything else: owned `Tensor op {Tensor|f64}` or f64 op f64.
                    _ => quote!((#aa) #op (#bb)),
                }
            }
            Expr::Cmp(kind, a, b, span) => {
                let aa = a.emit(vars);
                let bb = b.emit(vars);
                match (a.is_scalar(), b.is_scalar()) {
                    // scalar on the left — swap so the tensor is the receiver.
                    (true, false) => {
                        let m = Ident::new(kind.swapped().method(), *span);
                        quote!((#bb).#m(#aa))
                    }
                    // scalar on the right — promote it.
                    (false, true) => {
                        let m = Ident::new(kind.method(), *span);
                        quote!((#aa).#m(#bb))
                    }
                    // tensor vs tensor (both-scalar is rejected in `check`).
                    _ => {
                        let m = Ident::new(kind.method(), *span);
                        quote!((#aa).#m(&(#bb)))
                    }
                }
            }
            Expr::MethodDsl { recv, method, args } => {
                let r = recv.emit(vars);
                let a = args.iter().map(|(e, mode)| match mode {
                    ArgMode::Ref => {
                        let x = e.emit(vars);
                        quote!(&(#x))
                    }
                    ArgMode::Scalar => e.emit(vars),
                    // Raw config literal — emit verbatim so Rust infers the type
                    // (a `usize`/`i32` axis/head-dim), not `f64`.
                    ArgMode::Raw => match e {
                        Expr::Num(t) => quote!(#t),
                        _ => e.emit(vars),
                    },
                });
                quote!((#r).#method(#(#a),*))
            }
            Expr::Method { recv, name, args } => {
                let r = recv.emit(vars);
                let a = method_args(args, vars);
                quote!((#r).#name(#(#a),*))
            }
            Expr::Linear { x, w, b, act, .. } => {
                let xx = x.emit(vars);
                let ww = w.emit(vars);
                let bb = b.emit(vars);
                let act_tok = match act {
                    Some(v) => quote!(Some(Activation::#v)),
                    None => quote!(None),
                };
                quote!((#xx).linear_act(&(#ww), &(#bb), #act_tok))
            }
            // A `bind`-collection element: clone the indexed outer Tensor so it
            // is adopted into the graph (like a scalar `bind` used in an expr).
            Expr::AdoptIndex {
                base,
                index,
                fields,
                inner,
            } => {
                let inner = inner.iter();
                quote!((#base[#index] #(. #fields)* #([#inner])*).clone())
            }
            Expr::Index { base, .. } => syn::Error::new(
                base.span(),
                "internal error: unresolved family index reached codegen",
            )
            .to_compile_error(),
            Expr::Call { name, .. } => syn::Error::new(
                name.span(),
                "internal error: unresolved call reached codegen",
            )
            .to_compile_error(),
        }
    }
}

/// Lower escape-hatch method args. A lone identifier naming a declared tensor
/// binding is auto-borrowed (`k` → `&k`), so `q.attention(k, v, 8, 64, mask)`
/// reads naturally. Every other argument — literals, enum paths
/// (`MaskKind::Causal`), explicit `&x`, closures, turbofish calls, or a
/// parenthesised group (including the `~value` by-value escape, desugared to
/// `(value)` at parse time) — is forwarded verbatim, so a scalar reaches an
/// `f32`/`usize` parameter by value and the full `Tensor` method API stays
/// reachable.
fn method_args(args: &[syn::Expr], vars: &HashSet<String>) -> Vec<TokenStream> {
    args.iter()
        .map(|a| {
            if let syn::Expr::Path(p) = a {
                if p.qself.is_none() && p.path.leading_colon.is_none() && p.path.segments.len() == 1
                {
                    let seg = &p.path.segments[0];
                    if seg.arguments.is_none() && vars.contains(&seg.ident.to_string()) {
                        let id = &seg.ident;
                        return quote!(&#id);
                    }
                }
            }
            // `cb[i]` (or `layers[i].w`, `layers[i].a.b`) where the ROOT is a
            // `bind`-adopted collection (in `vars`) is auto-borrowed like a bare
            // tensor arg — `&(layers[0].w)` reaches a `&Tensor` parameter without
            // moving out of the `Vec`. Only an index/field chain qualifies (a
            // bare path is handled above); the root peels through `[…]`/`.field`.
            if matches!(a, syn::Expr::Index(_) | syn::Expr::Field(_)) {
                if let Some(root) = root_collection_ident(a) {
                    if vars.contains(&root.to_string()) {
                        return quote!(&(#a));
                    }
                }
            }
            quote!(#a)
        })
        .collect()
}

/// The root identifier of an index/field access chain (`layers[i].a.b` → `layers`),
/// peeling `[…]` (`Index`) and `.field` (`Field`) down to a single-segment path.
/// `None` for anything else (a computed base, a multi-segment path, etc.).
fn root_collection_ident(e: &syn::Expr) -> Option<&syn::Ident> {
    match e {
        syn::Expr::Field(f) => root_collection_ident(&f.base),
        syn::Expr::Index(ix) => root_collection_ident(&ix.expr),
        syn::Expr::Path(p)
            if p.qself.is_none()
                && p.path.leading_colon.is_none()
                && p.path.segments.len() == 1
                && p.path.segments[0].arguments.is_none() =>
        {
            Some(&p.path.segments[0].ident)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::rlx_build_impl;
    use quote::quote;

    fn expand(ts: proc_macro2::TokenStream) -> String {
        rlx_build_impl(ts).to_string()
    }

    #[test]
    fn good_program_lowers_cleanly() {
        let out = expand(quote! {
            graph "mlp";
            input x: [2, 4];
            param w: [4, 3];
            let y = gelu(x @ w);
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("matmul"));
        assert!(out.contains("gelu"));
    }

    #[test]
    fn unknown_binding_is_a_spanned_error() {
        let out = expand(quote! {
            input x: [2, 4];
            let y = x @ w;   // `w` never declared
            out y;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("unknown binding"));
    }

    #[test]
    fn matmul_on_scalar_is_rejected() {
        let out = expand(quote! {
            input x: [2, 4];
            let y = x @ 2.0;
            out y;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("tensor operands"));
    }

    #[test]
    fn scalar_let_is_rejected() {
        let out = expand(quote! {
            input x: [4];
            let s = 2.0 * 3.0;   // pure scalar — not a tensor
            let y = x * s;
            out y;
        });
        assert!(out.contains("compile_error"));
    }

    #[test]
    fn unknown_output_is_rejected() {
        let out = expand(quote! {
            input x: [2, 4];
            param w: [4, 3];
            let y = x @ w;
            out z;   // `z` never declared
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("unknown output binding"));
    }

    #[test]
    fn typo_in_method_arg_is_caught() {
        let out = expand(quote! {
            input x: [2, 4, 8];
            param wk: [8, 8];
            let k = x @ wk;
            let a = x.attention(k, vv, 8, 8, MaskKind::Causal);
            out a;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("unknown binding"));
        assert!(out.contains("vv"));
    }

    #[test]
    fn external_value_passes_through_parenthesized() {
        let out = expand(quote! {
            input x: [2, 4];
            let a = x.softmax((axis));
            out a;
        });
        assert!(!out.contains("compile_error"), "{out}");
    }

    #[test]
    fn enum_path_and_literal_args_are_not_flagged() {
        let out = expand(quote! {
            input x: [2, 4, 8];
            param wk: [8, 8];
            let k = x @ wk;
            let a = x.attention(k, k, 8, 8, MaskKind::Causal);
            out a;
        });
        assert!(!out.contains("compile_error"), "{out}");
    }

    #[test]
    fn matmul_binds_like_multiply_left_to_right() {
        let out = expand(quote! {
            input x: [2, 4];
            param w: [4, 3];
            param s: [2, 3];
            let y = x @ w * s;
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        let mm = out.find("matmul").expect("matmul present");
        let star = out.find(") * (").expect("outer multiply present");
        assert!(mm < star, "multiply should wrap the matmul: {out}");
    }

    // ── New-feature coverage ───────────────────────────────────────────

    #[test]
    fn comparison_and_select_sugar() {
        let out = expand(quote! {
            input x: [4];
            let y = select(x > 0.0, x, 0.0);
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains(". gt"), "{out}");
        assert!(out.contains("where_"), "{out}");
    }

    #[test]
    fn binary_sugar_maximum_and_pow() {
        let out = expand(quote! {
            input x: [4];
            input w: [4];
            let a = maximum(x, w);
            let b = a ** 2;
            out b;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("maximum"), "{out}");
        assert!(out.contains(". pow"), "{out}");
    }

    #[test]
    fn fn_inline_expands_body() {
        let out = expand(quote! {
            fn block(x, w) { let h = gelu(x @ w); }
            input a: [2, 4];
            param w1: [4, 4];
            param w2: [4, 4];
            let o1 = block(a, w1);
            let o2 = block(o1, w2);
            out o2;
        });
        assert!(!out.contains("compile_error"), "{out}");
        // Two inlined `gelu`s, one per call.
        assert_eq!(out.matches("gelu").count(), 2, "{out}");
    }

    #[test]
    fn fn_arity_mismatch_is_an_error() {
        let out = expand(quote! {
            fn block(x, w) { let h = x @ w; }
            input a: [2, 4];
            let o = block(a);
            out o;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("takes 2 argument"));
    }

    #[test]
    fn repeat_unrolls_body() {
        let out = expand(quote! {
            input x: [4, 4];
            param w: [4, 4];
            repeat 3 {
                let x = gelu(x @ w);
            }
            out x;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert_eq!(out.matches("gelu").count(), 3, "{out}");
    }

    #[test]
    fn runtime_repeat_emits_a_for_loop_not_an_unroll() {
        // A runtime count → a Rust `for _ in 0..(n)` loop; the body appears ONCE
        // (unlike a literal `repeat`, which duplicates the body per iteration).
        let out = expand(quote! {
            input x: [4, 4];
            param w: [4, 4];
            repeat n {
                let x = gelu(x @ w);
            }
            out x;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("for _ in 0 .."), "{out}");
        assert_eq!(out.matches("gelu").count(), 1, "body emitted once: {out}");
        // The rebound `x` is threaded through a fresh loop-carry cell.
        assert!(out.contains("__rlx_carry_x"), "{out}");
    }

    #[test]
    fn runtime_repeat_field_access_count() {
        // The count may be any brace-free expression (e.g. a config field).
        let out = expand(quote! {
            input x: [4, 4];
            param w: [4, 4];
            repeat cfg.n_layer {
                let x = x @ w;
            }
            out x;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("cfg . n_layer"), "{out}");
    }

    #[test]
    fn bind_makes_an_outer_tensor_a_known_binding() {
        // A `bind`-adopted name is auto-borrowed like a declared binding, and is
        // NOT rejected as an unknown method argument.
        let out = expand(quote! {
            bind idx, cb;
            input x: [2, 4];
            let y = x.synth_matmul(idx, cb, 2u32, 4u32);
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("synth_matmul"), "{out}");
    }

    #[test]
    fn bind_collection_indexed_by_repeat_index() {
        // `bind cb[], idx[];` + `cb[i]`/`idx[i]` in a method arg → each unrolled
        // iteration clones the outer `Vec` element and borrows it (`&(idx[0])`).
        let out = expand(quote! {
            input x: [2, 4];
            bind cb[], idx[];
            let h = x;
            repeat i in 0..2 {
                let h = h + x.synth_matmul(idx[i], cb[i], 2u32, 4u32);
            }
            out h;
        });
        assert!(!out.contains("compile_error"), "{out}");
        // Two unrolled synth_matmuls, indices materialized to 0 and 1.
        assert_eq!(out.matches("synth_matmul").count(), 2, "{out}");
        assert!(out.contains("idx [0]") || out.contains("idx[0]"), "{out}");
        assert!(out.contains("idx [1]") || out.contains("idx[1]"), "{out}");
    }

    #[test]
    fn bind_collection_expr_position() {
        // `ws[i]` in DSL-expr position lowers to `ws[i].clone()` (adopt).
        let out = expand(quote! {
            input x: [4, 4];
            bind ws[];
            let h = x;
            repeat i in 0..2 { let h = h @ ws[i]; }
            out h;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("ws [0]") || out.contains("ws[0]"), "{out}");
        assert!(out.contains(". clone"), "{out}");
    }

    #[test]
    fn bind_collection_field_access_in_method_arg() {
        // `layers[i].idx` / `layers[i].cb` — a `.field` after the collection index
        // is borrowed with the repeat index substituted per unrolled iteration.
        let out = expand(quote! {
            input x: [2, 4];
            bind layers[];
            let h = x;
            repeat i in 0..2 {
                let h = h + x.synth_matmul(layers[i].idx, layers[i].cb, 2u32, 4u32);
            }
            out h;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert_eq!(out.matches("synth_matmul").count(), 2, "{out}");
        // Indices materialized (0 and 1) and the field borrowed by reference.
        assert!(
            out.contains("layers [0] . idx") || out.contains("layers[0].idx"),
            "{out}"
        );
        assert!(
            out.contains("layers [1] . cb") || out.contains("layers[1].cb"),
            "{out}"
        );
        assert!(out.contains("& (layers [0] . idx"), "auto-borrowed: {out}");
    }

    #[test]
    fn bind_collection_field_access_in_expr_position() {
        // `layers[i].w` as a matmul operand → `(layers[i].w).clone()` (adopt).
        let out = expand(quote! {
            input x: [4, 4];
            bind layers[];
            let h = x;
            repeat i in 0..2 { let h = h @ layers[i].w; }
            out h;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(
            out.contains("layers [0] . w") || out.contains("layers[0].w"),
            "{out}"
        );
        assert!(out.contains(". clone"), "{out}");
    }

    #[test]
    fn field_access_on_param_family_is_rejected() {
        // A `.field` is only valid on a `bind` collection, not a param family.
        let out = expand(quote! {
            input x: [1, 4];
            param w[2]: [4, 4];
            repeat i in 0..2 { let x = x @ w[i].foo; }
            out x;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("only valid on a `bind`-adopted collection"));
    }

    #[test]
    fn bind_collection_nonempty_brackets_is_rejected() {
        let out = expand(quote! {
            input x: [2, 4];
            bind cb[3];
            let y = x + cb;
            out y;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("EMPTY brackets"));
    }

    #[test]
    fn bind_inside_repeat_is_rejected() {
        let out = expand(quote! {
            input x: [4, 4];
            repeat 2 {
                bind w;
                let x = x @ w;
            }
            out x;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("inside `repeat`"));
    }

    #[test]
    fn repeat_rejects_param_declaration() {
        let out = expand(quote! {
            input x: [4, 4];
            repeat 2 {
                param w: [4, 4];
                let x = x @ w;
            }
            out x;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("inside `repeat`"));
    }

    #[test]
    fn scan_lowers_to_scan_block() {
        let out = expand(quote! {
            input h0: [1, 8];
            param w: [8, 8];
            scan h = h0 for 4 {
                let h = relu(h @ w);
            }
            out h;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("scan_block"), "{out}");
    }

    #[test]
    fn scan_empty_body_is_rejected() {
        let out = expand(quote! {
            input h0: [1, 8];
            scan h = h0 for 4 { }
            out h;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("empty body"));
    }

    #[test]
    fn scan_inside_repeat_is_rejected() {
        let out = expand(quote! {
            input x: [4, 4];
            param w: [4, 4];
            repeat 2 {
                scan h = x for 3 { let h = h @ w; }
            }
            out x;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("inside `repeat`"));
    }

    #[test]
    fn array_const_lowers_to_constant_nd() {
        let out = expand(quote! {
            input x: [2, 2];
            const mask = [[1.0, 0.0], [0.0, 1.0]] : F32;
            let y = x * mask;
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        assert!(out.contains("constant_nd"), "{out}");
    }

    #[test]
    fn ragged_array_const_is_rejected() {
        let out = expand(quote! {
            input x: [2, 2];
            const mask = [[1.0, 0.0], [0.0]] : F32;
            let y = x * mask;
            out y;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("ragged"));
    }

    #[test]
    fn static_matmul_mismatch_is_caught() {
        let out = expand(quote! {
            input x: [2, 4];
            param w: [8, 3];   // inner dims 4 vs 8
            let y = x @ w;
            out y;
        });
        assert!(out.contains("compile_error"));
        assert!(out.contains("inner dimensions differ"));
    }

    #[test]
    fn dynamic_matmul_is_not_flagged() {
        let out = expand(quote! {
            input x: [?, 4];
            param w: [4, 3];
            let y = x @ w;
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
    }
}
