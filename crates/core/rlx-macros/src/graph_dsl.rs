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
//! * this proc macro owns *parsing, checking, and lowering* — a real Pratt
//!   parser, a semantic pass that reports precise spanned errors, and
//!   **path-free** `Tensor` method/operator output resolved against the
//!   wrapper's `use`.
//!
//! # Grammar
//! ```text
//! program   := ( "graph" STRING ";" )?  stmt*
//! stmt      := "input"  IDENT ":" shape ";"
//!            | "param"  IDENT ":" shape ";"
//!            | "const"  IDENT "=" ("-")? LIT ":" DTYPE ";"
//!            | "let"    IDENT "=" expr ";"
//!            | ("out"|"output") IDENT ("," IDENT)* ";"
//! shape     := "[" <tokens forwarded verbatim to shape!> "]"
//! expr      := term (("+"|"-"|"*"|"/"|"@") term)*
//! term      := "-" term
//!            | "(" expr ")"
//!            | NUMBER
//!            | IDENT
//!            | IDENT "(" expr ("," expr)* ")"           // f(x) → x.f(); matmul(a,b)
//!            | term "." IDENT "(" <Rust exprs> ")"      // escape hatch
//! ```
//!
//! In an escape-hatch `.method(args)`, a bare identifier is a binding
//! reference — it's validated and auto-borrowed (`k` → `&k`). Any other arg
//! (literal, enum path, `&x`, call, closure) is raw Rust; wrap an external
//! value as `(value)` to pass it through unchecked.
//!
//! # Precedence
//! Tightest → loosest: postfix `.method(…)` > unary `-` > (`@` `*` `/`) >
//! (`+` `-`). `@` shares a band with `* /` and is left-associative — matching
//! NumPy/Python, so `x @ w * s` is `(x @ w) * s`, not `x @ (w * s)`.
//! Elementwise ops broadcast; a scalar literal on either side of `+ - * /`
//! is promoted (`x * 2.0`, `0.5 * x`).

use proc_macro2::{Punct, Spacing, Span, TokenStream, TokenTree};
use quote::quote;
use std::collections::HashSet;
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
    if let Err(e) = dsl.check() {
        return e.to_compile_error();
    }
    dsl.codegen()
}

// ── AST ──────────────────────────────────────────────────────────────────

struct GraphDsl {
    name: Option<LitStr>,
    stmts: Vec<Stmt>,
}

enum Stmt {
    Decl {
        kind: DeclKind,
        name: Ident,
        shape: TokenStream,
    },
    Const {
        name: Ident,
        neg: bool,
        value: Lit,
        dtype: Ident,
    },
    Let {
        name: Ident,
        expr: Expr,
    },
    Out {
        names: Vec<Ident>,
    },
}

enum DeclKind {
    Input,
    Param,
}

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
    /// Method call — args are raw Rust expressions (the escape hatch).
    Method {
        recv: Box<Expr>,
        name: Ident,
        args: Vec<syn::Expr>,
    },
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

fn parse_stmt(input: ParseStream) -> syn::Result<Stmt> {
    if input.peek(Token![let]) {
        input.parse::<Token![let]>()?;
        let name = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let expr = parse_expr(input, 0)?;
        input.parse::<Token![;]>()?;
        return Ok(Stmt::Let { name, expr });
    }

    if input.peek(Token![const]) {
        input.parse::<Token![const]>()?;
        let name = input.parse::<Ident>()?;
        input.parse::<Token![=]>()?;
        let neg = input.peek(Token![-]);
        if neg {
            input.parse::<Token![-]>()?;
        }
        let value = input.parse::<Lit>()?;
        input.parse::<Token![:]>()?;
        let dtype = input.parse::<Ident>()?;
        input.parse::<Token![;]>()?;
        return Ok(Stmt::Const {
            name,
            neg,
            value,
            dtype,
        });
    }

    // Contextual-keyword statements: input / param / out / output.
    let kw = input.parse::<Ident>()?;
    match kw.to_string().as_str() {
        "input" | "param" => {
            let name = input.parse::<Ident>()?;
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
            Ok(Stmt::Decl { kind, name, shape })
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
        other => Err(syn::Error::new(
            kw.span(),
            format!(
                "unknown rlx! statement `{other}` — expected one of \
                 `input`, `param`, `const`, `let`, `out`"
            ),
        )),
    }
}

/// A binary operator and its left binding power (higher = tighter). `@` shares
/// the `* /` band to match NumPy/Python; all four are left-associative.
enum InfixOp {
    Bin(BinKind),
    Mat,
}

fn peek_infix(input: ParseStream) -> Option<(InfixOp, u8)> {
    if input.peek(Token![+]) {
        Some((InfixOp::Bin(BinKind::Add), 1))
    } else if input.peek(Token![-]) {
        Some((InfixOp::Bin(BinKind::Sub), 1))
    } else if input.peek(Token![*]) {
        Some((InfixOp::Bin(BinKind::Mul), 3))
    } else if input.peek(Token![/]) {
        Some((InfixOp::Bin(BinKind::Div), 3))
    } else if input.peek(Token![@]) {
        Some((InfixOp::Mat, 3))
    } else {
        None
    }
}

/// Pratt parser. `min_bp` is the minimum binding power that keeps a binary
/// operator in this sub-expression (precedence climbing).
fn parse_expr(input: ParseStream, min_bp: u8) -> syn::Result<Expr> {
    let mut lhs = parse_prefix(input)?;

    while let Some((op, l_bp)) = peek_infix(input) {
        if l_bp < min_bp {
            break;
        }
        let span = input.span(); // the operator token's span
        consume_infix(input)?;
        let rhs = parse_expr(input, l_bp + 1)?; // +1 ⇒ left-associative
        lhs = match op {
            InfixOp::Mat => Expr::MatMul(Box::new(lhs), Box::new(rhs), span),
            InfixOp::Bin(kind) => Expr::Bin(kind, Box::new(lhs), Box::new(rhs), span),
        };
    }

    Ok(lhs)
}

fn consume_infix(input: ParseStream) -> syn::Result<()> {
    if input.peek(Token![+]) {
        input.parse::<Token![+]>().map(drop)
    } else if input.peek(Token![-]) {
        input.parse::<Token![-]>().map(drop)
    } else if input.peek(Token![*]) {
        input.parse::<Token![*]>().map(drop)
    } else if input.peek(Token![/]) {
        input.parse::<Token![/]>().map(drop)
    } else {
        input.parse::<Token![@]>().map(drop)
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
        // Escape hatch: method args are real Rust expressions. Parsing them as
        // `syn::Expr` (not raw tokens) makes comma-splitting correct even with
        // turbofish, closures, or nested generics.
        let parsed = Punctuated::<syn::Expr, Token![,]>::parse_terminated(&content)?;
        e = Expr::Method {
            recv: Box::new(e),
            name,
            args: parsed.into_iter().collect(),
        };
    }
    Ok(e)
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

    // Identifier: either a bare tensor var or a function-call sugar.
    if input.peek(Ident) {
        let id = input.parse::<Ident>()?;
        if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let args = parse_arg_list(&content)?;
            return make_call(id, args);
        }
        return Ok(Expr::Var(id));
    }

    Err(input.error("expected a tensor expression (a binding name, `f(x)`, `a @ b`, or `(…)`)"))
}

/// Arguments to `f(…)` sugar are DSL expressions (so `gelu(x @ w + b)` works),
/// unlike `.method(…)` escape-hatch args which are raw Rust.
fn parse_arg_list(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let mut args = Vec::new();
    if input.is_empty() {
        return Ok(args);
    }
    loop {
        args.push(parse_expr(input, 0)?);
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        } else {
            break;
        }
    }
    Ok(args)
}

/// Map `f(args)` sugar to an AST node:
///  * `matmul(a, b)` / `mm(a, b)` → `a @ b`
///  * `f(x)`                      → `x.f()`  (any no-extra-arg method)
fn make_call(id: Ident, mut args: Vec<Expr>) -> syn::Result<Expr> {
    let name = id.to_string();
    if name == "matmul" || name == "mm" {
        if args.len() != 2 {
            return Err(syn::Error::new(
                id.span(),
                format!("`{name}(a, b)` takes exactly two operands"),
            ));
        }
        let b = args.pop().unwrap();
        let a = args.pop().unwrap();
        return Ok(Expr::MatMul(Box::new(a), Box::new(b), id.span()));
    }
    match args.len() {
        1 => Ok(Expr::Method {
            recv: Box::new(args.pop().unwrap()),
            name: id,
            args: Vec::new(),
        }),
        _ => Err(syn::Error::new(
            id.span(),
            format!(
                "`{name}(…)` with {} arguments has no sugar — call it as a \
                 method instead, e.g. `x.{name}(…)`",
                args.len()
            ),
        )),
    }
}

// ── Semantic checks ──────────────────────────────────────────────────────

impl GraphDsl {
    /// Resolve bindings and reject the mistakes that would otherwise surface as
    /// cryptic type errors in generated code: unknown names, matmul on a
    /// scalar, and `let`s that don't produce a tensor.
    fn check(&self) -> syn::Result<()> {
        // Every declared binding (for out-of-order `out` resolution).
        let mut declared: HashSet<String> = HashSet::new();
        for s in &self.stmts {
            if let Stmt::Decl { name, .. } | Stmt::Const { name, .. } | Stmt::Let { name, .. } = s {
                declared.insert(name.to_string());
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
            }
        }

        // A graph needs at least one output: an explicit `out`, or a `let` to
        // fall back on.
        if !has_output && !self.stmts.iter().any(|s| matches!(s, Stmt::Let { .. })) {
            return Err(syn::Error::new(
                Span::call_site(),
                "rlx! graph has no output — add `out <name>;` or at least one `let`",
            ));
        }
        Ok(())
    }
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
        Expr::Method { recv, args, .. } => {
            check_expr(recv, scope)?;
            for a in args {
                check_method_arg(a, scope)?;
            }
            Ok(())
        }
    }
}

/// A **bare identifier** in an escape-hatch method arg is a binding reference
/// (auto-borrowed at codegen), so a bare name that resolves to no binding is
/// almost always a typo — reject it with a hint. Everything else — a literal,
/// an enum path (`MaskKind::Causal`), `&x`, a call, a closure, or a
/// parenthesised group — is raw Rust and left to the compiler, which is also
/// the documented escape: wrap an external value as `(value)` to opt out.
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
            if let Stmt::Decl { name, .. } | Stmt::Const { name, .. } | Stmt::Let { name, .. } = s {
                vars.insert(name.to_string());
            }
        }

        let mut body = Vec::new();
        let mut outs: Vec<Ident> = Vec::new();
        let mut last_let: Option<Ident> = None;

        for s in &self.stmts {
            match s {
                Stmt::Decl { kind, name, shape } => {
                    let sname = LitStr::new(&name.to_string(), name.span());
                    let ctor = match kind {
                        DeclKind::Input => quote!(input),
                        DeclKind::Param => quote!(param),
                    };
                    body.push(quote! {
                        let #name = #scope.#ctor(#sname, shape!(#shape));
                    });
                }
                Stmt::Const {
                    name,
                    neg,
                    value,
                    dtype,
                } => {
                    let sign = if *neg { quote!(-) } else { quote!() };
                    body.push(quote! {
                        let #name = #scope.constant((#sign #value) as f64, DType::#dtype);
                    });
                }
                Stmt::Let { name, expr } => {
                    let e = expr.emit(&vars);
                    body.push(quote! { let #name = #e; });
                    last_let = Some(name.clone());
                }
                Stmt::Out { names } => outs.extend(names.iter().cloned()),
            }
        }

        if outs.is_empty() {
            if let Some(l) = last_let {
                outs.push(l);
            }
        }

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

impl Expr {
    /// A sub-expression is a *scalar* if it is built purely from numeric
    /// literals — those get promoted (`x * 2.0`), the rest stay tensors.
    fn is_scalar(&self) -> bool {
        match self {
            Expr::Num(_) => true,
            Expr::Neg(e) => e.is_scalar(),
            Expr::Bin(_, a, b, _) => a.is_scalar() && b.is_scalar(),
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
            Expr::Method { recv, name, args } => {
                let r = recv.emit(vars);
                let a = method_args(args, vars);
                quote!((#r).#name(#(#a),*))
            }
        }
    }
}

/// Lower escape-hatch method args. A lone identifier naming a declared tensor
/// binding is auto-borrowed (`k` → `&k`), so `q.attention(k, v, 8, 64, mask)`
/// reads naturally. Every other argument — literals, enum paths
/// (`MaskKind::Causal`), explicit `&x`, closures, turbofish calls — is
/// forwarded verbatim, so the full `Tensor` method API stays reachable.
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
            quote!(#a)
        })
        .collect()
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
        // A bare-ident arg naming no binding (`vv`, a typo for `v`) is a DSL
        // error now, not a downstream "cannot find value" surprise.
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
        // `(axis)` opts out of binding validation — it's raw Rust.
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
        // NumPy precedence: `x @ w * s` is `(x @ w) * s`. The lowering nests
        // the `*` outside the `.matmul(...)` call.
        let out = expand(quote! {
            input x: [2, 4];
            param w: [4, 3];
            param s: [2, 3];
            let y = x @ w * s;
            out y;
        });
        assert!(!out.contains("compile_error"), "{out}");
        // `.matmul (...) * (...)` — the multiply wraps the matmul, not vice versa.
        let mm = out.find("matmul").expect("matmul present");
        let star = out.find(") * (").expect("outer multiply present");
        assert!(mm < star, "multiply should wrap the matmul: {out}");
    }
}
