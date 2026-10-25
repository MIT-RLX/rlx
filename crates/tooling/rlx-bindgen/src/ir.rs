// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The whole C ABI as one typed list, for the FFI-language emitters.
//!
//! Both tables — the op table (graph builders) and the runtime API table
//! (`api.idl`) — lower here to [`Func`]s whose parameters and results use one
//! kind vocabulary, each checked against the exported C function. An emitter
//! (Go, Java, C#, Ruby, Julia, Zig) then needs one marshaling rule per kind,
//! not one per function, and every language gets every function.

use rlx_bind::OpSpec;
use rlx_bind::api::{Api, Returns, Role};

use crate::header::CFn;

/// A handle type, `Rlx<Name>` in C.
#[derive(Clone, Debug)]
pub struct Class {
    pub name: String,
    pub doc: String,
    /// `rlx_<c>_…`.
    pub c: String,
    pub feature: Option<String>,
}

impl Class {
    pub fn c_free(&self) -> String {
        format!("rlx_{}_free", self.c)
    }
}

/// One argument: canonical snake_case name, kind, default literal.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub kind: String,
    pub default: Option<Lit>,
}

impl Param {
    /// Whether a caller may leave it out (it has a default, or `None` is a value).
    pub fn optional(&self) -> bool {
        self.default.is_some() || self.kind.ends_with('?')
    }
}

/// One C-callable function.
#[derive(Clone, Debug)]
pub struct Func {
    /// The owning class (index into [`Ir::classes`]); graph builders belong to `Graph`.
    pub class: Option<usize>,
    pub name: String,
    pub role: Role,
    pub doc: String,
    pub params: Vec<Param>,
    /// Named results, in out-parameter order (a constructor's handle included).
    pub rets: Vec<(String, String)>,
    /// The result is the C return value (`bool`, a static `const char*`), not
    /// an out-parameter behind an `int32_t` status.
    pub direct: bool,
    pub c: String,
    pub feature: Option<String>,
    /// Takes host callbacks or updates arrays in place: each language writes it.
    pub manual: bool,
    /// A graph builder from the op table.
    pub op: bool,
    /// The JS spelling (method part only) — what rlx-js calls it.
    pub js: String,
    /// rlx-js reaches this static as a top-level function (`openGguf`).
    pub js_free: bool,
}

/// The lowered ABI.
#[derive(Clone, Debug)]
pub struct Ir {
    pub classes: Vec<Class>,
    pub funcs: Vec<Func>,
}

impl Ir {
    pub fn class_of(&self, f: &Func) -> Option<&Class> {
        f.class.map(|i| &self.classes[i])
    }

    pub fn members<'a>(&'a self, c: &'a Class) -> impl Iterator<Item = &'a Func> {
        let i = self.classes.iter().position(|k| k.name == c.name);
        self.funcs.iter().filter(move |f| f.class == i)
    }

    pub fn free_funcs(&self) -> impl Iterator<Item = &Func> {
        self.funcs.iter().filter(|f| f.class.is_none())
    }
}

// ── literals ────────────────────────────────────────────────

/// A default, parsed from the tables' neutral literals.
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    None,
    Bool(bool),
    /// As written (`-1`, `1e-5`, `0.0`).
    Num(String),
    Str(String),
    List(Vec<Lit>),
}

impl Lit {
    pub fn parse(text: &str) -> Lit {
        let t = crate::kinds::tidy(text.trim());
        let t = t.trim();
        match t {
            "None" | "none" | "nil" | "null" => Lit::None,
            "true" => Lit::Bool(true),
            "false" => Lit::Bool(false),
            _ if t.starts_with('"') => Lit::Str(t.trim_matches('"').to_string()),
            _ if t.starts_with('[') || t.starts_with("vec![") => {
                let inner = t
                    .trim_start_matches("vec!")
                    .trim_start_matches('[')
                    .trim_end_matches(']');
                Lit::List(
                    inner
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(Lit::parse)
                        .collect(),
                )
            }
            _ => Lit::Num(t.replace(' ', "")),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Lit::Num(n) if n.contains('.') || n.contains('e') || n.contains("inf"))
    }
}

// ── C expansion of every kind ───────────────────────────────

/// The C parameter types an argument kind takes.
pub fn c_args(kind: &str) -> Vec<String> {
    let v: &[&str] = match kind {
        "str" | "str?" | "json" | "json?" | "value" | "value?" => &["const char*"],
        "bool" => &["bool"],
        "f32" => &["float"],
        "f64" => &["double"],
        "i32" => &["int32_t"],
        "i64" => &["int64_t"],
        "u32" | "node" | "node?" => &["uint32_t"],
        "u64" => &["uint64_t"],
        "usize" => &["size_t"],
        "u64?" => &["const uint64_t*"],
        "usize?" => &["const size_t*"],
        "f32?" => &["const float*"],
        "f32s" | "frows2" | "frows6" => &["const float*", "size_t"],
        "u8s" => &["const uint8_t*", "size_t"],
        "u32s" | "nodes" => &["const uint32_t*", "size_t"],
        "usizes" | "usizes?" | "rows2" => &["const size_t*", "size_t"],
        "i64s" => &["const int64_t*", "size_t"],
        "strs" | "strs?" => &["const char* const*", "size_t"],
        "inputs" => &[
            "const char* const*",
            "const float* const*",
            "const size_t*",
            "size_t",
        ],
        "typed_inputs" => &[
            "const char* const*",
            "const uint8_t* const*",
            "const size_t*",
            "const char* const*",
            "size_t",
        ],
        "graph" | "graph&" | "graph?" => &["RlxGraph*"],
        other => {
            let class = other.trim_end_matches('&');
            return vec![format!("Rlx{class}*")];
        }
    };
    v.iter().map(|s| s.to_string()).collect()
}

/// The C out-parameter types a result kind takes.
pub fn c_outs(kind: &str) -> Vec<String> {
    let v: &[&str] = match kind {
        "str" | "str?" | "json" | "object" | "rows" | "shapes" | "shape" | "strings" => &["char**"],
        "f32s" => &["RlxF32s*"],
        "u8s" => &["RlxBytes*"],
        "u32s" | "nodes" => &["RlxU32s*"],
        "outputs" => &["RlxOutputs**"],
        "typed_outputs" => &["RlxTypedOutputs**"],
        "tensors" => &["RlxTensorMap**"],
        "bool" => &["bool*"],
        "usize" => &["size_t*"],
        "u32" | "node" => &["uint32_t*"],
        "u64" => &["uint64_t*"],
        "f32" => &["float*"],
        "f32?" => &["float*", "bool*"],
        "graph" => &["RlxGraph**"],
        class => return vec![format!("Rlx{class}**")],
    };
    v.iter().map(|s| s.to_string()).collect()
}

/// An op-table argument kind in the shared vocabulary.
fn op_kind(kind: &str) -> &str {
    match kind {
        "node" | "nodes" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "bool"
        | "usizes" | "i64s" | "f32s" => kind,
        "opt_node" => "node?",
        "pair" | "triple" => "usizes",
        "opt_usizes" | "opt_pair" => "usizes?",
        "f32x3" => "f32s",
        "bytes" => "u8s",
        "opt_usize" => "usize?",
        "opt_f32" => "f32?",
        "pads" => "rows2",
        "bands" => "frows2",
        "sos" => "frows6",
        "graph" => "graph",
        "opt_graph" => "graph?",
        "opt_str" | "opt_activation" => "str?",
        _ => "str",
    }
}

/// An op-table result kind as named results.
fn op_rets(ret: &str) -> Vec<(String, String)> {
    let one = |k: &str| vec![("result".to_string(), k.to_string())];
    match ret {
        "node" => one("node"),
        "pair" => vec![
            ("first".into(), "node".into()),
            ("second".into(), "node".into()),
        ],
        "nodes" => one("nodes"),
        "unit" => vec![],
        "usize" => one("usize"),
        "shape" => one("shape"),
        "strings" => one("strings"),
        _ => one("str"),
    }
}

/// Lower both tables, checking every function against the C ABI.
pub fn build(ops: &[OpSpec], api: &Api, fns: &[CFn]) -> Result<Ir, Vec<String>> {
    let by_name = |n: &str| fns.iter().find(|f| f.name == n);
    let mut ir = Ir {
        classes: api
            .classes
            .iter()
            .map(|c| Class {
                name: c.name.clone(),
                doc: c.doc.clone(),
                c: c.c.clone(),
                feature: c.feature.clone(),
            })
            .collect(),
        funcs: Vec::new(),
    };
    let graph = ir.classes.iter().position(|c| c.name == "Graph");
    let mut errors = Vec::new();

    for e in &api.entries {
        let Some(c) = &e.names.c else { continue };
        let Some(f) = by_name(c) else {
            errors.push(format!("{}: no C function `{c}`", e.name));
            continue;
        };
        let rets = match (e.role, &e.returns) {
            (Role::Constructor, _) => {
                vec![("handle".into(), api.classes[e.class.unwrap()].name.clone())]
            }
            (_, Returns::Unit) => vec![],
            (_, Returns::One(k)) => vec![("result".into(), k.clone())],
            (_, Returns::Tuple(items)) => items.clone(),
        };
        ir.funcs.push(Func {
            class: e.class,
            name: e.name.clone(),
            role: e.role,
            doc: e.doc.clone(),
            params: e
                .params
                .iter()
                .map(|p| Param {
                    name: p.name.clone(),
                    kind: p.kind.clone(),
                    default: p.default.as_deref().map(Lit::parse),
                })
                .collect(),
            rets,
            direct: f.ret != "int32_t",
            c: c.clone(),
            feature: e.feature.clone(),
            manual: e.manual,
            op: false,
            js: e
                .names
                .js
                .as_deref()
                .map(|n| n.rsplit('.').next().unwrap_or(n).to_string())
                .unwrap_or_else(|| camel(&e.name)),
            js_free: e.role == Role::Static
                && e.names.js.as_deref().is_some_and(|n| !n.contains('.')),
        });
    }
    for op in ops {
        let c = format!("rlx_graph_{}", op.c_name());
        let Some(f) = by_name(&c) else {
            errors.push(format!("op {}: no C function `{c}`", op.name));
            continue;
        };
        ir.funcs.push(Func {
            class: graph,
            name: op.name.trim_end_matches('_').to_string(),
            role: Role::Method,
            doc: op.doc.lines().map(str::trim).collect::<Vec<_>>().join("\n"),
            params: op
                .args
                .iter()
                .map(|a| Param {
                    name: a.name.to_string(),
                    kind: op_kind(a.kind).to_string(),
                    default: a.default.map(Lit::parse).filter(|l| *l != Lit::None),
                })
                .collect(),
            rets: op_rets(op.ret),
            direct: f.ret != "int32_t",
            c,
            feature: None,
            manual: false,
            op: true,
            js: op
                .js_names()
                .first()
                .cloned()
                .unwrap_or_else(|| camel(op.name)),
            js_free: false,
        });
    }

    // Every function's C signature must be what its kinds expand to.
    for func in ir.funcs.iter().filter(|f| !f.manual) {
        let f = by_name(&func.c).expect("looked up above");
        let mut want = Vec::new();
        if func.role == Role::Method {
            want.push(format!("Rlx{}*", ir.classes[func.class.unwrap()].name));
        }
        for p in &func.params {
            want.extend(c_args(&p.kind));
        }
        if !func.direct {
            for (_, k) in &func.rets {
                want.extend(c_outs(k));
            }
        }
        let got: Vec<String> = f.params.iter().map(|(t, _)| t.clone()).collect();
        if got != want {
            errors.push(format!(
                "`{}` takes ({}) but its row expands to ({})",
                func.c,
                got.join(", "),
                want.join(", ")
            ));
        }
    }
    if errors.is_empty() {
        Ok(ir)
    } else {
        Err(errors)
    }
}

/// snake_case → PascalCase.
pub fn pascal(snake: &str) -> String {
    rlx_bind::api::type_name(snake)
}

/// snake_case → camelCase.
pub fn camel(snake: &str) -> String {
    rlx_bind::manifest::camel(snake)
}

/// Doc lines with a prefix (`/// `, `# `, `// `).
pub fn doc_lines(doc: &str, prefix: &str) -> String {
    let mut s = String::new();
    for line in doc.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            s.push_str(prefix.trim_end());
        } else {
            s.push_str(prefix);
            s.push_str(line);
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_tables_lower_against_the_real_c_abi() {
        let root = crate::workspace_root();
        let fns = crate::capi_functions(&root);
        let ir = build(rlx_bind::manifest(), rlx_bind::api::api(), &fns)
            .unwrap_or_else(|e| panic!("{}", e.join("\n")));
        assert!(ir.funcs.iter().filter(|f| f.op).count() > 150);
        let conv = ir.funcs.iter().find(|f| f.name == "conv2d").unwrap();
        assert_eq!(conv.params[2].kind, "usizes");
        assert_eq!(
            conv.params[2].default,
            Some(Lit::List(vec![Lit::Num("1".into()), Lit::Num("1".into())]))
        );
    }

    #[test]
    fn literals_parse() {
        assert_eq!(Lit::parse("- 1"), Lit::Num("-1".into()));
        assert_eq!(Lit::parse("\"causal\""), Lit::Str("causal".into()));
        assert_eq!(Lit::parse("None"), Lit::None);
        assert!(Lit::parse("1e-5").is_float());
    }
}
