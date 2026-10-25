// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.rlxmodel`: the text form of rlx IR, and its stable encoding.
//!
//! ```text
//! rlxmodel 1
//! opset 1
//! rlx 0.2.18
//! model {name: "affine", version: "1.0.0"}
//!
//! graph @affine {
//!   %x = input : f32[1, 2]
//!   %w = param : f32[2, 2]
//!   %2 = MatMul %x, %w : f32[1, 2]
//!   %3 = Activation(Gelu) %2 : f32[1, 2]
//!   return %3
//! }
//! ```
//!
//! The header says which format and opset the file uses, which rlx wrote
//! it, optionally the oldest rlx that may read it (`requires 0.2.19`), and
//! the model's own block: its `name` and `version`, and anything its runtime
//! reads. Only `rlxmodel` and `opset` are required ([`Module`]).
//!
//! A node is `%label = Op operands : dtype[dims]`, with optional `name "…"`
//! and `origin {…}` after it. A value's label is its node's name when that
//! is a label (`%q_proj`, `%layers.0.k`: a letter or `_`, then letters,
//! digits, `_`, `.`) no other node has, else its position (`%7`). An input
//! or param named like its value is `%x = input` / `%w = param`. A dim is a size or a symbol (`?0`). An op and its
//! attributes are written as the [`Op`](crate::Op) variant itself:
//! `Reshape{new_shape: [1, -1]}`, `Cast(F16)`, `Narrow{axis: 1, start: 0,
//! len: 4}`. Nested graphs (the bodies of `If`, `While`, `Scan`) are
//! `graph @name { … }` blocks in place. Values are numbers (`nan`, `inf`,
//! `-inf` included; a NaN other than the canonical one keeps its bits as
//! `nan(0xffc00000)`), `"strings"`, `true` / `false`, `none`, `[lists]`,
//! `{field: value}` structs, `#{key: value}` maps and `b64"…"` bytes.
//! `//` starts a comment. `return %3 as logits` names an output
//! ([`Graph::name_output`](crate::Graph::name_output)); a name that is not a
//! label is quoted (`as "present.0.key"`).
//!
//! **Stability.** Ops, variants and fields are written by name, so reordering
//! the [`Op`](crate::Op) enum, or adding a variant, leaves every file
//! readable. Renaming a variant or field needs `#[serde(alias = "old")]`, and
//! a new field needs `#[serde(default)]` so files written before it still
//! read. The checked-in `tests/fixtures/opset1.rlxmodel` fails the tests when
//! that is missed. [`OPSET`] changes when an op's meaning changes, and a file
//! from a newer opset is refused rather than misread.

mod text;
mod value;

pub use text::ParseError;
pub use value::Error;

use crate::Graph;

/// The `.rlxmodel` format version this crate writes and reads.
pub const FORMAT_VERSION: u32 = 1;

/// The op-set version: what the ops mean. Files from a newer opset are
/// refused.
pub const OPSET: u32 = 1;

/// The file extension, without the dot.
pub const EXTENSION: &str = "rlxmodel";

/// Why a graph could not be written as text, or read from it.
#[derive(Debug, Clone, PartialEq)]
pub enum RlxModelError {
    /// Malformed text.
    Parse(ParseError),
    /// Well-formed text that is not a valid graph (an unknown op, a missing
    /// field), or a graph serde could not encode.
    Value(Error),
    /// A format or opset newer than this rlx, or a model that `requires` a
    /// newer rlx.
    Version(String),
}

impl std::fmt::Display for RlxModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "rlxmodel: {e}"),
            Self::Value(e) => write!(f, "rlxmodel: {e}"),
            Self::Version(e) => write!(f, "rlxmodel: {e}"),
        }
    }
}

impl std::error::Error for RlxModelError {}

impl From<ParseError> for RlxModelError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

impl From<Error> for RlxModelError {
    fn from(e: Error) -> Self {
        Self::Value(e)
    }
}

/// The rlx version this crate is, written as `rlx …` in every file.
pub const RLX_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A whole `.rlxmodel` file: its graphs, the model they make up, and the
/// versions that wrote and may read it.
///
/// ```text
/// rlxmodel 1
/// opset 1
/// rlx 0.2.18
/// requires 0.2.18
/// dims {seq: 0}
/// model {
///   name: "multilingual-e5-base",
///   version: "1.0.0",
///   authors: ["Liang Wang", "Nan Yang", "Furu Wei"],
///   license: "MIT",
///   repository: "https://github.com/microsoft/unilm",
///   homepage: "https://huggingface.co/intfloat/multilingual-e5-base",
///   card: """
/// # Multilingual-E5-base
///
/// Text embeddings by weakly-supervised contrastive pre-training.
/// """,
///   task: "embedding",
/// }
///
/// graph @encode { … }
/// ```
///
/// The model block is free-form. These fields have accessors, and tools
/// read them by these names: `name`, `version`, `authors`, `license` (SPDX,
/// one or a list), `repository`, `homepage`, `card` (the model card's
/// Markdown body) and `card_data` (its metadata, the YAML front matter of a
/// Hugging Face `README.md`, under Hugging Face's own keys). A text with
/// newlines prints as a raw `"""` block, verbatim.
#[derive(Debug, Clone, Default)]
pub struct Module {
    /// The model's own description: its `name` and `version`, and whatever
    /// else its runtime reads (`task`, entry points, weights). Free-form, as
    /// JSON.
    pub model: Option<serde_json::Value>,
    /// The rlx that wrote the file (`rlx 0.2.18`). [`write_module`] always
    /// writes [`RLX_VERSION`]; reading keeps what the file says.
    pub rlx: Option<String>,
    /// Names of the graphs' symbolic dims (`dims {seq: 0}`): a shape then
    /// reads `f32[1, seq]` rather than `f32[1, ?0]`.
    pub dims: Vec<(String, u32)>,
    /// The oldest rlx that may read the file (`requires 0.2.19`), for a model
    /// that needs a later runtime than its opset says. An older rlx refuses
    /// it.
    pub requires: Option<String>,
    pub graphs: Vec<Graph>,
}

impl Module {
    /// A module of `graphs`, with no model block.
    pub fn new(graphs: Vec<Graph>) -> Self {
        Self {
            graphs,
            ..Self::default()
        }
    }

    /// The model's `name`, if its block has one.
    pub fn model_name(&self) -> Option<&str> {
        self.model.as_ref()?.get("name")?.as_str()
    }

    /// The model's own `version`, if its block has one.
    pub fn model_version(&self) -> Option<&str> {
        self.model.as_ref()?.get("version")?.as_str()
    }

    fn model_str(&self, key: &str) -> Option<&str> {
        self.model.as_ref()?.get(key)?.as_str()
    }

    /// `authors`: a list of names (`"Name <email>"` is fine), or one string.
    pub fn authors(&self) -> Vec<&str> {
        strings(self.model.as_ref().and_then(|m| m.get("authors")))
    }

    /// `license`: SPDX expressions (`"MIT"`, `"Apache-2.0 OR MIT"`), one string
    /// or a list. A model whose weights and code differ lists both.
    pub fn licenses(&self) -> Vec<&str> {
        let own = strings(self.model.as_ref().and_then(|m| m.get("license")));
        if !own.is_empty() {
            return own;
        }
        strings(self.card_data().and_then(|d| d.get("license")))
    }

    /// `repository`: where its source lives (a GitHub URL, say).
    pub fn repository(&self) -> Option<&str> {
        self.model_str("repository")
    }

    /// `homepage`: the model's page (a Hugging Face model page, a paper).
    pub fn homepage(&self) -> Option<&str> {
        self.model_str("homepage")
    }

    /// `card`: the model card's Markdown body.
    pub fn card(&self) -> Option<&str> {
        self.model_str("card")
    }

    /// `card_data`: the model card's metadata, under Hugging Face's keys
    /// (the YAML front matter of a Hub `README.md`: `language`, `license`,
    /// `tags`, `datasets`, `base_model`, `pipeline_tag`, `model-index`, …).
    pub fn card_data(&self) -> Option<&serde_json::Value> {
        self.model.as_ref()?.get("card_data")
    }
}

/// A string, or the strings of a list.
fn strings(v: Option<&serde_json::Value>) -> Vec<&str> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.as_str()],
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_str()).collect(),
        _ => Vec::new(),
    }
}

/// `major.minor.patch` as numbers; a missing part is 0, and a pre-release
/// suffix (`-rc1`) is ignored.
fn version_parts(v: &str) -> Option<[u64; 3]> {
    let core = v.split(['-', '+']).next()?;
    let mut out = [0u64; 3];
    for (i, part) in core.split('.').enumerate() {
        *out.get_mut(i)? = part.parse().ok()?;
    }
    Some(out)
}

fn version_word(out: &mut String, v: &str) {
    if v.chars().all(|c| c.is_ascii_digit() || c == '.') && !v.is_empty() {
        out.push_str(v);
    } else {
        text::value(out, &value::Value::Str(v.to_string()), 0);
    }
}

/// Every shape in `v` (a struct of `dims` and `dtype`), its symbolic dims
/// renamed by `f`.
fn map_dims(
    v: &mut value::Value,
    f: &mut dyn FnMut(&mut value::Value) -> Result<(), String>,
) -> Result<(), String> {
    use value::{Payload, Value};
    map_shape_dims(v, &mut |d| match d {
        Value::Variant(name, Payload::Tuple(p)) if name == "Dynamic" && p.len() == 1 => {
            f(&mut p[0])
        }
        _ => Ok(()),
    })
}

/// Every dim of every shape in `v`, whole (`Static(n)`, `Dynamic(s)`, a
/// derived dim), passed to `f`.
fn map_shape_dims(
    v: &mut value::Value,
    f: &mut dyn FnMut(&mut value::Value) -> Result<(), String>,
) -> Result<(), String> {
    use value::{Payload, Value};
    match v {
        Value::Struct(fields) => {
            let is_shape = fields.len() == 2 && fields[0].0 == "dims" && fields[1].0 == "dtype";
            for (k, x) in fields.iter_mut() {
                if is_shape && k == "dims" {
                    if let Value::Seq(dims) = x {
                        for d in dims {
                            f(d)?;
                        }
                    }
                } else {
                    map_shape_dims(x, f)?;
                }
            }
        }
        Value::Seq(items) => {
            for x in items {
                map_shape_dims(x, f)?;
            }
        }
        Value::Variant(_, Payload::Tuple(items)) => {
            for x in items {
                map_shape_dims(x, f)?;
            }
        }
        Value::Variant(_, Payload::Struct(fields)) => {
            for (_, x) in fields {
                map_shape_dims(x, f)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// The derived dims a read graph's shapes write (`seq*16`), each now a
/// symbol of its own past every other, and the definitions of those.
fn derived_dims(
    g: &mut value::Value,
    dims: &[(String, u32)],
) -> Result<Vec<crate::dynamic::SymDef>, String> {
    use value::{Payload, Value};
    let num = |v: &Value| match v {
        Value::Num(s) => s.parse::<u32>().ok(),
        Value::Int(n) => u32::try_from(*n).ok(),
        _ => None,
    };
    let mut next = dims.iter().map(|(_, s)| s + 1).max().unwrap_or(0);
    map_dims(g, &mut |d| {
        if let Some(s) = num(d) {
            next = next.max(s + 1);
        }
        Ok(())
    })?;
    let mut defs: Vec<crate::dynamic::SymDef> = Vec::new();
    map_shape_dims(g, &mut |d| {
        let Value::Variant(k, Payload::Tuple(p)) = d else {
            return Ok(());
        };
        if k != text::DERIVED_DIM || !(3..=9).contains(&p.len()) {
            return Ok(());
        }
        let symbol = |v: &Value| -> Result<u32, String> {
            match v {
                Value::Str(name) => dims
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, s)| *s)
                    .ok_or_else(|| format!("dim `{name}` is not declared in the header's `dims`")),
                other => num(other).ok_or_else(|| "a derived dim's base is not a symbol".into()),
            }
        };
        let of = symbol(&p[0])?;
        let times = p
            .get(3)
            .filter(|v| !matches!(v, Value::Unit))
            .map(symbol)
            .transpose()?;
        let int = |v: &Value| match v {
            Value::Int(n) => i64::try_from(*n).ok(),
            _ => None,
        };
        let (Some(scale), Some(offset)) = (int(&p[1]), int(&p[2])) else {
            return Err("a derived dim's scale and offset are integers".into());
        };
        let at = |i: usize, default: i64| match p.get(i) {
            Some(v) => int(v).ok_or("a derived dim's terms are integers"),
            None => Ok(default),
        };
        let (linear, inner, div, plus_scale) = (at(4, 0)?, at(5, 0)?, at(6, 1)?, at(8, 0)?);
        let plus = p
            .get(7)
            .filter(|v| !matches!(v, Value::Unit))
            .map(symbol)
            .transpose()?;
        let found = crate::dynamic::SymDef {
            sym: 0,
            of,
            times,
            scale,
            linear,
            inner,
            div,
            plus,
            plus_scale,
            offset,
        };
        let def = match defs
            .iter()
            .find(|x| crate::dynamic::SymDef { sym: 0, ..**x } == found)
        {
            Some(x) => *x,
            None => {
                let x = crate::dynamic::SymDef { sym: next, ..found };
                next += 1;
                defs.push(x);
                x
            }
        };
        *d = Value::Variant(
            "Dynamic".into(),
            Payload::Tuple(vec![Value::Num(def.sym.to_string())]),
        );
        Ok(())
    })?;
    Ok(defs)
}

/// `module` as `.rlxmodel` text, stamped with this rlx's version.
pub fn write_module(module: &Module) -> Result<String, RlxModelError> {
    let mut out = format!("rlxmodel {FORMAT_VERSION}\nopset {OPSET}\nrlx ");
    version_word(&mut out, RLX_VERSION);
    out.push('\n');
    if let Some(r) = &module.requires {
        out.push_str("requires ");
        version_word(&mut out, r);
        out.push('\n');
    }
    if !module.dims.is_empty() {
        out.push_str("dims {");
        for (i, (name, sym)) in module.dims.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = std::fmt::Write::write_fmt(&mut out, format_args!("{name}: {sym}"));
        }
        out.push_str("}\n");
    }
    if let Some(m) = &module.model {
        out.push_str("model ");
        text::struct_lines(&mut out, &value::from_json(m));
        out.push('\n');
    }
    for g in &module.graphs {
        out.push('\n');
        let mut v = value::to_value(g)?;
        // Variables: `[input id, output index]`, which the text writes as
        // `var` inputs and `%k = %k_next` outputs.
        if !g.vars().is_empty()
            && let value::Value::Struct(fields) = &mut v
        {
            let pairs = g
                .vars()
                .iter()
                .map(|(i, o)| {
                    value::Value::Seq(vec![
                        value::Value::Int(i.0.into()),
                        value::Value::Int(*o as i128),
                    ])
                })
                .collect();
            fields.push((text::VARS.into(), value::Value::Seq(pairs)));
        }
        // Output names: `[output index, name]`, written `return %7 as logits`.
        if !g.output_names().is_empty()
            && let value::Value::Struct(fields) = &mut v
        {
            let pairs = g
                .output_names()
                .iter()
                .map(|(i, n)| {
                    value::Value::Seq(vec![
                        value::Value::Int(*i as i128),
                        value::Value::Str(n.clone()),
                    ])
                })
                .collect();
            fields.push((text::OUTPUT_NAMES.into(), value::Value::Seq(pairs)));
        }
        // Derived dims by their base: `seq*16`.
        if !g.sym_defs().is_empty() {
            map_shape_dims(&mut v, &mut |d| {
                use value::{Payload, Value};
                let Value::Variant(k, Payload::Tuple(p)) = d else {
                    return Ok(());
                };
                let n = match p.as_slice() {
                    [Value::Int(n)] if k == "Dynamic" => *n,
                    _ => return Ok(()),
                };
                let Some(def) = g.sym_defs().iter().find(|x| i128::from(x.sym) == n) else {
                    return Ok(());
                };
                let name = |sym: u32| {
                    module
                        .dims
                        .iter()
                        .find(|(_, s)| *s == sym)
                        .map_or(Value::Int(sym.into()), |(name, _)| Value::Str(name.clone()))
                };
                let mut parts = vec![
                    name(def.of),
                    Value::Int(def.scale.into()),
                    Value::Int(def.offset.into()),
                ];
                if let Some(t) = def.times {
                    parts.push(name(t));
                    if def.linear != 0 {
                        parts.push(Value::Int(def.linear.into()));
                    }
                }
                if def.div != 1 || def.plus.is_some() {
                    if parts.len() < 4 {
                        parts.push(Value::Unit);
                    }
                    if parts.len() < 5 {
                        parts.push(Value::Int(0));
                    }
                    parts.push(Value::Int(def.inner.into()));
                    parts.push(Value::Int(def.div.into()));
                }
                if let Some(q) = def.plus {
                    parts.push(name(q));
                    parts.push(Value::Int(def.plus_scale.into()));
                }
                *d = Value::Variant(text::DERIVED_DIM.into(), Payload::Tuple(parts));
                Ok(())
            })
            .map_err(Error)?;
        }
        // The reader numbers each derived dim anew, past every other symbol,
        // in the order the shapes name it: an attribute naming a derived
        // symbol names it by that number.
        let read_as: Vec<(u32, u32)> = if g.sym_defs().is_empty() {
            Vec::new()
        } else {
            let read = derived_dims(&mut v.clone(), &module.dims).map_err(Error)?;
            g.sym_defs()
                .iter()
                .filter_map(|d| {
                    read.iter()
                        .find(|r| {
                            crate::dynamic::SymDef { sym: 0, ..**r }
                                == crate::dynamic::SymDef { sym: 0, ..*d }
                        })
                        .map(|r| (d.sym, r.sym))
                })
                .collect()
        };
        // Symbolic attributes in place: `pos: upper`.
        for a in g.sym_attrs() {
            let sym = read_as
                .iter()
                .find(|(s, _)| *s == a.sym)
                .map_or(a.sym, |(_, r)| *r);
            let name = module
                .dims
                .iter()
                .find(|(_, s)| *s == sym)
                .map_or_else(|| format!("?{sym}"), |(n, _)| n.clone());
            let path: Vec<String> = ["nodes".to_string(), a.node.0.to_string(), "op".to_string()]
                .into_iter()
                .chain(a.path.iter().cloned())
                .collect();
            let leaf = value::at_mut(&mut v, &path).ok_or_else(|| {
                Error(format!("{} has no attribute {}", a.node, a.path.join(".")))
            })?;
            *leaf = text::sym_attr(name, i128::from(a.scale), i128::from(a.offset));
        }
        if !module.dims.is_empty() {
            map_dims(&mut v, &mut |d| {
                let n = match d {
                    value::Value::Int(n) => *n,
                    _ => return Ok(()),
                };
                if let Some((name, _)) = module.dims.iter().find(|(_, s)| i128::from(*s) == n) {
                    *d = value::Value::Str(name.clone());
                }
                Ok(())
            })
            .map_err(Error)?;
        }
        text::value(&mut out, &v, 0);
        out.push('\n');
    }
    Ok(out)
}

/// `graphs` as one `.rlxmodel` text, with no model block.
pub fn write_graphs(graphs: &[&Graph]) -> Result<String, RlxModelError> {
    write_module(&Module::new(graphs.iter().map(|g| (*g).clone()).collect()))
}

/// One graph as `.rlxmodel` text.
pub fn to_text(graph: &Graph) -> Result<String, RlxModelError> {
    write_graphs(&[graph])
}

/// A whole `.rlxmodel` text. Refused: a newer format or opset, or a
/// `requires` newer than [`RLX_VERSION`].
pub fn read_module(src: &str) -> Result<Module, RlxModelError> {
    let mut p = text::Parser::new(src)?;
    for (word, newest) in [("rlxmodel", FORMAT_VERSION), ("opset", OPSET)] {
        p.expect_ident(word)?;
        let n = p.number()?;
        let v: u32 = n
            .parse()
            .map_err(|_| p.err(format!("`{word}` takes a version number, found {n}")))?;
        if v > newest {
            return Err(RlxModelError::Version(format!(
                "{word} {v} is newer than this rlx reads ({newest}); update rlx"
            )));
        }
    }
    let mut module = Module::default();
    loop {
        let word = ["rlx", "requires", "model", "dims"]
            .into_iter()
            .find(|w| p.is_ident(w));
        let Some(word) = word else { break };
        p.expect_ident(word)?;
        let at = p.err("");
        let v = p.value()?;
        match word {
            "model" => module.model = Some(value::to_json(v)?),
            "dims" => {
                let value::Value::Struct(fields) = v else {
                    return Err(at_error(&at, "`dims` takes {name: symbol, …}".into()));
                };
                for (name, n) in fields {
                    let sym = match &n {
                        value::Value::Num(s) => s.parse::<u32>().ok(),
                        _ => None,
                    }
                    .ok_or_else(|| at_error(&at, format!("dim `{name}` needs a symbol number")))?;
                    module.dims.push((name, sym));
                }
            }
            _ => {
                let s = match v {
                    value::Value::Num(s) | value::Value::Str(s) => s,
                    value::Value::Variant(s, value::Payload::Unit) => s,
                    _ => return Err(at_error(&at, format!("`{word}` takes a version"))),
                };
                if version_parts(&s).is_none() {
                    return Err(at_error(&at, format!("`{s}` is not a version")));
                }
                if word == "rlx" {
                    module.rlx = Some(s);
                } else {
                    module.requires = Some(s);
                }
            }
        }
    }
    if let Some(req) = &module.requires
        && version_parts(req) > version_parts(RLX_VERSION)
    {
        return Err(RlxModelError::Version(format!(
            "this model requires rlx {req} or newer (this is {RLX_VERSION}); update rlx"
        )));
    }
    while !p.at_eof() {
        let at = p.err("");
        let mut g = p.graph()?;
        let dims = &module.dims;
        map_dims(&mut g, &mut |d| {
            if let value::Value::Str(name) = d {
                let sym = dims
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, s)| *s)
                    .ok_or_else(|| {
                        format!("dim `{name}` is not declared in the header's `dims`")
                    })?;
                *d = value::Value::Num(sym.to_string());
            }
            Ok(())
        })
        .map_err(|e| at_error(&at, e))?;
        let defs = derived_dims(&mut g, dims).map_err(|e| at_error(&at, e))?;
        let attrs = sym_attrs_of(&mut g, dims).map_err(|e| at_error(&at, e))?;
        let vars = take_vars(&mut g);
        let output_names = take_output_names(&mut g);
        module.graphs.push(
            value::from_value::<Graph>(g)
                .map(|mut graph| {
                    graph.set_sym_attrs(attrs);
                    graph.set_vars(vars);
                    graph.set_output_names(output_names);
                    graph.set_sym_defs(defs);
                    graph
                })
                .map_err(|e| Error(format!("graph at {}:{}: {e}", at.line, at.column)))?,
        );
    }
    Ok(module)
}

/// Take a parsed graph's variables out of its value (`[input, output]`).
fn take_vars(g: &mut value::Value) -> Vec<(crate::NodeId, usize)> {
    let value::Value::Struct(fields) = g else {
        return Vec::new();
    };
    let Some(i) = fields.iter().position(|(k, _)| k == text::VARS) else {
        return Vec::new();
    };
    let (_, v) = fields.remove(i);
    let value::Value::Seq(pairs) = v else {
        return Vec::new();
    };
    let num = |v: &value::Value| match v {
        value::Value::Num(n) => n.parse::<u32>().ok(),
        value::Value::Int(i) => u32::try_from(*i).ok(),
        _ => None,
    };
    pairs
        .iter()
        .filter_map(|p| match p {
            value::Value::Seq(p) if p.len() == 2 => {
                Some((crate::NodeId(num(&p[0])?), num(&p[1])? as usize))
            }
            _ => None,
        })
        .collect()
}

/// Take a parsed graph's output names out of its value (`[output, name]`).
fn take_output_names(g: &mut value::Value) -> Vec<(usize, String)> {
    let value::Value::Struct(fields) = g else {
        return Vec::new();
    };
    let Some(i) = fields.iter().position(|(k, _)| k == text::OUTPUT_NAMES) else {
        return Vec::new();
    };
    let (_, v) = fields.remove(i);
    let value::Value::Seq(pairs) = v else {
        return Vec::new();
    };
    pairs
        .into_iter()
        .filter_map(|p| match p {
            value::Value::Seq(mut p) if p.len() == 2 => {
                let value::Value::Str(name) = p.pop()? else {
                    return None;
                };
                let index = match &p[0] {
                    value::Value::Num(n) => n.parse::<usize>().ok(),
                    value::Value::Int(i) => usize::try_from(*i).ok(),
                    _ => None,
                }?;
                Some((index, name))
            }
            _ => None,
        })
        .collect()
}

/// Take the symbolic attributes out of a parsed graph's ops (`pos: upper`,
/// `start: ?0-1`), leaving 0 in their place.
fn sym_attrs_of(
    g: &mut value::Value,
    dims: &[(String, u32)],
) -> Result<Vec<crate::dynamic::SymAttr>, String> {
    use value::{Payload, Value};
    let sym_of = |name: &str| -> Result<u32, String> {
        if let Some(n) = name.strip_prefix('?') {
            return n.parse().map_err(|_| format!("bad symbol `{name}`"));
        }
        dims.iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| *s)
            .ok_or_else(|| format!("dim `{name}` is not declared in the header's `dims`"))
    };
    let Value::Struct(fields) = g else {
        return Ok(Vec::new());
    };
    let Some((_, Value::Seq(nodes))) = fields.iter_mut().find(|(k, _)| k == "nodes") else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for node in nodes {
        let Value::Struct(nf) = node else { continue };
        let id = nf
            .iter()
            .find(|(k, _)| k == "id")
            .and_then(|(_, v)| match v {
                Value::Num(n) => n.parse::<u32>().ok(),
                _ => None,
            })
            .unwrap_or(0);
        let Some((_, op)) = nf.iter_mut().find(|(k, _)| k == "op") else {
            continue;
        };
        let mut found = Vec::new();
        let is_dim = |n: &str| dims.iter().any(|(d, _)| d == n);
        value::take_leaves(
            op,
            &mut |v| match v {
                Value::Variant(n, Payload::Tuple(_)) if n == text::SYM_ATTR => {
                    Some(Value::Num("0".into()))
                }
                Value::Variant(n, Payload::Unit) if is_dim(n) => Some(Value::Num("0".into())),
                _ => None,
            },
            &mut Vec::new(),
            &mut found,
        );
        for (path, marker) in found {
            let (name, scale, offset) = match marker {
                Value::Variant(n, Payload::Unit) => (n, 1, 0),
                Value::Variant(_, Payload::Tuple(p)) => match (&p[0], &p[1], p.get(2)) {
                    (Value::Str(n), Value::Int(o), None) => (n.clone(), 1, *o),
                    (Value::Str(n), Value::Int(o), Some(Value::Int(s))) => (n.clone(), *s, *o),
                    _ => continue,
                },
                _ => continue,
            };
            out.push(crate::dynamic::SymAttr {
                node: crate::NodeId(id),
                path,
                sym: sym_of(&name)?,
                scale: scale as i64,
                offset: offset as i64,
            });
        }
    }
    Ok(out)
}

fn at_error(at: &ParseError, message: String) -> RlxModelError {
    RlxModelError::Parse(ParseError {
        message,
        ..at.clone()
    })
}

/// Every graph in a `.rlxmodel` text.
pub fn read_graphs(src: &str) -> Result<Vec<Graph>, RlxModelError> {
    Ok(read_module(src)?.graphs)
}

/// The one graph in a `.rlxmodel` text.
pub fn from_text(src: &str) -> Result<Graph, RlxModelError> {
    let mut graphs = read_graphs(src)?;
    match graphs.len() {
        1 => Ok(graphs.remove(0)),
        n => Err(RlxModelError::Value(Error(format!(
            "expected one graph, found {n}"
        )))),
    }
}

/// Where two ops differ, if only in integer attributes: each attribute's
/// field path and its two values. For [`crate::dynamic::generalize`].
pub(crate) fn op_int_diffs(a: &crate::Op, b: &crate::Op) -> Option<Vec<(Vec<String>, i128, i128)>> {
    value::int_diffs(&value::to_value(a).ok()?, &value::to_value(b).ok()?)
}

/// `op` with the integer attribute at `path` set to `x`.
pub(crate) fn set_op_int(op: &crate::Op, path: &[String], x: i128) -> Result<crate::Op, String> {
    let mut v = value::to_value(op).map_err(|e| e.0)?;
    let leaf = value::at_mut(&mut v, path)
        .ok_or_else(|| format!("{op} has no attribute {}", path.join(".")))?;
    *leaf = value::Value::Int(x);
    value::from_value(v).map_err(|e| e.0)
}

/// `op` in the value syntax, if JSON cannot carry it: for
/// [`crate::text`], whose op payloads are JSON.
pub(crate) fn op_text_if_not_json(op: &crate::Op) -> Option<String> {
    let v = value::to_value(op).ok()?;
    value::has_non_finite(&v).then(|| {
        let mut out = String::new();
        text::value(&mut out, &v, 0);
        out
    })
}

/// An op written in the value syntax (`Clamp{min: -inf, max: 0}`).
pub(crate) fn op_from_value_text(src: &str) -> Result<crate::Op, String> {
    let mut p = text::Parser::new(src).map_err(|e| e.message)?;
    let v = p.value().map_err(|e| e.message)?;
    if !p.at_eof() {
        return Err(format!("unexpected text after the op in `{src}`"));
    }
    value::from_value(v).map_err(|e| e.0)
}

#[cfg(test)]
mod tests;
