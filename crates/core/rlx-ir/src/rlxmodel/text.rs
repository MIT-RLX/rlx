// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Printing and parsing `.rlxmodel` text over the [`Value`] tree. The graph
//! layout (`graph @name { %0 = Op … : f32[…] … return … }`) is fixed here;
//! everything inside an op is the generic value syntax.

use std::collections::HashMap;
use std::fmt::Write;

use super::value::{Payload, Value};

// ── printing ────────────────────────────────────────────────

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && c.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !KEYWORDS.contains(&s)
}

const KEYWORDS: &[&str] = &["none", "true", "false", "inf", "nan", "graph", "return"];

/// The marker of a symbolic attribute in the value tree: `(symbol name,
/// offset)`. Not a name any op can have.
pub(crate) const SYM_ATTR: &str = "?sym";
/// A derived dim in a shape, `seq*16+1` or `h*w*4`: `[base, scale,
/// offset]` or `[base, scale, offset, second base]`, each base a symbol's
/// name or number.
pub(crate) const DERIVED_DIM: &str = "?dim";

fn string(out: &mut String, s: &str) {
    // A multi-line text (a model card, a chat template) as a raw block:
    // `"""` and a newline, the text exactly, then `"""`.
    if s.contains('\n') && !s.contains("\"\"\"") && !s.contains('\r') {
        out.push_str("\"\"\"\n");
        out.push_str(s);
        out.push_str("\"\"\"");
        return;
    }
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{{{:x}}}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn float<F: std::fmt::Debug + Into<f64> + Copy>(out: &mut String, f: F, nan_bits: Option<String>) {
    let x: f64 = f.into();
    if x.is_nan() {
        // The canonical NaN is `nan`; any other (a sign, a payload) keeps
        // its exact bits.
        match nan_bits {
            Some(bits) => {
                let _ = write!(out, "nan({bits})");
            }
            None => out.push_str("nan"),
        }
    } else if x.is_infinite() {
        out.push_str(if x > 0.0 { "inf" } else { "-inf" });
    } else {
        // Debug: the shortest text that reads back to the same value.
        let _ = write!(out, "{f:?}");
    }
}

fn f32_nan_bits(f: f32) -> Option<String> {
    (f.to_bits() != f32::NAN.to_bits()).then(|| format!("0x{:08x}", f.to_bits()))
}

fn f64_nan_bits(f: f64) -> Option<String> {
    (f.to_bits() != f64::NAN.to_bits()).then(|| format!("0x{:016x}", f.to_bits()))
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64(out: &mut String, bytes: &[u8]) {
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
}

fn unbase64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let digits: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !digits.len().is_multiple_of(4) {
        return None;
    }
    for chunk in digits.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for &c in chunk {
            let d = match c {
                b'=' => {
                    pad += 1;
                    0
                }
                _ if pad > 0 => return None,
                _ => B64.iter().position(|&b| b == c)? as u32,
            };
            n = (n << 6) | d;
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&bytes[..3 - pad.min(2)]);
    }
    Some(out)
}

/// The parts of a serialized `Graph`: name, nodes, outputs, and the
/// `.rlxmodel` side fields (variables, output names).
type GraphParts<'a> = (
    &'a Value,
    &'a [Value],
    &'a [Value],
    &'a [Value],
    &'a [Value],
);

/// The fields of a serialized `Graph`, if `fields` is one.
fn graph_fields(fields: &[(String, Value)]) -> Option<GraphParts<'_>> {
    let [
        (n, name),
        (ns, Value::Seq(nodes)),
        (o, Value::Seq(outputs)),
        rest @ ..,
    ] = fields
    else {
        return None;
    };
    if n != "name" || ns != "nodes" || o != "outputs" {
        return None;
    }
    let (mut vars, mut names): (&[Value], &[Value]) = (&[], &[]);
    for (k, v) in rest {
        match (k.as_str(), v) {
            (VARS, Value::Seq(v)) => vars = v,
            (OUTPUT_NAMES, Value::Seq(v)) => names = v,
            _ => return None,
        }
    }
    Some((name, nodes, outputs, vars, names))
}

/// The graph field `.rlxmodel` carries output names in: `[output index,
/// name]` pairs.
pub(crate) const OUTPUT_NAMES: &str = "output_names";

/// The graph field `.rlxmodel` carries its variables in: `[input id,
/// output index]` pairs.
pub(crate) const VARS: &str = "vars";

fn var_pairs(vars: &[Value]) -> Vec<(String, usize)> {
    vars.iter()
        .filter_map(|p| match p {
            Value::Seq(p) if p.len() == 2 => {
                let out = match &p[1] {
                    Value::Int(i) => usize::try_from(*i).ok(),
                    Value::Num(n) => n.parse().ok(),
                    _ => None,
                }?;
                Some((text_of(&p[0]), out))
            }
            _ => None,
        })
        .collect()
}

/// `[output index, name]` pairs.
fn name_pairs(names: &[Value]) -> Vec<(usize, String)> {
    names
        .iter()
        .filter_map(|p| match p {
            Value::Seq(p) if p.len() == 2 => {
                let index = match &p[0] {
                    Value::Int(i) => usize::try_from(*i).ok(),
                    Value::Num(n) => n.parse().ok(),
                    _ => None,
                }?;
                let Value::Str(name) = &p[1] else { return None };
                Some((index, name.clone()))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn value(out: &mut String, v: &Value, indent: usize) {
    match v {
        Value::Unit => out.push_str("()"),
        Value::None => out.push_str("none"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Byte(b) => {
            let _ = write!(out, "{b}");
        }
        Value::F32(f) => float(out, *f, f32_nan_bits(*f)),
        Value::F64(f) => float(out, *f, f64_nan_bits(*f)),
        Value::Num(s) => out.push_str(s),
        Value::Str(s) => string(out, s),
        Value::Bytes(b) => {
            out.push_str("b64\"");
            base64(out, b);
            out.push('"');
        }
        Value::Seq(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                value(out, it, indent);
            }
            out.push(']');
        }
        Value::Map(entries) => {
            out.push_str("#{");
            for (i, (k, it)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                value(out, k, indent);
                out.push_str(": ");
                value(out, it, indent);
            }
            out.push('}');
        }
        Value::Struct(fields) => match graph_fields(fields) {
            Some((name, nodes, outputs, vars, names)) => {
                graph(out, name, nodes, outputs, vars, names, indent)
            }
            None => struct_fields(out, fields, indent),
        },
        // A symbolic attribute: `upper`, `seq-1`, `cap+2`, `seq*-1+1000`.
        Value::Variant(name, Payload::Tuple(p))
            if name == SYM_ATTR && (p.len() == 2 || p.len() == 3) =>
        {
            if let Value::Str(sym) = &p[0] {
                out.push_str(sym);
            }
            if let Some(Value::Int(s)) = p.get(2) {
                let _ = write!(out, "*{s}");
            }
            match &p[1] {
                Value::Int(0) => {}
                Value::Int(o) if *o > 0 => {
                    let _ = write!(out, "+{o}");
                }
                other => value(out, other, indent),
            }
        }
        Value::Variant(name, payload) => {
            out.push_str(name);
            match payload {
                Payload::Unit => {}
                Payload::Tuple(items) => {
                    out.push('(');
                    for (i, it) in items.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        value(out, it, indent);
                    }
                    out.push(')');
                }
                Payload::Struct(fields) => struct_fields(out, fields, indent),
            }
        }
    }
}

/// `{` one field per line `}`, for the model block.
pub(crate) fn struct_lines(out: &mut String, v: &Value) {
    let Value::Struct(fields) = v else {
        return value(out, v, 0);
    };
    // Who and what first, the long card last, everything else between.
    const FIRST: &[&str] = &[
        "name",
        "version",
        "authors",
        "license",
        "repository",
        "homepage",
    ];
    let rank = |k: &str| match FIRST.iter().position(|f| *f == k) {
        Some(i) => i,
        None if k == "card" => FIRST.len() + 1,
        None => FIRST.len(),
    };
    let mut fields: Vec<&(String, Value)> = fields.iter().collect();
    fields.sort_by_key(|(k, _)| rank(k));
    out.push_str("{\n");
    for (k, it) in fields {
        out.push_str("  ");
        if is_ident(k) {
            out.push_str(k);
        } else {
            string(out, k);
        }
        out.push_str(": ");
        pretty(out, it, 1);
        out.push_str(",\n");
    }
    out.push('}');
}

/// Lines this long or longer break a struct or list onto one line per field.
const LINE: usize = 100;

/// `v` on one line when that fits; a struct or list that does not, one
/// field or item per line, indented to `depth`.
fn pretty(out: &mut String, v: &Value, depth: usize) {
    let mut one = String::new();
    value(&mut one, v, depth);
    let fits = !one.contains('\n') && one.len() + 2 * depth < LINE;
    let pad = "  ".repeat(depth + 1);
    match v {
        Value::Struct(fields) if !fits && !fields.is_empty() => {
            out.push_str("{\n");
            for (k, it) in fields.iter().filter(|(_, x)| *x != Value::None) {
                out.push_str(&pad);
                if is_ident(k) {
                    out.push_str(k);
                } else {
                    string(out, k);
                }
                out.push_str(": ");
                pretty(out, it, depth + 1);
                out.push_str(",\n");
            }
            out.push_str(&"  ".repeat(depth));
            out.push('}');
        }
        Value::Seq(items) if !fits && !items.is_empty() => {
            out.push_str("[\n");
            for it in items {
                out.push_str(&pad);
                pretty(out, it, depth + 1);
                out.push_str(",\n");
            }
            out.push_str(&"  ".repeat(depth));
            out.push(']');
        }
        _ => out.push_str(&one),
    }
}

fn struct_fields(out: &mut String, fields: &[(String, Value)], indent: usize) {
    out.push('{');
    // An optional field that is `none` says nothing, and reads back as
    // `none` when it is left out.
    let fields: Vec<&(String, Value)> = fields.iter().filter(|(_, v)| *v != Value::None).collect();
    for (i, (k, it)) in fields.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        if is_ident(k) {
            out.push_str(k);
        } else {
            string(out, k);
        }
        out.push_str(": ");
        value(out, it, indent);
    }
    out.push('}');
}

fn field<'a>(fields: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn shape(out: &mut String, v: &Value) {
    let Value::Struct(fields) = v else {
        return value(out, v, 0);
    };
    match field(fields, "dtype") {
        Some(Value::Variant(d, Payload::Unit)) => out.push_str(&d.to_ascii_lowercase()),
        Some(other) => value(out, other, 0),
        None => {}
    }
    out.push('[');
    if let Some(Value::Seq(dims)) = field(fields, "dims") {
        for (i, d) in dims.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match d {
                Value::Variant(k, Payload::Tuple(n)) if k == "Static" && n.len() == 1 => {
                    value(out, &n[0], 0)
                }
                // A named symbol (`dims {seq: 0}` in the header) prints by name.
                Value::Variant(k, Payload::Tuple(n))
                    if k == "Dynamic" && matches!(n.as_slice(), [Value::Str(_)]) =>
                {
                    if let [Value::Str(name)] = n.as_slice() {
                        out.push_str(name);
                    }
                }
                Value::Variant(k, Payload::Tuple(n)) if k == "Dynamic" && n.len() == 1 => {
                    out.push('?');
                    value(out, &n[0], 0);
                }
                Value::Variant(k, Payload::Tuple(n))
                    if k == DERIVED_DIM && (3..=9).contains(&n.len()) =>
                {
                    let base = |out: &mut String, b: &Value| match b {
                        Value::Str(name) => out.push_str(name),
                        b => {
                            out.push('?');
                            value(out, b, 0);
                        }
                    };
                    // `(…)/div`: the part before the division in brackets,
                    // its offset `inner`; `n[2]` after.
                    let div = match n.get(6) {
                        Some(Value::Int(d)) if *d != 1 => Some(*d),
                        _ => None,
                    };
                    if div.is_some() {
                        out.push('(');
                    }
                    base(out, &n[0]);
                    if let Some(t) = n.get(3).filter(|t| !matches!(t, Value::Unit)) {
                        out.push('*');
                        base(out, t);
                    }
                    if let Value::Int(scale) = n[1]
                        && scale != 1
                    {
                        let _ = write!(out, "*{scale}");
                    }
                    // A square's linear term: `+seq`, `-seq*3`.
                    if let Some(Value::Int(l)) = n.get(4).filter(|l| !matches!(l, Value::Int(0))) {
                        out.push(if *l < 0 { '-' } else { '+' });
                        base(out, &n[0]);
                        if l.unsigned_abs() != 1 {
                            let _ = write!(out, "*{}", l.unsigned_abs());
                        }
                    }
                    let offset = |out: &mut String, o: &Value| match o {
                        Value::Int(o) if *o > 0 => {
                            let _ = write!(out, "+{o}");
                        }
                        Value::Int(o) if *o < 0 => {
                            let _ = write!(out, "{o}");
                        }
                        _ => {}
                    };
                    if let Some(d) = div {
                        if let Some(inner) = n.get(5) {
                            offset(out, inner);
                        }
                        let _ = write!(out, ")/{d}");
                    }
                    // Another symbol: `+duration*1`, `+frames*-2`.
                    if let (Some(q), Some(Value::Int(k))) = (n.get(7), n.get(8))
                        && !matches!(q, Value::Unit)
                    {
                        out.push('+');
                        base(out, q);
                        let _ = write!(out, "*{k}");
                    }
                    offset(out, &n[2]);
                }
                other => value(out, other, 0),
            }
        }
    }
    out.push(']');
}

/// The most elements a constant prints as values; a larger one stays bytes.
const CONST_MAX: usize = 256;

/// A constant's bytes as values of `dtype` (`const [0.0, 1.0]`), for the
/// dtypes that print exactly; `None` for the rest (half floats, complex),
/// and for data that is not `numel` elements (a stripped weight).
fn const_values(data: &[u8], dtype: &str, numel: usize) -> Option<Vec<String>> {
    let size = match dtype {
        "f64" | "i64" => 8,
        "f32" | "i32" | "u32" => 4,
        "i16" => 2,
        "i8" | "u8" | "bool" => 1,
        _ => return None,
    };
    if numel > CONST_MAX || data.len() != numel * size {
        return None;
    }
    Some(
        data.chunks_exact(size)
            .map(|c| {
                let mut out = String::new();
                match dtype {
                    "f32" => {
                        let f = f32::from_le_bytes(c.try_into().expect("4 bytes"));
                        float(&mut out, f, f32_nan_bits(f));
                    }
                    "f64" => {
                        let f = f64::from_le_bytes(c.try_into().expect("8 bytes"));
                        float(&mut out, f, f64_nan_bits(f));
                    }
                    "i64" => out = i64::from_le_bytes(c.try_into().expect("8 bytes")).to_string(),
                    "i32" => out = i32::from_le_bytes(c.try_into().expect("4 bytes")).to_string(),
                    "u32" => out = u32::from_le_bytes(c.try_into().expect("4 bytes")).to_string(),
                    "i16" => out = i16::from_le_bytes(c.try_into().expect("2 bytes")).to_string(),
                    "i8" => out = (c[0] as i8).to_string(),
                    "bool" if c[0] <= 1 => out = (c[0] == 1).to_string(),
                    _ => out = c[0].to_string(),
                }
                out
            })
            .collect(),
    )
}

/// `const` values as the bytes of `dtype`.
fn const_bytes(values: &[Value], dtype: &str) -> Result<Vec<u8>, String> {
    let num = |v: &Value| -> Result<String, String> {
        match v {
            Value::Num(n) => Ok(n.clone()),
            Value::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            other => Err(format!("a const holds numbers, found {other:?}")),
        }
    };
    let bad = |n: &str| format!("`{n}` is not a {dtype}");
    let mut out = Vec::new();
    for v in values {
        let n = num(v)?;
        match dtype {
            "f32" => out.extend(
                super::value::parse_f32_text(&n)
                    .map_err(|_| bad(&n))?
                    .to_le_bytes(),
            ),
            "f64" => out.extend(
                super::value::parse_f64_text(&n)
                    .map_err(|_| bad(&n))?
                    .to_le_bytes(),
            ),
            "i64" => out.extend(n.parse::<i64>().map_err(|_| bad(&n))?.to_le_bytes()),
            "i32" => out.extend(n.parse::<i32>().map_err(|_| bad(&n))?.to_le_bytes()),
            "u32" => out.extend(n.parse::<u32>().map_err(|_| bad(&n))?.to_le_bytes()),
            "i16" => out.extend(n.parse::<i16>().map_err(|_| bad(&n))?.to_le_bytes()),
            "i8" => out.extend(n.parse::<i8>().map_err(|_| bad(&n))?.to_le_bytes()),
            "u8" | "bool" => out.push(n.parse::<u8>().map_err(|_| bad(&n))?),
            other => {
                return Err(format!(
                    "a `const` of {other} is written as `Constant{{data: b64\"…\"}}`"
                ));
            }
        }
    }
    Ok(out)
}

/// A node name that can be its value's label (`%q_proj`, `%layers.0.k`):
/// starts with a letter or `_`, then letters, digits, `_` and `.`.
fn is_label(s: &str) -> bool {
    let mut c = s.chars();
    c.next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && c.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn text_of(v: &Value) -> String {
    let mut s = String::new();
    value(&mut s, v, 0);
    s
}

/// Each node's label, by its id's text: its name where that is a label no
/// other node has, else its number.
fn labels(nodes: &[Value]) -> HashMap<String, String> {
    let names: Vec<Option<&str>> = nodes
        .iter()
        .map(|n| match n {
            Value::Struct(f) => match field(f, "name") {
                Some(Value::Str(s)) if is_label(s) => Some(s.as_str()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let mut count: HashMap<&str, usize> = HashMap::new();
    for n in names.iter().flatten() {
        *count.entry(n).or_default() += 1;
    }
    nodes
        .iter()
        .zip(&names)
        .filter_map(|(n, name)| {
            let Value::Struct(f) = n else { return None };
            let id = text_of(field(f, "id")?);
            let label = match name {
                Some(n) if count[n] == 1 => (*n).to_string(),
                _ => id.clone(),
            };
            Some((id, label))
        })
        .collect()
}

fn graph(
    out: &mut String,
    name: &Value,
    nodes: &[Value],
    outputs: &[Value],
    vars: &[Value],
    names: &[Value],
    indent: usize,
) {
    let vars = var_pairs(vars);
    let names = name_pairs(names);
    let pad = "  ".repeat(indent + 1);
    out.push_str("graph @");
    match name {
        Value::Str(s) if is_ident(s) => out.push_str(s),
        other => value(out, other, indent),
    }
    out.push_str(" {\n");
    let labels = labels(nodes);
    let label_of = |v: &Value| {
        let id = text_of(v);
        labels.get(&id).cloned().unwrap_or(id)
    };
    for n in nodes {
        let Value::Struct(f) = n else { continue };
        out.push_str(&pad);
        let label = field(f, "id").map(label_of).unwrap_or_default();
        let named = !label.starts_with(|c: char| c.is_ascii_digit());
        let id_text = field(f, "id").map(text_of).unwrap_or_default();
        let is_var = vars.iter().any(|(i, _)| *i == id_text);
        let var_sugar = is_var
            && named
            && matches!(field(f, "op"), Some(Value::Variant(k, Payload::Struct(of)))
                if k == "Input" && matches!(of.as_slice(), [(n, Value::Str(v))] if n == "name" && *v == label));
        out.push('%');
        out.push_str(&label);
        out.push_str(" = ");
        // `%c = const [0.0, 1.0] : f32[2]`: a small constant as its values.
        let shape_of = field(f, "shape");
        let dtype = match shape_of {
            Some(Value::Struct(sf)) => match field(sf, "dtype") {
                Some(Value::Variant(d, Payload::Unit)) => d.to_ascii_lowercase(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        let numel = match shape_of {
            Some(Value::Struct(sf)) => match field(sf, "dims") {
                Some(Value::Seq(d)) => d.iter().try_fold(1usize, |acc, x| match x {
                    Value::Variant(k, Payload::Tuple(n)) if k == "Static" && n.len() == 1 => {
                        match &n[0] {
                            Value::Int(v) => Some(acc * (*v as usize)),
                            _ => None,
                        }
                    }
                    _ => None,
                }),
                _ => None,
            },
            _ => None,
        };
        let scalar = matches!(shape_of, Some(Value::Struct(sf)) if matches!(field(sf, "dims"), Some(Value::Seq(d)) if d.is_empty()));
        let consts = match (field(f, "op"), numel) {
            (Some(Value::Variant(k, Payload::Struct(of))), Some(numel)) if k == "Constant" => {
                match of.as_slice() {
                    [(d, Value::Bytes(b))] if d == "data" => const_values(b, &dtype, numel),
                    _ => None,
                }
            }
            _ => None,
        };
        match field(f, "op") {
            _ if consts.is_some() => {
                let vals = consts.unwrap_or_default();
                out.push_str("const ");
                if scalar && vals.len() == 1 {
                    out.push_str(&vals[0]);
                } else {
                    out.push('[');
                    out.push_str(&vals.join(", "));
                    out.push(']');
                }
            }
            // `%x = input`: an input or param named like its value; `var`
            // for an input that is a variable.
            Some(Value::Variant(k, Payload::Struct(of)))
                if named
                    && (k == "Input" || k == "Param")
                    && matches!(of.as_slice(), [(n, Value::Str(v))] if n == "name" && *v == label) =>
            {
                out.push_str(match k.as_str() {
                    "Param" => "param",
                    _ if is_var => "var",
                    _ => "input",
                });
            }
            Some(op) => value(out, op, indent + 1),
            None => {}
        }
        if let Some(Value::Seq(inputs)) = field(f, "inputs") {
            for (i, x) in inputs.iter().enumerate() {
                out.push_str(if i == 0 { " %" } else { ", %" });
                out.push_str(&label_of(x));
            }
        }
        out.push_str(" : ");
        if let Some(s) = field(f, "shape") {
            shape(out, s);
        }
        for extra in ["name", "origin"] {
            match field(f, extra) {
                None | Some(Value::None) => {}
                // A named value's name is its label.
                Some(_) if extra == "name" && named => {}
                Some(v) => {
                    let _ = write!(out, " {extra} ");
                    value(out, v, indent + 1);
                }
            }
        }
        if is_var && !var_sugar {
            out.push_str(" var");
        }
        out.push('\n');
    }
    // `%k = %k_next`: an output that is variable `%k`'s next value;
    // `%7 as logits`: a named output.
    let items: Vec<String> = outputs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut item = match vars.iter().find(|(_, index)| *index == i) {
                Some((var, _)) => {
                    let var_label = labels.get(var).cloned().unwrap_or_else(|| var.clone());
                    format!("%{var_label} = %{}", label_of(o))
                }
                None => format!("%{}", label_of(o)),
            };
            if let Some((_, name)) = names.iter().find(|(index, _)| *index == i) {
                item.push_str(" as ");
                if is_ident(name) {
                    item.push_str(name);
                } else {
                    string(&mut item, name);
                }
            }
            item
        })
        .collect();
    out.push_str(&pad);
    out.push_str("return");
    let one_line = items.join(", ");
    if pad.len() + 7 + one_line.len() < LINE {
        if !items.is_empty() {
            out.push(' ');
            out.push_str(&one_line);
        }
    } else {
        // One value per line.
        for (i, item) in items.iter().enumerate() {
            out.push('\n');
            out.push_str(&pad);
            out.push_str("  ");
            out.push_str(item);
            if i + 1 < items.len() {
                out.push(',');
            }
        }
    }
    out.push('\n');
    out.push_str(&"  ".repeat(indent));
    out.push('}');
}

// ── parsing ─────────────────────────────────────────────────

/// Where and why the text failed to parse.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Num(String),
    /// `%12`
    Node(String),
    /// `?3`
    Sym(String),
    Bytes(Vec<u8>),
    Punct(char),
    /// `#{`
    MapOpen,
    Eof,
}

#[derive(Clone)]
struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

pub(crate) struct Parser<'a> {
    lex: Lexer<'a>,
    tok: Tok,
    at: (usize, usize),
}

impl<'a> Lexer<'a> {
    fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            line: self.line,
            column: self.col,
            message: msg.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek()?;
        self.pos += 1;
        if c == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn skip_space(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => {
                    self.bump();
                }
                Some(b'/') if self.src.get(self.pos + 1) == Some(&b'/') => {
                    while self.peek().is_some_and(|c| c != b'\n') {
                        self.bump();
                    }
                }
                _ => return,
            }
        }
    }

    fn take_while(&mut self, f: impl Fn(u8) -> bool) -> String {
        let start = self.pos;
        while self.peek().is_some_and(&f) {
            self.bump();
        }
        String::from_utf8_lossy(&self.src[start..self.pos]).into_owned()
    }

    /// `"""` newline text `"""`: the text verbatim, no escapes.
    fn raw_string(&mut self) -> Result<String, ParseError> {
        for _ in 0..3 {
            self.bump();
        }
        if self.bump() != Some(b'\n') {
            return Err(self.err("a `\"\"\"` text starts on the next line"));
        }
        let start = self.pos;
        let len = self.src[start..]
            .windows(3)
            .position(|w| w == b"\"\"\"")
            .ok_or_else(|| self.err("unterminated `\"\"\"` text"))?;
        for _ in 0..len + 3 {
            self.bump();
        }
        String::from_utf8(self.src[start..start + len].to_vec())
            .map_err(|_| self.err("text is not UTF-8"))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.bump(); // opening quote
        let mut out = Vec::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => break,
                Some(b'\\') => match self.bump() {
                    Some(b'"') => out.push(b'"'),
                    Some(b'\\') => out.push(b'\\'),
                    Some(b'n') => out.push(b'\n'),
                    Some(b't') => out.push(b'\t'),
                    Some(b'r') => out.push(b'\r'),
                    Some(b'u') => {
                        if self.bump() != Some(b'{') {
                            return Err(self.err("expected `\\u{…}`"));
                        }
                        let hex = self.take_while(|c| c.is_ascii_hexdigit());
                        if self.bump() != Some(b'}') {
                            return Err(self.err("expected `}` after `\\u{…`"));
                        }
                        let c = u32::from_str_radix(&hex, 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or_else(|| self.err(format!("bad escape \\u{{{hex}}}")))?;
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return Err(self.err("unknown escape")),
                },
                Some(c) => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("string is not UTF-8"))
    }

    fn next(&mut self) -> Result<Tok, ParseError> {
        self.skip_space();
        let Some(c) = self.peek() else {
            return Ok(Tok::Eof);
        };
        Ok(match c {
            b'"' if self.src[self.pos..].starts_with(b"\"\"\"") => Tok::Str(self.raw_string()?),
            b'"' => Tok::Str(self.string()?),
            b'%' => {
                self.bump();
                Tok::Node(self.take_while(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.'))
            }
            b'?' => {
                self.bump();
                Tok::Sym(self.take_while(|c| c.is_ascii_digit()))
            }
            b'#' if self.src.get(self.pos + 1) == Some(&b'{') => {
                self.bump();
                self.bump();
                Tok::MapOpen
            }
            b'-' | b'0'..=b'9' => {
                let start = self.pos;
                self.bump();
                // A number, with `e-3` / `e+3` exponents, or `-inf`.
                while let Some(c) = self.peek() {
                    let exp =
                        matches!(c, b'-' | b'+') && matches!(self.src[self.pos - 1], b'e' | b'E');
                    if c.is_ascii_alphanumeric() || c == b'.' || exp {
                        self.bump();
                    } else {
                        break;
                    }
                }
                Tok::Num(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let id = self.take_while(|c| c.is_ascii_alphanumeric() || c == b'_');
                if id == "b64" && self.peek() == Some(b'"') {
                    let s = self.string()?;
                    Tok::Bytes(unbase64(&s).ok_or_else(|| self.err("bad base64"))?)
                } else {
                    Tok::Ident(id)
                }
            }
            c => {
                self.bump();
                Tok::Punct(c as char)
            }
        })
    }
}

impl<'a> Parser<'a> {
    pub(crate) fn new(src: &'a str) -> Result<Self, ParseError> {
        let mut lex = Lexer {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        };
        lex.skip_space();
        let at = (lex.line, lex.col);
        let tok = lex.next()?;
        Ok(Self { lex, tok, at })
    }

    pub(crate) fn err(&self, msg: impl Into<String>) -> ParseError {
        ParseError {
            line: self.at.0,
            column: self.at.1,
            message: msg.into(),
        }
    }

    fn advance(&mut self) -> Result<Tok, ParseError> {
        self.lex.skip_space();
        self.at = (self.lex.line, self.lex.col);
        let next = self.lex.next()?;
        Ok(std::mem::replace(&mut self.tok, next))
    }

    /// The token after the current one, not consumed.
    fn peek_tok(&self) -> Result<Tok, ParseError> {
        let mut lex = self.lex.clone();
        lex.skip_space();
        lex.next()
    }

    pub(crate) fn at_eof(&self) -> bool {
        self.tok == Tok::Eof
    }

    fn is_punct(&self, c: char) -> bool {
        self.tok == Tok::Punct(c)
    }

    fn eat_punct(&mut self, c: char) -> Result<bool, ParseError> {
        if self.is_punct(c) {
            self.advance()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn expect_punct(&mut self, c: char) -> Result<(), ParseError> {
        if self.eat_punct(c)? {
            Ok(())
        } else {
            Err(self.err(format!("expected `{c}`, found {}", self.describe())))
        }
    }

    pub(crate) fn is_ident(&self, word: &str) -> bool {
        matches!(&self.tok, Tok::Ident(w) if w == word)
    }

    pub(crate) fn expect_ident(&mut self, word: &str) -> Result<(), ParseError> {
        if self.is_ident(word) {
            self.advance()?;
            Ok(())
        } else {
            Err(self.err(format!("expected `{word}`, found {}", self.describe())))
        }
    }

    pub(crate) fn number(&mut self) -> Result<String, ParseError> {
        match self.advance()? {
            Tok::Num(n) => Ok(n),
            other => Err(self.err(format!("expected a number, found {}", show(&other)))),
        }
    }

    fn describe(&self) -> String {
        show(&self.tok)
    }

    /// A value's label: `%3` or `%q_proj`.
    fn label(&mut self) -> Result<String, ParseError> {
        match self.advance()? {
            Tok::Node(n) if !n.is_empty() => Ok(n),
            other => Err(self.err(format!("expected `%n` or `%name`, found {}", show(&other)))),
        }
    }

    /// A reference to a value defined above it.
    fn node_ref(
        &mut self,
        labels: &HashMap<String, usize>,
        defined: usize,
    ) -> Result<Value, ParseError> {
        let at = self.err("");
        let l = self.label()?;
        let id = if l.bytes().all(|c| c.is_ascii_digit()) {
            l.parse::<usize>().ok().filter(|&n| n < defined)
        } else {
            labels.get(&l).copied()
        };
        match id {
            Some(id) => Ok(Value::Num(id.to_string())),
            None => Err(ParseError {
                message: format!("%{l} is not a value defined above"),
                ..at
            }),
        }
    }

    /// A comma-separated list until `close`.
    fn list<T>(
        &mut self,
        close: char,
        mut item: impl FnMut(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        let mut out = Vec::new();
        while !self.is_punct(close) {
            out.push(item(self)?);
            if !self.eat_punct(',')? {
                break;
            }
        }
        self.expect_punct(close)?;
        Ok(out)
    }

    fn fields(&mut self) -> Result<Vec<(String, Value)>, ParseError> {
        self.list('}', |p| {
            let key = match p.advance()? {
                Tok::Ident(k) => k,
                Tok::Str(k) => k,
                other => {
                    return Err(p.err(format!("expected a field name, found {}", show(&other))));
                }
            };
            p.expect_punct(':')?;
            Ok((key, p.value()?))
        })
    }

    pub(crate) fn value(&mut self) -> Result<Value, ParseError> {
        if self.is_ident("graph") {
            return self.graph();
        }
        Ok(match self.advance()? {
            Tok::Num(n) => Value::Num(n),
            Tok::Str(s) => Value::Str(s),
            Tok::Bytes(b) => Value::Bytes(b),
            Tok::Ident(w) => match w.as_str() {
                "none" => Value::None,
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                "nan" if self.is_punct('(') => {
                    self.advance()?;
                    let bits = self.number()?;
                    self.expect_punct(')')?;
                    Value::Num(format!("nan({bits})"))
                }
                "inf" | "nan" => Value::Num(w),
                _ => {
                    if self.eat_punct('(')? {
                        Value::Variant(w, Payload::Tuple(self.list(')', Self::value)?))
                    } else if self.eat_punct('{')? {
                        Value::Variant(w, Payload::Struct(self.fields()?))
                    } else if self.is_punct('*') {
                        // `seq*-1+1000`: a scaled symbol.
                        self.advance()?;
                        let n = self.number()?;
                        let scale = n
                            .parse::<i128>()
                            .map_err(|_| self.err(format!("expected a scale, found {n}")))?;
                        let off = self.sym_offset()?.unwrap_or(0);
                        sym_attr(w, scale, off)
                    } else if let Some(off) = self.sym_offset()? {
                        sym_attr(w, 1, off)
                    } else {
                        Value::Variant(w, Payload::Unit)
                    }
                }
            },
            // `?3` / `?3+1`: an unnamed symbol as an attribute.
            Tok::Sym(n) if !n.is_empty() => {
                let off = self.sym_offset()?.unwrap_or(0);
                sym_attr(format!("?{n}"), 1, off)
            }
            Tok::Punct('[') => Value::Seq(self.list(']', Self::value)?),
            Tok::Punct('{') => Value::Struct(self.fields()?),
            Tok::Punct('(') => {
                self.expect_punct(')')?;
                Value::Unit
            }
            Tok::MapOpen => Value::Map(self.list('}', |p| {
                let k = p.value()?;
                p.expect_punct(':')?;
                Ok((k, p.value()?))
            })?),
            other => return Err(self.err(format!("expected a value, found {}", show(&other)))),
        })
    }

    /// `+n` or `-n` after a symbol's name.
    fn sym_offset(&mut self) -> Result<Option<i128>, ParseError> {
        if self.eat_punct('+')? {
            let n = self.number()?;
            return n
                .parse()
                .map(Some)
                .map_err(|_| self.err(format!("expected an offset, found {n}")));
        }
        if let Tok::Num(n) = &self.tok
            && n.starts_with('-')
        {
            let n = n.clone();
            self.advance()?;
            return n
                .parse()
                .map(Some)
                .map_err(|_| self.err(format!("expected an offset, found {n}")));
        }
        Ok(None)
    }

    fn shape(&mut self) -> Result<Value, ParseError> {
        let dtype = match self.advance()? {
            Tok::Ident(d) => d,
            other => return Err(self.err(format!("expected a dtype, found {}", show(&other)))),
        };
        self.expect_punct('[')?;
        let dims = self.list(']', Self::dim)?;
        Ok(Value::Struct(vec![
            ("dims".into(), Value::Seq(dims)),
            ("dtype".into(), Value::Variant(dtype, Payload::Unit)),
        ]))
    }

    /// `+name*k`: another symbol in a derived dim.
    fn plus_term(p: &mut Self) -> Result<Option<(Value, i128)>, ParseError> {
        if !p.is_punct('+') || !matches!(p.peek_tok()?, Tok::Ident(_) | Tok::Sym(_)) {
            return Ok(None);
        }
        p.advance()?;
        let q = match p.advance()? {
            Tok::Ident(n) => Value::Str(n),
            Tok::Sym(n) => Value::Num(n),
            t => return Err(p.err(format!("expected a dim, found {}", show(&t)))),
        };
        p.expect_punct('*')?;
        let n = p.number()?;
        let k = n
            .parse::<i128>()
            .map_err(|_| p.err(format!("expected a coefficient, found {n}")))?;
        Ok(Some((q, k)))
    }

    /// One dim of a shape: `768`, `seq`, `seq*16-16`, `h*w*4`,
    /// `seq*seq*2+seq-1`, `(audio)/256+1`.
    fn dim(p: &mut Self) -> Result<Value, ParseError> {
        // `(…)/div±offset`: a quotient, rounded down.
        if p.eat_punct('(')? {
            let at = p.err("");
            let parts = match Self::dim(p)? {
                Value::Variant(k, Payload::Tuple(b)) if k == "Dynamic" => {
                    vec![b[0].clone(), Value::Int(1), Value::Int(0)]
                }
                Value::Variant(k, Payload::Tuple(parts)) if k == DERIVED_DIM => parts,
                _ => {
                    return Err(ParseError {
                        message: "a quotient of a fixed size".into(),
                        ..at
                    });
                }
            };
            p.expect_punct(')')?;
            p.expect_punct('/')?;
            let n = p.number()?;
            let div = n
                .parse::<i128>()
                .map_err(|_| p.err(format!("expected a divisor, found {n}")))?;
            let plus = Self::plus_term(p)?;
            let outer = p.sym_offset()?.unwrap_or(0);
            let mut parts = parts;
            let inner = std::mem::replace(&mut parts[2], Value::Int(outer));
            if parts.len() < 4 {
                parts.push(Value::Unit);
            }
            if parts.len() < 5 {
                parts.push(Value::Int(0));
            }
            parts.push(inner);
            parts.push(Value::Int(div));
            if let Some((q, k)) = plus {
                parts.push(q);
                parts.push(Value::Int(k));
            }
            return Ok(Value::Variant(DERIVED_DIM.into(), Payload::Tuple(parts)));
        }
        let base = match p.advance()? {
            Tok::Num(n) => {
                return Ok(Value::Variant(
                    "Static".into(),
                    Payload::Tuple(vec![Value::Num(n)]),
                ));
            }
            Tok::Sym(s) if !s.is_empty() => Value::Num(s),
            // A symbol by name; the header's `dims` maps it to its number.
            Tok::Ident(name) => Value::Str(name),
            other => return Err(p.err(format!("expected a dim, found {}", show(&other)))),
        };
        // `seq*16-16`: a dim derived from the symbol; `h*w*4`, from
        // the product of two.
        let mut other = None;
        let mut scale = 1;
        while p.eat_punct('*')? {
            match p.advance()? {
                Tok::Num(n) => {
                    scale = n
                        .parse::<i128>()
                        .map_err(|_| p.err(format!("expected a scale, found {n}")))?;
                }
                Tok::Ident(name) if other.is_none() => other = Some(Value::Str(name)),
                Tok::Sym(s) if other.is_none() && !s.is_empty() => other = Some(Value::Num(s)),
                t => {
                    return Err(p.err(format!("expected a scale or a dim, found {}", show(&t))));
                }
            }
        }
        // A square's linear term (`seq*seq*2+seq-1`, `-seq*3`): the
        // base's name again, signed, and its coefficient.
        let mut linear = 0i128;
        let named = |v: &Value| match v {
            Value::Str(s) | Value::Num(s) => s.clone(),
            _ => String::new(),
        };
        let sign = match &p.tok {
            Tok::Punct('+') if other.is_some() => {
                let next = p.peek_tok()?;
                matches!(next, Tok::Ident(ref n) if *n == named(&base)).then_some(1)
            }
            Tok::Num(n) if other.is_some() && n[1..] == named(&base) && n.starts_with('-') => {
                Some(-1)
            }
            _ => None,
        };
        if let Some(sign) = sign {
            if sign > 0 {
                p.advance()?;
            }
            p.advance()?;
            linear = sign;
            if p.eat_punct('*')? {
                let n = p.number()?;
                linear *= n
                    .parse::<i128>()
                    .map_err(|_| p.err(format!("expected a coefficient, found {n}")))?;
            }
        }
        let plus = Self::plus_term(p)?;
        let offset = p.sym_offset()?.unwrap_or(0);
        if let Some((q, k)) = plus {
            let mut parts = vec![base, Value::Int(scale), Value::Int(offset)];
            parts.push(other.unwrap_or(Value::Unit));
            parts.push(Value::Int(linear));
            parts.extend([Value::Int(0), Value::Int(1), q, Value::Int(k)]);
            return Ok(Value::Variant(DERIVED_DIM.into(), Payload::Tuple(parts)));
        }
        Ok(if scale == 1 && offset == 0 && other.is_none() {
            Value::Variant("Dynamic".into(), Payload::Tuple(vec![base]))
        } else {
            let mut parts = vec![base, Value::Int(scale), Value::Int(offset)];
            parts.extend(other);
            if linear != 0 {
                parts.push(Value::Int(linear));
            }
            Value::Variant(DERIVED_DIM.into(), Payload::Tuple(parts))
        })
    }

    /// `graph @name { … }`, as the serialized `Graph` struct.
    pub(crate) fn graph(&mut self) -> Result<Value, ParseError> {
        self.expect_ident("graph")?;
        self.expect_punct('@')?;
        let name = match self.advance()? {
            Tok::Ident(n) | Tok::Str(n) => n,
            other => return Err(self.err(format!("expected a graph name, found {}", show(&other)))),
        };
        self.expect_punct('{')?;
        let mut nodes = Vec::new();
        // Each value's position, by label: node ids are positions.
        let mut labels: HashMap<String, usize> = HashMap::new();
        // Inputs declared `var`, by id text, and the `[id, output]` pairs.
        let mut var_inputs: Vec<String> = Vec::new();
        let mut vars: Vec<Value> = Vec::new();
        let mut output_names: Vec<Value> = Vec::new();
        let outputs = loop {
            if self.is_ident("return") {
                self.advance()?;
                let mut outs = Vec::new();
                while matches!(self.tok, Tok::Node(_)) {
                    let at = self.err("");
                    let first = self.node_ref(&labels, nodes.len())?;
                    // `%k = %k_next`: the output is variable `%k`'s next value.
                    if self.eat_punct('=')? {
                        let var = text_of(&first);
                        if !var_inputs.contains(&var) {
                            return Err(ParseError {
                                message: "the value before `=` in `return` is not a `var`".into(),
                                ..at
                            });
                        }
                        vars.push(Value::Seq(vec![first, Value::Num(outs.len().to_string())]));
                        outs.push(self.node_ref(&labels, nodes.len())?);
                    } else {
                        outs.push(first);
                    }
                    // `%7 as logits`: the output's name.
                    if self.is_ident("as") {
                        self.advance()?;
                        let name = match self.advance()? {
                            Tok::Ident(n) | Tok::Str(n) => n,
                            other => {
                                return Err(self.err(format!(
                                    "expected an output name after `as`, found {}",
                                    show(&other)
                                )));
                            }
                        };
                        output_names.push(Value::Seq(vec![
                            Value::Num((outs.len() - 1).to_string()),
                            Value::Str(name),
                        ]));
                    }
                    if !self.eat_punct(',')? {
                        break;
                    }
                }
                self.expect_punct('}')?;
                break outs;
            }
            let at = self.err("");
            let position = nodes.len();
            let label = self.label()?;
            let named = !label.bytes().all(|c| c.is_ascii_digit());
            if named {
                if labels.insert(label.clone(), position).is_some() {
                    return Err(ParseError {
                        message: format!("%{label} is defined twice"),
                        ..at
                    });
                }
            } else if label.parse::<usize>().ok() != Some(position) {
                return Err(ParseError {
                    message: format!(
                        "%{label} is value {position} here: number values in order, or name them"
                    ),
                    ..at
                });
            }
            self.expect_punct('=')?;
            // `const 0.0` / `const [0.0, 1.0]`: values, encoded once the
            // shape gives their dtype.
            let consts = if self.is_ident("const") {
                self.advance()?;
                Some(match self.value()? {
                    Value::Seq(items) => items,
                    one => vec![one],
                })
            } else {
                None
            };
            let mut op = if consts.is_some() {
                Value::Unit
            } else {
                self.value()?
            };
            // `%x = input` / `%w = param`: named like its value.
            if let Value::Variant(k, Payload::Unit) = &op
                && (k == "input" || k == "param" || k == "var")
            {
                if !named {
                    return Err(ParseError {
                        message: format!("`{k}` needs a named value: `%x = {k} : …`"),
                        ..at
                    });
                }
                if k == "var" {
                    var_inputs.push(position.to_string());
                }
                let variant = if k == "param" { "Param" } else { "Input" };
                op = Value::Variant(
                    variant.into(),
                    Payload::Struct(vec![("name".into(), Value::Str(label.clone()))]),
                );
            }
            let mut inputs = Vec::new();
            while matches!(self.tok, Tok::Node(_)) {
                inputs.push(self.node_ref(&labels, position)?);
                if !self.eat_punct(',')? {
                    break;
                }
            }
            self.expect_punct(':')?;
            let shape = self.shape()?;
            if let Some(values) = consts {
                let dtype = match &shape {
                    Value::Struct(sf) => match field(sf, "dtype") {
                        Some(Value::Variant(d, _)) => d.to_ascii_lowercase(),
                        _ => String::new(),
                    },
                    _ => String::new(),
                };
                let numel = match &shape {
                    Value::Struct(sf) => match field(sf, "dims") {
                        Some(Value::Seq(d)) => d.iter().try_fold(1usize, |acc, x| match x {
                            Value::Variant(k, Payload::Tuple(n)) if k == "Static" => match &n[0] {
                                Value::Num(v) => v.parse::<usize>().ok().map(|v| acc * v),
                                _ => None,
                            },
                            _ => None,
                        }),
                        _ => None,
                    },
                    _ => None,
                };
                if numel != Some(values.len()) {
                    return Err(ParseError {
                        message: format!(
                            "a const of {} values for a shape of {} elements",
                            values.len(),
                            numel.map_or("unknown".into(), |n| n.to_string())
                        ),
                        ..at
                    });
                }
                let data = const_bytes(&values, &dtype).map_err(|message| ParseError {
                    message,
                    ..at.clone()
                })?;
                op = Value::Variant(
                    "Constant".into(),
                    Payload::Struct(vec![("data".into(), Value::Bytes(data))]),
                );
            }
            let mut f = vec![
                ("id".into(), Value::Num(position.to_string())),
                ("op".into(), op),
                ("inputs".into(), Value::Seq(inputs)),
                ("shape".into(), shape),
            ];
            if named {
                f.push(("name".into(), Value::Str(label)));
            }
            for extra in ["name", "origin"] {
                if self.is_ident(extra) {
                    self.advance()?;
                    let v = self.value()?;
                    f.retain(|(k, _)| k != extra);
                    f.push((extra.into(), v));
                }
            }
            // A ` var` suffix: an input that is a variable.
            if self.is_ident("var") {
                self.advance()?;
                var_inputs.push(position.to_string());
            }
            nodes.push(Value::Struct(f));
        };
        let mut fields = vec![
            ("name".into(), Value::Str(name)),
            ("nodes".into(), Value::Seq(nodes)),
            ("outputs".into(), Value::Seq(outputs)),
        ];
        if !vars.is_empty() {
            fields.push((VARS.into(), Value::Seq(vars)));
        }
        if !output_names.is_empty() {
            fields.push((OUTPUT_NAMES.into(), Value::Seq(output_names)));
        }
        Ok(Value::Struct(fields))
    }
}

fn show(t: &Tok) -> String {
    match t {
        Tok::Ident(w) => format!("`{w}`"),
        Tok::Str(s) => format!("\"{s}\""),
        Tok::Num(n) => n.clone(),
        Tok::Node(n) => format!("%{n}"),
        Tok::Sym(s) => format!("?{s}"),
        Tok::Bytes(_) => "bytes".into(),
        Tok::Punct(c) => format!("`{c}`"),
        Tok::MapOpen => "`#{`".into(),
        Tok::Eof => "the end of the text".into(),
    }
}

/// A symbolic attribute's value: `name * scale + offset` (the scale left
/// out when it is 1).
pub(crate) fn sym_attr(name: String, scale: i128, offset: i128) -> Value {
    let mut parts = vec![Value::Str(name), Value::Int(offset)];
    if scale != 1 {
        parts.push(Value::Int(scale));
    }
    Value::Variant(SYM_ATTR.into(), Payload::Tuple(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_every_length() {
        for n in 0..10u8 {
            let bytes: Vec<u8> = (0..n).map(|i| i.wrapping_mul(97)).collect();
            let mut s = String::new();
            base64(&mut s, &bytes);
            assert_eq!(unbase64(&s).unwrap(), bytes, "{s}");
        }
    }
}
