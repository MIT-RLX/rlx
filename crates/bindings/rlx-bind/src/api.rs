// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runtime API table (`api.idl`) as data.
//!
//! The op table ([`crate::manifest()`]) covers the graph builders; this covers
//! everything else a binding exposes — sessions, devices, autodiff, weights,
//! training, text, distributed nodes — one [`Entry`] per class member or free
//! function, named in every language. `rlx-bindgen` checks it against the C
//! ABI and generates Swift from it; rlx-js and pyrlx test their exports
//! against [`contract`] in both directions. The grammar is at the top of
//! `api.idl`.

use std::sync::OnceLock;

use crate::manifest::camel;

const SOURCE: &str = include_str!("api.idl");

/// What an entry is to its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A free function.
    Function,
    /// A class's constructor (`new`).
    Constructor,
    /// A static method.
    Static,
    /// An instance method.
    Method,
}

/// One argument: its name, kind (see `api.idl`) and default literal.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub kind: String,
    pub default: Option<String>,
}

/// What an entry returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Returns {
    Unit,
    One(String),
    /// Named values, in order.
    Tuple(Vec<(String, String)>),
}

impl Returns {
    /// Each returned kind, in order.
    pub fn kinds(&self) -> Vec<&str> {
        match self {
            Returns::Unit => Vec::new(),
            Returns::One(k) => vec![k.as_str()],
            Returns::Tuple(items) => items.iter().map(|(_, k)| k.as_str()).collect(),
        }
    }
}

/// An entry's name in one language: `None` when that language does not have
/// it (and [`Entry::why`] says why).
#[derive(Clone, Debug, Default)]
pub struct Names {
    /// Path from the `pyrlx` module root (`"Session.compile"`, `"load_tokenizer"`).
    pub py: Option<String>,
    /// Path from the `rlx` global (`"Session.compile"`, `"openGguf"`).
    pub js: Option<String>,
    /// The C function.
    pub c: Option<String>,
    /// The Swift member (`"compile"`, `"init"`) or free function.
    pub swift: Option<String>,
}

/// A handle type.
#[derive(Clone, Debug)]
pub struct Class {
    /// Canonical (and Swift) name, e.g. `"Session"`.
    pub name: String,
    pub doc: String,
    /// The class in Python / JS, `None` when absent.
    pub py: Option<String>,
    pub js: Option<String>,
    /// The C stem: `rlx_<c>_new`, `rlx_<c>_free`, opaque `Rlx<Name>`.
    pub c: String,
    pub feature: Option<String>,
    pub why: Option<String>,
    /// Its Swift class is hand-written (`Graph`, which carries the generated
    /// builders).
    pub manual: bool,
}

impl Class {
    /// The opaque C type, `Rlx<Name>`.
    pub fn c_type(&self) -> String {
        format!("Rlx{}", self.name)
    }

    /// The C function that frees a handle.
    pub fn c_free(&self) -> String {
        format!("rlx_{}_free", self.c)
    }
}

/// One class member or free function.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The owning class (index into [`Api::classes`]), `None` for a function.
    pub class: Option<usize>,
    /// Canonical snake_case name.
    pub name: String,
    pub role: Role,
    pub doc: String,
    pub params: Vec<Param>,
    pub returns: Returns,
    pub names: Names,
    /// Further Python / JS spellings of the same entry (full paths).
    pub py_also: Vec<String>,
    pub js_also: Vec<String>,
    /// The rlx-capi feature it needs (its own, else its class's).
    pub feature: Option<String>,
    /// Hand-written in Swift (callbacks, in-place buffers).
    pub manual: bool,
    /// Why a language lacks it.
    pub why: Option<String>,
}

/// Names a binding exports that are not entries, and why.
#[derive(Clone, Debug)]
pub struct Host {
    pub lang: String,
    pub names: Vec<String>,
    pub why: String,
}

/// The whole table.
#[derive(Clone, Debug)]
pub struct Api {
    pub classes: Vec<Class>,
    pub entries: Vec<Entry>,
    pub hosts: Vec<Host>,
}

impl Api {
    /// The owning class of `e`.
    pub fn class_of(&self, e: &Entry) -> Option<&Class> {
        e.class.map(|i| &self.classes[i])
    }

    /// Every entry of class `name`.
    pub fn members<'a>(&'a self, class: &'a Class) -> impl Iterator<Item = &'a Entry> {
        let idx = self.classes.iter().position(|c| c.name == class.name);
        self.entries.iter().filter(move |e| e.class == idx)
    }

    /// Host-only names for `lang` (`"py"` or `"js"`).
    pub fn host(&self, lang: &str) -> Vec<&str> {
        self.hosts
            .iter()
            .filter(|h| h.lang == lang)
            .flat_map(|h| h.names.iter().map(String::as_str))
            .collect()
    }
}

/// The table, parsed once.
pub fn api() -> &'static Api {
    static API: OnceLock<Api> = OnceLock::new();
    API.get_or_init(|| parse(SOURCE).unwrap_or_else(|e| panic!("rlx-bind api.idl: {e}")))
}

/// Every name `lang` (`"py"` or `"js"`) must export, given which rlx-capi
/// features its build has: `(path, also)` per entry — the main path first,
/// then its extra spellings. Classes the language has come first, as their
/// bare paths.
pub fn contract(lang: &str, enabled: impl Fn(&str) -> bool) -> Vec<String> {
    let a = api();
    let on = |f: &Option<String>| f.as_deref().is_none_or(&enabled);
    let mut out = Vec::new();
    for c in &a.classes {
        let name = if lang == "py" { &c.py } else { &c.js };
        if let Some(n) = name
            && on(&c.feature)
        {
            out.push(n.clone());
        }
    }
    for e in &a.entries {
        if !on(&e.feature) {
            continue;
        }
        let (main, also) = match lang {
            "py" => (&e.names.py, &e.py_also),
            _ => (&e.names.js, &e.js_also),
        };
        out.extend(main.iter().cloned());
        out.extend(also.iter().cloned());
    }
    out.sort();
    out.dedup();
    out
}

// ── parsing ─────────────────────────────────────────────────

fn snake_to_camel_upper(s: &str) -> String {
    let c = camel(s);
    let mut chars = c.chars();
    match chars.next() {
        Some(f) => f.to_uppercase().chain(chars).collect(),
        None => c,
    }
}

fn class_stem(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

/// `key=value` / `key="quoted value"` / bare `flag` tokens.
fn attrs(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.peek().is_none() {
            return Ok(out);
        }
        let mut key = String::new();
        while let Some(&c) = chars.peek() {
            if c == '=' || c.is_whitespace() {
                break;
            }
            key.push(c);
            chars.next();
        }
        if chars.peek() != Some(&'=') {
            out.push((key, String::new()));
            continue;
        }
        chars.next();
        let mut value = String::new();
        if chars.peek() == Some(&'"') {
            chars.next();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some(c) => value.push(c),
                    None => return Err(format!("unterminated quote after {key}=")),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                value.push(c);
                chars.next();
            }
        }
        out.push((key, value));
    }
}

/// Split `text` on top-level commas (none of the kinds nest, but tuples do).
fn split_top(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut quoted = false;
    for c in text.chars() {
        match c {
            '"' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            ',' if depth == 0 && !quoted => {
                out.push(std::mem::take(&mut cur).trim().to_string());
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn params(text: &str) -> Result<Vec<Param>, String> {
    split_top(text)
        .into_iter()
        .map(|p| {
            let (name, rest) = p
                .split_once(':')
                .ok_or_else(|| format!("argument `{p}` needs `name: kind`"))?;
            let (kind, default) = match rest.split_once('=') {
                Some((k, d)) => (k.trim(), Some(d.trim().to_string())),
                None => (rest.trim(), None),
            };
            Ok(Param {
                name: name.trim().to_string(),
                kind: kind.to_string(),
                default,
            })
        })
        .collect()
}

fn returns(text: &str) -> Result<Returns, String> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(Returns::Unit);
    }
    if let Some(inner) = t.strip_prefix('(').and_then(|r| r.strip_suffix(')')) {
        let items = params(inner)?
            .into_iter()
            .map(|p| (p.name, p.kind))
            .collect();
        return Ok(Returns::Tuple(items));
    }
    Ok(Returns::One(t.to_string()))
}

/// `name(args) -> ret [attrs]` → its parts.
fn signature(line: &str) -> Result<(String, Vec<Param>, Returns, Vec<(String, String)>), String> {
    let open = line.find('(').ok_or("expected `(`")?;
    let name = line[..open].trim().to_string();
    let mut depth = 0;
    let mut close = None;
    for (i, c) in line[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close.ok_or("unbalanced `(`")?;
    let args = params(&line[open + 1..close])?;
    let mut rest = line[close + 1..].trim();
    let mut at = Vec::new();
    if rest.ends_with(']') {
        let start = rest.rfind('[').ok_or("unbalanced `]`")?;
        at = attrs(&rest[start + 1..rest.len() - 1])?;
        rest = rest[..start].trim();
    }
    let ret = match rest.strip_prefix("->") {
        Some(r) => returns(r)?,
        None if rest.is_empty() => Returns::Unit,
        None => return Err(format!("unexpected `{rest}`")),
    };
    Ok((name, args, ret, at))
}

fn lang_name(value: &str) -> Option<String> {
    (value != "-").then(|| value.to_string())
}

fn list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parse `api.idl` text.
pub fn parse(src: &str) -> Result<Api, String> {
    let mut api = Api {
        classes: Vec::new(),
        entries: Vec::new(),
        hosts: Vec::new(),
    };
    let mut doc: Vec<String> = Vec::new();
    let mut class: Option<usize> = None;
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let lineno = i + 1;
        let line = lines[i].trim();
        i += 1;
        let at = |e: String| format!("line {lineno}: {e}");
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(d) = line.strip_prefix("///") {
            doc.push(d.strip_prefix(' ').unwrap_or(d).to_string());
            continue;
        }
        let take_doc = |doc: &mut Vec<String>| std::mem::take(doc).join("\n");
        if line == "end" {
            if class.take().is_none() {
                return Err(at("`end` without a class".into()));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("host ") {
            let (lang, names) = rest
                .split_once(':')
                .ok_or_else(|| at("host needs `lang:`".into()))?;
            let why_line = lines.get(i).map(|l| l.trim()).unwrap_or("");
            let why = attrs(why_line)
                .map_err(at)?
                .into_iter()
                .find(|(k, _)| k == "why")
                .map(|(_, v)| v)
                .ok_or_else(|| at("host needs a `why=` line".into()))?;
            i += 1;
            api.hosts.push(Host {
                lang: lang.trim().to_string(),
                names: names.split_whitespace().map(str::to_string).collect(),
                why,
            });
            continue;
        }
        if let Some(rest) = line.strip_prefix("class ") {
            if class.is_some() {
                return Err(at("class inside a class (missing `end`)".into()));
            }
            let (name, at_text) = match rest.split_once('[') {
                Some((n, a)) => (n.trim(), a.trim_end_matches(']')),
                None => (rest.trim(), ""),
            };
            let mut c = Class {
                name: name.to_string(),
                doc: take_doc(&mut doc),
                py: Some(name.to_string()),
                js: Some(name.to_string()),
                c: class_stem(name),
                feature: None,
                why: None,
                manual: false,
            };
            for (k, v) in attrs(at_text).map_err(at)? {
                match k.as_str() {
                    "manual" => c.manual = true,
                    "py" => c.py = lang_name(&v),
                    "js" => c.js = lang_name(&v),
                    "c" => c.c = v,
                    "feature" => c.feature = Some(v),
                    "why" => c.why = Some(v),
                    other => return Err(at(format!("unknown class attribute `{other}`"))),
                }
            }
            if (c.py.is_none() || c.js.is_none()) && c.why.is_none() {
                return Err(at(format!("class {name}: a missing language needs why=")));
            }
            api.classes.push(c);
            class = Some(api.classes.len() - 1);
            continue;
        }
        let (role, sig) = if let Some(rest) = line.strip_prefix("fn ") {
            if class.is_some() {
                return Err(at("`fn` inside a class".into()));
            }
            (Role::Function, rest)
        } else if class.is_none() {
            return Err(at(format!("unexpected `{line}` outside a class")));
        } else if let Some(rest) = line.strip_prefix("static ") {
            (Role::Static, rest)
        } else if line.starts_with("new(") {
            (Role::Constructor, line)
        } else {
            (Role::Method, line)
        };
        let (name, params, ret, attr) = signature(sig).map_err(at)?;
        let owner = class.map(|ci| api.classes[ci].clone());
        let names = default_names(owner.as_ref(), &name, role);
        let mut e = Entry {
            class,
            name,
            role,
            doc: take_doc(&mut doc),
            params,
            returns: ret,
            names,
            py_also: Vec::new(),
            js_also: Vec::new(),
            feature: owner.as_ref().and_then(|c| c.feature.clone()),
            manual: false,
            why: None,
        };
        for (k, v) in attr {
            match k.as_str() {
                "py" => e.names.py = lang_name(&v),
                "js" => e.names.js = lang_name(&v),
                "c" => e.names.c = lang_name(&v),
                "swift" => e.names.swift = lang_name(&v),
                "py_also" => e.py_also = list(&v),
                "js_also" => e.js_also = list(&v),
                "feature" => e.feature = Some(v),
                "manual" => e.manual = true,
                "why" => e.why = Some(v),
                other => return Err(at(format!("unknown attribute `{other}`"))),
            }
        }
        // A class a language lacks takes its members with it.
        if let Some(c) = &owner {
            if c.py.is_none() {
                e.names.py = None;
            }
            if c.js.is_none() {
                e.names.js = None;
            }
            if e.why.is_none() {
                e.why = c.why.clone();
            }
        }
        let names = &e.names;
        if (names.py.is_none() || names.js.is_none() || names.c.is_none() || names.swift.is_none())
            && e.why.is_none()
        {
            return Err(at(format!("{}: a missing language needs why=", e.name)));
        }
        api.entries.push(e);
    }
    if class.is_some() {
        return Err("a class is missing its `end`".into());
    }
    Ok(api)
}

fn default_names(owner: Option<&Class>, name: &str, role: Role) -> Names {
    let Some(c) = owner else {
        return Names {
            py: Some(name.to_string()),
            js: Some(camel(name)),
            c: Some(format!("rlx_{name}")),
            swift: Some(camel(name)),
        };
    };
    let member = |cls: &Option<String>, member: &str| cls.as_ref().map(|k| format!("{k}.{member}"));
    match role {
        Role::Constructor => Names {
            py: c.py.clone(),
            js: c.js.clone(),
            c: Some(format!("rlx_{}_new", c.c)),
            swift: Some("init".into()),
        },
        _ => Names {
            py: member(&c.py, name),
            js: member(&c.js, &camel(name)),
            c: Some(format!("rlx_{}_{name}", c.c)),
            swift: Some(camel(name)),
        },
    }
}

/// `"Session"` → `"Session"`, `"gguf_writer"` → `"GgufWriter"`.
pub fn type_name(snake: &str) -> String {
    snake_to_camel_upper(snake)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_parses_and_every_gap_is_explained() {
        let a = api();
        assert!(a.classes.iter().any(|c| c.name == "Session"));
        assert!(a.entries.len() > 100, "{} entries", a.entries.len());
        for e in &a.entries {
            let n = &e.names;
            if n.py.is_none() || n.js.is_none() || n.c.is_none() || n.swift.is_none() {
                assert!(e.why.is_some(), "{} lacks why", e.name);
            }
        }
    }

    #[test]
    fn default_names_follow_each_language() {
        let a = api();
        let find = |class: &str, name: &str| {
            a.entries
                .iter()
                .find(|e| a.class_of(e).map(|c| c.name.as_str()) == Some(class) && e.name == name)
                .unwrap()
        };
        let e = find("Compiled", "set_param_typed");
        assert_eq!(e.names.py.as_deref(), Some("CompiledGraph.set_param_typed"));
        assert_eq!(e.names.js.as_deref(), Some("Compiled.setParamTyped"));
        assert_eq!(e.names.c.as_deref(), Some("rlx_compiled_set_param_typed"));
        assert_eq!(e.names.swift.as_deref(), Some("setParamTyped"));
        let w = find("GgufWriter", "add_f32");
        assert_eq!(w.names.c.as_deref(), Some("rlx_gguf_writer_add_f32"));
        assert_eq!(w.names.py, None);
        let ctor = find("Session", "new");
        assert_eq!(ctor.names.py.as_deref(), Some("Session"));
        assert_eq!(ctor.names.swift.as_deref(), Some("init"));
        let tup = find("Runner", "run");
        assert_eq!(tup.returns.kinds(), vec!["str", "outputs"]);
    }

    #[test]
    fn contract_follows_features() {
        let all = contract("py", |_| true);
        let none = contract("py", |_| false);
        assert!(all.contains(&"GgufFile.tensor_info".to_string()));
        assert!(!none.contains(&"GgufFile.tensor_info".to_string()));
        assert!(none.contains(&"Session.compile_with".to_string()));
    }
}
