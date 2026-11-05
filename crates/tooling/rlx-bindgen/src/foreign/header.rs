//! A plain C header read into the foreign binding IR: the structures, the
//! callback types and the exported functions (with the comments before
//! them as their docs). A small reader for the C of a library's API
//! header, not a C parser: `typedef struct ... { ... } name, *pname;`,
//! `typedef int (Name)(...);` and `EXPORT ret name(params);` (the export
//! macro configured), with the configured macros defined.

/// A C type: its base name (typedefs of pointers resolved) and how many
/// pointers deep
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ty {
    pub base: String,
    pub ptr: u8,
    pub konst: bool,
}

impl Ty {
    pub fn is(&self, base: &str, ptr: u8) -> bool {
        self.base == base && self.ptr == ptr
    }
}

/// A named value: a parameter or a field
#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty: Ty,
    pub doc: String,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub name: String,
    pub doc: String,
    pub fields: Vec<Field>,
}

/// A callback type (`typedef int (Name)(...)`)
#[derive(Clone, Debug)]
pub struct Callback {
    pub name: String,
    pub doc: String,
    pub ret: Ty,
    pub params: Vec<Ty>,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub name: String,
    pub doc: String,
    pub ret: Ty,
    pub params: Vec<Field>,
}

#[derive(Clone, Debug, Default)]
pub struct Header {
    pub structs: Vec<Struct>,
    pub callbacks: Vec<Callback>,
    pub funcs: Vec<Func>,
}

/// `#ifdef`/`#ifndef` sections kept as if `defines` were defined and the
/// other macros not; the other preprocessor lines dropped
fn preprocess(text: &str, defines: &[String]) -> String {
    let mut out = String::new();
    // per open #if: whether its lines are kept
    let mut stack: Vec<bool> = Vec::new();
    for line in text.lines() {
        let t = line.trim_start();
        let keep = stack.iter().all(|&k| k);
        if let Some(rest) = t.strip_prefix("#ifdef") {
            stack.push(defines.iter().any(|d| d == rest.trim()));
        } else if let Some(rest) = t.strip_prefix("#ifndef") {
            stack.push(!defines.iter().any(|d| d == rest.trim()));
        } else if t.starts_with("#if") {
            stack.push(false);
        } else if t.starts_with("#else") {
            if let Some(k) = stack.last_mut() {
                *k = !*k;
            }
        } else if t.starts_with("#endif") {
            stack.pop();
        } else if t.starts_with('#') {
        } else if keep {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// The text with its comments taken out, and the comments with where they
/// ended in the stripped text
fn strip_comments(text: &str) -> (String, Vec<(usize, String)>) {
    let b: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut comments = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '/' && b.get(i + 1) == Some(&'*') {
            let start = i + 2;
            let mut j = start;
            while j + 1 < b.len() && !(b[j] == '*' && b[j + 1] == '/') {
                j += 1;
            }
            comments.push((out.len(), b[start..j.min(b.len())].iter().collect()));
            i = j + 2;
        } else if b[i] == '/' && b.get(i + 1) == Some(&'/') {
            let start = i + 2;
            let mut j = start;
            while j < b.len() && b[j] != '\n' {
                j += 1;
            }
            comments.push((out.len(), b[start..j].iter().collect()));
            i = j;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    (out, comments)
}

/// A comment as doc text: its lines trimmed
fn doc_text(c: &str) -> String {
    c.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The comment that ends right before `pos` (only white space between)
fn doc_before(text: &str, comments: &[(usize, String)], pos: usize) -> String {
    comments
        .iter()
        .rev()
        .find(|(end, _)| *end <= pos && text[*end..pos].trim().is_empty())
        .map(|(_, c)| doc_text(c))
        .unwrap_or_default()
}

/// The comment on the same line after `pos` (a field's)
fn doc_after(text: &str, comments: &[(usize, String)], pos: usize) -> String {
    comments
        .iter()
        .find(|(end, _)| *end >= pos && !text[pos..*end].contains('\n'))
        .map(|(_, c)| doc_text(c))
        .unwrap_or_default()
}

/// `type name` (or a bare type): the type and the name
fn split_decl(s: &str, pointer_typedefs: &[(String, String)]) -> (Ty, String) {
    let s = s.trim();
    let konst = s.starts_with("const ");
    let s = s.trim_start_matches("const ").replace('*', " * ");
    let mut words: Vec<&str> = s.split_whitespace().collect();
    let stars = words.iter().filter(|w| **w == "*").count() as u8;
    words.retain(|w| *w != "*" && *w != "struct");
    let (base, name) = match words.len() {
        0 => (String::new(), String::new()),
        1 => (words[0].to_string(), String::new()),
        _ => (
            words[..words.len() - 1].join(" "),
            words[words.len() - 1].to_string(),
        ),
    };
    // pvector_info = vector_info *
    let (base, extra) = match pointer_typedefs.iter().find(|(p, _)| *p == base) {
        Some((_, t)) => (t.clone(), 1),
        None => (base, 0),
    };
    let base = if base == "NG_BOOL" {
        "bool".to_string()
    } else {
        base
    };
    let base = if base == "ngcomplex" {
        "ngcomplex_t".to_string()
    } else {
        base
    };
    (
        Ty {
            base,
            ptr: stars + extra,
            konst,
        },
        name,
    )
}

/// The header read: `export` marks an exported function
pub fn read(text: &str, export: &str, defines: &[String]) -> Header {
    let text = preprocess(text, defines);
    let (text, comments) = strip_comments(&text);
    let mut h = Header::default();
    let mut ptr_typedefs: Vec<(String, String)> = Vec::new();
    // the structures: `struct ngcomplex {...};` and `typedef struct x {...} x, *px;`
    let mut i = 0;
    while let Some(k) = text[i..].find("struct ") {
        let at = i + k;
        let Some(open) = text[at..].find('{').map(|o| at + o) else {
            break;
        };
        let head = text[at..open].trim();
        // a declaration, not a use (`struct ngcomplex ngcomplex_t;`)
        if head.contains(';') || head.split_whitespace().count() > 2 {
            i = at + 7;
            continue;
        }
        let close = open + text[open..].find('}').unwrap();
        let semi = close + text[close..].find(';').unwrap();
        let tail = text[close + 1..semi].trim();
        let tag = head.trim_start_matches("struct").trim().to_string();
        let mut names = tail.split(',').map(str::trim).filter(|s| !s.is_empty());
        let name = names.next().map(str::to_string).unwrap_or_else(|| {
            if tag == "ngcomplex" {
                "ngcomplex_t".into()
            } else {
                tag.clone()
            }
        });
        for n in names {
            if let Some(p) = n.strip_prefix('*') {
                ptr_typedefs.push((p.trim().to_string(), name.clone()));
            }
        }
        let start = text[..at]
            .rfind("typedef")
            .filter(|&t| text[t..at].trim() == "typedef")
            .unwrap_or(at);
        let doc = doc_before(&text, &comments, start);
        let mut fields = Vec::new();
        let mut p = open + 1;
        for decl in text[open + 1..close].split(';') {
            let end = p + decl.len();
            if !decl.trim().is_empty() {
                let (ty, fname) = split_decl(decl, &ptr_typedefs);
                fields.push(Field {
                    name: fname,
                    ty,
                    doc: doc_after(&text, &comments, end),
                });
            }
            p = end + 1;
        }
        h.structs.push(Struct { name, doc, fields });
        i = semi;
    }
    // the callbacks: `typedef int (Name)(...);`
    let mut i = 0;
    while let Some(k) = text[i..].find("typedef") {
        let at = i + k;
        let semi = at + text[at..].find(';').unwrap();
        let decl = &text[at + 7..semi];
        if let (Some(a), Some(b)) = (decl.find('('), decl.find(')')) {
            if !decl[..a].contains('{')
                && decl[a + 1..b]
                    .trim()
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_')
            {
                let ret = split_decl(&decl[..a], &ptr_typedefs).0;
                let name = decl[a + 1..b].trim().to_string();
                let rest = &decl[b + 1..];
                let inner = &rest[rest.find('(').unwrap() + 1..rest.rfind(')').unwrap()];
                let params = inner
                    .split(',')
                    .map(|p| split_decl(p, &ptr_typedefs).0)
                    .collect();
                // the comment after it, or before it
                let after = doc_after(&text, &comments, semi + 1);
                let doc = if after.is_empty() {
                    doc_before(&text, &comments, at)
                } else {
                    after
                };
                h.callbacks.push(Callback {
                    name,
                    doc,
                    ret,
                    params,
                });
            }
        }
        i = semi;
    }
    // the functions: `EXPORT ret name(params);`
    let mut i = 0;
    while let Some(k) = text[i..].find(export) {
        let at = i + k;
        let semi = at + text[at..].find(';').unwrap();
        let decl = &text[at + export.len()..semi];
        let open = decl.find('(').unwrap();
        let (ret, name) = split_decl(&decl[..open], &ptr_typedefs);
        let inner = &decl[open + 1..decl.rfind(')').unwrap()];
        let mut params = Vec::new();
        if inner.trim() != "void" && !inner.trim().is_empty() {
            for (n, p) in inner.split(',').enumerate() {
                let (ty, pname) = split_decl(p, &ptr_typedefs);
                params.push(Field {
                    name: if pname.is_empty() {
                        format!("arg{n}")
                    } else {
                        pname
                    },
                    ty,
                    doc: String::new(),
                });
            }
        }
        h.funcs.push(Func {
            name,
            doc: doc_before(&text, &comments, at),
            ret,
            params,
        });
        i = semi;
    }
    h
}
