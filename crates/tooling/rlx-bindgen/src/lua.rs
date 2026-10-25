// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Lua binding (Lua 5.4+), from the lowered [`Ir`]: two files.
//!
//! * `rlx-lua/src/rlx_core.c` — a C module with one function per rlx-capi
//!   function, driven by C types alone: handles are light userdata, arrays and
//!   string lists are Lua tables, and every out-parameter comes back as an
//!   extra return value (strings and `{ptr, len}` arrays copied and freed on
//!   the spot).
//! * `rlx-lua/rlx/generated.lua` — the API: a metatable class per handle type
//!   (freed by `__gc` and `<close>`), optional parameters in a trailing
//!   options table (`g:softmax(x, {axis = 1})`), JSON options and reports.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

// ── the C module ────────────────────────────────────────────

fn c_function(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let _ = writeln!(s, "static int l_{}(lua_State* L) {{", f.c);
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut args = Vec::new();
    let mut rets = Vec::new();
    let mut arg = 0usize; // Lua argument index (1-based) of the next input
    for (i, (t, out)) in params.iter().enumerate() {
        if *out {
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    args.push(format!("&o{i}"));
                    let push = match st {
                        "RlxF32s" => format!("rlx_lua_push_f32s(L, o{i});"),
                        "RlxBytes" => format!("rlx_lua_push_bytes(L, o{i});"),
                        _ => format!("rlx_lua_push_u32s(L, o{i});"),
                    };
                    rets.push(push);
                }
                "char**" => {
                    pre.push(format!("char* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("rlx_lua_push_string(L, o{i});"));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("lua_pushboolean(L, o{i});"));
                }
                "float*" => {
                    pre.push(format!("float o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("lua_pushnumber(L, o{i});"));
                }
                "uint32_t*" | "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("lua_pushinteger(L, (lua_Integer)o{i});"));
                }
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("rlx_lua_push_ptr(L, o{i});"));
                }
            }
            continue;
        }
        arg += 1;
        let a = arg;
        match t.as_str() {
            "const char*" => args.push(format!("luaL_optstring(L, {a}, NULL)")),
            "bool" => args.push(format!("lua_toboolean(L, {a})")),
            "float" => args.push(format!("(float)luaL_checknumber(L, {a})")),
            "double" => args.push(format!("luaL_checknumber(L, {a})")),
            "int32_t" | "uint32_t" | "int64_t" | "uint64_t" | "size_t" => {
                args.push(format!("({t})luaL_checkinteger(L, {a})"))
            }
            "const float*" => {
                pre.push(format!("float* c{i} = rlx_lua_floats(L, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const uint8_t*" => args.push(format!("(const uint8_t*)luaL_optstring(L, {a}, NULL)")),
            "const uint32_t*" => {
                pre.push(format!("uint32_t* c{i} = rlx_lua_u32s(L, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const int64_t*" => {
                pre.push(format!("int64_t* c{i} = rlx_lua_i64s(L, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const size_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                pre.push(format!("{ct}* c{i} = ({ct}*)rlx_lua_sizes(L, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const char* const*" => {
                pre.push(format!("const char** c{i} = rlx_lua_strings(L, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free((void*)c{i});"));
            }
            "const float* const*" => {
                pre.push(format!("RlxLuaArrays c{i} = rlx_lua_float_arrays(L, {a});"));
                args.push(format!("(const float* const*)c{i}.ptrs"));
                post.push(format!("rlx_lua_arrays_free(&c{i});"));
            }
            "const uint8_t* const*" => {
                pre.push(format!(
                    "const uint8_t** c{i} = (const uint8_t**)rlx_lua_strings(L, {a});"
                ));
                args.push(format!("c{i}"));
                post.push(format!("free((void*)c{i});"));
            }
            handle => args.push(format!("({handle})rlx_lua_ptr(L, {a})")),
        }
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let call = format!("{}({})", f.c, args.join(", "));
    if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => {
                let _ = writeln!(s, "    lua_pushstring(L, {call});");
            }
            _ => {
                let _ = writeln!(s, "    lua_pushboolean(L, {call});");
            }
        }
        for line in post.iter().rev() {
            let _ = writeln!(s, "    {line}");
        }
        let _ = writeln!(s, "    return 1;\n}}\n");
        return;
    }
    let _ = writeln!(s, "    int32_t rc = {call};");
    for line in post.iter().rev() {
        let _ = writeln!(s, "    {line}");
    }
    let _ = writeln!(s, "    lua_pushinteger(L, rc);");
    let _ = writeln!(s, "    if (rc != RLX_OK) return 1;");
    for r in &rets {
        let _ = writeln!(s, "    {r}");
    }
    let _ = writeln!(s, "    return {};\n}}\n", rets.len() + 1);
}

/// `rlx_core.c`.
pub fn c_module(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* The `rlx_core` Lua C module: one function per rlx-capi function, each\n\
         \x20* returning the status and then its results. Build with rlx_lua_support.c.\n\
         \x20*/\n\n\
         #include \"rlx_lua.h\"\n\n",
    );
    let mut names = Vec::new();
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let gate = f
            .feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")));
        if let Some(g) = &gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        c_function(&mut s, ir, f);
        if gate.is_some() {
            s.push_str("#endif\n\n");
        }
        names.push((f.c.clone(), gate));
    }
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "static int l_{free}(lua_State* L) {{\n    {free}((Rlx{n}*)rlx_lua_ptr(L, 1));\n    return 0;\n}}\n",
            free = c.c_free(),
            n = c.name
        );
        names.push((c.c_free(), None));
    }
    s.push_str("static const luaL_Reg RLX_CORE[] = {\n");
    for (n, gate) in &names {
        if let Some(g) = gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        let _ = writeln!(s, "    {{\"{n}\", l_{n}}},");
        if gate.is_some() {
            s.push_str("#endif\n");
        }
    }
    s.push_str(
        "    {NULL, NULL},\n};\n\n\
         int luaopen_rlx_core(lua_State* L) {\n\
         \x20   luaL_newlib(L, RLX_CORE);\n\
         \x20   rlx_lua_register_support(L);\n\
         \x20   return 1;\n\
         }\n",
    );
    s
}

// ── the Lua API ─────────────────────────────────────────────

const KEYWORDS: &[&str] = &[
    "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto", "if", "in",
    "local", "nil", "not", "or", "repeat", "return", "then", "true", "until", "while",
];

fn local(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "nil".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "math.huge"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Lua expressions for one parameter's C-module arguments.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "json" | "value" => vec![format!("S.json.encode({v})")],
        "json?" | "value?" => vec![format!("{v} ~= nil and S.json.encode({v}) or nil")],
        "node?" => vec![format!("{v} or S.NONE")],
        "u64?" | "usize?" | "f32?" => vec![format!("{v} ~= nil and {{{v}}} or nil")],
        "f32s" | "u32s" | "nodes" | "usizes" | "i64s" => vec![v.into(), format!("#{v}")],
        "u8s" => vec![v.into(), format!("#{v}")],
        "usizes?" => vec![v.into(), format!("{v} and #{v} or 0")],
        "rows2" => vec![format!("S.rows({v}, 2, true)"), format!("#{v}")],
        "frows2" => vec![format!("S.rows({v}, 2, false)"), format!("#{v}")],
        "frows6" => vec![format!("S.rows({v}, 6, false)"), format!("#{v}")],
        "strs" => vec![v.into(), format!("#{v}")],
        "strs?" => vec![v.into(), format!("{v} and #{v} or 0")],
        "inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.n"),
            ]
        }
        "typed_inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.dtypes"),
                format!("{n}.n"),
            ]
        }
        "graph?" => vec![format!("{v} and {v}.handle or nil")],
        "str" | "str?" | "bool" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize"
        | "node" => vec![v.into()],
        _ => vec![format!("{v}.handle")],
    }
}

/// How many Lua values a result kind comes back as, and the conversion.
fn convert(kind: &str, vals: &[String]) -> String {
    let v = &vals[0];
    match kind {
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            format!("S.json.decode({v})")
        }
        "outputs" => format!("S.take_outputs({v})"),
        "typed_outputs" => format!("S.take_typed_outputs({v})"),
        "tensors" => format!("S.take_tensors({v})"),
        "f32?" => format!("({} and {v} or nil)", vals[1]),
        "graph" => format!("S.wrap(M.Graph, {v})"),
        k if k.starts_with(char::is_uppercase) => format!("S.wrap(M.{k}, {v})"),
        _ => v.clone(),
    }
}

fn width(kind: &str) -> usize {
    if kind == "f32?" { 2 } else { 1 }
}

fn func(s: &mut String, ir: &Ir, f: &Func) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let mut params: Vec<String> = required.iter().map(|p| local(&p.name)).collect();
    if !optional.is_empty() {
        params.push("opts".into());
    }
    s.push_str(&doc_lines(&f.doc, "--- "));
    let head = match (f.role, &class) {
        (Role::Constructor, Some(c)) => format!("function M.{c}.new({})", params.join(", ")),
        (Role::Static, Some(c)) => {
            format!("function M.{c}.{}({})", local(&f.name), params.join(", "))
        }
        (Role::Method, Some(c)) => {
            format!("function M.{c}:{}({})", local(&f.name), params.join(", "))
        }
        _ => format!("function M.{}({})", local(&f.name), params.join(", ")),
    };
    let _ = writeln!(s, "{head}");
    if !optional.is_empty() {
        let _ = writeln!(s, "  opts = opts or {{}}");
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("self.handle".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            match &p.default {
                Some(d) => format!("(opts.{0} == nil and {1} or opts.{0})", p.name, literal(d)),
                None => format!("opts.{}", p.name),
            }
        } else {
            local(&p.name)
        };
        match p.kind.as_str() {
            "inputs" => {
                let _ = writeln!(s, "  local in_{} = S.inputs({v})", p.name);
            }
            "typed_inputs" => {
                let _ = writeln!(s, "  local in_{} = S.typed_inputs({v})", p.name);
            }
            _ => {}
        }
        args.extend(lower(p, &v));
    }
    let call = format!("core.{}({})", f.c, args.join(", "));
    if f.direct {
        let _ = writeln!(s, "  return {call}\nend\n");
        return;
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let mut names = vec!["rc".to_string()];
    let mut groups = Vec::new();
    for (i, (_, k)) in rets.iter().enumerate() {
        let vals: Vec<String> = (0..width(k)).map(|j| format!("r{i}_{j}")).collect();
        names.extend(vals.iter().cloned());
        groups.push((k.clone(), vals));
    }
    let _ = writeln!(s, "  local {} = {call}", names.join(", "));
    let _ = writeln!(s, "  S.check(rc)");
    match (f.role, groups.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "  return S.wrap(M.{}, r0_0)", class.unwrap());
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "  return {}", convert(&groups[0].0, &groups[0].1));
        }
        _ => {
            let parts: Vec<String> = rets
                .iter()
                .zip(&groups)
                .map(|((n, _), (k, vals))| format!("{n} = {}", convert(k, vals)))
                .collect();
            let _ = writeln!(s, "  return {{ {} }}", parts.join(", "));
        }
    }
    s.push_str("end\n\n");
}

/// `rlx/generated.lua`.
pub fn lua_api(ir: &Ir) -> String {
    let mut s = String::from(
        "-- RLX — versatile ML compiler + runtime.\n\
         -- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         -- SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         -- @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         -- do not edit; regenerate with `just gen-bindings`. Loaded by rlx.lua.\n\
         \n\
         return function(M, core, S)\n\n",
    );
    for c in &ir.classes {
        s.push_str(&doc_lines(&c.doc, "--- "));
        let _ = writeln!(
            s,
            "M.{n} = S.class(\"{n}\", \"{free}\")\n",
            n = c.name,
            free = c.c_free()
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        func(&mut s, ir, f);
    }
    s.push_str("end\n");
    s
}
