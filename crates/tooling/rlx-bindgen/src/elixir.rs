// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Elixir binding, from the lowered [`Ir`]: two files.
//!
//! * `rlx-elixir/c_src/rlx_nif.c` — one NIF per rlx-capi function (a dirty
//!   I/O NIF, since a call may block — a mesh rank — or take long), driven by
//!   C types alone: handles are resources (freed by the class's C free
//!   function), strings binaries (`nil` is none), arrays lists. A call
//!   returns `{:ok, result}` or `{:error, kind, message}`.
//! * `rlx-elixir/lib/rlx/generated.ex` — the API: a module per handle class
//!   (`Rlx.Graph.matmul(g, a, b)`), free functions in `Rlx`, optional
//!   parameters as a keyword list with the tables' defaults
//!   (`Rlx.Graph.softmax(g, x, axis: 1)`); failures raise `Rlx.Error`.
//!   Options and reports are maps (JSON).

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

// ── the NIF ─────────────────────────────────────────────────

fn class_of_type(ir: &Ir, c: &str) -> Option<String> {
    let name = c.strip_prefix("Rlx")?.trim_end_matches('*');
    ir.classes
        .iter()
        .find(|k| k.name == name)
        .map(|k| k.c.clone())
}

fn nif(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let _ = writeln!(
        s,
        "static ERL_NIF_TERM n_{}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {{\n    (void)argc;",
        f.c
    );
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut call = Vec::new();
    let mut rets = Vec::new();
    let mut a = 0usize;
    for (i, (t, out)) in params.iter().enumerate() {
        if *out {
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    call.push(format!("&o{i}"));
                    let take = match st {
                        "RlxF32s" => "rlx_ex_f32s",
                        "RlxBytes" => "rlx_ex_bytes",
                        _ => "rlx_ex_u32s",
                    };
                    rets.push(format!("{take}(env, o{i})"));
                }
                "char**" => {
                    pre.push(format!("char* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ex_string(env, o{i})"));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("enif_make_atom(env, o{i} ? \"true\" : \"false\")"));
                }
                "float*" => {
                    pre.push(format!("float o{i} = 0;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("enif_make_double(env, o{i})"));
                }
                "uint32_t*" | "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("enif_make_uint64(env, (ErlNifUInt64)o{i})"));
                }
                "RlxOutputs**" => {
                    pre.push(format!("RlxOutputs* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ex_outputs(env, o{i})"));
                }
                "RlxTypedOutputs**" => {
                    pre.push(format!("RlxTypedOutputs* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ex_typed_outputs(env, o{i})"));
                }
                "RlxTensorMap**" => {
                    pre.push(format!("RlxTensorMap* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    rets.push(format!("rlx_ex_tensors(env, o{i})"));
                }
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    call.push(format!("&o{i}"));
                    let c = class_of_type(ir, base).expect("an out handle is a class");
                    rets.push(format!("rlx_ex_handle(env, RES_{c}, o{i})"));
                }
            }
            continue;
        }
        let v = format!("argv[{a}]");
        a += 1;
        let need = |kind: &str| {
            format!(
                "if (!c{i}.ok) return rlx_ex_badarg(env, {a0}, \"{kind}\");",
                a0 = a - 1
            )
        };
        match t.as_str() {
            "const char*" => {
                pre.push(format!("RlxExStr c{i} = rlx_ex_cstr(env, {v});"));
                pre.push(need("a string or nil"));
                call.push(format!("c{i}.s"));
                post.push(format!("free(c{i}.s);"));
            }
            "bool" => call.push(format!("rlx_ex_bool(env, {v})")),
            "float" => call.push(format!("(float)rlx_ex_double(env, {v})")),
            "double" => call.push(format!("rlx_ex_double(env, {v})")),
            "int32_t" | "int64_t" => call.push(format!("({t})rlx_ex_i64(env, {v})")),
            "uint32_t" | "uint64_t" | "size_t" => call.push(format!("({t})rlx_ex_u64(env, {v})")),
            "const float*" => {
                pre.push(format!("RlxExArr c{i} = rlx_ex_floats(env, {v});"));
                pre.push(need("a list of numbers"));
                call.push(format!("(const float*)c{i}.p"));
                post.push(format!("free(c{i}.p);"));
            }
            "const uint8_t*" => {
                pre.push(format!("ErlNifBinary c{i}; if (!enif_inspect_binary(env, {v}, &c{i})) return rlx_ex_badarg(env, {}, \"a binary\");", a - 1));
                call.push(format!("(c{i}.size ? c{i}.data : NULL)"));
            }
            "const uint32_t*" | "const int64_t*" | "const size_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                let f = match ct {
                    "uint32_t" => "rlx_ex_u32s_in",
                    "int64_t" => "rlx_ex_i64s_in",
                    _ => "rlx_ex_u64s_in",
                };
                pre.push(format!("RlxExArr c{i} = {f}(env, {v});"));
                pre.push(need("a list of integers"));
                call.push(format!("({ct}*)c{i}.p"));
                post.push(format!("free(c{i}.p);"));
            }
            "const char* const*" => {
                pre.push(format!("RlxExList c{i} = rlx_ex_strings(env, {v});"));
                pre.push(need("a list of strings"));
                call.push(format!("(const char* const*)c{i}.ptrs"));
                post.push(format!("rlx_ex_list_free(&c{i});"));
            }
            "const float* const*" => {
                pre.push(format!("RlxExList c{i} = rlx_ex_float_lists(env, {v});"));
                pre.push(need("a list of number lists"));
                call.push(format!("(const float* const*)c{i}.ptrs"));
                post.push(format!("rlx_ex_list_free(&c{i});"));
            }
            "const uint8_t* const*" => {
                pre.push(format!("RlxExList c{i} = rlx_ex_binaries(env, {v});"));
                pre.push(need("a list of binaries"));
                call.push(format!("(const uint8_t* const*)c{i}.ptrs"));
                post.push(format!("rlx_ex_list_free(&c{i});"));
            }
            handle => {
                let c = class_of_type(ir, handle).expect("a handle argument is a class");
                pre.push(format!("void* c{i} = NULL;"));
                pre.push(format!(
                    "if (!rlx_ex_ptr(env, RES_{c}, {v}, &c{i})) return rlx_ex_badarg(env, {}, \"an rlx {c} or nil\");",
                    a - 1
                ));
                call.push(format!("({handle})c{i}"));
            }
        }
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let c = format!("{}({})", f.c, call.join(", "));
    if f.direct {
        let wrap = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => format!("rlx_ex_cstring(env, {c})"),
            _ => format!("enif_make_atom(env, {c} ? \"true\" : \"false\")"),
        };
        let _ = writeln!(s, "    ERL_NIF_TERM r = rlx_ex_ok(env, {wrap});");
        for line in post.iter().rev() {
            let _ = writeln!(s, "    {line}");
        }
        let _ = writeln!(s, "    return r;\n}}\n");
        return;
    }
    let _ = writeln!(s, "    int32_t rc = {c};");
    for line in post.iter().rev() {
        let _ = writeln!(s, "    {line}");
    }
    let _ = writeln!(s, "    if (rc != RLX_OK) return rlx_ex_error(env, rc);");
    let result = match rets.len() {
        0 => "enif_make_atom(env, \"nil\")".to_string(),
        1 => rets[0].clone(),
        k => format!("enif_make_tuple{k}(env, {})", rets.join(", ")),
    };
    let _ = writeln!(s, "    return rlx_ex_ok(env, {result});\n}}\n");
}

/// `c_src/rlx_nif.c`.
pub fn nif_source(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* The Elixir NIFs (Elixir.Rlx.Native): one per rlx-capi function.\n\
         \x20* Conversions are in rlx_nif_support.c.\n\
         \x20*/\n\n\
         #include \"rlx_nif.h\"\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "ErlNifResourceType* RES_{c};\nstatic void dtor_{c}(ErlNifEnv* env, void* obj) {{\n    (void)env;\n    void** p = (void**)obj;\n    if (*p) {{ {free}((Rlx{n}*)*p); *p = NULL; }}\n}}\n\
             static ERL_NIF_TERM n_{free}(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {{\n    (void)argc;\n    void** p = NULL;\n    if (enif_get_resource(env, argv[0], RES_{c}, (void**)&p) && *p) {{ {free}((Rlx{n}*)*p); *p = NULL; }}\n    return enif_make_atom(env, \"ok\");\n}}\n",
            c = c.c,
            free = c.c_free(),
            n = c.name
        );
    }
    let mut table = Vec::new();
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let gate = f
            .feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")));
        if let Some(g) = &gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        nif(&mut s, ir, f);
        if gate.is_some() {
            s.push_str("#endif\n\n");
        }
        let n = c_params(ir, f).iter().filter(|(_, o)| !*o).count();
        let dirty = if f.direct {
            "0"
        } else {
            "ERL_NIF_DIRTY_JOB_IO_BOUND"
        };
        table.push((format!("{{\"{c}\", {n}, n_{c}, {dirty}}}", c = f.c), gate));
    }
    for c in &ir.classes {
        table.push((
            format!("{{\"{free}\", 1, n_{free}, 0}}", free = c.c_free()),
            None,
        ));
    }
    s.push_str("static ErlNifFunc FUNCS[] = {\n");
    for (row, gate) in &table {
        if let Some(g) = gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        let _ = writeln!(s, "    {row},");
        if gate.is_some() {
            s.push_str("#endif\n");
        }
    }
    s.push_str("    RLX_EX_SUPPORT_FUNCS\n};\n\nstatic int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM info) {\n    (void)priv;\n    (void)info;\n    int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;\n");
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "    RES_{c} = enif_open_resource_type(env, NULL, \"rlx_{c}\", dtor_{c}, flags, NULL);\n    if (!RES_{c}) return 1;",
            c = c.c
        );
    }
    s.push_str("    return rlx_ex_load(env);\n}\n\nERL_NIF_INIT(Elixir.Rlx.Native, FUNCS, load, NULL, NULL, NULL)\n");
    s
}

// ── the Elixir API ──────────────────────────────────────────

const RESERVED: &[&str] = &[
    "do", "end", "fn", "nil", "true", "false", "when", "and", "or", "not", "in", "catch", "rescue",
    "after", "else",
];

fn ident(n: &str) -> String {
    if RESERVED.contains(&n) {
        format!("{n}_")
    } else {
        n.to_string()
    }
}

/// Elixir's Kernel functions a module function by that name would clash
/// with when called unqualified inside the module.
fn fun_name(f: &Func) -> String {
    match f.name.as_str() {
        "div" | "rem" | "abs" | "max" | "min" | "round" | "trunc" | "floor" | "ceil" | "elem"
        | "exit" | "self" | "send" | "spawn" | "apply" | "length" | "not" | "raise" | "throw"
        | "to_string" | "inspect" | "binding" => format!("{}_", f.name),
        n => ident(n),
    }
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "nil".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) if n.contains("inf") => {
            if n.starts_with('-') {
                ":neg_infinity".into()
            } else {
                ":infinity".into()
            }
        }
        // Elixir floats need a decimal point: 1.0e-5.
        Lit::Num(n) if n.contains('e') && !n.contains('.') => n.replacen('e', ".0e", 1),
        Lit::Num(n) => {
            if matches!(kind, "f32" | "f64" | "f32?") && !n.contains('.') {
                format!("{n}.0")
            } else {
                n.clone()
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) if kind == "u8s" && items.is_empty() => "\"\"".into(),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(|i| literal(i, "f32")).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// Elixir expressions for one parameter's NIF arguments.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "json" | "value" => vec![format!("S.json({v})")],
        "json?" | "value?" => vec![format!("S.json_or_nil({v})")],
        "node?" => vec![format!("({v} || S.none())")],
        "u64?" | "usize?" | "f32?" => vec![format!("S.box({v})")],
        "f32s" | "u32s" | "nodes" | "usizes" | "i64s" | "strs" => {
            vec![v.into(), format!("length({v})")]
        }
        "u8s" => vec![v.into(), format!("byte_size({v})")],
        "usizes?" | "strs?" => vec![format!("({v} || [])"), format!("length({v} || [])")],
        "rows2" => vec![format!("S.rows({v}, 2, true)"), format!("length({v})")],
        "frows2" => vec![format!("S.rows({v}, 2, false)"), format!("length({v})")],
        "frows6" => vec![format!("S.rows({v}, 6, false)"), format!("length({v})")],
        "inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("length({n}.names)"),
            ]
        }
        "typed_inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names"),
                format!("{n}.data"),
                format!("{n}.lens"),
                format!("{n}.dtypes"),
                format!("length({n}.names)"),
            ]
        }
        _ => vec![v.into()],
    }
}

fn convert(kind: &str, v: &str) -> String {
    match kind {
        "str" => format!("({v} || \"\")"),
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => format!("S.decode({v})"),
        "f32?" => format!("S.optional({v})"),
        _ => v.into(),
    }
}

fn width(kind: &str) -> usize {
    if kind == "f32?" { 2 } else { 1 }
}

fn func(s: &mut String, ir: &Ir, f: &Func) {
    let name = match f.role {
        Role::Constructor => "new".to_string(),
        _ => fun_name(f),
    };
    let mut ps: Vec<String> = Vec::new();
    if f.role == Role::Method {
        ps.push("this".into());
    }
    ps.extend(
        f.params
            .iter()
            .filter(|p| !p.optional())
            .map(|p| ident(&p.name)),
    );
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    if !optional.is_empty() {
        ps.push("opts \\\\ []".into());
    }
    let doc = doc_lines(&f.doc, "");
    if !doc.trim().is_empty() {
        let _ = writeln!(
            s,
            "  @doc \"\"\"\n{}  \"\"\"",
            doc.lines().map(|l| format!("  {l}\n")).collect::<String>()
        );
    }
    if f.manual {
        // Hand-written in lib/rlx/manual.ex.
        let class = ir.class_of(f).unwrap().name.to_ascii_lowercase();
        let args = match (class.as_str(), f.name.as_str()) {
            ("optimizer", "step") => "this, params, grads",
            ("trainer", "run") => "this, steps, batch, opts \\\\ []",
            _ => "this, args",
        };
        let _ = writeln!(
            s,
            "  defdelegate {}({args}), to: Rlx.Manual, as: :{class}_{}\n",
            f.name, f.name
        );
        return;
    }
    let _ = writeln!(s, "  def {name}({}) do", ps.join(", "));
    if !optional.is_empty() {
        let defaults: Vec<String> = optional
            .iter()
            .map(|p| {
                let d = p
                    .default
                    .as_ref()
                    .map_or("nil".into(), |l| literal(l, &p.kind));
                format!("{}: {d}", p.name)
            })
            .collect();
        let _ = writeln!(s, "    o = S.opts(opts, [{}])", defaults.join(", "));
    }
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("this".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("o.{}", p.name)
        } else {
            ident(&p.name)
        };
        match p.kind.as_str() {
            "inputs" => {
                let _ = writeln!(s, "    in_{} = S.inputs({v})", p.name);
            }
            "typed_inputs" => {
                let _ = writeln!(s, "    in_{} = S.typed_inputs({v})", p.name);
            }
            _ => {}
        }
        args.extend(lower(p, &v));
    }
    let call = format!("S.unwrap(Rlx.Native.{}({}))", f.c, args.join(", "));
    if f.direct || f.role == Role::Constructor {
        let _ = writeln!(s, "    {call}\n  end\n");
        return;
    }
    match f.rets.as_slice() {
        [] => {
            let _ = writeln!(s, "    {call}\n    :ok\n  end\n");
        }
        [(_, k)] if width(k) == 1 => {
            let _ = writeln!(s, "    {}\n  end\n", convert(k, &call));
        }
        rs => {
            let mut names = Vec::new();
            let mut conv = Vec::new();
            for (i, (n, k)) in rs.iter().enumerate() {
                if width(k) == 2 {
                    names.push(format!("r{i}"));
                    names.push(format!("r{i}k"));
                    conv.push(format!("{n}: S.optional({{r{i}, r{i}k}})"));
                } else {
                    names.push(format!("r{i}"));
                    conv.push(format!("{n}: {}", convert(k, &format!("r{i}"))));
                }
            }
            if rs.len() == 1 {
                let _ = writeln!(
                    s,
                    "    {{{}}} = {call}\n    {}\n  end\n",
                    names.join(", "),
                    conv[0].split_once(": ").unwrap().1
                );
            } else {
                let _ = writeln!(
                    s,
                    "    {{{}}} = {call}\n    %{{{}}}\n  end\n",
                    names.join(", "),
                    conv.join(", ")
                );
            }
        }
    }
}

/// `lib/rlx/generated.ex`.
pub fn api(ir: &Ir) -> String {
    let mut s = String::from(
        "# RLX — versatile ML compiler + runtime.\n\
         # Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         # SPDX-License-Identifier: MIT OR Apache-2.0\n\
         #\n\
         # @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         # do not edit; regenerate with `just gen-bindings`.\n\n\
         defmodule Rlx.Native do\n\
         \x20 @moduledoc false\n\
         \x20 @on_load :load\n\n\
         \x20 # load_nif must be called from the module that owns the NIFs.\n\x20 def load, do: :erlang.load_nif(Rlx.Support.nif_path(), 0)\n\n",
    );
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let n = c_params(ir, f).iter().filter(|(_, o)| !*o).count();
        let args: Vec<String> = (0..n).map(|i| format!("_a{i}")).collect();
        let _ = writeln!(
            s,
            "  def {}({}), do: :erlang.nif_error(:not_loaded)",
            f.c,
            args.join(", ")
        );
    }
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "  def {}(_h), do: :erlang.nif_error(:not_loaded)",
            c.c_free()
        );
    }
    s.push_str(
        "  def support_optimizer_step(_h, _names, _params, _grads), do: :erlang.nif_error(:not_loaded)\n\
         end\n\n\
         defmodule Rlx do\n\
         \x20 @moduledoc \"rlx for Elixir: free functions; a module per class (`Rlx.Graph`, `Rlx.Session`, …).\"\n\
         \x20 alias Rlx.Support, as: S\n\n\
         \x20 @doc \"\\\"No node\\\", for an optional node argument.\"\n\
         \x20 def none, do: S.none()\n\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, ir, f);
    }
    s.truncate(s.trim_end().len());
    s.push_str("\nend\n\n");
    for c in &ir.classes {
        let _ = writeln!(s, "defmodule Rlx.{} do", c.name);
        let doc = doc_lines(&c.doc, "");
        let _ = writeln!(
            s,
            "  @moduledoc \"\"\"\n{}  \"\"\"",
            doc.lines().map(|l| format!("  {l}\n")).collect::<String>()
        );
        let _ = writeln!(
            s,
            "  alias Rlx.Support, as: S\n\n  @doc \"Free the handle now (it is unusable afterwards).\"\n  def close(this), do: Rlx.Native.{}(this)\n",
            c.c_free()
        );
        for f in ir.members(c) {
            func(&mut s, ir, f);
        }
        s.truncate(s.trim_end().len());
        s.push_str("\nend\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
