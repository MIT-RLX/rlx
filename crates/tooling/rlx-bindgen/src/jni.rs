// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-kotlin/src/main/c/rlx_jni.c`: a JNI function per rlx-capi function,
//! for Android (NDK) and the desktop JVM, from the lowered [`Ir`].
//!
//! Driven by C parameter types alone: handles cross as `jlong`, strings as
//! `jstring`, arrays as Java arrays (pointer + its own length argument, as in
//! C), lists of strings / arrays as `Object[]`, and every out-parameter as a
//! one- (or two-, for `{ptr, len}`) element Java array the Kotlin side reads.
//! `org.rlx.kt.Native` (generated in `kotlin.rs`) declares the matching
//! `external fun`s.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, c_args, c_outs};

/// The C parameter types of a function, in order, and whether each is a result.
pub fn c_params(ir: &Ir, f: &Func) -> Vec<(String, bool)> {
    let mut v = Vec::new();
    if f.role == Role::Method {
        v.push((format!("Rlx{}*", ir.class_of(f).unwrap().name), false));
    }
    for p in &f.params {
        v.extend(c_args(&p.kind).into_iter().map(|t| (t, false)));
    }
    if !f.direct {
        let rets: Vec<String> = match f.role {
            Role::Constructor => vec![ir.class_of(f).unwrap().name.clone()],
            _ => f.rets.iter().map(|(_, k)| k.clone()).collect(),
        };
        for k in rets {
            v.extend(c_outs(&k).into_iter().map(|t| (t, true)));
        }
    }
    v
}

/// The JNI type of a C parameter.
pub fn jni_ty(c: &str, out: bool) -> &'static str {
    if out {
        return match c {
            "bool*" => "jbooleanArray",
            "uint32_t*" => "jintArray",
            "float*" => "jfloatArray",
            _ => "jlongArray", // pointers, size_t, uint64_t, {ptr, len}
        };
    }
    match c {
        "const char*" => "jstring",
        "bool" => "jboolean",
        "float" => "jfloat",
        "double" => "jdouble",
        "int32_t" | "uint32_t" => "jint",
        "int64_t" | "uint64_t" | "size_t" => "jlong",
        "const float*" => "jfloatArray",
        "const uint8_t*" => "jbyteArray",
        "const uint32_t*" => "jintArray",
        "const int64_t*" | "const size_t*" | "const uint64_t*" => "jlongArray",
        "const char* const*" | "const float* const*" | "const uint8_t* const*" => "jobjectArray",
        _ => "jlong", // handles
    }
}

/// The Kotlin type of a JNI parameter, for `external fun`.
pub fn kotlin_ty(jni: &str) -> &'static str {
    match jni {
        "jstring" => "String?",
        "jboolean" => "Boolean",
        "jfloat" => "Float",
        "jdouble" => "Double",
        "jint" => "Int",
        "jlong" => "Long",
        "jfloatArray" => "FloatArray?",
        "jbyteArray" => "ByteArray?",
        "jintArray" => "IntArray?",
        "jlongArray" => "LongArray?",
        "jbooleanArray" => "BooleanArray?",
        _ => "Array<Any?>?",
    }
}

/// JNI name mangling: `_` → `_1`.
pub fn mangle(name: &str) -> String {
    name.replace('_', "_1")
}

fn function(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let ret = if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "jstring",
            _ => "jboolean",
        }
    } else {
        "jint"
    };
    let decl: Vec<String> = params
        .iter()
        .enumerate()
        .map(|(i, (t, out))| format!("{} a{i}", jni_ty(t, *out)))
        .collect();
    let _ = writeln!(
        s,
        "JNIEXPORT {ret} JNICALL Java_org_rlx_kt_Native_{}(JNIEnv* env, jclass cls{}{}) {{",
        mangle(&f.c),
        if decl.is_empty() { "" } else { ", " },
        decl.join(", ")
    );
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut args = Vec::new();
    for (i, (t, out)) in params.iter().enumerate() {
        let a = format!("a{i}");
        if *out {
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    args.push(format!("&o{i}"));
                    post.push(format!(
                        "if ({a}) {{ jlong v[2] = {{(jlong)(intptr_t)o{i}.ptr, (jlong)o{i}.len}}; (*env)->SetLongArrayRegion(env, {a}, 0, 2, v); }}"
                    ));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    args.push(format!("&o{i}"));
                    post.push(format!("if ({a}) {{ jboolean v = o{i}; (*env)->SetBooleanArrayRegion(env, {a}, 0, 1, &v); }}"));
                }
                "uint32_t*" => {
                    pre.push(format!("uint32_t o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    post.push(format!("if ({a}) {{ jint v = (jint)o{i}; (*env)->SetIntArrayRegion(env, {a}, 0, 1, &v); }}"));
                }
                "float*" => {
                    pre.push(format!("float o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    post.push(format!("if ({a}) {{ jfloat v = o{i}; (*env)->SetFloatArrayRegion(env, {a}, 0, 1, &v); }}"));
                }
                "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    post.push(format!("if ({a}) {{ jlong v = (jlong)o{i}; (*env)->SetLongArrayRegion(env, {a}, 0, 1, &v); }}"));
                }
                // char** and Rlx…**: a pointer result.
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    post.push(format!("if ({a}) {{ jlong v = (jlong)(intptr_t)o{i}; (*env)->SetLongArrayRegion(env, {a}, 0, 1, &v); }}"));
                }
            }
            continue;
        }
        match t.as_str() {
            "const char*" => {
                pre.push(format!(
                    "const char* c{i} = {a} ? (*env)->GetStringUTFChars(env, {a}, NULL) : NULL;"
                ));
                args.push(format!("c{i}"));
                post.push(format!(
                    "if ({a}) (*env)->ReleaseStringUTFChars(env, {a}, c{i});"
                ));
            }
            "bool" => args.push(format!("{a} != JNI_FALSE")),
            "float" | "double" => args.push(a),
            "int32_t" => args.push(format!("(int32_t){a}")),
            "uint32_t" => args.push(format!("(uint32_t){a}")),
            "int64_t" => args.push(format!("(int64_t){a}")),
            "uint64_t" => args.push(format!("(uint64_t){a}")),
            "size_t" => args.push(format!("(size_t){a}")),
            "const float*" => {
                pre.push(format!(
                    "jfloat* c{i} = {a} ? (*env)->GetFloatArrayElements(env, {a}, NULL) : NULL;"
                ));
                args.push(format!("(const float*)c{i}"));
                post.push(format!(
                    "if ({a}) (*env)->ReleaseFloatArrayElements(env, {a}, c{i}, JNI_ABORT);"
                ));
            }
            "const uint8_t*" => {
                pre.push(format!(
                    "jbyte* c{i} = {a} ? (*env)->GetByteArrayElements(env, {a}, NULL) : NULL;"
                ));
                args.push(format!("(const uint8_t*)c{i}"));
                post.push(format!(
                    "if ({a}) (*env)->ReleaseByteArrayElements(env, {a}, c{i}, JNI_ABORT);"
                ));
            }
            "const uint32_t*" => {
                pre.push(format!(
                    "jint* c{i} = {a} ? (*env)->GetIntArrayElements(env, {a}, NULL) : NULL;"
                ));
                args.push(format!("(const uint32_t*)c{i}"));
                post.push(format!(
                    "if ({a}) (*env)->ReleaseIntArrayElements(env, {a}, c{i}, JNI_ABORT);"
                ));
            }
            "const int64_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                pre.push(format!(
                    "jlong* c{i} = {a} ? (*env)->GetLongArrayElements(env, {a}, NULL) : NULL;"
                ));
                args.push(format!("(const {ct}*)c{i}"));
                post.push(format!(
                    "if ({a}) (*env)->ReleaseLongArrayElements(env, {a}, c{i}, JNI_ABORT);"
                ));
            }
            "const size_t*" => {
                pre.push(format!("size_t* c{i} = rlx_jni_sizes(env, {a});"));
                args.push(format!("c{i}"));
                post.push(format!("free(c{i});"));
            }
            "const char* const*" => {
                pre.push(format!("RlxJniStrings c{i} = rlx_jni_strings(env, {a});"));
                args.push(format!("(const char* const*)c{i}.ptrs"));
                post.push(format!("rlx_jni_strings_release(env, {a}, &c{i});"));
            }
            "const float* const*" => {
                pre.push(format!("RlxJniArrays c{i} = rlx_jni_arrays(env, {a}, 0);"));
                args.push(format!("(const float* const*)c{i}.ptrs"));
                post.push(format!("rlx_jni_arrays_release(env, {a}, &c{i});"));
            }
            "const uint8_t* const*" => {
                pre.push(format!("RlxJniArrays c{i} = rlx_jni_arrays(env, {a}, 1);"));
                args.push(format!("(const uint8_t* const*)c{i}.ptrs"));
                post.push(format!("rlx_jni_arrays_release(env, {a}, &c{i});"));
            }
            handle => args.push(format!("({})(intptr_t){a}", handle)),
        }
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let call = format!("{}({})", f.c, args.join(", "));
    match ret {
        "jstring" => {
            let _ = writeln!(s, "    jstring r = (*env)->NewStringUTF(env, {call});");
        }
        "jboolean" => {
            let _ = writeln!(s, "    jboolean r = {call} ? JNI_TRUE : JNI_FALSE;");
        }
        _ => {
            let _ = writeln!(s, "    jint r = (jint){call};");
        }
    }
    for line in post.iter().rev() {
        let _ = writeln!(s, "    {line}");
    }
    let _ = writeln!(s, "    (void)cls;\n    return r;\n}}\n");
}

/// `defaults`: rlx-capi's default features; a function behind any other one is
/// compiled only with `-DRLX_FEATURE_<NAME>` (the C library must have it too).
pub fn generate(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* One JNI function per rlx-capi function, for `org.rlx.kt.Native`. Build with\n\
         \x20* rlx_jni_support.c into librlx_jni (Android: the NDK; desktop: any C\n\
         \x20* compiler and the JDK's headers), linked against librlx_capi.\n\
         \x20*/\n\n\
         #include \"rlx_jni.h\"\n\n",
    );
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let gate = f
            .feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")));
        if let Some(g) = &gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        function(&mut s, ir, f);
        if gate.is_some() {
            s.push_str("#endif\n\n");
        }
    }
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "JNIEXPORT void JNICALL Java_org_rlx_kt_Native_{}(JNIEnv* env, jclass cls, jlong h) {{\n    (void)env; (void)cls;\n    {}((Rlx{}*)(intptr_t)h);\n}}\n",
            mangle(&c.c_free()),
            c.c_free(),
            c.name
        );
    }
    s
}

/// The `external fun` declarations of `org.rlx.Native`.
pub fn kotlin_natives(ir: &Ir) -> String {
    let mut s = String::new();
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let params: Vec<String> = c_params(ir, f)
            .iter()
            .enumerate()
            .map(|(i, (t, out))| format!("a{i}: {}", kotlin_ty(jni_ty(t, *out))))
            .collect();
        let ret = if f.direct {
            match f.rets.first().map(|(_, k)| k.as_str()) {
                Some("str") => "String",
                _ => "Boolean",
            }
        } else {
            "Int"
        };
        let _ = writeln!(
            s,
            "    @JvmStatic external fun {}({}): {ret}",
            f.c,
            params.join(", ")
        );
    }
    for c in &ir.classes {
        let _ = writeln!(s, "    @JvmStatic external fun {}(h: Long)", c.c_free());
    }
    s
}
