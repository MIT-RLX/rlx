// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.openGguf(path)` — read a checkpoint from script.
//!
//! ```js
//! const f = rlx.openGguf("model.gguf");
//! for (const name of f.tensorNames()) {
//!   const { dims, dtype } = f.info(name);
//!   compiled.setParam(name, f.tensor(name));   // dequantized to f32
//! }
//! ```
//!
//! The file is mmap'd, so opening a 4 GB checkpoint costs no resident memory;
//! `tensor()` is what materializes (and dequantizes) one tensor at a time.
//!
//! Shapes come back in GGML order — innermost dimension first, the reverse of
//! safetensors.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};

use crate::convert::*;
use crate::handle::{self, CLASS_GGUF};

fn file_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut GgufFile> {
    handle::borrow_mut(ctx, this, CLASS_GGUF, "GGUF file handle")
}

fn f_open(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "openGguf")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    match GgufFile::from_path_mmap(&path) {
        Ok(file) => Ok(handle::wrap(ctx, CLASS_GGUF, file)),
        Err(e) => ctx.throw_internal(&format!("openGguf('{path}'): {e}")),
    }
}

fn m_tensor_names(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let mut names: Vec<String> = file_of(ctx, this)?.tensors.keys().cloned().collect();
    // HashMap order is not stable across runs; a script that writes params in
    // iteration order deserves the same order every time.
    names.sort();
    Ok(new_string_array(ctx, &names))
}

fn m_info(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let (dims, dtype, n) = {
        let file = file_of(ctx, this)?;
        match file.get(&name) {
            Some(t) => (t.shape.clone(), format!("{:?}", t.dtype), t.n_elements()),
            None => return Ok(Value::Null),
        }
    };
    let dims = new_usize_array(ctx, &dims);
    let dtype = new_string(ctx, &dtype);
    Ok(new_object(
        ctx,
        vec![
            ("dims", dims),
            ("dtype", dtype),
            ("elements", Value::number(n as f64)),
        ],
    ))
}

/// One tensor, dequantized to f32.
fn m_tensor(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let data = {
        let file = file_of(ctx, this)?;
        match file.dequant_f32(&name) {
            Ok((data, _dims)) => data,
            Err(e) => return ctx.throw_internal(&format!("tensor('{name}'): {e}")),
        }
    };
    Ok(new_f32_array(ctx, &data))
}

/// One metadata entry as a native JS value.
///
/// A big tokenizer vocabulary is an `Array` of 150k strings; materializing it
/// eagerly for every `metadata()` call would dwarf the rest of the header, so
/// arrays past `ARRAY_PREVIEW` report their length instead of their contents.
/// `metadataValue(key)` returns the whole thing when a script wants it.
const ARRAY_PREVIEW: usize = 64;

fn meta_to_js(ctx: &mut Context, value: &rlx_gguf::MetaValue, full_arrays: bool) -> Value {
    use rlx_gguf::MetaValue as M;
    match value {
        M::U8(v) => Value::number(*v as f64),
        M::I8(v) => Value::number(*v as f64),
        M::U16(v) => Value::number(*v as f64),
        M::I16(v) => Value::number(*v as f64),
        M::U32(v) => Value::number(*v as f64),
        M::I32(v) => Value::number(*v as f64),
        // u64/i64 past 2^53 cannot survive a JS number; the header fields that
        // use them (counts, offsets) are far below it.
        M::U64(v) => Value::number(*v as f64),
        M::I64(v) => Value::number(*v as f64),
        M::F32(v) => Value::number(*v as f64),
        M::F64(v) => Value::number(*v),
        M::Bool(v) => Value::Bool(*v),
        M::String(s) => new_string(ctx, s),
        M::Array(items) => {
            if !full_arrays && items.len() > ARRAY_PREVIEW {
                let text = format!("[{} items — use metadataValue(key)]", items.len());
                return new_string(ctx, &text);
            }
            let values: Vec<Value> = items
                .iter()
                .map(|item| meta_to_js(ctx, item, full_arrays))
                .collect();
            new_array(ctx, values)
        }
    }
}

/// Header metadata — architecture, context length, rope base, …
fn m_metadata(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let mut entries: Vec<(String, rlx_gguf::MetaValue)> = file_of(ctx, this)?
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let obj = ctx.new_object();
    for (key, value) in entries {
        let js = meta_to_js(ctx, &value, false);
        ctx.define_value(&obj, &key, js, quickrs_core::object::PropFlags::C_W_E);
    }
    Ok(Value::Object(obj))
}

/// One metadata key, arrays included in full.
fn m_metadata_value(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let key = ctx.to_rust_string(arg(args, 0))?;
    let value = file_of(ctx, this)?.metadata.get(&key).cloned();
    Ok(match value {
        Some(v) => meta_to_js(ctx, &v, true),
        None => Value::Null,
    })
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let file = file_of(ctx, this)?;
        format!(
            "[rlx.Gguf v{} tensors={}]",
            file.version,
            file.tensors.len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── standalone quantization ─────────────────────────────────

fn parse_ggml_type(ctx: &mut Context, name: &str) -> JsResult<GgmlType> {
    match GgmlType::from_name(name) {
        Some(t) => Ok(t),
        None => ctx.throw_type(&format!(
            "unknown GGUF dtype '{name}' — expected e.g. Q4_K, Q6_K, IQ2_XXS, TQ2_0"
        )),
    }
}

/// `quantize(Float32Array, "Q4_K")` → `Uint8Array` of packed bytes.
///
/// The length must divide the scheme's block size (256 for the K/IQ/TQ
/// families, 32 for Q4_0 / IQ4_NL / MXFP4).
fn f_quantize(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let data = to_f32_vec(ctx, arg(args, 0), "quantize weights")?;
    let label = ctx.to_rust_string(arg(args, 1))?;
    let ggml = parse_ggml_type(ctx, &label)?;
    match rlx_gguf::quantize(&data, ggml) {
        Ok(packed) => Ok(new_u8_array(ctx, &packed)),
        Err(e) => ctx.throw_internal(&format!("quantize({label}): {e}")),
    }
}

/// `dequantize(Uint8Array, "Q4_K", elements?)` → `Float32Array`.
///
/// `elements` is inferred from the packed length when omitted, which is
/// unambiguous for every scheme whose block size divides evenly.
fn f_dequantize(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bytes = to_bytes(ctx, arg(args, 0), "dequantize packed bytes")?;
    let label = ctx.to_rust_string(arg(args, 1))?;
    let ggml = parse_ggml_type(ctx, &label)?;
    let n = if is_nullish(arg(args, 2)) {
        match (1..=bytes.len().saturating_mul(256))
            .find(|&n| rlx_gguf::bytes_for_public(ggml, n).is_some_and(|b| b == bytes.len()))
        {
            Some(n) => n,
            None => {
                return ctx.throw_type(&format!(
                    "dequantize: cannot infer element count for {label} from {} packed bytes — \
                     pass it explicitly",
                    bytes.len()
                ));
            }
        }
    } else {
        to_usize(ctx, arg(args, 2), "dequantize elements")?
    };
    match rlx_gguf::dequant_typed(ggml, &bytes, n, "dequantize") {
        Ok(values) => Ok(new_f32_array(ctx, &values)),
        Err(e) => ctx.throw_internal(&format!("dequantize({label}, n={n}): {e}")),
    }
}

// ── writing ─────────────────────────────────────────────────

/// One `{data, shape, dtype}` tensor spec. `data` may be f32 (quantized on
/// the way out) or already-packed bytes.
fn read_tensor_spec(
    ctx: &mut Context,
    name: &str,
    spec: &Value,
) -> JsResult<(Vec<usize>, GgmlType, Vec<u8>)> {
    let shape_v = field(ctx, spec, "shape")?;
    let dtype_v = field(ctx, spec, "dtype")?;
    let data_v = field(ctx, spec, "data")?;
    if is_nullish(&shape_v) || is_nullish(&dtype_v) || is_nullish(&data_v) {
        return ctx.throw_type(&format!(
            "writeGguf tensor '{name}': expected {{data, shape, dtype}}"
        ));
    }
    let shape = to_usize_vec(ctx, &shape_v, "tensor shape")?;
    let label = ctx.to_rust_string(&dtype_v)?;
    let ggml = parse_ggml_type(ctx, &label)?;

    // A Float32Array means "quantize this for me"; anything else is already
    // in the target layout and is written through untouched.
    let is_f32_input = matches!(
        Context::typed_array_type(&data_v),
        Some(quickrs_core::builtins::typedarray::ElementType::Float32)
    );
    let bytes = if is_f32_input && ggml != GgmlType::F32 {
        let values = to_f32_vec(ctx, &data_v, &format!("tensor '{name}'"))?;
        match rlx_gguf::quantize(&values, ggml) {
            Ok(packed) => packed,
            Err(e) => return ctx.throw_internal(&format!("writeGguf '{name}': {e}")),
        }
    } else {
        to_bytes(ctx, &data_v, &format!("tensor '{name}'"))?
    };
    Ok((shape, ggml, bytes))
}

/// `writeGguf(path, {name: {data, shape, dtype}}, {architecture, metadata})`.
fn f_write(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "writeGguf")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let tensors = arg(args, 1).clone();
    let options = arg(args, 2).clone();

    let mut writer = GgufWriter::new();
    let arch_v = field(ctx, &options, "architecture")?;
    if !is_nullish(&arch_v) {
        let arch = ctx.to_rust_string(&arch_v)?;
        writer.set_arch(&arch);
    }
    let meta_v = field(ctx, &options, "metadata")?;
    if !is_nullish(&meta_v) {
        for key in own_keys(ctx, &meta_v)? {
            let value = field(ctx, &meta_v, &key)?;
            let entry = match &value {
                Value::Bool(b) => MetaValue::Bool(*b),
                Value::Int(n) => MetaValue::I64(*n as i64),
                Value::Float(x) => MetaValue::F64(*x),
                _ => MetaValue::String(ctx.to_rust_string(&value)?),
            };
            writer.set_meta(key, entry);
        }
    }

    let mut names = own_keys(ctx, &tensors)?;
    names.sort();
    for name in &names {
        let spec = field(ctx, &tensors, name)?;
        let (shape, ggml, bytes) = read_tensor_spec(ctx, name, &spec)?;
        if let Err(e) = writer.add_tensor_bytes(name.clone(), shape, ggml, bytes) {
            return ctx.throw_internal(&format!("writeGguf tensor '{name}': {e}"));
        }
    }
    match writer.write_to_path(&path) {
        Ok(()) => Ok(Value::number(names.len() as f64)),
        Err(e) => ctx.throw_internal(&format!("writeGguf('{path}'): {e:#}")),
    }
}

// ── safetensors → GGUF ──────────────────────────────────────

/// `convertToGguf(src, out, "Q4_K", {architecture, skipNormBias, overrides})`.
///
/// `skipNormBias` defaults **on**: quantizing 1-D norm and bias tensors costs
/// accuracy for almost no space, since they are a rounding error of the total.
#[cfg(feature = "gguf-convert")]
fn f_convert(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use rlx_gguf_convert::{Converter, Scheme};

    fn parse_scheme(ctx: &mut Context, s: &str) -> JsResult<Scheme> {
        match Scheme::parse(s) {
            Ok(v) => Ok(v),
            Err(e) => ctx.throw_type(&format!("unknown quant scheme '{s}': {e}")),
        }
    }

    if let Some(refused) = handle::require_fs(ctx, "convertToGguf")? {
        return Ok(refused);
    }
    let input = ctx.to_rust_string(arg(args, 0))?;
    let output = ctx.to_rust_string(arg(args, 1))?;
    let scheme_label = to_string_or(ctx, arg(args, 2), "Q4_K")?;
    let default_scheme = parse_scheme(ctx, &scheme_label)?;
    let options = arg(args, 3).clone();

    let mut converter = match Converter::from_safetensors(&input) {
        Ok(c) => c.default_scheme(default_scheme),
        Err(e) => return ctx.throw_internal(&format!("convertToGguf('{input}'): {e:#}")),
    };

    let arch_v = field(ctx, &options, "architecture")?;
    if !is_nullish(&arch_v) {
        let arch = ctx.to_rust_string(&arch_v)?;
        converter = converter.architecture(&arch);
    }
    let overrides_v = field(ctx, &options, "overrides")?;
    if !is_nullish(&overrides_v) {
        for name in own_keys(ctx, &overrides_v)? {
            let value = field(ctx, &overrides_v, &name)?;
            let label = ctx.to_rust_string(&value)?;
            let scheme = parse_scheme(ctx, &label)?;
            converter = converter.scheme_for_name(name, scheme);
        }
    }
    let skip_v = field(ctx, &options, "skipNormBias")?;
    if is_nullish(&skip_v) || to_bool(&skip_v) {
        converter = converter.skip_quant_for(|name, shape| {
            shape.len() < 2 || name.contains("norm") || name.contains("bias")
        });
    }

    let report = match converter.write_gguf(&output) {
        Ok(r) => r,
        Err(e) => return ctx.throw_internal(&format!("convertToGguf write: {e:#}")),
    };
    let out_path = new_string(ctx, &report.output_path.to_string_lossy());
    Ok(new_object(
        ctx,
        vec![
            ("tensors", Value::number(report.tensors as f64)),
            ("inputBytes", Value::number(report.input_bytes as f64)),
            ("outputBytes", Value::number(report.output_bytes as f64)),
            (
                "compressionRatio",
                Value::number(report.compression_ratio()),
            ),
            ("outputPath", out_path),
        ],
    ))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    // No constructor: a GGUF handle only ever comes from `rlx.openGguf`.
    js_class! {
        ctx, namespace;
        name: "Gguf",
        class: CLASS_GGUF,
        methods: {
            "tensorNames" => m_tensor_names, 0;
            "info" => m_info, 1;
            "tensor" => m_tensor, 1;
            "metadata" => m_metadata, 0;
            "metadataValue" => m_metadata_value, 1;
            "toString" => m_to_string, 0;
        }
    };
    js_functions! {
        ctx, namespace;
        "openGguf" => f_open, 1;
        "quantize" => f_quantize, 2;
        "dequantize" => f_dequantize, 3;
        "writeGguf" => f_write, 3;
    };
    #[cfg(feature = "gguf-convert")]
    js_functions! {
        ctx, namespace;
        "convertToGguf" => f_convert, 4;
    };
}
