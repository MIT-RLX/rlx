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
use rlx_bind::runtime::io::gguf::{self, Gguf, TensorData, TensorSpec};

use crate::api::bind::{from_json, lift, to_json};
use crate::convert::*;
use crate::handle::{self, CLASS_GGUF};

fn file_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Gguf> {
    handle::borrow_mut(ctx, this, CLASS_GGUF, "GGUF file handle")
}

/// `openGguf(path)` → a handle that maps the file rather than reading it.
fn f_open(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "openGguf")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let file = lift(ctx, Gguf::open(&path))?;
    Ok(handle::wrap(ctx, CLASS_GGUF, file))
}

fn m_tensor_names(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = file_of(ctx, this)?.tensor_names();
    Ok(new_string_array(ctx, &names))
}

/// `{dims, dtype, elements}`, or `null` for an unknown name.
fn m_info(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let Some(info) = file_of(ctx, this)?.info(&name) else {
        return Ok(Value::Null);
    };
    let dims = new_usize_array(ctx, &info.shape);
    let dtype = new_string(ctx, info.dtype);
    Ok(new_object(
        ctx,
        vec![
            ("dims", dims),
            ("dtype", dtype),
            ("elements", Value::number(info.n_elements as f64)),
        ],
    ))
}

/// The tensor dequantized to f32.
fn m_tensor(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let r = file_of(ctx, this)?.dequant(&name);
    let data = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &data))
}

/// The tensor's stored bytes, still in its GGML layout.
fn m_tensor_bytes(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let r = file_of(ctx, this)?.bytes(&name);
    let data = lift(ctx, r)?;
    Ok(new_u8_array(ctx, &data))
}

/// Every metadata key; long arrays (a vocabulary) read as a placeholder —
/// `metadataValue(key)` returns them in full.
fn m_metadata(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let json = file_of(ctx, this)?.metadata(false);
    Ok(from_json(ctx, &json))
}

fn m_metadata_value(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let key = ctx.to_rust_string(arg(args, 0))?;
    Ok(match file_of(ctx, this)?.metadata_value(&key) {
        Some(v) => from_json(ctx, &v),
        None => Value::Null,
    })
}

fn m_architecture(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    Ok(match file_of(ctx, this)?.architecture() {
        Some(a) => new_string(ctx, &a),
        None => Value::Null,
    })
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let file = file_of(ctx, this)?;
        format!(
            "[rlx.Gguf v{} tensors={}]",
            file.version(),
            file.inner.tensors.len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── standalone quantization ─────────────────────────────────

/// `quantize(Float32Array, "Q4_K")` → `Uint8Array` of packed blocks.
fn f_quantize(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let data = to_f32_vec(ctx, arg(args, 0), "quantize weights")?;
    let label = ctx.to_rust_string(arg(args, 1))?;
    let packed = lift(ctx, gguf::quantize(&data, &label))?;
    Ok(new_u8_array(ctx, &packed))
}

/// `dequantize(bytes, "Q4_K", elements?)` → `Float32Array`.
fn f_dequantize(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bytes = to_bytes(ctx, arg(args, 0), "dequantize packed bytes")?;
    let label = ctx.to_rust_string(arg(args, 1))?;
    let n = if is_nullish(arg(args, 2)) {
        None
    } else {
        Some(to_usize(ctx, arg(args, 2), "dequantize elements")?)
    };
    let values = lift(ctx, gguf::dequantize(&bytes, &label, n))?;
    Ok(new_f32_array(ctx, &values))
}

// ── writing ─────────────────────────────────────────────────

/// `{data, shape, dtype}` — a `Float32Array` is quantized to `dtype`, anything
/// else is taken as already packed.
fn read_tensor_spec(ctx: &mut Context, name: &str, spec: &Value) -> JsResult<TensorSpec> {
    let shape_v = field(ctx, spec, "shape")?;
    let dtype_v = field(ctx, spec, "dtype")?;
    let data_v = field(ctx, spec, "data")?;
    if is_nullish(&shape_v) || is_nullish(&dtype_v) || is_nullish(&data_v) {
        return ctx.throw_type(&format!(
            "writeGguf tensor '{name}': expected {{data, shape, dtype}}"
        ));
    }
    let shape = to_usize_vec(ctx, &shape_v, "tensor shape")?;
    let dtype = ctx.to_rust_string(&dtype_v)?;
    let is_f32 = matches!(
        Context::typed_array_type(&data_v),
        Some(quickrs_core::builtins::typedarray::ElementType::Float32)
    );
    let data = if is_f32 {
        TensorData::F32(to_f32_vec(ctx, &data_v, &format!("tensor '{name}'"))?)
    } else {
        TensorData::Bytes(to_bytes(ctx, &data_v, &format!("tensor '{name}'"))?)
    };
    Ok(TensorSpec {
        name: name.to_string(),
        shape,
        dtype,
        data,
    })
}

/// `writeGguf(path, {name: {data, shape, dtype}}, {architecture, metadata})`
/// → tensor count.
fn f_write(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "writeGguf")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let tensors = arg(args, 1).clone();
    let options = arg(args, 2).clone();
    let arch_v = field(ctx, &options, "architecture")?;
    let arch = if is_nullish(&arch_v) {
        None
    } else {
        Some(ctx.to_rust_string(&arch_v)?)
    };
    let meta_v = field(ctx, &options, "metadata")?;
    let metadata = to_json(ctx, &meta_v)?;
    let mut specs = Vec::new();
    for name in own_keys(ctx, &tensors)? {
        let spec = field(ctx, &tensors, &name)?;
        specs.push(read_tensor_spec(ctx, &name, &spec)?);
    }
    let count = lift(ctx, gguf::write(&path, specs, arch.as_deref(), &metadata))?;
    Ok(Value::number(count as f64))
}

// ── safetensors → GGUF ──────────────────────────────────────

/// `convertToGguf(input, output, scheme = "Q4_K", {architecture, overrides,
/// skipNormBias})` → `{tensors, inputBytes, outputBytes, compressionRatio,
/// outputPath, schemes}`.
#[cfg(feature = "gguf-convert")]
fn f_convert(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use rlx_bind::runtime::io::convert;

    if let Some(refused) = handle::require_fs(ctx, "convertToGguf")? {
        return Ok(refused);
    }
    let input = ctx.to_rust_string(arg(args, 0))?;
    let output = ctx.to_rust_string(arg(args, 1))?;
    let scheme = to_string_or(ctx, arg(args, 2), "Q4_K")?;
    let opts = to_json(ctx, arg(args, 3))?;
    let report = lift(
        ctx,
        convert::convert_to_gguf(&input, &output, &scheme, &opts),
    )?;
    let json = convert::report_json(&report);
    Ok(crate::api::bind::report(ctx, &json))
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
            "tensorBytes" => m_tensor_bytes, 1;
            "architecture" => m_architecture, 0;
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
