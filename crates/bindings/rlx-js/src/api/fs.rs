// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! File reading and writing — **not installed by default**.
//!
//! [`Runtime::new`](crate::Runtime::new) deliberately gives a script no way to
//! touch the filesystem: an embedded engine that can read anything is a very
//! different security story from one that cannot, and "no I/O a script can
//! reach unless an embedder hands it one" is only true if it is actually true.
//!
//! The `rlx-js` CLI hands it over, because a script run from a shell already
//! has the user's authority. A library embedder opts in explicitly:
//!
//! ```no_run
//! let mut rt = rlx_js::Runtime::new();
//! rt.allow_filesystem();          // now rlx.readFile etc. exist
//! ```
//!
//! Training needs data, and a dataset loader is the one thing a pure-graph API
//! cannot supply — hence a real `readFile` rather than a bundled fixture.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};

use crate::convert::*;

/// `readFile(path)` → `Uint8Array`. Errors carry the OS message, which is what
/// tells a caller "no such file" apart from "permission denied".
fn f_read_file(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    match std::fs::read(&path) {
        Ok(bytes) => Ok(new_u8_array(ctx, &bytes)),
        Err(e) => ctx.throw_internal(&format!("readFile('{path}'): {e}")),
    }
}

/// `readText(path)` → string. Invalid UTF-8 is an error rather than a silent
/// replacement-character substitution.
fn f_read_text(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(new_string(ctx, &text)),
        Err(e) => ctx.throw_internal(&format!("readText('{path}'): {e}")),
    }
}

/// `writeFile(path, data)` → bytes written. `data` may be a string or any
/// typed array.
fn f_write_file(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    let data = arg(args, 1).clone();
    let bytes = if data.as_string().is_some() {
        ctx.to_rust_string(&data)?.into_bytes()
    } else {
        to_bytes(ctx, &data, "writeFile data")?
    };
    match std::fs::write(&path, &bytes) {
        Ok(()) => Ok(Value::number(bytes.len() as f64)),
        Err(e) => ctx.throw_internal(&format!("writeFile('{path}'): {e}")),
    }
}

/// `fileExists(path)` — a plain predicate, so a loader can search candidate
/// directories without try/catch around every one.
fn f_exists(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    Ok(Value::Bool(std::path::Path::new(&path).is_file()))
}

/// `fileSize(path)` → bytes, or `null` when it is not a readable file. Lets a
/// script check a large dataset is intact before spending time on it.
fn f_size(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    Ok(match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() => Value::number(meta.len() as f64),
        _ => Value::Null,
    })
}

/// `env(name)` → string or `null`. Just enough for a script to find a cache
/// directory (`$HOME`, `$RLX_MODELS_DIR`) without hardcoding a path.
fn f_env(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    Ok(match std::env::var(&name) {
        Ok(value) => new_string(ctx, &value),
        Err(_) => Value::Null,
    })
}

/// `readFileRange(path, offset, length)` → `Uint8Array`.
///
/// A dataset that does not fit in memory is read a window at a time; `readFile`
/// on a 40 GB shard is not a strategy. `length` past the end clamps rather than
/// erroring, so a caller can ask for a block without first checking the size.
fn f_read_range(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use std::io::{Read, Seek, SeekFrom};

    let path = ctx.to_rust_string(arg(args, 0))?;
    let offset = to_usize(ctx, arg(args, 1), "readFileRange offset")?;
    let length = to_usize(ctx, arg(args, 2), "readFileRange length")?;
    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => return ctx.throw_internal(&format!("readFileRange('{path}'): {e}")),
    };
    // Clamp to what is actually there before allocating: `length` comes from
    // script, and a mistaken 1e12 should be a short read, not an OOM.
    let available = file
        .metadata()
        .map(|m| (m.len() as usize).saturating_sub(offset))
        .unwrap_or(length);
    let length = length.min(available);
    if let Err(e) = file.seek(SeekFrom::Start(offset as u64)) {
        return ctx.throw_internal(&format!("readFileRange('{path}') seek: {e}"));
    }
    let mut buffer = vec![0u8; length];
    let mut filled = 0usize;
    while filled < length {
        match file.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => return ctx.throw_internal(&format!("readFileRange('{path}'): {e}")),
        }
    }
    buffer.truncate(filled);
    Ok(new_u8_array(ctx, &buffer))
}

/// `readFileInto(path, target, fileOffset?)` → bytes read.
///
/// Fills a caller-owned `Uint8Array`, so a streaming loop over a large dataset
/// allocates one buffer instead of one per block.
fn f_read_into(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    use quickrs_core::object::ObjectData;
    use std::io::{Read, Seek, SeekFrom};

    let path = ctx.to_rust_string(arg(args, 0))?;
    let target = arg(args, 1).clone();
    let offset = if is_nullish(arg(args, 2)) {
        0
    } else {
        to_usize(ctx, arg(args, 2), "readFileInto offset")?
    };
    let Some((buffer, window_offset, window_len)) = u8_window(&target) else {
        return ctx.throw_type("readFileInto: target must be a Uint8Array");
    };

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => return ctx.throw_internal(&format!("readFileInto('{path}'): {e}")),
    };
    if let Err(e) = file.seek(SeekFrom::Start(offset as u64)) {
        return ctx.throw_internal(&format!("readFileInto('{path}') seek: {e}"));
    }
    let mut staging = vec![0u8; window_len];
    let mut filled = 0usize;
    while filled < window_len {
        match file.read(&mut staging[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => return ctx.throw_internal(&format!("readFileInto('{path}'): {e}")),
        }
    }
    {
        let mut borrowed = buffer.borrow_mut();
        if let ObjectData::Buffer(data) = &mut borrowed.data {
            if data.detached || window_offset + filled > data.bytes.len() {
                return ctx.throw_type("readFileInto: the target buffer went away mid-read");
            }
            data.bytes[window_offset..window_offset + filled].copy_from_slice(&staging[..filled]);
        }
    }
    Ok(Value::number(filled as f64))
}

/// The `(buffer, offset, len)` a `Uint8Array` covers.
fn u8_window(v: &Value) -> Option<(quickrs_core::gc::Gc<JsObject>, usize, usize)> {
    use quickrs_core::builtins::typedarray::ElementType;
    use quickrs_core::object::ObjectData;

    if !matches!(
        Context::typed_array_type(v),
        Some(ElementType::Uint8) | Some(ElementType::Uint8Clamped)
    ) {
        return None;
    }
    let obj = v.as_object()?;
    let borrowed = obj.borrow();
    let ObjectData::View(view) = &borrowed.data else {
        return None;
    };
    let buffer = view.buffer.clone();
    let offset = view.byte_offset;
    let len = match view.byte_length {
        Some(n) => n,
        None => {
            let b = buffer.borrow();
            let ObjectData::Buffer(data) = &b.data else {
                return None;
            };
            data.bytes.len().saturating_sub(offset)
        }
    };
    Some((buffer, offset, len))
}

/// `listDir(path)` → sorted entry names. Sorted because directory order is not
/// stable across filesystems and a dataset loader should be reproducible.
fn f_list_dir(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    let entries = match std::fs::read_dir(&path) {
        Ok(entries) => entries,
        Err(e) => return ctx.throw_internal(&format!("listDir('{path}'): {e}")),
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(new_string_array(ctx, &names))
}

/// `mkdirs(path)` — create a directory and its parents; already existing is ok.
fn f_mkdirs(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    match std::fs::create_dir_all(&path) {
        Ok(()) => Ok(Value::Bool(true)),
        Err(e) => ctx.throw_internal(&format!("mkdirs('{path}'): {e}")),
    }
}

/// `removeFile(path)` → whether something was removed. A missing file is
/// `false`, not an error: "make sure this is gone" is the common intent.
fn f_remove(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let path = ctx.to_rust_string(arg(args, 0))?;
    if !std::path::Path::new(&path).exists() {
        return Ok(Value::Bool(false));
    }
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(Value::Bool(true)),
        Err(e) => ctx.throw_internal(&format!("removeFile('{path}'): {e}")),
    }
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "readFile" => f_read_file, 1;
        "readText" => f_read_text, 1;
        "writeFile" => f_write_file, 2;
        "fileExists" => f_exists, 1;
        "fileSize" => f_size, 1;
        "env" => f_env, 1;
        "readFileRange" => f_read_range, 3;
        "readFileInto" => f_read_into, 3;
        "listDir" => f_list_dir, 1;
        "mkdirs" => f_mkdirs, 1;
        "removeFile" => f_remove, 1;
    };
}
