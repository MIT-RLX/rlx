// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weights on disk: GGUF (feature `gguf`), the MLX / DDUF / NeMo / PyTorch
//! loaders and `.rlxp` packages (feature `weights`).

#[cfg(any(feature = "gguf", feature = "weights"))]
use std::ffi::c_char;

#[cfg(any(feature = "gguf", feature = "weights"))]
use crate::ffi::{arg, guard, out};

// ── GGUF ────────────────────────────────────────────────────

#[cfg(feature = "gguf")]
pub use gguf_api::*;

#[cfg(feature = "gguf")]
mod gguf_api {
    use rlx_bind::runtime::io::gguf::{self, Gguf, TensorData, TensorSpec};

    use super::*;
    use crate::ffi::{RlxBytes, RlxF32s, borrow, handle, release};

    /// An open, memory-mapped GGUF file.
    pub struct RlxGguf(Gguf);

    /// Tensors queued for `rlx_gguf_writer_write`.
    pub struct RlxGgufWriter(Vec<TensorSpec>);

    /// Open a GGUF file (memory-mapped).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_open(path: *const c_char, out: *mut *mut RlxGguf) -> i32 {
        guard(|| unsafe {
            let path = arg::str(path, "rlx_gguf_open path")?;
            out::put(out, handle(RlxGguf(Gguf::open(path)?)), "rlx_gguf_open out")
        })
    }

    /// Free a GGUF handle (`NULL` is a no-op).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_free(f: *mut RlxGguf) {
        unsafe { release(f) }
    }

    /// Tensor names, sorted, as a JSON array.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_tensor_names(f: *mut RlxGguf, out: *mut *mut c_char) -> i32 {
        guard(|| unsafe {
            let names = borrow(f, "gguf")?.0.tensor_names();
            out::strings(out, &names, "rlx_gguf_tensor_names out")
        })
    }

    /// `{"name", "shape", "dtype", "n_elements"}` as JSON, or `NULL` for an
    /// unknown name.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_info(
        f: *mut RlxGguf,
        name: *const c_char,
        out: *mut *mut c_char,
    ) -> i32 {
        guard(|| unsafe {
            let name = arg::str(name, "rlx_gguf_info name")?;
            match borrow(f, "gguf")?.0.info(name) {
                Some(t) => {
                    let v = rlx_bind::runtime::serde_json::json!({
                        "name": t.name, "shape": t.shape, "dtype": t.dtype, "n_elements": t.n_elements,
                    });
                    out::json(out, &v, "rlx_gguf_info out")
                }
                None => out::put(out, std::ptr::null_mut(), "rlx_gguf_info out"),
            }
        })
    }

    /// A tensor dequantized to f32 (free with `rlx_f32s_free`).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_tensor(
        f: *mut RlxGguf,
        name: *const c_char,
        out: *mut RlxF32s,
    ) -> i32 {
        guard(|| unsafe {
            let name = arg::str(name, "rlx_gguf_tensor name")?;
            let data = borrow(f, "gguf")?.0.dequant(name)?;
            out::put(out, data.into(), "rlx_gguf_tensor out")
        })
    }

    /// A tensor's stored bytes, still in its GGML layout (free with `rlx_bytes_free`).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_tensor_bytes(
        f: *mut RlxGguf,
        name: *const c_char,
        out: *mut RlxBytes,
    ) -> i32 {
        guard(|| unsafe {
            let name = arg::str(name, "rlx_gguf_tensor_bytes name")?;
            let data = borrow(f, "gguf")?.0.bytes(name)?;
            out::put(out, data.into(), "rlx_gguf_tensor_bytes out")
        })
    }

    /// Every metadata key as a JSON object; arrays past 64 entries read as a
    /// placeholder unless `full_arrays`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_metadata(
        f: *mut RlxGguf,
        full_arrays: bool,
        out: *mut *mut c_char,
    ) -> i32 {
        guard(|| unsafe {
            let meta = borrow(f, "gguf")?.0.metadata(full_arrays);
            out::json(out, &meta, "rlx_gguf_metadata out")
        })
    }

    /// `general.architecture`, or `NULL` when the file does not set it.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_architecture(f: *mut RlxGguf, out: *mut *mut c_char) -> i32 {
        guard(|| unsafe {
            match borrow(f, "gguf")?.0.architecture() {
                Some(a) => out::string(out, &a, "rlx_gguf_architecture out"),
                None => out::put(out, std::ptr::null_mut(), "rlx_gguf_architecture out"),
            }
        })
    }

    /// One metadata value in full, as JSON (`null` when the key is absent).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_metadata_value(
        f: *mut RlxGguf,
        key: *const c_char,
        out: *mut *mut c_char,
    ) -> i32 {
        guard(|| unsafe {
            let key = arg::str(key, "rlx_gguf_metadata_value key")?;
            let v = borrow(f, "gguf")?
                .0
                .metadata_value(key)
                .unwrap_or(rlx_bind::runtime::Value::Null);
            out::json(out, &v, "rlx_gguf_metadata_value out")
        })
    }

    /// safetensors (or PyTorch / ONNX) → GGUF at `dst`, quantized to `scheme`
    /// (`NULL` = `"Q4_K"`) by default. `options_json`: `{architecture,
    /// overrides: {tensor: scheme}, skip_norm_bias}`. Writes the report as JSON
    /// (feature `gguf-convert`).
    #[cfg(feature = "gguf-convert")]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_convert_to_gguf(
        src: *const c_char,
        dst: *const c_char,
        scheme: *const c_char,
        options_json: *const c_char,
        out: *mut *mut c_char,
    ) -> i32 {
        use rlx_bind::runtime::io::convert;
        guard(|| unsafe {
            let src = arg::str(src, "rlx_convert_to_gguf src")?;
            let dst = arg::str(dst, "rlx_convert_to_gguf dst")?;
            let scheme = arg::str_or(scheme, "Q4_K", "rlx_convert_to_gguf scheme")?;
            let opts = arg::json(options_json, "rlx_convert_to_gguf options")?;
            let report = convert::convert_to_gguf(src, dst, &scheme, &opts)?;
            out::json(
                out,
                &convert::report_json(&report),
                "rlx_convert_to_gguf out",
            )
        })
    }

    /// Pack f32 weights into a GGML type (`"Q4_K"`, …).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_quantize(
        data: *const f32,
        len: usize,
        dtype: *const c_char,
        out: *mut RlxBytes,
    ) -> i32 {
        guard(|| unsafe {
            let data = arg::slice(data, len, "rlx_quantize data")?;
            let dtype = arg::str(dtype, "rlx_quantize dtype")?;
            out::put(out, gguf::quantize(data, dtype)?.into(), "rlx_quantize out")
        })
    }

    /// Unpack GGML bytes to f32; `n` (`NULL` = inferred) is the element count.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_dequantize(
        data: *const u8,
        len: usize,
        dtype: *const c_char,
        n: *const usize,
        out: *mut RlxF32s,
    ) -> i32 {
        guard(|| unsafe {
            let data = arg::slice(data, len, "rlx_dequantize data")?;
            let dtype = arg::str(dtype, "rlx_dequantize dtype")?;
            out::put(
                out,
                gguf::dequantize(data, dtype, arg::opt(n))?.into(),
                "rlx_dequantize out",
            )
        })
    }

    /// Start a GGUF file to write.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_writer_new(out: *mut *mut RlxGgufWriter) -> i32 {
        guard(|| unsafe {
            out::put(
                out,
                handle(RlxGgufWriter(Vec::new())),
                "rlx_gguf_writer_new out",
            )
        })
    }

    /// Free a writer (`NULL` is a no-op).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_writer_free(w: *mut RlxGgufWriter) {
        unsafe { release(w) }
    }

    /// Queue an f32 tensor, quantized to `dtype` on write.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_writer_add_f32(
        w: *mut RlxGgufWriter,
        name: *const c_char,
        shape: *const usize,
        ndim: usize,
        dtype: *const c_char,
        data: *const f32,
        len: usize,
    ) -> i32 {
        guard(|| unsafe {
            let spec = TensorSpec {
                name: arg::str(name, "rlx_gguf_writer_add_f32 name")?.to_string(),
                shape: arg::vec(shape, ndim, "rlx_gguf_writer_add_f32 shape")?,
                dtype: arg::str(dtype, "rlx_gguf_writer_add_f32 dtype")?.to_string(),
                data: TensorData::F32(arg::vec(data, len, "rlx_gguf_writer_add_f32 data")?),
            };
            borrow(w, "gguf writer")?.0.push(spec);
            Ok(())
        })
    }

    /// Queue a tensor already in `dtype`'s packed layout.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_writer_add_bytes(
        w: *mut RlxGgufWriter,
        name: *const c_char,
        shape: *const usize,
        ndim: usize,
        dtype: *const c_char,
        data: *const u8,
        len: usize,
    ) -> i32 {
        guard(|| unsafe {
            let spec = TensorSpec {
                name: arg::str(name, "rlx_gguf_writer_add_bytes name")?.to_string(),
                shape: arg::vec(shape, ndim, "rlx_gguf_writer_add_bytes shape")?,
                dtype: arg::str(dtype, "rlx_gguf_writer_add_bytes dtype")?.to_string(),
                data: TensorData::Bytes(arg::vec(data, len, "rlx_gguf_writer_add_bytes data")?),
            };
            borrow(w, "gguf writer")?.0.push(spec);
            Ok(())
        })
    }

    /// Write the queued tensors to `path` (the writer is emptied).
    /// `architecture` and `metadata_json` (an object) may be `NULL`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_gguf_writer_write(
        w: *mut RlxGgufWriter,
        path: *const c_char,
        architecture: *const c_char,
        metadata_json: *const c_char,
        count: *mut usize,
    ) -> i32 {
        guard(|| unsafe {
            let path = arg::str(path, "rlx_gguf_writer_write path")?;
            let arch = arg::opt_str(architecture, "rlx_gguf_writer_write architecture")?;
            let meta = arg::json(metadata_json, "rlx_gguf_writer_write metadata")?;
            let specs = std::mem::take(&mut borrow(w, "gguf writer")?.0);
            let n = gguf::write(path, specs, arch.as_deref(), &meta)?;
            out::put(count, n, "rlx_gguf_writer_write count")
        })
    }
}

// ── weight loaders / .rlxp ──────────────────────────────────

#[cfg(feature = "weights")]
pub use weights_api::*;

#[cfg(feature = "weights")]
mod weights_api {
    use rlx_bind::runtime::io::weights;

    use super::*;
    use crate::ffi::{RlxTensorMap, handle};

    /// Every tensor of a checkpoint as f32. `format` is `"mlx"`, `"dduf"`,
    /// `"nemo"` or `"pt"`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_load_weights(
        format: *const c_char,
        path: *const c_char,
        out: *mut *mut RlxTensorMap,
    ) -> i32 {
        guard(|| unsafe {
            let format = arg::str(format, "rlx_load_weights format")?;
            let path = arg::str(path, "rlx_load_weights path")?;
            let entries = weights::load(format, path)?;
            out::put(
                out,
                handle(RlxTensorMap::new(entries)),
                "rlx_load_weights out",
            )
        })
    }

    /// A package's manifest as JSON.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_rlxp_open(path: *const c_char, out: *mut *mut c_char) -> i32 {
        guard(|| unsafe {
            let path = arg::str(path, "rlx_rlxp_open path")?;
            out::json(out, &weights::open_rlxp(path)?, "rlx_rlxp_open out")
        })
    }

    /// Check every tensor and sidecar hash; the report as JSON.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_rlxp_verify(path: *const c_char, out: *mut *mut c_char) -> i32 {
        guard(|| unsafe {
            let path = arg::str(path, "rlx_rlxp_verify path")?;
            out::json(out, &weights::verify_rlxp(path)?, "rlx_rlxp_verify out")
        })
    }

    /// Repackage `src` (a `from` = `"gguf"` / `"mlx"` / `"dduf"` / `"nemo"` /
    /// `"pt"` checkpoint) as `.rlxp` at `dst`. `options_json` (`NULL` =
    /// defaults): `{"container": "flat", "include_graph": false, …}`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rlx_to_rlxp(
        from: *const c_char,
        src: *const c_char,
        dst: *const c_char,
        options_json: *const c_char,
    ) -> i32 {
        guard(|| unsafe {
            let from = arg::str(from, "rlx_to_rlxp from")?;
            let src = arg::str(src, "rlx_to_rlxp src")?;
            let dst = arg::str(dst, "rlx_to_rlxp dst")?;
            let opts = arg::json(options_json, "rlx_to_rlxp options")?;
            weights::to_rlxp(from, src, dst, &opts)
        })
    }
}
