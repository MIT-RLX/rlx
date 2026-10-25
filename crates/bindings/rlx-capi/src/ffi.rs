// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Status codes, the last-error slot, owned buffers, and the argument / result
//! converters every entry point uses.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use rlx_bind::{ApiError, ApiResult, ErrorKind};

/// Success.
pub const RLX_OK: i32 = 0;
/// A wrong argument: unknown name, bad shape, wrong arity.
pub const RLX_ERR_TYPE: i32 = -1;
/// An index or value out of range (a node id the graph does not have).
pub const RLX_ERR_RANGE: i32 = -2;
/// Well-formed but not carried out: I/O, a backend not in this build.
pub const RLX_ERR_RUNTIME: i32 = -3;
/// A Rust assertion inside rlx, caught at the boundary.
pub const RLX_ERR_PANIC: i32 = -4;
/// A file that does not exist.
pub const RLX_ERR_NOT_FOUND: i32 = -5;
/// "No node" for an optional node argument.
pub const RLX_NONE: u32 = u32::MAX;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: &str) {
    let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
}

/// The message of the last failed call on this thread (empty if none). Valid
/// until the next failing call on this thread; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn rlx_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// The library version, e.g. `"0.2.18"`. Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn rlx_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// A `NULL` where a pointer was required — a wrong argument (`RLX_ERR_TYPE`).
pub(crate) fn null(what: &str) -> ApiError {
    ApiError::new(ErrorKind::Type, format!("{what}: NULL pointer"))
}

/// Run `f`, returning its status and recording any error.
pub(crate) fn guard(f: impl FnOnce() -> ApiResult<()>) -> i32 {
    rlx_bind::panics::install_hook();
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => RLX_OK,
        Ok(Err(e)) => {
            set_error(&e.message);
            e.kind.code()
        }
        Err(payload) => {
            set_error(&format!(
                "{} (a Rust assertion inside rlx, surfaced here rather than aborting)",
                rlx_bind::panics::message(payload.as_ref())
            ));
            RLX_ERR_PANIC
        }
    }
}

// ── owned buffers ───────────────────────────────────────────

/// A `uint32_t` array rlx allocated; free with `rlx_u32s_free`.
#[repr(C)]
pub struct RlxU32s {
    pub ptr: *mut u32,
    pub len: usize,
}

/// A `float` array rlx allocated; free with `rlx_f32s_free`.
#[repr(C)]
pub struct RlxF32s {
    pub ptr: *mut f32,
    pub len: usize,
}

/// A byte array rlx allocated; free with `rlx_bytes_free`.
#[repr(C)]
pub struct RlxBytes {
    pub ptr: *mut u8,
    pub len: usize,
}

fn into_raw<T>(v: Vec<T>) -> (*mut T, usize) {
    let boxed = v.into_boxed_slice();
    let len = boxed.len();
    (Box::into_raw(boxed).cast(), len)
}

unsafe fn drop_raw<T>(ptr: *mut T, len: usize) {
    if !ptr.is_null() {
        // SAFETY: `ptr`/`len` came from `into_raw` (a boxed slice).
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

impl From<Vec<u32>> for RlxU32s {
    fn from(v: Vec<u32>) -> Self {
        let (ptr, len) = into_raw(v);
        Self { ptr, len }
    }
}

impl From<Vec<f32>> for RlxF32s {
    fn from(v: Vec<f32>) -> Self {
        let (ptr, len) = into_raw(v);
        Self { ptr, len }
    }
}

impl From<Vec<u8>> for RlxBytes {
    fn from(v: Vec<u8>) -> Self {
        let (ptr, len) = into_raw(v);
        Self { ptr, len }
    }
}

/// Free a `uint32_t` array rlx returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_u32s_free(v: RlxU32s) {
    unsafe { drop_raw(v.ptr, v.len) }
}

/// Free a `float` array rlx returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_f32s_free(v: RlxF32s) {
    unsafe { drop_raw(v.ptr, v.len) }
}

/// Free a byte array rlx returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_bytes_free(v: RlxBytes) {
    unsafe { drop_raw(v.ptr, v.len) }
}

/// Free a `uint32_t` array through a pointer to it — for FFIs that cannot pass
/// a struct by value (Ruby's Fiddle). Same as `rlx_u32s_free(*v)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_u32s_release(v: *mut RlxU32s) {
    if let Some(v) = unsafe { v.as_mut() } {
        unsafe { drop_raw(v.ptr, v.len) };
        v.ptr = std::ptr::null_mut();
        v.len = 0;
    }
}

/// Free a `float` array through a pointer to it — see `rlx_u32s_release`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_f32s_release(v: *mut RlxF32s) {
    if let Some(v) = unsafe { v.as_mut() } {
        unsafe { drop_raw(v.ptr, v.len) };
        v.ptr = std::ptr::null_mut();
        v.len = 0;
    }
}

/// Free a byte array through a pointer to it — see `rlx_u32s_release`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_bytes_release(v: *mut RlxBytes) {
    if let Some(v) = unsafe { v.as_mut() } {
        unsafe { drop_raw(v.ptr, v.len) };
        v.ptr = std::ptr::null_mut();
        v.len = 0;
    }
}

/// Free a string rlx returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: every string rlx hands out came from `CString::into_raw`.
        drop(unsafe { CString::from_raw(s) });
    }
}

// ── argument readers ────────────────────────────────────────

pub(crate) mod arg {
    use super::*;

    /// A required UTF-8 string.
    pub unsafe fn str<'a>(p: *const c_char, what: &str) -> ApiResult<&'a str> {
        if p.is_null() {
            return Err(null(what));
        }
        // SAFETY: the caller passes a NUL-terminated string.
        unsafe { CStr::from_ptr(p) }
            .to_str()
            .map_err(|_| ApiError::type_(format!("{what}: not valid UTF-8")))
    }

    /// An optional string: `NULL` → `None`.
    pub unsafe fn opt_str(p: *const c_char, what: &str) -> ApiResult<Option<String>> {
        if p.is_null() {
            return Ok(None);
        }
        Ok(Some(unsafe { str(p, what) }?.to_string()))
    }

    /// A defaulted string: `NULL` → `default`.
    pub unsafe fn str_or(p: *const c_char, default: &str, what: &str) -> ApiResult<String> {
        Ok(unsafe { opt_str(p, what) }?.unwrap_or_else(|| default.to_string()))
    }

    /// An array: `NULL` with length 0 is empty.
    pub unsafe fn slice<'a, T>(p: *const T, len: usize, what: &str) -> ApiResult<&'a [T]> {
        if len == 0 {
            return Ok(&[]);
        }
        if p.is_null() {
            return Err(null(what));
        }
        // SAFETY: the caller passes `len` readable elements.
        Ok(unsafe { std::slice::from_raw_parts(p, len) })
    }

    pub unsafe fn vec<T: Clone>(p: *const T, len: usize, what: &str) -> ApiResult<Vec<T>> {
        Ok(unsafe { slice(p, len, what) }?.to_vec())
    }

    /// An optional array: `NULL` → `None` (whatever `len` says).
    pub unsafe fn opt_vec<T: Clone>(
        p: *const T,
        len: usize,
        what: &str,
    ) -> ApiResult<Option<Vec<T>>> {
        if p.is_null() {
            return Ok(None);
        }
        Ok(Some(unsafe { vec(p, len, what) }?))
    }

    /// An optional scalar: `NULL` → `None`.
    pub unsafe fn opt<T: Copy>(p: *const T) -> Option<T> {
        // SAFETY: a non-null pointer to one readable `T`.
        (!p.is_null()).then(|| unsafe { *p })
    }

    pub fn opt_node(id: u32) -> Option<u32> {
        (id != RLX_NONE).then_some(id)
    }

    /// `rows` rows of `width` values, flattened.
    pub unsafe fn rows<T: Clone>(
        p: *const T,
        rows: usize,
        width: usize,
        what: &str,
    ) -> ApiResult<Vec<Vec<T>>> {
        let flat = unsafe { slice(p, rows * width, what) }?;
        Ok(flat.chunks(width).map(<[T]>::to_vec).collect())
    }

    /// A list of strings: `n` NUL-terminated pointers.
    pub unsafe fn strs(p: *const *const c_char, n: usize, what: &str) -> ApiResult<Vec<String>> {
        let ptrs = unsafe { slice(p, n, what) }?;
        ptrs.iter()
            .map(|s| unsafe { str(*s, what) }.map(str::to_string))
            .collect()
    }

    /// A JSON options string: `NULL` → null.
    pub unsafe fn json(p: *const c_char, what: &str) -> ApiResult<rlx_bind::runtime::Value> {
        match unsafe { opt_str(p, what) }? {
            None => Ok(rlx_bind::runtime::Value::Null),
            Some(text) => rlx_bind::runtime::serde_json::from_str(&text)
                .map_err(|e| ApiError::type_(format!("{what}: invalid JSON: {e}"))),
        }
    }

    /// `n` named f32 inputs: parallel arrays of names, data pointers, lengths.
    pub unsafe fn inputs<'a>(
        names: *const *const c_char,
        data: *const *const f32,
        lens: *const usize,
        n: usize,
        what: &str,
    ) -> ApiResult<Vec<(&'a str, &'a [f32])>> {
        let names = unsafe { slice(names, n, what) }?;
        let data = unsafe { slice(data, n, what) }?;
        let lens = unsafe { slice(lens, n, what) }?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let name = unsafe { str(names[i], what) }?;
            let values = unsafe { slice(data[i], lens[i], what) }?;
            out.push((name, values));
        }
        Ok(out)
    }
}

// ── result writers ──────────────────────────────────────────

pub(crate) mod out {
    use super::*;

    /// Write `v` through `p`.
    pub unsafe fn put<T>(p: *mut T, v: T, what: &str) -> ApiResult<()> {
        if p.is_null() {
            return Err(null(what));
        }
        // SAFETY: a non-null pointer to writable storage for one `T`.
        unsafe { p.write(v) };
        Ok(())
    }

    /// A string the caller frees with `rlx_string_free`.
    pub unsafe fn string(p: *mut *mut c_char, s: &str, what: &str) -> ApiResult<()> {
        let c = CString::new(s.replace('\0', " ")).unwrap_or_default();
        unsafe { put(p, c.into_raw(), what) }
    }

    /// JSON text the caller frees with `rlx_string_free`.
    pub unsafe fn json(
        p: *mut *mut c_char,
        v: &rlx_bind::runtime::Value,
        what: &str,
    ) -> ApiResult<()> {
        unsafe { string(p, &v.to_string(), what) }
    }

    pub unsafe fn strings(p: *mut *mut c_char, items: &[String], what: &str) -> ApiResult<()> {
        let v = rlx_bind::runtime::serde_json::json!(items);
        unsafe { json(p, &v, what) }
    }

    pub unsafe fn shape(
        p: *mut *mut c_char,
        (dims, dtype): (Vec<i64>, String),
        what: &str,
    ) -> ApiResult<()> {
        let v = rlx_bind::runtime::serde_json::json!({ "dims": dims, "dtype": dtype });
        unsafe { json(p, &v, what) }
    }
}

/// Wrap a value as a heap handle for C.
pub(crate) fn handle<T>(v: T) -> *mut T {
    Box::into_raw(Box::new(v))
}

/// Borrow a handle.
pub(crate) unsafe fn borrow<'a, T>(p: *mut T, what: &str) -> ApiResult<&'a mut T> {
    // SAFETY: a non-null handle from `handle`, not yet freed.
    unsafe { p.as_mut() }.ok_or_else(|| null(what))
}

/// Free a handle (`NULL` is a no-op).
pub(crate) unsafe fn release<T>(p: *mut T) {
    if !p.is_null() {
        // SAFETY: a handle from `handle`, freed once.
        drop(unsafe { Box::from_raw(p) });
    }
}

// ── named tensors ───────────────────────────────────────────

/// Named f32 tensors (weights, trainer params): iterate with
/// `rlx_tensor_map_len` / `rlx_tensor_map_get`, free with `rlx_tensor_map_free`.
pub struct RlxTensorMap {
    pub(crate) entries: Vec<(CString, Vec<f32>)>,
}

impl RlxTensorMap {
    #[cfg(any(feature = "training", feature = "weights"))]
    pub(crate) fn new(entries: Vec<(String, Vec<f32>)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(|(n, d)| (CString::new(n.replace('\0', " ")).unwrap_or_default(), d))
                .collect(),
        }
    }
}

/// How many tensors the map holds (`0` for `NULL`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tensor_map_len(m: *const RlxTensorMap) -> usize {
    unsafe { m.as_ref() }.map_or(0, |m| m.entries.len())
}

/// Borrow entry `i`: name and data stay valid until `rlx_tensor_map_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tensor_map_get(
    m: *const RlxTensorMap,
    i: usize,
    name: *mut *const c_char,
    data: *mut *const f32,
    len: *mut usize,
) -> i32 {
    guard(|| unsafe {
        let m = m.as_ref().ok_or_else(|| null("tensor map"))?;
        let (n, d) = m
            .entries
            .get(i)
            .ok_or_else(|| ApiError::range(format!("tensor {i} of {}", m.entries.len())))?;
        out::put(name, n.as_ptr(), "rlx_tensor_map_get name")?;
        out::put(data, d.as_ptr(), "rlx_tensor_map_get data")?;
        out::put(len, d.len(), "rlx_tensor_map_get len")
    })
}

/// Free a tensor map (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tensor_map_free(m: *mut RlxTensorMap) {
    unsafe { release(m) }
}
