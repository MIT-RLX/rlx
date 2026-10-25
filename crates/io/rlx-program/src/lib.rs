// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX programs: the host logic of a model — tokenizing, phonemizing, a
//! search, a decode loop — as a WebAssembly module a package carries beside
//! its graphs (an entry of kind `program`). rlx runs the module in a
//! sandbox; the module runs the package's functions through rlx, on any
//! device, and does the rest itself.
//!
//! A program is a type implementing [`Program`], exported with
//! [`program!`], built for `wasm32-unknown-unknown` as a `cdylib`:
//!
//! ```ignore
//! struct Echo;
//! impl rlx_program::Program for Echo {
//!     fn init(_config: serde_json::Value) -> Result<Self, String> { Ok(Echo) }
//!     fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
//!         let out = rlx_program::run("double", req.tensors)?;
//!         Ok(Message::tensors(out))
//!     }
//! }
//! rlx_program::program!(Echo);
//! ```
//!
//! The ABI, both sides of which live here:
//!
//! - the module exports `rlx_alloc(len) -> ptr`, `rlx_free(ptr, len)`,
//!   `rlx_init(config) -> response` and `rlx_call(task, request) ->
//!   response`, each message a `(ptr, len)` in its memory, each response a
//!   `u64` (`ptr << 32 | len`) the caller frees;
//! - rlx gives it `rlx.run(function, message) -> response` (a package
//!   function, its tensors in and out; a body `{from: {input: slot}, keep:
//!   {output: slot}}` reads and keeps tensors in rlx's slots, so a decoder's
//!   caches never cross into the module — `run_with`), `rlx.host(service,
//!   message) -> response` (the package's files and tokenizers, the slots)
//!   and `rlx.log(text)`.
//!
//! A [`Message`] is a JSON body and named tensors ([`encode`] /
//! [`decode`]); an error is a body `{"error": "…"}`.

use serde_json::{Value, json};

/// The format's tag and version, the first bytes of every message.
pub const MAGIC: &[u8; 4] = b"RLXM";

/// A named tensor: its element type (`"f32"`, `"i32"`, `"i64"`, `"u8"`,
/// `"f16"`, `"bf16"`, `"bool"`), shape and little-endian bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(name: impl Into<String>, dtype: &str, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.to_string(),
            shape,
            data,
        }
    }
    pub fn f32(name: impl Into<String>, shape: Vec<usize>, values: &[f32]) -> Self {
        Self::new(
            name,
            "f32",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    pub fn i32(name: impl Into<String>, shape: Vec<usize>, values: &[i32]) -> Self {
        Self::new(
            name,
            "i32",
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
    }
    /// The values as f32, whatever the element type.
    pub fn to_f32(&self) -> Vec<f32> {
        let d = &self.data;
        match self.dtype.as_str() {
            "i32" => d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
                .collect(),
            "i64" => d
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(b);
                    i64::from_le_bytes(a) as f32
                })
                .collect(),
            "u8" | "bool" => d.iter().map(|&b| f32::from(b)).collect(),
            "i8" => d.iter().map(|&b| f32::from(b as i8)).collect(),
            "f16" => d
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            "bf16" => d
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
            _ => d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        }
    }
    /// The values as i64, whatever the element type (floats truncated).
    pub fn to_i64(&self) -> Vec<i64> {
        match self.dtype.as_str() {
            "i64" => self
                .data
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(b);
                    i64::from_le_bytes(a)
                })
                .collect(),
            "i32" => self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect(),
            _ => self.to_f32().into_iter().map(|v| v as i64).collect(),
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// A request or a response: a JSON body and named tensors.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Message {
    pub body: Value,
    pub tensors: Vec<Tensor>,
}

impl Message {
    pub fn new(body: Value, tensors: Vec<Tensor>) -> Self {
        Self { body, tensors }
    }
    /// A body alone.
    pub fn body(body: Value) -> Self {
        Self {
            body,
            tensors: Vec::new(),
        }
    }
    /// Tensors alone.
    pub fn tensors(tensors: Vec<Tensor>) -> Self {
        Self {
            body: Value::Null,
            tensors,
        }
    }
    /// The error response for `message`.
    pub fn error(message: impl Into<String>) -> Self {
        Self::body(json!({"error": message.into()}))
    }
    /// The error a response says, when it is one.
    pub fn as_error(&self) -> Option<&str> {
        self.body.get("error").and_then(Value::as_str)
    }
    /// The tensor `name`.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// `message` as bytes: `RLXM`, the header's length (u32 LE), the header
/// (`{"body": …, "tensors": [{name, dtype, shape, len}, …]}`), then each
/// tensor's bytes in order.
pub fn encode(message: &Message) -> Vec<u8> {
    let tensors: Vec<Value> = message
        .tensors
        .iter()
        .map(|t| json!({"name": t.name, "dtype": t.dtype, "shape": t.shape, "len": t.data.len()}))
        .collect();
    let header = json!({"body": message.body, "tensors": tensors}).to_string();
    let total: usize = message.tensors.iter().map(|t| t.data.len()).sum();
    let mut out = Vec::with_capacity(8 + header.len() + total);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for t in &message.tensors {
        out.extend_from_slice(&t.data);
    }
    out
}

/// The message `bytes` hold ([`encode`]'s format).
pub fn decode(bytes: &[u8]) -> Result<Message, String> {
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        return Err("not an RLXM message".into());
    }
    let n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let header = bytes
        .get(8..8 + n)
        .ok_or("the message ends inside its header")?;
    let header: Value =
        serde_json::from_slice(header).map_err(|e| format!("the message header: {e}"))?;
    let mut at = 8 + n;
    let mut tensors = Vec::new();
    for t in header["tensors"].as_array().into_iter().flatten() {
        let len = t["len"].as_u64().ok_or("a tensor without its length")? as usize;
        let data = bytes
            .get(at..at + len)
            .ok_or("the message ends inside a tensor")?
            .to_vec();
        at += len;
        tensors.push(Tensor {
            name: t["name"].as_str().unwrap_or_default().to_string(),
            dtype: t["dtype"].as_str().unwrap_or("f32").to_string(),
            shape: t["shape"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .collect(),
            data,
        });
    }
    Ok(Message {
        body: header["body"].clone(),
        tensors,
    })
}

/// A model's host logic. rlx calls [`Program::init`] once, with the entry's
/// `config`, then [`Program::call`] for each task asked of the model:
///
/// - `"run"`: tensors in, tensors out;
/// - `"transcribe"`: tensor `audio` (f32), body `{sample_rate}` → body
///   `{text}`;
/// - `"synthesize"`: body `{text, options}` → tensor `audio` (f32), body
///   `{sample_rate}`;
/// - `"embed"`: body `{texts}` → tensor `embeddings` `[n, d]`;
/// - `"generate"`: body `{prompt, options}` → body `{text}`;
/// - `"chat"`: body `{messages: [{role, content}], options}` → body `{text}`;
/// - `"score"`: body `{query, passages}` → body `{scores}`.
pub trait Program: Sized {
    fn init(config: Value) -> Result<Self, String>;
    fn call(&mut self, task: &str, request: Message) -> Result<Message, String>;
}

#[cfg(target_arch = "wasm32")]
mod guest {
    use super::*;

    #[link(wasm_import_module = "rlx")]
    unsafe extern "C" {
        #[link_name = "run"]
        fn rlx_run(name: u32, name_len: u32, msg: u32, msg_len: u32) -> u64;
        #[link_name = "host"]
        fn rlx_host(name: u32, name_len: u32, msg: u32, msg_len: u32) -> u64;
        #[link_name = "log"]
        fn rlx_log(text: u32, len: u32);
    }

    /// Take the response `packed` (`ptr << 32 | len`), allocated by
    /// [`alloc`], back.
    pub(crate) fn take(packed: u64) -> Vec<u8> {
        let (ptr, len) = ((packed >> 32) as usize, (packed & 0xffff_ffff) as usize);
        if len == 0 {
            return Vec::new();
        }
        // SAFETY: the host wrote `len` bytes into a buffer `alloc` made
        // with exactly that capacity.
        unsafe { Vec::from_raw_parts(ptr as *mut u8, len, len) }
    }

    /// Leak `bytes` to the caller as `ptr << 32 | len`.
    pub(crate) fn give(bytes: Vec<u8>) -> u64 {
        let mut b = bytes.into_boxed_slice();
        let (ptr, len) = (b.as_mut_ptr() as u64, b.len() as u64);
        std::mem::forget(b);
        (ptr << 32) | len
    }

    pub(crate) fn call_import(
        f: unsafe extern "C" fn(u32, u32, u32, u32) -> u64,
        name: &str,
        message: &Message,
    ) -> Result<Message, String> {
        let bytes = encode(message);
        // SAFETY: both buffers live across the call.
        let packed = unsafe {
            f(
                name.as_ptr() as u32,
                name.len() as u32,
                bytes.as_ptr() as u32,
                bytes.len() as u32,
            )
        };
        let response = decode(&take(packed))?;
        match response.as_error() {
            Some(e) => Err(e.to_string()),
            None => Ok(response),
        }
    }

    pub fn run(function: &str, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, String> {
        Ok(call_import(rlx_run, function, &Message::tensors(inputs))?.tensors)
    }

    pub fn run_with(
        function: &str,
        inputs: Vec<Tensor>,
        from: &[(&str, &str)],
        keep: &[(&str, &str)],
    ) -> Result<Vec<Tensor>, String> {
        let map = |pairs: &[(&str, &str)]| {
            Value::Object(
                pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
                    .collect(),
            )
        };
        let body = json!({"from": map(from), "keep": map(keep)});
        Ok(call_import(rlx_run, function, &Message::new(body, inputs))?.tensors)
    }

    /// [`run`] with symbols no input's shape gives sized by name (the
    /// package's `dims`): an output as long as the program asks.
    pub fn run_sized(
        function: &str,
        inputs: Vec<Tensor>,
        dims: &[(&str, usize)],
    ) -> Result<Vec<Tensor>, String> {
        let dims: serde_json::Map<String, Value> = dims
            .iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();
        let body = json!({ "dims": dims });
        Ok(call_import(rlx_run, function, &Message::new(body, inputs))?.tensors)
    }

    /// [`run`] with the request body as the host reads it: `from` and
    /// `keep` (as [`run_with`]) and `dims` (as [`run_sized`]) together.
    pub fn run_body(
        function: &str,
        inputs: Vec<Tensor>,
        body: Value,
    ) -> Result<Vec<Tensor>, String> {
        Ok(call_import(rlx_run, function, &Message::new(body, inputs))?.tensors)
    }

    pub fn host(service: &str, request: Message) -> Result<Message, String> {
        call_import(rlx_host, service, &request)
    }

    pub fn log(text: &str) {
        // SAFETY: `text` lives across the call.
        unsafe { rlx_log(text.as_ptr() as u32, text.len() as u32) }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn rlx_alloc(len: u32) -> u32 {
        let mut b = vec![0u8; len as usize].into_boxed_slice();
        let ptr = b.as_mut_ptr() as u32;
        std::mem::forget(b);
        ptr
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn rlx_free(ptr: u32, len: u32) {
        if len == 0 {
            return;
        }
        // SAFETY: `ptr` came from `rlx_alloc(len)` or `give`.
        unsafe {
            drop(Vec::from_raw_parts(
                ptr as *mut u8,
                len as usize,
                len as usize,
            ))
        };
    }
}

#[cfg(target_arch = "wasm32")]
pub use guest::{host, log, run, run_body, run_sized, run_with};

/// What [`program!`] exports call; not for use otherwise.
#[doc(hidden)]
pub mod export {
    #[cfg(target_arch = "wasm32")]
    use super::*;

    /// The bytes the host passed at `(ptr, len)`, its allocation taken back.
    ///
    /// # Safety
    /// `(ptr, len)` is a buffer `rlx_alloc(len)` made.
    pub unsafe fn input(ptr: u32, len: u32) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        unsafe { Vec::from_raw_parts(ptr as *mut u8, len as usize, len as usize) }
    }

    /// `result` as a response, leaked to the host.
    #[cfg(target_arch = "wasm32")]
    pub fn respond(result: Result<Message, String>) -> u64 {
        let m = result.unwrap_or_else(Message::error);
        super::guest::give(encode(&m))
    }
}

/// Export `$t` (a [`Program`]) as the module's `rlx_init` and `rlx_call`.
#[macro_export]
macro_rules! program {
    ($t:ty) => {
        static mut RLX_PROGRAM: ::core::option::Option<$t> = ::core::option::Option::None;

        #[unsafe(no_mangle)]
        pub extern "C" fn rlx_init(ptr: u32, len: u32) -> u64 {
            // SAFETY: the host wrote the config into a buffer of ours.
            let bytes = unsafe { $crate::export::input(ptr, len) };
            let result = $crate::decode(&bytes).and_then(|m| <$t as $crate::Program>::init(m.body));
            match result {
                Ok(p) => {
                    // SAFETY: a module runs one call at a time.
                    unsafe { *::core::ptr::addr_of_mut!(RLX_PROGRAM) = Some(p) };
                    $crate::export::respond(Ok($crate::Message::default()))
                }
                Err(e) => $crate::export::respond(Err(e)),
            }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn rlx_call(task: u32, task_len: u32, ptr: u32, len: u32) -> u64 {
            // SAFETY: the host wrote both into buffers of ours.
            let task = unsafe { $crate::export::input(task, task_len) };
            let bytes = unsafe { $crate::export::input(ptr, len) };
            let task = ::std::string::String::from_utf8_lossy(&task).into_owned();
            // SAFETY: a module runs one call at a time.
            let program = unsafe { (*::core::ptr::addr_of_mut!(RLX_PROGRAM)).as_mut() };
            let result = match program {
                Some(p) => {
                    $crate::decode(&bytes).and_then(|m| <$t as $crate::Program>::call(p, &task, m))
                }
                None => Err("the program was not initialized".to_string()),
            };
            $crate::export::respond(result)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_round_trips() {
        let m = Message::new(
            json!({"text": "hello", "n": 3}),
            vec![
                Tensor::f32("audio", vec![2, 2], &[1.0, -2.5, 0.0, 3.25]),
                Tensor::i32("ids", vec![3], &[7, -1, 42]),
                Tensor::new("empty", "u8", vec![0], vec![]),
            ],
        );
        let back = decode(&encode(&m)).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.tensor("ids").unwrap().to_i64(), [7, -1, 42]);
        assert_eq!(
            back.tensor("audio").unwrap().to_f32(),
            [1.0, -2.5, 0.0, 3.25]
        );
        assert!(decode(b"nope").is_err());
        assert_eq!(Message::error("bad").as_error(), Some("bad"));
    }

    #[test]
    fn half_floats_decode() {
        let t = Tensor::new(
            "h",
            "f16",
            vec![4],
            [0x3c00u16, 0xc000, 0x0001, 0x7bff]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        let v = t.to_f32();
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], -2.0);
        assert!((v[2] - 5.960_464_5e-8).abs() < 1e-12);
        assert_eq!(v[3], 65504.0);
    }
}
