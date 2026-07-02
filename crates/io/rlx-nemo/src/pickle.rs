// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! A minimal, **non-executing** pickle virtual machine covering exactly
//! the opcode subset `torch.save` emits for a state dict. It does not
//! import modules or call arbitrary code: `GLOBAL`/`STACK_GLOBAL` push a
//! symbolic `(module, name)` reference, `REDUCE` only knows the handful
//! of torch reconstructors that build tensors, and `persistent_id`
//! resolves the storage tuple into a typed [`StorageRef`].
//!
//! The end product is a [`Value`] tree from which [`crate::torch`]
//! harvests the `name -> TensorMeta` table.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Result, anyhow, bail};

use crate::dtype::DType;

/// A torch storage referenced by a tensor: where in the zip its bytes
/// live (`key` → `data/<key>` member) and how to interpret them.
#[derive(Debug, Clone)]
pub struct StorageRef {
    pub dtype: DType,
    pub key: String,
    /// Element count declared by the persistent id. Retained for
    /// faithfulness/cross-checking; the on-disk entry size is authoritative.
    #[allow(dead_code)]
    pub numel: usize,
}

/// A tensor reconstructed by `torch._utils._rebuild_tensor_v2`.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub storage_key: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    /// Element offset of the view into its storage.
    pub offset: usize,
}

/// A decoded pickle value. Mutable containers are `Rc<RefCell<…>>` so
/// memo references and `SETITEMS`/`APPENDS`/`BUILD` mutate in place.
///
/// Some scalar variants (`Float`, `Bytes`) are produced by the VM to
/// faithfully decode the stream even though the state-dict harvester
/// never reads them back — hence the blanket `dead_code` allowance.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<Vec<u8>>),
    Tuple(Rc<Vec<Value>>),
    List(Rc<RefCell<Vec<Value>>>),
    /// Order-preserving mapping (dicts and OrderedDicts alike).
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(Rc<str>, Rc<str>),
    Storage(StorageRef),
    Tensor(TensorMeta),
    /// A `REDUCE`/`NEWOBJ` result we don't model — kept so traversal can
    /// skip it without error.
    Opaque,
    /// Internal stack marker for `MARK`/…/`SETITEMS` framing.
    Mark,
}

impl Value {
    fn as_i64(&self) -> Result<i64> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Bool(b) => Ok(i64::from(*b)),
            other => Err(anyhow!("expected int, got {other:?}")),
        }
    }
    fn as_usize_vec(&self) -> Result<Vec<usize>> {
        let items = match self {
            Value::Tuple(t) => t.as_slice().to_vec(),
            Value::List(l) => l.borrow().clone(),
            other => bail!("expected tuple/list of ints, got {other:?}"),
        };
        items
            .iter()
            .map(|v| {
                let n = v.as_i64()?;
                usize::try_from(n).map_err(|_| anyhow!("negative dim {n}"))
            })
            .collect()
    }
    fn as_str(&self) -> Result<String> {
        match self {
            Value::Str(s) => Ok(s.to_string()),
            Value::Int(i) => Ok(i.to_string()),
            other => Err(anyhow!("expected str, got {other:?}")),
        }
    }
}

/// Unpickle the top-level object from a `data.pkl` byte stream.
pub fn unpickle(data: &[u8]) -> Result<Value> {
    Machine::new(data).run()
}

struct Machine<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
    /// Next index used by the `MEMOIZE` opcode.
    memo_len: u32,
}

impl<'a> Machine<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
            memo_len: 0,
        }
    }

    // ── byte-stream readers ──────────────────────────────────────────
    fn u8(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| anyhow!("pickle: unexpected EOF"))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| anyhow!("pickle: read past EOF"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u16le(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32le(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32le(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    /// A newline-terminated text line (used by `GLOBAL`, text-mode ints).
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.u8()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.data[start..self.pos - 1]).into_owned())
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| anyhow!("pickle: stack underflow"))
    }

    /// Pop everything pushed since the last `MARK` (consuming the mark).
    fn pop_to_mark(&mut self) -> Result<Vec<Value>> {
        let idx = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or_else(|| anyhow!("pickle: no MARK on stack"))?;
        let items = self.stack.split_off(idx + 1);
        self.stack.pop(); // remove the mark itself
        Ok(items)
    }

    fn memoize(&mut self) -> Result<()> {
        let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| anyhow!("pickle: MEMOIZE with empty stack"))?;
        self.memo.insert(self.memo_len, top);
        self.memo_len += 1;
        Ok(())
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                // PROTO
                0x80 => {
                    let _proto = self.u8()?;
                }
                // FRAME (8-byte length, advisory)
                0x95 => {
                    self.take(8)?;
                }
                // STOP
                b'.' => return self.pop(),
                // MARK
                b'(' => self.stack.push(Value::Mark),
                // NONE / booleans
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                // empty containers
                b'}' => self
                    .stack
                    .push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                b']' => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b')' => self.stack.push(Value::Tuple(Rc::new(Vec::new()))),
                // ints
                b'K' => {
                    let v = self.u8()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'M' => {
                    let v = self.u16le()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'J' => {
                    let v = self.i32le()? as i64;
                    self.stack.push(Value::Int(v));
                }
                // LONG1 / LONG4: little-endian signed of n bytes
                0x8a => {
                    let n = self.u8()? as usize;
                    let v = self.read_long(n)?;
                    self.stack.push(Value::Int(v));
                }
                0x8b => {
                    let n = self.u32le()? as usize;
                    let v = self.read_long(n)?;
                    self.stack.push(Value::Int(v));
                }
                // BININT (text 'I' int) — rare, but cheap to support
                b'I' => {
                    let line = self.line()?;
                    let v: i64 = line.trim().parse().unwrap_or(0);
                    self.stack.push(Value::Int(v));
                }
                // BINFLOAT (big-endian f64)
                b'G' => {
                    let b = self.take(8)?;
                    let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    self.stack.push(Value::Float(v));
                }
                // unicode strings
                b'X' => {
                    let n = self.u32le()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(Rc::from(s.as_str())));
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(Rc::from(s.as_str())));
                }
                0x8d => {
                    let b = self.take(8)?;
                    let n = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
                        as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(Rc::from(s.as_str())));
                }
                // bytes
                b'C' => {
                    let n = self.u8()? as usize;
                    let v = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(Rc::new(v)));
                }
                b'B' => {
                    let n = self.u32le()? as usize;
                    let v = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(Rc::new(v)));
                }
                // short binstring 'U' (latin-1) — treat as str
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(Rc::from(s.as_str())));
                }
                // tuples
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a])));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a, b])));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a, b, c])));
                }
                b't' => {
                    let items = self.pop_to_mark()?;
                    self.stack.push(Value::Tuple(Rc::new(items)));
                }
                // list building
                b'a' => {
                    let v = self.pop()?;
                    self.with_list_top(|l| l.push(v))?;
                }
                b'e' => {
                    let items = self.pop_to_mark()?;
                    self.with_list_top(|l| l.extend(items))?;
                }
                // dict building
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    self.with_dict_top(|d| d.push((k, v)))?;
                }
                b'u' => {
                    let items = self.pop_to_mark()?;
                    if items.len() % 2 != 0 {
                        bail!("pickle: SETITEMS with odd count");
                    }
                    self.with_dict_top(|d| {
                        for pair in items.chunks_exact(2) {
                            d.push((pair[0].clone(), pair[1].clone()));
                        }
                    })?;
                }
                // memo
                0x94 => self.memoize()?,
                b'q' => {
                    let idx = self.u8()? as u32;
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle: BINPUT empty stack"))?;
                    self.memo.insert(idx, top);
                }
                b'r' => {
                    let idx = self.u32le()?;
                    let top = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle: LONG_BINPUT empty stack"))?;
                    self.memo.insert(idx, top);
                }
                b'h' => {
                    let idx = self.u8()? as u32;
                    let v = self
                        .memo
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle: BINGET miss {idx}"))?;
                    self.stack.push(v);
                }
                b'j' => {
                    let idx = self.u32le()?;
                    let v = self
                        .memo
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| anyhow!("pickle: LONG_BINGET miss {idx}"))?;
                    self.stack.push(v);
                }
                // globals
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(
                        Rc::from(module.as_str()),
                        Rc::from(name.as_str()),
                    ));
                }
                0x93 => {
                    let name = self.pop()?.as_str()?;
                    let module = self.pop()?.as_str()?;
                    self.stack.push(Value::Global(
                        Rc::from(module.as_str()),
                        Rc::from(name.as_str()),
                    ));
                }
                // persistent id (binary): top is the persid tuple
                b'Q' => {
                    let pid = self.pop()?;
                    let resolved = self.persistent_load(&pid)?;
                    self.stack.push(resolved);
                }
                // REDUCE: apply callable(args)
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let v = self.reduce(&func, &args)?;
                    self.stack.push(v);
                }
                // NEWOBJ: cls.__new__(cls, *args) — we don't model classes
                0x81 => {
                    let _args = self.pop()?;
                    let _cls = self.pop()?;
                    self.stack.push(Value::Opaque);
                }
                // BUILD: apply __setstate__ to the object below
                b'b' => {
                    let state = self.pop()?;
                    self.build(&state)?;
                }
                // EMPTY_SET / FROZENSET / ADDITEMS — not used by torch
                other => bail!(
                    "pickle: unsupported opcode 0x{other:02x} ('{}') at offset {}",
                    other as char,
                    self.pos - 1
                ),
            }
        }
    }

    fn read_long(&mut self, n: usize) -> Result<i64> {
        if n == 0 {
            return Ok(0);
        }
        let bytes = self.take(n)?;
        let mut v: i64 = 0;
        for (i, &b) in bytes.iter().enumerate() {
            v |= (b as i64) << (8 * i);
        }
        // sign-extend from the top byte.
        let bits = 8 * n;
        if bits < 64 && (bytes[n - 1] & 0x80) != 0 {
            v |= -1i64 << bits;
        }
        Ok(v)
    }

    fn with_list_top(&mut self, f: impl FnOnce(&mut Vec<Value>)) -> Result<()> {
        match self.stack.last() {
            Some(Value::List(l)) => {
                f(&mut l.borrow_mut());
                Ok(())
            }
            other => Err(anyhow!("pickle: APPEND(S) on non-list {other:?}")),
        }
    }
    fn with_dict_top(&mut self, f: impl FnOnce(&mut Vec<(Value, Value)>)) -> Result<()> {
        match self.stack.last() {
            Some(Value::Dict(d)) => {
                f(&mut d.borrow_mut());
                Ok(())
            }
            other => Err(anyhow!("pickle: SETITEM(S) on non-dict {other:?}")),
        }
    }

    /// Resolve a torch storage persistent id:
    /// `("storage", <StorageType>, key, location, numel)`.
    fn persistent_load(&self, pid: &Value) -> Result<Value> {
        let t = match pid {
            Value::Tuple(t) => t,
            other => bail!("pickle: persistent id must be a tuple, got {other:?}"),
        };
        if t.len() < 5 {
            bail!(
                "pickle: storage persistent id has {} elements (<5)",
                t.len()
            );
        }
        let tag = t[0].as_str().unwrap_or_default();
        if tag != "storage" {
            bail!("pickle: unsupported persistent id tag {tag:?}");
        }
        let dtype = match &t[1] {
            Value::Global(_, name) => DType::from_storage_name(name)
                .ok_or_else(|| anyhow!("pickle: unknown storage type {name}"))?,
            other => bail!("pickle: storage type must be a global, got {other:?}"),
        };
        let key = t[2].as_str()?;
        let numel = usize::try_from(t[4].as_i64()?).unwrap_or(0);
        Ok(Value::Storage(StorageRef { dtype, key, numel }))
    }

    /// Apply the small set of torch reconstructor callables we recognize.
    fn reduce(&self, func: &Value, args: &Value) -> Result<Value> {
        let (module, name) = match func {
            Value::Global(m, n) => (m.as_ref(), n.as_ref()),
            // Some streams reduce a memoized class; treat unknown as opaque.
            _ => return Ok(Value::Opaque),
        };
        let argv: Vec<Value> = match args {
            Value::Tuple(t) => t.as_slice().to_vec(),
            Value::None => Vec::new(),
            other => vec![other.clone()],
        };

        match (module, name) {
            ("collections", "OrderedDict") => Ok(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
            (_, "_rebuild_tensor_v2") | (_, "_rebuild_tensor") => self.rebuild_tensor(&argv),
            (_, "_rebuild_parameter") => {
                // (data, requires_grad, backward_hooks) → the tensor.
                argv.into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("pickle: _rebuild_parameter with no data"))
            }
            _ => Ok(Value::Opaque),
        }
    }

    fn rebuild_tensor(&self, argv: &[Value]) -> Result<Value> {
        // _rebuild_tensor_v2(storage, storage_offset, size, stride, ...)
        if argv.len() < 4 {
            bail!(
                "pickle: _rebuild_tensor_v2 needs ≥4 args, got {}",
                argv.len()
            );
        }
        let storage = match &argv[0] {
            Value::Storage(s) => s.clone(),
            other => bail!("pickle: tensor storage arg is {other:?}"),
        };
        let offset = usize::try_from(argv[1].as_i64()?).unwrap_or(0);
        let shape = argv[2].as_usize_vec()?;
        let stride = argv[3].as_usize_vec()?;
        Ok(Value::Tensor(TensorMeta {
            storage_key: storage.key,
            dtype: storage.dtype,
            shape,
            stride,
            offset,
        }))
    }

    /// `BUILD` (`obj.__setstate__(state)`). For the dict-like objects in a
    /// torch state dict this merges `state` (itself a mapping) into `obj`.
    fn build(&mut self, state: &Value) -> Result<()> {
        let Some(top) = self.stack.last().cloned() else {
            bail!("pickle: BUILD with empty stack");
        };
        if let Value::Dict(dst) = top {
            match state {
                Value::Dict(src) => dst.borrow_mut().extend(src.borrow().iter().cloned()),
                // OrderedDict.__setstate__ can receive a (state, items) tuple.
                Value::Tuple(t) => {
                    for item in t.iter() {
                        if let Value::Dict(src) = item {
                            dst.borrow_mut().extend(src.borrow().iter().cloned());
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpickle_simple_dict() {
        // pickletools-equivalent of {"a": 1, "b": 2}, protocol 2.
        // \x80\x02 } q\x00 ( X\x01..a K\x01 X\x01..b K\x02 u .
        let mut p: Vec<u8> = vec![0x80, 0x02, b'}', b'q', 0x00, b'('];
        p.extend_from_slice(&[b'X', 1, 0, 0, 0, b'a', b'K', 1]);
        p.extend_from_slice(&[b'X', 1, 0, 0, 0, b'b', b'K', 2]);
        p.push(b'u');
        p.push(b'.');
        let v = unpickle(&p).unwrap();
        match v {
            Value::Dict(d) => {
                let d = d.borrow();
                assert_eq!(d.len(), 2);
                assert!(matches!(&d[0].0, Value::Str(s) if s.as_ref() == "a"));
                assert!(matches!(d[0].1, Value::Int(1)));
                assert!(matches!(d[1].1, Value::Int(2)));
            }
            other => panic!("expected dict, got {other:?}"),
        }
    }

    #[test]
    fn long1_signed() {
        let mut m = Machine::new(&[]);
        // 0x80 as LONG1 single byte == -128 (sign extended).
        m.data = &[0x80];
        m.pos = 0;
        assert_eq!(m.read_long(1).unwrap(), -128);
        m.data = &[0xff, 0x00];
        m.pos = 0;
        assert_eq!(m.read_long(2).unwrap(), 255);
    }
}
