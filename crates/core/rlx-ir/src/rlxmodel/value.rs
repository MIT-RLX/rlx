// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The value tree between serde and the `.rlxmodel` text: any `Serialize`
//! becomes a [`Value`] that keeps enum variants and struct fields by name,
//! and a [`Value`] deserializes back into the type. Floats keep `NaN` and
//! `±inf` (JSON cannot), and a `Vec<u8>` stays bytes.

use serde::de::{self, DeserializeSeed, IntoDeserializer, Visitor};
use serde::ser::{self, Serialize};

/// A serialized value, by name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Unit,
    None,
    Bool(bool),
    Int(i128),
    /// A `u8` (a list of only these is [`Value::Bytes`]).
    Byte(u8),
    F32(f32),
    F64(f64),
    /// A number as written in text, read as whatever type wants it.
    Num(String),
    Str(String),
    Bytes(Vec<u8>),
    Seq(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Struct(Vec<(String, Value)>),
    /// An enum variant: unit (`[]` payload, no braces), tuple or struct.
    Variant(String, Payload),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Payload {
    Unit,
    /// Newtype and tuple variants alike: `Name(a, b)`.
    Tuple(Vec<Value>),
    Struct(Vec<(String, Value)>),
}

/// A serialization or deserialization error.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self(msg.to_string())
    }
}

impl de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self(msg.to_string())
    }
}

type Result<T> = std::result::Result<T, Error>;

pub(crate) fn to_value<T: Serialize + ?Sized>(v: &T) -> Result<Value> {
    v.serialize(Ser)
}

pub(crate) fn from_value<T: de::DeserializeOwned>(v: Value) -> Result<T> {
    T::deserialize(v)
}

// ── serialize ───────────────────────────────────────────────

struct Ser;

pub(crate) struct SerSeq(Vec<Value>);
pub(crate) struct SerVariant(&'static str, Vec<Value>);
pub(crate) struct SerMap(Vec<(Value, Value)>, Option<Value>);
pub(crate) struct SerStruct(Vec<(String, Value)>);
pub(crate) struct SerStructVariant(&'static str, Vec<(String, Value)>);

/// A list whose every element was a `u8` is bytes.
fn seq(items: Vec<Value>) -> Value {
    if !items.is_empty() && items.iter().all(|v| matches!(v, Value::Byte(_))) {
        return Value::Bytes(
            items
                .into_iter()
                .map(|v| match v {
                    Value::Byte(b) => b,
                    _ => unreachable!(),
                })
                .collect(),
        );
    }
    Value::Seq(items)
}

impl ser::Serializer for Ser {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = SerSeq;
    type SerializeTuple = SerSeq;
    type SerializeTupleStruct = SerSeq;
    type SerializeTupleVariant = SerVariant;
    type SerializeMap = SerMap;
    type SerializeStruct = SerStruct;
    type SerializeStructVariant = SerStructVariant;

    fn serialize_bool(self, v: bool) -> Result<Value> {
        Ok(Value::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_i16(self, v: i16) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_i32(self, v: i32) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_i64(self, v: i64) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_i128(self, v: i128) -> Result<Value> {
        Ok(Value::Int(v))
    }
    fn serialize_u8(self, v: u8) -> Result<Value> {
        Ok(Value::Byte(v))
    }
    fn serialize_u16(self, v: u16) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_u32(self, v: u32) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_u64(self, v: u64) -> Result<Value> {
        Ok(Value::Int(v.into()))
    }
    fn serialize_u128(self, v: u128) -> Result<Value> {
        i128::try_from(v)
            .map(Value::Int)
            .map_err(|_| Error(format!("{v} is out of range")))
    }
    fn serialize_f32(self, v: f32) -> Result<Value> {
        Ok(Value::F32(v))
    }
    fn serialize_f64(self, v: f64) -> Result<Value> {
        Ok(Value::F64(v))
    }
    fn serialize_char(self, v: char) -> Result<Value> {
        Ok(Value::Str(v.to_string()))
    }
    fn serialize_str(self, v: &str) -> Result<Value> {
        Ok(Value::Str(v.to_string()))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value> {
        Ok(Value::Bytes(v.to_vec()))
    }
    fn serialize_none(self) -> Result<Value> {
        Ok(Value::None)
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<Value> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Unit)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value> {
        Ok(Value::Unit)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value> {
        Ok(Value::Variant(variant.into(), Payload::Unit))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        v: &T,
    ) -> Result<Value> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<Value> {
        Ok(Value::Variant(
            variant.into(),
            Payload::Tuple(vec![v.serialize(Ser)?]),
        ))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<SerSeq> {
        Ok(SerSeq(Vec::with_capacity(len.unwrap_or(0))))
    }
    fn serialize_tuple(self, len: usize) -> Result<SerSeq> {
        Ok(SerSeq(Vec::with_capacity(len)))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<SerSeq> {
        Ok(SerSeq(Vec::with_capacity(len)))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<SerVariant> {
        Ok(SerVariant(variant, Vec::with_capacity(len)))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<SerMap> {
        Ok(SerMap(Vec::new(), None))
    }
    fn serialize_struct(self, _: &'static str, len: usize) -> Result<SerStruct> {
        Ok(SerStruct(Vec::with_capacity(len)))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<SerStructVariant> {
        Ok(SerStructVariant(variant, Vec::with_capacity(len)))
    }
}

impl ser::SerializeSeq for SerSeq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.0.push(v.serialize(Ser)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(seq(self.0))
    }
}

impl ser::SerializeTuple for SerSeq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, v)
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Seq(self.0))
    }
}

impl ser::SerializeTupleStruct for SerSeq {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        ser::SerializeSeq::serialize_element(self, v)
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Seq(self.0))
    }
}

impl ser::SerializeTupleVariant for SerVariant {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        self.1.push(v.serialize(Ser)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Variant(self.0.into(), Payload::Tuple(self.1)))
    }
}

impl ser::SerializeMap for SerMap {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<()> {
        self.1 = Some(k.serialize(Ser)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<()> {
        let k = self
            .1
            .take()
            .ok_or_else(|| Error("map value before its key".into()))?;
        self.0.push((k, v.serialize(Ser)?));
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Map(self.0))
    }
}

impl ser::SerializeStruct for SerStruct {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<()> {
        self.0.push((k.into(), v.serialize(Ser)?));
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Struct(self.0))
    }
}

impl ser::SerializeStructVariant for SerStructVariant {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<()> {
        self.1.push((k.into(), v.serialize(Ser)?));
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Variant(self.0.into(), Payload::Struct(self.1)))
    }
}

// ── deserialize ─────────────────────────────────────────────

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::Unit => "()",
            Value::None => "none",
            Value::Bool(_) => "a bool",
            Value::Int(_) | Value::Byte(_) | Value::Num(_) => "a number",
            Value::F32(_) | Value::F64(_) => "a float",
            Value::Str(_) => "a string",
            Value::Bytes(_) => "bytes",
            Value::Seq(_) => "a list",
            Value::Map(_) => "a map",
            Value::Struct(_) => "a struct",
            Value::Variant(..) => "an enum variant",
        }
    }

    fn expected(&self, what: &str) -> Error {
        Error(format!("expected {what}, found {}", self.kind()))
    }

    fn int(&self) -> Result<i128> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Byte(b) => Ok((*b).into()),
            Value::Num(s) => s
                .parse::<i128>()
                .map_err(|_| Error(format!("expected an integer, found {s}"))),
            other => Err(other.expected("an integer")),
        }
    }

    fn f64(&self) -> Result<f64> {
        match self {
            Value::F64(f) => Ok(*f),
            Value::F32(f) => Ok((*f).into()),
            Value::Int(i) => Ok(*i as f64),
            Value::Byte(b) => Ok((*b).into()),
            Value::Num(s) => parse_f64(s),
            other => Err(other.expected("a number")),
        }
    }

    fn f32(&self) -> Result<f32> {
        match self {
            Value::F32(f) => Ok(*f),
            // Parsed directly as f32, so a printed f32 reads back exactly.
            Value::Num(s) => parse_f32(s),
            other => other.f64().map(|f| f as f32),
        }
    }
}

/// `nan(0x…)`: a NaN by its bits — 8 hex digits an f32's, 16 an f64's.
fn nan_bits(s: &str) -> Result<Option<(u64, bool)>> {
    let Some(hex) = s.strip_prefix("nan(0x").and_then(|h| h.strip_suffix(')')) else {
        return Ok(None);
    };
    let bits = u64::from_str_radix(hex, 16).map_err(|_| Error(format!("bad NaN bits in {s}")))?;
    Ok(Some((bits, hex.len() > 8)))
}

fn parse_f64(s: &str) -> Result<f64> {
    if let Some((bits, wide)) = nan_bits(s)? {
        return Ok(if wide {
            f64::from_bits(bits)
        } else {
            f32::from_bits(bits as u32).into()
        });
    }
    match s {
        "nan" => Ok(f64::NAN),
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => s
            .parse::<f64>()
            .map_err(|_| Error(format!("expected a number, found {s}"))),
    }
}

fn parse_f32(s: &str) -> Result<f32> {
    if let Some((bits, wide)) = nan_bits(s)? {
        return Ok(if wide {
            f64::from_bits(bits) as f32
        } else {
            f32::from_bits(bits as u32)
        });
    }
    match s {
        "nan" => Ok(f32::NAN),
        "inf" => Ok(f32::INFINITY),
        "-inf" => Ok(f32::NEG_INFINITY),
        _ => s
            .parse::<f32>()
            .map_err(|_| Error(format!("expected a number, found {s}"))),
    }
}

macro_rules! de_int {
    ($($method:ident => $t:ty, $visit:ident;)*) => {$(
        fn $method<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
            let i = self.int()?;
            let x = <$t>::try_from(i)
                .map_err(|_| Error(format!("{i} is out of range for {}", stringify!($t))))?;
            v.$visit(x)
        }
    )*};
}

impl<'de> de::Deserializer<'de> for Value {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Unit => v.visit_unit(),
            Value::None => v.visit_none(),
            Value::Bool(b) => v.visit_bool(b),
            Value::Int(i) => match i64::try_from(i) {
                Ok(i) => v.visit_i64(i),
                Err(_) => v.visit_i128(i),
            },
            Value::Byte(b) => v.visit_u8(b),
            Value::F32(f) => v.visit_f32(f),
            Value::F64(f) => v.visit_f64(f),
            Value::Num(s) => match s.parse::<i64>() {
                Ok(i) => v.visit_i64(i),
                Err(_) => v.visit_f64(parse_f64(&s)?),
            },
            Value::Str(s) => v.visit_string(s),
            Value::Bytes(b) => v.visit_byte_buf(b),
            Value::Seq(items) => v.visit_seq(SeqDe(items.into_iter())),
            Value::Map(m) => v.visit_map(MapDe::new(m)),
            Value::Struct(fields) => v.visit_map(MapDe::fields(fields)),
            Value::Variant(..) => v.visit_enum(EnumDe(self, &[])),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Bool(b) => v.visit_bool(b),
            other => Err(other.expected("a bool")),
        }
    }

    de_int! {
        deserialize_i8 => i8, visit_i8;
        deserialize_i16 => i16, visit_i16;
        deserialize_i32 => i32, visit_i32;
        deserialize_i64 => i64, visit_i64;
        deserialize_i128 => i128, visit_i128;
        deserialize_u8 => u8, visit_u8;
        deserialize_u16 => u16, visit_u16;
        deserialize_u32 => u32, visit_u32;
        deserialize_u64 => u64, visit_u64;
        deserialize_u128 => u128, visit_u128;
    }

    fn deserialize_f32<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_f32(self.f32()?)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_f64(self.f64()?)
    }

    fn deserialize_char<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_string(v)
    }

    fn deserialize_str<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_string(v)
    }

    fn deserialize_string<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Str(s) => v.visit_string(s),
            other => Err(other.expected("a string")),
        }
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_byte_buf(v)
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Bytes(b) => v.visit_byte_buf(b),
            Value::Seq(_) => self.deserialize_seq(v),
            other => Err(other.expected("bytes")),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::None => v.visit_none(),
            other => v.visit_some(other),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Unit => v.visit_unit(),
            other => Err(other.expected("()")),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(self, _: &'static str, v: V) -> Result<V::Value> {
        self.deserialize_unit(v)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        v: V,
    ) -> Result<V::Value> {
        v.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Seq(items) => v.visit_seq(SeqDe(items.into_iter())),
            Value::Bytes(b) => v.visit_seq(SeqDe(
                b.into_iter()
                    .map(Value::Byte)
                    .collect::<Vec<_>>()
                    .into_iter(),
            )),
            other => Err(other.expected("a list")),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _: usize, v: V) -> Result<V::Value> {
        self.deserialize_seq(v)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _: &'static str,
        _: usize,
        v: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(v)
    }

    fn deserialize_map<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match self {
            Value::Map(m) => v.visit_map(MapDe::new(m)),
            Value::Struct(fields) => v.visit_map(MapDe::fields(fields)),
            other => Err(other.expected("a map")),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        _: &'static [&'static str],
        v: V,
    ) -> Result<V::Value> {
        match self {
            Value::Struct(fields) => v.visit_map(MapDe::fields(fields)),
            Value::Map(m) => v.visit_map(MapDe::new(m)),
            other => Err(other.expected(&format!("a `{name}`"))),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value> {
        match self {
            Value::Variant(..) => v.visit_enum(EnumDe(self, variants)),
            other => Err(other.expected(&format!("a `{name}` variant"))),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_string(v)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_unit()
    }
}

struct SeqDe(std::vec::IntoIter<Value>);

impl<'de> de::SeqAccess<'de> for SeqDe {
    type Error = Error;
    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        self.0.next().map(|v| seed.deserialize(v)).transpose()
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.0.len())
    }
}

struct MapDe {
    entries: std::vec::IntoIter<(Value, Value)>,
    value: Option<Value>,
}

impl MapDe {
    fn new(m: Vec<(Value, Value)>) -> Self {
        Self {
            entries: m.into_iter(),
            value: None,
        }
    }

    fn fields(fields: Vec<(String, Value)>) -> Self {
        Self::new(
            fields
                .into_iter()
                .map(|(k, v)| (Value::Str(k), v))
                .collect(),
        )
    }
}

impl<'de> de::MapAccess<'de> for MapDe {
    type Error = Error;
    fn next_key_seed<K: DeserializeSeed<'de>>(&mut self, seed: K) -> Result<Option<K::Value>> {
        match self.entries.next() {
            Some((k, v)) => {
                self.value = Some(v);
                seed.deserialize(k).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value> {
        let v = self
            .value
            .take()
            .ok_or_else(|| Error("map key without a value".into()))?;
        seed.deserialize(v)
    }
}

/// An enum variant, matched by name. Exact first, then ignoring case, so
/// the text can spell `F32` as `f32`.
struct EnumDe(Value, &'static [&'static str]);

impl<'de> de::EnumAccess<'de> for EnumDe {
    type Error = Error;
    type Variant = Payload;

    fn variant_seed<V: DeserializeSeed<'de>>(self, seed: V) -> Result<(V::Value, Payload)> {
        let Value::Variant(name, payload) = self.0 else {
            return Err(self.0.expected("an enum variant"));
        };
        let name = if self.1.contains(&name.as_str()) {
            name
        } else {
            match self.1.iter().find(|v| v.eq_ignore_ascii_case(&name)) {
                Some(v) => (*v).to_string(),
                None => name,
            }
        };
        let id: de::value::StringDeserializer<Error> = name.into_deserializer();
        Ok((seed.deserialize(id)?, payload))
    }
}

impl<'de> de::VariantAccess<'de> for Payload {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        match self {
            Payload::Unit => Ok(()),
            Payload::Tuple(v) if v.is_empty() => Ok(()),
            _ => Err(Error("expected a variant without fields".into())),
        }
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value> {
        match self {
            Payload::Tuple(mut v) if v.len() == 1 => seed.deserialize(v.remove(0)),
            _ => Err(Error("expected a variant with one value: `Name(x)`".into())),
        }
    }

    fn tuple_variant<V: Visitor<'de>>(self, _: usize, v: V) -> Result<V::Value> {
        match self {
            Payload::Tuple(items) => v.visit_seq(SeqDe(items.into_iter())),
            Payload::Unit => v.visit_seq(SeqDe(Vec::new().into_iter())),
            Payload::Struct(_) => Err(Error("expected `Name(a, b)`, found `Name{…}`".into())),
        }
    }

    fn struct_variant<V: Visitor<'de>>(self, _: &'static [&'static str], v: V) -> Result<V::Value> {
        match self {
            Payload::Struct(fields) => v.visit_map(MapDe::fields(fields)),
            Payload::Unit => v.visit_map(MapDe::new(Vec::new())),
            Payload::Tuple(_) => Err(Error("expected `Name{field: …}`, found `Name(…)`".into())),
        }
    }
}

// ── JSON, for the free-form `model { … }` block ─────────────

pub(crate) fn from_json(v: &serde_json::Value) -> Value {
    use serde_json::Value as J;
    match v {
        J::Null => Value::None,
        J::Bool(b) => Value::Bool(*b),
        J::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => Value::Int(i.into()),
            (_, Some(u)) => Value::Int(u.into()),
            _ => Value::F64(n.as_f64().unwrap_or(f64::NAN)),
        },
        J::String(s) => Value::Str(s.clone()),
        J::Array(a) => Value::Seq(a.iter().map(from_json).collect()),
        J::Object(o) => Value::Struct(o.iter().map(|(k, v)| (k.clone(), from_json(v))).collect()),
    }
}

/// A bare word (`task: embedding`) reads as the string it spells.
pub(crate) fn to_json(v: Value) -> Result<serde_json::Value> {
    use serde_json::Value as J;
    let number = |f: f64| {
        serde_json::Number::from_f64(f)
            .map(J::Number)
            .ok_or_else(|| Error(format!("{f} has no JSON form")))
    };
    Ok(match v {
        Value::Unit | Value::None => J::Null,
        Value::Bool(b) => J::Bool(b),
        Value::Int(i) => match i64::try_from(i) {
            Ok(i) => J::from(i),
            Err(_) => J::from(u64::try_from(i).map_err(|_| Error(format!("{i} is out of range")))?),
        },
        Value::Byte(b) => J::from(b),
        Value::F32(f) => number(f.into())?,
        Value::F64(f) => number(f)?,
        Value::Num(s) => match s.parse::<i64>() {
            Ok(i) => J::from(i),
            Err(_) => match s.parse::<u64>() {
                Ok(u) => J::from(u),
                Err(_) => number(parse_f64(&s)?)?,
            },
        },
        Value::Str(s) => J::String(s),
        Value::Bytes(b) => J::Array(b.into_iter().map(J::from).collect()),
        Value::Seq(items) => J::Array(items.into_iter().map(to_json).collect::<Result<_>>()?),
        Value::Struct(fields) => J::Object(
            fields
                .into_iter()
                .map(|(k, v)| Ok((k, to_json(v)?)))
                .collect::<Result<_>>()?,
        ),
        Value::Map(entries) => J::Object(
            entries
                .into_iter()
                .map(|(k, v)| match k {
                    Value::Str(k) => Ok((k, to_json(v)?)),
                    other => Err(Error(format!(
                        "a JSON key must be a string, found {}",
                        other.kind()
                    ))),
                })
                .collect::<Result<_>>()?,
        ),
        Value::Variant(name, Payload::Unit) => J::String(name),
        Value::Variant(name, Payload::Tuple(mut items)) if items.len() == 1 => {
            serde_json::json!({ name: to_json(items.remove(0))? })
        }
        Value::Variant(name, Payload::Tuple(items)) => serde_json::json!({
            name: items.into_iter().map(to_json).collect::<Result<Vec<_>>>()?
        }),
        Value::Variant(name, Payload::Struct(fields)) => {
            serde_json::json!({ name: to_json(Value::Struct(fields))? })
        }
    })
}

/// True if `v` holds a float JSON has no form for (`NaN`, `±inf`).
pub(crate) fn has_non_finite(v: &Value) -> bool {
    match v {
        Value::F32(f) => !f.is_finite(),
        Value::F64(f) => !f.is_finite(),
        Value::Seq(items) => items.iter().any(has_non_finite),
        Value::Map(entries) => entries
            .iter()
            .any(|(k, v)| has_non_finite(k) || has_non_finite(v)),
        Value::Struct(fields) => fields.iter().any(|(_, v)| has_non_finite(v)),
        Value::Variant(_, Payload::Tuple(items)) => items.iter().any(has_non_finite),
        Value::Variant(_, Payload::Struct(fields)) => fields.iter().any(|(_, v)| has_non_finite(v)),
        _ => false,
    }
}

// ── field paths, for symbolic attributes ────────────────────

/// The children of `v` with their path segment: a field name, or an index.
fn children_mut(v: &mut Value) -> Vec<(String, &mut Value)> {
    match v {
        Value::Seq(items) | Value::Variant(_, Payload::Tuple(items)) => items
            .iter_mut()
            .enumerate()
            .map(|(i, x)| (i.to_string(), x))
            .collect(),
        Value::Struct(fields) | Value::Variant(_, Payload::Struct(fields)) => {
            fields.iter_mut().map(|(k, x)| (k.clone(), x)).collect()
        }
        _ => Vec::new(),
    }
}

fn children(v: &Value) -> Vec<(String, &Value)> {
    match v {
        Value::Seq(items) | Value::Variant(_, Payload::Tuple(items)) => items
            .iter()
            .enumerate()
            .map(|(i, x)| (i.to_string(), x))
            .collect(),
        Value::Struct(fields) | Value::Variant(_, Payload::Struct(fields)) => {
            fields.iter().map(|(k, x)| (k.clone(), x)).collect()
        }
        _ => Vec::new(),
    }
}

fn is_int(v: &Value) -> bool {
    matches!(v, Value::Int(_) | Value::Byte(_) | Value::Num(_)) && v.int().is_ok()
}

/// Where `a` and `b` differ, if only in integer leaves: each such leaf's
/// path and its two values. `None` if anything else differs.
pub(crate) fn int_diffs(a: &Value, b: &Value) -> Option<Vec<(Vec<String>, i128, i128)>> {
    fn walk(
        a: &Value,
        b: &Value,
        path: &mut Vec<String>,
        out: &mut Vec<(Vec<String>, i128, i128)>,
    ) -> bool {
        if is_int(a) && is_int(b) {
            let (x, y) = (a.int().unwrap_or(0), b.int().unwrap_or(0));
            if x != y {
                out.push((path.clone(), x, y));
            }
            return true;
        }
        let same_shape = match (a, b) {
            (Value::Variant(n, p), Value::Variant(m, q)) => {
                n == m && std::mem::discriminant(p) == std::mem::discriminant(q)
            }
            (Value::Seq(x), Value::Seq(y)) => x.len() == y.len(),
            (Value::Struct(x), Value::Struct(y)) => x.len() == y.len(),
            _ => return a == b,
        };
        if !same_shape {
            return false;
        }
        let (ca, cb) = (children(a), children(b));
        if ca.len() != cb.len() {
            return false;
        }
        for ((ka, va), (kb, vb)) in ca.into_iter().zip(cb) {
            if ka != kb {
                return false;
            }
            path.push(ka);
            let ok = walk(va, vb, path, out);
            path.pop();
            if !ok {
                return false;
            }
        }
        true
    }
    let mut out = Vec::new();
    walk(a, b, &mut Vec::new(), &mut out).then_some(out)
}

/// The leaf at `path`.
pub(crate) fn at_mut<'a>(v: &'a mut Value, path: &[String]) -> Option<&'a mut Value> {
    let Some((first, rest)) = path.split_first() else {
        return Some(v);
    };
    let child = children_mut(v).into_iter().find(|(k, _)| k == first)?.1;
    at_mut(child, rest)
}

/// Every leaf of `v` that `pick` takes, with its path, replaced by what
/// `pick` returns for it.
pub(crate) fn take_leaves(
    v: &mut Value,
    pick: &mut dyn FnMut(&Value) -> Option<Value>,
    path: &mut Vec<String>,
    out: &mut Vec<(Vec<String>, Value)>,
) {
    if let Some(replacement) = pick(v) {
        out.push((path.clone(), std::mem::replace(v, replacement)));
        return;
    }
    for (k, child) in children_mut(v) {
        path.push(k);
        take_leaves(child, pick, path, out);
        path.pop();
    }
}

/// An `f32` as `.rlxmodel` writes it (`nan`, `-inf`, `nan(0x…)` included).
pub(crate) fn parse_f32_text(s: &str) -> Result<f32> {
    parse_f32(s)
}

/// An `f64` as `.rlxmodel` writes it.
pub(crate) fn parse_f64_text(s: &str) -> Result<f64> {
    parse_f64(s)
}
