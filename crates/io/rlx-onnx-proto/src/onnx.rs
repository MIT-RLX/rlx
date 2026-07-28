// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// This file is generated. Do not edit
// @generated

// https://github.com/Manishearth/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]
#![allow(static_mut_refs)]
#![allow(mismatched_lifetime_syntaxes)]

#![cfg_attr(rustfmt, rustfmt_skip)]

#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
#![allow(unused_results)]

use protobuf::Message as Message_imported_for_functions;
use protobuf::ProtobufEnum as ProtobufEnum_imported_for_functions;

#[derive(PartialEq,Clone,Default)]
pub struct AttributeProto {
    // message fields
    name: ::protobuf::SingularField<::std::string::String>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    field_type: ::std::option::Option<AttributeProto_AttributeType>,
    f: ::std::option::Option<f32>,
    i: ::std::option::Option<i64>,
    s: ::protobuf::SingularField<::std::vec::Vec<u8>>,
    t: ::protobuf::SingularPtrField<TensorProto>,
    g: ::protobuf::SingularPtrField<GraphProto>,
    floats: ::std::vec::Vec<f32>,
    ints: ::std::vec::Vec<i64>,
    strings: ::protobuf::RepeatedField<::std::vec::Vec<u8>>,
    tensors: ::protobuf::RepeatedField<TensorProto>,
    graphs: ::protobuf::RepeatedField<GraphProto>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for AttributeProto {}

impl AttributeProto {
    pub fn new() -> AttributeProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static AttributeProto {
        static mut instance: ::protobuf::lazy::Lazy<AttributeProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const AttributeProto,
        };
        unsafe {
            instance.get(AttributeProto::new)
        }
    }

    // optional string name = 1;

    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        if self.name.is_none() {
            self.name.set_default();
        }
        self.name.as_mut().unwrap()
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        self.name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_name(&self) -> &str {
        match self.name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.name
    }

    fn mut_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.name
    }

    // optional string doc_string = 13;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }

    // optional .onnx.AttributeProto.AttributeType type = 20;

    pub fn clear_field_type(&mut self) {
        self.field_type = ::std::option::Option::None;
    }

    pub fn has_field_type(&self) -> bool {
        self.field_type.is_some()
    }

    // Param is passed by value, moved
    pub fn set_field_type(&mut self, v: AttributeProto_AttributeType) {
        self.field_type = ::std::option::Option::Some(v);
    }

    pub fn get_field_type(&self) -> AttributeProto_AttributeType {
        self.field_type.unwrap_or(AttributeProto_AttributeType::UNDEFINED)
    }

    fn get_field_type_for_reflect(&self) -> &::std::option::Option<AttributeProto_AttributeType> {
        &self.field_type
    }

    fn mut_field_type_for_reflect(&mut self) -> &mut ::std::option::Option<AttributeProto_AttributeType> {
        &mut self.field_type
    }

    // optional float f = 2;

    pub fn clear_f(&mut self) {
        self.f = ::std::option::Option::None;
    }

    pub fn has_f(&self) -> bool {
        self.f.is_some()
    }

    // Param is passed by value, moved
    pub fn set_f(&mut self, v: f32) {
        self.f = ::std::option::Option::Some(v);
    }

    pub fn get_f(&self) -> f32 {
        self.f.unwrap_or(0.)
    }

    fn get_f_for_reflect(&self) -> &::std::option::Option<f32> {
        &self.f
    }

    fn mut_f_for_reflect(&mut self) -> &mut ::std::option::Option<f32> {
        &mut self.f
    }

    // optional int64 i = 3;

    pub fn clear_i(&mut self) {
        self.i = ::std::option::Option::None;
    }

    pub fn has_i(&self) -> bool {
        self.i.is_some()
    }

    // Param is passed by value, moved
    pub fn set_i(&mut self, v: i64) {
        self.i = ::std::option::Option::Some(v);
    }

    pub fn get_i(&self) -> i64 {
        self.i.unwrap_or(0)
    }

    fn get_i_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.i
    }

    fn mut_i_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.i
    }

    // optional bytes s = 4;

    pub fn clear_s(&mut self) {
        self.s.clear();
    }

    pub fn has_s(&self) -> bool {
        self.s.is_some()
    }

    // Param is passed by value, moved
    pub fn set_s(&mut self, v: ::std::vec::Vec<u8>) {
        self.s = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_s(&mut self) -> &mut ::std::vec::Vec<u8> {
        if self.s.is_none() {
            self.s.set_default();
        }
        self.s.as_mut().unwrap()
    }

    // Take field
    pub fn take_s(&mut self) -> ::std::vec::Vec<u8> {
        self.s.take().unwrap_or_else(|| ::std::vec::Vec::new())
    }

    pub fn get_s(&self) -> &[u8] {
        match self.s.as_ref() {
            Some(v) => &v,
            None => &[],
        }
    }

    fn get_s_for_reflect(&self) -> &::protobuf::SingularField<::std::vec::Vec<u8>> {
        &self.s
    }

    fn mut_s_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::vec::Vec<u8>> {
        &mut self.s
    }

    // optional .onnx.TensorProto t = 5;

    pub fn clear_t(&mut self) {
        self.t.clear();
    }

    pub fn has_t(&self) -> bool {
        self.t.is_some()
    }

    // Param is passed by value, moved
    pub fn set_t(&mut self, v: TensorProto) {
        self.t = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_t(&mut self) -> &mut TensorProto {
        if self.t.is_none() {
            self.t.set_default();
        }
        self.t.as_mut().unwrap()
    }

    // Take field
    pub fn take_t(&mut self) -> TensorProto {
        self.t.take().unwrap_or_else(|| TensorProto::new())
    }

    pub fn get_t(&self) -> &TensorProto {
        self.t.as_ref().unwrap_or_else(|| TensorProto::default_instance())
    }

    fn get_t_for_reflect(&self) -> &::protobuf::SingularPtrField<TensorProto> {
        &self.t
    }

    fn mut_t_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<TensorProto> {
        &mut self.t
    }

    // optional .onnx.GraphProto g = 6;

    pub fn clear_g(&mut self) {
        self.g.clear();
    }

    pub fn has_g(&self) -> bool {
        self.g.is_some()
    }

    // Param is passed by value, moved
    pub fn set_g(&mut self, v: GraphProto) {
        self.g = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_g(&mut self) -> &mut GraphProto {
        if self.g.is_none() {
            self.g.set_default();
        }
        self.g.as_mut().unwrap()
    }

    // Take field
    pub fn take_g(&mut self) -> GraphProto {
        self.g.take().unwrap_or_else(|| GraphProto::new())
    }

    pub fn get_g(&self) -> &GraphProto {
        self.g.as_ref().unwrap_or_else(|| GraphProto::default_instance())
    }

    fn get_g_for_reflect(&self) -> &::protobuf::SingularPtrField<GraphProto> {
        &self.g
    }

    fn mut_g_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<GraphProto> {
        &mut self.g
    }

    // repeated float floats = 7;

    pub fn clear_floats(&mut self) {
        self.floats.clear();
    }

    // Param is passed by value, moved
    pub fn set_floats(&mut self, v: ::std::vec::Vec<f32>) {
        self.floats = v;
    }

    // Mutable pointer to the field.
    pub fn mut_floats(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.floats
    }

    // Take field
    pub fn take_floats(&mut self) -> ::std::vec::Vec<f32> {
        ::std::mem::replace(&mut self.floats, ::std::vec::Vec::new())
    }

    pub fn get_floats(&self) -> &[f32] {
        &self.floats
    }

    fn get_floats_for_reflect(&self) -> &::std::vec::Vec<f32> {
        &self.floats
    }

    fn mut_floats_for_reflect(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.floats
    }

    // repeated int64 ints = 8;

    pub fn clear_ints(&mut self) {
        self.ints.clear();
    }

    // Param is passed by value, moved
    pub fn set_ints(&mut self, v: ::std::vec::Vec<i64>) {
        self.ints = v;
    }

    // Mutable pointer to the field.
    pub fn mut_ints(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.ints
    }

    // Take field
    pub fn take_ints(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.ints, ::std::vec::Vec::new())
    }

    pub fn get_ints(&self) -> &[i64] {
        &self.ints
    }

    fn get_ints_for_reflect(&self) -> &::std::vec::Vec<i64> {
        &self.ints
    }

    fn mut_ints_for_reflect(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.ints
    }

    // repeated bytes strings = 9;

    pub fn clear_strings(&mut self) {
        self.strings.clear();
    }

    // Param is passed by value, moved
    pub fn set_strings(&mut self, v: ::protobuf::RepeatedField<::std::vec::Vec<u8>>) {
        self.strings = v;
    }

    // Mutable pointer to the field.
    pub fn mut_strings(&mut self) -> &mut ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &mut self.strings
    }

    // Take field
    pub fn take_strings(&mut self) -> ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        ::std::mem::replace(&mut self.strings, ::protobuf::RepeatedField::new())
    }

    pub fn get_strings(&self) -> &[::std::vec::Vec<u8>] {
        &self.strings
    }

    fn get_strings_for_reflect(&self) -> &::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &self.strings
    }

    fn mut_strings_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &mut self.strings
    }

    // repeated .onnx.TensorProto tensors = 10;

    pub fn clear_tensors(&mut self) {
        self.tensors.clear();
    }

    // Param is passed by value, moved
    pub fn set_tensors(&mut self, v: ::protobuf::RepeatedField<TensorProto>) {
        self.tensors = v;
    }

    // Mutable pointer to the field.
    pub fn mut_tensors(&mut self) -> &mut ::protobuf::RepeatedField<TensorProto> {
        &mut self.tensors
    }

    // Take field
    pub fn take_tensors(&mut self) -> ::protobuf::RepeatedField<TensorProto> {
        ::std::mem::replace(&mut self.tensors, ::protobuf::RepeatedField::new())
    }

    pub fn get_tensors(&self) -> &[TensorProto] {
        &self.tensors
    }

    fn get_tensors_for_reflect(&self) -> &::protobuf::RepeatedField<TensorProto> {
        &self.tensors
    }

    fn mut_tensors_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<TensorProto> {
        &mut self.tensors
    }

    // repeated .onnx.GraphProto graphs = 11;

    pub fn clear_graphs(&mut self) {
        self.graphs.clear();
    }

    // Param is passed by value, moved
    pub fn set_graphs(&mut self, v: ::protobuf::RepeatedField<GraphProto>) {
        self.graphs = v;
    }

    // Mutable pointer to the field.
    pub fn mut_graphs(&mut self) -> &mut ::protobuf::RepeatedField<GraphProto> {
        &mut self.graphs
    }

    // Take field
    pub fn take_graphs(&mut self) -> ::protobuf::RepeatedField<GraphProto> {
        ::std::mem::replace(&mut self.graphs, ::protobuf::RepeatedField::new())
    }

    pub fn get_graphs(&self) -> &[GraphProto] {
        &self.graphs
    }

    fn get_graphs_for_reflect(&self) -> &::protobuf::RepeatedField<GraphProto> {
        &self.graphs
    }

    fn mut_graphs_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<GraphProto> {
        &mut self.graphs
    }
}

impl ::protobuf::Message for AttributeProto {
    fn is_initialized(&self) -> bool {
        for v in &self.t {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.g {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.tensors {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.graphs {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.name)?;
                },
                13 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                20 => {
                    ::protobuf::rt::read_proto2_enum_with_unknown_fields_into(wire_type, is, &mut self.field_type, 20, &mut self.unknown_fields)?
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeFixed32 {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_float()?;
                    self.f = ::std::option::Option::Some(tmp);
                },
                3 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.i = ::std::option::Option::Some(tmp);
                },
                4 => {
                    ::protobuf::rt::read_singular_bytes_into(wire_type, is, &mut self.s)?;
                },
                5 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.t)?;
                },
                6 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.g)?;
                },
                7 => {
                    ::protobuf::rt::read_repeated_float_into(wire_type, is, &mut self.floats)?;
                },
                8 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.ints)?;
                },
                9 => {
                    ::protobuf::rt::read_repeated_bytes_into(wire_type, is, &mut self.strings)?;
                },
                10 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.tensors)?;
                },
                11 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.graphs)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.name.as_ref() {
            my_size += ::protobuf::rt::string_size(1, &v);
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(13, &v);
        }
        if let Some(v) = self.field_type {
            my_size += ::protobuf::rt::enum_size(20, v);
        }
        if let Some(v) = self.f {
            my_size += 5;
        }
        if let Some(v) = self.i {
            my_size += ::protobuf::rt::value_size(3, v, ::protobuf::wire_format::WireTypeVarint);
        }
        if let Some(ref v) = self.s.as_ref() {
            my_size += ::protobuf::rt::bytes_size(4, &v);
        }
        if let Some(ref v) = self.t.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if let Some(ref v) = self.g.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        my_size += 5 * self.floats.len() as u32;
        for value in &self.ints {
            my_size += ::protobuf::rt::value_size(8, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        for value in &self.strings {
            my_size += ::protobuf::rt::bytes_size(9, &value);
        };
        for value in &self.tensors {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        for value in &self.graphs {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.name.as_ref() {
            os.write_string(1, &v)?;
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(13, &v)?;
        }
        if let Some(v) = self.field_type {
            os.write_enum(20, v.value())?;
        }
        if let Some(v) = self.f {
            os.write_float(2, v)?;
        }
        if let Some(v) = self.i {
            os.write_int64(3, v)?;
        }
        if let Some(ref v) = self.s.as_ref() {
            os.write_bytes(4, &v)?;
        }
        if let Some(ref v) = self.t.as_ref() {
            os.write_tag(5, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if let Some(ref v) = self.g.as_ref() {
            os.write_tag(6, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        for v in &self.floats {
            os.write_float(7, *v)?;
        };
        for v in &self.ints {
            os.write_int64(8, *v)?;
        };
        for v in &self.strings {
            os.write_bytes(9, &v)?;
        };
        for v in &self.tensors {
            os.write_tag(10, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        for v in &self.graphs {
            os.write_tag(11, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for AttributeProto {
    fn new() -> AttributeProto {
        AttributeProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<AttributeProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "name",
                    AttributeProto::get_name_for_reflect,
                    AttributeProto::mut_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    AttributeProto::get_doc_string_for_reflect,
                    AttributeProto::mut_doc_string_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeEnum<AttributeProto_AttributeType>>(
                    "type",
                    AttributeProto::get_field_type_for_reflect,
                    AttributeProto::mut_field_type_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                    "f",
                    AttributeProto::get_f_for_reflect,
                    AttributeProto::mut_f_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "i",
                    AttributeProto::get_i_for_reflect,
                    AttributeProto::mut_i_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeBytes>(
                    "s",
                    AttributeProto::get_s_for_reflect,
                    AttributeProto::mut_s_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorProto>>(
                    "t",
                    AttributeProto::get_t_for_reflect,
                    AttributeProto::mut_t_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<GraphProto>>(
                    "g",
                    AttributeProto::get_g_for_reflect,
                    AttributeProto::mut_g_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                    "floats",
                    AttributeProto::get_floats_for_reflect,
                    AttributeProto::mut_floats_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "ints",
                    AttributeProto::get_ints_for_reflect,
                    AttributeProto::mut_ints_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeBytes>(
                    "strings",
                    AttributeProto::get_strings_for_reflect,
                    AttributeProto::mut_strings_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorProto>>(
                    "tensors",
                    AttributeProto::get_tensors_for_reflect,
                    AttributeProto::mut_tensors_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<GraphProto>>(
                    "graphs",
                    AttributeProto::get_graphs_for_reflect,
                    AttributeProto::mut_graphs_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<AttributeProto>(
                    "AttributeProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for AttributeProto {
    fn clear(&mut self) {
        self.clear_name();
        self.clear_doc_string();
        self.clear_field_type();
        self.clear_f();
        self.clear_i();
        self.clear_s();
        self.clear_t();
        self.clear_g();
        self.clear_floats();
        self.clear_ints();
        self.clear_strings();
        self.clear_tensors();
        self.clear_graphs();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for AttributeProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for AttributeProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum AttributeProto_AttributeType {
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
}

impl ::protobuf::ProtobufEnum for AttributeProto_AttributeType {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<AttributeProto_AttributeType> {
        match value {
            0 => ::std::option::Option::Some(AttributeProto_AttributeType::UNDEFINED),
            1 => ::std::option::Option::Some(AttributeProto_AttributeType::FLOAT),
            2 => ::std::option::Option::Some(AttributeProto_AttributeType::INT),
            3 => ::std::option::Option::Some(AttributeProto_AttributeType::STRING),
            4 => ::std::option::Option::Some(AttributeProto_AttributeType::TENSOR),
            5 => ::std::option::Option::Some(AttributeProto_AttributeType::GRAPH),
            6 => ::std::option::Option::Some(AttributeProto_AttributeType::FLOATS),
            7 => ::std::option::Option::Some(AttributeProto_AttributeType::INTS),
            8 => ::std::option::Option::Some(AttributeProto_AttributeType::STRINGS),
            9 => ::std::option::Option::Some(AttributeProto_AttributeType::TENSORS),
            10 => ::std::option::Option::Some(AttributeProto_AttributeType::GRAPHS),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [AttributeProto_AttributeType] = &[
            AttributeProto_AttributeType::UNDEFINED,
            AttributeProto_AttributeType::FLOAT,
            AttributeProto_AttributeType::INT,
            AttributeProto_AttributeType::STRING,
            AttributeProto_AttributeType::TENSOR,
            AttributeProto_AttributeType::GRAPH,
            AttributeProto_AttributeType::FLOATS,
            AttributeProto_AttributeType::INTS,
            AttributeProto_AttributeType::STRINGS,
            AttributeProto_AttributeType::TENSORS,
            AttributeProto_AttributeType::GRAPHS,
        ];
        values
    }

    fn enum_descriptor_static(_: ::std::option::Option<AttributeProto_AttributeType>) -> &'static ::protobuf::reflect::EnumDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::EnumDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::EnumDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                ::protobuf::reflect::EnumDescriptor::new("AttributeProto_AttributeType", file_descriptor_proto())
            })
        }
    }
}

impl ::std::marker::Copy for AttributeProto_AttributeType {
}

impl ::protobuf::reflect::ProtobufValue for AttributeProto_AttributeType {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Enum(self.descriptor())
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct ValueInfoProto {
    // message fields
    name: ::protobuf::SingularField<::std::string::String>,
    field_type: ::protobuf::SingularPtrField<TypeProto>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for ValueInfoProto {}

impl ValueInfoProto {
    pub fn new() -> ValueInfoProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static ValueInfoProto {
        static mut instance: ::protobuf::lazy::Lazy<ValueInfoProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ValueInfoProto,
        };
        unsafe {
            instance.get(ValueInfoProto::new)
        }
    }

    // optional string name = 1;

    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        if self.name.is_none() {
            self.name.set_default();
        }
        self.name.as_mut().unwrap()
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        self.name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_name(&self) -> &str {
        match self.name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.name
    }

    fn mut_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.name
    }

    // optional .onnx.TypeProto type = 2;

    pub fn clear_field_type(&mut self) {
        self.field_type.clear();
    }

    pub fn has_field_type(&self) -> bool {
        self.field_type.is_some()
    }

    // Param is passed by value, moved
    pub fn set_field_type(&mut self, v: TypeProto) {
        self.field_type = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_field_type(&mut self) -> &mut TypeProto {
        if self.field_type.is_none() {
            self.field_type.set_default();
        }
        self.field_type.as_mut().unwrap()
    }

    // Take field
    pub fn take_field_type(&mut self) -> TypeProto {
        self.field_type.take().unwrap_or_else(|| TypeProto::new())
    }

    pub fn get_field_type(&self) -> &TypeProto {
        self.field_type.as_ref().unwrap_or_else(|| TypeProto::default_instance())
    }

    fn get_field_type_for_reflect(&self) -> &::protobuf::SingularPtrField<TypeProto> {
        &self.field_type
    }

    fn mut_field_type_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<TypeProto> {
        &mut self.field_type
    }

    // optional string doc_string = 3;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }
}

impl ::protobuf::Message for ValueInfoProto {
    fn is_initialized(&self) -> bool {
        for v in &self.field_type {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.name)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.field_type)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.name.as_ref() {
            my_size += ::protobuf::rt::string_size(1, &v);
        }
        if let Some(ref v) = self.field_type.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(3, &v);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.name.as_ref() {
            os.write_string(1, &v)?;
        }
        if let Some(ref v) = self.field_type.as_ref() {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(3, &v)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for ValueInfoProto {
    fn new() -> ValueInfoProto {
        ValueInfoProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<ValueInfoProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "name",
                    ValueInfoProto::get_name_for_reflect,
                    ValueInfoProto::mut_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TypeProto>>(
                    "type",
                    ValueInfoProto::get_field_type_for_reflect,
                    ValueInfoProto::mut_field_type_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    ValueInfoProto::get_doc_string_for_reflect,
                    ValueInfoProto::mut_doc_string_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<ValueInfoProto>(
                    "ValueInfoProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for ValueInfoProto {
    fn clear(&mut self) {
        self.clear_name();
        self.clear_field_type();
        self.clear_doc_string();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ValueInfoProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ValueInfoProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct NodeProto {
    // message fields
    input: ::protobuf::RepeatedField<::std::string::String>,
    output: ::protobuf::RepeatedField<::std::string::String>,
    name: ::protobuf::SingularField<::std::string::String>,
    op_type: ::protobuf::SingularField<::std::string::String>,
    domain: ::protobuf::SingularField<::std::string::String>,
    attribute: ::protobuf::RepeatedField<AttributeProto>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for NodeProto {}

impl NodeProto {
    pub fn new() -> NodeProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static NodeProto {
        static mut instance: ::protobuf::lazy::Lazy<NodeProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const NodeProto,
        };
        unsafe {
            instance.get(NodeProto::new)
        }
    }

    // repeated string input = 1;

    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    // Param is passed by value, moved
    pub fn set_input(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.input = v;
    }

    // Mutable pointer to the field.
    pub fn mut_input(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.input
    }

    // Take field
    pub fn take_input(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.input, ::protobuf::RepeatedField::new())
    }

    pub fn get_input(&self) -> &[::std::string::String] {
        &self.input
    }

    fn get_input_for_reflect(&self) -> &::protobuf::RepeatedField<::std::string::String> {
        &self.input
    }

    fn mut_input_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.input
    }

    // repeated string output = 2;

    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    // Param is passed by value, moved
    pub fn set_output(&mut self, v: ::protobuf::RepeatedField<::std::string::String>) {
        self.output = v;
    }

    // Mutable pointer to the field.
    pub fn mut_output(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.output
    }

    // Take field
    pub fn take_output(&mut self) -> ::protobuf::RepeatedField<::std::string::String> {
        ::std::mem::replace(&mut self.output, ::protobuf::RepeatedField::new())
    }

    pub fn get_output(&self) -> &[::std::string::String] {
        &self.output
    }

    fn get_output_for_reflect(&self) -> &::protobuf::RepeatedField<::std::string::String> {
        &self.output
    }

    fn mut_output_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<::std::string::String> {
        &mut self.output
    }

    // optional string name = 3;

    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        if self.name.is_none() {
            self.name.set_default();
        }
        self.name.as_mut().unwrap()
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        self.name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_name(&self) -> &str {
        match self.name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.name
    }

    fn mut_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.name
    }

    // optional string op_type = 4;

    pub fn clear_op_type(&mut self) {
        self.op_type.clear();
    }

    pub fn has_op_type(&self) -> bool {
        self.op_type.is_some()
    }

    // Param is passed by value, moved
    pub fn set_op_type(&mut self, v: ::std::string::String) {
        self.op_type = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_op_type(&mut self) -> &mut ::std::string::String {
        if self.op_type.is_none() {
            self.op_type.set_default();
        }
        self.op_type.as_mut().unwrap()
    }

    // Take field
    pub fn take_op_type(&mut self) -> ::std::string::String {
        self.op_type.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_op_type(&self) -> &str {
        match self.op_type.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_op_type_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.op_type
    }

    fn mut_op_type_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.op_type
    }

    // optional string domain = 7;

    pub fn clear_domain(&mut self) {
        self.domain.clear();
    }

    pub fn has_domain(&self) -> bool {
        self.domain.is_some()
    }

    // Param is passed by value, moved
    pub fn set_domain(&mut self, v: ::std::string::String) {
        self.domain = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_domain(&mut self) -> &mut ::std::string::String {
        if self.domain.is_none() {
            self.domain.set_default();
        }
        self.domain.as_mut().unwrap()
    }

    // Take field
    pub fn take_domain(&mut self) -> ::std::string::String {
        self.domain.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_domain(&self) -> &str {
        match self.domain.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_domain_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.domain
    }

    fn mut_domain_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.domain
    }

    // repeated .onnx.AttributeProto attribute = 5;

    pub fn clear_attribute(&mut self) {
        self.attribute.clear();
    }

    // Param is passed by value, moved
    pub fn set_attribute(&mut self, v: ::protobuf::RepeatedField<AttributeProto>) {
        self.attribute = v;
    }

    // Mutable pointer to the field.
    pub fn mut_attribute(&mut self) -> &mut ::protobuf::RepeatedField<AttributeProto> {
        &mut self.attribute
    }

    // Take field
    pub fn take_attribute(&mut self) -> ::protobuf::RepeatedField<AttributeProto> {
        ::std::mem::replace(&mut self.attribute, ::protobuf::RepeatedField::new())
    }

    pub fn get_attribute(&self) -> &[AttributeProto] {
        &self.attribute
    }

    fn get_attribute_for_reflect(&self) -> &::protobuf::RepeatedField<AttributeProto> {
        &self.attribute
    }

    fn mut_attribute_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<AttributeProto> {
        &mut self.attribute
    }

    // optional string doc_string = 6;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }
}

impl ::protobuf::Message for NodeProto {
    fn is_initialized(&self) -> bool {
        for v in &self.attribute {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.input)?;
                },
                2 => {
                    ::protobuf::rt::read_repeated_string_into(wire_type, is, &mut self.output)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.name)?;
                },
                4 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.op_type)?;
                },
                7 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.domain)?;
                },
                5 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.attribute)?;
                },
                6 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        for value in &self.input {
            my_size += ::protobuf::rt::string_size(1, &value);
        };
        for value in &self.output {
            my_size += ::protobuf::rt::string_size(2, &value);
        };
        if let Some(ref v) = self.name.as_ref() {
            my_size += ::protobuf::rt::string_size(3, &v);
        }
        if let Some(ref v) = self.op_type.as_ref() {
            my_size += ::protobuf::rt::string_size(4, &v);
        }
        if let Some(ref v) = self.domain.as_ref() {
            my_size += ::protobuf::rt::string_size(7, &v);
        }
        for value in &self.attribute {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(6, &v);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        for v in &self.input {
            os.write_string(1, &v)?;
        };
        for v in &self.output {
            os.write_string(2, &v)?;
        };
        if let Some(ref v) = self.name.as_ref() {
            os.write_string(3, &v)?;
        }
        if let Some(ref v) = self.op_type.as_ref() {
            os.write_string(4, &v)?;
        }
        if let Some(ref v) = self.domain.as_ref() {
            os.write_string(7, &v)?;
        }
        for v in &self.attribute {
            os.write_tag(5, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(6, &v)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for NodeProto {
    fn new() -> NodeProto {
        NodeProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<NodeProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "input",
                    NodeProto::get_input_for_reflect,
                    NodeProto::mut_input_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "output",
                    NodeProto::get_output_for_reflect,
                    NodeProto::mut_output_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "name",
                    NodeProto::get_name_for_reflect,
                    NodeProto::mut_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "op_type",
                    NodeProto::get_op_type_for_reflect,
                    NodeProto::mut_op_type_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "domain",
                    NodeProto::get_domain_for_reflect,
                    NodeProto::mut_domain_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<AttributeProto>>(
                    "attribute",
                    NodeProto::get_attribute_for_reflect,
                    NodeProto::mut_attribute_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    NodeProto::get_doc_string_for_reflect,
                    NodeProto::mut_doc_string_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<NodeProto>(
                    "NodeProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for NodeProto {
    fn clear(&mut self) {
        self.clear_input();
        self.clear_output();
        self.clear_name();
        self.clear_op_type();
        self.clear_domain();
        self.clear_attribute();
        self.clear_doc_string();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for NodeProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for NodeProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct ModelProto {
    // message fields
    ir_version: ::std::option::Option<i64>,
    opset_import: ::protobuf::RepeatedField<OperatorSetIdProto>,
    producer_name: ::protobuf::SingularField<::std::string::String>,
    producer_version: ::protobuf::SingularField<::std::string::String>,
    domain: ::protobuf::SingularField<::std::string::String>,
    model_version: ::std::option::Option<i64>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    graph: ::protobuf::SingularPtrField<GraphProto>,
    metadata_props: ::protobuf::RepeatedField<StringStringEntryProto>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for ModelProto {}

impl ModelProto {
    pub fn new() -> ModelProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static ModelProto {
        static mut instance: ::protobuf::lazy::Lazy<ModelProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ModelProto,
        };
        unsafe {
            instance.get(ModelProto::new)
        }
    }

    // optional int64 ir_version = 1;

    pub fn clear_ir_version(&mut self) {
        self.ir_version = ::std::option::Option::None;
    }

    pub fn has_ir_version(&self) -> bool {
        self.ir_version.is_some()
    }

    // Param is passed by value, moved
    pub fn set_ir_version(&mut self, v: i64) {
        self.ir_version = ::std::option::Option::Some(v);
    }

    pub fn get_ir_version(&self) -> i64 {
        self.ir_version.unwrap_or(0)
    }

    fn get_ir_version_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.ir_version
    }

    fn mut_ir_version_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.ir_version
    }

    // repeated .onnx.OperatorSetIdProto opset_import = 8;

    pub fn clear_opset_import(&mut self) {
        self.opset_import.clear();
    }

    // Param is passed by value, moved
    pub fn set_opset_import(&mut self, v: ::protobuf::RepeatedField<OperatorSetIdProto>) {
        self.opset_import = v;
    }

    // Mutable pointer to the field.
    pub fn mut_opset_import(&mut self) -> &mut ::protobuf::RepeatedField<OperatorSetIdProto> {
        &mut self.opset_import
    }

    // Take field
    pub fn take_opset_import(&mut self) -> ::protobuf::RepeatedField<OperatorSetIdProto> {
        ::std::mem::replace(&mut self.opset_import, ::protobuf::RepeatedField::new())
    }

    pub fn get_opset_import(&self) -> &[OperatorSetIdProto] {
        &self.opset_import
    }

    fn get_opset_import_for_reflect(&self) -> &::protobuf::RepeatedField<OperatorSetIdProto> {
        &self.opset_import
    }

    fn mut_opset_import_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<OperatorSetIdProto> {
        &mut self.opset_import
    }

    // optional string producer_name = 2;

    pub fn clear_producer_name(&mut self) {
        self.producer_name.clear();
    }

    pub fn has_producer_name(&self) -> bool {
        self.producer_name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_producer_name(&mut self, v: ::std::string::String) {
        self.producer_name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_producer_name(&mut self) -> &mut ::std::string::String {
        if self.producer_name.is_none() {
            self.producer_name.set_default();
        }
        self.producer_name.as_mut().unwrap()
    }

    // Take field
    pub fn take_producer_name(&mut self) -> ::std::string::String {
        self.producer_name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_producer_name(&self) -> &str {
        match self.producer_name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_producer_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.producer_name
    }

    fn mut_producer_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.producer_name
    }

    // optional string producer_version = 3;

    pub fn clear_producer_version(&mut self) {
        self.producer_version.clear();
    }

    pub fn has_producer_version(&self) -> bool {
        self.producer_version.is_some()
    }

    // Param is passed by value, moved
    pub fn set_producer_version(&mut self, v: ::std::string::String) {
        self.producer_version = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_producer_version(&mut self) -> &mut ::std::string::String {
        if self.producer_version.is_none() {
            self.producer_version.set_default();
        }
        self.producer_version.as_mut().unwrap()
    }

    // Take field
    pub fn take_producer_version(&mut self) -> ::std::string::String {
        self.producer_version.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_producer_version(&self) -> &str {
        match self.producer_version.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_producer_version_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.producer_version
    }

    fn mut_producer_version_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.producer_version
    }

    // optional string domain = 4;

    pub fn clear_domain(&mut self) {
        self.domain.clear();
    }

    pub fn has_domain(&self) -> bool {
        self.domain.is_some()
    }

    // Param is passed by value, moved
    pub fn set_domain(&mut self, v: ::std::string::String) {
        self.domain = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_domain(&mut self) -> &mut ::std::string::String {
        if self.domain.is_none() {
            self.domain.set_default();
        }
        self.domain.as_mut().unwrap()
    }

    // Take field
    pub fn take_domain(&mut self) -> ::std::string::String {
        self.domain.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_domain(&self) -> &str {
        match self.domain.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_domain_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.domain
    }

    fn mut_domain_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.domain
    }

    // optional int64 model_version = 5;

    pub fn clear_model_version(&mut self) {
        self.model_version = ::std::option::Option::None;
    }

    pub fn has_model_version(&self) -> bool {
        self.model_version.is_some()
    }

    // Param is passed by value, moved
    pub fn set_model_version(&mut self, v: i64) {
        self.model_version = ::std::option::Option::Some(v);
    }

    pub fn get_model_version(&self) -> i64 {
        self.model_version.unwrap_or(0)
    }

    fn get_model_version_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.model_version
    }

    fn mut_model_version_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.model_version
    }

    // optional string doc_string = 6;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }

    // optional .onnx.GraphProto graph = 7;

    pub fn clear_graph(&mut self) {
        self.graph.clear();
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    // Param is passed by value, moved
    pub fn set_graph(&mut self, v: GraphProto) {
        self.graph = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_graph(&mut self) -> &mut GraphProto {
        if self.graph.is_none() {
            self.graph.set_default();
        }
        self.graph.as_mut().unwrap()
    }

    // Take field
    pub fn take_graph(&mut self) -> GraphProto {
        self.graph.take().unwrap_or_else(|| GraphProto::new())
    }

    pub fn get_graph(&self) -> &GraphProto {
        self.graph.as_ref().unwrap_or_else(|| GraphProto::default_instance())
    }

    fn get_graph_for_reflect(&self) -> &::protobuf::SingularPtrField<GraphProto> {
        &self.graph
    }

    fn mut_graph_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<GraphProto> {
        &mut self.graph
    }

    // repeated .onnx.StringStringEntryProto metadata_props = 14;

    pub fn clear_metadata_props(&mut self) {
        self.metadata_props.clear();
    }

    // Param is passed by value, moved
    pub fn set_metadata_props(&mut self, v: ::protobuf::RepeatedField<StringStringEntryProto>) {
        self.metadata_props = v;
    }

    // Mutable pointer to the field.
    pub fn mut_metadata_props(&mut self) -> &mut ::protobuf::RepeatedField<StringStringEntryProto> {
        &mut self.metadata_props
    }

    // Take field
    pub fn take_metadata_props(&mut self) -> ::protobuf::RepeatedField<StringStringEntryProto> {
        ::std::mem::replace(&mut self.metadata_props, ::protobuf::RepeatedField::new())
    }

    pub fn get_metadata_props(&self) -> &[StringStringEntryProto] {
        &self.metadata_props
    }

    fn get_metadata_props_for_reflect(&self) -> &::protobuf::RepeatedField<StringStringEntryProto> {
        &self.metadata_props
    }

    fn mut_metadata_props_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<StringStringEntryProto> {
        &mut self.metadata_props
    }
}

impl ::protobuf::Message for ModelProto {
    fn is_initialized(&self) -> bool {
        for v in &self.opset_import {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.graph {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.metadata_props {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.ir_version = ::std::option::Option::Some(tmp);
                },
                8 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.opset_import)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.producer_name)?;
                },
                3 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.producer_version)?;
                },
                4 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.domain)?;
                },
                5 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.model_version = ::std::option::Option::Some(tmp);
                },
                6 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                7 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.graph)?;
                },
                14 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.metadata_props)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(v) = self.ir_version {
            my_size += ::protobuf::rt::value_size(1, v, ::protobuf::wire_format::WireTypeVarint);
        }
        for value in &self.opset_import {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        if let Some(ref v) = self.producer_name.as_ref() {
            my_size += ::protobuf::rt::string_size(2, &v);
        }
        if let Some(ref v) = self.producer_version.as_ref() {
            my_size += ::protobuf::rt::string_size(3, &v);
        }
        if let Some(ref v) = self.domain.as_ref() {
            my_size += ::protobuf::rt::string_size(4, &v);
        }
        if let Some(v) = self.model_version {
            my_size += ::protobuf::rt::value_size(5, v, ::protobuf::wire_format::WireTypeVarint);
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(6, &v);
        }
        if let Some(ref v) = self.graph.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        for value in &self.metadata_props {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(v) = self.ir_version {
            os.write_int64(1, v)?;
        }
        for v in &self.opset_import {
            os.write_tag(8, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        if let Some(ref v) = self.producer_name.as_ref() {
            os.write_string(2, &v)?;
        }
        if let Some(ref v) = self.producer_version.as_ref() {
            os.write_string(3, &v)?;
        }
        if let Some(ref v) = self.domain.as_ref() {
            os.write_string(4, &v)?;
        }
        if let Some(v) = self.model_version {
            os.write_int64(5, v)?;
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(6, &v)?;
        }
        if let Some(ref v) = self.graph.as_ref() {
            os.write_tag(7, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        for v in &self.metadata_props {
            os.write_tag(14, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for ModelProto {
    fn new() -> ModelProto {
        ModelProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<ModelProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "ir_version",
                    ModelProto::get_ir_version_for_reflect,
                    ModelProto::mut_ir_version_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<OperatorSetIdProto>>(
                    "opset_import",
                    ModelProto::get_opset_import_for_reflect,
                    ModelProto::mut_opset_import_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "producer_name",
                    ModelProto::get_producer_name_for_reflect,
                    ModelProto::mut_producer_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "producer_version",
                    ModelProto::get_producer_version_for_reflect,
                    ModelProto::mut_producer_version_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "domain",
                    ModelProto::get_domain_for_reflect,
                    ModelProto::mut_domain_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "model_version",
                    ModelProto::get_model_version_for_reflect,
                    ModelProto::mut_model_version_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    ModelProto::get_doc_string_for_reflect,
                    ModelProto::mut_doc_string_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<GraphProto>>(
                    "graph",
                    ModelProto::get_graph_for_reflect,
                    ModelProto::mut_graph_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<StringStringEntryProto>>(
                    "metadata_props",
                    ModelProto::get_metadata_props_for_reflect,
                    ModelProto::mut_metadata_props_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<ModelProto>(
                    "ModelProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for ModelProto {
    fn clear(&mut self) {
        self.clear_ir_version();
        self.clear_opset_import();
        self.clear_producer_name();
        self.clear_producer_version();
        self.clear_domain();
        self.clear_model_version();
        self.clear_doc_string();
        self.clear_graph();
        self.clear_metadata_props();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for ModelProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ModelProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct StringStringEntryProto {
    // message fields
    key: ::protobuf::SingularField<::std::string::String>,
    value: ::protobuf::SingularField<::std::string::String>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for StringStringEntryProto {}

impl StringStringEntryProto {
    pub fn new() -> StringStringEntryProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static StringStringEntryProto {
        static mut instance: ::protobuf::lazy::Lazy<StringStringEntryProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const StringStringEntryProto,
        };
        unsafe {
            instance.get(StringStringEntryProto::new)
        }
    }

    // optional string key = 1;

    pub fn clear_key(&mut self) {
        self.key.clear();
    }

    pub fn has_key(&self) -> bool {
        self.key.is_some()
    }

    // Param is passed by value, moved
    pub fn set_key(&mut self, v: ::std::string::String) {
        self.key = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_key(&mut self) -> &mut ::std::string::String {
        if self.key.is_none() {
            self.key.set_default();
        }
        self.key.as_mut().unwrap()
    }

    // Take field
    pub fn take_key(&mut self) -> ::std::string::String {
        self.key.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_key(&self) -> &str {
        match self.key.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_key_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.key
    }

    fn mut_key_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.key
    }

    // optional string value = 2;

    pub fn clear_value(&mut self) {
        self.value.clear();
    }

    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }

    // Param is passed by value, moved
    pub fn set_value(&mut self, v: ::std::string::String) {
        self.value = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_value(&mut self) -> &mut ::std::string::String {
        if self.value.is_none() {
            self.value.set_default();
        }
        self.value.as_mut().unwrap()
    }

    // Take field
    pub fn take_value(&mut self) -> ::std::string::String {
        self.value.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_value(&self) -> &str {
        match self.value.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_value_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.value
    }

    fn mut_value_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.value
    }
}

impl ::protobuf::Message for StringStringEntryProto {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.key)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.value)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.key.as_ref() {
            my_size += ::protobuf::rt::string_size(1, &v);
        }
        if let Some(ref v) = self.value.as_ref() {
            my_size += ::protobuf::rt::string_size(2, &v);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.key.as_ref() {
            os.write_string(1, &v)?;
        }
        if let Some(ref v) = self.value.as_ref() {
            os.write_string(2, &v)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for StringStringEntryProto {
    fn new() -> StringStringEntryProto {
        StringStringEntryProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<StringStringEntryProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "key",
                    StringStringEntryProto::get_key_for_reflect,
                    StringStringEntryProto::mut_key_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "value",
                    StringStringEntryProto::get_value_for_reflect,
                    StringStringEntryProto::mut_value_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<StringStringEntryProto>(
                    "StringStringEntryProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for StringStringEntryProto {
    fn clear(&mut self) {
        self.clear_key();
        self.clear_value();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for StringStringEntryProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for StringStringEntryProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct GraphProto {
    // message fields
    node: ::protobuf::RepeatedField<NodeProto>,
    name: ::protobuf::SingularField<::std::string::String>,
    initializer: ::protobuf::RepeatedField<TensorProto>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    input: ::protobuf::RepeatedField<ValueInfoProto>,
    output: ::protobuf::RepeatedField<ValueInfoProto>,
    value_info: ::protobuf::RepeatedField<ValueInfoProto>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for GraphProto {}

impl GraphProto {
    pub fn new() -> GraphProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static GraphProto {
        static mut instance: ::protobuf::lazy::Lazy<GraphProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const GraphProto,
        };
        unsafe {
            instance.get(GraphProto::new)
        }
    }

    // repeated .onnx.NodeProto node = 1;

    pub fn clear_node(&mut self) {
        self.node.clear();
    }

    // Param is passed by value, moved
    pub fn set_node(&mut self, v: ::protobuf::RepeatedField<NodeProto>) {
        self.node = v;
    }

    // Mutable pointer to the field.
    pub fn mut_node(&mut self) -> &mut ::protobuf::RepeatedField<NodeProto> {
        &mut self.node
    }

    // Take field
    pub fn take_node(&mut self) -> ::protobuf::RepeatedField<NodeProto> {
        ::std::mem::replace(&mut self.node, ::protobuf::RepeatedField::new())
    }

    pub fn get_node(&self) -> &[NodeProto] {
        &self.node
    }

    fn get_node_for_reflect(&self) -> &::protobuf::RepeatedField<NodeProto> {
        &self.node
    }

    fn mut_node_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<NodeProto> {
        &mut self.node
    }

    // optional string name = 2;

    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        if self.name.is_none() {
            self.name.set_default();
        }
        self.name.as_mut().unwrap()
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        self.name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_name(&self) -> &str {
        match self.name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.name
    }

    fn mut_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.name
    }

    // repeated .onnx.TensorProto initializer = 5;

    pub fn clear_initializer(&mut self) {
        self.initializer.clear();
    }

    // Param is passed by value, moved
    pub fn set_initializer(&mut self, v: ::protobuf::RepeatedField<TensorProto>) {
        self.initializer = v;
    }

    // Mutable pointer to the field.
    pub fn mut_initializer(&mut self) -> &mut ::protobuf::RepeatedField<TensorProto> {
        &mut self.initializer
    }

    // Take field
    pub fn take_initializer(&mut self) -> ::protobuf::RepeatedField<TensorProto> {
        ::std::mem::replace(&mut self.initializer, ::protobuf::RepeatedField::new())
    }

    pub fn get_initializer(&self) -> &[TensorProto] {
        &self.initializer
    }

    fn get_initializer_for_reflect(&self) -> &::protobuf::RepeatedField<TensorProto> {
        &self.initializer
    }

    fn mut_initializer_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<TensorProto> {
        &mut self.initializer
    }

    // optional string doc_string = 10;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }

    // repeated .onnx.ValueInfoProto input = 11;

    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    // Param is passed by value, moved
    pub fn set_input(&mut self, v: ::protobuf::RepeatedField<ValueInfoProto>) {
        self.input = v;
    }

    // Mutable pointer to the field.
    pub fn mut_input(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.input
    }

    // Take field
    pub fn take_input(&mut self) -> ::protobuf::RepeatedField<ValueInfoProto> {
        ::std::mem::replace(&mut self.input, ::protobuf::RepeatedField::new())
    }

    pub fn get_input(&self) -> &[ValueInfoProto] {
        &self.input
    }

    fn get_input_for_reflect(&self) -> &::protobuf::RepeatedField<ValueInfoProto> {
        &self.input
    }

    fn mut_input_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.input
    }

    // repeated .onnx.ValueInfoProto output = 12;

    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    // Param is passed by value, moved
    pub fn set_output(&mut self, v: ::protobuf::RepeatedField<ValueInfoProto>) {
        self.output = v;
    }

    // Mutable pointer to the field.
    pub fn mut_output(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.output
    }

    // Take field
    pub fn take_output(&mut self) -> ::protobuf::RepeatedField<ValueInfoProto> {
        ::std::mem::replace(&mut self.output, ::protobuf::RepeatedField::new())
    }

    pub fn get_output(&self) -> &[ValueInfoProto] {
        &self.output
    }

    fn get_output_for_reflect(&self) -> &::protobuf::RepeatedField<ValueInfoProto> {
        &self.output
    }

    fn mut_output_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.output
    }

    // repeated .onnx.ValueInfoProto value_info = 13;

    pub fn clear_value_info(&mut self) {
        self.value_info.clear();
    }

    // Param is passed by value, moved
    pub fn set_value_info(&mut self, v: ::protobuf::RepeatedField<ValueInfoProto>) {
        self.value_info = v;
    }

    // Mutable pointer to the field.
    pub fn mut_value_info(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.value_info
    }

    // Take field
    pub fn take_value_info(&mut self) -> ::protobuf::RepeatedField<ValueInfoProto> {
        ::std::mem::replace(&mut self.value_info, ::protobuf::RepeatedField::new())
    }

    pub fn get_value_info(&self) -> &[ValueInfoProto] {
        &self.value_info
    }

    fn get_value_info_for_reflect(&self) -> &::protobuf::RepeatedField<ValueInfoProto> {
        &self.value_info
    }

    fn mut_value_info_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<ValueInfoProto> {
        &mut self.value_info
    }
}

impl ::protobuf::Message for GraphProto {
    fn is_initialized(&self) -> bool {
        for v in &self.node {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.initializer {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.input {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.output {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.value_info {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.node)?;
                },
                2 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.name)?;
                },
                5 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.initializer)?;
                },
                10 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                11 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.input)?;
                },
                12 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.output)?;
                },
                13 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.value_info)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        for value in &self.node {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        if let Some(ref v) = self.name.as_ref() {
            my_size += ::protobuf::rt::string_size(2, &v);
        }
        for value in &self.initializer {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(10, &v);
        }
        for value in &self.input {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        for value in &self.output {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        for value in &self.value_info {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        for v in &self.node {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        if let Some(ref v) = self.name.as_ref() {
            os.write_string(2, &v)?;
        }
        for v in &self.initializer {
            os.write_tag(5, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(10, &v)?;
        }
        for v in &self.input {
            os.write_tag(11, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        for v in &self.output {
            os.write_tag(12, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        for v in &self.value_info {
            os.write_tag(13, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for GraphProto {
    fn new() -> GraphProto {
        GraphProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<GraphProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<NodeProto>>(
                    "node",
                    GraphProto::get_node_for_reflect,
                    GraphProto::mut_node_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "name",
                    GraphProto::get_name_for_reflect,
                    GraphProto::mut_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorProto>>(
                    "initializer",
                    GraphProto::get_initializer_for_reflect,
                    GraphProto::mut_initializer_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    GraphProto::get_doc_string_for_reflect,
                    GraphProto::mut_doc_string_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<ValueInfoProto>>(
                    "input",
                    GraphProto::get_input_for_reflect,
                    GraphProto::mut_input_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<ValueInfoProto>>(
                    "output",
                    GraphProto::get_output_for_reflect,
                    GraphProto::mut_output_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<ValueInfoProto>>(
                    "value_info",
                    GraphProto::get_value_info_for_reflect,
                    GraphProto::mut_value_info_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<GraphProto>(
                    "GraphProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for GraphProto {
    fn clear(&mut self) {
        self.clear_node();
        self.clear_name();
        self.clear_initializer();
        self.clear_doc_string();
        self.clear_input();
        self.clear_output();
        self.clear_value_info();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for GraphProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for GraphProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TensorProto {
    // message fields
    dims: ::std::vec::Vec<i64>,
    data_type: ::std::option::Option<TensorProto_DataType>,
    segment: ::protobuf::SingularPtrField<TensorProto_Segment>,
    float_data: ::std::vec::Vec<f32>,
    int32_data: ::std::vec::Vec<i32>,
    string_data: ::protobuf::RepeatedField<::std::vec::Vec<u8>>,
    int64_data: ::std::vec::Vec<i64>,
    name: ::protobuf::SingularField<::std::string::String>,
    doc_string: ::protobuf::SingularField<::std::string::String>,
    raw_data: ::protobuf::SingularField<::std::vec::Vec<u8>>,
    double_data: ::std::vec::Vec<f64>,
    uint64_data: ::std::vec::Vec<u64>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TensorProto {}

impl TensorProto {
    pub fn new() -> TensorProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TensorProto {
        static mut instance: ::protobuf::lazy::Lazy<TensorProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TensorProto,
        };
        unsafe {
            instance.get(TensorProto::new)
        }
    }

    // repeated int64 dims = 1;

    pub fn clear_dims(&mut self) {
        self.dims.clear();
    }

    // Param is passed by value, moved
    pub fn set_dims(&mut self, v: ::std::vec::Vec<i64>) {
        self.dims = v;
    }

    // Mutable pointer to the field.
    pub fn mut_dims(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.dims
    }

    // Take field
    pub fn take_dims(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.dims, ::std::vec::Vec::new())
    }

    pub fn get_dims(&self) -> &[i64] {
        &self.dims
    }

    fn get_dims_for_reflect(&self) -> &::std::vec::Vec<i64> {
        &self.dims
    }

    fn mut_dims_for_reflect(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.dims
    }

    // optional .onnx.TensorProto.DataType data_type = 2;

    pub fn clear_data_type(&mut self) {
        self.data_type = ::std::option::Option::None;
    }

    pub fn has_data_type(&self) -> bool {
        self.data_type.is_some()
    }

    // Param is passed by value, moved
    pub fn set_data_type(&mut self, v: TensorProto_DataType) {
        self.data_type = ::std::option::Option::Some(v);
    }

    pub fn get_data_type(&self) -> TensorProto_DataType {
        self.data_type.unwrap_or(TensorProto_DataType::UNDEFINED)
    }

    fn get_data_type_for_reflect(&self) -> &::std::option::Option<TensorProto_DataType> {
        &self.data_type
    }

    fn mut_data_type_for_reflect(&mut self) -> &mut ::std::option::Option<TensorProto_DataType> {
        &mut self.data_type
    }

    // optional .onnx.TensorProto.Segment segment = 3;

    pub fn clear_segment(&mut self) {
        self.segment.clear();
    }

    pub fn has_segment(&self) -> bool {
        self.segment.is_some()
    }

    // Param is passed by value, moved
    pub fn set_segment(&mut self, v: TensorProto_Segment) {
        self.segment = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_segment(&mut self) -> &mut TensorProto_Segment {
        if self.segment.is_none() {
            self.segment.set_default();
        }
        self.segment.as_mut().unwrap()
    }

    // Take field
    pub fn take_segment(&mut self) -> TensorProto_Segment {
        self.segment.take().unwrap_or_else(|| TensorProto_Segment::new())
    }

    pub fn get_segment(&self) -> &TensorProto_Segment {
        self.segment.as_ref().unwrap_or_else(|| TensorProto_Segment::default_instance())
    }

    fn get_segment_for_reflect(&self) -> &::protobuf::SingularPtrField<TensorProto_Segment> {
        &self.segment
    }

    fn mut_segment_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<TensorProto_Segment> {
        &mut self.segment
    }

    // repeated float float_data = 4;

    pub fn clear_float_data(&mut self) {
        self.float_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_float_data(&mut self, v: ::std::vec::Vec<f32>) {
        self.float_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_float_data(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.float_data
    }

    // Take field
    pub fn take_float_data(&mut self) -> ::std::vec::Vec<f32> {
        ::std::mem::replace(&mut self.float_data, ::std::vec::Vec::new())
    }

    pub fn get_float_data(&self) -> &[f32] {
        &self.float_data
    }

    fn get_float_data_for_reflect(&self) -> &::std::vec::Vec<f32> {
        &self.float_data
    }

    fn mut_float_data_for_reflect(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.float_data
    }

    // repeated int32 int32_data = 5;

    pub fn clear_int32_data(&mut self) {
        self.int32_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_int32_data(&mut self, v: ::std::vec::Vec<i32>) {
        self.int32_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_int32_data(&mut self) -> &mut ::std::vec::Vec<i32> {
        &mut self.int32_data
    }

    // Take field
    pub fn take_int32_data(&mut self) -> ::std::vec::Vec<i32> {
        ::std::mem::replace(&mut self.int32_data, ::std::vec::Vec::new())
    }

    pub fn get_int32_data(&self) -> &[i32] {
        &self.int32_data
    }

    fn get_int32_data_for_reflect(&self) -> &::std::vec::Vec<i32> {
        &self.int32_data
    }

    fn mut_int32_data_for_reflect(&mut self) -> &mut ::std::vec::Vec<i32> {
        &mut self.int32_data
    }

    // repeated bytes string_data = 6;

    pub fn clear_string_data(&mut self) {
        self.string_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_string_data(&mut self, v: ::protobuf::RepeatedField<::std::vec::Vec<u8>>) {
        self.string_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_string_data(&mut self) -> &mut ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &mut self.string_data
    }

    // Take field
    pub fn take_string_data(&mut self) -> ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        ::std::mem::replace(&mut self.string_data, ::protobuf::RepeatedField::new())
    }

    pub fn get_string_data(&self) -> &[::std::vec::Vec<u8>] {
        &self.string_data
    }

    fn get_string_data_for_reflect(&self) -> &::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &self.string_data
    }

    fn mut_string_data_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<::std::vec::Vec<u8>> {
        &mut self.string_data
    }

    // repeated int64 int64_data = 7;

    pub fn clear_int64_data(&mut self) {
        self.int64_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_int64_data(&mut self, v: ::std::vec::Vec<i64>) {
        self.int64_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_int64_data(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.int64_data
    }

    // Take field
    pub fn take_int64_data(&mut self) -> ::std::vec::Vec<i64> {
        ::std::mem::replace(&mut self.int64_data, ::std::vec::Vec::new())
    }

    pub fn get_int64_data(&self) -> &[i64] {
        &self.int64_data
    }

    fn get_int64_data_for_reflect(&self) -> &::std::vec::Vec<i64> {
        &self.int64_data
    }

    fn mut_int64_data_for_reflect(&mut self) -> &mut ::std::vec::Vec<i64> {
        &mut self.int64_data
    }

    // optional string name = 8;

    pub fn clear_name(&mut self) {
        self.name.clear();
    }

    pub fn has_name(&self) -> bool {
        self.name.is_some()
    }

    // Param is passed by value, moved
    pub fn set_name(&mut self, v: ::std::string::String) {
        self.name = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_name(&mut self) -> &mut ::std::string::String {
        if self.name.is_none() {
            self.name.set_default();
        }
        self.name.as_mut().unwrap()
    }

    // Take field
    pub fn take_name(&mut self) -> ::std::string::String {
        self.name.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_name(&self) -> &str {
        match self.name.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_name_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.name
    }

    fn mut_name_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.name
    }

    // optional string doc_string = 12;

    pub fn clear_doc_string(&mut self) {
        self.doc_string.clear();
    }

    pub fn has_doc_string(&self) -> bool {
        self.doc_string.is_some()
    }

    // Param is passed by value, moved
    pub fn set_doc_string(&mut self, v: ::std::string::String) {
        self.doc_string = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_doc_string(&mut self) -> &mut ::std::string::String {
        if self.doc_string.is_none() {
            self.doc_string.set_default();
        }
        self.doc_string.as_mut().unwrap()
    }

    // Take field
    pub fn take_doc_string(&mut self) -> ::std::string::String {
        self.doc_string.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_doc_string(&self) -> &str {
        match self.doc_string.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_doc_string_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.doc_string
    }

    fn mut_doc_string_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.doc_string
    }

    // optional bytes raw_data = 9;

    pub fn clear_raw_data(&mut self) {
        self.raw_data.clear();
    }

    pub fn has_raw_data(&self) -> bool {
        self.raw_data.is_some()
    }

    // Param is passed by value, moved
    pub fn set_raw_data(&mut self, v: ::std::vec::Vec<u8>) {
        self.raw_data = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_raw_data(&mut self) -> &mut ::std::vec::Vec<u8> {
        if self.raw_data.is_none() {
            self.raw_data.set_default();
        }
        self.raw_data.as_mut().unwrap()
    }

    // Take field
    pub fn take_raw_data(&mut self) -> ::std::vec::Vec<u8> {
        self.raw_data.take().unwrap_or_else(|| ::std::vec::Vec::new())
    }

    pub fn get_raw_data(&self) -> &[u8] {
        match self.raw_data.as_ref() {
            Some(v) => &v,
            None => &[],
        }
    }

    fn get_raw_data_for_reflect(&self) -> &::protobuf::SingularField<::std::vec::Vec<u8>> {
        &self.raw_data
    }

    fn mut_raw_data_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::vec::Vec<u8>> {
        &mut self.raw_data
    }

    // repeated double double_data = 10;

    pub fn clear_double_data(&mut self) {
        self.double_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_double_data(&mut self, v: ::std::vec::Vec<f64>) {
        self.double_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_double_data(&mut self) -> &mut ::std::vec::Vec<f64> {
        &mut self.double_data
    }

    // Take field
    pub fn take_double_data(&mut self) -> ::std::vec::Vec<f64> {
        ::std::mem::replace(&mut self.double_data, ::std::vec::Vec::new())
    }

    pub fn get_double_data(&self) -> &[f64] {
        &self.double_data
    }

    fn get_double_data_for_reflect(&self) -> &::std::vec::Vec<f64> {
        &self.double_data
    }

    fn mut_double_data_for_reflect(&mut self) -> &mut ::std::vec::Vec<f64> {
        &mut self.double_data
    }

    // repeated uint64 uint64_data = 11;

    pub fn clear_uint64_data(&mut self) {
        self.uint64_data.clear();
    }

    // Param is passed by value, moved
    pub fn set_uint64_data(&mut self, v: ::std::vec::Vec<u64>) {
        self.uint64_data = v;
    }

    // Mutable pointer to the field.
    pub fn mut_uint64_data(&mut self) -> &mut ::std::vec::Vec<u64> {
        &mut self.uint64_data
    }

    // Take field
    pub fn take_uint64_data(&mut self) -> ::std::vec::Vec<u64> {
        ::std::mem::replace(&mut self.uint64_data, ::std::vec::Vec::new())
    }

    pub fn get_uint64_data(&self) -> &[u64] {
        &self.uint64_data
    }

    fn get_uint64_data_for_reflect(&self) -> &::std::vec::Vec<u64> {
        &self.uint64_data
    }

    fn mut_uint64_data_for_reflect(&mut self) -> &mut ::std::vec::Vec<u64> {
        &mut self.uint64_data
    }
}

impl ::protobuf::Message for TensorProto {
    fn is_initialized(&self) -> bool {
        for v in &self.segment {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.dims)?;
                },
                2 => {
                    ::protobuf::rt::read_proto2_enum_with_unknown_fields_into(wire_type, is, &mut self.data_type, 2, &mut self.unknown_fields)?
                },
                3 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.segment)?;
                },
                4 => {
                    ::protobuf::rt::read_repeated_float_into(wire_type, is, &mut self.float_data)?;
                },
                5 => {
                    ::protobuf::rt::read_repeated_int32_into(wire_type, is, &mut self.int32_data)?;
                },
                6 => {
                    ::protobuf::rt::read_repeated_bytes_into(wire_type, is, &mut self.string_data)?;
                },
                7 => {
                    ::protobuf::rt::read_repeated_int64_into(wire_type, is, &mut self.int64_data)?;
                },
                8 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.name)?;
                },
                12 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.doc_string)?;
                },
                9 => {
                    ::protobuf::rt::read_singular_bytes_into(wire_type, is, &mut self.raw_data)?;
                },
                10 => {
                    ::protobuf::rt::read_repeated_double_into(wire_type, is, &mut self.double_data)?;
                },
                11 => {
                    ::protobuf::rt::read_repeated_uint64_into(wire_type, is, &mut self.uint64_data)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        for value in &self.dims {
            my_size += ::protobuf::rt::value_size(1, *value, ::protobuf::wire_format::WireTypeVarint);
        };
        if let Some(v) = self.data_type {
            my_size += ::protobuf::rt::enum_size(2, v);
        }
        if let Some(ref v) = self.segment.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        if !self.float_data.is_empty() {
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size((self.float_data.len() * 4) as u32) + (self.float_data.len() * 4) as u32;
        }
        if !self.int32_data.is_empty() {
            my_size += ::protobuf::rt::vec_packed_varint_size(5, &self.int32_data);
        }
        for value in &self.string_data {
            my_size += ::protobuf::rt::bytes_size(6, &value);
        };
        if !self.int64_data.is_empty() {
            my_size += ::protobuf::rt::vec_packed_varint_size(7, &self.int64_data);
        }
        if let Some(ref v) = self.name.as_ref() {
            my_size += ::protobuf::rt::string_size(8, &v);
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            my_size += ::protobuf::rt::string_size(12, &v);
        }
        if let Some(ref v) = self.raw_data.as_ref() {
            my_size += ::protobuf::rt::bytes_size(9, &v);
        }
        if !self.double_data.is_empty() {
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size((self.double_data.len() * 8) as u32) + (self.double_data.len() * 8) as u32;
        }
        if !self.uint64_data.is_empty() {
            my_size += ::protobuf::rt::vec_packed_varint_size(11, &self.uint64_data);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        for v in &self.dims {
            os.write_int64(1, *v)?;
        };
        if let Some(v) = self.data_type {
            os.write_enum(2, v.value())?;
        }
        if let Some(ref v) = self.segment.as_ref() {
            os.write_tag(3, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        if !self.float_data.is_empty() {
            os.write_tag(4, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            // TODO: Data size is computed again, it should be cached
            os.write_raw_varint32((self.float_data.len() * 4) as u32)?;
            for v in &self.float_data {
                os.write_float_no_tag(*v)?;
            };
        }
        if !self.int32_data.is_empty() {
            os.write_tag(5, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            // TODO: Data size is computed again, it should be cached
            os.write_raw_varint32(::protobuf::rt::vec_packed_varint_data_size(&self.int32_data))?;
            for v in &self.int32_data {
                os.write_int32_no_tag(*v)?;
            };
        }
        for v in &self.string_data {
            os.write_bytes(6, &v)?;
        };
        if !self.int64_data.is_empty() {
            os.write_tag(7, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            // TODO: Data size is computed again, it should be cached
            os.write_raw_varint32(::protobuf::rt::vec_packed_varint_data_size(&self.int64_data))?;
            for v in &self.int64_data {
                os.write_int64_no_tag(*v)?;
            };
        }
        if let Some(ref v) = self.name.as_ref() {
            os.write_string(8, &v)?;
        }
        if let Some(ref v) = self.doc_string.as_ref() {
            os.write_string(12, &v)?;
        }
        if let Some(ref v) = self.raw_data.as_ref() {
            os.write_bytes(9, &v)?;
        }
        if !self.double_data.is_empty() {
            os.write_tag(10, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            // TODO: Data size is computed again, it should be cached
            os.write_raw_varint32((self.double_data.len() * 8) as u32)?;
            for v in &self.double_data {
                os.write_double_no_tag(*v)?;
            };
        }
        if !self.uint64_data.is_empty() {
            os.write_tag(11, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            // TODO: Data size is computed again, it should be cached
            os.write_raw_varint32(::protobuf::rt::vec_packed_varint_data_size(&self.uint64_data))?;
            for v in &self.uint64_data {
                os.write_uint64_no_tag(*v)?;
            };
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TensorProto {
    fn new() -> TensorProto {
        TensorProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<TensorProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "dims",
                    TensorProto::get_dims_for_reflect,
                    TensorProto::mut_dims_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeEnum<TensorProto_DataType>>(
                    "data_type",
                    TensorProto::get_data_type_for_reflect,
                    TensorProto::mut_data_type_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorProto_Segment>>(
                    "segment",
                    TensorProto::get_segment_for_reflect,
                    TensorProto::mut_segment_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                    "float_data",
                    TensorProto::get_float_data_for_reflect,
                    TensorProto::mut_float_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt32>(
                    "int32_data",
                    TensorProto::get_int32_data_for_reflect,
                    TensorProto::mut_int32_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeBytes>(
                    "string_data",
                    TensorProto::get_string_data_for_reflect,
                    TensorProto::mut_string_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "int64_data",
                    TensorProto::get_int64_data_for_reflect,
                    TensorProto::mut_int64_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "name",
                    TensorProto::get_name_for_reflect,
                    TensorProto::mut_name_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "doc_string",
                    TensorProto::get_doc_string_for_reflect,
                    TensorProto::mut_doc_string_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeBytes>(
                    "raw_data",
                    TensorProto::get_raw_data_for_reflect,
                    TensorProto::mut_raw_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeDouble>(
                    "double_data",
                    TensorProto::get_double_data_for_reflect,
                    TensorProto::mut_double_data_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeUint64>(
                    "uint64_data",
                    TensorProto::get_uint64_data_for_reflect,
                    TensorProto::mut_uint64_data_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TensorProto>(
                    "TensorProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TensorProto {
    fn clear(&mut self) {
        self.clear_dims();
        self.clear_data_type();
        self.clear_segment();
        self.clear_float_data();
        self.clear_int32_data();
        self.clear_string_data();
        self.clear_int64_data();
        self.clear_name();
        self.clear_doc_string();
        self.clear_raw_data();
        self.clear_double_data();
        self.clear_uint64_data();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TensorProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TensorProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TensorProto_Segment {
    // message fields
    begin: ::std::option::Option<i64>,
    end: ::std::option::Option<i64>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TensorProto_Segment {}

impl TensorProto_Segment {
    pub fn new() -> TensorProto_Segment {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TensorProto_Segment {
        static mut instance: ::protobuf::lazy::Lazy<TensorProto_Segment> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TensorProto_Segment,
        };
        unsafe {
            instance.get(TensorProto_Segment::new)
        }
    }

    // optional int64 begin = 1;

    pub fn clear_begin(&mut self) {
        self.begin = ::std::option::Option::None;
    }

    pub fn has_begin(&self) -> bool {
        self.begin.is_some()
    }

    // Param is passed by value, moved
    pub fn set_begin(&mut self, v: i64) {
        self.begin = ::std::option::Option::Some(v);
    }

    pub fn get_begin(&self) -> i64 {
        self.begin.unwrap_or(0)
    }

    fn get_begin_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.begin
    }

    fn mut_begin_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.begin
    }

    // optional int64 end = 2;

    pub fn clear_end(&mut self) {
        self.end = ::std::option::Option::None;
    }

    pub fn has_end(&self) -> bool {
        self.end.is_some()
    }

    // Param is passed by value, moved
    pub fn set_end(&mut self, v: i64) {
        self.end = ::std::option::Option::Some(v);
    }

    pub fn get_end(&self) -> i64 {
        self.end.unwrap_or(0)
    }

    fn get_end_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.end
    }

    fn mut_end_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.end
    }
}

impl ::protobuf::Message for TensorProto_Segment {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.begin = ::std::option::Option::Some(tmp);
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.end = ::std::option::Option::Some(tmp);
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(v) = self.begin {
            my_size += ::protobuf::rt::value_size(1, v, ::protobuf::wire_format::WireTypeVarint);
        }
        if let Some(v) = self.end {
            my_size += ::protobuf::rt::value_size(2, v, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(v) = self.begin {
            os.write_int64(1, v)?;
        }
        if let Some(v) = self.end {
            os.write_int64(2, v)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TensorProto_Segment {
    fn new() -> TensorProto_Segment {
        TensorProto_Segment::new()
    }

    fn descriptor_static(_: ::std::option::Option<TensorProto_Segment>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "begin",
                    TensorProto_Segment::get_begin_for_reflect,
                    TensorProto_Segment::mut_begin_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "end",
                    TensorProto_Segment::get_end_for_reflect,
                    TensorProto_Segment::mut_end_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TensorProto_Segment>(
                    "TensorProto_Segment",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TensorProto_Segment {
    fn clear(&mut self) {
        self.clear_begin();
        self.clear_end();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TensorProto_Segment {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TensorProto_Segment {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum TensorProto_DataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
}

impl ::protobuf::ProtobufEnum for TensorProto_DataType {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<TensorProto_DataType> {
        match value {
            0 => ::std::option::Option::Some(TensorProto_DataType::UNDEFINED),
            1 => ::std::option::Option::Some(TensorProto_DataType::FLOAT),
            2 => ::std::option::Option::Some(TensorProto_DataType::UINT8),
            3 => ::std::option::Option::Some(TensorProto_DataType::INT8),
            4 => ::std::option::Option::Some(TensorProto_DataType::UINT16),
            5 => ::std::option::Option::Some(TensorProto_DataType::INT16),
            6 => ::std::option::Option::Some(TensorProto_DataType::INT32),
            7 => ::std::option::Option::Some(TensorProto_DataType::INT64),
            8 => ::std::option::Option::Some(TensorProto_DataType::STRING),
            9 => ::std::option::Option::Some(TensorProto_DataType::BOOL),
            10 => ::std::option::Option::Some(TensorProto_DataType::FLOAT16),
            11 => ::std::option::Option::Some(TensorProto_DataType::DOUBLE),
            12 => ::std::option::Option::Some(TensorProto_DataType::UINT32),
            13 => ::std::option::Option::Some(TensorProto_DataType::UINT64),
            14 => ::std::option::Option::Some(TensorProto_DataType::COMPLEX64),
            15 => ::std::option::Option::Some(TensorProto_DataType::COMPLEX128),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [TensorProto_DataType] = &[
            TensorProto_DataType::UNDEFINED,
            TensorProto_DataType::FLOAT,
            TensorProto_DataType::UINT8,
            TensorProto_DataType::INT8,
            TensorProto_DataType::UINT16,
            TensorProto_DataType::INT16,
            TensorProto_DataType::INT32,
            TensorProto_DataType::INT64,
            TensorProto_DataType::STRING,
            TensorProto_DataType::BOOL,
            TensorProto_DataType::FLOAT16,
            TensorProto_DataType::DOUBLE,
            TensorProto_DataType::UINT32,
            TensorProto_DataType::UINT64,
            TensorProto_DataType::COMPLEX64,
            TensorProto_DataType::COMPLEX128,
        ];
        values
    }

    fn enum_descriptor_static(_: ::std::option::Option<TensorProto_DataType>) -> &'static ::protobuf::reflect::EnumDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::EnumDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::EnumDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                ::protobuf::reflect::EnumDescriptor::new("TensorProto_DataType", file_descriptor_proto())
            })
        }
    }
}

impl ::std::marker::Copy for TensorProto_DataType {
}

impl ::protobuf::reflect::ProtobufValue for TensorProto_DataType {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Enum(self.descriptor())
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TensorShapeProto {
    // message fields
    dim: ::protobuf::RepeatedField<TensorShapeProto_Dimension>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TensorShapeProto {}

impl TensorShapeProto {
    pub fn new() -> TensorShapeProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TensorShapeProto {
        static mut instance: ::protobuf::lazy::Lazy<TensorShapeProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TensorShapeProto,
        };
        unsafe {
            instance.get(TensorShapeProto::new)
        }
    }

    // repeated .onnx.TensorShapeProto.Dimension dim = 1;

    pub fn clear_dim(&mut self) {
        self.dim.clear();
    }

    // Param is passed by value, moved
    pub fn set_dim(&mut self, v: ::protobuf::RepeatedField<TensorShapeProto_Dimension>) {
        self.dim = v;
    }

    // Mutable pointer to the field.
    pub fn mut_dim(&mut self) -> &mut ::protobuf::RepeatedField<TensorShapeProto_Dimension> {
        &mut self.dim
    }

    // Take field
    pub fn take_dim(&mut self) -> ::protobuf::RepeatedField<TensorShapeProto_Dimension> {
        ::std::mem::replace(&mut self.dim, ::protobuf::RepeatedField::new())
    }

    pub fn get_dim(&self) -> &[TensorShapeProto_Dimension] {
        &self.dim
    }

    fn get_dim_for_reflect(&self) -> &::protobuf::RepeatedField<TensorShapeProto_Dimension> {
        &self.dim
    }

    fn mut_dim_for_reflect(&mut self) -> &mut ::protobuf::RepeatedField<TensorShapeProto_Dimension> {
        &mut self.dim
    }
}

impl ::protobuf::Message for TensorShapeProto {
    fn is_initialized(&self) -> bool {
        for v in &self.dim {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.dim)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        for value in &self.dim {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        for v in &self.dim {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TensorShapeProto {
    fn new() -> TensorShapeProto {
        TensorShapeProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<TensorShapeProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorShapeProto_Dimension>>(
                    "dim",
                    TensorShapeProto::get_dim_for_reflect,
                    TensorShapeProto::mut_dim_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TensorShapeProto>(
                    "TensorShapeProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TensorShapeProto {
    fn clear(&mut self) {
        self.clear_dim();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TensorShapeProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TensorShapeProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TensorShapeProto_Dimension {
    // message oneof groups
    value: ::std::option::Option<TensorShapeProto_Dimension_oneof_value>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TensorShapeProto_Dimension {}

#[derive(Clone,PartialEq)]
pub enum TensorShapeProto_Dimension_oneof_value {
    dim_value(i64),
    dim_param(::std::string::String),
}

impl TensorShapeProto_Dimension {
    pub fn new() -> TensorShapeProto_Dimension {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TensorShapeProto_Dimension {
        static mut instance: ::protobuf::lazy::Lazy<TensorShapeProto_Dimension> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TensorShapeProto_Dimension,
        };
        unsafe {
            instance.get(TensorShapeProto_Dimension::new)
        }
    }

    // optional int64 dim_value = 1;

    pub fn clear_dim_value(&mut self) {
        self.value = ::std::option::Option::None;
    }

    pub fn has_dim_value(&self) -> bool {
        match self.value {
            ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_value(..)) => true,
            _ => false,
        }
    }

    // Param is passed by value, moved
    pub fn set_dim_value(&mut self, v: i64) {
        self.value = ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_value(v))
    }

    pub fn get_dim_value(&self) -> i64 {
        match self.value {
            ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_value(v)) => v,
            _ => 0,
        }
    }

    // optional string dim_param = 2;

    pub fn clear_dim_param(&mut self) {
        self.value = ::std::option::Option::None;
    }

    pub fn has_dim_param(&self) -> bool {
        match self.value {
            ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(..)) => true,
            _ => false,
        }
    }

    // Param is passed by value, moved
    pub fn set_dim_param(&mut self, v: ::std::string::String) {
        self.value = ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(v))
    }

    // Mutable pointer to the field.
    pub fn mut_dim_param(&mut self) -> &mut ::std::string::String {
        if let ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(_)) = self.value {
        } else {
            self.value = ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(::std::string::String::new()));
        }
        match self.value {
            ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(ref mut v)) => v,
            _ => panic!(),
        }
    }

    // Take field
    pub fn take_dim_param(&mut self) -> ::std::string::String {
        if self.has_dim_param() {
            match self.value.take() {
                ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(v)) => v,
                _ => panic!(),
            }
        } else {
            ::std::string::String::new()
        }
    }

    pub fn get_dim_param(&self) -> &str {
        match self.value {
            ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(ref v)) => v,
            _ => "",
        }
    }
}

impl ::protobuf::Message for TensorShapeProto_Dimension {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    self.value = ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_value(is.read_int64()?));
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeLengthDelimited {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    self.value = ::std::option::Option::Some(TensorShapeProto_Dimension_oneof_value::dim_param(is.read_string()?));
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let ::std::option::Option::Some(ref v) = self.value {
            match v {
                &TensorShapeProto_Dimension_oneof_value::dim_value(v) => {
                    my_size += ::protobuf::rt::value_size(1, v, ::protobuf::wire_format::WireTypeVarint);
                },
                &TensorShapeProto_Dimension_oneof_value::dim_param(ref v) => {
                    my_size += ::protobuf::rt::string_size(2, &v);
                },
            };
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let ::std::option::Option::Some(ref v) = self.value {
            match v {
                &TensorShapeProto_Dimension_oneof_value::dim_value(v) => {
                    os.write_int64(1, v)?;
                },
                &TensorShapeProto_Dimension_oneof_value::dim_param(ref v) => {
                    os.write_string(2, v)?;
                },
            };
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TensorShapeProto_Dimension {
    fn new() -> TensorShapeProto_Dimension {
        TensorShapeProto_Dimension::new()
    }

    fn descriptor_static(_: ::std::option::Option<TensorShapeProto_Dimension>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_i64_accessor::<_>(
                    "dim_value",
                    TensorShapeProto_Dimension::has_dim_value,
                    TensorShapeProto_Dimension::get_dim_value,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_string_accessor::<_>(
                    "dim_param",
                    TensorShapeProto_Dimension::has_dim_param,
                    TensorShapeProto_Dimension::get_dim_param,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TensorShapeProto_Dimension>(
                    "TensorShapeProto_Dimension",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TensorShapeProto_Dimension {
    fn clear(&mut self) {
        self.clear_dim_value();
        self.clear_dim_param();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TensorShapeProto_Dimension {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TensorShapeProto_Dimension {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TypeProto {
    // message oneof groups
    value: ::std::option::Option<TypeProto_oneof_value>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TypeProto {}

#[derive(Clone,PartialEq)]
pub enum TypeProto_oneof_value {
    tensor_type(TypeProto_Tensor),
}

impl TypeProto {
    pub fn new() -> TypeProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TypeProto {
        static mut instance: ::protobuf::lazy::Lazy<TypeProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TypeProto,
        };
        unsafe {
            instance.get(TypeProto::new)
        }
    }

    // optional .onnx.TypeProto.Tensor tensor_type = 1;

    pub fn clear_tensor_type(&mut self) {
        self.value = ::std::option::Option::None;
    }

    pub fn has_tensor_type(&self) -> bool {
        match self.value {
            ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(..)) => true,
            _ => false,
        }
    }

    // Param is passed by value, moved
    pub fn set_tensor_type(&mut self, v: TypeProto_Tensor) {
        self.value = ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(v))
    }

    // Mutable pointer to the field.
    pub fn mut_tensor_type(&mut self) -> &mut TypeProto_Tensor {
        if let ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(_)) = self.value {
        } else {
            self.value = ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(TypeProto_Tensor::new()));
        }
        match self.value {
            ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(ref mut v)) => v,
            _ => panic!(),
        }
    }

    // Take field
    pub fn take_tensor_type(&mut self) -> TypeProto_Tensor {
        if self.has_tensor_type() {
            match self.value.take() {
                ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(v)) => v,
                _ => panic!(),
            }
        } else {
            TypeProto_Tensor::new()
        }
    }

    pub fn get_tensor_type(&self) -> &TypeProto_Tensor {
        match self.value {
            ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(ref v)) => v,
            _ => TypeProto_Tensor::default_instance(),
        }
    }
}

impl ::protobuf::Message for TypeProto {
    fn is_initialized(&self) -> bool {
        if let Some(TypeProto_oneof_value::tensor_type(ref v)) = self.value {
            if !v.is_initialized() {
                return false;
            }
        }
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeLengthDelimited {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    self.value = ::std::option::Option::Some(TypeProto_oneof_value::tensor_type(is.read_message()?));
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let ::std::option::Option::Some(ref v) = self.value {
            match v {
                &TypeProto_oneof_value::tensor_type(ref v) => {
                    let len = v.compute_size();
                    my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
                },
            };
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let ::std::option::Option::Some(ref v) = self.value {
            match v {
                &TypeProto_oneof_value::tensor_type(ref v) => {
                    os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
                    os.write_raw_varint32(v.get_cached_size())?;
                    v.write_to_with_cached_sizes(os)?;
                },
            };
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TypeProto {
    fn new() -> TypeProto {
        TypeProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<TypeProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_message_accessor::<_, TypeProto_Tensor>(
                    "tensor_type",
                    TypeProto::has_tensor_type,
                    TypeProto::get_tensor_type,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TypeProto>(
                    "TypeProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TypeProto {
    fn clear(&mut self) {
        self.clear_tensor_type();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TypeProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TypeProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct TypeProto_Tensor {
    // message fields
    elem_type: ::std::option::Option<TensorProto_DataType>,
    shape: ::protobuf::SingularPtrField<TensorShapeProto>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for TypeProto_Tensor {}

impl TypeProto_Tensor {
    pub fn new() -> TypeProto_Tensor {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static TypeProto_Tensor {
        static mut instance: ::protobuf::lazy::Lazy<TypeProto_Tensor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const TypeProto_Tensor,
        };
        unsafe {
            instance.get(TypeProto_Tensor::new)
        }
    }

    // optional .onnx.TensorProto.DataType elem_type = 1;

    pub fn clear_elem_type(&mut self) {
        self.elem_type = ::std::option::Option::None;
    }

    pub fn has_elem_type(&self) -> bool {
        self.elem_type.is_some()
    }

    // Param is passed by value, moved
    pub fn set_elem_type(&mut self, v: TensorProto_DataType) {
        self.elem_type = ::std::option::Option::Some(v);
    }

    pub fn get_elem_type(&self) -> TensorProto_DataType {
        self.elem_type.unwrap_or(TensorProto_DataType::UNDEFINED)
    }

    fn get_elem_type_for_reflect(&self) -> &::std::option::Option<TensorProto_DataType> {
        &self.elem_type
    }

    fn mut_elem_type_for_reflect(&mut self) -> &mut ::std::option::Option<TensorProto_DataType> {
        &mut self.elem_type
    }

    // optional .onnx.TensorShapeProto shape = 2;

    pub fn clear_shape(&mut self) {
        self.shape.clear();
    }

    pub fn has_shape(&self) -> bool {
        self.shape.is_some()
    }

    // Param is passed by value, moved
    pub fn set_shape(&mut self, v: TensorShapeProto) {
        self.shape = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_shape(&mut self) -> &mut TensorShapeProto {
        if self.shape.is_none() {
            self.shape.set_default();
        }
        self.shape.as_mut().unwrap()
    }

    // Take field
    pub fn take_shape(&mut self) -> TensorShapeProto {
        self.shape.take().unwrap_or_else(|| TensorShapeProto::new())
    }

    pub fn get_shape(&self) -> &TensorShapeProto {
        self.shape.as_ref().unwrap_or_else(|| TensorShapeProto::default_instance())
    }

    fn get_shape_for_reflect(&self) -> &::protobuf::SingularPtrField<TensorShapeProto> {
        &self.shape
    }

    fn mut_shape_for_reflect(&mut self) -> &mut ::protobuf::SingularPtrField<TensorShapeProto> {
        &mut self.shape
    }
}

impl ::protobuf::Message for TypeProto_Tensor {
    fn is_initialized(&self) -> bool {
        for v in &self.shape {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_proto2_enum_with_unknown_fields_into(wire_type, is, &mut self.elem_type, 1, &mut self.unknown_fields)?
                },
                2 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.shape)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(v) = self.elem_type {
            my_size += ::protobuf::rt::enum_size(1, v);
        }
        if let Some(ref v) = self.shape.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(v) = self.elem_type {
            os.write_enum(1, v.value())?;
        }
        if let Some(ref v) = self.shape.as_ref() {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for TypeProto_Tensor {
    fn new() -> TypeProto_Tensor {
        TypeProto_Tensor::new()
    }

    fn descriptor_static(_: ::std::option::Option<TypeProto_Tensor>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeEnum<TensorProto_DataType>>(
                    "elem_type",
                    TypeProto_Tensor::get_elem_type_for_reflect,
                    TypeProto_Tensor::mut_elem_type_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<TensorShapeProto>>(
                    "shape",
                    TypeProto_Tensor::get_shape_for_reflect,
                    TypeProto_Tensor::mut_shape_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<TypeProto_Tensor>(
                    "TypeProto_Tensor",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for TypeProto_Tensor {
    fn clear(&mut self) {
        self.clear_elem_type();
        self.clear_shape();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for TypeProto_Tensor {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for TypeProto_Tensor {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct OperatorSetIdProto {
    // message fields
    domain: ::protobuf::SingularField<::std::string::String>,
    version: ::std::option::Option<i64>,
    // special fields
    unknown_fields: ::protobuf::UnknownFields,
    cached_size: ::protobuf::CachedSize,
}

// see codegen.rs for the explanation why impl Sync explicitly
unsafe impl ::std::marker::Sync for OperatorSetIdProto {}

impl OperatorSetIdProto {
    pub fn new() -> OperatorSetIdProto {
        ::std::default::Default::default()
    }

    pub fn default_instance() -> &'static OperatorSetIdProto {
        static mut instance: ::protobuf::lazy::Lazy<OperatorSetIdProto> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const OperatorSetIdProto,
        };
        unsafe {
            instance.get(OperatorSetIdProto::new)
        }
    }

    // optional string domain = 1;

    pub fn clear_domain(&mut self) {
        self.domain.clear();
    }

    pub fn has_domain(&self) -> bool {
        self.domain.is_some()
    }

    // Param is passed by value, moved
    pub fn set_domain(&mut self, v: ::std::string::String) {
        self.domain = ::protobuf::SingularField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_domain(&mut self) -> &mut ::std::string::String {
        if self.domain.is_none() {
            self.domain.set_default();
        }
        self.domain.as_mut().unwrap()
    }

    // Take field
    pub fn take_domain(&mut self) -> ::std::string::String {
        self.domain.take().unwrap_or_else(|| ::std::string::String::new())
    }

    pub fn get_domain(&self) -> &str {
        match self.domain.as_ref() {
            Some(v) => &v,
            None => "",
        }
    }

    fn get_domain_for_reflect(&self) -> &::protobuf::SingularField<::std::string::String> {
        &self.domain
    }

    fn mut_domain_for_reflect(&mut self) -> &mut ::protobuf::SingularField<::std::string::String> {
        &mut self.domain
    }

    // optional int64 version = 2;

    pub fn clear_version(&mut self) {
        self.version = ::std::option::Option::None;
    }

    pub fn has_version(&self) -> bool {
        self.version.is_some()
    }

    // Param is passed by value, moved
    pub fn set_version(&mut self, v: i64) {
        self.version = ::std::option::Option::Some(v);
    }

    pub fn get_version(&self) -> i64 {
        self.version.unwrap_or(0)
    }

    fn get_version_for_reflect(&self) -> &::std::option::Option<i64> {
        &self.version
    }

    fn mut_version_for_reflect(&mut self) -> &mut ::std::option::Option<i64> {
        &mut self.version
    }
}

impl ::protobuf::Message for OperatorSetIdProto {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_string_into(wire_type, is, &mut self.domain)?;
                },
                2 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_int64()?;
                    self.version = ::std::option::Option::Some(tmp);
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.domain.as_ref() {
            my_size += ::protobuf::rt::string_size(1, &v);
        }
        if let Some(v) = self.version {
            my_size += ::protobuf::rt::value_size(2, v, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.domain.as_ref() {
            os.write_string(1, &v)?;
        }
        if let Some(v) = self.version {
            os.write_int64(2, v)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn ::std::any::Any {
        self as &dyn ::std::any::Any
    }
    fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
        self as &mut dyn ::std::any::Any
    }
    fn into_any(self: Box<Self>) -> ::std::boxed::Box<dyn ::std::any::Any> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        ::protobuf::MessageStatic::descriptor_static(None::<Self>)
    }
}

impl ::protobuf::MessageStatic for OperatorSetIdProto {
    fn new() -> OperatorSetIdProto {
        OperatorSetIdProto::new()
    }

    fn descriptor_static(_: ::std::option::Option<OperatorSetIdProto>) -> &'static ::protobuf::reflect::MessageDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::MessageDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                let mut fields = ::std::vec::Vec::new();
                fields.push(::protobuf::reflect::accessor::make_singular_field_accessor::<_, ::protobuf::types::ProtobufTypeString>(
                    "domain",
                    OperatorSetIdProto::get_domain_for_reflect,
                    OperatorSetIdProto::mut_domain_for_reflect,
                ));
                fields.push(::protobuf::reflect::accessor::make_option_accessor::<_, ::protobuf::types::ProtobufTypeInt64>(
                    "version",
                    OperatorSetIdProto::get_version_for_reflect,
                    OperatorSetIdProto::mut_version_for_reflect,
                ));
                ::protobuf::reflect::MessageDescriptor::new::<OperatorSetIdProto>(
                    "OperatorSetIdProto",
                    fields,
                    file_descriptor_proto()
                )
            })
        }
    }
}

impl ::protobuf::Clear for OperatorSetIdProto {
    fn clear(&mut self) {
        self.clear_domain();
        self.clear_version();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for OperatorSetIdProto {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for OperatorSetIdProto {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Message(self)
    }
}

#[derive(Clone,PartialEq,Eq,Debug,Hash)]
pub enum Version {
    _START_VERSION = 0,
    IR_VERSION_2017_10_10 = 1,
    IR_VERSION_2017_10_30 = 2,
    IR_VERSION = 3,
}

impl ::protobuf::ProtobufEnum for Version {
    fn value(&self) -> i32 {
        *self as i32
    }

    fn from_i32(value: i32) -> ::std::option::Option<Version> {
        match value {
            0 => ::std::option::Option::Some(Version::_START_VERSION),
            1 => ::std::option::Option::Some(Version::IR_VERSION_2017_10_10),
            2 => ::std::option::Option::Some(Version::IR_VERSION_2017_10_30),
            3 => ::std::option::Option::Some(Version::IR_VERSION),
            _ => ::std::option::Option::None
        }
    }

    fn values() -> &'static [Self] {
        static values: &'static [Version] = &[
            Version::_START_VERSION,
            Version::IR_VERSION_2017_10_10,
            Version::IR_VERSION_2017_10_30,
            Version::IR_VERSION,
        ];
        values
    }

    fn enum_descriptor_static(_: ::std::option::Option<Version>) -> &'static ::protobuf::reflect::EnumDescriptor {
        static mut descriptor: ::protobuf::lazy::Lazy<::protobuf::reflect::EnumDescriptor> = ::protobuf::lazy::Lazy {
            lock: ::protobuf::lazy::ONCE_INIT,
            ptr: 0 as *const ::protobuf::reflect::EnumDescriptor,
        };
        unsafe {
            descriptor.get(|| {
                ::protobuf::reflect::EnumDescriptor::new("Version", file_descriptor_proto())
            })
        }
    }
}

impl ::std::marker::Copy for Version {
}

impl ::protobuf::reflect::ProtobufValue for Version {
    fn as_ref(&self) -> ::protobuf::reflect::ProtobufValueRef {
        ::protobuf::reflect::ProtobufValueRef::Enum(self.descriptor())
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n\nonnx.proto\x12\x04onnx\"\x97\x04\n\x0eAttributeProto\x12\x12\n\x04na\
    me\x18\x01\x20\x01(\tR\x04name\x12\x1d\n\ndoc_string\x18\r\x20\x01(\tR\t\
    docString\x126\n\x04type\x18\x14\x20\x01(\x0e2\".onnx.AttributeProto.Att\
    ributeTypeR\x04type\x12\x0c\n\x01f\x18\x02\x20\x01(\x02R\x01f\x12\x0c\n\
    \x01i\x18\x03\x20\x01(\x03R\x01i\x12\x0c\n\x01s\x18\x04\x20\x01(\x0cR\
    \x01s\x12\x1f\n\x01t\x18\x05\x20\x01(\x0b2\x11.onnx.TensorProtoR\x01t\
    \x12\x1e\n\x01g\x18\x06\x20\x01(\x0b2\x10.onnx.GraphProtoR\x01g\x12\x16\
    \n\x06floats\x18\x07\x20\x03(\x02R\x06floats\x12\x12\n\x04ints\x18\x08\
    \x20\x03(\x03R\x04ints\x12\x18\n\x07strings\x18\t\x20\x03(\x0cR\x07strin\
    gs\x12+\n\x07tensors\x18\n\x20\x03(\x0b2\x11.onnx.TensorProtoR\x07tensor\
    s\x12(\n\x06graphs\x18\x0b\x20\x03(\x0b2\x10.onnx.GraphProtoR\x06graphs\
    \"\x91\x01\n\rAttributeType\x12\r\n\tUNDEFINED\x10\0\x12\t\n\x05FLOAT\
    \x10\x01\x12\x07\n\x03INT\x10\x02\x12\n\n\x06STRING\x10\x03\x12\n\n\x06T\
    ENSOR\x10\x04\x12\t\n\x05GRAPH\x10\x05\x12\n\n\x06FLOATS\x10\x06\x12\x08\
    \n\x04INTS\x10\x07\x12\x0b\n\x07STRINGS\x10\x08\x12\x0b\n\x07TENSORS\x10\
    \t\x12\n\n\x06GRAPHS\x10\n\"h\n\x0eValueInfoProto\x12\x12\n\x04name\x18\
    \x01\x20\x01(\tR\x04name\x12#\n\x04type\x18\x02\x20\x01(\x0b2\x0f.onnx.T\
    ypeProtoR\x04type\x12\x1d\n\ndoc_string\x18\x03\x20\x01(\tR\tdocString\"\
    \xd1\x01\n\tNodeProto\x12\x14\n\x05input\x18\x01\x20\x03(\tR\x05input\
    \x12\x16\n\x06output\x18\x02\x20\x03(\tR\x06output\x12\x12\n\x04name\x18\
    \x03\x20\x01(\tR\x04name\x12\x17\n\x07op_type\x18\x04\x20\x01(\tR\x06opT\
    ype\x12\x16\n\x06domain\x18\x07\x20\x01(\tR\x06domain\x122\n\tattribute\
    \x18\x05\x20\x03(\x0b2\x14.onnx.AttributeProtoR\tattribute\x12\x1d\n\ndo\
    c_string\x18\x06\x20\x01(\tR\tdocString\"\x81\x03\n\nModelProto\x12\x1d\
    \n\nir_version\x18\x01\x20\x01(\x03R\tirVersion\x12;\n\x0copset_import\
    \x18\x08\x20\x03(\x0b2\x18.onnx.OperatorSetIdProtoR\x0bopsetImport\x12#\
    \n\rproducer_name\x18\x02\x20\x01(\tR\x0cproducerName\x12)\n\x10producer\
    _version\x18\x03\x20\x01(\tR\x0fproducerVersion\x12\x16\n\x06domain\x18\
    \x04\x20\x01(\tR\x06domain\x12#\n\rmodel_version\x18\x05\x20\x01(\x03R\
    \x0cmodelVersion\x12\x1d\n\ndoc_string\x18\x06\x20\x01(\tR\tdocString\
    \x12&\n\x05graph\x18\x07\x20\x01(\x0b2\x10.onnx.GraphProtoR\x05graph\x12\
    C\n\x0emetadata_props\x18\x0e\x20\x03(\x0b2\x1c.onnx.StringStringEntryPr\
    otoR\rmetadataProps\"@\n\x16StringStringEntryProto\x12\x10\n\x03key\x18\
    \x01\x20\x01(\tR\x03key\x12\x14\n\x05value\x18\x02\x20\x01(\tR\x05value\
    \"\xa8\x02\n\nGraphProto\x12#\n\x04node\x18\x01\x20\x03(\x0b2\x0f.onnx.N\
    odeProtoR\x04node\x12\x12\n\x04name\x18\x02\x20\x01(\tR\x04name\x123\n\
    \x0binitializer\x18\x05\x20\x03(\x0b2\x11.onnx.TensorProtoR\x0binitializ\
    er\x12\x1d\n\ndoc_string\x18\n\x20\x01(\tR\tdocString\x12*\n\x05input\
    \x18\x0b\x20\x03(\x0b2\x14.onnx.ValueInfoProtoR\x05input\x12,\n\x06outpu\
    t\x18\x0c\x20\x03(\x0b2\x14.onnx.ValueInfoProtoR\x06output\x123\n\nvalue\
    _info\x18\r\x20\x03(\x0b2\x14.onnx.ValueInfoProtoR\tvalueInfo\"\xb3\x05\
    \n\x0bTensorProto\x12\x12\n\x04dims\x18\x01\x20\x03(\x03R\x04dims\x127\n\
    \tdata_type\x18\x02\x20\x01(\x0e2\x1a.onnx.TensorProto.DataTypeR\x08data\
    Type\x123\n\x07segment\x18\x03\x20\x01(\x0b2\x19.onnx.TensorProto.Segmen\
    tR\x07segment\x12!\n\nfloat_data\x18\x04\x20\x03(\x02R\tfloatDataB\x02\
    \x10\x01\x12!\n\nint32_data\x18\x05\x20\x03(\x05R\tint32DataB\x02\x10\
    \x01\x12\x1f\n\x0bstring_data\x18\x06\x20\x03(\x0cR\nstringData\x12!\n\n\
    int64_data\x18\x07\x20\x03(\x03R\tint64DataB\x02\x10\x01\x12\x12\n\x04na\
    me\x18\x08\x20\x01(\tR\x04name\x12\x1d\n\ndoc_string\x18\x0c\x20\x01(\tR\
    \tdocString\x12\x19\n\x08raw_data\x18\t\x20\x01(\x0cR\x07rawData\x12#\n\
    \x0bdouble_data\x18\n\x20\x03(\x01R\ndoubleDataB\x02\x10\x01\x12#\n\x0bu\
    int64_data\x18\x0b\x20\x03(\x04R\nuint64DataB\x02\x10\x01\x1a1\n\x07Segm\
    ent\x12\x14\n\x05begin\x18\x01\x20\x01(\x03R\x05begin\x12\x10\n\x03end\
    \x18\x02\x20\x01(\x03R\x03end\"\xcc\x01\n\x08DataType\x12\r\n\tUNDEFINED\
    \x10\0\x12\t\n\x05FLOAT\x10\x01\x12\t\n\x05UINT8\x10\x02\x12\x08\n\x04IN\
    T8\x10\x03\x12\n\n\x06UINT16\x10\x04\x12\t\n\x05INT16\x10\x05\x12\t\n\
    \x05INT32\x10\x06\x12\t\n\x05INT64\x10\x07\x12\n\n\x06STRING\x10\x08\x12\
    \x08\n\x04BOOL\x10\t\x12\x0b\n\x07FLOAT16\x10\n\x12\n\n\x06DOUBLE\x10\
    \x0b\x12\n\n\x06UINT32\x10\x0c\x12\n\n\x06UINT64\x10\r\x12\r\n\tCOMPLEX6\
    4\x10\x0e\x12\x0e\n\nCOMPLEX128\x10\x0f\"\x9a\x01\n\x10TensorShapeProto\
    \x122\n\x03dim\x18\x01\x20\x03(\x0b2\x20.onnx.TensorShapeProto.Dimension\
    R\x03dim\x1aR\n\tDimension\x12\x1d\n\tdim_value\x18\x01\x20\x01(\x03H\0R\
    \x08dimValue\x12\x1d\n\tdim_param\x18\x02\x20\x01(\tH\0R\x08dimParamB\
    \x07\n\x05value\"\xc0\x01\n\tTypeProto\x129\n\x0btensor_type\x18\x01\x20\
    \x01(\x0b2\x16.onnx.TypeProto.TensorH\0R\ntensorType\x1ao\n\x06Tensor\
    \x127\n\telem_type\x18\x01\x20\x01(\x0e2\x1a.onnx.TensorProto.DataTypeR\
    \x08elemType\x12,\n\x05shape\x18\x02\x20\x01(\x0b2\x16.onnx.TensorShapeP\
    rotoR\x05shapeB\x07\n\x05value\"F\n\x12OperatorSetIdProto\x12\x16\n\x06d\
    omain\x18\x01\x20\x01(\tR\x06domain\x12\x18\n\x07version\x18\x02\x20\x01\
    (\x03R\x07version*c\n\x07Version\x12\x12\n\x0e_START_VERSION\x10\0\x12\
    \x19\n\x15IR_VERSION_2017_10_10\x10\x01\x12\x19\n\x15IR_VERSION_2017_10_\
    30\x10\x02\x12\x0e\n\nIR_VERSION\x10\x03\
";

static mut file_descriptor_proto_lazy: ::protobuf::lazy::Lazy<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::lazy::Lazy {
    lock: ::protobuf::lazy::ONCE_INIT,
    ptr: 0 as *const ::protobuf::descriptor::FileDescriptorProto,
};

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    unsafe {
        file_descriptor_proto_lazy.get(|| {
            parse_descriptor_proto()
        })
    }
}
