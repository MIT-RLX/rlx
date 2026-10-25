// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated code stands on: loading the library, errors, the handle
// base, and moving values to and from C.

part of '../rlx.dart';

final DynamicLibrary _lib = _open();

/// `RLX_CAPI_LIB` when set, else the library bundled with the package
/// (`native/<os>-<arch>/`), else librlx_capi on the loader's search path.
DynamicLibrary _open() {
  final path = Platform.environment['RLX_CAPI_LIB'];
  if (path != null && path.isNotEmpty) return DynamicLibrary.open(path);
  final file = Platform.isMacOS
      ? 'librlx_capi.dylib'
      : Platform.isWindows
          ? 'rlx_capi.dll'
          : 'librlx_capi.so';
  return DynamicLibrary.open(_bundled(file) ?? file);
}

String? _bundled(String file) {
  final platforms = {
    Abi.macosArm64: 'macos-arm64',
    Abi.macosX64: 'macos-x64',
    Abi.linuxX64: 'linux-x64',
    Abi.linuxArm64: 'linux-arm64',
    Abi.windowsX64: 'windows-x64',
  };
  final platform = platforms[Abi.current()];
  final lib = Isolate.resolvePackageUriSync(Uri.parse('package:rlx/rlx.dart'));
  if (platform == null || lib == null) return null;
  final candidate = File.fromUri(lib.resolve('../native/$platform/$file'));
  return candidate.existsSync() ? candidate.path : null;
}

/// "No node", for an optional node argument.
const int none = 0xFFFFFFFF;

/// What kind of failure an [RlxException] is (the `RLX_ERR_*` status).
enum RlxErrorKind { type, range, runtime, panic, notFound, other }

/// A failed rlx call.
class RlxException implements Exception {
  final RlxErrorKind kind;
  final String message;
  RlxException(this.kind, this.message);
  @override
  String toString() => 'RlxException(${kind.name}): $message';
}

/// A tensor of any dtype: its little-endian bytes and dtype name.
class TypedTensor {
  final Uint8List data;
  final String dtype;
  const TypedTensor(this.data, this.dtype);
}

final _lastError =
    _lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>('rlx_last_error');

void _check(int rc) {
  if (rc == 0) return;
  final kind = switch (rc) {
    -1 => RlxErrorKind.type,
    -2 => RlxErrorKind.range,
    -3 => RlxErrorKind.runtime,
    -4 => RlxErrorKind.panic,
    -5 => RlxErrorKind.notFound,
    _ => RlxErrorKind.other,
  };
  throw RlxException(kind, _lastError().toDartString());
}

/// The base of every handle class: freed when collected, or now with [close].
abstract base class _Handle implements Finalizable {
  Pointer<Void> _h;
  final Pointer<NativeFunction<Void Function(Pointer<Void>)>> _release;
  final NativeFinalizer _finalizer;

  _Handle(this._h, this._release, this._finalizer) {
    if (_h == nullptr) throw RlxException(RlxErrorKind.runtime, 'rlx returned NULL');
    _finalizer.attach(this, _h, detach: this);
  }

  /// Free the handle now (it is unusable afterwards).
  void close() {
    if (_h == nullptr) return;
    _finalizer.detach(this);
    _release.asFunction<void Function(Pointer<Void>)>()(_h);
    _h = nullptr;
  }
}

// ── arguments ──────────────────────────────────────────────────

/// `{ptr, len}` as rlx returns arrays.
final class _Slice extends Struct {
  external Pointer<Void> ptr;
  @Size()
  external int len;
}

Pointer<Void> _str(Allocator a, String s) => s.toNativeUtf8(allocator: a).cast();
Pointer<Void> _strOrNull(Allocator a, String? s) => s == null ? nullptr : _str(a, s);
Pointer<Void> _json(Allocator a, Object? v) => v == null ? nullptr : _str(a, jsonEncode(v));

Pointer<Void> _boxU64(Allocator a, int? v) {
  if (v == null) return nullptr;
  final p = a<Uint64>()..value = v;
  return p.cast();
}

Pointer<Void> _boxF32(Allocator a, double? v) {
  if (v == null) return nullptr;
  final p = a<Float>()..value = v;
  return p.cast();
}

Pointer<Void> _floats(Allocator a, List<double> v) {
  if (v.isEmpty) return nullptr;
  final p = a<Float>(v.length);
  p.asTypedList(v.length).setAll(0, v);
  return p.cast();
}

Pointer<Void> _bytes(Allocator a, List<int> v) {
  if (v.isEmpty) return nullptr;
  final p = a<Uint8>(v.length);
  p.asTypedList(v.length).setAll(0, v);
  return p.cast();
}

Pointer<Void> _u32s(Allocator a, List<int> v) {
  if (v.isEmpty) return nullptr;
  final p = a<Uint32>(v.length);
  p.asTypedList(v.length).setAll(0, v);
  return p.cast();
}

Pointer<Void> _u64s(Allocator a, List<int> v) {
  if (v.isEmpty) return nullptr;
  final p = a<Uint64>(v.length);
  p.asTypedList(v.length).setAll(0, v);
  return p.cast();
}

Pointer<Void> _i64s(Allocator a, List<int> v) {
  if (v.isEmpty) return nullptr;
  final p = a<Int64>(v.length);
  p.asTypedList(v.length).setAll(0, v);
  return p.cast();
}

Pointer<Void> _strs(Allocator a, List<String>? v) {
  if (v == null || v.isEmpty) return nullptr;
  final p = a<Pointer<Void>>(v.length);
  for (var i = 0; i < v.length; i++) {
    p[i] = _str(a, v[i]);
  }
  return p.cast();
}

List<int> _rows(List<List<int>> rows, int w, bool splat) => [
      for (final (i, r) in rows.indexed)
        if (r.length == w)
          ...r
        else if (splat && r.length == 1)
          ...List.filled(w, r[0])
        else
          throw RlxException(RlxErrorKind.type, 'row $i needs $w values, got ${r.length}'),
    ];

List<double> _frows(List<List<double>> rows, int w) => [
      for (final (i, r) in rows.indexed)
        if (r.length == w)
          ...r
        else
          throw RlxException(RlxErrorKind.type, 'row $i needs $w values, got ${r.length}'),
    ];

/// Named tensors as the C ABI takes them: names, data pointers, lengths.
class _Inputs {
  final Pointer<Void> names, data, lens, dtypes;
  final int n;
  _Inputs(this.names, this.data, this.lens, this.dtypes, this.n);
}

_Inputs _inputs(Allocator a, Map<String, List<double>> m) {
  final names = m.keys.toList()..sort();
  final data = a<Pointer<Void>>(names.isEmpty ? 1 : names.length);
  final lens = a<Size>(names.isEmpty ? 1 : names.length);
  for (var i = 0; i < names.length; i++) {
    final v = m[names[i]]!;
    data[i] = _floats(a, v);
    lens[i] = v.length;
  }
  return _Inputs(_strs(a, names), data.cast(), lens.cast(), nullptr, names.length);
}

_Inputs _typedInputs(Allocator a, Map<String, TypedTensor> m) {
  final names = m.keys.toList()..sort();
  final data = a<Pointer<Void>>(names.isEmpty ? 1 : names.length);
  final lens = a<Size>(names.isEmpty ? 1 : names.length);
  for (var i = 0; i < names.length; i++) {
    final t = m[names[i]]!;
    data[i] = _bytes(a, t.data);
    lens[i] = t.data.length;
  }
  final dtypes = _strs(a, [for (final n in names) m[n]!.dtype]);
  return _Inputs(_strs(a, names), data.cast(), lens.cast(), dtypes, names.length);
}

// ── results ────────────────────────────────────────────────────

final _stringFree =
    _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('rlx_string_free');
final _f32sRelease =
    _lib.lookupFunction<Void Function(Pointer<_Slice>), void Function(Pointer<_Slice>)>('rlx_f32s_release');
final _bytesRelease =
    _lib.lookupFunction<Void Function(Pointer<_Slice>), void Function(Pointer<_Slice>)>('rlx_bytes_release');
final _u32sRelease =
    _lib.lookupFunction<Void Function(Pointer<_Slice>), void Function(Pointer<_Slice>)>('rlx_u32s_release');

String _takeString(Pointer<Pointer<Void>> o) => _takeStringOrNull(o) ?? '';

String? _takeStringOrNull(Pointer<Pointer<Void>> o) {
  final p = o.value;
  if (p == nullptr) return null;
  final s = p.cast<Utf8>().toDartString();
  _stringFree(p);
  return s;
}

dynamic _takeJson(Pointer<Pointer<Void>> o) => jsonDecode(_takeString(o));

Float32List _takeF32s(Pointer<_Slice> o) {
  final s = o.ref;
  final v = s.len == 0 ? Float32List(0) : Float32List.fromList(s.ptr.cast<Float>().asTypedList(s.len));
  _f32sRelease(o);
  return v;
}

Uint8List _takeBytes(Pointer<_Slice> o) {
  final s = o.ref;
  final v = s.len == 0 ? Uint8List(0) : Uint8List.fromList(s.ptr.cast<Uint8>().asTypedList(s.len));
  _bytesRelease(o);
  return v;
}

List<int> _takeU32s(Pointer<_Slice> o) {
  final s = o.ref;
  final v = s.len == 0 ? <int>[] : List<int>.of(s.ptr.cast<Uint32>().asTypedList(s.len));
  _u32sRelease(o);
  return v;
}

final _outputsLen =
    _lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('rlx_outputs_len');
final _outputsGet = _lib.lookupFunction<
    Int32 Function(Pointer<Void>, Size, Pointer<Pointer<Float>>, Pointer<Size>),
    int Function(Pointer<Void>, int, Pointer<Pointer<Float>>, Pointer<Size>)>('rlx_outputs_get');
final _outputsFree =
    _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('rlx_outputs_free');

List<Float32List> _takeOutputs(Pointer<Void> o) => using((a) {
      final d = a<Pointer<Float>>();
      final len = a<Size>();
      final out = [
        for (var i = 0; i < _outputsLen(o); i++)
          if (_outputsGet(o, i, d, len) == 0)
            len.value == 0 ? Float32List(0) : Float32List.fromList(d.value.asTypedList(len.value)),
      ];
      _outputsFree(o);
      return out;
    });

final _typedLen =
    _lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('rlx_typed_outputs_len');
final _typedGet = _lib.lookupFunction<
    Int32 Function(Pointer<Void>, Size, Pointer<Pointer<Uint8>>, Pointer<Size>, Pointer<Pointer<Utf8>>),
    int Function(Pointer<Void>, int, Pointer<Pointer<Uint8>>, Pointer<Size>,
        Pointer<Pointer<Utf8>>)>('rlx_typed_outputs_get');
final _typedFree =
    _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('rlx_typed_outputs_free');

List<TypedTensor> _takeTypedOutputs(Pointer<Void> o) => using((a) {
      final d = a<Pointer<Uint8>>();
      final len = a<Size>();
      final dt = a<Pointer<Utf8>>();
      final out = [
        for (var i = 0; i < _typedLen(o); i++)
          if (_typedGet(o, i, d, len, dt) == 0)
            TypedTensor(
                len.value == 0 ? Uint8List(0) : Uint8List.fromList(d.value.asTypedList(len.value)),
                dt.value.toDartString()),
      ];
      _typedFree(o);
      return out;
    });

final _mapLen =
    _lib.lookupFunction<Size Function(Pointer<Void>), int Function(Pointer<Void>)>('rlx_tensor_map_len');
final _mapGet = _lib.lookupFunction<
    Int32 Function(Pointer<Void>, Size, Pointer<Pointer<Utf8>>, Pointer<Pointer<Float>>, Pointer<Size>),
    int Function(Pointer<Void>, int, Pointer<Pointer<Utf8>>, Pointer<Pointer<Float>>,
        Pointer<Size>)>('rlx_tensor_map_get');
final _mapFree =
    _lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('rlx_tensor_map_free');

Map<String, Float32List> _takeTensors(Pointer<Void> m) => using((a) {
      final name = a<Pointer<Utf8>>();
      final d = a<Pointer<Float>>();
      final len = a<Size>();
      final out = <String, Float32List>{};
      for (var i = 0; i < _mapLen(m); i++) {
        if (_mapGet(m, i, name, d, len) != 0) continue;
        out[name.value.toDartString()] =
            len.value == 0 ? Float32List(0) : Float32List.fromList(d.value.asTypedList(len.value));
      }
      _mapFree(m);
      return out;
    });
