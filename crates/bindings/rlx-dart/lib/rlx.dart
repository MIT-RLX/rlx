// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// rlx for Dart, over its C ABI (`librlx_capi`): every C function, generated
/// by rlx-bindgen from rlx-bind's tables (`src/generated.dart`), on the
/// hand-written support in `src/support.dart` and `src/manual.dart`.
///
/// The library is `RLX_CAPI_LIB` when set, else `librlx_capi` on the
/// platform's library search path.
library;

import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

part 'src/support.dart';
part 'src/generated.dart';
part 'src/manual.dart';
