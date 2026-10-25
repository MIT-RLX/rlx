// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//   dart run example/loading.dart

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:rlx/rlx.dart';

void main() {
  final tmp = Directory.systemTemp.createTempSync('rlx-example-dart-');
  final w = [for (var i = 0; i < 64; i++) (i % 9) / 4 - 1];

  final gguf = '${tmp.path}/demo.gguf';
  final writer = GgufWriter()
    ..addF32('proj.weight', [8, 8], 'F32', w)
    ..addF32('proj.weight_q8', [8, 8], 'Q8_0', w);
  writer.write(gguf, architecture: 'demo', metadata: {'demo.layers': 1});
  final f = Gguf.open(gguf);
  print('gguf architecture: ${f.architecture()}');
  print('gguf tensors: ${f.tensorNames().join(' ')}');
  final exact = f.tensor('proj.weight'), q8 = f.tensor('proj.weight_q8');
  var same = true, maxd = 0.0;
  for (var i = 0; i < 64; i++) {
    same &= exact[i] == w[i];
    maxd = (exact[i] - q8[i]).abs() > maxd ? (exact[i] - q8[i]).abs() : maxd;
  }
  print('gguf f32 exact: $same q8 max|Δ|: ${maxd.toStringAsFixed(4)}');

  final g = Graph(name: 'proj');
  final x = g.input('x', [1, 8]);
  g.setOutputs([g.matmul(x, g.param('proj.weight', [8, 8]))]);
  final c = Session().compile(g);
  c.setParam('proj.weight', exact);
  final y = c.run({'x': List.filled(8, 1.0)})[0];
  print('gguf -> graph: ${y.take(3).map((v) => v.toStringAsFixed(2)).join(' ')}');

  final st = '${tmp.path}/demo.safetensors';
  var header = '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}';
  header += ' ' * ((8 - header.length % 8) % 8);
  final bytes = BytesBuilder()
    ..add((ByteData(8)..setUint64(0, header.length, Endian.little)).buffer.asUint8List())
    ..add(utf8.encode(header))
    ..add(Float32List.fromList(w).buffer.asUint8List());
  File(st).writeAsBytesSync(bytes.toBytes());
  final loaded = loadWeights('safetensors', st);
  var exact2 = true;
  for (var i = 0; i < 64; i++) {
    exact2 &= loaded['proj.weight']![i] == w[i];
  }
  print('safetensors: [proj.weight] exact: $exact2');

  final packed = quantize(w, 'Q8_0');
  print('quantize q8_0: ${w.length * 4} -> ${packed.length} bytes, '
      'round trip ${dequantize(packed, 'Q8_0', count: w.length).length} values');
  tmp.deleteSync(recursive: true);
  print('argmax: ${argmax([0.1, 2, 0.3])}');
}
