// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//   dart run example/backends.dart

import 'package:rlx/rlx.dart';

Graph model() {
  final g = Graph(name: 'mlp');
  final x = g.input('x', [4, 16]);
  g.setOutputs([g.tanh(g.matmul(x, g.param('w', [16, 16])))]);
  return g;
}

void main() {
  final w = [for (var i = 0; i < 256; i++) (i % 5) / 10 - 0.2];
  final x = {
    'x': [for (var i = 0; i < 64; i++) (i % 7) / 7]
  };

  final available = devices();
  print('devices: ${available.join(' ')}');
  print('metal available: ${isAvailable('metal')}');
  print('fastest for graph: ${fastestDeviceFor(model())}');
  List<double>? reference;
  for (final device in available) {
    final c = Session(device: device).compile(model());
    c.setParam('w', w);
    final y = c.run(x)[0];
    reference ??= y;
    var d = 0.0;
    for (var i = 0; i < y.length; i++) {
      final e = (y[i] - reference[i]).abs();
      if (e > d) d = e;
    }
    print('session $device: max|Δ| vs first = ${d.toStringAsExponential(2)}');
  }
  final flex = FlexibleSession(policy: {
    'prefer': ['metal', 'cpu']
  });
  print('flexible session picked: ${flex.compile(model()).device()}');
  final runner = Runner(model());
  runner.setParam('w', w);
  print('runner warmed: ${runner.warmAll().join(' ')}');
  for (final row in runner.benchmark(x, runs: 3)) {
    print('  bench ${row['device']}: exec_ms=${row['exec_ms']}');
  }
  print('runner ran on: ${runner.run(x).device}');
  print('run_try chain: ${runner.runTry(['metal', 'cpu'], x).device}');
  final router = Router(model());
  router.setParam('w', w);
  print('router ran on: ${router.runChain(x).device}');
}
