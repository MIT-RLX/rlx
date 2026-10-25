// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The Dart binding against the real library.

import 'dart:io';

import 'package:rlx/rlx.dart';
import 'package:test/test.dart';

(Graph, int) lossGraph() {
  final g = Graph(name: 'mse');
  final x = g.input('x', [1, 2]), y = g.input('y', [1, 2]);
  final w = g.param('w', [2, 2]);
  final d = g.sub(g.matmul(x, w), y);
  g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);
  return (g, w);
}

void main() {
  test('inference', () {
    final (g, _) = lossGraph();
    final m = Session().compile(g);
    expect(g.isConsumed(), isTrue);
    m.setParam('w', [1, 0, 0, 1]);
    expect(m.run({'x': [1, 2], 'y': [0, 0]})[0][0], 2.5);
    expect(m.device(), 'cpu');
  });

  test('errors carry their kind', () {
    expect(() => Graph().relu(42),
        throwsA(isA<RlxException>().having((e) => e.kind, 'kind', RlxErrorKind.range)));
  });

  test('training', () {
    final (g, w) = lossGraph();
    final bwd = Session().compile(grad(g, [w]));
    bwd.setParam('w', [1, 0, 0, 1]);
    expect(bwd.run({'x': [1, 2], 'y': [0, 0], 'd_output': [1]})[1][3], 4);
    final t = Trainer(g.clone(), ['w'], {'w': [0, 0, 0, 0]}, options: {
      'optimizer': {'kind': 'adamw', 'lr': 0.1}
    });
    final r = t.run(300, (_) => {'x': [1, 1], 'y': [1, -1]}, onStep: (_, l) => l > 1e-6);
    expect(r.lastLoss!, lessThan(1e-4));
    final opt = Optimizer(kind: 'sgd', options: {'lr': 0.5});
    final p = {'p': <double>[1, -1]};
    opt.step(p, {'p': [2, -2]});
    expect(p['p'], [0, 0]);
  });

  test('loading', () {
    final path = '${Directory.systemTemp.path}/rlx-dart-test.gguf';
    final w = GgufWriter()..addF32('w', [4], 'F32', [0.5, -1, 2, 0.25]);
    expect(w.write(path, architecture: 'demo'), 1);
    final f = Gguf.open(path);
    expect(f.architecture(), 'demo');
    expect(f.tensor('w')[2], 2);
    expect(f.info('w')['dtype'], 'F32');
    File(path).deleteSync();
  });

  test('backends and a one-rank mesh', () {
    expect(devices().first, 'cpu');
    expect(isAvailable('cpu'), isTrue);
    final (g, _) = lossGraph();
    final r = Runner(g, policy: {
      'only': ['cpu']
    });
    r.setParam('w', [1, 0, 0, 1]);
    final res = r.run({'x': [1, 2], 'y': [0, 0]});
    expect(res.device, 'cpu');
    expect(res.outputs[0][0], 2.5);
    expect(Group.connect(0, 1).allReduce([2, 3], op: 'max')[1], 3);
  });
}
