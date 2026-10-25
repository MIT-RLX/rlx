// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.
//   dart run example/training.dart [device]

import 'dart:io';
import 'dart:math';

import 'package:rlx/rlx.dart';

const wTrue = [2.0, -1.0, 0.5];
const batch = 8;

Map<String, List<double>> data(int step) {
  final x = <double>[], y = <double>[];
  for (var i = 0; i < batch; i++) {
    var s = 0.0;
    for (var k = 0; k < 3; k++) {
      final v = ((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1;
      x.add(v);
      s += v * wTrue[k];
    }
    y.add(s);
  }
  return {'x': x, 'y': y};
}

String fmt(List<double> v) => v.map((e) => e.toStringAsFixed(3)).join(' ');

void main(List<String> args) {
  final device = args.isEmpty ? 'cpu' : args[0];
  final g = Graph(name: 'regression');
  final x = g.input('x', [batch, 3]), y = g.input('y', [batch, 1]);
  final w = g.param('w', [3, 1]);
  final d = g.sub(g.matmul(x, w), y);
  g.setOutputs([g.mean(g.mul(d, d), [0, 1])]);

  final trainer = Trainer(g.clone(), ['w'], {'w': [0, 0, 0]}, options: {
    'device': device,
    'optimizer': {'kind': 'adamw', 'lr': 0.1},
  });
  final first = trainer.evaluate(data(0));
  trainer.run(400, data,
      lr: (s) => 0.1 * 0.5 * (1 + cos(pi * s / 400)), onStep: (_, loss) => loss > 1e-6);
  print('trainer loss: ${first.toStringAsFixed(4)} -> '
      '${trainer.evaluate(data(0)).toStringAsFixed(6)} in ${trainer.steps()} steps');
  print('trainer w: ${fmt(trainer.params()['w']!)}');

  final path = '${Directory.systemTemp.path}/rlx-example-train-dart.gguf';
  final written = trainer.save(path);
  final restored = Trainer(g.clone(), ['w'], {'w': [0, 0, 0]}, options: {'device': device});
  print('checkpoint: $written tensors, restored step ${restored.load(path)['step']}');
  File(path).deleteSync();
  restored.freeze(names: ['w']);
  final before = fmt(restored.params()['w']!);
  restored.step(data(1));
  print('frozen: ${restored.frozen()} unchanged: ${fmt(restored.params()['w']!) == before}');
  restored.unfreeze();

  final backward = Session(device: device).compile(grad(g, [w])); // outputs [loss, dw]
  final opt = Optimizer(kind: 'sgd', options: {'lr': 0.2});
  final params = {'w': <double>[0, 0, 0]};
  var loss = 0.0;
  for (var step = 0; step < 300; step++) {
    backward.setParam('w', params['w']!);
    final outs = backward.run({...data(step), 'd_output': [1]});
    loss = outs[0][0];
    opt.step(params, {'w': outs[1]});
  }
  print('manual loss: ${loss.toStringAsFixed(6)} w: ${fmt(params['w']!)}');
}
