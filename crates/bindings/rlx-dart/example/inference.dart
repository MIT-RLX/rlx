// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//   dart run example/inference.dart [device]
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

import 'package:rlx/rlx.dart';

List<double> weights(int n, int seed) =>
    [for (var i = 0; i < n; i++) ((i * 7919 + seed * 104729) % 1000) / 1000 - 0.5];

void main(List<String> args) {
  final device = args.isEmpty ? 'cpu' : args[0];
  final g = Graph(name: 'classifier');
  final x = g.input('x', [1, 4]);
  final hidden = g.relu(g.matmul(x, g.param('w1', [4, 8])));
  g.setOutputs([g.softmax(g.matmul(hidden, g.param('w2', [8, 3])))]);

  final model = Session(device: device).compile(g); // consumes g
  print('compiled on: ${model.device()}');
  model.setParams({'w1': weights(32, 1), 'w2': weights(24, 2)});
  final probs = model.run({'x': [0.5, -1, 2, 0.25]})[0];
  print('probs: ${probs.map((p) => p.toStringAsFixed(4)).join(' ')}');
  print('probs sum: ${probs.reduce((a, b) => a + b).toStringAsFixed(4)}');
  print('output shapes: ${model.outputShapes()} dtypes: ${model.outputDtypes()}');
  model.close();
}
