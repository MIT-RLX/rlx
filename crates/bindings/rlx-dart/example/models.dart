// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib dart run example/models.dart

import 'dart:io';

import 'package:rlx/rlx.dart';

void main() {
  final g = Graph(name: 'affine');
  final x = g.input('x', [1, 2]);
  g.setOutputs([g.matmul(x, g.param('w', [2, 2]))]);
  final path = '${Directory.systemTemp.path}/rlx-example-dart.rlxp';
  saveModel(g, path, {'w': [2, 0, 0, 3]},
      model: {'task': 'graph', 'family': 'affine', 'outputs': ['y']});

  final m = Model.open(path, options: {'device': 'cpu'});
  final info = m.info();
  print('model: ${info['family']} (${info['task']}) on ${info['device']}');
  final y = m.run({'x': [1, 1]})[0];
  print('model run: ${y[0].toStringAsFixed(3)} ${y[1].toStringAsFixed(3)}');
  m.close();
  File(path).deleteSync();

  final lm = Model.open('demo:count');
  final lmInfo = lm.info();
  print('plugin model: ${lmInfo['family']} (${lmInfo['task']})');
  print('model text: ${lm.generateText('c', options: {'max_tokens': 4})}');
  print('model ids: ${lm.generate([2], maxTokens: 3).join(' ')}');

  // By task: a chat, speech out and back in, embeddings.
  print('chat: ${lm.chat([{'role': 'user', 'content': 'c'}], options: {'max_tokens': 4})}');
  lm.close();
  final voice = Model.open('demo:voice');
  final speech = voice.synthesize('hello');
  print('voice: ${speech.audio.length} samples at ${speech.sampleRate} Hz');
  final ear = Model.open('demo:speech');
  print('heard: ${ear.transcribe(speech.audio, sampleRate: speech.sampleRate)}');
  final embedder = Model.open('demo:embed');
  final rows = embedder.embed(['hello', 'sky']) as List;
  print('embed: ${rows.expand((r) => r as List).map((v) => (v as num).toInt()).join(' ')}');
  for (final model in [voice, ear, embedder]) {
    model.close();
  }
}
