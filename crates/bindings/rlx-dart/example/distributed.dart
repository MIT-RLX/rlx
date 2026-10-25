// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: a three-rank mesh on loopback, one isolate per rank (across
// machines, one process per host with the same peers). Rank 0 coordinates:
// collectives, pipelined inference (a stage shipped per worker),
// data-parallel training where every replica ends identical.
//   dart run example/distributed.dart

import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:rlx/rlx.dart';

const world = 3, perRank = 16;

void worker((int, List<String>, SendPort) args) {
  final (rank, peers, done) = args;
  final g = Group.connect(rank, world, options: {'peers': peers, 'device': 'cpu'});
  collectives(g, rank);
  g.serveWorker(options: {'steps': 2});
  done.send(List<double>.from(g.serveTrainer()['params']['w'] as List));
}

void collectives(Group g, int rank) {
  final sum = g.allReduce([rank + 1.0], op: 'sum');
  final gathered = g.allGather([rank + 1.0]);
  final shared = g.broadcast(0, [rank == 0 ? 42.0 : 0]); // the root's length
  if (rank == 0) {
    print('mesh: rank ${g.rank()} of ${g.world()}, leader: ${g.isLeader()}');
    print('all_reduce sum: ${sum.join(',')}');
    print('all_gather: ${gathered.join(',')}');
    print('broadcast: ${shared.join(',')}');
  }
}

Future<void> main() async {
  final peers = loopbackPeers(world);
  final results = [for (var r = 1; r < world; r++) ReceivePort()];
  for (var r = 1; r < world; r++) {
    await Isolate.spawn(worker, (r, peers, results[r - 1].sendPort));
  }
  final g = Group.connect(0, world, options: {'peers': peers, 'device': 'cpu'});
  collectives(g, 0);
  final tmp = Directory.systemTemp.createTempSync('rlx-dist-dart-');
  String blob(String name, List<double> v) {
    final path = '${tmp.path}/$name';
    File(path).writeAsBytesSync(Float32List.fromList(v).buffer.asUint8List());
    return 'file://$path';
  }

  final stageW = [
    [for (var i = 0; i < 16; i++) (i * 5 % 7) / 7 - 0.3],
    [for (var i = 0; i < 16; i++) (i * 3 % 5) / 5 - 0.2],
  ];
  print('workers: ${g.collectCaps().map((c) => 'rank ${c['rank']} on ${c['platform']}').join(', ')}');
  final stage = Graph(name: 'stage');
  stage.setOutputs([stage.relu(stage.matmul(stage.input('x', [1, 4]), stage.param('w', [4, 4])))]);
  for (var r = 1; r < world; r++) {
    g.shipStage(r, stage, {
      'device': 'cpu',
      'weights': {'w': blob('w$r.bin', stageW[(r - 1) % 2])}
    });
  }
  for (final xv in <List<double>>[
    [1, -1, 0.5, 2],
    [0.25, 0.5, -0.75, 1],
  ]) {
    g.sendActivation(1, xv);
    print('pipeline: ${g.recvActivation(world - 1).map((v) => v.toStringAsFixed(3)).join(' ')}');
  }

  g.collectCaps();
  final lr = Graph(name: 'lr');
  final X = lr.input('X', [4, 3]), y = lr.input('y', [4, 1]), wv = lr.param('w', [3, 1]);
  final d = lr.sub(lr.matmul(X, wv), y);
  lr.setOutputs([lr.mean(lr.mul(d, d), [0, 1])]);
  final bwd = grad(lr, [wv]);
  const wTrue = [1.5, -2.0, 0.5];
  const n = perRank * world;
  final xAll = [for (var i = 0; i < n * 3; i++) (i * 37 % 23) / 11.5 - 1];
  final yAll = [
    for (var i = 0; i < n; i++) [for (var k = 0; k < 3; k++) xAll[i * 3 + k] * wTrue[k]].reduce((a, b) => a + b)
  ];
  final uris = {'X': blob('X.bin', xAll), 'y': blob('y.bin', yAll), 'w': blob('w0.bin', [0, 0, 0])};
  Map<String, Object> spec(int r) => {
        'params': [
          {'name': 'w', 'uri': uris['w']!}
        ],
        'data': [
          {'input': 'X', 'uri': uris['X']!, 'elem': 3, 'shard_start': r * perRank, 'shard_len': perRank},
          {'input': 'y', 'uri': uris['y']!, 'elem': 1, 'shard_start': r * perRank, 'shard_len': perRank},
        ],
        'batch': 4,
        'lr_per_epoch': List.filled(40, 0.2),
        'device': 'cpu',
      };
  for (var r = 1; r < world; r++) {
    g.shipTrain(r, bwd, spec(r));
  }
  final mine = g.runTrain(bwd, spec(0));
  final metrics = mine['metrics'];
  print('training loss: ${(metrics['first_loss'] as num).toStringAsFixed(4)} -> '
      '${(metrics['last_loss'] as num).toStringAsExponential(2)}');
  final w0 = List<double>.from(mine['params']['w'] as List);
  print('rank 0 w: ${w0.map((v) => v.toStringAsFixed(3)).join(' ')} (true: 1.5 -2 0.5)');
  for (var r = 1; r < world; r++) {
    final theirs = await results[r - 1].first as List<double>;
    final agree = [for (var k = 0; k < w0.length; k++) (theirs[k] - w0[k]).abs() < 1e-5].every((b) => b);
    print('rank $r w agrees: $agree');
  }
  tmp.deleteSync(recursive: true);
}
