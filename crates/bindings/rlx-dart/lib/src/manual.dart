// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The two rows whose shape is not a plain call: Optimizer.step updates the
// caller's lists in place, Trainer.run takes callbacks.

part of '../rlx.dart';

final _optimizerStep = _lib.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Float>>, Pointer<Void>, Pointer<Size>,
        Pointer<Void>, Pointer<Void>, Size),
    int Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Float>>, Pointer<Void>, Pointer<Size>,
        Pointer<Void>, Pointer<Void>, int)>('rlx_optimizer_step');

extension OptimizerStep on Optimizer {
  /// Update each `params[name]` in place from `grads[name]`.
  void step(Map<String, List<double>> params, Map<String, List<double>> grads) {
    final names = params.keys.toList()..sort();
    using((a) {
      final n = names.length;
      final ps = a<Pointer<Float>>(n == 0 ? 1 : n);
      final gs = a<Pointer<Void>>(n == 0 ? 1 : n);
      final lens = a<Size>(n == 0 ? 1 : n);
      for (var i = 0; i < n; i++) {
        final p = params[names[i]]!;
        final g = grads[names[i]] ??
            (throw RlxException(RlxErrorKind.type, "Optimizer.step: no gradient for '${names[i]}'"));
        ps[i] = _floats(a, p).cast();
        gs[i] = _floats(a, g);
        lens[i] = p.length;
      }
      _check(_optimizerStep(_h, _strs(a, names), ps, gs.cast(), lens, nullptr, nullptr, n));
      for (var i = 0; i < n; i++) {
        params[names[i]]!.setAll(0, ps[i].asTypedList(lens[i]));
      }
    });
  }
}

/// What [TrainerRun.run] did.
typedef TrainSummary = ({
  int steps,
  bool stopped,
  double? firstLoss,
  double? lastLoss,
  double? meanLoss,
  double stepsPerSecond,
  double? gradNorm,
});

extension TrainerRun on Trainer {
  /// Train `steps` steps, pulling inputs from `batch(step)` (null: data
  /// exhausted). `lr(step)` schedules the learning rate; `onStep(step, loss)`
  /// returning false stops early (called every `every` steps).
  TrainSummary run(
    int steps,
    Map<String, List<double>>? Function(int step) batch, {
    double Function(int step)? lr,
    bool Function(int step, double loss)? onStep,
    int every = 1,
  }) {
    every = every < 1 ? 1 : every;
    final start = this.steps();
    final losses = <double>[];
    var stopped = false;
    final clock = Stopwatch()..start();
    for (var i = 0; i < steps; i++) {
      final s = start + i;
      final b = batch(s);
      if (b == null) break;
      if (lr != null) setLr(lr(s));
      final loss = step(b);
      losses.add(loss);
      if (onStep != null && (i + 1) % every == 0 && !onStep(s, loss)) {
        stopped = true;
        break;
      }
    }
    final secs = clock.elapsedMicroseconds / 1e6;
    return (
      steps: losses.length,
      stopped: stopped,
      firstLoss: losses.isEmpty ? null : losses.first,
      lastLoss: losses.isEmpty ? null : losses.last,
      meanLoss: losses.isEmpty ? null : losses.reduce((x, y) => x + y) / losses.length,
      stepsPerSecond: losses.length / (secs > 1e-9 ? secs : 1e-9),
      gradNorm: gradNorm(),
    );
  }
}
