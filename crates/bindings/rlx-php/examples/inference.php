<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//   php examples/inference.php [device]
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Graph;
use Rlx\Session;

$device = $argv[1] ?? 'cpu';
$weights = fn (int $n, int $seed) => array_map(fn ($i) => (($i * 7919 + $seed * 104729) % 1000) / 1000 - 0.5, range(0, $n - 1));

$g = new Graph(name: 'classifier');
$x = $g->input('x', [1, 4]);
$hidden = $g->relu($g->matmul($x, $g->param('w1', [4, 8])));
$g->setOutputs([$g->softmax($g->matmul($hidden, $g->param('w2', [8, 3])))]);

$model = (new Session(device: $device))->compile($g); // consumes $g
echo "compiled on: {$model->device()}\n";
$model->setParams(['w1' => $weights(32, 1), 'w2' => $weights(24, 2)]);
$probs = $model->run(['x' => [0.5, -1, 2, 0.25]])[0];
echo 'probs: ', implode(' ', array_map(fn ($p) => sprintf('%.4f', $p), $probs)), "\n";
printf("probs sum: %.4f\n", array_sum($probs));
echo 'output shapes: ', json_encode($model->outputShapes()), ' dtypes: ', json_encode($model->outputDtypes()), "\n";
