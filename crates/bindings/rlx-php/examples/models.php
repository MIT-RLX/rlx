<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.so php examples/models.php

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Graph;
use Rlx\Model;

use function Rlx\saveModel;

$g = new Graph(name: 'affine');
$x = $g->input('x', [1, 2]);
$g->setOutputs([$g->matmul($x, $g->param('w', [2, 2]))]);
$path = sys_get_temp_dir() . '/rlx-example-php.rlxp';
saveModel($g, $path, ['w' => [2, 0, 0, 3]], model: ['task' => 'graph', 'family' => 'affine', 'outputs' => ['y']]);

$m = Model::open($path, ['device' => 'cpu']);
$info = $m->info();
echo "model: {$info['family']} ({$info['task']}) on {$info['device']}\n";
$y = $m->run(['x' => [1, 1]])[0];
printf("model run: %.3f %.3f\n", $y[0], $y[1]);
unlink($path);

$lm = Model::open('demo:count');
$lmInfo = $lm->info();
echo "plugin model: {$lmInfo['family']} ({$lmInfo['task']})\n";
echo 'model text: ', $lm->generateText('c', ['max_tokens' => 4]), "\n";
echo 'model ids: ', implode(' ', $lm->generate([2], maxTokens: 3)), "\n";

// By task: a chat, speech out and back in, embeddings.
echo 'chat: ', $lm->chat([['role' => 'user', 'content' => 'c']], ['max_tokens' => 4]), "\n";
$speech = Model::open('demo:voice')->synthesize('hello');
printf("voice: %d samples at %d Hz\n", count($speech['audio']), $speech['sampleRate']);
echo 'heard: ', Model::open('demo:speech')->transcribe($speech['audio'], $speech['sampleRate']), "\n";
$rows = Model::open('demo:embed')->embed(['hello', 'sky']);
echo 'embed: ', implode(' ', array_map('intval', array_merge(...$rows))), "\n";
