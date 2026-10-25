<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.
//   php examples/training.php [device]

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Graph;
use Rlx\Optimizer;
use Rlx\Session;
use Rlx\Trainer;

$device = $argv[1] ?? 'cpu';
const W_TRUE = [2, -1, 0.5];
const BATCH = 8;

function data(int $step): array
{
    $x = [];
    $y = [];
    for ($i = 0; $i < BATCH; $i++) {
        $s = 0.0;
        for ($k = 0; $k < 3; $k++) {
            $v = (($step * 31 + $i * 7 + $k * 3) % 17) / 8.5 - 1;
            $x[] = $v;
            $s += $v * W_TRUE[$k];
        }
        $y[] = $s;
    }
    return ['x' => $x, 'y' => $y];
}
$fmt = fn (array $v) => implode(' ', array_map(fn ($e) => sprintf('%.3f', $e), $v));

$g = new Graph(name: 'regression');
$x = $g->input('x', [BATCH, 3]);
$y = $g->input('y', [BATCH, 1]);
$w = $g->param('w', [3, 1]);
$d = $g->sub($g->matmul($x, $w), $y);
$g->setOutputs([$g->mean($g->mul($d, $d), [0, 1])]);

$trainer = new Trainer($g->clone(), ['w'], ['w' => [0, 0, 0]],
    options: ['device' => $device, 'optimizer' => ['kind' => 'adamw', 'lr' => 0.1]]);
$first = $trainer->evaluate(data(0));
$trainer->run(400, 'data',
    lr: fn ($s) => 0.1 * 0.5 * (1 + cos(M_PI * $s / 400)),
    onStep: fn ($s, $loss) => $loss > 1e-6);
printf("trainer loss: %.4f -> %.6f in %d steps\n", $first, $trainer->evaluate(data(0)), $trainer->steps());
echo 'trainer w: ', $fmt($trainer->params()['w']), "\n";

$path = sys_get_temp_dir() . '/rlx-example-train-php.gguf';
$written = $trainer->save($path);
$restored = new Trainer($g->clone(), ['w'], ['w' => [0, 0, 0]], options: ['device' => $device]);
printf("checkpoint: %d tensors, restored step %d\n", $written, $restored->load($path)['step']);
unlink($path);
$restored->freeze(names: ['w']);
$before = $fmt($restored->params()['w']);
$restored->step(data(1));
echo 'frozen: ', json_encode($restored->frozen()), ' unchanged: ', $fmt($restored->params()['w']) === $before ? 'true' : 'false', "\n";
$restored->unfreeze();

$backward = (new Session(device: $device))->compile(Rlx\grad($g, [$w])); // outputs [loss, dw]
$opt = new Optimizer(kind: 'sgd', options: ['lr' => 0.2]);
$params = ['w' => [0.0, 0.0, 0.0]];
$loss = 0.0;
for ($step = 0; $step < 300; $step++) {
    $backward->setParam('w', $params['w']);
    $outs = $backward->run(data($step) + ['d_output' => [1]]);
    $loss = $outs[0][0];
    $opt->step($params, ['w' => $outs[1]]);
}
printf("manual loss: %.6f w: %s\n", $loss, $fmt($params['w']));
