<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The PHP binding against the real library; exits non-zero on the first failure.

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Gguf;
use Rlx\GgufWriter;
use Rlx\Graph;
use Rlx\Group;
use Rlx\Optimizer;
use Rlx\RlxException;
use Rlx\Runner;
use Rlx\Session;
use Rlx\Trainer;

$checks = 0;
function check(bool $ok, string $what): void
{
    global $checks;
    $checks++;
    if (!$ok) {
        fwrite(STDERR, "failed: $what\n");
        exit(1);
    }
}
function lossGraph(): array
{
    $g = new Graph(name: 'mse');
    $x = $g->input('x', [1, 2]);
    $y = $g->input('y', [1, 2]);
    $w = $g->param('w', [2, 2]);
    $d = $g->sub($g->matmul($x, $w), $y);
    $g->setOutputs([$g->mean($g->mul($d, $d), [0, 1])]);
    return [$g, $w];
}

// inference
[$g] = lossGraph();
$m = (new Session())->compile($g);
check($g->isConsumed(), 'compile consumes');
$m->setParam('w', [1, 0, 0, 1]);
check($m->run(['x' => [1, 2], 'y' => [0, 0]])[0][0] == 2.5, 'loss 2.5');
check($m->device() === 'cpu', 'device');

// errors
try {
    (new Graph())->relu(42);
    check(false, 'range error raised');
} catch (RlxException $e) {
    check($e->kind === 'range', 'range error');
}

// training
[$g, $w] = lossGraph();
$bwd = (new Session())->compile(Rlx\grad($g, [$w]));
$bwd->setParam('w', [1, 0, 0, 1]);
check($bwd->run(['x' => [1, 2], 'y' => [0, 0], 'd_output' => [1]])[1][3] == 4, 'dw');
$t = new Trainer($g->clone(), ['w'], ['w' => [0, 0, 0, 0]], options: ['optimizer' => ['kind' => 'adamw', 'lr' => 0.1]]);
$r = $t->run(300, fn () => ['x' => [1, 1], 'y' => [1, -1]], onStep: fn ($s, $l) => $l > 1e-6);
check($r['lastLoss'] < 1e-4, 'trainer converges');
$opt = new Optimizer(kind: 'sgd', options: ['lr' => 0.5]);
$p = ['p' => [1.0, -1.0]];
$opt->step($p, ['p' => [2, -2]]);
check($p['p'] == [0, 0], 'optimizer in place');

// loading
$path = sys_get_temp_dir() . '/rlx-php-test.gguf';
$gw = new GgufWriter();
$gw->addF32('w', [4], 'F32', [0.5, -1, 2, 0.25]);
check($gw->write($path, architecture: 'demo') === 1, 'write');
$f = Gguf::open($path);
check($f->architecture() === 'demo', 'architecture');
check($f->tensor('w')[2] == 2, 'tensor');
check($f->info('w')['dtype'] === 'F32', 'info');
unlink($path);

// backends and a one-rank mesh
check(Rlx\devices()[0] === 'cpu' && Rlx\isAvailable('cpu'), 'cpu');
[$g] = lossGraph();
$runner = new Runner($g, policy: ['only' => ['cpu']]);
$runner->setParam('w', [1, 0, 0, 1]);
$res = $runner->run(['x' => [1, 2], 'y' => [0, 0]]);
check($res['device'] === 'cpu' && $res['outputs'][0][0] == 2.5, 'runner');
check(Group::connect(0, 1)->allReduce([2, 3], op: 'max')[1] == 3, 'all_reduce');
echo "test_rlx: $checks checks passed\n";
