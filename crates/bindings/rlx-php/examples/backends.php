<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//   php examples/backends.php

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\FlexibleSession;
use Rlx\Graph;
use Rlx\Router;
use Rlx\Runner;
use Rlx\Session;

function model(): Graph
{
    $g = new Graph(name: 'mlp');
    $x = $g->input('x', [4, 16]);
    $g->setOutputs([$g->tanh($g->matmul($x, $g->param('w', [16, 16])))]);
    return $g;
}
$w = array_map(fn ($i) => ($i % 5) / 10 - 0.2, range(0, 255));
$x = ['x' => array_map(fn ($i) => ($i % 7) / 7, range(0, 63))];

$available = Rlx\devices();
echo 'devices: ', implode(' ', $available), "\n";
echo 'metal available: ', Rlx\isAvailable('metal') ? 'true' : 'false', "\n";
echo 'fastest for graph: ', Rlx\fastestDeviceFor(model()), "\n";
$reference = null;
foreach ($available as $device) {
    $c = (new Session(device: $device))->compile(model());
    $c->setParam('w', $w);
    $y = $c->run($x)[0];
    $reference ??= $y;
    printf("session %s: max|Δ| vs first = %.2e\n", $device, max(array_map(fn ($a, $b) => abs($a - $b), $y, $reference)));
}
$flex = new FlexibleSession(policy: ['prefer' => ['metal', 'cpu']]);
echo 'flexible session picked: ', $flex->compile(model())->device(), "\n";
$runner = new Runner(model());
$runner->setParam('w', $w);
echo 'runner warmed: ', implode(' ', $runner->warmAll()), "\n";
foreach ($runner->benchmark($x, runs: 3) as $row) {
    echo "  bench {$row['device']}: exec_ms={$row['exec_ms']}\n";
}
echo 'runner ran on: ', $runner->run($x)['device'], "\n";
echo 'run_try chain: ', $runner->runTry(['metal', 'cpu'], $x)['device'], "\n";
$router = new Router(model());
$router->setParam('w', $w);
echo 'router ran on: ', $router->runChain($x)['device'], "\n";
