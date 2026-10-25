<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: PHP is single-threaded, so each rank is a process; start one per
// rank with the same peers (here three on loopback; across machines, one per
// host):
//   php examples/distributed.php 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   php examples/distributed.php 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   php examples/distributed.php 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
// Rank 0 coordinates: collectives, pipelined inference (a stage shipped per
// worker), data-parallel training. With no arguments it is a mesh of one.

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Graph;
use Rlx\Group;

$rank = (int) ($argv[1] ?? 0);
$world = (int) ($argv[2] ?? 1);
const PER_RANK = 16;
$blobs = [];
$blob = function (string $name, array $values) use (&$blobs, $rank): string {
    $path = sys_get_temp_dir() . "/rlx-dist-php-$rank-$name";
    file_put_contents($path, pack('g*', ...$values));
    $blobs[] = $path;
    return "file://$path";
};

$options = ['device' => 'cpu'];
if (isset($argv[3])) {
    $options['peers'] = explode(',', $argv[3]);
}
$g = Group::connect($rank, $world, options: $options);
$sum = $g->allReduce([$rank + 1], op: 'sum');
$gathered = $g->allGather([$rank + 1]);
$shared = $g->broadcast(0, [$rank === 0 ? 42 : 0]); // the root's length
if ($rank === 0) {
    printf("mesh: rank %d of %d, leader: %s\n", $g->rank(), $g->world(), $g->isLeader() ? 'true' : 'false');
    echo 'all_reduce sum: ', implode(',', $sum), "\n";
    echo 'all_gather: ', implode(',', $gathered), "\n";
    echo 'broadcast: ', implode(',', $shared), "\n";
}
if ($world === 1) {
    echo "a mesh of one: pass rank world peers to start more ranks\n";
    exit(0);
}
if ($rank !== 0) {
    $g->serveWorker(options: ['steps' => 2]);
    $g->send(0, 1, $g->serveTrainer()['params']['w']); // rank 0 compares
    exit(0);
}

$stageW = [
    array_map(fn ($i) => ($i * 5 % 7) / 7 - 0.3, range(0, 15)),
    array_map(fn ($i) => ($i * 3 % 5) / 5 - 0.2, range(0, 15)),
];
echo 'workers: ', implode(', ', array_map(fn ($c) => "rank {$c['rank']} on {$c['platform']}", $g->collectCaps())), "\n";
$stage = new Graph(name: 'stage');
$stage->setOutputs([$stage->relu($stage->matmul($stage->input('x', [1, 4]), $stage->param('w', [4, 4])))]);
for ($r = 1; $r < $world; $r++) {
    $g->shipStage($r, $stage, ['device' => 'cpu', 'weights' => ['w' => $blob("w$r.bin", $stageW[($r - 1) % 2])]]);
}
foreach ([[1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]] as $xv) {
    $g->sendActivation(1, $xv);
    echo 'pipeline: ', implode(' ', array_map(fn ($v) => sprintf('%.3f', $v), $g->recvActivation($world - 1))), "\n";
}

$g->collectCaps();
$lr = new Graph(name: 'lr');
$X = $lr->input('X', [4, 3]);
$Y = $lr->input('y', [4, 1]);
$wv = $lr->param('w', [3, 1]);
$d = $lr->sub($lr->matmul($X, $wv), $Y);
$lr->setOutputs([$lr->mean($lr->mul($d, $d), [0, 1])]);
$bwd = Rlx\grad($lr, [$wv]);
$wTrue = [1.5, -2, 0.5];
$n = PER_RANK * $world;
$xAll = array_map(fn ($i) => ($i * 37 % 23) / 11.5 - 1, range(0, $n * 3 - 1));
$yAll = [];
for ($i = 0; $i < $n; $i++) {
    $yAll[] = $xAll[$i * 3] * $wTrue[0] + $xAll[$i * 3 + 1] * $wTrue[1] + $xAll[$i * 3 + 2] * $wTrue[2];
}
$uris = ['X' => $blob('X.bin', $xAll), 'y' => $blob('y.bin', $yAll), 'w' => $blob('w0.bin', [0, 0, 0])];
$spec = fn (int $r) => [
    'params' => [['name' => 'w', 'uri' => $uris['w']]],
    'data' => [
        ['input' => 'X', 'uri' => $uris['X'], 'elem' => 3, 'shard_start' => $r * PER_RANK, 'shard_len' => PER_RANK],
        ['input' => 'y', 'uri' => $uris['y'], 'elem' => 1, 'shard_start' => $r * PER_RANK, 'shard_len' => PER_RANK],
    ],
    'batch' => 4,
    'lr_per_epoch' => array_fill(0, 40, 0.2),
    'device' => 'cpu',
];
for ($r = 1; $r < $world; $r++) {
    $g->shipTrain($r, $bwd, $spec($r));
}
$mine = $g->runTrain($bwd, $spec(0));
printf("training loss: %.4f -> %.2e\n", $mine['metrics']['first_loss'], $mine['metrics']['last_loss']);
$w0 = $mine['params']['w'];
printf("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)\n", ...$w0);
for ($r = 1; $r < $world; $r++) {
    $theirs = $g->recv($r, 1);
    $agree = count(array_filter(array_map(fn ($a, $b) => abs($a - $b) < 1e-5, $theirs, $w0))) === count($w0);
    echo "rank $r w agrees: ", $agree ? 'true' : 'false', "\n";
}
array_map('unlink', $blobs);
