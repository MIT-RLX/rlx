<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//   php examples/loading.php

declare(strict_types=1);

require __DIR__ . '/../src/rlx.php';

use Rlx\Gguf;
use Rlx\GgufWriter;
use Rlx\Graph;
use Rlx\Session;

$tmp = sys_get_temp_dir();
$w = array_map(fn ($i) => ($i % 9) / 4 - 1, range(0, 63));

$gguf = "$tmp/rlx-example-php.gguf";
$writer = new GgufWriter();
$writer->addF32('proj.weight', [8, 8], 'F32', $w);
$writer->addF32('proj.weight_q8', [8, 8], 'Q8_0', $w);
$writer->write($gguf, architecture: 'demo', metadata: ['demo.layers' => 1]);
$f = Gguf::open($gguf);
echo "gguf architecture: {$f->architecture()}\n";
echo 'gguf tensors: ', implode(' ', $f->tensorNames()), "\n";
$exact = $f->tensor('proj.weight');
$q8 = $f->tensor('proj.weight_q8');
$maxd = max(array_map(fn ($a, $b) => abs($a - $b), $exact, $q8));
printf("gguf f32 exact: %s q8 max|Δ|: %.4f\n", $exact == $w ? 'true' : 'false', $maxd);

$g = new Graph(name: 'proj');
$x = $g->input('x', [1, 8]);
$g->setOutputs([$g->matmul($x, $g->param('proj.weight', [8, 8]))]);
$c = (new Session())->compile($g);
$c->setParam('proj.weight', $exact);
$y = $c->run(['x' => array_fill(0, 8, 1.0)])[0];
printf("gguf -> graph: %.2f %.2f %.2f\n", $y[0], $y[1], $y[2]);

$st = "$tmp/rlx-example-php.safetensors";
$header = '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}';
$header .= str_repeat(' ', (8 - strlen($header) % 8) % 8);
file_put_contents($st, pack('P', strlen($header)) . $header . pack('g*', ...$w));
$loaded = Rlx\loadWeights('safetensors', $st);
echo 'safetensors: [proj.weight] exact: ', $loaded['proj.weight'] == $w ? 'true' : 'false', "\n";

$packed = Rlx\quantize($w, 'Q8_0');
printf("quantize q8_0: %d -> %d bytes, round trip %d values\n", count($w) * 4, strlen($packed),
    count(Rlx\dequantize($packed, 'Q8_0', count: count($w))));
unlink($gguf);
unlink($st);
echo 'argmax: ', Rlx\argmax([0.1, 2, 0.3]), "\n";
