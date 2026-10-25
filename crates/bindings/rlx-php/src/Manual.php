<?php
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The two rows whose shape is not a plain call: Optimizer::step updates the
// caller's arrays in place (by reference), Trainer::run takes callbacks.

declare(strict_types=1);

namespace Rlx\Manual;

use FFI;
use Rlx\RlxException;
use Rlx\Support as S;

trait OptimizerRows
{
    /** Update each `$params[$name]` in place from `$grads[$name]`. */
    public function step(array &$params, array $grads): void
    {
        $names = array_map('strval', array_keys($params));
        sort($names, SORT_STRING);
        $n = count($names);
        $k = [];
        $ps = S::ffi()->new('float*[' . max($n, 1) . ']');
        $gs = S::ffi()->new('float*[' . max($n, 1) . ']');
        $lens = S::ffi()->new('size_t[' . max($n, 1) . ']');
        $bufs = [];
        foreach ($names as $i => $name) {
            if (!array_key_exists($name, $grads)) {
                throw new RlxException('type', "Optimizer step: no gradient for '$name'");
            }
            $bufs[$i] = S::arr($k, 'float', $params[$name]);
            $ps[$i] = $bufs[$i] === null ? null : S::ffi()->cast('float*', $bufs[$i]);
            $g = S::arr($k, 'float', $grads[$name]);
            $gs[$i] = $g === null ? null : S::ffi()->cast('float*', $g);
            $lens[$i] = count($params[$name]);
        }
        S::check(S::ffi()->rlx_optimizer_step($this->h, S::strs($k, $names), $ps, $gs, $lens, null, null, $n));
        foreach ($names as $i => $name) {
            $params[$name] = [];
            for ($j = 0; $j < $lens[$i]; $j++) {
                $params[$name][] = $bufs[$i][$j];
            }
        }
    }
}

trait TrainerRows
{
    /**
     * Train `$steps` steps, pulling inputs from `$batch($step)` (null: data
     * exhausted). `$lr($step)` schedules the rate; `$onStep($step, $loss)`
     * returning false stops early (called every `$every` steps).
     */
    public function run(int $steps, callable $batch, ?callable $lr = null, ?callable $onStep = null, int $every = 1): array
    {
        $every = max($every, 1);
        $start = $this->steps();
        $losses = [];
        $stopped = false;
        $t0 = hrtime(true);
        for ($i = 0; $i < $steps; $i++) {
            $s = $start + $i;
            $b = $batch($s);
            if ($b === null) {
                break;
            }
            if ($lr !== null) {
                $this->setLr($lr($s));
            }
            $loss = $this->step($b);
            $losses[] = $loss;
            if ($onStep !== null && ($i + 1) % $every === 0 && !$onStep($s, $loss)) {
                $stopped = true;
                break;
            }
        }
        $secs = max((hrtime(true) - $t0) / 1e9, 1e-9);
        return [
            'steps' => count($losses),
            'stopped' => $stopped,
            'firstLoss' => $losses[0] ?? null,
            'lastLoss' => $losses ? $losses[count($losses) - 1] : null,
            'meanLoss' => $losses ? array_sum($losses) / count($losses) : null,
            'stepsPerSecond' => count($losses) / $secs,
            'gradNorm' => $this->gradNorm(),
        ];
    }
}
