// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// The two rows whose shape is not a plain call, as UFCS functions:
/// `opt.step(params, grads)` updates the caller's slices in place,
/// `trainer.run(steps, batch, …)` takes callbacks.
module rlx.manual;

import core.time : MonoTime;
import std.algorithm : sort;
import std.string : toStringz;
import std.typecons : Nullable, nullable;

import rlx.generated;
import rlx.support;

/// Update each `params[name]` in place from `grads[name]`.
void step(Optimizer opt, float[][string] params, const(float[])[string] grads)
{
    auto names = params.keys.dup;
    names.sort();
    const(char)*[] cnames;
    float*[] ps;
    const(float)*[] gs;
    size_t[] lens;
    foreach (name; names)
    {
        auto g = name in grads;
        if (g is null)
            throw new RlxException(RlxErrorKind.type, "Optimizer.step: no gradient for '" ~ name ~ "'");
        cnames ~= name.toStringz;
        ps ~= params[name].ptr;
        gs ~= (*g).ptr;
        lens ~= params[name].length;
    }
    checkRc(rlx_optimizer_step(opt.h, cnames.ptr, ps.ptr, gs.ptr, lens.ptr, null, null, names.length));
}

/// What `Trainer.run` did.
struct TrainSummary
{
    size_t steps;
    bool stopped;
    Nullable!float firstLoss, lastLoss, meanLoss, gradNorm;
    double stepsPerSecond;
}

/// Train `steps` steps, pulling inputs from `batch(step)` (null: data
/// exhausted). `lr(step)` schedules the rate; `onStep(step, loss)` returning
/// false stops early (called every `every` steps).
TrainSummary run(Trainer t, size_t steps, const(float[])[string] delegate(size_t step) batch,
    double delegate(size_t step) lr = null, bool delegate(size_t step, float loss) onStep = null,
    size_t every = 1)
{
    every = every < 1 ? 1 : every;
    immutable start = t.steps();
    float[] losses;
    TrainSummary r;
    immutable t0 = MonoTime.currTime;
    foreach (i; 0 .. steps)
    {
        immutable s = start + i;
        auto b = batch(s);
        if (b is null)
            break;
        if (lr !is null)
            t.setLr(cast(float) lr(s));
        immutable loss = t.step(b);
        losses ~= loss;
        if (onStep !is null && (i + 1) % every == 0 && !onStep(s, loss))
        {
            r.stopped = true;
            break;
        }
    }
    immutable secs = (MonoTime.currTime - t0).total!"nsecs" / 1e9;
    r.steps = losses.length;
    if (losses.length)
    {
        r.firstLoss = losses[0];
        r.lastLoss = losses[$ - 1];
        float sum = 0;
        foreach (l; losses)
            sum += l;
        r.meanLoss = sum / losses.length;
    }
    r.stepsPerSecond = losses.length / (secs > 1e-9 ? secs : 1e-9);
    r.gradNorm = t.gradNorm();
    return r;
}
