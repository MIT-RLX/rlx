// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.LongFunction;

import static java.lang.foreign.ValueLayout.*;

/**
 * The rows of the runtime API table marked {@code manual}: an optimizer step
 * that updates the caller's arrays in place, and the training loop (driven
 * from Java over {@code step}).
 */
public final class Manual {
    private Manual() {}

    /** {@code Optimizer.step}. */
    public interface OptimizerRows {
        /** Update each {@code params[name]} in place from {@code grads[name]}. */
        default void step(Map<String, float[]> params, Map<String, float[]> grads) {
            var names = new ArrayList<>(new TreeMap<>(params).keySet());
            int n = names.size();
            try (Arena a = Arena.ofConfined()) {
                MemorySegment cn = a.allocate(ADDRESS, Math.max(n, 1));
                MemorySegment ps = a.allocate(ADDRESS, Math.max(n, 1));
                MemorySegment gs = a.allocate(ADDRESS, Math.max(n, 1));
                MemorySegment ls = a.allocate(JAVA_LONG, Math.max(n, 1));
                List<MemorySegment> bufs = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    float[] p = params.get(names.get(i));
                    float[] g = grads.get(names.get(i));
                    if (g == null || g.length != p.length) {
                        throw new RLXException(RLXException.Kind.TYPE, "Optimizer.step: no matching gradient for '" + names.get(i) + "'");
                    }
                    MemorySegment pb = a.allocateFrom(JAVA_FLOAT, p);
                    bufs.add(pb);
                    cn.setAtIndex(ADDRESS, i, a.allocateFrom(names.get(i)));
                    ps.setAtIndex(ADDRESS, i, pb);
                    gs.setAtIndex(ADDRESS, i, a.allocateFrom(JAVA_FLOAT, g));
                    ls.setAtIndex(JAVA_LONG, i, p.length);
                }
                Support.check((int) Support.call("rlx_optimizer_step",
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG),
                        ((Support.Handle) this).h, cn, ps, gs, ls, MemorySegment.NULL, MemorySegment.NULL, (long) n));
                for (int i = 0; i < n; i++) {
                    float[] back = bufs.get(i).toArray(JAVA_FLOAT);
                    System.arraycopy(back, 0, params.get(names.get(i)), 0, back.length);
                }
            }
        }
    }

    /** Called after a step with its loss; return {@code false} to stop. */
    @FunctionalInterface
    public interface OnStep {
        boolean apply(long step, float loss);
    }

    /** Options for {@code Trainer.run}. */
    public static final class RunOpts {
        /** The learning rate per step ({@code null}: unchanged). */
        public LongFunction<Float> lr;
        /** Called every {@link #every} steps. */
        public OnStep onStep;
        public int every = 1;

        public RunOpts lr(LongFunction<Float> f) { this.lr = f; return this; }
        public RunOpts onStep(OnStep f) { this.onStep = f; return this; }
        public RunOpts every(int n) { this.every = n; return this; }
    }

    /** {@code Trainer.run}. */
    public interface TrainerRows {
        long steps();
        float step(Map<String, float[]> batch);
        void setLr(float lr);
        Float gradNorm();

        /** Train {@code steps} steps, pulling inputs from {@code batch} ({@code null}: data exhausted). */
        default Map<String, Object> run(int steps, LongFunction<Map<String, float[]>> batch) {
            return run(steps, batch, null);
        }

        /** Train with a learning-rate schedule and a per-step callback; returns the summary. */
        default Map<String, Object> run(int steps, LongFunction<Map<String, float[]>> batch, RunOpts opts) {
            if (opts == null) opts = new RunOpts();
            int every = Math.max(opts.every, 1);
            long start = steps();
            List<Float> losses = new ArrayList<>();
            boolean stopped = false;
            long t0 = System.nanoTime();
            for (int i = 0; i < steps; i++) {
                long s = start + i;
                Map<String, float[]> b = batch.apply(s);
                if (b == null) break;
                if (opts.lr != null) setLr(opts.lr.apply(s));
                float loss = step(b);
                losses.add(loss);
                if (opts.onStep != null && (i + 1) % every == 0 && !opts.onStep.apply(s, loss)) {
                    stopped = true;
                    break;
                }
            }
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("steps", losses.size());
            summary.put("stopped", stopped);
            summary.put("first_loss", losses.isEmpty() ? null : losses.get(0));
            summary.put("last_loss", losses.isEmpty() ? null : losses.get(losses.size() - 1));
            summary.put("mean_loss", losses.stream().mapToDouble(Float::doubleValue).average().orElse(Double.NaN));
            summary.put("steps_per_second", losses.size() / Math.max((System.nanoTime() - t0) / 1e9, 1e-9));
            summary.put("grad_norm", gradNorm());
            return summary;
        }
    }
}
