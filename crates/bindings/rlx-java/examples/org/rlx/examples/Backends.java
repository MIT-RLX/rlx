// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.util.List;
import java.util.Map;
import org.rlx.RLX;

/**
 * Picking a backend: what this build has, what suits a graph, and the ways to
 * choose — by name, by cost model, by policy, by benchmark, per run.
 */
public final class Backends {
    static RLX.Graph model() {
        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("mlp"));
        int x = g.input("x", new long[] {4, 16});
        g.setOutputs(new int[] {g.tanh(g.matmul(x, g.param("w", new long[] {16, 16})))});
        return g;
    }

    public static void main(String[] args) {
        float[] xs = new float[64];
        for (int i = 0; i < 64; i++) xs[i] = (i % 7) / 7f;
        var x = Map.of("x", xs);
        float[] w = new float[256];
        for (int i = 0; i < 256; i++) w[i] = (i % 5) / 10f - 0.2f;

        List<String> available = RLX.devices();
        System.out.println("devices: " + String.join(" ", available));
        System.out.println("metal available: " + RLX.isAvailable("metal"));
        System.out.println("fastest for graph: " + RLX.fastestDeviceFor(model()));
        float[] reference = null;
        for (String device : available) {
            var c = new RLX.Session(new RLX.Session.SessionOpts().device(device)).compile(model());
            c.setParam("w", w);
            float[] y = c.run(x)[0];
            if (reference == null) reference = y;
            double d = 0;
            for (int i = 0; i < y.length; i++) d = Math.max(d, Math.abs(y[i] - reference[i]));
            System.out.printf("session %s: max|Δ| vs first = %.2e%n", device, d);
        }
        var flex = new RLX.FlexibleSession(new RLX.FlexibleSession.FlexibleSessionOpts().policy(Map.of("prefer", List.of("metal", "cpu"))));
        System.out.println("flexible session picked: " + flex.compile(model()).device());
        var runner = new RLX.Runner(model());
        runner.setParam("w", w);
        System.out.println("runner warmed: " + String.join(" ", runner.warmAll()));
        for (var row : runner.benchmark(x, new RLX.Runner.BenchmarkOpts().runs(3L))) {
            System.out.println("  bench " + row.get("device") + ": exec_ms=" + row.get("exec_ms"));
        }
        System.out.println("runner ran on: " + runner.run(x).device());
        System.out.println("run_try chain: " + runner.runTry(new String[] {"metal", "cpu"}, x).device());
        var router = new RLX.Router(model());
        router.setParam("w", w);
        System.out.println("router ran on: " + router.runChain(x).device());
    }
}
