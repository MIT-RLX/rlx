// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.rlx.RLX;

/**
 * Inference: build a graph, compile it for a device, set its weights, run it.
 * A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
 */
public final class Inference {
    static float[] weights(int n, int seed) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) w[i] = ((i * 7919 + seed * 104729) % 1000) / 1000f - 0.5f;
        return w;
    }

    public static void main(String[] args) {
        String device = args.length > 0 ? args[0] : "cpu";
        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("classifier"));
        int x = g.input("x", new long[] {1, 4});
        int hidden = g.relu(g.matmul(x, g.param("w1", new long[] {4, 8})));
        g.setOutputs(new int[] {g.softmax(g.matmul(hidden, g.param("w2", new long[] {8, 3})))});

        try (var model = new RLX.Session(new RLX.Session.SessionOpts().device(device)).compile(g)) { // consumes g
            System.out.println("compiled on: " + model.device());
            model.setParams(Map.of("w1", weights(32, 1), "w2", weights(24, 2)));
            float[] probs = model.run(Map.of("x", new float[] {0.5f, -1, 2, 0.25f}))[0];
            System.out.println("probs: " + IntStream.range(0, probs.length)
                    .mapToObj(i -> String.format("%.4f", probs[i])).collect(Collectors.joining(" ")));
            float sum = 0;
            for (float p : probs) sum += p;
            System.out.printf("probs sum: %.4f%n", sum);
            System.out.println("output shapes: " + Arrays.deepToString(model.outputShapes()) + " dtypes: " + model.outputDtypes());
        }
    }
}
