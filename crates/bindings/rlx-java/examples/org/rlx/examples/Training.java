// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.rlx.Manual;
import org.rlx.RLX;

/**
 * Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
 * the same fit by hand with {@code grad} and an Optimizer.
 */
public final class Training {
    static final float[] W_TRUE = {2, -1, 0.5f};
    static final int BATCH = 8;

    static Map<String, float[]> data(long step) {
        float[] x = new float[BATCH * 3];
        float[] y = new float[BATCH];
        for (int i = 0; i < BATCH; i++) {
            for (int k = 0; k < 3; k++) {
                x[i * 3 + k] = ((step * 31 + i * 7 + k * 3) % 17) / 8.5f - 1;
                y[i] += x[i * 3 + k] * W_TRUE[k];
            }
        }
        return new HashMap<>(Map.of("x", x, "y", y));
    }

    static String fmt(float[] v) {
        return IntStream.range(0, v.length).mapToObj(i -> String.format("%.3f", v[i])).collect(Collectors.joining(" "));
    }

    public static void main(String[] args) throws Exception {
        String device = args.length > 0 ? args[0] : "cpu";
        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("regression"));
        int x = g.input("x", new long[] {BATCH, 3});
        int y = g.input("y", new long[] {BATCH, 1});
        int w = g.param("w", new long[] {3, 1});
        int d = g.sub(g.matmul(x, w), y);
        g.setOutputs(new int[] {g.mean(g.mul(d, d), new long[] {0, 1})});

        var trainer = new RLX.Trainer(g.clone(), new String[] {"w"}, Map.of("w", new float[3]),
                new RLX.Trainer.TrainerOpts().options(Map.of("device", device, "optimizer", Map.of("kind", "adamw", "lr", 0.1))));
        float first = trainer.evaluate(data(0));
        trainer.run(400, Training::data, new Manual.RunOpts()
                .lr(s -> (float) (0.1 * 0.5 * (1 + Math.cos(Math.PI * s / 400))))
                .onStep((s, loss) -> loss > 1e-6));
        System.out.printf("trainer loss: %.4f -> %.6f in %d steps%n", first, trainer.evaluate(data(0)), trainer.steps());
        System.out.println("trainer w: " + fmt(trainer.params().get("w")));

        var path = Files.createTempFile("rlx-example-train-java", ".gguf");
        long written = trainer.save(path.toString());
        var restored = new RLX.Trainer(g.clone(), new String[] {"w"}, Map.of("w", new float[3]),
                new RLX.Trainer.TrainerOpts().options(Map.of("device", device)));
        System.out.println("checkpoint: " + written + " tensors, restored step " + restored.load(path.toString()).get("step"));
        Files.deleteIfExists(path);
        restored.freeze(new RLX.Trainer.FreezeOpts().names(new String[] {"w"}));
        float[] before = restored.params().get("w");
        restored.step(data(1));
        System.out.println("frozen: " + restored.frozen() + " unchanged: " + Arrays.equals(restored.params().get("w"), before));
        restored.unfreeze();

        var backward = new RLX.Session(new RLX.Session.SessionOpts().device(device)).compile(RLX.grad(g, new int[] {w}));
        var opt = new RLX.Optimizer(new RLX.Optimizer.OptimizerOpts().kind("sgd").options(Map.of("lr", 0.2)));
        float[] params = new float[3];
        float loss = 0;
        for (int step = 0; step < 300; step++) {
            backward.setParam("w", params);
            var feed = data(step);
            feed.put("d_output", new float[] {1});
            float[][] outs = backward.run(feed);
            loss = outs[0][0];
            opt.step(Map.of("w", params), Map.of("w", outs[1]));
        }
        System.out.printf("manual loss: %.6f w: %s%n", loss, fmt(params));
    }
}
