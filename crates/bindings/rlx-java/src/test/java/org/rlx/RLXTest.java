// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * The Java binding against the real library — plain {@code java}, no test
 * framework, so it runs anywhere a JDK is: exits non-zero on the first failure.
 */
public final class RLXTest {
    private static int checks = 0;

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) throw new AssertionError("failed: " + what);
    }

    /** loss = mean((x·w − y)²), x/y [1, 2], w [2, 2]; returns {graph, w}. */
    private static Object[] lossGraph() {
        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("mse"));
        int x = g.input("x", new long[] {1, 2});
        int y = g.input("y", new long[] {1, 2});
        int w = g.param("w", new long[] {2, 2});
        int d = g.sub(g.matmul(x, w), y);
        g.setOutputs(new int[] {g.mean(g.mul(d, d), new long[] {0, 1})});
        return new Object[] {g, w};
    }

    public static void main(String[] args) throws Exception {
        // inference
        var gw = lossGraph();
        var g = (RLX.Graph) gw[0];
        try (var m = new RLX.Session().compile(g)) {
            check(g.isConsumed(), "compile consumes");
            m.setParam("w", new float[] {1, 0, 0, 1});
            check(m.run(Map.of("x", new float[] {1, 2}, "y", new float[] {0, 0}))[0][0] == 2.5f, "loss 2.5");
            check(m.device().equals("cpu"), "device");
        }

        // errors
        try {
            new RLX.Graph().relu(42);
            check(false, "bad node throws");
        } catch (RLXException e) {
            check(e.kind() == RLXException.Kind.RANGE && e.getMessage().contains("does not exist"), "range error");
        }

        // training
        gw = lossGraph();
        g = (RLX.Graph) gw[0];
        var bwd = new RLX.Session().compile(RLX.grad(g, new int[] {(int) gw[1]}));
        bwd.setParam("w", new float[] {1, 0, 0, 1});
        var outs = bwd.run(Map.of("x", new float[] {1, 2}, "y", new float[] {0, 0}, "d_output", new float[] {1}));
        check(outs[1][3] == 4f, "dw");
        var t = new RLX.Trainer(g.clone(), new String[] {"w"}, Map.of("w", new float[4]),
                new RLX.Trainer.TrainerOpts().options(Map.of("optimizer", Map.of("kind", "adamw", "lr", 0.1))));
        var batch = Map.of("x", new float[] {1, 1}, "y", new float[] {1, -1});
        var summary = t.run(300, s -> batch, new Manual.RunOpts().onStep((s, l) -> l > 1e-6));
        check((Float) summary.get("last_loss") < 1e-4f, "trainer converges");
        Path ck = Files.createTempFile("rlx-java", ".gguf");
        check(t.save(ck.toString()) >= 1, "checkpoint");
        Files.deleteIfExists(ck);
        var opt = new RLX.Optimizer(new RLX.Optimizer.OptimizerOpts().kind("sgd").options(Map.of("lr", 0.5)));
        float[] p = {1, -1};
        opt.step(Map.of("p", p), Map.of("p", new float[] {2, -2}));
        check(p[0] == 0 && p[1] == 0, "optimizer in place");

        // loading
        Path path = Files.createTempFile("rlx-java", ".gguf");
        float[] w = {0.5f, -1, 2, 0.25f};
        try (var wr = new RLX.GgufWriter()) {
            wr.addF32("w", new long[] {4}, "F32", w);
            check(wr.write(path.toString(), new RLX.GgufWriter.WriteOpts().architecture("demo")) == 1, "write");
        }
        try (var f = RLX.Gguf.open(path.toString())) {
            check("demo".equals(f.architecture()), "architecture");
            check(java.util.Arrays.equals(f.tensor("w"), w), "tensor");
            check("F32".equals(f.info("w").get("dtype")), "info");
        }
        Files.deleteIfExists(path);

        // backends
        check(RLX.devices().contains("cpu") && RLX.isAvailable("cpu"), "cpu");
        g = (RLX.Graph) lossGraph()[0];
        var r = new RLX.Runner(g, new RLX.Runner.RunnerOpts().policy(Map.of("only", List.of("cpu"))));
        r.setParam("w", new float[] {1, 0, 0, 1});
        var res = r.run(Map.of("x", new float[] {1, 2}, "y", new float[] {0, 0}));
        check(res.device().equals("cpu") && res.outputs()[0][0] == 2.5f, "runner");

        // a two-rank mesh, the worker on another thread
        var peers = RLX.loopbackPeers(2);
        var worker = CompletableFuture.supplyAsync(() -> {
            var wg = RLX.Group.connect(1, 2, new RLX.Group.ConnectOpts().options(Map.of("peers", peers)));
            wg.allReduce(new float[] {3}, new RLX.Group.AllReduceOpts().op("max"));
            return wg.serveWorker(new RLX.Group.ServeWorkerOpts().options(Map.of("steps", 1)));
        });
        var mesh = RLX.Group.connect(0, 2, new RLX.Group.ConnectOpts().options(Map.of("peers", peers)));
        check(mesh.allReduce(new float[] {5}, new RLX.Group.AllReduceOpts().op("max"))[0] == 5f, "all_reduce");
        check(mesh.collectCaps().size() == 1, "caps");
        var s = new RLX.Graph();
        int sx = s.input("x", new long[] {2});
        s.setOutputs(new int[] {s.add(sx, sx)});
        mesh.shipStage(1, s, Map.of("device", "cpu"));
        mesh.sendActivation(1, new float[] {1, 2});
        float[] y = mesh.recvActivation(1);
        check(y[0] == 2 && y[1] == 4, "activation");
        check(((Number) worker.get().get("activations")).intValue() == 1, "worker served");

        System.out.println("RLXTest: " + checks + " checks passed");
    }
}
