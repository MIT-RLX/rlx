// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import org.rlx.RLX;

/**
 * Distributed: three ranks as threads on loopback; across machines, one rank
 * per host with the same peers. Collectives, pipelined inference (a stage
 * shipped per worker), data-parallel training.
 */
public final class Distributed {
    static final int WORLD = 3, PER_RANK = 16;
    static Path tmp;

    static String blob(String name, float[] values) throws Exception {
        var b = ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) b.putFloat(v);
        Path p = tmp.resolve(name);
        Files.write(p, b.array());
        return "file://" + p;
    }

    static float[] matvec(float[] x, float[] w) {
        float[] out = new float[4];
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int i = 0; i < 4; i++) s += x[i] * w[i * 4 + j];
            out[j] = Math.max(0, s);
        }
        return out;
    }

    static Map<String, Object> spec(int rank, Map<String, String> uris) {
        int start = rank * PER_RANK;
        return Map.of(
                "params", List.of(Map.of("name", "w", "uri", uris.get("w"))),
                "data", List.of(
                        Map.of("input", "X", "uri", uris.get("X"), "elem", 3, "shard_start", start, "shard_len", PER_RANK),
                        Map.of("input", "y", "uri", uris.get("y"), "elem", 1, "shard_start", start, "shard_len", PER_RANK)),
                "batch", 4, "lr_per_epoch", Collections.nCopies(40, 0.2), "device", "cpu");
    }

    @SuppressWarnings("unchecked")
    static float[] weightsOf(Map<String, Object> report) {
        var w = (List<Object>) ((Map<String, Object>) report.get("params")).get("w");
        float[] out = new float[w.size()];
        for (int i = 0; i < out.length; i++) out[i] = ((Number) w.get(i)).floatValue();
        return out;
    }

    public static void main(String[] args) throws Exception {
        tmp = Files.createTempDirectory("rlx-dist-java");
        float[][] stageW = new float[3][16];
        for (int i = 0; i < 16; i++) {
            stageW[1][i] = (i * 5 % 7) / 7f - 0.3f;
            stageW[2][i] = (i * 3 % 5) / 5f - 0.2f;
        }
        float[] wTrue = {1.5f, -2, 0.5f};
        float[] xAll = new float[PER_RANK * WORLD * 3];
        for (int i = 0; i < xAll.length; i++) xAll[i] = (i * 37 % 23) / 11.5f - 1;
        float[] yAll = new float[PER_RANK * WORLD];
        for (int i = 0; i < yAll.length; i++) for (int k = 0; k < 3; k++) yAll[i] += xAll[i * 3 + k] * wTrue[k];

        var peers = RLX.loopbackPeers(WORLD);
        var connect = new RLX.Group.ConnectOpts().options(Map.of("peers", peers, "device", "cpu"));
        List<CompletableFuture<float[]>> workers = new ArrayList<>();
        for (int r = 1; r < WORLD; r++) {
            final int rank = r;
            workers.add(CompletableFuture.supplyAsync(() -> {
                var g = RLX.Group.connect(rank, WORLD, connect);
                g.allReduce(new float[] {rank + 1}, new RLX.Group.AllReduceOpts().op("sum"));
                g.allGather(new float[] {rank + 1});
                g.broadcast(0, new float[] {0}); // the root's length; it overwrites
                g.serveWorker(new RLX.Group.ServeWorkerOpts().options(Map.of("steps", 2)));
                return weightsOf(g.serveTrainer());
            }));
        }

        var g = RLX.Group.connect(0, WORLD, connect);
        System.out.println("mesh: rank " + g.rank() + " of " + g.world() + ", leader: " + g.isLeader());
        System.out.println("all_reduce sum: " + java.util.Arrays.toString(g.allReduce(new float[] {1}, new RLX.Group.AllReduceOpts().op("sum"))));
        System.out.println("all_gather: " + java.util.Arrays.toString(g.allGather(new float[] {1})));
        System.out.println("broadcast: " + java.util.Arrays.toString(g.broadcast(0, new float[] {42})));
        var caps = g.collectCaps();
        System.out.println("workers: " + String.join(", ", caps.stream().map(c -> "rank " + c.get("rank") + " on " + c.get("platform")).toList()));
        var stage = new RLX.Graph(new RLX.Graph.GraphOpts().name("stage"));
        int sx = stage.input("x", new long[] {1, 4});
        stage.setOutputs(new int[] {stage.relu(stage.matmul(sx, stage.param("w", new long[] {4, 4})))});
        for (int r = 1; r < WORLD; r++) {
            g.shipStage(r, stage, Map.of("device", "cpu", "weights", Map.of("w", blob("w" + r + ".bin", stageW[r]))));
        }
        for (float[] xv : new float[][] {{1, -1, 0.5f, 2}, {0.25f, 0.5f, -0.75f, 1}}) {
            g.sendActivation(1, xv);
            float[] yv = g.recvActivation(WORLD - 1);
            float[] want = matvec(matvec(xv, stageW[1]), stageW[2]);
            double d = 0;
            for (int i = 0; i < 4; i++) d = Math.max(d, Math.abs(yv[i] - want[i]));
            System.out.printf("pipeline max|Δ| vs local = %.1e%n", d);
        }

        g.collectCaps();
        var lr = new RLX.Graph(new RLX.Graph.GraphOpts().name("lr"));
        int X = lr.input("X", new long[] {4, 3});
        int Y = lr.input("y", new long[] {4, 1});
        int wv = lr.param("w", new long[] {3, 1});
        int d = lr.sub(lr.matmul(X, wv), Y);
        lr.setOutputs(new int[] {lr.mean(lr.mul(d, d), new long[] {0, 1})});
        var bwd = RLX.grad(lr, new int[] {wv});
        var uris = Map.of("X", blob("X.bin", xAll), "y", blob("y.bin", yAll), "w", blob("w0.bin", new float[3]));
        for (int r = 1; r < WORLD; r++) g.shipTrain(r, bwd, spec(r, uris));
        var mine = g.runTrain(bwd, spec(0, uris));
        @SuppressWarnings("unchecked")
        var m = (Map<String, Object>) mine.get("metrics");
        System.out.printf("training loss: %.4f -> %.2e%n", ((Number) m.get("first_loss")).doubleValue(), ((Number) m.get("last_loss")).doubleValue());
        float[] w0 = weightsOf(mine);
        System.out.printf("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)%n", w0[0], w0[1], w0[2]);
        for (int r = 1; r < WORLD; r++) {
            float[] theirs = workers.get(r - 1).get();
            boolean agree = true;
            for (int i = 0; i < 3; i++) agree &= Math.abs(theirs[i] - w0[i]) < 1e-5;
            System.out.println("rank " + r + " w agrees: " + agree);
        }
    }
}
