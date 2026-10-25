// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Map;
import org.rlx.RLX;

/**
 * Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
 * into a compiled graph, and quantizing in memory. RLX_TOKENIZER=&lt;tokenizer.json&gt;
 * adds the tokenizer.
 */
public final class Loading {
    public static void main(String[] args) throws Exception {
        float[] w = new float[64];
        for (int i = 0; i < 64; i++) w[i] = (i % 9) / 4f - 1;
        var gguf = Files.createTempFile("rlx-example-java", ".gguf");
        try (var writer = new RLX.GgufWriter()) {
            writer.addF32("proj.weight", new long[] {8, 8}, "F32", w);
            writer.addF32("proj.weight_q8", new long[] {8, 8}, "Q8_0", w);
            writer.write(gguf.toString(), new RLX.GgufWriter.WriteOpts().architecture("demo").metadata(Map.of("demo.layers", 1)));
        }
        float[] exact;
        try (var f = RLX.Gguf.open(gguf.toString())) {
            System.out.println("gguf architecture: " + f.architecture());
            System.out.println("gguf tensors: " + String.join(" ", f.tensorNames()));
            exact = f.tensor("proj.weight");
            float[] q8 = f.tensor("proj.weight_q8");
            double maxd = 0;
            for (int i = 0; i < 64; i++) maxd = Math.max(maxd, Math.abs(exact[i] - q8[i]));
            System.out.printf("gguf f32 exact: %s q8 max|Δ|: %.4f%n", Arrays.equals(exact, w), maxd);
        }

        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("proj"));
        int x = g.input("x", new long[] {1, 8});
        g.setOutputs(new int[] {g.matmul(x, g.param("proj.weight", new long[] {8, 8}))});
        var c = new RLX.Session().compile(g);
        c.setParam("proj.weight", exact);
        float[] ones = new float[8];
        Arrays.fill(ones, 1);
        float[] y = c.run(Map.of("x", ones))[0];
        System.out.printf("gguf -> graph: %.2f %.2f %.2f%n", y[0], y[1], y[2]);

        var st = Files.createTempFile("rlx-example-java", ".safetensors");
        String header = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0," + 4 * w.length + "]}}";
        while (header.length() % 8 != 0) header += " ";
        var buf = ByteBuffer.allocate(8 + header.length() + 4 * w.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.putLong(header.length()).put(header.getBytes(StandardCharsets.UTF_8));
        for (float v : w) buf.putFloat(v);
        Files.write(st, buf.array());
        var loaded = RLX.loadWeights("safetensors", st.toString());
        System.out.println("safetensors: " + loaded.keySet() + " exact: " + Arrays.equals(loaded.get("proj.weight"), w));

        byte[] packed = RLX.quantize(w, "Q8_0");
        float[] back = RLX.dequantize(packed, "Q8_0", new RLX.DequantizeOpts().count((long) w.length));
        System.out.println("quantize q8_0: " + 4 * w.length + " -> " + packed.length + " bytes, round trip " + back.length + " values");
        Files.deleteIfExists(gguf);
        Files.deleteIfExists(st);

        String tok = System.getenv("RLX_TOKENIZER");
        if (tok != null) {
            try (var t = RLX.Tokenizer.load(tok)) {
                int[] ids = t.encode("Hello from Java");
                System.out.println("tokenizer: " + ids.length + " ids -> \"" + t.decode(ids) + "\"");
            }
        } else {
            System.out.println("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)");
        }
        System.out.println("argmax: " + RLX.argmax(new float[] {0.1f, 2, 0.3f}));
    }
}
