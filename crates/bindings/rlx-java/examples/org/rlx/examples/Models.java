// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.rlx.RLX;

/**
 * Models: any provider's model, opened by path or id and run by name. A graph
 * saved as an .rlxp package runs anywhere; a plugin's models (a downstream collection's LLMs,
 * or the reference plugin's demo:count) load through RLX_MODEL_PLUGINS.
 */
public final class Models {
    public static void main(String[] args) throws Exception {
        var g = new RLX.Graph(new RLX.Graph.GraphOpts().name("affine"));
        int x = g.input("x", new long[] {1, 2});
        g.setOutputs(new int[] {g.matmul(x, g.param("w", new long[] {2, 2}))});
        Path path = Path.of(System.getProperty("java.io.tmpdir"), "rlx-example-java.rlxp");
        RLX.saveModel(g, path.toString(), Map.of("w", new float[] {2, 0, 0, 3}),
                new RLX.SaveModelOpts().model(Map.of("task", "graph", "family", "affine", "outputs", List.of("y"))));

        try (var m = RLX.Model.open(path.toString(), new RLX.Model.OpenOpts().options(Map.of("device", "cpu")))) {
            var info = m.info();
            System.out.println("model: " + info.get("family") + " (" + info.get("task") + ") on " + info.get("device"));
            float[] y = m.run(Map.of("x", new float[] {1, 1}))[0];
            System.out.printf("model run: %.3f %.3f%n", y[0], y[1]);
        } finally {
            Files.deleteIfExists(path);
        }

        try (var lm = RLX.Model.open("demo:count")) {
            var info = lm.info();
            System.out.println("plugin model: " + info.get("family") + " (" + info.get("task") + ")");
            System.out.println("model text: " + lm.generateText("c", new RLX.Model.GenerateTextOpts().options(Map.of("max_tokens", 4))));
            int[] ids = lm.generate(new int[] {2}, new RLX.Model.GenerateOpts().maxTokens(3));
            System.out.println("model ids: " + Arrays.stream(ids).mapToObj(String::valueOf).collect(Collectors.joining(" ")));

            // By task: a chat, speech out and back in, embeddings.
            var msgs = List.of(Map.of("role", "user", "content", "c"));
            System.out.println("chat: " + lm.chat(msgs, new RLX.Model.ChatOpts().options(Map.of("max_tokens", 4))));
        }
        try (var voice = RLX.Model.open("demo:voice"); var ear = RLX.Model.open("demo:speech");
                var embedder = RLX.Model.open("demo:embed")) {
            var speech = voice.synthesize("hello");
            System.out.println("voice: " + speech.audio().length + " samples at " + speech.sampleRate() + " Hz");
            System.out.println("heard: " + ear.transcribe(speech.audio(), new RLX.Model.TranscribeOpts().sampleRate(speech.sampleRate())));
            var rows = (List<?>) embedder.embed(new String[] {"hello", "sky"});
            System.out.println("embed: " + rows.stream()
                    .flatMap(r -> ((List<?>) r).stream())
                    .map(v -> String.valueOf(((Number) v).intValue()))
                    .collect(Collectors.joining(" ")));
        }
    }
}
