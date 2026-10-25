// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

using Rlx;

/// Models: any provider's model, opened by path or id and run by name. A graph
/// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
/// collection's LLMs, or the reference plugin's demo:count) load through
/// RLX_MODEL_PLUGINS.
static class Models
{
    public static void Run()
    {
        var g = new RLX.Graph("affine");
        var x = g.Input("x", new long[] { 1, 2 });
        g.SetOutputs(new[] { g.Matmul(x, g.Param("w", new long[] { 2, 2 })) });
        var path = Path.Combine(Path.GetTempPath(), "rlx-example-csharp.rlxp");
        RLX.SaveModel(g, path, new Dictionary<string, float[]> { ["w"] = new float[] { 2, 0, 0, 3 } },
            new { task = "graph", family = "affine", outputs = new[] { "y" } });

        using (var m = RLX.Model.Open(path, new { device = "cpu" }))
        {
            var info = m.Info();
            Console.WriteLine($"model: {info["family"]} ({info["task"]}) on {info["device"]}");
            var y = m.Run(new Dictionary<string, float[]> { ["x"] = new float[] { 1, 1 } })[0];
            Console.WriteLine($"model run: {y[0]:F3} {y[1]:F3}");
        }
        File.Delete(path);

        using var lm = RLX.Model.Open("demo:count");
        var lmInfo = lm.Info();
        Console.WriteLine($"plugin model: {lmInfo["family"]} ({lmInfo["task"]})");
        Console.WriteLine($"model text: {lm.GenerateText("c", new { max_tokens = 4 })}");
        Console.WriteLine($"model ids: {string.Join(" ", lm.Generate(new uint[] { 2 }, 3))}");

        // By task: a chat, speech out and back in, embeddings.
        Console.WriteLine($"chat: {lm.Chat(new[] { new { role = "user", content = "c" } }, new { max_tokens = 4 })}");
        using var voice = RLX.Model.Open("demo:voice");
        var speech = voice.Synthesize("hello");
        Console.WriteLine($"voice: {speech.Audio.Length} samples at {speech.SampleRate} Hz");
        using var ear = RLX.Model.Open("demo:speech");
        Console.WriteLine($"heard: {ear.Transcribe(speech.Audio, speech.SampleRate)}");
        using var embedder = RLX.Model.Open("demo:embed");
        var rows = embedder.Embed(new[] { "hello", "sky" })!.AsArray();
        var values = rows.SelectMany(r => r!.AsArray().Select(v => ((int)v!.GetValue<double>()).ToString()));
        Console.WriteLine($"embed: {string.Join(" ", values)}");
    }
}
