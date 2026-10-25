// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//   ldc2 -i -Isource examples/models.d -L-lrlx_capi && RLX_MODEL_PLUGINS=… ./models
module models;

import std.algorithm : map;
import std.array : join;
import std.conv : to;
import std.file : remove, tempDir;
import std.path : buildPath;
import std.stdio : writefln, writeln;
import rlx;

void main()
{
    auto g = new Graph(name: "affine");
    immutable x = g.input("x", [1, 2]);
    g.setOutputs([g.matmul(x, g.param("w", [2, 2]))]);
    immutable path = buildPath(tempDir, "rlx-example-d.rlxp");
    saveModel(g, path, ["w": [2.0f, 0, 0, 3]],
        parseJSON(`{"task": "graph", "family": "affine", "outputs": ["y"]}`));

    auto m = Model.open(path, parseJSON(`{"device": "cpu"}`));
    auto info = m.info();
    writefln("model: %s (%s) on %s", info["family"].str, info["task"].str, info["device"].str);
    auto y = m.run(["x": [1.0f, 1]])[0];
    writefln("model run: %.3f %.3f", y[0], y[1]);
    remove(path);

    auto lm = Model.open("demo:count");
    auto lmInfo = lm.info();
    writefln("plugin model: %s (%s)", lmInfo["family"].str, lmInfo["task"].str);
    writeln("model text: ", lm.generateText("c", parseJSON(`{"max_tokens": 4}`)));
    writeln("model ids: ", lm.generate([2], 3).map!(to!string).join(" "));

    // By task: a chat, speech out and back in, embeddings.
    writeln("chat: ", lm.chat(parseJSON(`[{"role": "user", "content": "c"}]`), parseJSON(`{"max_tokens": 4}`)));
    auto speech = Model.open("demo:voice").synthesize("hello");
    writefln("voice: %d samples at %d Hz", speech.audio.length, speech.sampleRate);
    writeln("heard: ", Model.open("demo:speech").transcribe(speech.audio, speech.sampleRate));
    string[] values;
    foreach (row; Model.open("demo:embed").embed(["hello", "sky"]).array)
        foreach (v; row.array)
            values ~= to!string(cast(int) v.get!double);
    writeln("embed: ", values.join(" "));
}
