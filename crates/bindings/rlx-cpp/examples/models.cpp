// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "rlx/rlx.hpp"

// A string field of a flat JSON object (enough for `info()`'s top level).
static std::string field(const std::string& json, const std::string& key) {
    auto at = json.find("\"" + key + "\":\"");
    if (at == std::string::npos) return "";
    at += key.size() + 4;
    return json.substr(at, json.find('"', at) - at);
}

int main() {
    rlx::Graph g({.name = "affine"});
    auto x = g.input("x", {1, 2});
    g.set_outputs({g.matmul(x, g.param("w", {2, 2}))});
    auto path = (std::filesystem::temp_directory_path() / "rlx-example-cpp.rlxp").string();
    rlx::save_model(g, path, {{"w", {2, 0, 0, 3}}},
                    {.model = R"({"task": "graph", "family": "affine", "outputs": ["y"]})"});
    {
        auto m = rlx::Model::open(path, {.options = R"({"device": "cpu"})"});
        auto info = m.info();
        std::printf("model: %s (%s) on %s\n", field(info, "family").c_str(), field(info, "task").c_str(),
                    field(info, "device").c_str());
        auto y = m.run({{"x", {1, 1}}})[0];
        std::printf("model run: %.3f %.3f\n", y[0], y[1]);
    }
    std::filesystem::remove(path);

    auto lm = rlx::Model::open("demo:count");
    auto info = lm.info();
    std::printf("plugin model: %s (%s)\n", field(info, "family").c_str(), field(info, "task").c_str());
    std::printf("model text: %s\n", lm.generate_text("c", {.options = R"({"max_tokens": 4})"}).c_str());
    std::printf("model ids:");
    for (auto id : lm.generate({2}, {.max_tokens = 3})) std::printf(" %u", id);
    std::printf("\n");

    // By task: a chat, speech out and back in, embeddings.
    std::printf("chat: %s\n",
                lm.chat(R"([{"role": "user", "content": "c"}])", {.options = R"({"max_tokens": 4})"}).c_str());
    auto speech = rlx::Model::open("demo:voice").synthesize("hello");
    std::printf("voice: %zu samples at %u Hz\n", speech.audio.size(), speech.sample_rate);
    auto heard = rlx::Model::open("demo:speech").transcribe(speech.audio, {.sample_rate = speech.sample_rate});
    std::printf("heard: %s\n", heard.c_str());
    // JSON [[5.0,2.0],[3.0,0.0]]: print every number in it.
    auto rows = rlx::Model::open("demo:embed").embed({"hello", "sky"});
    std::printf("embed:");
    for (const char* p = rows.c_str(); *p;) {
        char* end;
        double v = std::strtod(p, &end);
        if (end != p) {
            std::printf(" %d", static_cast<int>(v));
            p = end;
        } else {
            ++p;
        }
    }
    std::printf("\n");
}
