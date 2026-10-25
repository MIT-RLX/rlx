// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.

#include <cmath>
#include <cstdio>

#include "rlx/rlx.hpp"

static rlx::Graph model() {
    rlx::Graph g({.name = "mlp"});
    auto x = g.input("x", {4, 16});
    g.set_outputs({g.tanh(g.matmul(x, g.param("w", {16, 16})))});
    return g;
}

int main() {
    std::vector<float> xs(64), w(256);
    for (int i = 0; i < 64; i++) xs[i] = (i % 7) / 7.0f;
    for (int i = 0; i < 256; i++) w[i] = (i % 5) / 10.0f - 0.2f;
    rlx::Inputs x{{"x", xs}};

    auto available = rlx::devices();
    std::printf("devices:");
    for (auto& d : available) std::printf(" %s", d.c_str());
    std::printf("\nmetal available: %s\n", rlx::is_available("metal") ? "true" : "false");
    auto probe = model();
    std::printf("fastest for graph: %s\n", rlx::fastest_device_for(probe).c_str());
    std::vector<float> reference;
    for (auto& device : available) {
        auto g = model();
        auto c = rlx::Session({.device = device}).compile(g);
        c.set_param("w", w);
        auto y = c.run(x)[0];
        if (reference.empty()) reference = y;
        float d = 0;
        for (size_t i = 0; i < y.size(); i++) d = std::fmax(d, std::fabs(y[i] - reference[i]));
        std::printf("session %s: max|Δ| vs first = %.2e\n", device.c_str(), d);
    }
    rlx::FlexibleSession flex({.policy = "{\"prefer\": [\"metal\", \"cpu\"]}"});
    auto g1 = model();
    std::printf("flexible session picked: %s\n", flex.compile(g1).device().c_str());
    auto g2 = model();
    rlx::Runner runner(g2);
    runner.set_param("w", w);
    std::printf("runner warmed: %s\n", runner.warm_all()[0].c_str());
    std::printf("bench: %s\n", runner.benchmark(x, {.runs = 3}).c_str());
    std::printf("runner ran on: %s\n", runner.run(x).device.c_str());
    std::printf("run_try chain: %s\n", runner.run_try({"metal", "cpu"}, x).device.c_str());
    auto g3 = model();
    rlx::Router router(g3);
    router.set_param("w", w);
    std::printf("router ran on: %s\n", router.run_chain(x).device.c_str());
}
