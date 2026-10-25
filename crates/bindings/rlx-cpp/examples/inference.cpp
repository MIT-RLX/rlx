// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

#include <cstdio>

#include "rlx/rlx.hpp"

static std::vector<float> weights(int n, int seed) {
    std::vector<float> w(n);
    for (int i = 0; i < n; i++) w[i] = ((i * 7919 + seed * 104729) % 1000) / 1000.0f - 0.5f;
    return w;
}

int main(int argc, char** argv) {
    std::string device = argc > 1 ? argv[1] : "cpu";
    rlx::Graph g({.name = "classifier"});
    auto x = g.input("x", {1, 4});
    auto hidden = g.relu(g.matmul(x, g.param("w1", {4, 8})));
    g.set_outputs({g.softmax(g.matmul(hidden, g.param("w2", {8, 3})))});

    auto model = rlx::Session({.device = device}).compile(g);  // consumes g
    std::printf("compiled on: %s\n", model.device().c_str());
    model.set_params({{"w1", weights(32, 1)}, {"w2", weights(24, 2)}});
    auto probs = model.run({{"x", {0.5f, -1, 2, 0.25f}}})[0];
    float sum = 0;
    std::printf("probs:");
    for (float p : probs) { std::printf(" %.4f", p); sum += p; }
    std::printf("\nprobs sum: %.4f\n", sum);
    auto shapes = model.output_shapes();
    std::printf("output shapes: [[%lld, %lld]] dtypes: [%s]\n", (long long)shapes[0][0], (long long)shapes[0][1], model.output_dtypes()[0].c_str());
}
