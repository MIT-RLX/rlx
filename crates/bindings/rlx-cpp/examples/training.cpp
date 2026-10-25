// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.

#include <cmath>
#include <cstdio>

#include "common.hpp"
#include "rlx/rlx.hpp"

constexpr int BATCH = 8;
static const float W_TRUE[3] = {2, -1, 0.5f};

static rlx::Inputs data(uint64_t step) {
    std::vector<float> x(BATCH * 3), y(BATCH);
    for (int i = 0; i < BATCH; i++)
        for (int k = 0; k < 3; k++) {
            x[i * 3 + k] = ((step * 31 + i * 7 + k * 3) % 17) / 8.5f - 1;
            y[i] += x[i * 3 + k] * W_TRUE[k];
        }
    return {{"x", x}, {"y", y}};
}

static void print3(const char* what, const std::vector<float>& w) {
    std::printf("%s%.3f %.3f %.3f\n", what, w[0], w[1], w[2]);
}

int main(int argc, char** argv) {
    std::string device = argc > 1 ? argv[1] : "cpu";
    rlx::Graph g({.name = "regression"});
    auto x = g.input("x", {BATCH, 3});
    auto y = g.input("y", {BATCH, 1});
    auto w = g.param("w", {3, 1});
    auto d = g.sub(g.matmul(x, w), y);
    g.set_outputs({g.mean(g.mul(d, d), {0, 1})});

    auto options = "{\"device\": \"" + device + "\", \"optimizer\": {\"kind\": \"adamw\", \"lr\": 0.1}}";
    auto forward = g.clone();
    rlx::Trainer trainer(forward, {"w"}, {{"w", {0, 0, 0}}}, {.options = options});
    float first = trainer.evaluate(data(0));
    trainer.run(400, [](uint64_t s) { return std::optional(data(s)); },
                {.lr = [](uint64_t s) { return float(0.1 * 0.5 * (1 + std::cos(M_PI * s / 400))); },
                 .on_step = [](uint64_t, float loss) { return loss > 1e-6f; }});
    std::printf("trainer loss: %.4f -> %.6f in %llu steps\n", first, trainer.evaluate(data(0)), (unsigned long long)trainer.steps());
    print3("trainer w: ", trainer.params()["w"]);

    auto path = scratch("rlx-example-train-cpp.gguf");
    auto written = trainer.save(path);
    auto again = g.clone();
    rlx::Trainer restored(again, {"w"}, {{"w", {0, 0, 0}}}, {.options = "{\"device\": \"" + device + "\"}"});
    std::printf("checkpoint: %zu tensors, restored step %.0f\n", written, json_number(restored.load(path), "step"));
    std::remove(path.c_str());
    restored.freeze({.names = std::vector<std::string>{"w"}});
    auto before = restored.params()["w"];
    restored.step(data(1));
    std::printf("frozen: [%s] unchanged: %s\n", restored.frozen()[0].c_str(), restored.params()["w"] == before ? "true" : "false");
    restored.unfreeze();

    auto backward_graph = rlx::grad(g, {w});  // outputs [loss, dw]
    auto backward = rlx::Session({.device = device}).compile(backward_graph);
    rlx::Optimizer opt({.kind = "sgd", .options = "{\"lr\": 0.2}"});
    rlx::Tensors params{{"w", {0, 0, 0}}};
    float loss = 0;
    for (uint64_t step = 0; step < 300; step++) {
        backward.set_param("w", params["w"]);
        auto feed = data(step);
        feed["d_output"] = {1};
        auto outs = backward.run(feed);
        loss = outs[0][0];
        opt.step(params, {{"w", outs[1]}});
    }
    std::printf("manual loss: %.6f w: %.3f %.3f %.3f\n", loss, params["w"][0], params["w"][1], params["w"][2]);
}
