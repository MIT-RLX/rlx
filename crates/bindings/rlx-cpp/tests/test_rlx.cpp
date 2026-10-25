// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The C++ binding against the real library; exits non-zero on the first failure.

#include <cstdio>
#include <cstdlib>
#include <future>

#include "rlx/rlx.hpp"

static int checks = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (!(cond)) { std::printf("FAILED: %s (line %d)\n", #cond, __LINE__); std::exit(1); } \
    } while (0)

static rlx::Graph loss_graph(rlx::Node* w_out = nullptr) {
    rlx::Graph g({.name = "mse"});
    auto x = g.input("x", {1, 2});
    auto y = g.input("y", {1, 2});
    auto w = g.param("w", {2, 2});
    auto d = g.sub(g.matmul(x, w), y);
    g.set_outputs({g.mean(g.mul(d, d), {0, 1})});
    if (w_out) *w_out = w;
    return g;
}

int main() {
    {  // inference
        auto g = loss_graph();
        auto m = rlx::Session().compile(g);
        CHECK(g.is_consumed());
        m.set_param("w", {1, 0, 0, 1});
        CHECK(m.run({{"x", {1, 2}}, {"y", {0, 0}}})[0][0] == 2.5f);
        CHECK(m.device() == "cpu");
        CHECK(m.output_dtypes() == std::vector<std::string>{"f32"});
    }
    {  // errors
        rlx::Graph g;
        try { g.relu(42); CHECK(false); }
        catch (const rlx::Error& e) { CHECK(e.kind() == rlx::Error::Kind::Range); }
        try { rlx::Session({.device = "no-such-device"}); CHECK(false); }
        catch (const rlx::Error&) { CHECK(true); }
    }
    {  // training
        rlx::Node w;
        auto g = loss_graph(&w);
        auto bg = rlx::grad(g, {w});
        auto bwd = rlx::Session().compile(bg);
        bwd.set_param("w", {1, 0, 0, 1});
        auto outs = bwd.run({{"x", {1, 2}}, {"y", {0, 0}}, {"d_output", {1}}});
        CHECK(outs[1] == (std::vector<float>{1, 2, 2, 4}));
        auto fwd = g.clone();
        rlx::Trainer t(fwd, {"w"}, {{"w", {0, 0, 0, 0}}}, {.options = "{\"optimizer\": {\"kind\": \"adamw\", \"lr\": 0.1}}"});
        rlx::Inputs batch{{"x", {1, 1}}, {"y", {1, -1}}};
        auto r = t.run(300, [&](uint64_t) { return std::optional(batch); }, {.on_step = [](uint64_t, float l) { return l > 1e-6f; }});
        CHECK(r.last_loss < 1e-4f);
        rlx::Optimizer opt({.kind = "sgd", .options = "{\"lr\": 0.5}"});
        rlx::Tensors p{{"p", {1, -1}}};
        opt.step(p, {{"p", {2, -2}}});
        CHECK(p["p"] == (std::vector<float>{0, 0}));
    }
    {  // loading
        std::string path = "rlx-cpp-test.gguf";
        std::vector<float> w{0.5f, -1, 2, 0.25f};
        rlx::GgufWriter wr;
        wr.add_f32("w", {4}, "F32", w);
        CHECK(wr.write(path, {.architecture = "demo"}) == 1);
        auto f = rlx::Gguf::open(path);
        CHECK(f.architecture() == "demo");
        CHECK(f.tensor("w") == w);
        std::remove(path.c_str());
    }
    {  // backends and a two-rank mesh
        CHECK(rlx::devices()[0] == "cpu" && rlx::is_available("cpu"));
        auto g = loss_graph();
        rlx::Runner r(g, {.policy = "{\"only\": [\"cpu\"]}"});
        r.set_param("w", {1, 0, 0, 1});
        auto res = r.run({{"x", {1, 2}}, {"y", {0, 0}}});
        CHECK(res.device == "cpu" && res.outputs[0][0] == 2.5f);
        auto peers = rlx::loopback_peers(2);
        auto opts = "{\"peers\": [\"" + peers[0] + "\", \"" + peers[1] + "\"]}";
        auto worker = std::async(std::launch::async, [opts] {
            auto wg = rlx::Group::connect(1, 2, {.options = opts});
            wg.all_reduce({3}, {.op = "max"});
            return wg.serve_worker({.options = "{\"steps\": 1}"});
        });
        auto mesh = rlx::Group::connect(0, 2, {.options = opts});
        CHECK(mesh.all_reduce({5}, {.op = "max"})[0] == 5);
        mesh.collect_caps();
        rlx::Graph s;
        auto sx = s.input("x", {2});
        s.set_outputs({s.add(sx, sx)});
        mesh.ship_stage(1, s, "{\"device\": \"cpu\"}");
        mesh.send_activation(1, {1, 2});
        CHECK(mesh.recv_activation(1) == (std::vector<float>{2, 4}));
        CHECK(worker.get().find("\"activations\":1") != std::string::npos);
    }
    std::printf("test_rlx: %d checks passed\n", checks);
}
