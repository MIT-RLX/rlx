// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: three ranks as threads on loopback; across machines, one rank
// per host with the same peers. Collectives, pipelined inference (a stage
// shipped per worker), data-parallel training.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>

#include "common.hpp"
#include "rlx/rlx.hpp"

constexpr int WORLD = 3, PER_RANK = 16;

static std::string blob(const std::string& name, const std::vector<float>& v) {
    auto path = scratch("rlx-dist-cpp-" + name);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
    return "file://" + path;
}

static std::vector<float> matvec(const std::vector<float>& x, const std::vector<float>& w) {
    std::vector<float> out(4);
    for (int j = 0; j < 4; j++) {
        float s = 0;
        for (int i = 0; i < 4; i++) s += x[i] * w[i * 4 + j];
        out[j] = std::fmax(0, s);
    }
    return out;
}

static std::string spec(int rank, const std::string& ux, const std::string& uy, const std::string& uw) {
    auto start = std::to_string(rank * PER_RANK), len = std::to_string(PER_RANK);
    return "{\"params\": [{\"name\": \"w\", \"uri\": \"" + uw + "\"}], \"data\": ["
           "{\"input\": \"X\", \"uri\": \"" + ux + "\", \"elem\": 3, \"shard_start\": " + start + ", \"shard_len\": " + len + "},"
           "{\"input\": \"y\", \"uri\": \"" + uy + "\", \"elem\": 1, \"shard_start\": " + start + ", \"shard_len\": " + len + "}],"
           " \"batch\": 4, \"lr\": 0.2, \"epochs\": 40, \"device\": \"cpu\"}";
}

int main() {
    std::vector<std::vector<float>> stage_w(3, std::vector<float>(16));
    for (int i = 0; i < 16; i++) {
        stage_w[1][i] = (i * 5 % 7) / 7.0f - 0.3f;
        stage_w[2][i] = (i * 3 % 5) / 5.0f - 0.2f;
    }
    const float w_true[3] = {1.5f, -2, 0.5f};
    std::vector<float> x_all(PER_RANK * WORLD * 3), y_all(PER_RANK * WORLD);
    for (size_t i = 0; i < x_all.size(); i++) x_all[i] = (i * 37 % 23) / 11.5f - 1;
    for (size_t i = 0; i < y_all.size(); i++) for (int k = 0; k < 3; k++) y_all[i] += x_all[i * 3 + k] * w_true[k];

    auto peers = rlx::loopback_peers(WORLD);
    std::string list = "[";
    for (size_t i = 0; i < peers.size(); i++) list += (i ? ",\"" : "\"") + peers[i] + "\"";
    auto options = "{\"peers\": " + list + "], \"device\": \"cpu\"}";

    std::vector<std::future<std::vector<double>>> workers;
    for (uint32_t rank = 1; rank < WORLD; rank++) {
        workers.push_back(std::async(std::launch::async, [rank, options] {
            auto g = rlx::Group::connect(rank, WORLD, {.options = options});
            g.all_reduce({float(rank + 1)}, {.op = "sum"});
            g.all_gather({float(rank + 1)});
            g.broadcast(0, {0});  // the root's length; it overwrites
            g.serve_worker({.options = "{\"steps\": 2}"});
            return json_numbers(g.serve_trainer(), "w");
        }));
    }

    auto g = rlx::Group::connect(0, WORLD, {.options = options});
    std::printf("mesh: rank %u of %u, leader: %s\n", g.rank(), g.world(), g.is_leader() ? "true" : "false");
    std::printf("all_reduce sum: [%g]\n", g.all_reduce({1}, {.op = "sum"})[0]);
    auto gathered = g.all_gather({1});
    std::printf("all_gather: [%g, %g, %g]\n", gathered[0], gathered[1], gathered[2]);
    std::printf("broadcast: [%g]\n", g.broadcast(0, {42})[0]);
    std::printf("workers: %s\n", g.collect_caps().c_str());

    rlx::Graph stage({.name = "stage"});
    auto sx = stage.input("x", {1, 4});
    stage.set_outputs({stage.relu(stage.matmul(sx, stage.param("w", {4, 4})))});
    for (uint32_t r = 1; r < WORLD; r++)
        g.ship_stage(r, stage, "{\"device\": \"cpu\", \"weights\": {\"w\": \"" + blob("w" + std::to_string(r) + ".bin", stage_w[r]) + "\"}}");
    for (auto xv : {std::vector<float>{1, -1, 0.5f, 2}, std::vector<float>{0.25f, 0.5f, -0.75f, 1}}) {
        g.send_activation(1, xv);
        auto yv = g.recv_activation(WORLD - 1);
        auto want = matvec(matvec(xv, stage_w[1]), stage_w[2]);
        float d = 0;
        for (int i = 0; i < 4; i++) d = std::fmax(d, std::fabs(yv[i] - want[i]));
        std::printf("pipeline max|Δ| vs local = %.1e\n", d);
    }

    g.collect_caps();
    rlx::Graph lr({.name = "lr"});
    auto X = lr.input("X", {4, 3});
    auto Y = lr.input("y", {4, 1});
    auto wv = lr.param("w", {3, 1});
    auto d = lr.sub(lr.matmul(X, wv), Y);
    lr.set_outputs({lr.mean(lr.mul(d, d), {0, 1})});
    auto bwd = rlx::grad(lr, {wv});
    auto ux = blob("X.bin", x_all), uy = blob("y.bin", y_all), uw = blob("w0.bin", {0, 0, 0});
    for (uint32_t r = 1; r < WORLD; r++) g.ship_train(r, bwd, spec(r, ux, uy, uw));
    auto mine = g.run_train(bwd, spec(0, ux, uy, uw));
    std::printf("training loss: %.4f -> %.2e\n", json_number(mine, "first_loss"), json_number(mine, "last_loss"));
    auto w0 = json_numbers(mine, "w");
    std::printf("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)\n", w0[0], w0[1], w0[2]);
    for (int r = 1; r < WORLD; r++) {
        auto theirs = workers[r - 1].get();
        bool agree = theirs.size() == 3;
        for (int i = 0; agree && i < 3; i++) agree = std::fabs(theirs[i] - w0[i]) < 1e-5;
        std::printf("rank %d w agrees: %s\n", r, agree ? "true" : "false");
    }
}
