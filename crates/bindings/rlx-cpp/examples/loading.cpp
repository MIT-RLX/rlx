// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "common.hpp"
#include "rlx/rlx.hpp"

int main() {
    std::vector<float> w(64);
    for (int i = 0; i < 64; i++) w[i] = (i % 9) / 4.0f - 1;
    auto gguf = scratch("rlx-example-cpp.gguf");
    {
        rlx::GgufWriter writer;
        writer.add_f32("proj.weight", {8, 8}, "F32", w);
        writer.add_f32("proj.weight_q8", {8, 8}, "Q8_0", w);
        writer.write(gguf, {.architecture = "demo", .metadata = "{\"demo.layers\": 1}"});
    }
    auto f = rlx::Gguf::open(gguf);
    std::printf("gguf architecture: %s\n", f.architecture().value_or("none").c_str());
    std::printf("gguf tensors:");
    for (auto& n : f.tensor_names()) std::printf(" %s", n.c_str());
    auto exact = f.tensor("proj.weight");
    auto q8 = f.tensor("proj.weight_q8");
    float maxd = 0;
    for (size_t i = 0; i < 64; i++) maxd = std::fmax(maxd, std::fabs(exact[i] - q8[i]));
    std::printf("\ngguf f32 exact: %s q8 max|Δ|: %.4f\n", exact == w ? "true" : "false", maxd);

    rlx::Graph g({.name = "proj"});
    auto x = g.input("x", {1, 8});
    g.set_outputs({g.matmul(x, g.param("proj.weight", {8, 8}))});
    auto c = rlx::Session().compile(g);
    c.set_param("proj.weight", exact);
    auto y = c.run({{"x", std::vector<float>(8, 1)}})[0];
    std::printf("gguf -> graph: %.2f %.2f %.2f\n", y[0], y[1], y[2]);

    auto st = scratch("rlx-example-cpp.safetensors");
    std::string header = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}";
    while (header.size() % 8) header += ' ';
    {
        std::ofstream out(st, std::ios::binary);
        uint64_t n = header.size();
        out.write(reinterpret_cast<const char*>(&n), 8);
        out << header;
        out.write(reinterpret_cast<const char*>(w.data()), 256);
    }
    auto loaded = rlx::load_weights("safetensors", st);
    std::printf("safetensors: [%s] exact: %s\n", loaded.begin()->first.c_str(), loaded["proj.weight"] == w ? "true" : "false");

    auto packed = rlx::quantize(w, "Q8_0");
    auto back = rlx::dequantize(packed, "Q8_0", {.count = w.size()});
    std::printf("quantize q8_0: %zu -> %zu bytes, round trip %zu values\n", w.size() * 4, packed.size(), back.size());
    std::remove(gguf.c_str());
    std::remove(st.c_str());
    std::printf("argmax: %u\n", rlx::argmax({0.1f, 2, 0.3f}));
}
