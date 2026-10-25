// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Small helpers the examples share: reports are JSON text in C++, and these
// pull a number or a number list out of one (use any JSON library for more).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

/// The numbers of the first JSON array after `"key":` in `json`.
inline std::vector<double> json_numbers(const std::string& json, const std::string& key) {
    std::vector<double> out;
    auto at = json.find("\"" + key + "\"");
    if (at == std::string::npos) return out;
    at = json.find('[', at);
    auto end = json.find(']', at);
    std::string body = json.substr(at + 1, end - at - 1);
    const char* p = body.c_str();
    while (*p) {
        char* next = nullptr;
        double v = std::strtod(p, &next);
        if (next == p) { p++; continue; }
        out.push_back(v);
        p = next;
    }
    return out;
}

/// The number after `"key":` in `json`.
inline double json_number(const std::string& json, const std::string& key) {
    auto at = json.find("\"" + key + "\":");
    return at == std::string::npos ? 0 : std::strtod(json.c_str() + at + key.size() + 3, nullptr);
}

/// A scratch path under TMPDIR (or the working directory).
inline std::string scratch(const std::string& name) {
    const char* dir = std::getenv("TMPDIR");
    std::string d = dir ? dir : ".";
    if (!d.empty() && d.back() == '/') d.pop_back();
    return d + "/" + name;
}
